"""
Portfolio Derisking: HRP + Correlation-Distance Clustering
==========================================================
1. Correlation-distance (1 - |ρ|) for clustering — stable, interpretable
2. Cluster in 10-D MDS, project to 2-D only for plotting
3. Hierarchical Risk Parity (López de Prado) for weight allocation
4. Hard sector caps (configurable)
5. Sampling guide with sector concentration awareness

Requirements:
    pip install yfinance scikit-learn pandas numpy matplotlib scipy
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.manifold import MDS
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import ConvexHull
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, leaves_list
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import warnings

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
TICKERS = [
    "OPFI","HERE","CALM","HGMCF","HMY","KARO","PDD","NONOF",
    "NVO","ATAT","ASRMF","ASR","PXED","WLKP","DRD","MNSO",
    "AATC","HRB","LKNCY","ANF","BLBD","RDY","GNMSF","MO","GMAB",
    "CIG","PNRG","ALAR","RSMDF","CCSI","WLTH","SBS",
    "THC","GAMB","NTES","ABEV","NETTF","ESEA","ASC","DDI","SLVM","PHM",
    "IDT","NATH","DECK","ZTO","TRMB","HESM","GRBK","TEO",
    "NUTX","TCMFF","HRMY","CBT","KOF","VEON","ITRN","ETST",
    "FIZZ","HLF","MIND","LNTH","GSL","GHC","OSPN",
]

N_CLUSTERS          = 8
MIN_TRADING_DAYS    = 100
PERIOD              = "1y"
MDS_HIGH_DIM        = 10      # cluster in this many dimensions
MAX_SECTOR_WEIGHT   = 0.20    # hard cap: 20% per sector
MAX_SINGLE_WEIGHT   = 0.05    # hard cap: 5% per name


# ═══════════════════════════════════════════════════════════════════════════════
# 1. FETCH PRICES
# ═══════════════════════════════════════════════════════════════════════════════
print(f"📥  Downloading {len(TICKERS)} tickers ({PERIOD})…")
raw = yf.download(TICKERS, period=PERIOD, auto_adjust=True, progress=True)
prices = raw["Close"]

valid = prices.columns[prices.count() >= MIN_TRADING_DAYS].tolist()
dropped = sorted(set(TICKERS) - set(valid))
if dropped:
    print(f"⚠️  Dropped (insufficient data): {dropped}")
print(f"✅  Proceeding with {len(valid)} tickers\n")

prices = prices[valid].ffill().bfill().dropna(axis=0, how="all")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. RETURNS + CORRELATION DISTANCE
# ═══════════════════════════════════════════════════════════════════════════════
rets = prices.pct_change().dropna().replace([np.inf, -np.inf], np.nan).dropna(axis=1)
valid = rets.columns.tolist()
n = len(valid)

corr = rets.corr().values
cov  = rets.cov().values

# Correlation distance: d = sqrt(0.5 * (1 - ρ))   (proper metric)
corr_dist = np.sqrt(0.5 * (1 - corr))
np.fill_diagonal(corr_dist, 0)
corr_dist = (corr_dist + corr_dist.T) / 2  # ensure symmetry


# ═══════════════════════════════════════════════════════════════════════════════
# 3. HIGH-DIMENSIONAL MDS → WARD CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════════
print(f"📐  MDS embedding: {MDS_HIGH_DIM}-D for clustering, 2-D for plotting…")

xy_high = MDS(n_components=MDS_HIGH_DIM, dissimilarity="precomputed",
              random_state=42, n_init=10, max_iter=1000,
              normalized_stress="auto").fit_transform(corr_dist)

xy_2d = MDS(n_components=2, dissimilarity="precomputed",
            random_state=42, n_init=10, max_iter=1000,
            normalized_stress="auto").fit_transform(corr_dist)

print(f"🔗  Ward clustering in {MDS_HIGH_DIM}-D → {N_CLUSTERS} clusters")
labels = AgglomerativeClustering(
    n_clusters=N_CLUSTERS, linkage="ward"
).fit_predict(xy_high)

# Relabel largest cluster = 0
counts = Counter(labels)
rank = {cl: r for r, (cl, _) in enumerate(counts.most_common())}
labels = np.array([rank[l] for l in labels])

for c in range(N_CLUSTERS):
    print(f"   Cluster {c}: {(labels == c).sum()} stocks")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SECTOR METADATA
# ═══════════════════════════════════════════════════════════════════════════════
print("🏷️   Fetching sector info…")
meta = {}
for t in valid:
    try:
        info = yf.Ticker(t).info
        meta[t] = dict(
            sector   = info.get("sector") or "Unknown",
            industry = info.get("industry") or "Unknown",
            name     = info.get("shortName") or info.get("longName") or t,
        )
    except Exception:
        meta[t] = dict(sector="Unknown", industry="Unknown", name=t)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. HRP  (Hierarchical Risk Parity — López de Prado)
# ═══════════════════════════════════════════════════════════════════════════════
print("⚖️   Computing HRP weights…")

def _get_quasi_diag(link):
    """Extract sorted order from linkage matrix (quasi-diagonal reordering)."""
    return list(leaves_list(link))

def _get_cluster_var(cov, cluster_items):
    """Inverse-variance portfolio variance for a sub-cluster."""
    sub_cov = cov[np.ix_(cluster_items, cluster_items)]
    inv_diag = 1.0 / np.diag(sub_cov)
    w = inv_diag / inv_diag.sum()
    return float(w @ sub_cov @ w)

def hrp_weights(returns_df):
    """
    Full HRP pipeline:
      1. Tree clustering on correlation distance
      2. Quasi-diagonalization
      3. Recursive bisection for weights
    """
    corr_ = returns_df.corr().values
    cov_  = returns_df.cov().values
    n_    = corr_.shape[0]

    # Distance & linkage
    dist_ = np.sqrt(0.5 * (1.0 - corr_))
    np.fill_diagonal(dist_, 0)
    dist_ = (dist_ + dist_.T) / 2
    condensed_ = squareform(dist_)
    link_ = linkage(condensed_, method="single")

    # Quasi-diagonal order
    sort_ix = _get_quasi_diag(link_)

    # Recursive bisection
    weights = np.ones(n_)

    def _bisect(items):
        if len(items) <= 1:
            return
        mid = len(items) // 2
        left  = items[:mid]
        right = items[mid:]

        var_left  = _get_cluster_var(cov_, left)
        var_right = _get_cluster_var(cov_, right)

        alpha = 1.0 - var_left / (var_left + var_right)

        weights[left]  *= alpha
        weights[right] *= 1.0 - alpha

        _bisect(left)
        _bisect(right)

    _bisect(sort_ix)

    # Normalize
    weights /= weights.sum()
    return weights

hrp_w = hrp_weights(rets)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. APPLY SECTOR CAPS
# ═══════════════════════════════════════════════════════════════════════════════
print("🔒  Applying sector & single-name caps…")

sector_list = [meta[t]["sector"] for t in valid]

def apply_caps(weights, sectors, max_sector, max_single, iterations=20):
    """Iteratively clip weights to sector and single-name caps, renormalize."""
    w = weights.copy()
    for _ in range(iterations):
        # Single-name cap
        excess = np.maximum(w - max_single, 0)
        w = np.minimum(w, max_single)

        # Redistribute excess proportionally to uncapped names
        if excess.sum() > 0:
            uncapped = w < max_single
            if uncapped.any():
                w[uncapped] += excess.sum() * (w[uncapped] / w[uncapped].sum())

        # Sector cap
        sector_arr = np.array(sectors)
        for sec in np.unique(sector_arr):
            mask = sector_arr == sec
            sec_total = w[mask].sum()
            if sec_total > max_sector:
                scale = max_sector / sec_total
                freed = w[mask].sum() - max_sector
                w[mask] *= scale
                # Redistribute to other sectors
                other = ~mask & (w > 0)
                if other.any():
                    w[other] += freed * (w[other] / w[other].sum())

        w = np.maximum(w, 0)
        w /= w.sum()

    return w

hrp_capped = apply_caps(hrp_w, sector_list, MAX_SECTOR_WEIGHT, MAX_SINGLE_WEIGHT)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. BUILD RESULTS TABLE
# ═══════════════════════════════════════════════════════════════════════════════
rows = []
for i, t in enumerate(valid):
    r = rets[t]
    rows.append(dict(
        ticker        = t,
        name          = meta[t]["name"],
        sector        = meta[t]["sector"],
        industry      = meta[t]["industry"],
        cluster       = int(labels[i]),
        x             = float(xy_2d[i, 0]),
        y             = float(xy_2d[i, 1]),
        annualReturn  = round(float((1 + r).prod() ** (252 / len(r)) - 1) * 100, 2),
        volatility    = round(float(r.std() * np.sqrt(252)) * 100, 2),
        hrp_weight    = round(float(hrp_w[i]) * 100, 3),
        capped_weight = round(float(hrp_capped[i]) * 100, 3),
    ))
df = pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. PRINT RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("CLUSTER COMPOSITION")
print("=" * 80)
for c in range(N_CLUSTERS):
    sub = df[df.cluster == c]
    print(f"\n── Cluster {c}  ({len(sub)} stocks) " + "─" * 50)
    for sec, grp in sub.groupby("sector"):
        tks = grp.sort_values("annualReturn", ascending=False).ticker.tolist()
        print(f"   {sec:30s}  {len(tks):2d}  {tks}")

print("\n" + "=" * 80)
print("SECTOR ALLOCATION  (HRP raw → capped)")
print("=" * 80)
sec_alloc = df.groupby("sector").agg(
    raw_wt=("hrp_weight", "sum"),
    capped_wt=("capped_weight", "sum"),
    count=("ticker", "count"),
).sort_values("capped_wt", ascending=False)
for sec, row in sec_alloc.iterrows():
    bar = "█" * int(row.capped_wt / 2)
    print(f"  {sec:30s}  {row['count']:2.0f} names  "
          f"raw={row.raw_wt:5.1f}%  capped={row.capped_wt:5.1f}%  {bar}")

print("\n" + "=" * 80)
print("TOP 20 HOLDINGS  (HRP capped)")
print("=" * 80)
top = df.nlargest(20, "capped_weight")
for _, r in top.iterrows():
    print(f"  {r.ticker:8s}  {r.capped_weight:5.2f}%  "
          f"C{r.cluster}  {r.sector:25s}  ret={r.annualReturn:+6.1f}%  vol={r.volatility:5.1f}%")

print("\n" + "=" * 80)
print("SAMPLING GUIDE  (pick 1–2 per cluster, diversify sectors)")
print("=" * 80)
for c in range(N_CLUSTERS):
    sub = df[df.cluster == c].sort_values("capped_weight", ascending=False)
    seen, picks = set(), []
    for _, r in sub.iterrows():
        if r.sector not in seen:
            seen.add(r.sector)
            picks.append(f"{r.ticker} [{r.sector[:15]}, w={r.capped_weight:.2f}%]")
    print(f"  C{c}: {' | '.join(picks[:5])}")

# Portfolio-level stats
w_vec = hrp_capped
port_ret  = float(w_vec @ df.annualReturn.values)
port_vol  = float(np.sqrt(w_vec @ cov @ w_vec) * np.sqrt(252) * 100)
port_sr   = port_ret / port_vol if port_vol > 0 else 0
n_eff     = 1.0 / (w_vec ** 2).sum()  # effective number of bets

print(f"\n{'─' * 60}")
print(f"  Portfolio expected return:   {port_ret:+.1f}%")
print(f"  Portfolio volatility:        {port_vol:.1f}%")
print(f"  Sharpe ratio (approx):       {port_sr:.2f}")
print(f"  Effective # of bets:         {n_eff:.1f} / {n}")
print(f"  Max single-name weight:      {hrp_capped.max()*100:.2f}%")
print(f"  Max sector weight:           {sec_alloc.capped_wt.max():.1f}%")
print(f"{'─' * 60}")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. PLOTS  (4-panel)
# ═══════════════════════════════════════════════════════════════════════════════
sectors_sorted = sorted(df.sector.unique())
n_sec = len(sectors_sorted)

sec_colors = {s: cm.tab20(i / max(n_sec - 1, 1)) for i, s in enumerate(sectors_sorted)}
cl_colors  = {c: cm.Set1(c / max(N_CLUSTERS - 1, 1)) for c in range(N_CLUSTERS)}
markers    = ["o", "s", "^", "D", "v", "P", "*", "X", "h", "p", "<", ">"]
shape_map  = {s: markers[i % len(markers)] for i, s in enumerate(sectors_sorted)}

fig = plt.figure(figsize=(24, 18))
fig.patch.set_facecolor("#fafafa")
gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.25)

# ── PANEL 1: Clusters colored by sector ──────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
ax.set_title("Clusters colored by SECTOR  ·  Hull = Cluster",
             fontsize=12, fontweight="bold", pad=10)

for sec in sectors_sorted:
    mask = df.sector == sec
    ax.scatter(df.loc[mask, "x"], df.loc[mask, "y"],
               c=[sec_colors[sec]], s=60, edgecolors="white",
               linewidths=0.5, label=sec, zorder=3)

for c in range(N_CLUSTERS):
    pts = df.loc[df.cluster == c, ["x", "y"]].values
    col = cl_colors[c]
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    if len(pts) >= 3:
        try:
            hull = ConvexHull(pts)
            verts = np.append(hull.vertices, hull.vertices[0])
            ax.fill(pts[verts, 0], pts[verts, 1],
                    color=col, alpha=0.06, zorder=1)
            ax.plot(pts[verts, 0], pts[verts, 1], "--",
                    color=col, alpha=0.5, lw=1.5)
        except Exception:
            pass
    ax.text(cx, cy, f"C{c}", fontsize=9, fontweight="bold", color=col,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col, alpha=0.8))

for _, r in df.iterrows():
    ax.annotate(r.ticker, (r.x, r.y), fontsize=5, color="#555",
                textcoords="offset points", xytext=(4, 4))
ax.legend(fontsize=6, loc="best", framealpha=0.9, title="Sector", title_fontsize=7)
ax.set_xlabel("MDS Dim 1"); ax.set_ylabel("MDS Dim 2")
ax.tick_params(labelsize=7); ax.grid(True, alpha=0.12)


# ── PANEL 2: Clusters colored by cluster ─────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
ax.set_title("Colored by CLUSTER  ·  Shape = Sector",
             fontsize=12, fontweight="bold", pad=10)

for c in range(N_CLUSTERS):
    sub = df[df.cluster == c]
    for sec in sub.sector.unique():
        mask = (df.cluster == c) & (df.sector == sec)
        ax.scatter(df.loc[mask, "x"], df.loc[mask, "y"],
                   c=[cl_colors[c]], marker=shape_map[sec],
                   s=60, edgecolors="white", linewidths=0.5, zorder=3)

cl_handles  = [mpatches.Patch(color=cl_colors[c], label=f"Cluster {c}")
               for c in range(N_CLUSTERS)]
sec_handles = [plt.Line2D([0], [0], marker=shape_map[s], color="gray",
               markerfacecolor="gray", markersize=6, linestyle="None", label=s)
               for s in sectors_sorted]
leg1 = ax.legend(handles=cl_handles, fontsize=6, loc="upper left",
                 framealpha=0.9, title="Cluster", title_fontsize=7)
ax.add_artist(leg1)
ax.legend(handles=sec_handles, fontsize=5, loc="lower right",
          framealpha=0.9, title="Sector (shape)", title_fontsize=6)

for _, r in df.iterrows():
    ax.annotate(r.ticker, (r.x, r.y), fontsize=5, color="#555",
                textcoords="offset points", xytext=(4, 4))
ax.set_xlabel("MDS Dim 1"); ax.set_ylabel("MDS Dim 2")
ax.tick_params(labelsize=7); ax.grid(True, alpha=0.12)


# ── PANEL 3: HRP weights bubble chart ────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
ax.set_title("HRP Capped Weights  ·  Bubble size = weight  ·  Color = Sector",
             fontsize=12, fontweight="bold", pad=10)

sizes = df.capped_weight / df.capped_weight.max() * 500 + 20
for sec in sectors_sorted:
    mask = df.sector == sec
    ax.scatter(df.loc[mask, "x"], df.loc[mask, "y"],
               s=sizes[mask], c=[sec_colors[sec]],
               alpha=0.7, edgecolors="white", linewidths=0.5,
               label=sec, zorder=3)

for _, r in df.iterrows():
    ax.annotate(f"{r.ticker}\n{r.capped_weight:.1f}%",
                (r.x, r.y), fontsize=4.5, color="#333",
                ha="center", va="center", fontweight="bold")
ax.legend(fontsize=6, loc="best", framealpha=0.9, title="Sector", title_fontsize=7)
ax.set_xlabel("MDS Dim 1"); ax.set_ylabel("MDS Dim 2")
ax.tick_params(labelsize=7); ax.grid(True, alpha=0.12)


# ── PANEL 4: Sector allocation bar chart ─────────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
ax.set_title("Sector Allocation  (HRP raw vs capped)",
             fontsize=12, fontweight="bold", pad=10)

sec_df = sec_alloc.sort_values("capped_wt")
y_pos = range(len(sec_df))
bars_raw = ax.barh(y_pos, sec_df.raw_wt, height=0.35, label="HRP raw",
                   color="#b0b8d0", edgecolor="white", linewidth=0.5)
bars_cap = ax.barh([y + 0.35 for y in y_pos], sec_df.capped_wt, height=0.35,
                   label="HRP capped",
                   color=[sec_colors.get(s, "#999") for s in sec_df.index],
                   edgecolor="white", linewidth=0.5)
ax.axvline(MAX_SECTOR_WEIGHT * 100, color="red", linestyle="--", lw=1.2,
           alpha=0.7, label=f"Cap ({MAX_SECTOR_WEIGHT*100:.0f}%)")
ax.set_yticks([y + 0.175 for y in y_pos])
ax.set_yticklabels(sec_df.index, fontsize=8)
ax.set_xlabel("Weight (%)", fontsize=9)
ax.legend(fontsize=7, loc="lower right")
ax.tick_params(labelsize=7)
ax.grid(True, axis="x", alpha=0.15)

# Add value labels
for i, (raw, cap) in enumerate(zip(sec_df.raw_wt, sec_df.capped_wt)):
    ax.text(cap + 0.3, i + 0.35, f"{cap:.1f}%", va="center", fontsize=6.5,
            fontweight="bold", color="#333")

plt.savefig("stock_clusters_hrp.png", dpi=180, bbox_inches="tight")
print("\n📊  Saved stock_clusters_hrp.png")
plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# 10. DENDROGRAM
# ═══════════════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(20, 7))
fig2.patch.set_facecolor("#fafafa")

condensed = squareform(corr_dist)
Z = linkage(condensed, method="ward")

# Color labels by sector
label_colors = {t: sec_colors[meta[t]["sector"]] for t in valid}
hex_colors = {}
for t, rgba in label_colors.items():
    hex_colors[t] = "#{:02x}{:02x}{:02x}".format(
        int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))

dn = dendrogram(Z, labels=valid, leaf_rotation=90, leaf_font_size=6,
                color_threshold=0, above_threshold_color="#999", ax=ax2)

# Color the x-axis labels by sector
xlbls = ax2.get_xticklabels()
for lbl in xlbls:
    t = lbl.get_text()
    if t in hex_colors:
        lbl.set_color(hex_colors[t])
        lbl.set_fontweight("bold")

ax2.set_title("Hierarchical Clustering Dendrogram  ·  Label color = Sector",
              fontsize=13, fontweight="bold", pad=12)
ax2.set_ylabel("Correlation Distance", fontsize=10)

# Add sector legend
sec_patches = [mpatches.Patch(color=sec_colors[s], label=s) for s in sectors_sorted]
ax2.legend(handles=sec_patches, fontsize=6, loc="upper right",
           framealpha=0.9, title="Sector", title_fontsize=7, ncol=2)

plt.tight_layout()
plt.savefig("stock_dendrogram.png", dpi=150, bbox_inches="tight")
print("📊  Saved stock_dendrogram.png")
plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# 11. EXPORT
# ═══════════════════════════════════════════════════════════════════════════════
df_out = df.sort_values(["cluster", "capped_weight"], ascending=[True, False])
df_out.to_csv("stock_clusters_hrp.csv", index=False)
print("📄  Saved stock_clusters_hrp.csv")