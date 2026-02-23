"""
output/visualizations.py
------------------------
All Plotly-based visualisations for the Bayesian Portfolio Simulation Platform.

Public API
----------
plot_fan_chart(percentile_paths, tickers, weights, horizon_days, risk_metrics) -> Figure
plot_return_distribution(portfolio_returns, risk_metrics) -> Figure
plot_trace(chain_samples, param_names, n_display) -> Figure
plot_posterior_densities(chain_samples, param_names, model_name, N) -> Figure
plot_rhat_table(diag) -> Figure
plot_ess_table(diag) -> Figure
plot_shrinkage(chain_samples, returns_hist, param_names, N) -> Figure
plot_var_breakdown(tickers, cvar_contrib) -> Figure
plot_model_graph_a() -> Figure
plot_model_graph_b() -> Figure
"""

import logging
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
BAND_COLOURS = {
    5  : "rgba(31, 119, 180, 0.10)",
    25 : "rgba(31, 119, 180, 0.20)",
    50 : "rgba(31, 119, 180, 0.00)",
    75 : "rgba(31, 119, 180, 0.20)",
    95 : "rgba(31, 119, 180, 0.10)",
}
MEDIAN_COLOUR  = "#1f77b4"
VAR_COLOUR     = "#d62728"
CVAR_COLOUR    = "#ff7f0e"
GREEN          = "#2ca02c"
RED            = "#d62728"


# ---------------------------------------------------------------------------
# 1. Fan chart
# ---------------------------------------------------------------------------

def plot_fan_chart(
    percentile_paths: dict,
    tickers: list[str],
    weights: np.ndarray,
    horizon_days: int,
    risk_metrics: dict,
) -> go.Figure:
    """Fan chart showing 5/25/50/75/95 percentile bands of portfolio returns."""
    T    = len(list(percentile_paths.values())[0])
    days = np.arange(T)

    fig = go.Figure()

    # Shaded bands (bottom to top: 5-95, 25-75)
    p5  = percentile_paths[5]  * 100
    p25 = percentile_paths[25] * 100
    p50 = percentile_paths[50] * 100
    p75 = percentile_paths[75] * 100
    p95 = percentile_paths[95] * 100

    # 5–95 band
    fig.add_trace(go.Scatter(
        x=np.concatenate([days, days[::-1]]),
        y=np.concatenate([p95, p5[::-1]]),
        fill="toself", fillcolor="rgba(31,119,180,0.12)",
        line=dict(width=0), name="5–95 pct", showlegend=True,
        hoverinfo="skip",
    ))

    # 25–75 band
    fig.add_trace(go.Scatter(
        x=np.concatenate([days, days[::-1]]),
        y=np.concatenate([p75, p25[::-1]]),
        fill="toself", fillcolor="rgba(31,119,180,0.25)",
        line=dict(width=0), name="25–75 pct", showlegend=True,
        hoverinfo="skip",
    ))

    # Median
    fig.add_trace(go.Scatter(
        x=days, y=p50, mode="lines",
        line=dict(color=MEDIAN_COLOUR, width=2.5),
        name="Median (50th pct)",
    ))

    # Zero line
    fig.add_hline(y=0, line=dict(color="grey", dash="dash", width=1))

    # VaR annotation
    var_pct = risk_metrics["var_5pct"] * 100
    fig.add_annotation(
        x=T - 1, y=var_pct,
        text=f"VaR(5%) = {var_pct:.1f}%",
        showarrow=True, arrowhead=2,
        font=dict(color=VAR_COLOUR, size=11),
    )

    fig.update_layout(
        title=f"Portfolio Return Fan Chart — {horizon_days}-Day Horizon",
        xaxis_title="Trading Days",
        yaxis_title="Cumulative Return (%)",
        legend=dict(x=0.01, y=0.99),
        template="plotly_white",
        height=500,
    )
    return fig


# ---------------------------------------------------------------------------
# 2. Return distribution histogram
# ---------------------------------------------------------------------------

def plot_return_distribution(
    portfolio_returns: np.ndarray,
    risk_metrics: dict,
) -> go.Figure:
    """Histogram of terminal portfolio returns with VaR and CVaR marked."""
    r   = portfolio_returns * 100   # convert to %
    var = risk_metrics["var_5pct"]  * 100
    cvar = risk_metrics["cvar_5pct"] * 100
    mu   = risk_metrics["expected_return"] * 100

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=r, nbinsx=80,
        marker_color="steelblue", opacity=0.75,
        name="Simulated returns",
    ))

    # VaR vertical line
    fig.add_vline(
        x=var, line=dict(color=VAR_COLOUR, width=2, dash="dash"),
        annotation_text=f"VaR(5%) {var:.1f}%",
        annotation_position="top right",
        annotation_font_color=VAR_COLOUR,
    )

    # CVaR vertical line
    fig.add_vline(
        x=cvar, line=dict(color=CVAR_COLOUR, width=2, dash="dot"),
        annotation_text=f"CVaR(5%) {cvar:.1f}%",
        annotation_position="top left",
        annotation_font_color=CVAR_COLOUR,
    )

    # Expected return
    fig.add_vline(
        x=mu, line=dict(color=GREEN, width=2),
        annotation_text=f"E[R] {mu:.1f}%",
        annotation_position="top right",
        annotation_font_color=GREEN,
    )

    fig.update_layout(
        title="Simulated Portfolio Return Distribution at Horizon",
        xaxis_title="Portfolio Return (%)",
        yaxis_title="Frequency",
        template="plotly_white",
        showlegend=False,
        height=420,
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Trace plots
# ---------------------------------------------------------------------------

def plot_trace(
    chain_samples: list[np.ndarray],
    param_names: list[str],
    n_display: int = 8,
) -> go.Figure:
    """Overlaid trace plots for the first n_display parameters across all chains."""
    n_chains = len(chain_samples)
    n_cols   = 2
    n_rows   = (n_display + 1) // 2

    params_to_show = param_names[:n_display]

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=params_to_show,
        vertical_spacing=0.08,
    )

    colours = px.colors.qualitative.Set1[:n_chains]

    for idx, pname in enumerate(params_to_show):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        for c, samples in enumerate(chain_samples):
            if idx < samples.shape[1]:
                fig.add_trace(
                    go.Scatter(
                        y=samples[:, idx],
                        mode="lines",
                        line=dict(color=colours[c % len(colours)], width=0.8),
                        name=f"Chain {c+1}" if idx == 0 else None,
                        showlegend=(idx == 0),
                        legendgroup=f"chain_{c}",
                    ),
                    row=row, col=col,
                )

    fig.update_layout(
        title="MCMC Trace Plots (post-warmup samples)",
        height=200 * n_rows,
        template="plotly_white",
    )
    return fig


# ---------------------------------------------------------------------------
# 4. Posterior density plots
# ---------------------------------------------------------------------------

def plot_posterior_densities(
    chain_samples: list[np.ndarray],
    param_names: list[str],
    model_name: str,
    N: int,
    n_display: int = 8,
) -> go.Figure:
    """Kernel density estimates for posterior marginals of key parameters."""
    combined = np.concatenate(chain_samples, axis=0)

    # Show the first n_display parameters
    params_to_show = param_names[:n_display]

    n_cols = 2
    n_rows = (n_display + 1) // 2

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=params_to_show,
        vertical_spacing=0.1,
    )

    for idx, pname in enumerate(params_to_show):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        if idx >= combined.shape[1]:
            continue

        x_data = combined[:, idx]
        kde_x, kde_y = _kde(x_data)

        fig.add_trace(
            go.Scatter(
                x=kde_x, y=kde_y,
                fill="tozeroy",
                fillcolor="rgba(31,119,180,0.25)",
                line=dict(color=MEDIAN_COLOUR),
                name=pname,
                showlegend=False,
            ),
            row=row, col=col,
        )

        # Mark posterior mean
        mu = x_data.mean()
        fig.add_vline(
            x=mu, line=dict(color="black", dash="dash", width=1),
            row=row, col=col,  # type: ignore
        )

    fig.update_layout(
        title=f"Posterior Marginal Densities — Model {model_name}",
        height=200 * n_rows,
        template="plotly_white",
    )
    return fig


# ---------------------------------------------------------------------------
# 5. R-hat table
# ---------------------------------------------------------------------------

def plot_rhat_table(diag: dict) -> go.Figure:
    """Colour-coded R-hat table."""
    params = diag["param"]
    rhat   = diag["rhat"]
    ok     = diag["rhat_ok"]

    colours = [GREEN if o else RED for o in ok]
    text    = [f"{r:.4f}" for r in rhat]

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["Parameter", "R-hat"],
            fill_color="lightgrey",
            font=dict(size=13, color="black"),
            align="left",
        ),
        cells=dict(
            values=[params, text],
            fill_color=[["white"] * len(params), colours],
            font=dict(size=12),
            align=["left", "center"],
        ),
    )])

    fig.update_layout(
        title="R-hat Convergence Diagnostics (threshold < 1.05)",
        template="plotly_white",
        height=min(600, 60 + 30 * len(params)),
    )
    return fig


# ---------------------------------------------------------------------------
# 6. ESS table
# ---------------------------------------------------------------------------

def plot_ess_table(diag: dict) -> go.Figure:
    """Colour-coded ESS table."""
    params = diag["param"]
    ess    = diag["ess"]
    ok     = diag["ess_ok"]

    colours = [GREEN if o else RED for o in ok]
    text    = [f"{e:.0f}" for e in ess]

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["Parameter", "ESS"],
            fill_color="lightgrey",
            font=dict(size=13, color="black"),
            align="left",
        ),
        cells=dict(
            values=[params, text],
            fill_color=[["white"] * len(params), colours],
            font=dict(size=12),
            align=["left", "center"],
        ),
    )])

    fig.update_layout(
        title="Effective Sample Size (threshold > 400)",
        template="plotly_white",
        height=min(600, 60 + 30 * len(params)),
    )
    return fig


# ---------------------------------------------------------------------------
# 7. Shrinkage plot
# ---------------------------------------------------------------------------

def plot_shrinkage(
    chain_samples: list[np.ndarray],
    returns_hist: np.ndarray,
    param_names: list[str],
    N: int,
) -> go.Figure:
    """Scatter plot: sample mean vs posterior mean for each stock's μ_i."""
    combined = np.concatenate(chain_samples, axis=0)

    # μ_i are the first N parameters (Model A) or same for Model B (indices 0..N-1 within mu_ik flatten)
    # For simplicity, extract the first N params labelled "mu_X"
    mu_indices = [i for i, n in enumerate(param_names[:combined.shape[1]]) if n.startswith("mu_") and "_" not in n[3:]]
    if not mu_indices:
        mu_indices = list(range(min(N, combined.shape[1])))

    post_means  = combined[:, mu_indices].mean(axis=0)
    sample_means = returns_hist[:len(mu_indices)].mean(axis=1)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=sample_means * 100,
        y=post_means   * 100,
        mode="markers+text",
        text=[f"Stock {i}" for i in range(len(mu_indices))],
        textposition="top center",
        marker=dict(size=10, color=MEDIAN_COLOUR),
        name="Stocks",
    ))

    # 1:1 reference line
    mn = min(sample_means.min(), post_means.min()) * 100 * 1.2
    mx = max(sample_means.max(), post_means.max()) * 100 * 1.2
    fig.add_trace(go.Scatter(
        x=[mn, mx], y=[mn, mx],
        mode="lines", line=dict(dash="dash", color="grey"),
        name="1:1 (no shrinkage)",
    ))

    fig.update_layout(
        title="Bayesian Shrinkage — Sample Mean vs Posterior Mean (μ_i)",
        xaxis_title="Sample Mean Daily Return (%)",
        yaxis_title="Posterior Mean Daily Return (%)",
        template="plotly_white",
        height=420,
    )
    return fig


# ---------------------------------------------------------------------------
# 8. CVaR contribution bar chart
# ---------------------------------------------------------------------------

def plot_var_breakdown(
    tickers: list[str],
    cvar_contrib: dict,
) -> go.Figure:
    """Horizontal bar chart of each stock's % contribution to portfolio CVaR."""
    pct = cvar_contrib["pct_contributions"] * 100

    colours = [RED if p > 0 else GREEN for p in pct]

    fig = go.Figure(go.Bar(
        y=tickers,
        x=pct,
        orientation="h",
        marker_color=colours,
        text=[f"{p:.1f}%" for p in pct],
        textposition="outside",
    ))

    fig.update_layout(
        title="Stock Contribution to Portfolio CVaR (5%)",
        xaxis_title="CVaR Contribution (%)",
        template="plotly_white",
        height=max(300, 40 * len(tickers) + 100),
        margin=dict(l=100),
    )
    return fig


# ---------------------------------------------------------------------------
# Kernel density estimate helper
# ---------------------------------------------------------------------------

def _kde(x: np.ndarray, n_points: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Simple Gaussian KDE."""
    from scipy.stats import gaussian_kde

    try:
        kde  = gaussian_kde(x, bw_method="scott")
        xmin = x.min() - 3 * x.std()
        xmax = x.max() + 3 * x.std()
        xs   = np.linspace(xmin, xmax, n_points)
        ys   = kde(xs)
    except Exception:
        xs = np.linspace(x.min(), x.max(), n_points)
        ys = np.zeros(n_points)

    return xs, ys


# ---------------------------------------------------------------------------
# 9. Hierarchical model plate diagrams
# ---------------------------------------------------------------------------

# ---- shared helpers -------------------------------------------------------

def _node_style(kind: str) -> dict:
    """Return marker kwargs for a given node kind."""
    styles = {
        "hyperprior": dict(symbol="circle", size=38, color="rgba(173,216,230,0.85)",
                           line=dict(color="#2c6fad", width=1.5)),
        "latent":     dict(symbol="circle", size=38, color="rgba(255,255,255,0.95)",
                           line=dict(color="#555", width=1.5)),
        "observed":   dict(symbol="circle", size=38, color="rgba(180,180,180,0.85)",
                           line=dict(color="#333", width=1.5)),
        "determ":     dict(symbol="diamond", size=28, color="rgba(255,230,150,0.90)",
                           line=dict(color="#aa8800", width=1.5)),
    }
    return styles.get(kind, styles["latent"])


def _add_nodes(fig, nodes: list[dict]):
    """Add all graph nodes as a single scatter trace per kind."""
    from collections import defaultdict
    groups: dict[str, list] = defaultdict(list)
    for nd in nodes:
        groups[nd["kind"]].append(nd)

    kind_label = {
        "hyperprior": "Hyperprior",
        "latent":     "Latent variable",
        "observed":   "Observed",
        "determ":     "Deterministic",
    }
    for kind, nd_list in groups.items():
        xs   = [n["x"] for n in nd_list]
        ys   = [n["y"] for n in nd_list]
        txts = [n["label"] for n in nd_list]
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="markers+text",
            text=txts,
            textfont=dict(size=12, color="black"),
            textposition="middle center",
            marker=_node_style(kind),
            name=kind_label[kind],
            legendgroup=kind,
            showlegend=True,
            hovertext=[n.get("desc", n["label"]) for n in nd_list],
            hoverinfo="text",
        ))


def _arrow(fig, x0, y0, x1, y1, color="#444"):
    """Add a directed edge arrow as a layout annotation."""
    fig.add_annotation(
        x=x1, y=y1, ax=x0, ay=y0,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True,
        arrowhead=2, arrowsize=1.2, arrowwidth=1.5,
        arrowcolor=color,
    )


def _plate(fig, x0, y0, x1, y1, label: str, color="rgba(100,100,200,0.08)"):
    """Add a plate (dashed rectangle) with a label in the bottom-left corner."""
    fig.add_shape(
        type="rect",
        x0=x0, y0=y0, x1=x1, y1=y1,
        line=dict(color="#666", width=1.5, dash="dash"),
        fillcolor=color,
    )
    fig.add_annotation(
        x=x0 + 0.05, y=y0 + 0.05,
        xref="x", yref="y",
        text=label,
        showarrow=False,
        font=dict(size=10, color="#555"),
        xanchor="left", yanchor="bottom",
    )


def _graph_layout(fig, title: str, width: int = 700, height: int = 620):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis=dict(visible=False, range=[-0.5, 7.5]),
        yaxis=dict(visible=False, range=[-0.5, 8.5], scaleanchor="x", scaleratio=1),
        template="plotly_white",
        width=width,
        height=height,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=-0.12,
            xanchor="center", x=0.5,
            font=dict(size=11),
        ),
        margin=dict(l=20, r=20, t=60, b=80),
    )


# ---- Model A --------------------------------------------------------------

def plot_model_graph_a() -> go.Figure:
    """Plate-notation DAG for Model A — Hierarchical Student-t.

    Hierarchy
    ---------
    Global hyperpriors → per-stock latents → per-(stock,day) latents → observed.

    Plates
    ------
    * Outer plate  i = 1…N  (stocks)
    * Inner plate  t = 1…T  (days, nested inside stock plate)
    """
    fig = go.Figure()

    # ------------------------------------------------------------------
    # Node layout  (x, y) in a loose grid; y increases upward
    # Columns:  0=left-edge, 3=centre, 6=right-edge
    # Rows:     8=top, 0=bottom
    # ------------------------------------------------------------------
    nodes = [
        # --- Global hyperpriors (row 7–8, spread horizontally) ---
        dict(x=0.8, y=7.6, kind="hyperprior", label="μ₀",
             desc="Global mean hyperprior\nμ₀ ~ N(0, 1)"),
        dict(x=2.2, y=7.6, kind="hyperprior", label="τ",
             desc="Mean shrinkage precision\nτ ~ Half-Normal(1)"),
        dict(x=3.8, y=7.6, kind="hyperprior", label="νₛ",
             desc="Shape hyper for σᵢ\nνₛ ~ Gamma(2,0.1)"),
        dict(x=5.2, y=7.6, kind="hyperprior", label="ξ",
             desc="Scale hyper for σᵢ\nξ ~ Half-Normal(1)"),
        dict(x=6.5, y=7.6, kind="hyperprior", label="ν",
             desc="Student-t dof\nν ~ Gamma(2, 0.1), ν > 2"),

        # --- Per-stock latents (row 5.5, stock plate) ---
        dict(x=1.5, y=5.4, kind="latent", label="μᵢ",
             desc="Per-stock mean\nμᵢ | μ₀,τ ~ N(μ₀, 1/τ²)"),
        dict(x=4.5, y=5.4, kind="latent", label="σᵢ",
             desc="Per-stock vol\nσᵢ | νₛ,ξ ~ InvGamma(νₛ/2, νₛξ²/2)"),

        # --- Per-(stock,day) latents (row 3.3, inner plate) ---
        dict(x=3.0, y=3.2, kind="latent", label="λᵢₜ",
             desc="Scale-mixture weight\nλᵢₜ | ν ~ Gamma(ν/2, ν/2)"),

        # --- Observed (row 1.5) ---
        dict(x=3.0, y=1.5, kind="observed", label="rᵢₜ",
             desc="Observed log-return\nrᵢₜ | μᵢ, σᵢ, λᵢₜ ~ N(μᵢ, σᵢ²/λᵢₜ)"),
    ]

    # ------------------------------------------------------------------
    # Edges  (parent → child)
    # ------------------------------------------------------------------
    edges = [
        # hyperpriors → per-stock
        (0.8, 7.6, 1.5, 5.4),   # μ₀ → μᵢ
        (2.2, 7.6, 1.5, 5.4),   # τ  → μᵢ
        (3.8, 7.6, 4.5, 5.4),   # νₛ → σᵢ
        (5.2, 7.6, 4.5, 5.4),   # ξ  → σᵢ
        (6.5, 7.6, 3.0, 3.2),   # ν  → λᵢₜ
        # per-stock → per-obs
        (1.5, 5.4, 3.0, 3.2),   # μᵢ → λᵢₜ  (via observed; draw to r)
        (1.5, 5.4, 3.0, 1.5),   # μᵢ → rᵢₜ
        (4.5, 5.4, 3.0, 1.5),   # σᵢ → rᵢₜ
        # λ → r
        (3.0, 3.2, 3.0, 1.5),   # λᵢₜ → rᵢₜ
    ]

    for (x0, y0, x1, y1) in edges:
        _arrow(fig, x0, y0, x1, y1)

    _add_nodes(fig, nodes)

    # ------------------------------------------------------------------
    # Plates
    # ------------------------------------------------------------------
    # Stock plate  i = 1…N
    _plate(fig, 0.3, 0.8, 5.5, 6.3, "i = 1 … N  (stocks)",
           color="rgba(100,180,255,0.06)")
    # Day plate  t = 1…T  (inner)
    _plate(fig, 1.5, 0.8, 5.0, 4.1, "t = 1 … T  (days)",
           color="rgba(60,60,200,0.05)")

    _graph_layout(fig, "Model A — Hierarchical Student-t (plate notation)")
    return fig


# ---- Model B --------------------------------------------------------------

def plot_model_graph_b() -> go.Figure:
    """Plate-notation DAG for Model B — Hierarchical Normal Mixture (K=2).

    Hierarchy
    ---------
    Global hyperpriors (per component k) → per-stock latents →
    per-(stock,day) regime variable → observed.

    Plates
    ------
    * Component plate  k = 0,1
    * Stock plate  i = 1…N
    * Day plate  t = 1…T
    """
    fig = go.Figure()

    # ------------------------------------------------------------------
    # Node layout
    # ------------------------------------------------------------------
    nodes = [
        # --- Component hyperpriors (row 7.6) ---
        dict(x=0.5, y=7.6, kind="hyperprior", label="μ₀ₖ",
             desc="Per-component mean hyper\nμ₀ₖ ~ N(0,1)"),
        dict(x=1.8, y=7.6, kind="hyperprior", label="τₖ",
             desc="Per-component shrinkage\nτₖ ~ Half-Normal(1)"),
        dict(x=3.1, y=7.6, kind="hyperprior", label="νₖ",
             desc="Per-component vol shape\nνₖ ~ Gamma(2,0.1)"),
        dict(x=4.4, y=7.6, kind="hyperprior", label="ξₖ",
             desc="Per-component vol scale\nξₖ ~ Half-Normal(1)"),
        dict(x=5.8, y=7.6, kind="hyperprior", label="α_π",
             desc="Beta hyperprior α\nα_π ~ Gamma(2,1)"),
        dict(x=7.0, y=7.6, kind="hyperprior", label="β_π",
             desc="Beta hyperprior β\nβ_π ~ Gamma(2,1)"),

        # --- Per-stock, per-component latents (row 5.4) ---
        dict(x=0.9, y=5.4, kind="latent", label="μᵢₖ",
             desc="Per-stock per-component mean\nμᵢₖ | μ₀ₖ,τₖ ~ N(μ₀ₖ, 1/τₖ²)"),
        dict(x=2.8, y=5.4, kind="latent", label="σᵢₖ",
             desc="Per-stock per-component vol\nσᵢₖ | νₖ,ξₖ ~ InvGamma(νₖ/2, νₖξₖ²/2)\n(σᵢ₀ < σᵢ₁ ordering)"),

        # --- Per-stock mixing weight (row 5.4, right) ---
        dict(x=5.2, y=5.4, kind="latent", label="πᵢ",
             desc="Per-stock mixing weight\nπᵢ | α_π,β_π ~ Beta(α_π, β_π)"),

        # --- Per-(stock,day) regime (row 3.2) ---
        dict(x=3.0, y=3.2, kind="latent", label="zᵢₜ",
             desc="Regime indicator\nzᵢₜ | πᵢ ~ Bernoulli(πᵢ)"),

        # --- Observed (row 1.5) ---
        dict(x=3.0, y=1.5, kind="observed", label="rᵢₜ",
             desc="Observed log-return\nrᵢₜ | zᵢₜ,μᵢₖ,σᵢₖ ~ N(μᵢ,zᵢₜ, σᵢ,zᵢₜ²)"),
    ]

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------
    edges = [
        # component hyperpriors → per-stock latents
        (0.5, 7.6, 0.9, 5.4),   # μ₀ₖ → μᵢₖ
        (1.8, 7.6, 0.9, 5.4),   # τₖ  → μᵢₖ
        (3.1, 7.6, 2.8, 5.4),   # νₖ  → σᵢₖ
        (4.4, 7.6, 2.8, 5.4),   # ξₖ  → σᵢₖ
        # Beta hyperpriors → πᵢ
        (5.8, 7.6, 5.2, 5.4),   # α_π → πᵢ
        (7.0, 7.6, 5.2, 5.4),   # β_π → πᵢ
        # per-stock → regime
        (5.2, 5.4, 3.0, 3.2),   # πᵢ → zᵢₜ
        # regime + params → observed
        (3.0, 3.2, 3.0, 1.5),   # zᵢₜ → rᵢₜ
        (0.9, 5.4, 3.0, 1.5),   # μᵢₖ → rᵢₜ
        (2.8, 5.4, 3.0, 1.5),   # σᵢₖ → rᵢₜ
    ]

    for (x0, y0, x1, y1) in edges:
        _arrow(fig, x0, y0, x1, y1)

    _add_nodes(fig, nodes)

    # ------------------------------------------------------------------
    # Plates
    # ------------------------------------------------------------------
    # Component plate  k = 0,1
    _plate(fig, 0.1, 4.7, 4.2, 8.3, "k = 0, 1  (regimes)",
           color="rgba(200,120,60,0.05)")
    # Stock plate  i = 1…N
    _plate(fig, 0.1, 0.8, 6.3, 6.2, "i = 1 … N  (stocks)",
           color="rgba(100,180,255,0.06)")
    # Day plate  t = 1…T  (inner)
    _plate(fig, 1.5, 0.8, 5.5, 4.1, "t = 1 … T  (days)",
           color="rgba(60,60,200,0.05)")

    _graph_layout(fig, "Model B — Hierarchical Normal Mixture, K = 2 (plate notation)",
                  width=780, height=640)
    return fig
