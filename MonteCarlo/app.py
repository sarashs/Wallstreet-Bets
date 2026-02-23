"""
app.py
------
Streamlit dashboard for the Bayesian Portfolio Simulation Platform.

Tabs
----
Tab 1 — Portfolio Setup & MCMC
    • Enter tickers and weights
    • Configure MCMC hyperparameters
    • Run MCMC (both models in sequence)
    • Inspect convergence diagnostics and WAIC comparison
    • Save posterior parameters to YAML + NPZ

Tab 2 — Monte Carlo Simulation
    • Load a saved YAML config (auto-loads session state if freshly run)
    • Set simulation horizon (days) and number of paths
    • Run forward Monte Carlo
    • Display fan chart, return distribution, VaR/CVaR metrics
    • Per-stock CVaR decomposition
    • Save simulation results as images

Usage
-----
    streamlit run MonteCarlo/app.py
"""

import sys
import os
import logging
import pathlib
import json
import time
import datetime

import numpy as np
import pandas as pd
import yaml
import streamlit as st

# ---- path setup -----------------------------------------------------------
ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ---- configure logging FIRST (before any module-level loggers fire) --------
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "app.log", mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)
logger.info("Streamlit app starting")

# ---- project imports -------------------------------------------------------
from data.fetcher import fetch_returns, compute_spearman_correlation
from models.model_a import ModelA
from models.model_b import ModelB
from mcmc.sampler import GibbsSampler
from mcmc.diagnostics import make_diagnostics_table
from selection.waic import select_model
from simulation.montecarlo import (
    run_forward_simulation,
    compute_risk_metrics,
    compute_cvar_contribution,
)
from output.visualizations import (
    plot_fan_chart,
    plot_return_distribution,
    plot_trace,
    plot_posterior_densities,
    plot_rhat_table,
    plot_ess_table,
    plot_shrinkage,
    plot_var_breakdown,
    plot_model_graph_a,
    plot_model_graph_b,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Bayesian Portfolio Simulator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
def _init_session():
    defaults = {
        "portfolio"    : [],              # list of {"ticker": str, "weight": float}
        "mcmc_result"  : None,            # dict returned by run_mcmc_pipeline()
        "sim_result"   : None,            # dict returned by run_forward_simulation()
        "loaded_config": None,            # dict loaded from YAML
        "log_messages" : [],              # list of str for in-app log display
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_session()


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _log(msg: str, level: str = "info"):
    """Write to both the Python logger and the in-app log panel."""
    getattr(logger, level)(msg)
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.log_messages.append(f"[{ts}] {msg}")
    # Keep last 200 entries
    if len(st.session_state.log_messages) > 200:
        st.session_state.log_messages = st.session_state.log_messages[-200:]


def _save_results(mcmc_result: dict, tickers: list, weights: list, history_days: int) -> pathlib.Path:
    """Persist posterior samples and metadata to disk.

    Files
    -----
    MonteCarlo/output/<timestamp>/config.yaml
    MonteCarlo/output/<timestamp>/posterior.npz
    """
    ts        = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = ROOT / "output" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path  = out_dir / "posterior.npz"
    yaml_path = out_dir / "config.yaml"

    comp   = mcmc_result["comparison"]
    sel    = comp["comparison"]["selected"]

    # --- Save posterior samples ---
    samples_key = f"samples_{sel.lower()}"
    chain_key   = f"chain_samples_{sel.lower()}"
    np.savez_compressed(
        npz_path,
        samples       = mcmc_result[samples_key],
        spearman_corr = mcmc_result["spearman_corr"],
        returns_hist  = mcmc_result["returns_hist"],
    )

    # --- Save config YAML ---
    diag_a = mcmc_result.get("diag_a", {})
    diag_b = mcmc_result.get("diag_b", {})

    cfg = {
        "version"        : "0.1.0",
        "created_at"     : ts,
        "tickers"        : tickers,
        "weights"        : [float(w) for w in weights],
        "history_days"   : history_days,
        "selected_model" : sel,
        "waic_a"         : float(comp["waic_a"]["waic"]),
        "waic_b"         : float(comp["waic_b"]["waic"]),
        "delta_waic"     : float(comp["comparison"]["delta_waic"]),
        "se_delta"       : float(comp["comparison"]["se_delta"]),
        "uncertain"      : bool(comp["comparison"]["uncertain"]),
        "rationale"      : comp["comparison"]["rationale"],
        "rhat_max_a"     : float(np.array(diag_a.get("rhat", [1.0])).max()),
        "rhat_max_b"     : float(np.array(diag_b.get("rhat", [1.0])).max()),
        "ess_min_a"      : float(np.array(diag_a.get("ess", [0.0])).min()),
        "ess_min_b"      : float(np.array(diag_b.get("ess", [0.0])).min()),
        "n_posterior_samples": int(mcmc_result[samples_key].shape[0]),
        "mcmc_settings"  : mcmc_result["mcmc_settings"],
        "posterior_file" : str(npz_path.name),
        "output_dir"     : str(out_dir),
    }

    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    _log(f"Results saved to {out_dir}")
    return yaml_path


def _load_config(yaml_path: str) -> dict:
    """Load config YAML and companion NPZ."""
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    out_dir  = pathlib.Path(cfg["output_dir"])
    npz_path = out_dir / cfg["posterior_file"]
    npz      = np.load(npz_path, allow_pickle=False)

    cfg["posterior_samples"] = npz["samples"]
    cfg["spearman_corr"]     = npz["spearman_corr"]
    cfg["returns_hist"]      = npz["returns_hist"]

    return cfg


# ---------------------------------------------------------------------------
# Core pipeline (runs MCMC for both models)
# ---------------------------------------------------------------------------

def run_mcmc_pipeline(
    tickers: list,
    weights: list,
    history_days: int,
    warmup: int,
    n_samples: int,
    n_chains: int,
    thinning: int,
    progress_bar,
    status_text,
) -> dict:
    """Fetch data, run MCMC for both models, compute WAIC, select model."""

    # --- 1. Data -----------------------------------------------------------
    status_text.text("Fetching historical price data from Yahoo Finance...")
    _log(f"Fetching data: {tickers}, lookback={history_days} days")

    data = fetch_returns(tickers, n_days=history_days)
    survived = data["tickers"]
    returns  = data["returns"]    # (N, T)
    N, T     = returns.shape

    div_yields = data.get("dividend_yields", {})
    n_payers   = sum(1 for y in div_yields.values() if y > 0)
    _log(
        f"Data fetched: {N} tickers × {T} days | winsorised={data['n_winsorised']} | "
        f"dividend payers={n_payers}/{N} (returns are TOTAL returns incl. dividends)"
    )

    if data["dropped_tickers"]:
        _log(f"Dropped tickers (insufficient data): {data['dropped_tickers']}", "warning")
        status_text.warning(
            f"Dropped tickers — insufficient history: {data['dropped_tickers']}. "
            "These were removed before alignment so other tickers keep their full window."
        )

    if data.get("short_history"):
        short_strs = [f"{t} ({d} days)" for t, d in data["short_history"]]
        _log(
            f"Short-history tickers (present but trimmed alignment window): {short_strs}",
            "warning",
        )
        status_text.warning(
            f"Short-history tickers kept but trimmed the aligned window: {short_strs}. "
            f"All tickers are now aligned to {T} trading days instead of the requested {history_days}. "
            "Consider removing these tickers if you need the full history for the others."
        )

    # Re-align weights to survived tickers
    ticker_to_w = dict(zip(tickers, weights))
    w_survived  = np.array([ticker_to_w.get(t, 1.0) for t in survived])
    w_survived  = w_survived / w_survived.sum()

    spearman = compute_spearman_correlation(returns)

    # --- 2. Model A MCMC ---------------------------------------------------
    status_text.text("Running MCMC for Model A (Hierarchical Student-t)...")
    _log("Starting Model A MCMC")
    progress_bar.progress(0.05)

    model_a  = ModelA()
    def cb_a(frac):
        progress_bar.progress(0.05 + frac * 0.35)

    sampler_a = GibbsSampler(
        model_a, returns,
        n_chains=n_chains, warmup=warmup,
        n_samples=n_samples, thinning=thinning,
        seed=42, progress_callback=cb_a,
    )
    result_a  = sampler_a.run()
    diag_a    = make_diagnostics_table(result_a["chain_samples"], result_a["param_names"])
    _log(f"Model A done. R-hat max={np.array(diag_a['rhat']).max():.4f}, ESS min={np.array(diag_a['ess']).min():.0f}")

    # --- 3. Model B MCMC ---------------------------------------------------
    status_text.text("Running MCMC for Model B (Normal Mixture Regime-switching)...")
    _log("Starting Model B MCMC")

    model_b  = ModelB()
    def cb_b(frac):
        progress_bar.progress(0.40 + frac * 0.35)

    sampler_b = GibbsSampler(
        model_b, returns,
        n_chains=n_chains, warmup=warmup,
        n_samples=n_samples, thinning=thinning,
        seed=123, progress_callback=cb_b,
    )
    result_b  = sampler_b.run()
    diag_b    = make_diagnostics_table(result_b["chain_samples"], result_b["param_names"])
    _log(f"Model B done. R-hat max={np.array(diag_b['rhat']).max():.4f}, ESS min={np.array(diag_b['ess']).min():.0f}")

    # --- 4. WAIC model selection -------------------------------------------
    status_text.text("Computing WAIC and selecting model...")
    _log("Computing WAIC")
    progress_bar.progress(0.78)

    comparison = select_model(
        model_a, model_b,
        result_a["samples"], result_b["samples"],
        returns, n_sub=300,
    )
    sel = comparison["comparison"]["selected"]
    _log(f"WAIC model selection: Model {sel}. {comparison['comparison']['rationale']}")
    progress_bar.progress(0.95)

    mcmc_settings = dict(
        warmup=warmup, n_samples=n_samples,
        n_chains=n_chains, thinning=thinning,
    )

    return dict(
        tickers          = survived,
        weights          = w_survived,
        returns_hist     = returns,
        spearman_corr    = spearman,
        samples_a        = result_a["samples"],
        samples_b        = result_b["samples"],
        chain_samples_a  = result_a["chain_samples"],
        chain_samples_b  = result_b["chain_samples"],
        param_names_a    = result_a["param_names"],
        param_names_b    = result_b["param_names"],
        diag_a           = diag_a,
        diag_b           = diag_b,
        comparison       = comparison,
        mcmc_settings    = mcmc_settings,
        dividend_yields  = data.get("dividend_yields", {}),
        dividend_history = data.get("dividend_history", {}),
    )


# ===========================================================================
# TAB 1 — Portfolio Setup & MCMC
# ===========================================================================

def render_tab1():
    st.header("Portfolio Setup & MCMC")
    st.markdown(
        "Enter your portfolio tickers and weights, configure the MCMC sampler, "
        "then click **Run MCMC** to fit both models and select the best one via WAIC."
    )

    # -----------------------------------------------------------------------
    # Portfolio input
    # -----------------------------------------------------------------------
    st.subheader("Portfolio Definition")

    col_add1, col_add2, col_add3 = st.columns([2, 1, 1])
    with col_add1:
        new_ticker = st.text_input("Ticker symbol", placeholder="e.g. AAPL", key="new_ticker").upper().strip()
    with col_add2:
        new_weight = st.number_input("Weight (market cap, any units)", min_value=0.01, value=1.0, step=0.1, key="new_weight")
    with col_add3:
        st.write("")
        st.write("")
        if st.button("Add ticker", use_container_width=True):
            if new_ticker:
                existing = [p["ticker"] for p in st.session_state.portfolio]
                if new_ticker in existing:
                    st.warning(f"{new_ticker} is already in the portfolio.")
                else:
                    st.session_state.portfolio.append({"ticker": new_ticker, "weight": new_weight})
                    _log(f"Added {new_ticker} with weight {new_weight}")

    # Bulk input
    with st.expander("Bulk entry (paste CSV: ticker,weight per line)"):
        bulk_text = st.text_area(
            "Format: AAPL,150\nMSFT,200\n...",
            height=120,
            key="bulk_text",
        )
        if st.button("Parse bulk input"):
            lines = [l.strip() for l in bulk_text.strip().splitlines() if l.strip()]
            added = 0
            for line in lines:
                parts = line.split(",")
                if len(parts) == 2:
                    t = parts[0].strip().upper()
                    try:
                        w = float(parts[1].strip())
                        existing = [p["ticker"] for p in st.session_state.portfolio]
                        if t not in existing and t:
                            st.session_state.portfolio.append({"ticker": t, "weight": w})
                            added += 1
                    except ValueError:
                        pass
            if added:
                st.success(f"Added {added} tickers.")
                _log(f"Bulk added {added} tickers")

    # Current portfolio table
    if st.session_state.portfolio:
        port_df = pd.DataFrame(st.session_state.portfolio)
        port_df["weight_pct"] = (port_df["weight"] / port_df["weight"].sum() * 100).round(2)

        edited = st.data_editor(
            port_df[["ticker", "weight"]],
            num_rows="dynamic",
            use_container_width=True,
            key="portfolio_editor",
        )
        # Sync back
        st.session_state.portfolio = edited.to_dict("records")

        # Summary
        n_total = len(port_df)
        col_s1, col_s2 = st.columns(2)
        col_s1.metric("Tickers", n_total)
        col_s2.metric("Sum of weights", f"{port_df['weight'].sum():.2f}")

    else:
        st.info("No tickers added yet.  Add tickers above or use bulk entry.")

    st.markdown("---")

    # -----------------------------------------------------------------------
    # MCMC configuration
    # -----------------------------------------------------------------------
    st.subheader("MCMC Configuration")

    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    with col_m1:
        history_days = st.number_input("History (trading days)", min_value=252, max_value=1260, value=504, step=63)
    with col_m2:
        warmup    = st.number_input("Warmup iterations", min_value=500, max_value=10000, value=2000, step=500)
    with col_m3:
        n_samples = st.number_input("Sampling iterations", min_value=500, max_value=20000, value=5000, step=500)
    with col_m4:
        n_chains  = st.number_input("Chains", min_value=2, max_value=8, value=4, step=1)
    with col_m5:
        thinning  = st.number_input("Thinning", min_value=1, max_value=10, value=2, step=1)

    n_stored = (n_samples // thinning) * n_chains
    st.caption(
        f"Total posterior draws after thinning: **{n_stored:,}** "
        f"({n_samples // thinning:,} per chain × {n_chains} chains)"
    )

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Run MCMC
    # -----------------------------------------------------------------------
    can_run = len(st.session_state.portfolio) >= 2

    if not can_run:
        st.warning("Please add at least 2 tickers to run the MCMC pipeline.")

    if st.button("Run MCMC (both models)", type="primary", disabled=not can_run):
        tickers = [p["ticker"] for p in st.session_state.portfolio]
        weights = [p["weight"]  for p in st.session_state.portfolio]

        prog_bar    = st.progress(0)
        status_text = st.empty()
        start_time  = time.time()

        try:
            result = run_mcmc_pipeline(
                tickers=tickers,
                weights=weights,
                history_days=int(history_days),
                warmup=int(warmup),
                n_samples=int(n_samples),
                n_chains=int(n_chains),
                thinning=int(thinning),
                progress_bar=prog_bar,
                status_text=status_text,
            )
            st.session_state.mcmc_result = result
            elapsed = time.time() - start_time
            prog_bar.progress(1.0)
            status_text.success(f"MCMC complete in {elapsed:.1f}s")
            _log(f"MCMC pipeline complete in {elapsed:.1f}s")
        except Exception as e:
            status_text.error(f"MCMC failed: {e}")
            _log(f"MCMC error: {e}", "error")
            logger.exception("MCMC pipeline error")

    # -----------------------------------------------------------------------
    # Results section (only visible after successful run)
    # -----------------------------------------------------------------------
    if st.session_state.mcmc_result is not None:
        r = st.session_state.mcmc_result
        st.markdown("---")
        st.subheader("MCMC Results")

        comp    = r["comparison"]
        sel     = comp["comparison"]["selected"]
        waic_a  = comp["waic_a"]["waic"]
        waic_b  = comp["waic_b"]["waic"]
        delta   = comp["comparison"]["delta_waic"]

        # ---- Portfolio + dividend summary ---------------------------------
        div_yields = r.get("dividend_yields", {})
        if div_yields:
            survived_tickers = r["tickers"]
            survived_weights = r["weights"]

            port_rows = []
            for tk, wt in zip(survived_tickers, survived_weights):
                dy = div_yields.get(tk, 0.0)
                port_rows.append({
                    "Ticker"         : tk,
                    "Weight (%)"     : f"{wt * 100:.1f}",
                    "Div Yield (TTM)": f"{dy * 100:.2f}%" if dy > 0 else "—",
                    "Return type"    : "Total return (price + dividends)" if dy > 0 else "Price return only",
                })
            port_summary_df = pd.DataFrame(port_rows)

            with st.expander("Portfolio summary — dividend yields (trailing 12 months)", expanded=True):
                st.dataframe(port_summary_df, use_container_width=True)
                n_pay = sum(1 for dy in div_yields.values() if dy > 0)
                if n_pay > 0:
                    st.caption(
                        f"**Note:** Log-returns used for MCMC are *total* returns — "
                        f"dividends are already embedded in the adjusted close prices "
                        f"(DRIP assumed). Yields shown above are for reference only."
                    )
                else:
                    st.caption("No dividend-paying tickers detected in trailing 12 months.")

        # WAIC summary card
        col_w1, col_w2, col_w3, col_w4 = st.columns(4)
        col_w1.metric("WAIC (Model A)", f"{waic_a:.1f}")
        col_w2.metric("WAIC (Model B)", f"{waic_b:.1f}")
        col_w3.metric("ΔWAIC (B − A)", f"{delta:+.1f}", delta_color="inverse")
        col_w4.metric("Selected Model", f"Model {sel}")

        st.info(comp["comparison"]["rationale"])

        # ---- Hierarchical model diagrams ----------------------------------
        with st.expander("Hierarchical model structure (graphical model diagrams)", expanded=False):
            st.markdown(
                "Plate notation for both candidate models. "
                "**Light blue** nodes = hyperpriors · **White** nodes = latent variables · "
                "**Grey** nodes = observed data · Dashed rectangles = plates (repeated structure)."
            )
            col_ga, col_gb = st.columns(2)
            with col_ga:
                st.plotly_chart(plot_model_graph_a(), use_container_width=True)
            with col_gb:
                st.plotly_chart(plot_model_graph_b(), use_container_width=True)

        # Diagnostics tabs
        diag_tab_a, diag_tab_b, trace_tab, posterior_tab = st.tabs([
            "Model A Diagnostics", "Model B Diagnostics", "Trace Plots", "Posterior Densities"
        ])

        with diag_tab_a:
            st.subheader("Model A — Convergence")
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.plotly_chart(plot_rhat_table(r["diag_a"]), use_container_width=True)
            with col_d2:
                st.plotly_chart(plot_ess_table(r["diag_a"]), use_container_width=True)

        with diag_tab_b:
            st.subheader("Model B — Convergence")
            col_d3, col_d4 = st.columns(2)
            with col_d3:
                st.plotly_chart(plot_rhat_table(r["diag_b"]), use_container_width=True)
            with col_d4:
                st.plotly_chart(plot_ess_table(r["diag_b"]), use_container_width=True)

        with trace_tab:
            st.subheader(f"Trace Plots — Selected Model {sel}")
            chain_s = r[f"chain_samples_{sel.lower()}"]
            pnames  = r[f"param_names_{sel.lower()}"]
            n_disp  = min(10, len(pnames), chain_s[0].shape[1])
            st.plotly_chart(plot_trace(chain_s, pnames, n_display=n_disp), use_container_width=True)

        with posterior_tab:
            st.subheader(f"Posterior Densities — Selected Model {sel}")
            chain_s = r[f"chain_samples_{sel.lower()}"]
            pnames  = r[f"param_names_{sel.lower()}"]
            n_disp  = min(8, len(pnames), chain_s[0].shape[1])
            st.plotly_chart(
                plot_posterior_densities(chain_s, pnames, sel, r["returns_hist"].shape[0], n_disp),
                use_container_width=True,
            )
            st.plotly_chart(
                plot_shrinkage(chain_s, r["returns_hist"], pnames, r["returns_hist"].shape[0]),
                use_container_width=True,
            )

        st.markdown("---")

        # Save results
        st.subheader("Save Parameters")
        if st.button("Save to YAML + NPZ", type="secondary"):
            try:
                yaml_path = _save_results(
                    r, r["tickers"], r["weights"].tolist(), int(history_days)
                )
                st.success(f"Saved: `{yaml_path}`")
                _log(f"Saved config to {yaml_path}")
            except Exception as e:
                st.error(f"Save failed: {e}")
                _log(f"Save error: {e}", "error")


# ===========================================================================
# TAB 2 — Monte Carlo Simulation
# ===========================================================================

def render_tab2():
    st.header("Monte Carlo Simulation")
    st.markdown(
        "Load saved MCMC parameters (or use the session result from Tab 1), "
        "configure the simulation horizon, and run the forward simulation."
    )

    # -----------------------------------------------------------------------
    # Load config
    # -----------------------------------------------------------------------
    st.subheader("Load MCMC Parameters")

    load_mode = st.radio(
        "Source",
        ["Use current session (Tab 1 result)", "Load from YAML file"],
        horizontal=True,
    )

    config_ready = False
    active_cfg   = None

    if load_mode == "Use current session (Tab 1 result)":
        if st.session_state.mcmc_result is not None:
            r   = st.session_state.mcmc_result
            sel = r["comparison"]["comparison"]["selected"]

            active_cfg = {
                "tickers"          : r["tickers"],
                "weights"          : r["weights"].tolist(),
                "selected_model"   : sel,
                "posterior_samples": r[f"samples_{sel.lower()}"],
                "spearman_corr"    : r["spearman_corr"],
                "returns_hist"     : r["returns_hist"],
            }
            config_ready = True
            st.success(f"Session result loaded: {len(r['tickers'])} tickers, Model {sel} selected.")
        else:
            st.warning("No session result found.  Please run MCMC in Tab 1 first, or load a YAML file.")

    else:
        output_root = ROOT / "output"
        yaml_files  = sorted(output_root.rglob("config.yaml"), reverse=True) if output_root.exists() else []

        if yaml_files:
            options     = [str(f) for f in yaml_files]
            chosen_yaml = st.selectbox("Select saved config", options)
            if st.button("Load selected config"):
                try:
                    active_cfg = _load_config(chosen_yaml)
                    st.session_state.loaded_config = active_cfg
                    config_ready = True
                    st.success(f"Loaded: {chosen_yaml}")
                    _log(f"Loaded config from {chosen_yaml}")
                except Exception as e:
                    st.error(f"Load failed: {e}")
                    _log(f"Load error: {e}", "error")
        else:
            st.info("No saved configs found.  Run MCMC and save from Tab 1 first.")

        if st.session_state.loaded_config is not None and not config_ready:
            active_cfg   = st.session_state.loaded_config
            config_ready = True
            st.info("Using previously loaded config.")

    # -----------------------------------------------------------------------
    # Show portfolio summary if config loaded
    # -----------------------------------------------------------------------
    if config_ready and active_cfg is not None:
        with st.expander("Portfolio summary"):
            cfg_tickers = active_cfg["tickers"]
            cfg_weights = np.asarray(active_cfg["weights"])
            cfg_weights = cfg_weights / cfg_weights.sum()

            summary_df = pd.DataFrame({
                "Ticker": cfg_tickers,
                "Weight (%)": (cfg_weights * 100).round(2),
            })
            st.dataframe(summary_df, use_container_width=True)
            st.caption(f"Selected model: **{active_cfg['selected_model']}**")

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Simulation configuration
    # -----------------------------------------------------------------------
    st.subheader("Simulation Settings")

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        horizon_days = st.number_input(
            "Simulation horizon (trading days)",
            min_value=21, max_value=1260, value=252, step=21,
            help="252 ≈ 1 year, 126 ≈ 6 months, 63 ≈ 1 quarter"
        )
    with col_s2:
        n_sim = st.number_input(
            "Number of simulation paths",
            min_value=1000, max_value=100_000, value=10_000, step=1000,
        )
    with col_s3:
        sim_seed = st.number_input("Random seed", min_value=0, max_value=99999, value=0)

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Run simulation
    # -----------------------------------------------------------------------
    if st.button("Run Monte Carlo Simulation", type="primary", disabled=not config_ready):
        if active_cfg is None:
            st.error("No config loaded.")
        else:
            prog_bar    = st.progress(0)
            status_text = st.empty()
            start_time  = time.time()

            status_text.text("Running forward Monte Carlo simulation...")
            _log(f"Starting simulation: horizon={horizon_days}, n_sim={n_sim}")

            try:
                result = run_forward_simulation(
                    selected_model    = active_cfg["selected_model"],
                    posterior_samples = np.asarray(active_cfg["posterior_samples"]),
                    returns_hist      = np.asarray(active_cfg["returns_hist"]),
                    spearman_corr     = np.asarray(active_cfg["spearman_corr"]),
                    weights           = np.asarray(active_cfg["weights"]),
                    tickers           = list(active_cfg["tickers"]),
                    horizon_days      = int(horizon_days),
                    n_sim             = int(n_sim),
                    seed              = int(sim_seed),
                    progress_callback = lambda f: prog_bar.progress(f),
                )
                st.session_state.sim_result = {
                    "result" : result,
                    "tickers": active_cfg["tickers"],
                    "horizon": int(horizon_days),
                }
                elapsed = time.time() - start_time
                prog_bar.progress(1.0)
                status_text.success(f"Simulation complete in {elapsed:.1f}s  ({int(n_sim):,} paths)")
                _log(f"Simulation complete in {elapsed:.1f}s")
            except Exception as e:
                status_text.error(f"Simulation failed: {e}")
                _log(f"Simulation error: {e}", "error")
                logger.exception("Simulation error")

    # -----------------------------------------------------------------------
    # Display results
    # -----------------------------------------------------------------------
    if st.session_state.sim_result is not None:
        sr      = st.session_state.sim_result
        result  = sr["result"]
        tickers = sr["tickers"]
        horizon = sr["horizon"]
        metrics = result["risk_metrics"]
        weights = result["weights"]

        st.markdown("---")
        st.subheader("Simulation Results")

        # ---- Key metrics row ----
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Expected Return",    f"{metrics['expected_return']*100:.1f}%")
        m2.metric("Median Return",      f"{metrics['median_return']*100:.1f}%")
        m3.metric("Volatility (ann.)",  f"{metrics['volatility_ann']*100:.1f}%")
        m4.metric("VaR (5%)",           f"{metrics['var_5pct']*100:.1f}%",  delta_color="inverse")
        m5.metric("CVaR (5%)",          f"{metrics['cvar_5pct']*100:.1f}%", delta_color="inverse")
        m6.metric("Prob. of Loss",      f"{metrics['prob_loss']*100:.1f}%",  delta_color="inverse")

        m7, m8, m9 = st.columns(3)
        m7.metric("95th Percentile", f"{metrics['pct_95']*100:.1f}%")
        m8.metric("Skewness",        f"{metrics['skewness']:.2f}")
        m9.metric("Excess Kurtosis", f"{metrics['excess_kurtosis']:.2f}")

        # ---- Fan chart ----
        st.plotly_chart(
            plot_fan_chart(result["percentile_paths"], tickers, weights, horizon, metrics),
            use_container_width=True,
        )

        # ---- Return distribution ----
        st.plotly_chart(
            plot_return_distribution(result["portfolio_returns"], metrics),
            use_container_width=True,
        )

        # ---- CVaR decomposition ----
        st.subheader("CVaR Decomposition by Stock")
        cvar_contrib = compute_cvar_contribution(
            result["portfolio_returns"],
            result["stock_returns"],
            weights,
        )
        st.plotly_chart(
            plot_var_breakdown(tickers, cvar_contrib),
            use_container_width=True,
        )

        # ---- Per-stock summary table ----
        with st.expander("Per-stock return summary"):
            stock_r = result["stock_returns"]   # (n_sim, N)
            rows = []
            for i, tk in enumerate(tickers):
                r_i = stock_r[:, i]
                rows.append({
                    "Ticker"     : tk,
                    "Weight (%)" : f"{weights[i]*100:.1f}",
                    "E[R]"       : f"{r_i.mean()*100:.1f}%",
                    "Vol (ann)"  : f"{r_i.std()*np.sqrt(252/horizon)*100:.1f}%",
                    "VaR(5%)"    : f"{np.percentile(r_i,5)*100:.1f}%",
                    "CVaR Contrib%": f"{cvar_contrib['pct_contributions'][i]*100:.1f}%",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # ---- Save results ----
        st.markdown("---")
        st.subheader("Save Simulation Results")
        if st.button("Save simulation figures"):
            ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = ROOT / "output" / f"sim_{ts}"
            out_dir.mkdir(parents=True, exist_ok=True)

            try:
                fan_fig  = plot_fan_chart(result["percentile_paths"], tickers, weights, horizon, metrics)
                dist_fig = plot_return_distribution(result["portfolio_returns"], metrics)
                var_fig  = plot_var_breakdown(tickers, cvar_contrib)

                fan_fig.write_image(str(out_dir / "fan_chart.png"))
                dist_fig.write_image(str(out_dir / "return_distribution.png"))
                var_fig.write_image(str(out_dir / "var_breakdown.png"))

                # Save metrics as YAML
                metrics_out = {k: float(v) if isinstance(v, (float, int, np.floating)) else v
                               for k, v in metrics.items()}
                with open(out_dir / "metrics.yaml", "w") as f:
                    yaml.dump(metrics_out, f)

                st.success(f"Figures and metrics saved to `{out_dir}`")
                _log(f"Simulation figures saved to {out_dir}")
            except Exception as e:
                st.error(f"Save failed: {e}")
                _log(f"Figure save error: {e}", "error")


# ===========================================================================
# Sidebar — activity log
# ===========================================================================

def render_sidebar():
    with st.sidebar:
        st.title("Activity Log")
        if st.button("Clear log"):
            st.session_state.log_messages = []

        log_text = "\n".join(reversed(st.session_state.log_messages[-50:]))
        st.text_area("", value=log_text, height=600, disabled=True, key="log_display")


# ===========================================================================
# Main
# ===========================================================================

def main():
    st.title("Bayesian Portfolio Simulation Platform")
    st.markdown(
        "A hierarchical Bayesian MCMC engine with automatic model selection (WAIC) "
        "and forward Monte Carlo simulation for portfolio risk analysis."
    )

    tab1, tab2 = st.tabs(["MCMC & Portfolio Setup", "Monte Carlo Simulation"])

    with tab1:
        render_tab1()

    with tab2:
        render_tab2()

    render_sidebar()


if __name__ == "__main__":
    main()
