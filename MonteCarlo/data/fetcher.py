"""
data/fetcher.py
---------------
Fetches historical adjusted-close prices from Yahoo Finance, computes daily
log-returns, aligns calendars across tickers, winsorises extreme values, and
returns a clean NumPy returns matrix plus metadata.

Dividend handling
-----------------
Yahoo Finance's adjusted-close price (returned when auto_adjust=True) is
backward-adjusted for BOTH stock splits AND cash dividends.  The adjustment
works by applying a multiplicative factor to all historical prices whenever a
dividend is paid, so that:

    log(adj_close_t / adj_close_{t-1})

equals the *total return* on day t — i.e. pure price appreciation PLUS the
dividend yield on ex-dividend dates.  This implicitly models full dividend
reinvestment (DRIP) without any separate accounting.

We additionally fetch the raw dividend history per ticker (in a separate API
call) so the user can inspect trailing 12-month dividend yields.  Those yields
are returned in the result dict under "dividend_yields" but play NO role in
the MCMC model itself, which already operates on total-return log-returns.

Public API
----------
fetch_returns(tickers, n_days=504) -> dict
compute_spearman_correlation(returns_matrix) -> np.ndarray
"""

import logging
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WINSORISE_LIMIT = 0.15   # ±15 % daily return cap (likely data errors beyond this)
MIN_OBSERVATIONS = 200   # minimum trading days required after cleaning


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def fetch_returns(tickers: list[str], n_days: int = 504) -> dict:
    """Fetch and clean historical daily log-returns for a list of tickers.

    Returns total-return log-returns: adjusted close prices from Yahoo Finance
    (auto_adjust=True) already embed dividend reinvestment, so
    log(adj_close_t / adj_close_{t-1}) captures both price appreciation and
    the dividend yield on each ex-dividend date.

    Parameters
    ----------
    tickers : list[str]
        Yahoo Finance ticker symbols (e.g. ["AAPL", "MSFT"]).
    n_days : int
        Number of *trading* days to use (default 504 ≈ 2 trading years).

    Returns
    -------
    dict with keys:
        "returns"          : np.ndarray (N, T) — total-return log-returns
        "tickers"          : list[str]  — tickers that survived cleaning
        "dates"            : list[str]  — ISO date strings per column
        "prices"           : np.ndarray (N, T) — dividend-adjusted close prices
        "n_winsorised"     : int        — total winsorised observations
        "dropped_tickers"  : list[str]  — tickers removed (insufficient data)
        "short_history"    : list[(str, int)] — tickers that trimmed alignment window
        "dividend_yields"  : dict[str, float] — trailing 12M yield per ticker
        "dividend_history" : dict[str, pd.Series] — raw dividend amounts per ticker

    Raises
    ------
    ValueError
        If fewer than 2 tickers survive the cleaning pipeline.
    """
    import yfinance as yf  # lazy import to keep module import fast

    logger.info("Fetching price data for %d tickers, lookback=%d days", len(tickers), n_days)

    end_date = pd.Timestamp.today()
    start_date = end_date - pd.Timedelta(days=n_days + 60)   # extra buffer for weekends/holidays

    # ---- Download adjusted close prices (total-return series) -------------
    # auto_adjust=True: Yahoo Finance backward-adjusts all historical prices
    # for splits AND dividends.  The resulting daily log-returns are therefore
    # TOTAL returns — price appreciation plus dividend income on ex-div dates.
    logger.debug("Downloading from Yahoo Finance: %s → %s", start_date.date(), end_date.date())
    raw = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )

    # yfinance returns multi-level columns when len(tickers) > 1
    if isinstance(raw.columns, pd.MultiIndex):
        price_df = raw["Close"]
    else:
        # Single ticker → flat columns
        price_df = raw[["Close"]].rename(columns={"Close": tickers[0]})

    logger.debug("Raw price shape after download: %s", price_df.shape)

    # ---- Trim to requested lookback ---------------------------------------
    # Keep only the most recent n_days trading days
    price_df = price_df.dropna(how="all").iloc[-n_days:]

    # ---- Log-returns ------------------------------------------------------
    log_ret_df = np.log(price_df / price_df.shift(1)).iloc[1:]
    logger.debug("Log-return shape before alignment: %s", log_ret_df.shape)

    # ---- Pre-alignment history check -------------------------------------
    # Identify tickers whose own history is shorter than the requested window.
    # These tickers will silently trim everyone else's effective sample if kept.
    # Drop them now, before alignment, so the remaining tickers keep their full
    # history rather than being truncated to the shortest common window.
    short_history = []
    pre_align_dropped = []
    for col in log_ret_df.columns:
        available = log_ret_df[col].notna().sum()
        if available < MIN_OBSERVATIONS:
            logger.warning(
                "Pre-alignment drop: %s has only %d trading days of history "
                "(min %d). Removing before alignment to protect other tickers.",
                col, available, MIN_OBSERVATIONS,
            )
            pre_align_dropped.append(col)
        elif available < n_days * 0.75:
            # Ticker has enough data to survive, but is significantly shorter
            # than requested — warn so the user knows alignment will be trimmed.
            logger.warning(
                "%s has only %d / %d requested trading days. "
                "Calendar alignment will shorten all other tickers to this window.",
                col, available, n_days,
            )
            short_history.append((col, int(available)))

    if pre_align_dropped:
        log_ret_df = log_ret_df.drop(columns=pre_align_dropped)
        price_df   = price_df.drop(columns=pre_align_dropped)
        logger.info(
            "Removed %d ticker(s) before alignment: %s",
            len(pre_align_dropped), pre_align_dropped,
        )

    if short_history:
        bottleneck_ticker, bottleneck_days = min(short_history, key=lambda x: x[1])
        logger.warning(
            "Shortest-history ticker after pre-drop: %s (%d days). "
            "All tickers will be aligned to this window.",
            bottleneck_ticker, bottleneck_days,
        )

    # ---- Drop rows where ANY remaining ticker has NaN (calendar alignment) --
    # This removes holidays / non-overlapping trading days.
    log_ret_df = log_ret_df.dropna(how="any")
    logger.debug("Log-return shape after alignment: %s", log_ret_df.shape)

    # ---- Winsorise --------------------------------------------------------
    n_winsorised = 0
    for col in log_ret_df.columns:
        too_high = log_ret_df[col] > WINSORISE_LIMIT
        too_low  = log_ret_df[col] < -WINSORISE_LIMIT
        if too_high.any() or too_low.any():
            count = too_high.sum() + too_low.sum()
            logger.warning(
                "Winsorising %d observations for %s (|r| > %.0f%%)", count, col, WINSORISE_LIMIT * 100
            )
            n_winsorised += count
        log_ret_df[col] = log_ret_df[col].clip(-WINSORISE_LIMIT, WINSORISE_LIMIT)

    # ---- Drop tickers with insufficient observations after alignment -------
    # (a ticker could pass the pre-alignment check but still end up short
    # after the any-NaN row drop, e.g. if it was listed mid-window)
    survived = []
    dropped = list(pre_align_dropped)   # start with pre-alignment casualties
    for col in log_ret_df.columns:
        valid = log_ret_df[col].notna().sum()
        if valid >= MIN_OBSERVATIONS:
            survived.append(col)
        else:
            logger.warning(
                "Post-alignment drop: %s has only %d valid observations (min %d)",
                col, valid, MIN_OBSERVATIONS,
            )
            dropped.append(col)

    if len(survived) < 2:
        raise ValueError(
            f"Only {len(survived)} ticker(s) passed data quality checks "
            f"(minimum 2 required). Dropped: {dropped}"
        )

    log_ret_df = log_ret_df[survived]
    price_df   = price_df[survived]

    # Align price_df dates to the same rows as log_ret_df
    price_df = price_df.loc[log_ret_df.index]

    returns_matrix = log_ret_df.to_numpy().T          # (N, T)
    prices_matrix  = price_df.to_numpy().T            # (N, T)
    dates          = [d.date().isoformat() for d in log_ret_df.index]

    # ---- Dividend data (display only; returns already include dividends) ----
    # Fetch raw dividend amounts per ticker so the user can see how much of
    # their expected return comes from income vs. price appreciation.
    # This does NOT change the MCMC inputs — the log-returns above already
    # embed dividends via the adjusted close price.
    dividend_history, dividend_yields = _fetch_dividend_data(
        survived, price_df, start_date
    )

    n_payers = sum(1 for y in dividend_yields.values() if y > 0)
    logger.info(
        "Data ready: %d tickers × %d days | winsorised=%d | dropped=%s | "
        "short_history=%s | dividend payers=%d/%d",
        len(survived), returns_matrix.shape[1], n_winsorised, dropped,
        [t for t, _ in short_history], n_payers, len(survived),
    )

    return {
        "returns":          returns_matrix,
        "tickers":          list(log_ret_df.columns),
        "dates":            dates,
        "prices":           prices_matrix,
        "n_winsorised":     n_winsorised,
        "dropped_tickers":  dropped,
        "short_history":    short_history,   # list of (ticker, n_days_available)
        "dividend_yields":  dividend_yields,  # dict {ticker: trailing 12M yield}
        "dividend_history": dividend_history, # dict {ticker: pd.Series of amounts}
    }


def _fetch_dividend_data(
    tickers: list[str],
    price_df: pd.DataFrame,
    start_date: pd.Timestamp,
) -> tuple[dict, dict]:
    """Fetch raw dividend history and compute trailing 12-month yields.

    Dividends are fetched via yf.Ticker().dividends (one call per ticker).
    These are informational only — the MCMC already uses total-return prices.

    Parameters
    ----------
    tickers : list[str]
        Surviving tickers (after cleaning).
    price_df : pd.DataFrame
        Dividend-adjusted close prices (for latest-price denominator).
    start_date : pd.Timestamp
        Start of the history window (used to filter dividend history).

    Returns
    -------
    (dividend_history, dividend_yields)
        dividend_history : dict {ticker: pd.Series}  — raw per-dividend amounts
        dividend_yields  : dict {ticker: float}       — trailing 12M yield (0.0 if none)
    """
    import yfinance as yf

    dividend_history: dict = {}
    dividend_yields:  dict = {}
    cutoff_ttm = pd.Timestamp.today() - pd.Timedelta(days=365)

    for ticker in tickers:
        try:
            divs = yf.Ticker(ticker).dividends

            # yfinance dividend index is timezone-aware; normalise to tz-naive
            if divs.index.tz is not None:
                divs.index = divs.index.tz_convert(None)

            # Restrict to the fetch window for history display
            divs_window = divs[divs.index >= start_date]
            dividend_history[ticker] = divs_window

            # Trailing 12-month total cash dividend per share
            divs_ttm   = divs[divs.index >= cutoff_ttm]
            annual_div = float(divs_ttm.sum())

            # Yield = annual dividend / latest adjusted close price
            # Using the adjusted close keeps the denominator consistent with the
            # total-return prices used throughout the rest of the pipeline.
            col_prices = price_df[ticker].dropna() if ticker in price_df.columns else pd.Series(dtype=float)
            if not col_prices.empty and annual_div > 0:
                latest_price = float(col_prices.iloc[-1])
                dividend_yields[ticker] = annual_div / latest_price if latest_price > 0 else 0.0
            else:
                dividend_yields[ticker] = 0.0

            if dividend_yields[ticker] > 0:
                logger.debug(
                    "%s: trailing 12M dividend = $%.4f/share, yield = %.2f%%",
                    ticker, annual_div, dividend_yields[ticker] * 100,
                )
            else:
                logger.debug("%s: no dividends in trailing 12 months", ticker)

        except Exception as exc:
            logger.debug("Could not fetch dividends for %s: %s", ticker, exc)
            dividend_history[ticker] = pd.Series(dtype=float)
            dividend_yields[ticker]  = 0.0

    n_payers = sum(1 for y in dividend_yields.values() if y > 0)
    logger.info(
        "Dividend data fetched: %d / %d tickers paid dividends in trailing 12M. "
        "Reminder: log-returns already embed dividends via adjusted close prices.",
        n_payers, len(tickers),
    )
    return dividend_history, dividend_yields


def compute_spearman_correlation(returns_matrix: np.ndarray) -> np.ndarray:
    """Compute the Spearman rank-correlation matrix of daily log-returns.

    Parameters
    ----------
    returns_matrix : np.ndarray, shape (N, T)

    Returns
    -------
    np.ndarray, shape (N, N) — Spearman correlation matrix.
        Guaranteed to be positive-definite by adding a small ridge if needed.
    """
    N = returns_matrix.shape[0]
    logger.debug("Computing Spearman correlation for %d tickers", N)

    if N == 1:
        return np.ones((1, 1))

    corr, _ = spearmanr(returns_matrix.T)          # spearmanr expects (obs, vars)

    if N == 2:
        # spearmanr returns a scalar for 2 variables
        C = np.array([[1.0, corr], [corr, 1.0]])
    else:
        C = np.asarray(corr)

    # Ensure positive-definiteness by adding a small ridge
    min_eig = np.linalg.eigvalsh(C).min()
    if min_eig < 1e-8:
        ridge = abs(min_eig) + 1e-6
        C += ridge * np.eye(N)
        logger.debug("Added ridge %.2e to Spearman matrix to ensure PD", ridge)

    # Renormalize diagonals to 1 (ridge can perturb them slightly)
    D = np.sqrt(np.diag(C))
    C = C / D[:, None] / D[None, :]

    logger.debug("Spearman correlation matrix computed, shape %s", C.shape)
    return C
