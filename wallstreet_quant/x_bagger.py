"""
X-Bagger Finder & Forward Screener.

Four-stage pipeline:
  Stage 1 (cheap, all tickers):     yfinance max-period download → max-drawup
                                    multiples over 5y / 10y / 20y windows.
  Stage 2 (expensive, filtered):    Per-winner LLM deep-dive: business model
                                    (web search) + SEC inflection analysis +
                                    macro context (web search). Disk-cached.
  Stage 3 (single call, filtered):  LLM consolidation of winner profiles into
                                    a `WinnerPatterns` fingerprint.
  Stage 4 (forward screener):       Rule-based pre-filter on the current
                                    universe (market cap, sector) followed by
                                    LLM scoring (0-100) against the pattern.
"""

import os
import json
import time
import pickle
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

try:
    from edgar_ai import (winner_business_model_analysis, winner_inflection_analysis,
                          winner_market_context, consolidate_winner_patterns,
                          score_against_pattern)
    from edgar_extractor import (fetch_10K_and_10Q_filings, extract_items_from_filing,
                                 trim_filing_text)
except ImportError:
    from wallstreet_quant.edgar_ai import (winner_business_model_analysis, winner_inflection_analysis,
                                           winner_market_context, consolidate_winner_patterns,
                                           score_against_pattern)
    from wallstreet_quant.edgar_extractor import (fetch_10K_and_10Q_filings, extract_items_from_filing,
                                                  trim_filing_text)

logger = logging.getLogger("XBagger")


# ---------------------------------------------------------------------------
# Main Scanner
# ---------------------------------------------------------------------------

class XBaggerScanner:
    """Find historical multi-baggers and screen current stocks against their pattern.

    Parameters
    ----------
    batch_delay : float
        Seconds between yfinance calls (rate-limiting).
    model : str
        OpenAI model used for non-consolidation analyses (default `gpt-5.2-pro`).
    consolidation_model : str
        OpenAI model used for the pattern-consolidation call (default `o3`).
    cache_dir : str or None
        Directory for caching Stage 2 winner-profile results so re-runs don't
        repay LLM cost. If *None*, caching is disabled.
    """

    WINDOWS = (5, 10, 20)  # years

    def __init__(self,
                 batch_delay: float = 0.25,
                 model: str = "gpt-5.2-pro",
                 consolidation_model: str = "o3",
                 cache_dir: Optional[str] = None):
        self.batch_delay = batch_delay
        self.model = model
        self.consolidation_model = consolidation_model
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Stage 1 — Historical scan
    # ------------------------------------------------------------------

    @staticmethod
    def _max_drawup(prices: np.ndarray, dates: pd.DatetimeIndex,
                    window_days: int,
                    volumes: Optional[np.ndarray] = None,
                    min_trough_price: float = 0.0,
                    min_trough_dollar_volume: float = 0.0) -> Dict[str, float]:
        """Maximum (peak / prior-trough) ratio inside a rolling window.

        For every t in the series, look back `window_days` and find
        max(close[t-W:t+1]) / min(close[t-W:t+1]). Return the largest such
        ratio across the whole series, plus the peak/trough dates.

        Quality filters (highly recommended to suppress penny-stock noise):
        - `min_trough_price`: skip troughs trading below this adjusted price.
          Most "1000x" runs in the SEC universe are sub-penny stocks going
          $0.0001 → $0.1; setting this to e.g. $1 kills that noise.
        - `min_trough_dollar_volume`: requires `volumes` array. Skip troughs
          where the 30-day-centered average dollar volume (price × volume)
          is below this threshold. Filters illiquid pump-and-dumps.
        """
        if len(prices) < 2:
            return {"multiple": 1.0, "peak_date": None, "trough_date": None,
                    "trough_price": None, "peak_price": None}

        best = {"multiple": 1.0, "peak_date": None, "trough_date": None,
                "trough_price": None, "peak_price": None}
        n = len(prices)

        # Pre-compute centered 30-day average dollar volume if needed
        avg_dv = None
        if volumes is not None and min_trough_dollar_volume > 0:
            dv = prices * volumes
            kernel = np.ones(30) / 30
            avg_dv = np.convolve(dv, kernel, mode="same")

        for end in range(1, n):
            start = max(0, end - window_days)
            window_slice = prices[start:end + 1]
            if len(window_slice) < 2:
                continue

            # Mask invalid trough positions to inf
            masked = window_slice.astype(float).copy()
            if min_trough_price > 0:
                masked = np.where(masked >= min_trough_price, masked, np.inf)
            if avg_dv is not None:
                window_dv = avg_dv[start:end + 1]
                masked = np.where(window_dv >= min_trough_dollar_volume,
                                  masked, np.inf)
            # The last position cannot be a trough (no peak after)
            masked[-1] = np.inf
            if not np.isfinite(masked).any():
                continue

            min_idx = int(np.argmin(masked))
            if not np.isfinite(masked[min_idx]):
                continue
            after_trough = window_slice[min_idx:]
            max_idx_local = int(np.argmax(after_trough))
            trough_price = float(after_trough[0])
            peak_price = float(after_trough[max_idx_local])
            if trough_price <= 0:
                continue
            multiple = peak_price / trough_price
            if multiple > best["multiple"]:
                best["multiple"] = float(multiple)
                best["trough_date"] = dates[start + min_idx].date().isoformat()
                best["peak_date"] = dates[start + min_idx + max_idx_local].date().isoformat()
                best["trough_price"] = trough_price
                best["peak_price"] = peak_price
        return best

    def find_historical_winners(self, tickers: List[str],
                                windows: Optional[List[int]] = None,
                                min_trough_price: float = 1.0,
                                min_trough_dollar_volume: float = 1e6) -> pd.DataFrame:
        """Stage 1: yfinance scan for max-drawup multiples over each window.

        Quality filters default to:
        - `min_trough_price=1.0` (drops sub-$1 penny-stock troughs)
        - `min_trough_dollar_volume=1e6` (drops trough days with <$1M average
          dollar volume — kills illiquid junk)
        """
        import yfinance as yf

        if windows is None:
            windows = list(self.WINDOWS)
        trading_days = {y: y * 252 for y in windows}

        print(f"[Stage 1] Scanning {len(tickers)} tickers for {windows}-year drawup multiples...")
        print(f"  Quality filters: min_trough_price=${min_trough_price}, "
              f"min_trough_dollar_volume=${min_trough_dollar_volume:,.0f}")
        rows = []
        for i, ticker in enumerate(tickers):
            if i % 100 == 0:
                print(f"  Progress: {i}/{len(tickers)}")
            try:
                df = yf.download(ticker, period="max", progress=False,
                                 auto_adjust=True, threads=False)
                if df.empty or len(df) < 252:
                    continue
                # Handle MultiIndex columns from yfinance
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                closes = df["Close"].values.astype(float)
                volumes = df["Volume"].values.astype(float) if "Volume" in df.columns else None
                dates = df.index

                row = {"ticker": ticker, "history_days": len(closes),
                       "first_date": dates[0].date().isoformat(),
                       "last_date": dates[-1].date().isoformat(),
                       "current_price": round(float(closes[-1]), 4)}
                for y in windows:
                    res = self._max_drawup(closes, dates, trading_days[y],
                                           volumes=volumes,
                                           min_trough_price=min_trough_price,
                                           min_trough_dollar_volume=min_trough_dollar_volume)
                    row[f"max_{y}y"] = round(res["multiple"], 2)
                    row[f"trough_date_{y}y"] = res["trough_date"]
                    row[f"peak_date_{y}y"] = res["peak_date"]
                    row[f"trough_price_{y}y"] = (round(res["trough_price"], 4)
                                                 if res["trough_price"] is not None else None)
                rows.append(row)
            except Exception as e:
                logger.debug(f"Stage 1 failed for {ticker}: {e}")
            time.sleep(self.batch_delay)

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(f"max_{windows[-1]}y", ascending=False).reset_index(drop=True)
        print(f"[Stage 1] Complete. {len(df)} tickers with usable history.")
        return df

    def revalidate_stage1(self, stage1_df: pd.DataFrame,
                          min_trough_price: float = 1.0,
                          min_trough_dollar_volume: float = 1e6,
                          windows: Optional[List[int]] = None) -> pd.DataFrame:
        """Re-run Stage 1 math on existing tickers with new quality filters.

        Use this to clean up an existing Stage 1 CSV without re-paying yfinance
        cost across the whole universe — only the tickers in `stage1_df` are
        re-fetched. Useful when you discover penny-stock noise in your results
        and want to apply stricter trough-quality filters.
        """
        return self.find_historical_winners(
            stage1_df["ticker"].tolist(),
            windows=windows,
            min_trough_price=min_trough_price,
            min_trough_dollar_volume=min_trough_dollar_volume,
        )

    @staticmethod
    def filter_winners(stage1_df: pd.DataFrame, window_years: int,
                       min_multiple: float) -> pd.DataFrame:
        """Apply a (window, multiple) gate to Stage 1 output."""
        col = f"max_{window_years}y"
        if col not in stage1_df.columns:
            raise ValueError(f"Stage 1 dataframe has no column '{col}'")
        return stage1_df[stage1_df[col] >= min_multiple].copy()

    # ------------------------------------------------------------------
    # Stage 2 — Winner deep-dive
    # ------------------------------------------------------------------

    def _cache_path(self, ticker: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        return os.path.join(self.cache_dir, f"{ticker}.pkl")

    def analyze_winner(self, ticker: str, trough_date: str, peak_date: str) -> dict:
        """Run business-model + inflection + macro analyses for a single winner."""
        cache_path = self._cache_path(ticker)
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        profile = {"ticker": ticker, "trough_date": trough_date, "peak_date": peak_date}

        # Business model + moat (web search)
        try:
            bm = winner_business_model_analysis(ticker, model=self.model)
            profile["business_model"] = bm.model_dump()
        except Exception as e:
            logger.warning(f"business_model failed for {ticker}: {e}")
            profile["business_model"] = {}

        # Inflection point from SEC filing near the trough
        try:
            filing_text = self._fetch_filing_near(ticker, trough_date)
            if filing_text:
                inf = winner_inflection_analysis(ticker, filing_text, trough_date,
                                                 peak_date, model=self.model)
                profile["inflection"] = inf.model_dump()
            else:
                profile["inflection"] = {}
        except Exception as e:
            logger.warning(f"inflection failed for {ticker}: {e}")
            profile["inflection"] = {}

        # Macro / sector context (web search)
        try:
            mc = winner_market_context(ticker, trough_date, peak_date, model=self.model)
            profile["market_context"] = mc.model_dump()
        except Exception as e:
            logger.warning(f"market_context failed for {ticker}: {e}")
            profile["market_context"] = {}

        if cache_path:
            with open(cache_path, "wb") as f:
                pickle.dump(profile, f)
        return profile

    @staticmethod
    def _fetch_filing_near(ticker: str, trough_date: str) -> str:
        """Fetch the 10-K or 10-Q closest to (and not after) the trough date."""
        try:
            trough = datetime.fromisoformat(trough_date).date()
        except Exception:
            return ""
        # Look at filings within +/- 1 year of the trough
        start = (trough - timedelta(days=365)).isoformat()
        end = (trough + timedelta(days=365)).isoformat()
        filings = fetch_10K_and_10Q_filings(
            ticker, start_date=start, end_date=end,
            form=["10-K", "10-Q", "20-F"], include_foreign=True,
        )
        if not filings:
            return ""
        filing = filings[0]
        try:
            items = extract_items_from_filing(filing, ["1A", "7"])
            text = items.get("7", "") or items.get("1A", "")
            return text or trim_filing_text(filing)
        except Exception:
            return trim_filing_text(filing)

    def analyze_winners(self, winners_df: pd.DataFrame, window_years: int) -> List[dict]:
        """Stage 2: run analyze_winner() for every row in `winners_df`."""
        trough_col = f"trough_date_{window_years}y"
        peak_col = f"peak_date_{window_years}y"
        print(f"[Stage 2] Analysing {len(winners_df)} winners...")
        profiles = []
        for i, row in winners_df.reset_index(drop=True).iterrows():
            ticker = row["ticker"]
            print(f"  [{i+1}/{len(winners_df)}] Analysing {ticker}...")
            profile = self.analyze_winner(
                ticker,
                trough_date=row[trough_col],
                peak_date=row[peak_col],
            )
            profile[f"max_{window_years}y"] = row[f"max_{window_years}y"]
            profiles.append(profile)
        return profiles

    # ------------------------------------------------------------------
    # Stage 3 — Pattern extraction
    # ------------------------------------------------------------------

    def extract_patterns(self, winner_profiles: List[dict]):
        """Stage 3: consolidate winner profiles into a single pattern fingerprint."""
        print(f"[Stage 3] Extracting patterns from {len(winner_profiles)} winner profiles...")
        profiles_json = json.dumps(winner_profiles, indent=2, default=str)
        patterns = consolidate_winner_patterns(profiles_json, model=self.consolidation_model)
        return patterns

    # ------------------------------------------------------------------
    # Stage 4 — Forward screener
    # ------------------------------------------------------------------

    @staticmethod
    def _quick_info(ticker: str) -> dict:
        """Cheap yfinance .info for pre-filtering."""
        import yfinance as yf
        try:
            info = yf.Ticker(ticker).info
            return {
                "ticker": ticker,
                "market_cap": info.get("marketCap") or 0,
                "sector": info.get("sector") or "",
                "industry": info.get("industry") or "",
                "float_shares": info.get("floatShares") or 0,
                "currency": info.get("currency") or "",
            }
        except Exception:
            return {"ticker": ticker, "market_cap": 0, "sector": "", "industry": "",
                    "float_shares": 0, "currency": ""}

    def prefilter_universe(self, tickers: List[str], patterns,
                           max_market_cap: float = 5e9,
                           sector_whitelist: Optional[List[str]] = None) -> pd.DataFrame:
        """Cheap yfinance pre-filter to drop obvious mismatches before LLM scoring."""
        print(f"[Stage 4a] Pre-filtering {len(tickers)} tickers...")
        rows = []
        for i, t in enumerate(tickers):
            if i % 200 == 0:
                print(f"  Pre-filter progress: {i}/{len(tickers)}")
            rows.append(self._quick_info(t))
            time.sleep(self.batch_delay)
        info_df = pd.DataFrame(rows)

        mask = (info_df["market_cap"] > 0) & (info_df["market_cap"] <= max_market_cap)
        if sector_whitelist:
            mask &= info_df["sector"].isin(sector_whitelist)
        kept = info_df[mask].copy()
        print(f"[Stage 4a] Pre-filter kept {len(kept)}/{len(info_df)} tickers "
              f"(market cap <= ${max_market_cap/1e9:.1f}B"
              f"{', sector in ' + str(sector_whitelist) if sector_whitelist else ''}).")
        return kept

    def score_current_stocks(self, tickers: List[str], patterns,
                             max_market_cap: float = 5e9,
                             sector_whitelist: Optional[List[str]] = None) -> pd.DataFrame:
        """Stage 4: pre-filter then LLM-score against the winner pattern."""
        kept = self.prefilter_universe(tickers, patterns,
                                       max_market_cap=max_market_cap,
                                       sector_whitelist=sector_whitelist)
        if kept.empty:
            return kept

        patterns_json = json.dumps(patterns.model_dump() if hasattr(patterns, "model_dump")
                                   else patterns, indent=2, default=str)

        print(f"[Stage 4b] LLM-scoring {len(kept)} pre-filtered tickers...")
        scores = []
        for i, ticker in enumerate(kept["ticker"]):
            if i % 25 == 0:
                print(f"  Scoring progress: {i}/{len(kept)}")
            try:
                s = score_against_pattern(ticker, patterns_json, model=self.model)
                scores.append({
                    "ticker": ticker,
                    "score": s.score,
                    "matched_traits": "; ".join(s.matched_traits),
                    "missing_traits": "; ".join(s.missing_traits),
                    "reasoning": s.reasoning,
                })
            except Exception as e:
                logger.warning(f"Stage 4 scoring failed for {ticker}: {e}")
                scores.append({"ticker": ticker, "score": 0, "matched_traits": "",
                               "missing_traits": "", "reasoning": f"error: {e}"})

        scores_df = pd.DataFrame(scores)
        out = kept.merge(scores_df, on="ticker", how="left")
        return out.sort_values("score", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def scan(self, tickers: List[str],
             window_years: int = 10,
             min_multiple: float = 100.0,
             score_universe: Optional[List[str]] = None,
             max_market_cap: float = 5e9,
             sector_whitelist: Optional[List[str]] = None) -> dict:
        """Run all four stages end-to-end. Returns a dict with intermediate outputs.

        Use this for one-shot runs. For tuning, prefer driving the stages
        individually from a notebook so you don't repay Stage 1 / Stage 2 cost.
        """
        print(f"{'='*80}")
        print(f"X-BAGGER SCANNER — {len(tickers)} tickers")
        print(f"Threshold: {min_multiple}x in {window_years}y")
        print(f"{'='*80}")

        stage1 = self.find_historical_winners(tickers)
        winners = self.filter_winners(stage1, window_years, min_multiple)
        if winners.empty:
            print(f"\nNo tickers cleared {min_multiple}x in {window_years}y. Exiting.")
            return {"stage1": stage1, "winners": winners, "profiles": [],
                    "patterns": None, "candidates": pd.DataFrame()}

        profiles = self.analyze_winners(winners, window_years)
        patterns = self.extract_patterns(profiles)

        candidates = pd.DataFrame()
        if score_universe is not None:
            candidates = self.score_current_stocks(
                score_universe, patterns,
                max_market_cap=max_market_cap,
                sector_whitelist=sector_whitelist,
            )

        return {"stage1": stage1, "winners": winners, "profiles": profiles,
                "patterns": patterns, "candidates": candidates}

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def save_results(self, results: dict, output_dir: str = ".") -> str:
        """Save all stage outputs to a single dated Excel."""
        today = datetime.now().strftime("%Y-%m-%d")
        filepath = os.path.join(output_dir, f"x_bagger_{today}.xlsx")

        patterns = results.get("patterns")
        if hasattr(patterns, "model_dump"):
            patterns_dict = patterns.model_dump()
        else:
            patterns_dict = patterns or {}

        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            stage1 = results.get("stage1", pd.DataFrame())
            if not stage1.empty:
                stage1.to_excel(writer, sheet_name="Historical Scan", index=False)

            winners = results.get("winners", pd.DataFrame())
            if not winners.empty:
                winners.to_excel(writer, sheet_name="Winners", index=False)

            profiles = results.get("profiles", [])
            if profiles:
                # Flatten the nested dict for Excel readability
                flat = []
                for p in profiles:
                    row = {"ticker": p["ticker"],
                           "trough_date": p.get("trough_date"),
                           "peak_date": p.get("peak_date")}
                    for sub in ("business_model", "inflection", "market_context"):
                        sub_dict = p.get(sub, {}) or {}
                        for k, v in sub_dict.items():
                            row[f"{sub}.{k}"] = (json.dumps(v) if isinstance(v, list)
                                                 else v)
                    flat.append(row)
                pd.DataFrame(flat).to_excel(writer, sheet_name="Winner Profiles", index=False)

            if patterns_dict:
                pd.DataFrame([{k: (json.dumps(v) if isinstance(v, list) else v)
                               for k, v in patterns_dict.items()}]).to_excel(
                    writer, sheet_name="Patterns", index=False)

            candidates = results.get("candidates", pd.DataFrame())
            if isinstance(candidates, pd.DataFrame) and not candidates.empty:
                candidates.to_excel(writer, sheet_name="Forward Candidates", index=False)

        print(f"Results saved to: {filepath}")
        return filepath
