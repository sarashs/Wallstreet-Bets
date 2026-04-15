"""
Multi-signal Short Squeeze Scanner.

Combines SEC Failure-to-Deliver data, volume/price technicals, and
LLM-powered SEC filing sentiment into a composite squeeze score.

Three-stage pipeline:
  Stage 1 (cheap, all tickers):  SEC FTD data + FMP/yfinance volume/price signals
  Stage 2 (expensive, filtered): SEC filing sentiment via 3 specialized LLM prompts
  Stage 3 (filtered):            LLM consolidation → buy/no-buy + assessment text
"""

import os
import io
import json
import time
import zipfile
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

import requests
import numpy as np
import pandas as pd
from pydantic import BaseModel
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

try:
    from edgar_ai import (squeeze_catalyst_detection, ownership_dynamics_analysis,
                          short_vulnerability_assessment)
    from edgar_extractor import (fetch_10K_and_10Q_filings, extract_items_from_filing,
                                 trim_filing_text)
except ImportError:
    from wallstreet_quant.edgar_ai import (squeeze_catalyst_detection, ownership_dynamics_analysis,
                                           short_vulnerability_assessment)
    from wallstreet_quant.edgar_extractor import (fetch_10K_and_10Q_filings, extract_items_from_filing,
                                                  trim_filing_text)

logger = logging.getLogger("ShortSqueeze")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=5, max=30))
def _fetch_json(url: str) -> dict:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _strength_to_num(s: str) -> int:
    """Map strength/severity strings to numeric values."""
    return {"strong": 3, "high": 3, "moderate": 2, "weak": 1, "low": 1, "none": 0}.get(s, 0)


# ---------------------------------------------------------------------------
# Pydantic model for LLM consolidation
# ---------------------------------------------------------------------------

class SqueezeAssessment(BaseModel):
    assessment: str
    buy_signal: bool


# ---------------------------------------------------------------------------
# Main Scanner
# ---------------------------------------------------------------------------

class ShortSqueezeScanner:
    """Multi-signal short squeeze scanner.

    Parameters
    ----------
    fmp_api_key : str or None
        Financial Modeling Prep API key.  If *None*, volume/price signals
        fall back to yfinance only.
    batch_delay : float
        Seconds to wait between API calls (rate-limiting).
    model : str
        OpenAI model used for SEC sentiment analysis.
    """

    FMP_BASE = "https://financialmodelingprep.com/api/v3"
    FTD_BASE = "https://www.sec.gov/files/data/fails-deliver-data"

    def __init__(self, fmp_api_key: Optional[str] = None,
                 batch_delay: float = 0.25,
                 model: str = "gpt-5.2-pro"):
        self.fmp_api_key = fmp_api_key
        self.batch_delay = batch_delay
        self.model = model
        self._ftd_cache: Optional[pd.DataFrame] = None
        self._openai = OpenAI()

    # ------------------------------------------------------------------
    # SEC FTD Signal Module
    # ------------------------------------------------------------------

    def _download_ftd_file(self, year: int, month: int, half: str) -> pd.DataFrame:
        """Download a single SEC FTD CSV.

        Parameters
        ----------
        year, month : int
        half : str  'a' (first half) or 'b' (second half)
        """
        ym = f"{year}{month:02d}"
        url = f"{self.FTD_BASE}/cnsfails{ym}{half}.zip"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                csv_name = zf.namelist()[0]
                with zf.open(csv_name) as f:
                    df = pd.read_csv(f, delimiter="|", dtype=str)
            # Normalise column names
            df.columns = [c.strip() for c in df.columns]
            # Parse columns
            df = df.rename(columns={
                "SETTLEMENT DATE": "date",
                "SYMBOL": "symbol",
                "QUANTITY (FAILS)": "fails",
                "PRICE": "price",
                "DESCRIPTION": "description",
                "CUSIP": "cusip",
            })
            df["fails"] = pd.to_numeric(df["fails"], errors="coerce").fillna(0).astype(int)
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
            df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
            return df
        except Exception as e:
            logger.debug(f"FTD file not available for {ym}{half}: {e}")
            return pd.DataFrame()

    def _load_recent_ftd_data(self, months_back: int = 3) -> pd.DataFrame:
        """Download and concatenate recent FTD files."""
        if self._ftd_cache is not None:
            return self._ftd_cache

        today = datetime.now()
        frames: List[pd.DataFrame] = []
        for delta in range(months_back + 1):
            dt = today - timedelta(days=30 * delta)
            y, m = dt.year, dt.month
            for half in ("a", "b"):
                df = self._download_ftd_file(y, m, half)
                if not df.empty:
                    frames.append(df)

        if frames:
            self._ftd_cache = pd.concat(frames, ignore_index=True)
        else:
            self._ftd_cache = pd.DataFrame()
        print(f"  FTD data loaded: {len(self._ftd_cache)} records across {months_back} months")
        return self._ftd_cache

    def compute_ftd_signals(self, ticker: str, ftd_df: pd.DataFrame,
                            avg_volume: float = 0) -> dict:
        """Compute FTD-based squeeze signals for a single ticker."""
        result = {
            "ftd_total_fails": 0, "ftd_avg_daily_fails": 0.0,
            "ftd_max_daily_fails": 0, "ftd_days_with_fails": 0,
            "ftd_trend": "no_data", "ftd_ratio": 0.0, "ftd_score": 0.0,
        }

        if ftd_df.empty:
            return result

        tk = ftd_df[ftd_df["symbol"].str.upper() == ticker.upper()]
        if tk.empty:
            return result

        daily = tk.groupby("date")["fails"].sum().sort_index()

        result["ftd_total_fails"] = int(daily.sum())
        result["ftd_avg_daily_fails"] = round(float(daily.mean()), 1)
        result["ftd_max_daily_fails"] = int(daily.max())
        result["ftd_days_with_fails"] = int((daily > 0).sum())

        # Trend: compare recent 2 weeks vs prior
        if len(daily) >= 10:
            mid = len(daily) // 2
            recent_mean = daily.iloc[mid:].mean()
            prior_mean = daily.iloc[:mid].mean()
            if prior_mean > 0:
                ratio = recent_mean / prior_mean
                if ratio > 1.3:
                    result["ftd_trend"] = "increasing"
                elif ratio < 0.7:
                    result["ftd_trend"] = "decreasing"
                else:
                    result["ftd_trend"] = "stable"
            else:
                result["ftd_trend"] = "increasing" if recent_mean > 0 else "stable"
        else:
            result["ftd_trend"] = "stable"

        # FTD ratio vs average volume
        if avg_volume > 0:
            result["ftd_ratio"] = round(result["ftd_avg_daily_fails"] / avg_volume, 4)

        # Score (0-30)
        score = 0.0
        # ftd_ratio component (0-15)
        score += min(result["ftd_ratio"] * 500, 15)
        # Trend bonus (0-8)
        if result["ftd_trend"] == "increasing":
            score += 8
        elif result["ftd_trend"] == "stable":
            score += 3
        # Persistence (0-7): days with fails / total days
        total_days = len(daily) if len(daily) > 0 else 1
        persistence = result["ftd_days_with_fails"] / total_days
        score += min(persistence * 7, 7)

        result["ftd_score"] = round(min(score, 30), 2)
        return result

    # ------------------------------------------------------------------
    # Volume / Price Signal Module
    # ------------------------------------------------------------------

    def _fetch_quote_fmp(self, ticker: str) -> Optional[dict]:
        if not self.fmp_api_key:
            return None
        try:
            data = _fetch_json(f"{self.FMP_BASE}/quote/{ticker}?apikey={self.fmp_api_key}")
            return data[0] if data else None
        except Exception as e:
            logger.debug(f"FMP quote failed for {ticker}: {e}")
            return None

    def _fetch_historical_fmp(self, ticker: str, days: int = 60) -> Optional[pd.DataFrame]:
        if not self.fmp_api_key:
            return None
        try:
            data = _fetch_json(
                f"{self.FMP_BASE}/historical-price-full/{ticker}?apikey={self.fmp_api_key}"
            )
            if "historical" not in data:
                return None
            df = pd.DataFrame(data["historical"][:days])
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            return df[["date", "open", "high", "low", "close", "volume"]]
        except Exception as e:
            logger.debug(f"FMP historical failed for {ticker}: {e}")
            return None

    def _fetch_historical_yfinance(self, ticker: str, days: int = 60) -> Optional[pd.DataFrame]:
        try:
            import yfinance as yf
            df = yf.download(ticker, period=f"{days + 15}d", progress=False, auto_adjust=True)
            if df.empty:
                return None
            df = df.reset_index()
            # Handle MultiIndex columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            df.columns = [c.lower() for c in df.columns]
            df = df.rename(columns={"adj close": "close"})
            return df[["date", "open", "high", "low", "close", "volume"]].tail(days).reset_index(drop=True)
        except Exception as e:
            logger.debug(f"yfinance failed for {ticker}: {e}")
            return None

    def compute_volume_price_signals(self, ticker: str) -> dict:
        """Compute volume and price-based squeeze signals."""
        result = {
            "price": 0.0, "market_cap": 0, "avg_volume": 0,
            "volume_spike_ratio": 0.0, "relative_volume_5d": 0.0,
            "price_momentum_5d": 0.0, "price_momentum_10d": 0.0,
            "bb_width": 0.0, "bb_squeeze": False, "vp_score": 0.0,
        }

        # Quote data from FMP
        quote = self._fetch_quote_fmp(ticker)
        if quote:
            result["price"] = quote.get("price", 0)
            result["market_cap"] = quote.get("marketCap", 0)
            result["avg_volume"] = quote.get("avgVolume", 0)
            avg_vol = quote.get("avgVolume", 1) or 1
            cur_vol = quote.get("volume", 0)
            result["volume_spike_ratio"] = round(cur_vol / avg_vol, 2)

        # Historical data (FMP with yfinance fallback)
        hist = self._fetch_historical_fmp(ticker)
        if hist is None or hist.empty:
            hist = self._fetch_historical_yfinance(ticker)
        if hist is None or hist.empty:
            return result

        closes = hist["close"].values
        volumes = hist["volume"].values

        # Fill quote data from history if FMP quote unavailable
        if result["price"] == 0 and len(closes) > 0:
            result["price"] = float(closes[-1])
        if result["avg_volume"] == 0 and len(volumes) > 0:
            result["avg_volume"] = int(np.mean(volumes))

        # Relative volume 5d
        if len(volumes) >= 10:
            recent_5d = np.mean(volumes[-5:])
            overall = np.mean(volumes)
            if overall > 0:
                result["relative_volume_5d"] = round(recent_5d / overall, 2)

        # Price momentum
        if len(closes) >= 6:
            result["price_momentum_5d"] = round((closes[-1] / closes[-6] - 1) * 100, 2)
        if len(closes) >= 11:
            result["price_momentum_10d"] = round((closes[-1] / closes[-11] - 1) * 100, 2)

        # Bollinger Band width & squeeze
        if len(closes) >= 20:
            sma20 = np.convolve(closes, np.ones(20) / 20, mode="valid")
            # Compute rolling std for the same window
            bb_widths = []
            for i in range(len(closes) - 19):
                window = closes[i:i + 20]
                std = np.std(window, ddof=1)
                mid = sma20[i]
                if mid > 0:
                    bb_widths.append(2 * 2 * std / mid)  # (upper-lower)/mid
                else:
                    bb_widths.append(0)
            if bb_widths:
                result["bb_width"] = round(bb_widths[-1], 4)
                # Squeeze = current width in lowest 10th percentile of its own history
                threshold = np.percentile(bb_widths, 10)
                result["bb_squeeze"] = bool(bb_widths[-1] <= threshold)

        # Volume/Price score (0-35)
        score = 0.0
        # Volume spike (0-10)
        vs = result["volume_spike_ratio"]
        if vs > 1:
            score += min((vs - 1) * 5, 10)
        # BB squeeze bonus (0-8)
        if result["bb_squeeze"]:
            score += 8
        # Momentum (0-10): use 5d momentum
        mom = result["price_momentum_5d"]
        if mom > 0:
            score += min(mom * 1.5, 10)
        # Small-cap bonus (0-7)
        mc = result["market_cap"]
        if 0 < mc < 500_000_000:
            score += 7
        elif 0 < mc < 2_000_000_000:
            score += 4

        result["vp_score"] = round(min(score, 35), 2)
        return result

    # ------------------------------------------------------------------
    # SEC Sentiment Module
    # ------------------------------------------------------------------

    def compute_sec_sentiment(self, ticker: str) -> dict:
        """Run short-squeeze-specific SEC filing analysis via LLM."""
        result = {
            "catalyst_strength": "none", "catalyst_summary": "",
            "float_pressure": "none", "ownership_summary": "",
            "vulnerability_level": "none", "vulnerability_summary": "",
            "sec_score": 0.0,
        }

        try:
            filings = fetch_10K_and_10Q_filings(
                ticker, latest_n=1,
                form=["10-K", "10-Q"],
                include_foreign=True,
            )
            if not filings:
                logger.warning(f"No SEC filing found for {ticker}")
                return result

            filing = filings[0]
            items = extract_items_from_filing(filing, ["1A", "7"])

            # Use extracted items or fall back to trimmed full filing
            item_7 = items.get("7", "")
            item_1a = items.get("1A", "")
            if not item_7 and not item_1a:
                fallback = trim_filing_text(filing)
                item_7 = item_7 or fallback
                item_1a = item_1a or fallback

            # Run the three analyses
            catalyst = squeeze_catalyst_detection(item_7 or item_1a, model=self.model)
            ownership = ownership_dynamics_analysis(item_1a or item_7, model=self.model)
            vulnerability = short_vulnerability_assessment(item_1a or item_7, model=self.model)

            result["catalyst_strength"] = catalyst.overall_catalyst_strength
            result["catalyst_summary"] = catalyst.summary
            result["float_pressure"] = ownership.float_pressure_assessment
            result["ownership_summary"] = ownership.summary
            result["vulnerability_level"] = vulnerability.overall_vulnerability
            result["vulnerability_summary"] = vulnerability.summary

            # SEC score (0-35)
            sec_score = 0.0
            sec_score += _strength_to_num(catalyst.overall_catalyst_strength) * 4   # 0-12
            sec_score += _strength_to_num(ownership.float_pressure_assessment) * 4  # 0-12
            sec_score += min(_strength_to_num(vulnerability.overall_vulnerability) * 3.67, 11)  # 0-11
            result["sec_score"] = round(min(sec_score, 35), 2)

        except Exception as e:
            logger.error(f"SEC sentiment failed for {ticker}: {e}")

        return result

    # ------------------------------------------------------------------
    # Consolidation
    # ------------------------------------------------------------------

    def _consolidate_squeeze_assessment(self, ticker: str,
                                        all_signals: dict) -> Tuple[str, bool]:
        """Use LLM to produce a final assessment and buy signal."""
        signals_json = json.dumps(all_signals, indent=2, default=str)

        prompt = f"""
        You are evaluating whether {ticker} is a short squeeze BUY candidate.

        Given the following signals (JSON), determine:
        1. A concise assessment (2-3 sentences) of the squeeze potential.
        2. A binary buy_signal: True only if the composite evidence strongly
           suggests an actionable short squeeze setup. Be CONSERVATIVE --
           require multiple confirming signals (high FTD + volume anomaly +
           positive catalyst or thin float).

        A buy_signal=True means: "there is meaningful evidence of a short squeeze
        developing and an investor should consider entering a position."

        SIGNALS:
        {signals_json}
        """
        try:
            resp = self._openai.responses.parse(
                model="o3",
                input=[
                    {"role": "system", "content":
                     "You are a rigorous quantitative analyst evaluating short squeeze setups. "
                     "Only recommend buy when multiple independent signals confirm. "
                     "False positives are costly — be conservative."},
                    {"role": "user", "content": prompt}
                ],
                text_format=SqueezeAssessment,
            ).output_parsed
            return resp.assessment, resp.buy_signal
        except Exception as e:
            logger.error(f"Consolidation failed for {ticker}: {e}")
            return f"Assessment failed: {e}", False

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def scan(self, tickers: List[str],
             ftd_score_threshold: float = 25.0,
             run_sec_sentiment: bool = True) -> pd.DataFrame:
        """Run the full multi-signal short squeeze scan.

        Parameters
        ----------
        tickers : list[str]
            Universe of tickers to scan.
        ftd_score_threshold : float
            Minimum (ftd_score + vp_score) to advance to Stage 2.
        run_sec_sentiment : bool
            If True, run expensive LLM sentiment on filtered candidates.
        """
        print(f"{'='*80}")
        print(f"SHORT SQUEEZE SCANNER — {len(tickers)} tickers")
        print(f"{'='*80}")

        # Stage 1: FTD + Volume/Price (cheap)
        print("\n[Stage 1] Loading SEC FTD data...")
        ftd_df = self._load_recent_ftd_data(months_back=3)

        print(f"[Stage 1] Computing FTD + volume/price signals for {len(tickers)} tickers...")
        stage1_results = []
        for i, ticker in enumerate(tickers):
            if i % 50 == 0:
                print(f"  Progress: {i}/{len(tickers)} ({len(stage1_results)} processed)")

            vp = self.compute_volume_price_signals(ticker)
            ftd = self.compute_ftd_signals(ticker, ftd_df, avg_volume=vp.get("avg_volume", 0))
            preliminary_score = ftd["ftd_score"] + vp["vp_score"]

            row = {"ticker": ticker, **ftd, **vp, "preliminary_score": preliminary_score}
            stage1_results.append(row)
            time.sleep(self.batch_delay)

        df = pd.DataFrame(stage1_results)
        if df.empty:
            print("No results from Stage 1.")
            return df

        # Filter for Stage 2
        candidates = df[df["preliminary_score"] >= ftd_score_threshold].copy()
        print(f"\n[Stage 1] Complete. {len(candidates)}/{len(df)} tickers passed "
              f"threshold ({ftd_score_threshold}).")

        if candidates.empty or not run_sec_sentiment:
            # Fill SEC columns with defaults
            for col in ["catalyst_strength", "catalyst_summary", "float_pressure",
                        "ownership_summary", "vulnerability_level", "vulnerability_summary",
                        "sec_score", "composite_score", "assessment", "buy_signal"]:
                candidates[col] = "" if col.endswith("summary") or col == "assessment" else (
                    False if col == "buy_signal" else 0.0 if "score" in col else "none"
                )
            return candidates.sort_values("preliminary_score", ascending=False).reset_index(drop=True)

        # Stage 2: SEC sentiment (expensive, filtered only)
        print(f"\n[Stage 2] Running SEC sentiment analysis on {len(candidates)} candidates...")
        sec_results = {}
        for i, ticker in enumerate(candidates["ticker"]):
            print(f"  [{i+1}/{len(candidates)}] Analyzing {ticker}...")
            sec_results[ticker] = self.compute_sec_sentiment(ticker)

        # Merge SEC results
        sec_df = pd.DataFrame.from_dict(sec_results, orient="index")
        sec_df.index.name = "ticker"
        sec_df = sec_df.reset_index()
        candidates = candidates.merge(sec_df, on="ticker", how="left")

        # Composite score
        candidates["composite_score"] = (
            candidates["ftd_score"] + candidates["vp_score"] +
            candidates["sec_score"].fillna(0)
        ).clip(upper=100).round(2)

        # Stage 3: LLM consolidation
        print(f"\n[Stage 3] Generating assessments for {len(candidates)} candidates...")
        assessments = []
        buy_signals = []
        for _, row in candidates.iterrows():
            signals = {
                "ticker": row["ticker"],
                "ftd_score": row["ftd_score"],
                "ftd_ratio": row["ftd_ratio"],
                "ftd_trend": row["ftd_trend"],
                "ftd_days_with_fails": row["ftd_days_with_fails"],
                "vp_score": row["vp_score"],
                "volume_spike_ratio": row["volume_spike_ratio"],
                "bb_squeeze": row["bb_squeeze"],
                "price_momentum_5d": row["price_momentum_5d"],
                "price_momentum_10d": row["price_momentum_10d"],
                "sec_score": row.get("sec_score", 0),
                "catalyst_strength": row.get("catalyst_strength", "none"),
                "float_pressure": row.get("float_pressure", "none"),
                "vulnerability_level": row.get("vulnerability_level", "none"),
                "composite_score": row["composite_score"],
            }
            assessment, buy = self._consolidate_squeeze_assessment(row["ticker"], signals)
            assessments.append(assessment)
            buy_signals.append(buy)

        candidates["assessment"] = assessments
        candidates["buy_signal"] = buy_signals

        # Final ordering
        col_order = [
            "ticker", "price", "market_cap",
            "ftd_total_fails", "ftd_avg_daily_fails", "ftd_ratio", "ftd_trend", "ftd_score",
            "volume_spike_ratio", "relative_volume_5d",
            "price_momentum_5d", "price_momentum_10d",
            "bb_width", "bb_squeeze", "vp_score",
            "catalyst_strength", "float_pressure", "vulnerability_level", "sec_score",
            "composite_score", "assessment", "buy_signal",
        ]
        # Keep only columns that exist
        col_order = [c for c in col_order if c in candidates.columns]
        extra_cols = [c for c in candidates.columns if c not in col_order]
        candidates = candidates[col_order + extra_cols]

        candidates = candidates.sort_values("composite_score", ascending=False).reset_index(drop=True)
        print(f"\n{'='*80}")
        print(f"Scan complete. {candidates['buy_signal'].sum()} buy signals out of "
              f"{len(candidates)} candidates.")
        print(f"{'='*80}")
        return candidates

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def save_results(self, df: pd.DataFrame, output_dir: str = ".") -> str:
        """Save results to Excel with two sheets."""
        today = datetime.now().strftime("%Y-%m-%d")
        filepath = os.path.join(output_dir, f"short_squeeze_{today}.xlsx")

        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            # Sheet 1: Buy candidates only
            buys = df[df["buy_signal"] == True].copy() if "buy_signal" in df.columns else pd.DataFrame()
            if not buys.empty:
                buys.to_excel(writer, sheet_name="Candidates", index=False)
            else:
                pd.DataFrame({"info": ["No buy candidates found"]}).to_excel(
                    writer, sheet_name="Candidates", index=False)

            # Sheet 2: All screened tickers
            df.to_excel(writer, sheet_name="All Screened", index=False)

        print(f"Results saved to: {filepath}")
        return filepath
