from typing import List, Dict
from pydantic import BaseModel
from enum import Enum
import pandas as pd
from tqdm import tqdm
import logging
from datetime import datetime
import os

# --- Setup logging with today's date as filename --- #
def setup_pipeline_logger():
    """Configure logging to file with today's date."""
    log_filename = datetime.now().strftime("%Y-%m-%d") + ".log"
    
    # Create logger
    logger = logging.getLogger("SecAnalysis")
    logger.setLevel(logging.DEBUG)
    
    # Remove any existing handlers to avoid duplicates
    logger.handlers = []
    
    # File handler - detailed logging
    file_handler = logging.FileHandler(log_filename, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_format)
    
    # Console handler - info level only
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_format)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Initialize logger
logger = setup_pipeline_logger()

try:
    from edgar_ai import *
    from edgar_extractor import extract_items_from_filing
    from edgar_extractor import fetch_10K_and_10Q_filings
    from edgar_extractor import trim_filing_text, llm_extract_sections
except:
    from wallstreet_quant.edgar_ai import *
    from wallstreet_quant.edgar_extractor import extract_items_from_filing
    from wallstreet_quant.edgar_extractor import fetch_10K_and_10Q_filings
    from wallstreet_quant.edgar_extractor import trim_filing_text, llm_extract_sections
import json
from openai import OpenAI

# Initialize client
client = OpenAI()

# Assumes your analysis functions are already defined in this file or imported elsewhere

# --------------------------------------------------------------------------- #
#  SEC-ANALYSIS WRAPPER CLASS                                                 #
# --------------------------------------------------------------------------- #
# --- define a consistent placeholder for failed extractions --- #
EXTRACTION_FAILED = {"error": "could not be extracted"}
class SecAnalysis:
    """
    High-level orchestrator.  Call an instance with a mapping:
        {ticker: [filing_obj_curr, filing_obj_prev], …}
    Each filing_obj must expose the minimal attributes accessed below.
    """

    def __call__(self, filings: Dict[str, List[object]], model="gpt-5.2-pro") -> pd.DataFrame:
        rows = []
        total_tickers = len(filings)
        skipped_no_filings = 0
        single_filing_mode = 0
        successful = 0
        failed = 0
        
        logger.info("=" * 60)
        logger.info(f"SEC ANALYSIS STARTED - Processing {total_tickers} tickers")
        logger.info(f"Model: {model}")
        logger.info("=" * 60)
        
        for ticker, flist in tqdm(filings.items()):
            logger.info(f"\n{'─' * 40}")
            logger.info(f"PROCESSING: {ticker}")
            
            # Skip tickers with no filings at all
            if not flist or len(flist) == 0:
                logger.warning(f"{ticker}: No filings found - SKIPPED (may not be a SEC filer)")
                skipped_no_filings += 1
                continue
            
            # For comparison analysis, we need at least 2 filings
            # If only 1 filing exists, use it for both current and previous (limited comparison)
            if len(flist) < 2:
                logger.warning(f"{ticker}: Only 1 filing found - using single filing mode (limited comparison)")
                single_filing_mode += 1
                flist = [flist[0], flist[0]]  # Use same filing for both
            
            # Log filing details
            try:
                logger.debug(f"{ticker}: Current filing: {flist[0].form} dated {flist[0].filing_date}")
                logger.debug(f"{ticker}: Previous filing: {flist[1].form} dated {flist[1].filing_date}")
            except Exception as e:
                logger.debug(f"{ticker}: Could not log filing details: {e}")

            # Pull the key items once
            items_needed = ['1', '1A', '3', '7', '9A']     # add others if your extractor supports them
            try:
                curr = extract_items_from_filing(flist[0], items_needed)   # latest filing
                logger.debug(f"{ticker}: Current filing - extracted items: {list(curr.keys())}")
            except Exception as e:
                logger.error(f"{ticker}: Recent filing extraction FAILED: {e}")
                # Use empty dicts to trigger fallback to whole filing
                curr = {}
            try:
                prev = extract_items_from_filing(flist[1], items_needed)   # older filing
                logger.debug(f"{ticker}: Previous filing - extracted items: {list(prev.keys())}")
            except Exception as e:
                logger.error(f"{ticker}: Previous filing extraction FAILED: {e}")
                # Use empty dicts to trigger fallback to whole filing
                prev = {}
            
            # --- LLM extraction fallback for missing sections ----------- #
            missing_curr = [item for item in items_needed if item not in curr or not curr[item]]
            missing_prev = [item for item in items_needed if item not in prev or not prev[item]]

            # Cache trimmed texts (lazy — only computed if needed)
            trimmed_curr = None
            trimmed_prev = None

            if missing_curr or missing_prev:
                trimmed_curr = trim_filing_text(flist[0])
                if not trimmed_curr:
                    logger.warning(f"{ticker}: Could not extract any text from current filing")
            if missing_prev:
                trimmed_prev = trim_filing_text(flist[1])
                if not trimmed_prev:
                    logger.warning(f"{ticker}: Could not extract any text from previous filing")

            # Single LLM call per filing to extract all missing sections
            if missing_curr and trimmed_curr:
                logger.info(f"{ticker}: Regex missed {missing_curr} — LLM extracting from trimmed text")
                try:
                    llm_sections = llm_extract_sections(trimmed_curr, missing_curr)
                    curr.update(llm_sections)
                except Exception as e:
                    logger.warning(f"{ticker}: LLM section extraction failed for curr: {e}")

            if missing_prev and trimmed_prev:
                logger.info(f"{ticker}: Regex missed {missing_prev} in prev filing — LLM extracting")
                try:
                    llm_sections = llm_extract_sections(trimmed_prev, missing_prev)
                    prev.update(llm_sections)
                except Exception as e:
                    logger.warning(f"{ticker}: LLM section extraction failed for prev: {e}")

            # --- section analyses with fallback to trimmed text ---------- #
            logger.debug(f"{ticker}: Starting LLM analysis...")

            try:
                if '1A' in prev and '1A' in curr and prev['1A'] and curr['1A']:
                    logger.debug(f"{ticker}: Risk Factor analysis - using extracted sections")
                    rf = risk_factor_analysis(prev['1A'], curr['1A'], model=model)
                else:
                    logger.warning(f"{ticker}: Risk Factor sections missing - using trimmed filing fallback")
                    rf = risk_factor_analysis(
                        trimmed_prev or trim_filing_text(flist[1]),
                        trimmed_curr or trim_filing_text(flist[0]),
                        model=model)
                logger.info(f"{ticker}: ✓ Risk Factor analysis completed")
            except Exception as e:
                logger.error(f"{ticker}: ✗ Risk Factor Analysis FAILED: {e}")
                rf = EXTRACTION_FAILED

            try:
                if '7' in curr and curr['7']:
                    logger.debug(f"{ticker}: MD&A analysis - using extracted section")
                    mda = mdad_analysis(curr['7'], model=model)
                else:
                    logger.warning(f"{ticker}: MD&A section missing - using trimmed filing fallback")
                    mda = mdad_analysis(trimmed_curr or trim_filing_text(flist[0]), model=model)
                logger.info(f"{ticker}: ✓ MD&A analysis completed")
            except Exception as e:
                logger.error(f"{ticker}: ✗ MD&A Analysis FAILED: {e}")
                mda = EXTRACTION_FAILED

            try:
                if '3' in curr and curr['3']:
                    logger.debug(f"{ticker}: Legal analysis - using extracted section")
                    leg = legal_matters(curr['3'], model=model)
                else:
                    logger.warning(f"{ticker}: Legal section missing - using trimmed filing fallback")
                    leg = legal_matters(trimmed_curr or trim_filing_text(flist[0]), model=model)
                logger.info(f"{ticker}: ✓ Legal analysis completed")
            except Exception as e:
                logger.error(f"{ticker}: ✗ Legal Matters Analysis FAILED: {e}")
                leg = EXTRACTION_FAILED

            try:
                if '9A' in curr and curr['9A']:
                    logger.debug(f"{ticker}: Controls analysis - using extracted section")
                    ctrl = control_status(curr['9A'], model=model)
                else:
                    logger.warning(f"{ticker}: Controls section missing - using trimmed filing fallback")
                    ctrl = control_status(trimmed_curr or trim_filing_text(flist[0]), model=model)
                logger.info(f"{ticker}: ✓ Controls analysis completed")
            except Exception as e:
                logger.error(f"{ticker}: ✗ Control Status Analysis FAILED: {e}")
                ctrl = EXTRACTION_FAILED

            try:
                if '1' in prev and '1' in curr and prev['1'] and curr['1']:
                    logger.debug(f"{ticker}: Business analysis - using extracted sections")
                    biz = business_info(prev['1'], curr['1'], model=model)
                else:
                    logger.warning(f"{ticker}: Business sections missing - using trimmed filing fallback")
                    biz = business_info(
                        trimmed_prev or trim_filing_text(flist[1]),
                        trimmed_curr or trim_filing_text(flist[0]),
                        model=model)
                logger.info(f"{ticker}: ✓ Business analysis completed")
            except Exception as e:
                logger.error(f"{ticker}: ✗ Business Info Analysis FAILED: {e}")
                biz = EXTRACTION_FAILED

            try:
                if '7' in prev and '7' in curr and prev['7'] and curr['7']:
                    tone = tone_shift_analysis(prev['7'], curr['7'], model=model)
                else:
                    logger.warning(f"{ticker}: MD&A sections missing for tone analysis - using trimmed filing fallback")
                    tone = tone_shift_analysis(
                        trimmed_prev or trim_filing_text(flist[1]),
                        trimmed_curr or trim_filing_text(flist[0]),
                        model=model)
                logger.info(f"{ticker}: ✓ Tone analysis completed")
            except Exception as e:
                logger.error(f"{ticker}: ✗ Tone Shift Analysis FAILED: {e}")
                tone = EXTRACTION_FAILED

            try:
                if '7' in curr and curr['7']:
                    logger.debug(f"{ticker}: Strategy analysis - using extracted section")
                    strat = strategy_summary_analysis(curr['7'], model=model)
                else:
                    logger.warning(f"{ticker}: MD&A section missing for strategy - using trimmed filing fallback")
                    strat = strategy_summary_analysis(trimmed_curr or trim_filing_text(flist[0]), model=model)
                logger.info(f"{ticker}: ✓ Strategy analysis completed")
            except Exception as e:
                logger.error(f"{ticker}: ✗ Strategy Summary Analysis FAILED: {e}")
                strat = EXTRACTION_FAILED

            try:
                if '1' in curr and curr['1']:
                    logger.debug(f"{ticker}: Human capital analysis - using extracted section")
                    hc = human_capital_analysis(curr['1'], model=model)
                else:
                    logger.warning(f"{ticker}: Business section missing for human capital - using trimmed filing fallback")
                    hc = human_capital_analysis(trimmed_curr or trim_filing_text(flist[0]), model=model)
                logger.info(f"{ticker}: ✓ Human capital analysis completed")
            except Exception as e:
                logger.error(f"{ticker}: ✗ Human Capital Analysis FAILED: {e}")
                hc = EXTRACTION_FAILED

            try:
                logger.debug(f"{ticker}: Fetching earnings call...")
                ecall = earnings_call(ticker)
                logger.info(f"{ticker}: ✓ Earnings call completed")
            except Exception as e:
                logger.error(f"{ticker}: ✗ Earnings call FAILED: {e}")
                ecall = EXTRACTION_FAILED

            # --- consolidate via LLM into short report + recommendation ---- #
            logger.debug(f"{ticker}: Consolidating results...")
            try:
                report, reco = self._consolidate_result(
                    ticker=ticker,
                    risk=rf,
                    mda=mda,
                    legal=leg,
                    controls=ctrl,
                    business=biz,
                    tone=tone,
                    strategy=strat,
                    human_capital=hc,
                    earnings_call=ecall,
                )
                logger.info(f"{ticker}: ✓ COMPLETED - Recommendation: {reco}")
                successful += 1
            except Exception as e:
                logger.error(f"{ticker}: ✗ Consolidation FAILED: {e}")
                report = "Analysis failed"
                reco = "unknown"
                failed += 1

            # --- flatten to dataframe row ---------------------------------- #
            rows.append({
                "ticker": ticker,
                "risk_factors": rf.model_dump() if hasattr(rf, "model_dump") else rf,
                "md&a": mda.model_dump() if hasattr(mda, "model_dump") else mda,
                "legal": leg.model_dump() if hasattr(leg, "model_dump") else leg,
                "controls": ctrl.model_dump() if hasattr(ctrl, "model_dump") else ctrl,
                "business": biz.model_dump() if hasattr(biz, "model_dump") else biz,
                "tone_shift": tone.model_dump() if hasattr(tone, "model_dump") else tone,
                "strategy_summary": strat.model_dump() if hasattr(strat, "model_dump") else strat,
                "human_cap_esg": hc.model_dump() if hasattr(hc, "model_dump") else hc,
                "earnings_call": ecall.model_dump() if hasattr(ecall, "model_dump") else ecall,
                "final_report": report,
                "recommendation": reco,
            })

        # --- Final summary --- #
        logger.info("\n" + "=" * 60)
        logger.info("SEC ANALYSIS COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Total tickers processed: {total_tickers}")
        logger.info(f"  ✓ Successful: {successful}")
        logger.info(f"  ✗ Failed: {failed}")
        logger.info(f"  ⊘ Skipped (no filings): {skipped_no_filings}")
        logger.info(f"  ⚠ Single filing mode: {single_filing_mode}")
        logger.info(f"Results DataFrame shape: {len(rows)} rows")
        logger.info("=" * 60)

        return pd.DataFrame(rows)

    # --------------------------------------------------------------------- #
    #  Helper: combine section outputs into single narrative + recommendation
    # --------------------------------------------------------------------- #
    @staticmethod
    def _consolidate_result(**kwargs) -> tuple[str, str]:  # Fixed return type
        ticker = kwargs["ticker"]
        logger.debug(f"{ticker}: Building consolidation prompt...")
        
        analysis_json = json.dumps({k: v.model_dump() if isinstance(v, BaseModel) else v
                                    for k, v in kwargs.items()
                                    if k not in {"ticker"}})
        
        # Log sizes for debugging
        logger.debug(f"{ticker}: Analysis JSON size: {len(analysis_json)} chars")

        prompt = f"""
        Summarise the following parsed SEC-filing
        analysis and earnings call (JSON) into a concise justification (≤ 300 words) and give an overall
        BUY recommendation: positive / neutral / negative.  Focus on key drivers,
        risks, guidance and tone. Be objective, rigorous and critical. You should explain your decision in the justification section, weighing pros and cons.

        JSON INPUT:
        {analysis_json}

        Return exactly:
        {{
          "report": "<short paragraph>",
          "recommendation": "positive|neutral|negative"
        }}
        """

        # Fixed the API call to use proper response format
        class ConsolidatedResponse(BaseModel):
            report: str
            recommendation: str

        logger.debug(f"{ticker}: Calling o3-pro model for consolidation...")
        resp = client.responses.parse(
            model="o3-pro",
            input=[{"role": "system", "content": "You are a rigorous investment analyst. Your job is to analyze the data you are provided and produce a buy signal. Weigh both positive catalysts (strong guidance, revenue growth, segment expansion, favorable tone shifts) and negative signals (new risks, legal exposure, weaknesses, hedging language) fairly. A 'positive' recommendation is appropriate when the positives materially outweigh the negatives — do not default to neutral."},
                   {"role": "user", "content": prompt}],
            text_format=ConsolidatedResponse,
        ).output_parsed
        
        logger.debug(f"{ticker}: Consolidation response received")

        return resp.report, resp.recommendation

if __name__=="__main__":
    sec_ai = SecAnalysis()
    NVDAfilings_list = fetch_10K_and_10Q_filings("NVDA", "2023-01-01", "2025-6-6",form=["10-K"])
    MSFTfilings_list = fetch_10K_and_10Q_filings("MSFT", "2023-01-01", "2025-6-6",form=["10-K"])
    df, _ = sec_ai({'NVDA':NVDAfilings_list, 'MSFT':MSFTfilings_list})