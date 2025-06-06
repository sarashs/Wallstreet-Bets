from typing import List, Dict
from pydantic import BaseModel
from enum import Enum
import pandas as pd
from edgar_ai import *
from edgar_extractor import extract_items_from_filing
import json
from openai import OpenAI

# Initialize client
client = OpenAI()

# Assumes your analysis functions are already defined in this file or imported elsewhere

# --------------------------------------------------------------------------- #
#  SEC-ANALYSIS WRAPPER CLASS                                                 #
# --------------------------------------------------------------------------- #
class SecAnalysis:
    """
    High-level orchestrator.  Call an instance with a mapping:
        {ticker: [filing_obj_curr, filing_obj_prev], …}
    Each filing_obj must expose the minimal attributes accessed below.
    """

    def __call__(self, filings: Dict[str, List[object]]) -> pd.DataFrame:
        rows = []
        for ticker, flist in filings.items():
            if len(flist) < 2:  # need prev + current
                raise ValueError(f"{ticker}: need at least two filings (previous & current)")

            # Pull the key items once
            items_needed = ['1', '1A', '3', '7', '9A']     # add others if your extractor supports them
            prev = extract_items_from_filing(flist[1], items_needed)   # older filing
            curr = extract_items_from_filing(flist[0], items_needed)   # latest filing
            
            # --- section extractions -------------------------------------- #
            rf    = risk_factor_analysis(prev['1A'],  curr['1A'])
            mda   = mdad_analysis(curr['7'])                      # only current MD&A needed here
            leg   = legal_matters(curr['3'])
            ctrl  = control_status(curr['9A'])
            biz   = business_info(prev['1'],    curr['1'])
            tone  = tone_shift_analysis(prev['7'], curr['7'])
            strat = strategy_summary_analysis(curr['7'])
            hc    = human_capital_analysis(curr['1'])             # Item 1 contains "Human Capital Resources"

            # --- consolidate via LLM into short report + recommendation ---- #
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
            )

            # --- flatten to dataframe row ---------------------------------- #
            rows.append({
                "ticker": ticker,
                "risk_factors": rf.model_dump(),
                "md&a": mda.model_dump(),
                "legal": leg.model_dump(),
                "controls": ctrl.model_dump(),
                "business": biz.model_dump(),
                "tone_shift": tone.model_dump(),
                "strategy_summary": strat.model_dump(),
                "human_cap_esg": hc.model_dump(),
                "final_report": report,
                "recommendation": reco,
            })

        return pd.DataFrame(rows)

    # --------------------------------------------------------------------- #
    #  Helper: combine section outputs into single narrative + recommendation
    # --------------------------------------------------------------------- #
    @staticmethod
    def _consolidate_result(**kwargs) -> tuple[str, str]:  # Fixed return type
        ticker = kwargs["ticker"]
        analysis_json = json.dumps({k: v.model_dump() if isinstance(v, BaseModel) else v
                                    for k, v in kwargs.items()
                                    if k not in {"ticker"}})

        prompt = f"""
        You are a senior equity analyst.  Summarise the following parsed SEC-filing
        analysis (JSON) into a concise report (≤ 200 words) and give an overall
        BUY recommendation: positive / neutral / negative.  Focus on key drivers,
        risks, guidance and tone.

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

        resp = client.responses.parse(
            model="gpt-4o",
            input=[{"role": "system", "content": "You are a rigorous sell-side analyst."},
                   {"role": "user", "content": prompt}],
            text_format=ConsolidatedResponse,
        ).output_parsed

        return resp.report, resp.recommendation

if __name__=="__main__":
    sec_ai = SecAnalysis()
    NVDAfilings_list = fetch_10K_and_10Q_filings("NVDA", "2023-01-01", "2025-6-6",form=["10-K"])
    MSFTfilings_list = fetch_10K_and_10Q_filings("MSFT", "2023-01-01", "2025-6-6",form=["10-K"])
    df, _ = sec_ai({'NVDA':NVDAfilings_list, 'MSFT':MSFTfilings_list})