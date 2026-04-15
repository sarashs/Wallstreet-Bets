from typing import List, Optional
from pydantic import BaseModel, Field
from openai import OpenAI
from enum import Enum
import warnings
try:
    from edgar_extractor import chunk_text
except:
    from wallstreet_quant.edgar_extractor import chunk_text

# Initialize OpenAI client
client = OpenAI()

class BuyRecommendation(str, Enum):
    positive = "positive"
    neutral = "neutral"
    negative = "negative"

# ---------- 1. Risk Factors  (Item 1A, 10-K; Part II-Item 1A, 10-Q) ----------
class RiskChange(BaseModel):
    description: str

class RiskFactorsChanges(BaseModel):
    changes_detected: bool
    added_risks: List[RiskChange] = []
    removed_risks: List[RiskChange] = []

def risk_factor_analysis(previous_1A_text: str, current_1A_text: str, model: str ="gpt-5.2-pro"):
    prompt_risk = f"""
    You are analyzing the 'Risk Factors' section (Item 1A) for the current and previous 10-K document and making a buy recommendation.  
    Task: compare it with the prior filing and return a response conforming to RiskFactorsChanges:
    - changes_detected: true/false  
    - added_risks / removed_risks: list each change (1-sentence description).
    
    Previous Item 1A:
    {previous_1A_text}
    ------------------
    Current Item 1A:
    {current_1A_text}
    """
    risk_out = client.responses.parse(
        model=model,
        input=[{"role":"system","content":"You are a rigorous SEC-filing analyst. You job is detect changes in risk factors. You should be aware that sometimes risk factors are hidden or implicit (sometimes on purpose)."},
                  {"role":"user","content":prompt_risk}],
        text_format=RiskFactorsChanges
    )
    risk_out = risk_out.output_parsed
    return risk_out

# ---------- 2. MD&A Drivers & Forward-Looking  (Item 7, 10-K; Item 2, 10-Q) ----------
class MDADetails(BaseModel):
    performance_drivers: List[str]
    forward_looking_statements: List[str]
    honest_opinion: str
    
def mdad_analysis(section_text: str, model: str = "gpt-5.2-pro"):
    prompt_mda = f"""
    From the MD&A (Item 7 / Item 2) extract:
    - performance_drivers: key factors mgmt says drove recent results (list).  
    - forward_looking_statements: explicit outlook or plans.
    - honest_opinion: management's honest opinion about the business
    Return as MDADetails. This opinion might be implicit or hidden in the text.
    
    SECTION:
    {section_text}
    """
    
    mda_out = client.responses.parse(
        model=model,
        input=[{"role":"system","content":"You are a rigorous SEC-filing analyst."},
                  {"role":"user","content":prompt_mda}],
        text_format=MDADetails
    )
    mda_out = mda_out.output_parsed
    return mda_out

# ---------- 3. Legal Proceedings  (Item 3, 10-K; Part II-Item 1, 10-Q) ----------
class Proceeding(BaseModel):
    case: str
    description: str
    potential_impact: str
    
class LegalMatters(BaseModel):
    material_legal_proceedings: bool
    proceedings: List[Proceeding] = []

def legal_matters(section_text: str, model: str = "gpt-5.2-pro"):
    prompt_legal = f"""
    Summarize all ongoing or pending matters in 'Legal Proceedings' (Item 3).
    - material_legal_proceedings: True/False based on whether they exist
    - proceedings: list of legal proceedings (1 or 2 sentences per proceeding)
    Return as LegalMatters.
    
    SECTION:
    {section_text}
    """
    
    legal_out = client.responses.parse(
        model=model,
        input=[{"role":"system","content":"You are a rigorous SEC-filing analyst."},
                  {"role":"user","content":prompt_legal}],
        text_format=LegalMatters
    )
    legal_out = legal_out.output_parsed
    return legal_out

# ---------- 4. Controls & Procedures  (Item 9A, 10-K; Item 4, 10-Q) ----------
class Weakness(BaseModel):
    area: str
    description: str

class ControlsStatus(BaseModel):
    material_weaknesses_disclosed: bool
    weaknesses: List[Weakness] = []

def control_status(section_text: str, model: str = "gpt-5.2-pro"):
    prompt_ctrl = f"""
    In 'Controls and Procedures' (Item 9A / Item 4) identify any material weaknesses.
    It might be the case that they try to bury this information in dense text and hide it.
    Return as ControlsStatus.
    
    SECTION:
    {section_text}
    """
    
    ctrl_out = client.responses.parse(
        model=model,
        input=[{"role":"system","content":"You are a rigorous SEC-filing analyst."},
                  {"role":"user","content":prompt_ctrl}],
        text_format=ControlsStatus
    )
    
    ctrl_out = ctrl_out.output_parsed
    return ctrl_out

# ---------- 5. Business Overview & Segments  (Item 1, 10-K) ----------
class Segment(BaseModel):
    name: str
    description: str

class BusinessInfo(BaseModel):
    segments: List[Segment]
    segment_changes: List[str]

def business_info(previous_1_text: str, current_1_text: str, model: str = "gpt-5.2-pro"):
    prompt_business = f"""
    From 'Business' (Item 1) list current reportable segments and note any changes.
    Return as BusinessInfo.
    
    Previous Item 1:
    {previous_1_text}
    ------------------
    Current Item 1:
    {current_1_text}
    """
    
    business_out = client.responses.parse(
        model=model,
        input=[{"role":"system","content":"You are a rigorous SEC-filing analyst."},
                  {"role":"user","content":prompt_business}],
        text_format=BusinessInfo
    )
    business_out = business_out.output_parsed
    return business_out

# ---------- 6. Language Shift (MD&A tone) ---------- item 7
class ShiftExample(BaseModel):
    previous: str
    current: str

class ToneShift(BaseModel):
    language_shift_detected: bool
    examples: List[ShiftExample] = []

def tone_shift_analysis(previous_mda: str, current_mda: str, model: str = "gpt-5.2-pro"):

    prompt_shift = f"""
    You are analyzing the language tone of the MD&A section. Your task is to identify tone or hedging shifts — such as moving from confident to cautious language, from definitive to speculative, or from optimistic to neutral.
    
    ⚠️ DO NOT include examples where the change is:
    - only in numbers (e.g., interest rates, amounts, dates)
    - minor wording updates with no tonal effect
    - additions of factual or compliance disclosures
    
    ✅ Include only examples that show **real tone shifts**, such as:
    - confident → cautious (e.g., "will" → "may")
    - optimistic → neutral (e.g., "strong results" → "stable performance")
    - reduced certainty or hedging
    
    Return the result in this format:
    ToneShift:
    - language_shift_detected: true/false
    - examples: up to 10 with (previous, current)
    
    Previous MD&A:
    {previous_mda}
    ------------------
    Current MD&A:
    {current_mda}
    """

    shift_out = client.responses.parse(
        model=model,
        input=[{"role": "system", "content": "You are a rigorous SEC-filing analyst."},
               {"role": "user", "content": prompt_shift}],
        text_format=ToneShift
    )
    return shift_out.output_parsed

# ---------- 7. Segment Performance & Strategy  (Item 7, 10-K; Item 2, 10-Q) ----------
class SegmentStrategy(BaseModel):
    name: str
    performance: str
    strategic_changes: Optional[str]

class GuidanceChange(BaseModel):
    previous_guidance: str
    revised_guidance: str

class StrategySummary(BaseModel):
    segments: List[SegmentStrategy]
    corporate_priorities: List[str]
    guidance_revisions: List[GuidanceChange]

def strategy_summary_analysis(section_text: str, model: str = "gpt-5.2-pro"):
    prompt_strategy = f"""
    From the MD&A section (Item 7, 10-K or Item 2, 10-Q), extract:
    - Each business segment's recent performance and any strategic changes
    - Corporate-level strategic priorities (forward-looking plans, themes)
    - Any updates to forward-looking guidance (e.g., revenue, margins)

    Return the result as StrategySummary.

    SECTION:
    {section_text}
    """
    strat_out = client.responses.parse(
        model=model,
        input=[{"role": "system", "content": "You are a rigorous SEC-filing analyst."},
               {"role": "user", "content": prompt_strategy}],
        text_format=StrategySummary
    )
    return strat_out.output_parsed

# ---------- 8. Human Capital & ESG  (Item 1 – “Human Capital Resources”) ----------
class HCDetail(BaseModel):
    category: str
    description: str

class HCReport(BaseModel):
    human_capital_changes: bool
    details: List[HCDetail] = []

def human_capital_analysis(section_text: str, model: str = "gpt-5.2-pro"):
    prompt_hc = f"""
    From the 'Human Capital Resources' / ESG discussion (Item 1), note any significant
    updates (headcount, retention, labor disputes, ESG initiatives).  
    Return as HCReport.

    SECTION:
    {section_text}
    """
    hc_out = client.responses.parse(
        model=model,
        input=[{"role": "system", "content": "You are a rigorous SEC-filing analyst."},
               {"role": "user", "content": prompt_hc}],
        text_format=HCReport
    )
    return hc_out.output_parsed

class EarningsCall(BaseModel):
    managements_opinion : str
    risks: List[str] = []
    growth: List[str] = []
    buy: bool

def earnings_call(ticker: str, model: str = "gpt-5.2-pro"):
    query =  f"""
        review the last ticker {ticker} earnings call?
    """
    instructions = """You are a stock analyst helping with finding and analyzinf earnings calls.
     decide on the following give a few bullet points for each of the following:    
    - managements_opinion: managements honest opinion of the company without the usual fluf and bullshit  
    - risks: main risk factors  
    - growth: main growth factors  
    - buy: buy signal based on your gatherings either True or False (if data could not be found then false)
    your output will help determine whether it is a good investment. Be very conservative in your buy signal.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
        resp = client.responses.parse(
            model="gpt-5.2-pro",
            tools=[{"type": "web_search_preview",
                    "search_context_size": "low"}],
            input=query,
            instructions=instructions,
            text_format=EarningsCall
        )
    resp = resp.output_parsed
    return resp

# ---------- 9. Competitive Landscape (Item 1A, 10-K; Item 1, 10-Q) ----------
class Competition(BaseModel):
    competitors_detected: bool
    competitors: List[str]

def competitors_analysis(company_name: str, section_item: str, model: str = "gpt-5.2-pro") -> set[str]:

    """ Analyze the given section text for mentions of competitors.
    """
    if not section_item:
        raise ValueError("section_item cannot be empty or None")

    test_competitors = set()
    client = OpenAI()
    chunks = chunk_text(section_item)

    for section_text in chunks:
        prompt = f"""
        Read the following section and return ONLY company names that are mentioned as competitors.
        
        STRICT RULES:
        1. ONLY return actual company names (e.g., "Microsoft Corporation", "Apple Inc.", "Amazon")
        2. DO NOT return product names (e.g., "Android", "Xbox", "iPhone", "Windows")
        3. DO NOT return service names (e.g., "Google Search", "Azure", "AWS")
        4. DO NOT return platform names (e.g., "iOS", "macOS", "Linux")
        5. DO NOT return generic terms (e.g., "competitors", "rivals", "market players")
        6. DO NOT return the company itself ({company_name}) as a competitor
        7. Only include if mentioned in a competitive context, not as clients or partners
        
        Examples of VALID competitors: "Apple Inc.", "Microsoft Corporation", "Meta Platforms"
        Examples of INVALID (products/services): "Android", "Xbox", "iPhone", "Google Search", "Office 365"
        
        Company Name: {company_name}
        SECTION chunk:
        {section_text}
        """

        mda_response = client.responses.parse(
            model=model,
            input=[
                {"role": "system", "content": "You are a rigorous SEC-filing analyst specializing in competitor identification. "
                "Your ONLY task is to extract actual COMPANY NAMES that are competitors, NOT products, services, or platforms. "
                "Be extremely strict: only return full company names like 'Apple Inc.', 'Microsoft Corporation', 'Meta Platforms', etc. "
                "NEVER return product names like 'Android', 'Xbox', 'iPhone', 'Windows', or service names like 'Google Search', 'AWS'. "
                "The goal is to identify competing CORPORATIONS, not their products or services. "
                "If you're unsure whether something is a company name or product name, exclude it. "
                "Only include companies explicitly mentioned as competitors in the text."},
                {"role": "user", "content": prompt}
            ],
            text_format=Competition
        )
        mda_out = mda_response.output_parsed

        for competitor in mda_out.competitors:
            if competitor.lower() in section_text.lower():
                test_competitors.add(competitor)
    return test_competitors

# ---------- 10. Short Squeeze Catalyst Detection (Item 7 MD&A) ----------
class CatalystItem(BaseModel):
    catalyst: str
    category: str  # turnaround, product_launch, regulatory_win, activist, restructuring, other
    strength: str  # strong, moderate, weak

class SqueezeCatalystAnalysis(BaseModel):
    catalysts_detected: bool
    catalysts: List[CatalystItem] = []
    overall_catalyst_strength: str  # strong, moderate, weak, none
    summary: str

def squeeze_catalyst_detection(section_text: str, model: str = "gpt-5.2-pro") -> SqueezeCatalystAnalysis:
    """Analyze SEC filing section for positive catalysts that could trigger short covering."""
    prompt = f"""
    You are analyzing an SEC filing section (MD&A or Risk Factors) specifically to identify
    POSITIVE CATALYSTS that could trigger short sellers to cover their positions, causing a
    short squeeze.

    Look for:
    1. TURNAROUND signals: improving margins, cost restructuring bearing fruit, return to profitability
    2. NEW PRODUCT/SERVICE launches: upcoming catalysts that could drive revenue surprise
    3. REGULATORY WINS: FDA approvals, patent grants, favorable rulings
    4. ACTIVIST INVESTOR involvement: known activists taking positions, board changes, strategic reviews
    5. RESTRUCTURING: divestitures, spin-offs, or strategic pivots that could unlock value
    6. GUIDANCE IMPROVEMENTS: raised forecasts, accelerating growth, beat-and-raise patterns

    For each catalyst found, assess its strength as a potential short squeeze trigger:
    - "strong": concrete, near-term, and material (e.g., FDA approval expected Q2)
    - "moderate": directional but uncertain timing (e.g., "exploring strategic alternatives")
    - "weak": vague or long-term (e.g., "investing in innovation")

    If the filing text is generic corporate boilerplate with no real catalysts, set
    catalysts_detected=false and overall_catalyst_strength="none".

    SECTION TEXT:
    {section_text}
    """
    output = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": "You are a rigorous SEC-filing analyst specializing in short squeeze dynamics. "
             "Your job is to identify positive catalysts that could force short sellers to cover. "
             "Be conservative -- only flag real, material catalysts, not generic corporate language."},
            {"role": "user", "content": prompt}
        ],
        text_format=SqueezeCatalystAnalysis
    ).output_parsed
    return output


# ---------- 11. Ownership Dynamics Analysis ----------
class OwnershipSignal(BaseModel):
    signal_type: str  # buyback, insider_buying, institutional_accumulation, concentrated_ownership, float_reduction
    description: str
    bullish: bool

class OwnershipDynamicsAnalysis(BaseModel):
    signals_detected: bool
    signals: List[OwnershipSignal] = []
    float_pressure_assessment: str  # high, moderate, low, none
    summary: str

def ownership_dynamics_analysis(section_text: str, model: str = "gpt-5.2-pro") -> OwnershipDynamicsAnalysis:
    """Analyze SEC filing for ownership dynamics that could contribute to a short squeeze."""
    prompt = f"""
    You are analyzing an SEC filing to identify OWNERSHIP DYNAMICS that could contribute
    to a short squeeze scenario. These signals reduce available float or indicate
    informed buying pressure.

    Look for mentions of:
    1. STOCK BUYBACKS: active repurchase programs, accelerated buybacks, new authorizations
    2. INSIDER BUYING: officers or directors purchasing shares on the open market
    3. INSTITUTIONAL ACCUMULATION: large holders increasing positions, 13D/13G filings mentioned
    4. CONCENTRATED OWNERSHIP: high insider/institutional ownership reducing tradeable float
    5. FLOAT REDUCTION: any mechanism reducing shares available for trading (conversions,
       lockups expiring but holders not selling)

    For each signal, note whether it is BULLISH for a squeeze scenario (reduces available
    shares for shorts to borrow) or not.

    Assess overall float_pressure_assessment:
    - "high": multiple signals of float reduction + active buying
    - "moderate": some buyback activity or moderate insider buying
    - "low": mentioned but immaterial
    - "none": no relevant signals found

    SECTION TEXT:
    {section_text}
    """
    output = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": "You are a rigorous SEC-filing analyst specializing in ownership structure "
             "and float analysis. Identify signals that reduce tradeable float or indicate informed buying pressure."},
            {"role": "user", "content": prompt}
        ],
        text_format=OwnershipDynamicsAnalysis
    ).output_parsed
    return output


# ---------- 12. Short Seller Vulnerability Assessment (Item 1A Risk Factors) ----------
class VulnerabilityFactor(BaseModel):
    factor: str
    description: str
    severity: str  # high, moderate, low

class ShortVulnerabilityAnalysis(BaseModel):
    vulnerability_detected: bool
    factors: List[VulnerabilityFactor] = []
    overall_vulnerability: str  # high, moderate, low, none
    summary: str

def short_vulnerability_assessment(section_text: str, model: str = "gpt-5.2-pro") -> ShortVulnerabilityAnalysis:
    """Analyze Risk Factors to assess how vulnerable short sellers are in this stock."""
    prompt = f"""
    You are analyzing an SEC filing's Risk Factors section to assess whether SHORT SELLERS
    are particularly VULNERABLE in this stock. This is the inverse of typical risk analysis --
    you are looking for factors that make the SHORT THESIS fragile.

    Look for:
    1. EXPLICIT SHORT SELLING MENTIONS: company discussing short selling of its stock,
       short squeezes, or stock price volatility driven by short interest
    2. THIN FLOAT warnings: low public float, high insider ownership, limited liquidity
    3. STOCK PRICE VOLATILITY language: acknowledgment of extreme price movements,
       meme stock dynamics, social media driven trading
    4. DEFENSIVE MEASURES: poison pills, shareholder rights plans, anti-dilution provisions
       that could hurt shorts
    5. BINARY EVENT RISK: upcoming catalysts (trials, rulings, earnings) where a positive
       outcome would cause rapid price appreciation
    6. HARD-TO-BORROW indicators: mentions of share lending, securities lending programs,
       or difficulty in borrowing shares

    Assess overall vulnerability:
    - "high": explicit short selling mentions + thin float + binary catalysts
    - "moderate": some vulnerability factors present
    - "low": standard risk factors, nothing specific to shorts
    - "none": no relevant factors

    SECTION TEXT:
    {section_text}
    """
    output = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": "You are a rigorous SEC-filing analyst. Your specific task is to assess "
             "how vulnerable short sellers are in this stock based on the risk factor disclosures. "
             "Focus on factors that could cause forced covering or rapid price appreciation."},
            {"role": "user", "content": prompt}
        ],
        text_format=ShortVulnerabilityAnalysis
    ).output_parsed
    return output


# Example usage:
#risk = risk_factor_analysis(prev['1A'], last['1A'], model="gpt-4o")
#mdad = mdad_analysis(last['7'], model="gpt-4o")
#legal = legal_matters(last['3'], "gpt-4o")
#ctrl = control_status(last['9A'])
#busi = business_info(prev['1'], last['1'])
#tone = tone_shift_analysis(prev['7'], last['7'])
#seg_perfomance = strategy_summary_analysis(last['7'])
#human_capital = human_capital_analysis(last['1'])