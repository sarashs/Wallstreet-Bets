# balance sheet extractor
### Methods:
# fetch_10K_and_10Q_filings: downloads 10ks and 10qs for a given range
# extract_financials: extrancts balance sheets, income statements and cashflow statements for a list of filings
# extract_items_from_filing: extracts the written part of filings for one file e.g., ["1A", "7", "3"]
# pip install edgartools
import sys
from typing import List
from edgar import *
from edgar.financials import Financials
import pandas as pd
import os
import re
import logging

from openai import OpenAI
from pydantic import BaseModel

logger = logging.getLogger("SecAnalysis")

edgar_identity = None
edgar_identity = os.getenv("edgar_identity") # must be a string like "user_name email@server.com" if you don't have an environment variable then set it mannually as I said: "user_name email@server.com"
assert edgar_identity is not None, 'edgar_identity environment variable must be a string like "user_name email@server.com"'
set_identity(edgar_identity)

from edgar import Company

def fetch_10K_and_10Q_filings(ticker: str, start_date: str = None, end_date: str = None, 
                               form: list = None, latest_only: bool = False,
                               latest_n: int = None, include_foreign: bool = True):
    """
    Fetches the 10-K and 10-Q filings for the given ticker within the specified date range.
    Also supports foreign filer forms (20-F annual reports, 6-K current reports).

    Note:
      - Make sure you have set your EDGAR identity (using set_identity) before calling this function.
      - The date filter should be in the form "YYYY-MM-DD:YYYY-MM-DD".

    Parameters:
        ticker (str): The stock ticker (e.g., "AAPL").
        start_date (str): The start date in "YYYY-MM-DD" format. Optional if latest_only/latest_n used.
        end_date (str): The end date in "YYYY-MM-DD" format. Optional if latest_only/latest_n used.
        form (list): List of form types to fetch (e.g., ['10-K'], ['10-Q'], or ['10-K', '10-Q']).
                     If None and latest_only/latest_n=True, fetches both 10-K and 10-Q.
                     If None and latest_only=False, defaults to ['10-K'].
        latest_only (bool): If True, returns only the single most recent filing. Equivalent to latest_n=1.
        latest_n (int): If set, returns the N most recent filings (sorted by date, newest first).
                        Useful for comparison analysis (e.g., latest_n=2 for current vs previous).
        include_foreign (bool): If True, also looks for 20-F (annual) and 6-K (quarterly) forms
                                for foreign companies when 10-K/10-Q are not found.

    Returns:
        list: A list of filing objects (or an empty list if no filings are found).
              If latest_only=True or latest_n is set, returns a list with the specified number of filings.
    """
    try:
        # Create a Company object for the given ticker
        company = Company(ticker)
        
        # Handle latest_only as latest_n=1
        if latest_only and latest_n is None:
            latest_n = 1
        
        if latest_n is not None:
            # Fetch all relevant filings and return the N most recent
            all_filings = []
            
            # Fetch domestic filings (10-K and 10-Q)
            filings_10k = company.get_filings(form=['10-K'])
            filings_10q = company.get_filings(form=['10-Q'])
            
            # Collect available filings using index access
            if filings_10k and len(filings_10k) > 0:
                count_10k = min(len(filings_10k), latest_n * 2)
                for i in range(count_10k):
                    all_filings.append(filings_10k[i])
            if filings_10q and len(filings_10q) > 0:
                count_10q = min(len(filings_10q), latest_n * 2)
                for i in range(count_10q):
                    all_filings.append(filings_10q[i])
            
            # If no domestic filings and include_foreign is True, try foreign forms
            if not all_filings and include_foreign:
                filings_20f = company.get_filings(form=['20-F'])
                filings_6k = company.get_filings(form=['6-K'])
                
                if filings_20f and len(filings_20f) > 0:
                    count_20f = min(len(filings_20f), latest_n * 2)
                    for i in range(count_20f):
                        all_filings.append(filings_20f[i])
                if filings_6k and len(filings_6k) > 0:
                    count_6k = min(len(filings_6k), latest_n * 2)
                    for i in range(count_6k):
                        all_filings.append(filings_6k[i])
            
            if not all_filings:
                print(f"No 10-K, 10-Q, 20-F, or 6-K filings found for {ticker}.")
                return []
            
            # Sort by filing date (newest first) and return top N
            all_filings.sort(key=lambda f: f.filing_date, reverse=True)
            return all_filings[:latest_n]
        
        # Standard behavior: fetch specific form types with date filtering
        if form is None:
            form = ['10-K']
            
        # Retrieve filings for the specified form(s)
        filings = company.get_filings(form=form)
        
        # Filter by date range if provided
        if start_date and end_date:
            filtered_filings = filings.filter(date=f"{start_date}:{end_date}")
        else:
            filtered_filings = filings
        
        if not filtered_filings:
            print(f"No {form} filings found for {ticker}" + 
                  (f" between {start_date} and {end_date}." if start_date and end_date else "."))
            return []
            
        return filtered_filings

    except Exception as e:
        print(f"An error occurred while fetching filings for {ticker}: {e}")
        return []


def extract_financials(filings):
    """
    Extracts financial statements from a list of filings.
    
    For each filing, the function:
      - Calls filing.obj() to get the data object (e.g. TenK/TenQ).
      - Checks that the object has a 'financials' attribute.
      - Extracts the balance sheet, income statement, and cashflow statement using:
            financials.get_balance_sheet()
            financials.get_income_statement()
            financials.get_cash_flow_statement()
    
    Parameters:
        filings (list): A list-like object of filing objects (e.g. from Company.get_filings()).
    
    Returns:
        tuple: Six items containing the extracted financial statements:
               (balance_sheets, income_statements, cashflow_statements, 
                balance_sheets_str, income_statements_str, cashflow_statements_str).
               Filings that do not have a data object or the requested financial statement(s) are skipped.
    """
    balance_sheets = []
    income_statements = []
    cashflow_statements = []
    
    for filing in filings:
        try:
            # Convert the filing to its data object (e.g., TenK or TenQ)
            data_obj = filing.obj()
            if data_obj is None:
                print("Filing has no data object. Skipping...")
                continue
            
            # Check that the data object contains financials
            if not hasattr(data_obj, "financials") or data_obj.financials is None:
                print("Filing has no financials. Skipping...")
                continue

            financials = data_obj.financials
            
            # Extract the individual financial statements.
            # If any of these methods are unavailable or return None, skip that particular statement.
            balance_sheet = financials.get_balance_sheet() if hasattr(financials, "get_balance_sheet") else None
            income_statement = financials.get_income_statement() if hasattr(financials, "get_income_statement") else None
            cashflow_statement = financials.get_cash_flow_statement() if hasattr(financials, "get_cash_flow_statement") else None
            
            if balance_sheet is not None:
                balance_sheets.append(balance_sheet)
            if income_statement is not None:
                income_statements.append(income_statement)
            if cashflow_statement is not None:
                cashflow_statements.append(cashflow_statement)
        
        except Exception as e:
            print(f"Error extracting financials from filing: {e}")
            continue

    # Convert to structured strings for LLM processing with enhanced metadata
    def format_statement_for_llm(statement, statement_type, index):
        parts = []
        parts.append(f"=== {statement_type} {index + 1} ===")
        
        # Add metadata if available
        if hasattr(statement, 'period_end_date'):
            parts.append(f"Period End Date: {statement.period_end_date}")
        if hasattr(statement, 'filing_date'):
            parts.append(f"Filing Date: {statement.filing_date}")
        if hasattr(statement, 'period_focus'):
            parts.append(f"Period Focus: {statement.period_focus}")
        if hasattr(statement, 'fiscal_year'):
            parts.append(f"Fiscal Year: {statement.fiscal_year}")
        if hasattr(statement, 'fiscal_period'):
            parts.append(f"Fiscal Period: {statement.fiscal_period}")
        
        parts.append("")  # Empty line before data
        
        # Use to_dataframe() but with better formatting
        df = statement.to_dataframe()
        if not df.empty:
            # Round numeric values for cleaner display
            df_display = df.copy()
            for col in df_display.select_dtypes(include=['float64', 'int64']).columns:
                df_display[col] = df_display[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "")
            parts.append(df_display.to_string())
        else:
            parts.append("No data available")
        
        return '\n'.join(parts)
    
    balance_sheet_parts = [format_statement_for_llm(item, "BALANCE SHEET", i) 
                          for i, item in enumerate(balance_sheets)]
    balance_sheets_str = '\n\n'.join(balance_sheet_parts)
    
    income_statement_parts = [format_statement_for_llm(item, "INCOME STATEMENT", i) 
                             for i, item in enumerate(income_statements)]
    income_statements_str = '\n\n'.join(income_statement_parts)
    
    cashflow_statement_parts = [format_statement_for_llm(item, "CASH FLOW STATEMENT", i) 
                               for i, item in enumerate(cashflow_statements)]
    cashflow_statements_str = '\n\n'.join(cashflow_statement_parts)
    
    return balance_sheets, income_statements, cashflow_statements, balance_sheets_str, income_statements_str, cashflow_statements_str


# ============================================================================
# IMPROVED SECTION EXTRACTION - Multi-strategy approach with fallbacks
# ============================================================================

# Standard 10-K item definitions with expected keywords
ITEM_DEFINITIONS_10K = {
    '1': {
        'keywords': ['BUSINESS', 'OVERVIEW', 'DESCRIPTION OF BUSINESS', 'OUR BUSINESS'],
        'title_patterns': [r'BUSINESS', r'DESCRIPTION\s+OF\s+BUSINESS']
    },
    '1A': {
        'keywords': ['RISK FACTORS', 'RISKS'],
        'title_patterns': [r'RISK\s+FACTORS']
    },
    '1B': {
        'keywords': ['UNRESOLVED STAFF COMMENTS', 'STAFF COMMENTS'],
        'title_patterns': [r'UNRESOLVED\s+STAFF\s+COMMENTS']
    },
    '1C': {
        'keywords': ['CYBERSECURITY'],
        'title_patterns': [r'CYBERSECURITY']
    },
    '2': {
        'keywords': ['PROPERTIES', 'PROPERTY'],
        'title_patterns': [r'PROPERTIES']
    },
    '3': {
        'keywords': ['LEGAL PROCEEDINGS', 'LITIGATION'],
        'title_patterns': [r'LEGAL\s+PROCEEDINGS']
    },
    '4': {
        'keywords': ['MINE SAFETY'],
        'title_patterns': [r'MINE\s+SAFETY']
    },
    '5': {
        'keywords': ['MARKET FOR REGISTRANT', 'COMMON EQUITY', 'STOCKHOLDER MATTERS'],
        'title_patterns': [r'MARKET\s+FOR.*COMMON\s+EQUITY']
    },
    '6': {
        'keywords': ['RESERVED', 'SELECTED FINANCIAL'],
        'title_patterns': [r'\[?RESERVED\]?', r'SELECTED\s+FINANCIAL']
    },
    '7': {
        'keywords': ["MANAGEMENT'S DISCUSSION", 'MD&A', 'FINANCIAL CONDITION', 'RESULTS OF OPERATIONS'],
        'title_patterns': [r"MANAGEMENT['']?S\s+DISCUSSION", r'MD\s*&\s*A']
    },
    '7A': {
        'keywords': ['QUANTITATIVE AND QUALITATIVE', 'MARKET RISK'],
        'title_patterns': [r'QUANTITATIVE\s+AND\s+QUALITATIVE.*MARKET\s+RISK']
    },
    '8': {
        'keywords': ['FINANCIAL STATEMENTS', 'SUPPLEMENTARY DATA'],
        'title_patterns': [r'FINANCIAL\s+STATEMENTS']
    },
    '9': {
        'keywords': ['CHANGES IN AND DISAGREEMENTS', 'ACCOUNTANTS'],
        'title_patterns': [r'CHANGES\s+IN\s+AND\s+DISAGREEMENTS']
    },
    '9A': {
        'keywords': ['CONTROLS AND PROCEDURES', 'INTERNAL CONTROL'],
        'title_patterns': [r'CONTROLS\s+AND\s+PROCEDURES']
    },
    '9B': {
        'keywords': ['OTHER INFORMATION'],
        'title_patterns': [r'OTHER\s+INFORMATION']
    },
    '9C': {
        'keywords': ['FOREIGN JURISDICTIONS', 'PREVENT INSPECTIONS'],
        'title_patterns': [r'DISCLOSURE.*FOREIGN\s+JURISDICTIONS']
    },
    '10': {
        'keywords': ['DIRECTORS', 'EXECUTIVE OFFICERS', 'CORPORATE GOVERNANCE'],
        'title_patterns': [r'DIRECTORS.*EXECUTIVE\s+OFFICERS']
    },
    '11': {
        'keywords': ['EXECUTIVE COMPENSATION'],
        'title_patterns': [r'EXECUTIVE\s+COMPENSATION']
    },
    '12': {
        'keywords': ['SECURITY OWNERSHIP', 'BENEFICIAL OWNERSHIP'],
        'title_patterns': [r'SECURITY\s+OWNERSHIP']
    },
    '13': {
        'keywords': ['CERTAIN RELATIONSHIPS', 'RELATED TRANSACTIONS'],
        'title_patterns': [r'CERTAIN\s+RELATIONSHIPS']
    },
    '14': {
        'keywords': ['PRINCIPAL ACCOUNTANT', 'ACCOUNTING FEES'],
        'title_patterns': [r'PRINCIPAL\s+ACCOUNT']
    },
    '15': {
        'keywords': ['EXHIBITS', 'FINANCIAL STATEMENT SCHEDULES'],
        'title_patterns': [r'EXHIBITS']
    }
}

# 10-Q Item Definitions
# 10-Q Structure:
# Part I - Financial Information: Items 1-4
# Part II - Other Information: Items 1-6
ITEM_DEFINITIONS_10Q = {
    # Part I - Financial Information
    'Part I Item 1': {
        'keywords': ['FINANCIAL STATEMENTS', 'CONDENSED', 'UNAUDITED'],
        'title_patterns': [r'FINANCIAL\s+STATEMENTS']
    },
    'Part I Item 2': {
        'keywords': ["MANAGEMENT'S DISCUSSION", "TRUSTEE'S DISCUSSION", 'MD&A', 'FINANCIAL CONDITION', 
                     'RESULTS OF OPERATIONS', 'DISCUSSION AND ANALYSIS'],
        'title_patterns': [r"MANAGEMENT['']?S\s+DISCUSSION", r"TRUSTEE['']?S\s+DISCUSSION", r'MD\s*&\s*A',
                          r'DISCUSSION\s+AND\s+ANALYSIS']
    },
    'Part I Item 3': {
        'keywords': ['QUANTITATIVE AND QUALITATIVE', 'MARKET RISK'],
        'title_patterns': [r'QUANTITATIVE\s+AND\s+QUALITATIVE.*MARKET\s+RISK']
    },
    'Part I Item 4': {
        'keywords': ['CONTROLS AND PROCEDURES', 'DISCLOSURE CONTROLS', 'CONTROLS'],
        'title_patterns': [r'CONTROLS\s+AND\s+PROCEDURES', r'CONTROLS']
    },
    # Part II - Other Information
    'Part II Item 1': {
        'keywords': ['LEGAL PROCEEDINGS', 'LITIGATION'],
        'title_patterns': [r'LEGAL\s+PROCEEDINGS']
    },
    'Part II Item 1A': {
        'keywords': ['RISK FACTORS', 'RISKS'],
        'title_patterns': [r'RISK\s+FACTORS']
    },
    'Part II Item 2': {
        'keywords': ['UNREGISTERED SALES', 'EQUITY SECURITIES', 'USE OF PROCEEDS'],
        'title_patterns': [r'UNREGISTERED\s+SALES', r'USE\s+OF\s+PROCEEDS']
    },
    'Part II Item 3': {
        'keywords': ['DEFAULTS', 'SENIOR SECURITIES'],
        'title_patterns': [r'DEFAULTS']
    },
    'Part II Item 4': {
        'keywords': ['MINE SAFETY'],
        'title_patterns': [r'MINE\s+SAFETY']
    },
    'Part II Item 5': {
        'keywords': ['OTHER INFORMATION'],
        'title_patterns': [r'OTHER\s+INFORMATION']
    },
    'Part II Item 6': {
        'keywords': ['EXHIBITS'],
        'title_patterns': [r'EXHIBITS']
    }
}

# Mapping from 10-K item requests to 10-Q equivalents
# This allows users to request items like '1A', '7', '3' and get the right 10-Q section
ITEM_10K_TO_10Q_MAP = {
    '1A': 'Part II Item 1A',   # Risk Factors
    '7': 'Part I Item 2',       # MD&A
    '7A': 'Part I Item 3',      # Quantitative/Qualitative Market Risk
    '3': 'Part II Item 1',      # Legal Proceedings
    '9A': 'Part I Item 4',      # Controls and Procedures
    '8': 'Part I Item 1',       # Financial Statements
}

# Backward compatibility alias
ITEM_DEFINITIONS = ITEM_DEFINITIONS_10K


def _normalize_text(text):
    """Normalize text for consistent matching."""
    # Replace various unicode spaces with regular spaces
    text = re.sub(r'[\u00A0\u2000-\u200B\u2028\u2029\xa0]', ' ', text)
    # Normalize multiple spaces/tabs to single space
    text = re.sub(r'[ \t]+', ' ', text)
    return text


def _find_all_item_mentions(full_text, item_num):
    """
    Find all mentions of a specific item in the document.
    Returns list of dicts with position, context, and header score.
    """
    # Pattern to match "ITEM X" or "ITEM X." or "ITEM X:"
    # Handle items like "1A" carefully - don't match "1A" when looking for "1"
    # Allow box-drawing characters (│, ╭, ╮, etc.) and other decorations before ITEM
    # The key is to find ITEM at roughly the start of a line (after newline + optional chars)
    if item_num[-1].isalpha():
        # Item like "1A", "7A" - require exact match
        pattern = re.compile(
            rf'(?:^|\n)[^\n]*?ITEM\s+{re.escape(item_num)}\b[\.\:\s]',
            re.IGNORECASE | re.MULTILINE
        )
    else:
        # Item like "1", "7" - don't match if followed by letter
        pattern = re.compile(
            rf'(?:^|\n)[^\n]*?ITEM\s+{re.escape(item_num)}(?![A-Z])[\.\:\s]',
            re.IGNORECASE | re.MULTILINE
        )
    
    mentions = []
    for match in pattern.finditer(full_text):
        # Find the actual position of "ITEM" within the match
        match_text = match.group(0)
        item_offset = match_text.upper().find('ITEM')
        pos = match.start() + item_offset
        
        # Get context after the match (up to 600 chars)
        context_end = min(pos + 600, len(full_text))
        context = full_text[pos:context_end]
        
        # Get header score (higher = more likely to be a real section header)
        header_score = _is_likely_section_header(full_text, pos, item_num)
        
        mentions.append({
            'position': pos,
            'context': context,
            'is_header': header_score > 0,  # backward compatibility
            'header_score': header_score,
            'match': match.group(0)
        })
    
    return mentions


def _is_likely_section_header(full_text, pos, item_num):
    """
    Determine if a position is likely a section header vs TOC entry or reference.
    Uses scoring system to make the determination.
    """
    # Get context before and after
    context_before = full_text[max(0, pos-200):pos]
    context_after = full_text[pos:min(len(full_text), pos+2000)]
    
    # Check for TOC indicators before this position
    toc_indicators = ['TABLE OF CONTENTS', 'INDEX', 'PAGE']
    is_in_toc = any(ind in context_before.upper() for ind in toc_indicators)
    
    # Check if followed by dotted line (TOC entry signature)
    has_dotted_line = bool(re.search(r'\.{5,}', context_after[:200]))
    
    # Check if followed by page number pattern typical of TOC
    has_page_ref = bool(re.search(r'\.{3,}\s*\d+\s*$', context_after[:200], re.MULTILINE))
    
    # Check for expected keywords in context after
    item_def = ITEM_DEFINITIONS.get(item_num, {})
    keywords = item_def.get('keywords', [])
    has_keywords = any(kw.upper() in context_after[:500].upper() for kw in keywords)
    
    # Check content length after - actual sections have substantial content
    # Look for next ITEM header (allowing for box-drawing characters)
    next_item = re.search(r'\n[^\n]*?ITEM\s+\d+[A-Z]?[\.\:\s]', context_after[100:], re.IGNORECASE)
    if next_item:
        section_length = next_item.start()
    else:
        section_length = len(context_after)
    
    # Check if this looks like a cross-reference (embedded in a sentence)
    # Cross-references often have phrases like "see", "refer to", "described in" before them
    # Also check for quote marks which indicate inline references
    # Include both ASCII and Unicode quote characters
    is_crossref = bool(re.search(r'(see|refer\s+to|described\s+in|discussed\s+in|set\s+forth\s+in|under)\s*["\'""'']*\s*$', 
                                  context_before[-60:], re.IGNORECASE))
    
    # Check if preceded by quote mark (inline reference like: see "Item 1A. Risk Factors")
    # Include Unicode quotes: " " ' '
    has_quote_before = bool(re.search(r'["\'""'']\s*$', context_before[-15:]))
    
    # Check if followed by a quote pattern (indicating it's a reference)
    # Pattern: "Item X. Title" for a discussion of..." or "Item X. Title," our...
    # Include Unicode quotes
    has_quote_after = bool(re.search(r'^Item\s+\d+[A-Z]?\.\s+[^│\n]{5,80}["\'""'']\s*(for|of|our|and|the)', 
                                      context_after[:200], re.IGNORECASE))
    
    # Check for box-drawing characters (indicates formatted header)
    has_box_drawing = bool(re.search(r'[─│╭╮╯╰┌┐└┘├┤┬┴┼]', context_before[-20:] + context_after[:50]))
    
    # Check if this appears to be within a paragraph (text continues after on same topic)
    # Real headers have structural break - new paragraph about the topic, not continuation
    is_inline_mention = bool(re.search(r'^Item\s+\d+[A-Z]?\.\s+\w+.*?(for\s+a\s+discussion|for\s+additional|for\s+more)', 
                                        context_after[:200], re.IGNORECASE))
    
    # Scoring system
    score = 0
    if is_in_toc:
        score -= 3
    if has_dotted_line:
        score -= 2
    if has_page_ref:
        score -= 2
    if is_crossref:
        score -= 3  # Strong negative signal for cross-references
    if has_quote_before:
        score -= 3  # Quote before = inline reference
    if has_quote_after:
        score -= 2  # "Item X. Title" for a discussion" is a reference
    if is_inline_mention:
        score -= 3  # "for a discussion of" pattern = cross-reference
    if has_box_drawing:
        score += 10  # Box-drawing characters VERY STRONGLY indicate a header
    if has_keywords:
        score += 2
    if section_length > 5000:
        score += 3  # Very long section = definitely a real header
    elif section_length > 1000:
        score += 2
    elif section_length > 500:
        score += 1
    elif section_length < 200:
        score -= 1  # Very short = likely not a real section
    
    # Check if starts on its own line (strong header indicator)
    line_start = full_text.rfind('\n', max(0, pos-5), pos)
    if line_start != -1 and pos - line_start < 10:
        score += 1
    
    return score


def _find_section_by_keyword(full_text, item_num, start_search_from=0):
    """
    Fallback: Find section using keyword patterns when other methods fail.
    """
    item_def = ITEM_DEFINITIONS.get(item_num, {})
    title_patterns = item_def.get('title_patterns', [])
    
    for title_pat in title_patterns:
        # Build pattern: ITEM X followed by title pattern
        # Allow box-drawing characters and other decorations before ITEM
        if item_num[-1].isalpha():
            pattern = re.compile(
                rf'(?:^|\n)[^\n]*?ITEM\s+{re.escape(item_num)}\b[\.:\s]*{title_pat}',
                re.IGNORECASE | re.MULTILINE
            )
        else:
            pattern = re.compile(
                rf'(?:^|\n)[^\n]*?ITEM\s+{re.escape(item_num)}(?![A-Z])[\.:\s]*{title_pat}',
                re.IGNORECASE | re.MULTILINE
            )
        
        matches = list(pattern.finditer(full_text, start_search_from))
        
        for match in matches:
            # Find the actual position of "ITEM" within the match
            match_text = match.group(0)
            item_offset = match_text.upper().find('ITEM')
            actual_pos = match.start() + item_offset
            if _is_likely_section_header(full_text, actual_pos, item_num):
                return actual_pos
    
    return None


def _estimate_toc_end(full_text):
    """
    Estimate where the table of contents ends.
    The actual content usually starts after this point.
    """
    # Strategy 1: Look for section headers with box-drawing characters (very reliable)
    # These indicate actual formatted section headers, not TOC entries
    box_header = re.search(r'[─│╭╮╯╰┌┐└┘├┤┬┴┼][^\n]*?ITEM\s+1\b[^A]', full_text[:100000], re.IGNORECASE)
    if box_header:
        return box_header.start()
    
    # Strategy 2: Look for "PART I" followed by section title (FINANCIAL INFORMATION, etc.)
    # This is the main section header that appears after TOC in 10-Q filings
    part_i_with_title = re.search(
        r'PART\s+I\s*[—\-–:]*\s*(FINANCIAL\s+INFORMATION|FINANCIAL|ITEM)',
        full_text[:50000], 
        re.IGNORECASE
    )
    if part_i_with_title:
        return part_i_with_title.start()
    
    # Strategy 3: Look for "Part I" header that's followed by "Item 1" shortly after
    part_i_pattern = re.compile(r'PART\s+I\b(?!\s*I)', re.IGNORECASE)
    for match in part_i_pattern.finditer(full_text[:50000]):
        # Check if "Item 1" appears within the next 200 chars
        after_text = full_text[match.start():match.start()+200]
        if re.search(r'Item\s+1[\.\:\s]', after_text, re.IGNORECASE):
            return match.start()
    
    # Strategy 4: Find Item 1 that is followed by substantial text (not page number)
    # TOC entries: "Item 1. Business 4" (followed by page number)
    # Real headers: "Item 1. Business" followed by actual content paragraphs
    item1_patterns = [
        r'\n[^\n]*?ITEM\s+1\b[^A][^\n]*BUSINESS[^\n]*\n',
        r'\n[^\n]*?ITEM\s+1[\.\:\s]+FINANCIAL\s+STATEMENTS[^\n]*\n',
        r'\n[^\n]*?ITEM\s+1[\.\:\s][^\n]*\n[^\n]+\n[^\n]+\n',  # Item 1 followed by content
    ]
    for pattern in item1_patterns:
        item1_re = re.compile(pattern, re.IGNORECASE)
        for match in item1_re.finditer(full_text[:100000]):
            # Check what follows - if it's substantial text (not another Item), this is likely the real header
            after_match = full_text[match.end():match.end()+500]
            # If the next 100 chars don't have another "Item X" pattern, this is likely the real section
            if not re.search(r'^\s*Item\s+\d', after_match[:100], re.IGNORECASE):
                return match.start()
    
    # Fallback: assume TOC is in first 5k chars (be conservative to avoid missing sections)
    return min(5000, len(full_text) // 10)


def extract_items_from_filing(filing_obj, items_to_extract, verbose=False):
    """
    Extracts textual sections (e.g., Item 1A, Item 7) from a filing object.
    Automatically detects filing type (10-K vs 10-Q) and uses appropriate item definitions.
    Uses multiple strategies with fallbacks to maximize extraction success.
    
    Strategies used (in order):
    1. Find all ITEM mentions and identify which are actual section headers
    2. Use keyword patterns to locate sections
    3. Skip TOC entries by estimating TOC boundary
    
    Parameters:
        filing_obj: A filing object returned by the 'edgar' package.
        items_to_extract: List of item names (e.g., ["1", "1A", "7", "3"]).
                         For 10-Q filings, you can still use 10-K item numbers
                         and they will be mapped to equivalent 10-Q sections.
        verbose: If True, print warnings about missing items. Default False for cleaner batch output.
        
    Returns:
        dict: Dictionary of item number → extracted text
    """
    try:
        # Get the full filing text
        full_text = filing_obj.text()
        full_text = _normalize_text(full_text)
        
        # Detect filing type from the filing object or text
        is_10q = _detect_filing_type(filing_obj, full_text) == '10-Q'
        
        # Estimate where TOC ends - we want to find sections AFTER this
        toc_end = _estimate_toc_end(full_text)
        
        if is_10q:
            # Use 10-Q extraction logic
            return _extract_10q_items(full_text, items_to_extract, toc_end, verbose=verbose)
        else:
            # Use 10-K extraction logic (original behavior)
            return _extract_10k_items(full_text, items_to_extract, toc_end, verbose=verbose)
        
    except Exception as e:
        print(f"Failed to extract items: {e}")
        import traceback
        traceback.print_exc()
        return {}


def _detect_filing_type(filing_obj, full_text):
    """
    Detect whether this is a 10-K or 10-Q filing.
    
    Parameters:
        filing_obj: Filing object from edgar package
        full_text: Normalized full text of the filing
        
    Returns:
        str: '10-K' or '10-Q'
    """
    # Try to get form type from filing object
    try:
        form = getattr(filing_obj, 'form', None)
        if form:
            if '10-Q' in str(form).upper():
                return '10-Q'
            elif '10-K' in str(form).upper():
                return '10-K'
    except:
        pass
    
    # Fallback: detect from text content
    # 10-Q has "QUARTERLY REPORT" and "Part I" / "Part II" structure
    first_10k = full_text[:20000].upper()
    
    if 'QUARTERLY REPORT' in first_10k or 'FORM 10-Q' in first_10k:
        return '10-Q'
    elif 'ANNUAL REPORT' in first_10k or 'FORM 10-K' in first_10k:
        return '10-K'
    
    # Check for Part I/Part II structure typical of 10-Q
    if re.search(r'PART\s+I\s*[-–—]?\s*FINANCIAL\s+INFORMATION', first_10k):
        return '10-Q'
    
    # Default to 10-K
    return '10-K'


def _extract_10k_items(full_text, items_to_extract, toc_end, verbose=False):
    """
    Extract items from a 10-K filing.
    
    Parameters:
        full_text: Normalized full text of the filing
        items_to_extract: List of item names (e.g., ["1", "1A", "7", "3"])
        toc_end: Estimated end position of table of contents
        verbose: If True, print warnings about missing items
        
    Returns:
        dict: Dictionary of item number → extracted text
    """
    # Build list of all items we need to find (for boundary detection)
    all_10k_items = ['1', '1A', '1B', '1C', '2', '3', '4', '5', '6', '7', '7A', '8', '9', '9A', '9B', '9C', '10', '11', '12', '13', '14', '15']
    
    # Find positions of all items
    all_item_positions = []
    
    for item_num in all_10k_items:
        position = None
        
        # Strategy 1: Find all mentions and pick the best one (highest score)
        mentions = _find_all_item_mentions(full_text, item_num)
        
        # Filter to mentions after TOC
        mentions_after_toc = [m for m in mentions if m['position'] > toc_end]
        
        # Pick the one with the highest header score (not just the first header)
        if mentions_after_toc:
            # Sort by header_score descending, then by position ascending (prefer earlier if tied)
            best_mention = max(mentions_after_toc, key=lambda m: (m['header_score'], -m['position']))
            # Only use it if score indicates it's likely a header
            if best_mention['header_score'] > 0:
                position = best_mention['position']
            else:
                # No clear header, use first mention after TOC
                position = mentions_after_toc[0]['position']
        
        # Strategy 2: Fallback to keyword-based search
        if position is None:
            position = _find_section_by_keyword(full_text, item_num, toc_end)
        
        if position is not None:
            all_item_positions.append({
                'item': item_num,
                'position': position
            })
    
    # Sort by position
    all_item_positions.sort(key=lambda x: x['position'])
    
    # Find end markers (SIGNATURES, EXHIBITS, etc.)
    end_markers = ['SIGNATURES', 'EXHIBIT INDEX', 'POWER OF ATTORNEY']
    end_position = len(full_text)
    for marker in end_markers:
        match = re.search(rf'(?:^|\n)\s*{marker}', full_text, re.IGNORECASE | re.MULTILINE)
        if match and match.start() > toc_end:
            end_position = min(end_position, match.start())
    
    # Extract content for each item
    all_extracted = {}
    
    for i, item_info in enumerate(all_item_positions):
        item_num = item_info['item']
        start_pos = item_info['position']
        
        # Find end position
        if i + 1 < len(all_item_positions):
            item_end = all_item_positions[i + 1]['position']
        else:
            item_end = end_position
        
        # Extract content
        content = full_text[start_pos:item_end].strip()
        
        # Validate content - should have reasonable length
        if content and len(content) > 50:
            all_extracted[item_num] = content
    
    # Return only requested items
    extracted = {}
    for item_num in items_to_extract:
        if item_num in all_extracted:
            extracted[item_num] = all_extracted[item_num]
        elif verbose:
            print(f"Warning: Could not extract Item {item_num} from 10-K")
    
    return extracted


def _extract_10q_items(full_text, items_to_extract, toc_end, verbose=False):
    """
    Extract items from a 10-Q filing.
    Handles the Part I / Part II structure of 10-Q filings.
    Accepts 10-K style item requests and maps them to 10-Q equivalents.
    
    Parameters:
        full_text: Normalized full text of the filing
        items_to_extract: List of item names (e.g., ["1A", "7", "3"])
                         These are 10-K item numbers that will be mapped to 10-Q sections
        toc_end: Estimated end position of table of contents
        verbose: If True, print warnings about missing items
        
    Returns:
        dict: Dictionary of original item number → extracted text
    """
    # 10-Q has two parts with their own items
    # Part I - Financial Information: Items 1-4
    # Part II - Other Information: Items 1-6
    
    all_10q_items = [
        ('Part I', '1'), ('Part I', '2'), ('Part I', '3'), ('Part I', '4'),
        ('Part II', '1'), ('Part II', '1A'), ('Part II', '2'), ('Part II', '3'), 
        ('Part II', '4'), ('Part II', '5'), ('Part II', '6')
    ]
    
    # Find positions of all items
    all_item_positions = []
    
    for part, item_num in all_10q_items:
        position = _find_10q_item_position(full_text, part, item_num, toc_end)
        
        if position is not None:
            full_item_name = f"{part} Item {item_num}"
            all_item_positions.append({
                'item': full_item_name,
                'part': part,
                'item_num': item_num,
                'position': position
            })
    
    # Sort by position
    all_item_positions.sort(key=lambda x: x['position'])
    
    # Find end markers
    end_markers = ['SIGNATURES', 'EXHIBIT INDEX', 'POWER OF ATTORNEY']
    end_position = len(full_text)
    for marker in end_markers:
        match = re.search(rf'(?:^|\n)\s*{marker}', full_text, re.IGNORECASE | re.MULTILINE)
        if match and match.start() > toc_end:
            end_position = min(end_position, match.start())
    
    # Extract content for each item
    all_extracted = {}
    
    for i, item_info in enumerate(all_item_positions):
        full_item_name = item_info['item']
        start_pos = item_info['position']
        
        # Find end position
        if i + 1 < len(all_item_positions):
            item_end = all_item_positions[i + 1]['position']
        else:
            item_end = end_position
        
        # Extract content
        content = full_text[start_pos:item_end].strip()
        
        # Validate content - should have reasonable length
        if content and len(content) > 50:
            all_extracted[full_item_name] = content
    
    # Map requested 10-K items to 10-Q items and return
    extracted = {}
    for item_num in items_to_extract:
        # Check if user requested a 10-K style item number
        if item_num in ITEM_10K_TO_10Q_MAP:
            mapped_item = ITEM_10K_TO_10Q_MAP[item_num]
            if mapped_item in all_extracted:
                extracted[item_num] = all_extracted[mapped_item]
            elif verbose:
                print(f"Warning: Could not extract Item {item_num} (mapped to {mapped_item}) from 10-Q")
        else:
            # Try direct lookup with Part prefix
            for full_name, content in all_extracted.items():
                if f"Item {item_num}" in full_name:
                    extracted[item_num] = content
                    break
            else:
                if verbose:
                    print(f"Warning: Could not extract Item {item_num} from 10-Q - no mapping found")
    
    return extracted


def trim_filing_text(filing_obj) -> str:
    """Return filing text with cover page, TOC, Item 8 (financials),
    signatures, and exhibits stripped out.  Deterministic regex-based
    trimming — no LLM cost."""
    try:
        raw = filing_obj.text()
        text = _normalize_text(raw)

        # --- detect filing type ---
        filing_type = _detect_filing_type(filing_obj, text)
        is_10q = (filing_type == '10-Q')

        # 1. Remove everything before TOC end (cover page + TOC)
        toc_end = _estimate_toc_end(text)
        text = text[toc_end:]

        # 2. Remove financial-statements section (Item 8 for 10-K,
        #    Part I Item 1 for 10-Q) — the largest section, unused by
        #    any of the 8 analysis functions.
        if is_10q:
            # Part I Item 1 = Financial Statements in 10-Q
            fs_start = re.search(
                r'(?:^|\n)[^\n]*?ITEM\s+1[\.\:\s\-–—]+FINANCIAL\s+STATEMENTS',
                text, re.IGNORECASE | re.MULTILINE)
            fs_end = re.search(
                r'(?:^|\n)[^\n]*?ITEM\s+2[\.\:\s\-–—]',
                text, re.IGNORECASE | re.MULTILINE)
            if fs_start and fs_end and fs_end.start() > fs_start.start():
                text = text[:fs_start.start()] + text[fs_end.start():]
        else:
            # Item 8 = Financial Statements in 10-K
            mentions_8 = _find_all_item_mentions(text, '8')
            mentions_9 = _find_all_item_mentions(text, '9')
            if mentions_8 and mentions_9:
                item8_pos = mentions_8[0]['position']
                item9_pos = mentions_9[0]['position']
                if item9_pos > item8_pos:
                    text = text[:item8_pos] + text[item9_pos:]

        # 3. Remove everything after SIGNATURES / EXHIBIT INDEX
        end_markers = ['SIGNATURES', 'EXHIBIT INDEX', 'POWER OF ATTORNEY']
        end_position = len(text)
        for marker in end_markers:
            match = re.search(rf'(?:^|\n)\s*{marker}',
                              text, re.IGNORECASE | re.MULTILINE)
            if match:
                end_position = min(end_position, match.start())
        text = text[:end_position]

        # 4. Strip XBRL inline markup residue
        text = re.sub(r'ix:\w+', '', text)

        # 5. Collapse runs of 3+ blank lines to 2
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

    except Exception as e:
        logger.warning(f"trim_filing_text failed ({e}) — returning raw text")
        return filing_obj.text()


# --- Pydantic models for LLM section extraction --- #

class ExtractedSection(BaseModel):
    item_number: str
    content: str

class ExtractedSections(BaseModel):
    sections: list[ExtractedSection]


def llm_extract_sections(trimmed_text: str, items_needed: List[str],
                          model: str = "gpt-4o-mini") -> dict:
    """Use a single cheap LLM call to extract all missing sections from
    the trimmed filing text.  Returns {item_number: content_text}."""
    if not items_needed:
        return {}

    # Build item descriptions for prompt
    item_descriptions = []
    for item in items_needed:
        defn = ITEM_DEFINITIONS_10K.get(item, {})
        keywords = defn.get('keywords', [])
        title = keywords[0] if keywords else f"Item {item}"
        item_descriptions.append(f"  - Item {item}: {title}")
    item_list_str = "\n".join(item_descriptions)

    prompt = (
        "You are an SEC filing section extractor. From the filing text below, "
        "extract the full content of each requested Item section. "
        "Return ONLY the content of each section — do not summarise.\n\n"
        f"Sections needed:\n{item_list_str}\n\n"
        "Filing text:\n" + trimmed_text
    )

    client = OpenAI()

    try:
        # If text is very long, chunk it and merge results
        if len(trimmed_text) > 120_000:
            chunks = chunk_text(trimmed_text, max_len=100_000, overlap=5_000)
            all_sections = {}
            for chunk in chunks:
                chunk_prompt = (
                    "You are an SEC filing section extractor. From the filing text below, "
                    "extract the full content of each requested Item section. "
                    "Return ONLY the content of each section — do not summarise. "
                    "If a section is not present in this chunk, omit it.\n\n"
                    f"Sections needed:\n{item_list_str}\n\n"
                    "Filing text:\n" + chunk
                )
                resp = client.responses.parse(
                    model=model,
                    input=[{"role": "user", "content": chunk_prompt}],
                    text_format=ExtractedSections,
                ).output_parsed
                for sec in resp.sections:
                    # Keep longest extraction per item across chunks
                    if sec.item_number in items_needed and len(sec.content) >= 100:
                        existing = all_sections.get(sec.item_number, "")
                        if len(sec.content) > len(existing):
                            all_sections[sec.item_number] = sec.content
            return all_sections
        else:
            resp = client.responses.parse(
                model=model,
                input=[{"role": "user", "content": prompt}],
                text_format=ExtractedSections,
            ).output_parsed
            return {
                sec.item_number: sec.content
                for sec in resp.sections
                if sec.item_number in items_needed and len(sec.content) >= 100
            }

    except Exception as e:
        logger.warning(f"llm_extract_sections failed: {e}")
        return {}


def _find_10q_item_position(full_text, part, item_num, toc_end):
    """
    Find the position of a specific item in a 10-Q filing.
    
    Parameters:
        full_text: Normalized full text
        part: 'Part I' or 'Part II'
        item_num: Item number within the part (e.g., '1', '1A', '2')
        toc_end: Estimated TOC end position
        
    Returns:
        int or None: Position of the item header, or None if not found
    """
    # Get expected keywords for this item
    full_item_key = f"{part} Item {item_num}"
    item_def = ITEM_DEFINITIONS_10Q.get(full_item_key, {})
    expected_keywords = [kw.upper() for kw in item_def.get('keywords', [])]
    
    # First, find all Part I and Part II SECTION HEADERS in the document (not references)
    # Section headers typically have formatting like "Part I. Financial Information" or box-drawing
    text_upper = full_text.upper()
    
    # Find real Part headers (with section titles, not inline references)
    part1_header_pos = None
    part2_header_pos = None
    
    # Look for Part I header - should be followed by "Financial Information" or similar
    # Include various dash types: em-dash (—), en-dash (–), hyphen (-), colon (:), spaces
    for m in re.finditer(r'PART\s+I\b(?!\s*I)[\.\:\s\-–—]*(?:FINANCIAL|ITEM)', text_upper[toc_end:]):
        # Check it's a header not a reference (look for "in Part I, Item" pattern = reference)
        context_before = full_text[max(0, toc_end + m.start() - 30):toc_end + m.start()]
        if not re.search(r'(in|to|of|see)\s*$', context_before, re.IGNORECASE):
            part1_header_pos = toc_end + m.start()
            break
    
    # Look for Part II header - should be followed by "Other Information" or similar
    for m in re.finditer(r'PART\s+II\b[\.\:\s\-–—]*(?:OTHER|ITEM)', text_upper[toc_end:]):
        context_before = full_text[max(0, toc_end + m.start() - 30):toc_end + m.start()]
        if not re.search(r'(in|to|of|see)\s*$', context_before, re.IGNORECASE):
            part2_header_pos = toc_end + m.start()
            break
    
    # Build pattern for 10-Q item headers
    if item_num[-1].isalpha():
        # Item like "1A" - exact match required
        pattern = re.compile(
            rf'(?:^|\n)[^\n]*?ITEM\s+{re.escape(item_num)}\b[\.\:\s\-–—]',
            re.IGNORECASE | re.MULTILINE
        )
    else:
        # Item like "1", "2" - don't match if followed by letter
        pattern = re.compile(
            rf'(?:^|\n)[^\n]*?ITEM\s+{re.escape(item_num)}(?![A-Z0-9])[\.\:\s\-–—]',
            re.IGNORECASE | re.MULTILINE
        )
    
    # Determine valid position range based on requested Part
    if part == 'Part I':
        if part1_header_pos is not None and part2_header_pos is not None:
            valid_range = (part1_header_pos, part2_header_pos)
        elif part1_header_pos is not None:
            valid_range = (part1_header_pos, len(full_text))
        else:
            valid_range = (toc_end, part2_header_pos if part2_header_pos else len(full_text))
    else:  # Part II
        if part2_header_pos is not None:
            valid_range = (part2_header_pos, len(full_text))
        else:
            # No Part II header found - look in second half of document
            valid_range = (len(full_text) // 2, len(full_text))
    
    # Find all matches in the valid range
    mentions = []
    for match in pattern.finditer(full_text):
        match_text = match.group(0)
        item_offset = match_text.upper().find('ITEM')
        pos = match.start() + item_offset
        
        # Check if position is in valid range for requested Part
        if valid_range[0] <= pos < valid_range[1]:
            # Get context after for keyword matching
            context_after = full_text[pos:min(len(full_text), pos+500)].upper()
            
            # Check if expected keywords are present in context
            has_expected_keywords = any(kw in context_after for kw in expected_keywords) if expected_keywords else True
            
            score = _is_likely_section_header(full_text, pos, item_num)
            # Boost score if keywords match
            if has_expected_keywords:
                score += 5
                
            mentions.append({
                'position': pos,
                'score': score,
                'has_keywords': has_expected_keywords
            })
    
    if not mentions:
        return None
    
    # Prefer mentions with keywords, then by score
    mentions.sort(key=lambda m: (m['has_keywords'], m['score']), reverse=True)
    
    # Return best match
    best = mentions[0]
    return best['position']


def get_table_of_contents(filing_obj):
    """
    Extracts the table of contents and returns it as a list of dictionaries.
    Uses ITEM_DEFINITIONS for keyword matching.
    
    Returns:
        list: List of dictionaries with 'item' and 'title' keys
    """
    try:
        full_text = filing_obj.text()
        full_text = _normalize_text(full_text)
        
        toc_list = []
        toc_end = _estimate_toc_end(full_text)
        toc_region = full_text[:toc_end]
        
        for item_num, item_def in ITEM_DEFINITIONS.items():
            keywords = item_def.get('keywords', [])
            
            # Pattern for TOC entries
            if item_num[-1].isalpha():
                pattern = re.compile(
                    rf'ITEM\s+{re.escape(item_num)}\b[\.:\s]*([^\n]+)',
                    re.IGNORECASE
                )
            else:
                pattern = re.compile(
                    rf'ITEM\s+{re.escape(item_num)}(?![A-Z])[\.:\s]*([^\n]+)',
                    re.IGNORECASE
                )
            
            for match in pattern.finditer(toc_region):
                title = match.group(1).strip()
                # Clean up title
                title = re.sub(r'\.{2,}.*$', '', title)  # Remove dotted leaders
                title = re.sub(r'\s+\d+\s*$', '', title)  # Remove page numbers
                title = title.strip()
                
                if title and any(kw.upper() in title.upper() for kw in keywords):
                    toc_list.append({
                        'item': item_num,
                        'title': title
                    })
                    break
        
        return toc_list
        
    except Exception as e:
        print(f"Failed to extract TOC: {e}")
        return []


def find_sections_using_toc(full_text, toc_list):
    """
    Legacy function - kept for compatibility.
    Find actual sections using the titles from the table of contents.
    """
    all_item_positions = []
    full_text = _normalize_text(full_text)
    toc_end = _estimate_toc_end(full_text)
   
    for toc_item in toc_list:
        item_num = toc_item['item']
        position = _find_section_by_keyword(full_text, item_num, toc_end)
        
        if position is not None:
            all_item_positions.append({
                'item': item_num,
                'position': position,
                'title': toc_item.get('title', '')
            })

    return all_item_positions


def chunk_text(text, max_len=30_000, overlap=1_000):
    """Split text into overlapping chunks for processing."""
    if max_len <= overlap:
        raise ValueError("max_len must exceed overlap")

    chunks, start, step = [], 0, max_len - overlap
    n = len(text)

    while start < n:
        end = min(start + max_len, n)
        chunks.append(text[start:end])
        if end == n:                # reached the tail
            break
        start += step               # always moves forward
    return chunks


#test
if __name__=="__main__":
    filings_list = fetch_10K_and_10Q_filings("nvda", "2023-01-01", "2025-2-15", form=["10-Q", "10-K"])
    balance_sheets, income_statements, cashflow_statements, balance_sheets_str, income_statements_str, cashflow_statements_str = extract_financials(filings_list)
    assert len(filings_list) > 0
    print(balance_sheets[0])
    
    # Test section extraction
    if filings_list:
        print("\n--- Testing Section Extraction ---")
        items = extract_items_from_filing(filings_list[0], ['1A', '7', '3', '9A'])
        for item_num, content in items.items():
            print(f"Item {item_num}: {len(content)} characters extracted")
