# balance sheet extractor
### Methods:
# fetch_10K_and_10Q_filings: downloads 10ks and 10qs for a given range
# extract_financials: extrancts balance sheets, income statements and cashflow statements for a list of filings
# extract_items_from_filing: extracts the written part of filings for one file e.g., ["1A", "7", "3"]
# pip install edgartools
import sys
from edgar import *
from edgar.financials import Financials
import pandas as pd
import os
import re

edgar_identity = None
edgar_identity = os.getenv("edgar_identity") # must be a string like "user_name email@server.com" if you don't have an environment variable then set it mannually as I said: "user_name email@server.com"
assert edgar_identity is not None, 'edgar_identity environment variable must be a string like "user_name email@server.com"'
set_identity(edgar_identity)

from edgar import Company

def fetch_10K_and_10Q_filings(ticker: str, start_date: str, end_date: str, form: list = ['10-K']):
    """
    Fetches the 10-K and 10-Q filings for the given ticker within the specified date range.

    Note:
      - Make sure you have set your EDGAR identity (using set_identity) before calling this function.
      - The date filter should be in the form "YYYY-MM-DD:YYYY-MM-DD".

    Parameters:
        ticker (str): The stock ticker (e.g., "AAPL").
        start_date (str): The start date in "YYYY-MM-DD" format.
        end_date (str): The end date in "YYYY-MM-DD" format.

    Returns:
        list: A list-like object of filing objects (or an empty list if no filings are found).
    """
    try:
        # Create a Company object for the given ticker
        company = Company(ticker)
        # Retrieve both 10-K and 10-Q filings for the company
        filings = company.get_filings(form=form) #"10-K",
        # Filter the filings based on the provided date range
        # The filter date string uses the format "start_date:end_date"
        filtered_filings = filings.filter(date=f"{start_date}:{end_date}")
        
        if not filtered_filings:
            print(f"No 10-K or 10-Q filings found for {ticker} between {start_date} and {end_date}.")
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

# Standard 10-K/10-Q item definitions with expected keywords
ITEM_DEFINITIONS = {
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
    Returns list of dicts with position, context, and header likelihood.
    """
    # Pattern to match "ITEM X" or "ITEM X." or "ITEM X:"
    # Handle items like "1A" carefully - don't match "1A" when looking for "1"
    if item_num[-1].isalpha():
        # Item like "1A", "7A" - require exact match
        pattern = re.compile(
            rf'(?:^|\n)\s*ITEM\s+{re.escape(item_num)}\b[\.\:\s]',
            re.IGNORECASE | re.MULTILINE
        )
    else:
        # Item like "1", "7" - don't match if followed by letter
        pattern = re.compile(
            rf'(?:^|\n)\s*ITEM\s+{re.escape(item_num)}(?![A-Z])[\.\:\s]',
            re.IGNORECASE | re.MULTILINE
        )
    
    mentions = []
    for match in pattern.finditer(full_text):
        pos = match.start()
        # Get context after the match (up to 600 chars)
        context_end = min(pos + 600, len(full_text))
        context = full_text[pos:context_end]
        
        # Determine if this is likely a header (actual section start)
        is_header = _is_likely_section_header(full_text, pos, item_num)
        
        mentions.append({
            'position': pos,
            'context': context,
            'is_header': is_header,
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
    next_item = re.search(r'\n\s*ITEM\s+\d+[A-Z]?[\.\:\s]', context_after[100:], re.IGNORECASE)
    if next_item:
        section_length = next_item.start()
    else:
        section_length = len(context_after)
    
    # Scoring system
    score = 0
    if is_in_toc:
        score -= 3
    if has_dotted_line:
        score -= 2
    if has_page_ref:
        score -= 2
    if has_keywords:
        score += 2
    if section_length > 1000:
        score += 2
    elif section_length > 500:
        score += 1
    
    # Check if starts on its own line (strong header indicator)
    line_start = full_text.rfind('\n', max(0, pos-5), pos)
    if line_start != -1 and pos - line_start < 10:
        score += 1
    
    return score > 0


def _find_section_by_keyword(full_text, item_num, start_search_from=0):
    """
    Fallback: Find section using keyword patterns when other methods fail.
    """
    item_def = ITEM_DEFINITIONS.get(item_num, {})
    title_patterns = item_def.get('title_patterns', [])
    
    for title_pat in title_patterns:
        # Build pattern: ITEM X followed by title pattern
        if item_num[-1].isalpha():
            pattern = re.compile(
                rf'(?:^|\n)\s*ITEM\s+{re.escape(item_num)}\b[\.:\s]*{title_pat}',
                re.IGNORECASE | re.MULTILINE
            )
        else:
            pattern = re.compile(
                rf'(?:^|\n)\s*ITEM\s+{re.escape(item_num)}(?![A-Z])[\.:\s]*{title_pat}',
                re.IGNORECASE | re.MULTILINE
            )
        
        matches = list(pattern.finditer(full_text, start_search_from))
        
        for match in matches:
            if _is_likely_section_header(full_text, match.start(), item_num):
                return match.start()
    
    return None


def _estimate_toc_end(full_text):
    """
    Estimate where the table of contents ends.
    The actual content usually starts after this point.
    """
    # Look for common TOC end markers
    toc_end_markers = [
        r'PART\s+I\s*\n',  # Start of Part I
        r'ITEM\s+1\b[^A].*?BUSINESS',  # Item 1 with Business heading
        r'FORWARD[\-\s]*LOOKING\s+STATEMENTS',
        r'SPECIAL\s+NOTE',
    ]
    
    min_pos = len(full_text)
    for marker in toc_end_markers:
        match = re.search(marker, full_text[:50000], re.IGNORECASE)
        if match:
            min_pos = min(min_pos, match.start())
    
    # If we found a reasonable marker, return it; otherwise assume TOC is in first 10k
    if min_pos < 50000:
        return min_pos
    
    return min(10000, len(full_text) // 10)


def extract_items_from_filing(filing_obj, items_to_extract):
    """
    Extracts textual sections (e.g., Item 1A, Item 7) from a filing object.
    Uses multiple strategies with fallbacks to maximize extraction success.
    
    Strategies used (in order):
    1. Find all ITEM mentions and identify which are actual section headers
    2. Use keyword patterns to locate sections
    3. Skip TOC entries by estimating TOC boundary
    
    Parameters:
        filing_obj: A filing object returned by the 'edgar' package.
        items_to_extract: List of item names (e.g., ["1", "1A", "7", "3"]).
        
    Returns:
        dict: Dictionary of item number → extracted text
    """
    try:
        # Get the full filing text
        full_text = filing_obj.text()
        full_text = _normalize_text(full_text)
        
        # Estimate where TOC ends - we want to find sections AFTER this
        toc_end = _estimate_toc_end(full_text)
        
        # Build list of all items we need to find (for boundary detection)
        all_10k_items = ['1', '1A', '1B', '1C', '2', '3', '4', '5', '6', '7', '7A', '8', '9', '9A', '9B', '9C', '10', '11', '12', '13', '14', '15']
        
        # Find positions of all items
        all_item_positions = []
        
        for item_num in all_10k_items:
            position = None
            
            # Strategy 1: Find all mentions and pick the best one (likely header)
            mentions = _find_all_item_mentions(full_text, item_num)
            
            # Filter to mentions after TOC
            mentions_after_toc = [m for m in mentions if m['position'] > toc_end]
            
            # Pick the first one that looks like a header
            header_mentions = [m for m in mentions_after_toc if m['is_header']]
            if header_mentions:
                position = header_mentions[0]['position']
            elif mentions_after_toc:
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
            else:
                print(f"Warning: Could not extract Item {item_num}")
        
        return extracted
        
    except Exception as e:
        print(f"Failed to extract items: {e}")
        import traceback
        traceback.print_exc()
        return {}


def extract_items_from_filing_old(filing_obj, items_to_extract):
    """
    DEPRECATED: Old extraction method kept for reference.
    Use extract_items_from_filing instead.
    """
    try:
        full_text = filing_obj.text()
        full_text = re.sub(r'\s+', ' ', full_text)
        full_text = full_text.replace('\n', ' ').replace('\r', ' ')
        
        item_pattern = re.compile(r'(ITEM\s+(\d+[A-Z]?)\.*)(.*?)((?=ITEM\s+\d+[A-Z]?\.?)|$)', re.IGNORECASE)
        matches = item_pattern.findall(full_text)
        extracted = {}
        
        for match in matches:
            item_number = match[1].upper()
            content = match[2].strip()
            if item_number in items_to_extract:
                extracted[item_number] = content
                
        return extracted
    except Exception as e:
        print(f"Failed to extract items: {e}")
        return {}


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
    Uses the original logic to find items in the TOC.
    
    Returns:
        list: List of dictionaries with 'item' and 'title' keys
    """
    try:
        full_text = filing_obj.text()
        full_text = re.sub(r'[ \t]+', ' ', full_text)
        
        # Define expected keywords for each item type (original logic)
        item_keywords = {
            '1': ['BUSINESS', 'OVERVIEW', 'DESCRIPTION', 'OPERATIONS'],
            '1A': ['RISK FACTORS', 'RISKS', 'RISK'],
            '1B': ['UNRESOLVED', 'STAFF', 'COMMENTS'],
            '1C': ['CYBERSECURITY'],
            '2': ['PROPERTIES', 'PROPERTY'],
            '3': ['LEGAL PROCEEDINGS', 'LITIGATION', 'LEGAL'],
            '4': ['MINE SAFETY', 'MINING'],
            '5': ['MARKET', 'REGISTRANT', 'SECURITIES'],
            '6': ['SELECTED FINANCIAL', 'FINANCIAL DATA'],
            '7': ['MANAGEMENT\'S DISCUSSION', 'MD&A', 'FINANCIAL CONDITION', 'RESULTS OF OPERATIONS'],
            '7A': ['QUANTITATIVE', 'QUALITATIVE', 'MARKET RISK'],
            '8': ['FINANCIAL STATEMENTS', 'CONSOLIDATED'],
            '9': ['CHANGES', 'DISAGREEMENTS', 'ACCOUNTANTS'],
            '9A': ['CONTROLS', 'PROCEDURES', 'INTERNAL CONTROL'],
            '9B': ['OTHER INFORMATION'],
            '10': ['DIRECTORS', 'EXECUTIVE OFFICERS', 'GOVERNANCE'],
            '11': ['EXECUTIVE COMPENSATION', 'COMPENSATION'],
            '12': ['SECURITY OWNERSHIP', 'BENEFICIAL OWNERSHIP'],
            '13': ['CERTAIN RELATIONSHIPS', 'RELATED TRANSACTIONS'],
            '14': ['PRINCIPAL ACCOUNTING', 'FEES', 'SERVICES'],
            '15': ['EXHIBITS', 'FINANCIAL STATEMENT']
        }
        
        toc_list = []
        all_possible_items = ['1', '1A', '1B', '1C', '2', '3', '4', '5', '6', '7', '7A', '8', '9', '9A', '9B', '10', '11', '12', '13', '14', '15', '16']
        
        for item_num in all_possible_items:
            # Find all mentions of this item (original pattern)
            item_pattern = re.compile(rf'ITEM[\s\u00A0\u2000-\u200B\u2028\u2029]+{re.escape(item_num)}(?:\.|:|\s|\u00A0|$)', re.IGNORECASE)
            
            for match in item_pattern.finditer(full_text):
                start_pos = match.start()
                
                # Check context around this mention for relevant keywords (original logic)
                context_start = max(0, start_pos - 200)
                context_end = min(len(full_text), start_pos + 500)
                context = full_text[context_start:context_end].upper()
                
                # Check if any of the expected keywords appear in the context
                keywords = item_keywords.get(item_num, [])
                keyword_found = any(keyword in context for keyword in keywords)
                
                if keyword_found:
                    # Find where this item would end (next ITEM or end of line for TOC)
                    next_item_pattern = re.compile(rf'ITEM[\s\u00A0\u2000-\u200B\u2028\u2029]+\d+[A-Z]?', re.IGNORECASE)
                    next_match = next_item_pattern.search(full_text, start_pos + len(match.group(0)))
                    
                    if next_match:
                        content_end = next_match.start()
                    else:
                        # Look for end of line or section
                        line_end = full_text.find('\n', start_pos + len(match.group(0)))
                        content_end = line_end if line_end != -1 else len(full_text)
                    
                    # Extract the potential TOC entry
                    toc_entry = full_text[start_pos:content_end].strip()
                    
                    # If this looks like a TOC entry (short, contains item keywords), save it
                    if len(toc_entry) < 1000:  # TOC entries are typically short
                        # Extract the full title (everything after "ITEM X" part)
                        title_match = re.search(rf'ITEM[\s\u00A0\u2000-\u200B\u2028\u2029]+{re.escape(item_num)}[.\s]*(.+)', toc_entry, re.IGNORECASE)
                        if title_match:
                            full_title = title_match.group(1).strip()
                            # Clean up the title (remove page numbers, dots, etc.)
                            full_title = re.sub(r'\.{2,}.*$', '', full_title)  # Remove dotted leaders and page numbers
                            full_title = re.sub(r'\s+\d+\s*$', '', full_title)  # Remove trailing page numbers
                            full_title = full_title.strip()
                            
                            if full_title:
                                toc_list.append({
                                    'item': item_num,
                                    'title': full_title,
                                    'toc_entry': toc_entry
                                })
                                break  # Found the TOC entry for this item, stop looking
        
        return toc_list
        
    except Exception as e:
        print(f"Failed to extract TOC: {e}")
        return []

def extract_items_from_filing(filing_obj, items_to_extract):
    """
    Extracts textual sections (e.g., Item 1A, Item 7) from a filing object.
    First finds the table of contents, then uses full titles to locate sections.
    
    Parameters:
        filing_obj: A filing object returned by the 'edgar' package.
        items_to_extract: List of item names (e.g., ["1", "1A", "7", "3"]).
        
    Returns:
        dict: Dictionary of item number → extracted text
    """
    try:
        # Get the full filing text
        full_text = filing_obj.text()
        
        # Light normalization - preserve structure
        full_text = re.sub(r'[ \t]+', ' ', full_text)  # Normalize spaces/tabs only
        
        # Step 1: Get table of contents as a list
        toc_list = get_table_of_contents(filing_obj)
        
        # Step 2: Use TOC to find actual sections
        all_item_positions = find_sections_using_toc(full_text, toc_list)
        
        # Sort all found items by position
        all_item_positions.sort(key=lambda x: x['position'])
        
        # Extract content between item positions
        all_extracted = {}
        
        for i, item_info in enumerate(all_item_positions):
            item_num = item_info['item']
            start_pos = item_info['position']
            
            # Find where this item's content ends
            if i + 1 < len(all_item_positions):
                # Next item starts here
                end_pos = all_item_positions[i + 1]['position']
            else:
                # Last item - look for common ending markers
                end_markers = ['SIGNATURES', 'EXHIBIT INDEX', 'EXHIBITS']
                end_pos = len(full_text)
                
                for marker in end_markers:
                    marker_pattern = re.compile(rf'(?:^|\n)\s*{marker}', re.IGNORECASE | re.MULTILINE)
                    marker_match = marker_pattern.search(full_text, start_pos)
                    if marker_match:
                        end_pos = min(end_pos, marker_match.start())
            
            # Extract the content
            content = full_text[start_pos:end_pos].strip()
            
            if content:
                all_extracted[item_num] = content
        
        # Now filter to return only the requested items
        extracted = {}
        for item_num in items_to_extract:
            if item_num in all_extracted:
                extracted[item_num] = all_extracted[item_num]
        
        return extracted
        
    except Exception as e:
        print(f"Failed to extract items: {e}")
        return {}

def find_sections_using_toc(full_text, toc_list):
    """
    Find actual sections using the titles from the table of contents.
    Uses the second match (first is TOC, second is actual section).
    """
    all_item_positions = []
   
    for toc_item in toc_list:
        item_num = toc_item['item']
        title = toc_item['title']

        # Clean the title to get the core title without page references
        # Remove common suffixes like "Pages X-Y", "Page X", "None"
        clean_title = re.sub(r'\s+Pages?\s*\d+.*$', '', title, flags=re.IGNORECASE)
        clean_title = re.sub(r'\s+None\s*$', '', clean_title, flags=re.IGNORECASE)
        if ':' in clean_title:
            clean_title = clean_title.split(':')[0].strip()    
        if not clean_title:
            continue

        # Create search patterns to find the actual section
        clean_title = clean_title.replace('\u2019', "'").replace('\u2018', "'")
        # 2) escape regex metachars
        escaped_title = re.escape(clean_title)

        # 3) relax spacing, punctuation, quotes, etc.
        escaped_title = escaped_title.replace(r'\ ', r'[\s\u00A0\u2000-\u200B\u2028\u2029]+')
        escaped_title = escaped_title.replace(r"'", r"[\'’]")    # straight or curly apostrophe, or none
        escaped_title = escaped_title.replace(r'\"', r'["“”]?')   # straight or curly quote, or none
        escaped_title = escaped_title.replace(r'\-', r'[-–—]')
        escaped_title = escaped_title.replace(r'\&', r'(?:&|and)')
        escaped_title = escaped_title.replace(r'\.', r'\.?')
        escaped_title = escaped_title.replace(r'\,', r'\,?')
        escaped_title = escaped_title.replace(r'\:', r'\:?')

        # 4) assign
        flexible_title = escaped_title

        # Multiple patterns to try
        patterns = [
            rf'ITEM\s+{re.escape(item_num)}\b[\s\S]*?{flexible_title}',
        ]

        found = False
        for pattern in patterns:
            try:
                item_pattern = re.compile(pattern, re.IGNORECASE | re.MULTILINE)

                # Find ALL matches in the document
                matches = list(item_pattern.finditer(full_text))

                # Take the second match if it exists (first is TOC, second is actual section)
                if len(matches) >= 2:
                    actual_section_match = matches[1]
                    position = actual_section_match.start()

                    all_item_positions.append({
                        'item': item_num,
                        'position': position,
                        'match_text': actual_section_match.group(0).strip(),
                        'title': clean_title
                    })
                    found = True
                    break

            except re.error:
                continue
            
        # If we didn't find a second match with any pattern, this item might not have a section
        if not found:
            print(f"Warning: Could not find actual section for Item {item_num}")

    return all_item_positions

def chunk_text(text, max_len=30_000, overlap=1_000):
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
    filings_list = fetch_10K_and_10Q_filings("nvda", "2023-01-01", "2025-2-15",form=["10-Q", "10-K"])
    balance_sheets, income_statements, cashflow_statements, balance_sheets_str, income_statements_str, cashflow_statements_str = extract_financials(filings_list)
    assert len(filings_list) > 0
    print(balance_sheets[0])