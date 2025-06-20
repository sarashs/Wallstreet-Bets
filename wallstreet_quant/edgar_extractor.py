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

def extract_items_from_filing_old(filing_obj, items_to_extract):
    """
    Extracts textual sections (e.g., Item 1A, Item 7) from a filing object.
    
    Parameters:
        filing_obj: A filing object returned by the 'edgar' package.
        items_to_extract: List of item names (e.g., ["1", "1A", "7", "3"]).
        
    Returns:
        dict: Dictionary of item number → extracted text
    """
    try:
        # Get the full filing text
        full_text = filing_obj.text()
        
        # Normalize whitespace and case (your original approach)
        full_text = re.sub(r'\s+', ' ', full_text)
        full_text = full_text.replace('\n', ' ').replace('\r', ' ')
        
        # Fixed pattern - the main issues were:
        # 1. \d+ only matches single digit, need \d+ for multiple digits
        # 2. Lookahead pattern should match the main pattern exactly
        item_pattern = re.compile(r'(ITEM\s+(\d+[A-Z]?)\.*)(.*?)((?=ITEM\s+\d+[A-Z]?\.?)|$)', re.IGNORECASE)
        
        # Extract all items
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


        print(f"Original title: '{title}'")
        print(f"Clean title: '{clean_title}'") 
        print(f"Flexible title: '{flexible_title}'")

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

#test
if __name__=="__main__":
    filings_list = fetch_10K_and_10Q_filings("nvda", "2023-01-01", "2025-2-15",form=["10-Q", "10-K"])
    balance_sheets, income_statements, cashflow_statements, balance_sheets_str, income_statements_str, cashflow_statements_str = extract_financials(filings_list)
    assert len(filings_list) > 0
    print(balance_sheets[0])