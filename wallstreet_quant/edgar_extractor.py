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

def extract_items_from_filing(filing_obj, items_to_extract):
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
#test
if __name__=="__main__":
    filings_list = fetch_10K_and_10Q_filings("nvda", "2023-01-01", "2025-2-15",form=["10-Q", "10-K"])
    balance_sheets, income_statements, cashflow_statements, balance_sheets_str, income_statements_str, cashflow_statements_str = extract_financials(filings_list)
    assert len(filings_list) > 0
    print(balance_sheets[0])