"""
A collection of tools for scanning and calculating financial metrics for stocks.

This module provides functions to:
- Scan for stocks that meet specific criteria (e.g., CANSLIM).
- Calculate and save a wide range of common financial metrics and technical indicators
  to the database, such as revenue growth, EPS growth, relative strength, and more.
- Find top competitors for a given stock using a hybrid approach of financial metrics
  and business summary text analysis.
- Calculate a proprietary fundamental score for companies.
"""
from datetime import datetime, timedelta
import logging
import numpy as np
import pandas as pd
import traceback, operator
from sklearn.impute import SimpleImputer
from sqlalchemy import func, update, desc
from sqlalchemy.orm import Session
from scipy import stats

from core.db import get_db
from core.model import Company, Exchange, Financials, PriceHistory, object_as_dict
from tools.data_manager import DataManager

logger = logging.getLogger(__name__)

canslim_scan_criteria = {  
    "Revenue Growth (YOY)": {"min": 25},
    "EPS Growth (YOY)": {"min": 20},
    "3-Year EPS CAGR": {"min": 25},
    "Return on Equity (ROE)": {"min": 17},
    "Debt-to-Equity Ratio": {"max": 1.5},
    "Institutional Ownership": {"min": 15, "max": 70},
    "P/E Ratio": {"min": 10, "max": 100},
    "Average Daily Volume": {"min": 100000},
    "Shares Outstanding": {"max": 50000000},
    "Price Relative to 52-Week High (%)": {"min": 70, "max": 110},
    "Relative Strength (Percentile)": {"min": 70},
    "Expanding Volume": {},
}

index_name_mapping = {
    "Total Revenue": ["Total Revenue", "TotalRevenue", "Revenue", "Total Revenues", "Total Revenues", "totalRevenue"],
    "Net Income": ["Net Income", "NetIncome", "Net Profit", "Net Earnings", "Net Income Continuous Operations", "NetIncomeContinuousOperations", "Net Income Discontinuous Operations", "NetIncomeDiscontinuousOperations","Net Income Common Stockholders", "NetIncomeCommonStockholders", "Net Income From Continuing And Discontinued Operation", "NetIncomeFromContinuingAndDiscontinuedOperation", "Net Income From Continuing Operation Net Minority Interest", "NetIncomeFromContinuingOperationNetMinorityInterest","Net Income Including Noncontrolling Interests", "NetIncomeIncludingNoncontrollingInterests", "netIncome"],
    "Shares (Diluted)": ["Shares (Diluted)", "SharesDiluted", "Diluted Average Shares", "DilutedAverageShares", "Weighted Average Diluted Shares", "WeightedAverageDilutedShares", "DilutedShares", "Diluted Shares"],
    "Operating Income": ["Operating Income", "OperatingIncome", "operatingIncome"], 
    "Basic Average Shares": ["Basic Average Shares", "BasicAverageShares"],
    "EBIT": ["EBIT", "ebit"], 
    "Interest Expense": ["Interest Expense", "InterestExpense"],
    "Dividends Paid": ["Dividends Paid", "DividendsPaid", "CashDividendsPaid", "dividendsPaid"], 
    "Cost Of Revenue": ["Cost Of Revenue", "CostOfRevenue", "costOfRevenue"],
    "Operating Cash Flow": ["Operating Cash Flow", "OperatingCashFlow", "totalCashFromOperatingActivities"],
    "Capital Expenditures": ["Capital Expenditures", "CapitalExpenditures", "CapitalExpenditure", "capitalExpenditures"],
    "Free Cash Flow": ["Free Cash Flow", "FreeCashFlow", "freeCashFlow"],
    "Short Term Debt": ["Short Term Debt", "ShortTermDebt", "Short Long Term Debt", "Current Debt", "CurrentDebt", "CurrentDebtAndCapitalLeaseObligation", "Commercial Paper"],
    "Long Term Debt": ["Long Term Debt", "LongTermDebt", "Long Term Liabilities", "Deferred Long Term Liabilities", "Long Term Debt And Capital Lease Obligation", "LongTermDebtAndCapitalLeaseObligation", "longTermDebt"],
    "Total Equity": ["Total Equity", "TotalEquity", "Stockholders Equity", "StockholdersEquity", "Total Stockholders Equity", "Other Stockholders Equity", "totalStockholderEquity"],
    "Current Assets": ["Current Assets", "CurrentAssets", "Total Current Assets", "Other Current Assets", "Current Deferred Assets"],
    "Total Assets": ["Total Assets", "TotalAssets", "totalAssets"],
    "Current Liabilities": ["Current Liabilities", "CurrentLiabilities", "Total Current Liabilities", "Other Current Liabilities", "Current Deferred Liabilities"],
    "Total Liabilities": ["Total Liabilities", "TotalLiabilitiesNetMinorityInterest", "Other Liabilities", "Total Non Current Liabilities", "DeferredTaxLiabilities", "Total Liabilities Net Minority Interest", "totalLiab"],
    "Cash And Equivalent": ["Cash Cash Equivalents And Short Term Investments", "CashAndCashEquivalents", "Cash Equivalents", "Cash Cash Equivalents And Federal Funds Sold", "cash"],
    "Receivables": ["Receivables", "Other Receivables", "Non Current Accounts Receivable", "Accounts Receivable", "Gross Accounts Receivable", "AccountsReceivable"],
    "Payables": ["Payables", "TaxPayables", "Accounts Payable", "Other Payable", "AccountsPayable", "PayablesAndAccruedExpenses", "TradeandOtherPayablesNonCurrent"],
    "Inventory": ["Inventory", "Finished Goods", "Raw Materials", "Other Inventories", "inventory"],
}

def standardize_index_names(df, index_name_mapping):
    """Standardizes index names in a DataFrame based on the given mapping.

    Args:
        df (pd.DataFrame): The DataFrame to standardize.
        index_name_mapping (dict): The mapping of standard names to variations.

    Returns:
        pd.DataFrame: A new DataFrame with standardized index names.
    """
    df_new = df.copy() #copy the dataframe

    if 'index' not in df_new.columns:
      return df_new #nothing to change
    
    new_index_values = []
    for idx in df_new['index'].unique():
        found = False
        for standard_name, variations in index_name_mapping.items():
            if idx in variations:
                new_index_values.append((idx, standard_name)); found = True; break
        if not found:
            logger.warning(f"Found new index name not in mapping: {idx}")
            # Do not modify the mapping. Just treat it as its own standard name for this run.
            new_index_values.append((idx, idx))

    for old, new in new_index_values:
       df_new.loc[df_new['index'] == old, 'index'] = new

    return df_new

def _get_passing_ids(companies, check_func):
    """Helper to apply a check function to a list of companies and return their IDs."""
    return {c.id for c in companies if check_func(c)}

def scan_canslim_stocks_from_db(market="us", canslim_scan_criteria=canslim_scan_criteria):
    """Scan all stocks in the database and return a list of company tickers that pass all the canslim_criteria."""
    db = next(get_db())
    try:
        # Get all companies in the specified market
        companies = db.query(Company).filter(Company.exchange.in_(db.query(Exchange.exchange_code).filter(Exchange.country_code == market))).all()
        logger.info(f"Total companies to scan: {len(companies)}")
        if not companies:
            return []

        # --- Refactored: Data-driven approach for checking criteria ---
        def is_valid_and_compare(value, op, threshold):
            return value is not None and op(value, threshold)

        checkers = {
            "Revenue Growth (YOY)": lambda c, v: is_valid_and_compare(c.revenuegrowth_quarterly_yoy, operator.ge, v["min"] / 100),
            "EPS Growth (YOY)": lambda c, v: is_valid_and_compare(c.earningsgrowth_quarterly_yoy, operator.ge, v["min"] / 100),
            "3-Year EPS CAGR": lambda c, v: is_valid_and_compare(c.eps_cagr_3year, operator.ge, v["min"] / 100),
            "Return on Equity (ROE)": lambda c, v: is_valid_and_compare(c.returnonequity, operator.ge, v["min"] / 100),
            "Debt-to-Equity Ratio": lambda c, v: is_valid_and_compare(c.debttoequity, operator.le, v["max"]),
            "Institutional Ownership": lambda c, v: c.heldpercentinstitutions is not None and v["min"] / 100 <= c.heldpercentinstitutions <= v["max"] / 100,
            "P/E Ratio": lambda c, v: c.trailingpe is not None and v["min"] <= c.trailingpe <= v["max"],
            "Average Daily Volume": lambda c, v: is_valid_and_compare(c.averagevolume, operator.ge, v["min"]),
            "Shares Outstanding": lambda c, v: is_valid_and_compare(c.sharesoutstanding, operator.le, v["max"]),
            "Price Relative to 52-Week High (%)": lambda c, v: c.price_relative_to_52week_high is not None and v["min"] <= c.price_relative_to_52week_high <= v["max"],
            "Relative Strength (Percentile)": lambda c, v: is_valid_and_compare(c.relative_strength_percentile_252, operator.ge, v["min"]),
            "Expanding Volume": lambda c, v: c.expanding_volume is True,
        }

        logger.info("running canslim scan...")
        passing_company_ids_by_metric = {}
        for metric, criteria in canslim_scan_criteria.items():
            if checker_func := checkers.get(metric):
                passing_company_ids_by_metric[metric] = _get_passing_ids(companies, lambda c: checker_func(c, criteria))

        # Find companies that pass all criteria
        if not passing_company_ids_by_metric:
            return []
        final_passing_company_ids = set.intersection(*passing_company_ids_by_metric.values())

        # Convert company_ids back to symbols
        passing_companies = db.query(Company).filter(Company.id.in_(final_passing_company_ids)).all()
        passing_tickers = [company.symbol for company in passing_companies]

        return passing_tickers

    except Exception as e:
        traceback.print_exc()
        logger.error(f"An unexpected error occurred during the scan: {e}")
        return []
    finally:
        db.close()

def calculate_and_save_common_values_for_scanner(market: str = "us", tickers: list[str] = None):
    """ Calculates and saves common values for all companies in the specified market. """
    db = next(get_db())
    try:
        if tickers:
            tickers = DataManager().resolve_symbols_batch(tickers)
        # Get all companies in the specified market
        if market and tickers:
            companies = db.query(Company).filter(Company.isactive == True, Company.symbol.in_(tickers), Company.exchange.in_(db.query(Exchange.exchange_code).filter(Exchange.country_code == market))).all()
        elif market:
            companies = db.query(Company).filter(Company.isactive == True, Company.exchange.in_(db.query(Exchange.exchange_code).filter(Exchange.country_code == market))).all()
        elif tickers:
            companies = db.query(Company).filter(Company.symbol.in_(tickers)).all()
            
        company_ids = [c.id for c in companies]
        logger.info(f"Total companies to run calculation: {len(company_ids)}")
        if not company_ids:
            return []

        calculate_and_save_common_values(db, company_ids, not tickers)

        # Update all companies' last_price_date to current data
        logger.info("updating companies' last_price_date")
        db.query(Company).filter(Company.id.in_(company_ids)).update({Company.last_price_date: func.current_date()}, synchronize_session=False)
        db.commit()
    except Exception as e:
        traceback.print_exc()
        logger.error(f"An unexpected error occurred during the scan: {e}")
        return []
    finally:
        db.close()

def calculate_and_save_common_values(db: Session, company_ids: list[int], calculate_percentile: bool = True):
    """
    Orchestrates the calculation and saving of various common metrics for a list of companies.

    This function acts as a dispatcher, calling specific calculation functions for:
    - Revenue and EPS growth
    - Price relative to 52-week high
    - Expanding volume
    - Other financial ratios
    - Quarterly trends
    - Relative strength and fundamental score percentiles (optional)

    Args:
        db: The database session.
        company_ids: A list of company IDs to process.
        calculate_percentile: If True, calculates and saves percentile-based scores.
                              Set to False to skip these more intensive calculations.
    """
    calculate_revenue_growth_yoy(db, company_ids)   
    calculate_eps_growth_yoy_and_cagr(db, company_ids)
    calculate_price_relative_to_52week_high(db, company_ids)
    calculate_expanding_volume(db, company_ids)
    calculate_and_save_other_ratios(db, company_ids)
    calculate_quarterly_trends(db, company_ids)
    if calculate_percentile:
        calculate_relative_strength_percentile(db, company_ids)
        fundamental_score_calculator = FundamentalScoreCalculator()
        fundamental_score_calculator.calculate_fundamental_score_and_percentile(db, company_ids)

    # calculate_return_on_equity(db, company_ids)      #use returnonequity from yfinance
    # calculate_debt_to_equity_ratio(db, company_ids)      #use debttoequity from yfinance
    # calculate_average_daily_volume(db, company_ids)   #use averagevolume from yfinance
    # calculate_shares_outstanding(db, company_ids)   #use sharesoutstanding from yfinance
    # calculate_pe_ratio_and_eps_trailing_twelve_months(db, company_ids)   #use trailingpe and epstrailingtwelvemonths from yfinance


def calculate_revenue_growth_yoy(db: Session, company_ids: list[int], batch_size=200):
    """Calculates Year-over-Year Revenue Growth and updates the Company table."""
    logger.info("running calculate_revenue_growth_yoy...")
    update_data = []
    for i in range(0, len(company_ids), batch_size):
        batch_ids = company_ids[i:i + batch_size]
        logger.info(f"Processing batch {i // batch_size + 1} of companies: {len(batch_ids)} companies")

        # Get the last 5 quarters of revenue for the current batch of companies
        revenues = db.query(Financials).filter(
            Financials.company_id.in_(batch_ids),
            Financials.type == "quarterly_income_statement",
            Financials.index.in_(index_name_mapping["Total Revenue"]),
        ).all()

        if not revenues:
            logger.warning(f"No data found for companies: {batch_ids}")
            continue

        revenues_df = pd.DataFrame([
            {
                'company_id': r.company_id,
                'report_date': r.report_date,
                'index': r.index,  # keep the old index
                'value': r.value,
            } for r in revenues
        ])

        # Combine data in a dataframe
        if not revenues_df.empty:
            # Standardize the index
            revenues_df = standardize_index_names(revenues_df, index_name_mapping)
            # set the good index
            revenues_df = revenues_df.pivot_table(index=['company_id', 'report_date'], columns='index', values='value').reset_index()
            # Group by company and sort by date
            revenues_df.sort_values(by=['company_id', 'report_date'], inplace=True)
            # Calculate current and previous year's revenue (quarterly)
            revenues_df['previous_revenue'] = revenues_df.groupby('company_id')['Total Revenue'].shift(4)  # Assuming quarterly reports
            # Calculate YOY growth
            revenues_df['yoy_growth'] = np.where(
                revenues_df["previous_revenue"] == 0,
                np.where(
                    revenues_df["Total Revenue"] > 0,
                    np.inf,
                    np.nan
                ),
                (revenues_df["Total Revenue"] - revenues_df["previous_revenue"]) / abs(revenues_df["previous_revenue"])
            )

            # Update Company table with the last calculated values
            for company_id, group in revenues_df.groupby('company_id'):
                last_row = group.iloc[-1]
                update_data.append({
                    "id": company_id,
                    "revenuegrowth_quarterly_yoy": float(last_row['yoy_growth']),
                    "last_revenue_report_date": last_row['report_date']
                })

    if update_data:
        #print(f"{update_data[0]=}")
        db.execute(update(Company), update_data)
        db.commit()

def calculate_eps_growth_yoy_and_cagr(db: Session, company_ids: list[int], batch_size=200):
    """Calculates EPS Growth (YOY) and 3-Year EPS CAGR, updating the Company table in a single bulk operation.

    Args:
        db: The database session.
        company_ids: A list of company IDs to process.
    """
    logger.info("running calculate_eps_growth_yoy_and_cagr...")
    update_data = []
    for i in range(0, len(company_ids), batch_size):
        batch_ids = company_ids[i:i + batch_size]
        logger.info(f"Processing batch {i // batch_size + 1} of companies: {len(batch_ids)} companies")

        # Get the last 5 quarters and 3 years of EPS data for the current batch of companies
        eps_values = db.query(Financials).filter(
            Financials.company_id.in_(batch_ids),
            Financials.type.in_(
                ["quarterly_income_statement", "annual_income_statement"]
            ),
            Financials.index.in_(
                index_name_mapping["Net Income"]
                + index_name_mapping["Shares (Diluted)"]
            ),
        ).all()

        if not eps_values:
            logger.warning(f"No data found for companies: {batch_ids}")
            continue

        eps_df = pd.DataFrame(
            [
                {
                    "company_id": e.company_id,
                    "report_date": e.report_date,
                    "type": e.type,
                    "index": e.index,
                    "value": e.value,
                }
                for e in eps_values
            ]
        )

        eps_df = standardize_index_names(eps_df, index_name_mapping)
        eps_df = eps_df.pivot_table(
            index=["company_id", "report_date", "type"],
            columns="index",
            values="value",
            fill_value=np.nan, # Changed fill_value to np.nan
        ).reset_index()

        # --- Fill missing "Shares (Diluted)" ---
        if "Shares (Diluted)" in eps_df.columns:
            eps_df.sort_values(by=["company_id", "report_date"], inplace=True)
            eps_df["Shares (Diluted)"] = eps_df.groupby("company_id")["Shares (Diluted)"].ffill() # Forward fill
            eps_df["Shares (Diluted)"] = eps_df.groupby("company_id")["Shares (Diluted)"].bfill() # Backward fill if needed

        if "Net Income" not in eps_df.columns or "Shares (Diluted)" not in eps_df.columns:
            logger.warning(f"missing Net Income or Shares (Diluted), {eps_df.columns=}")
            continue

        eps_df["EPS"] = eps_df["Net Income"] / eps_df["Shares (Diluted)"]

        eps_q = eps_df[eps_df["type"] == "quarterly_income_statement"].copy()
        eps_a = eps_df[eps_df["type"] == "annual_income_statement"].copy()

        # **Correct Date Handling**
        eps_q["report_date"] = pd.to_datetime(eps_q["report_date"], utc=True)  # Convert to date and specify utc=True
        eps_a["report_date"] = pd.to_datetime(eps_a["report_date"], utc=True)

        # EPS Growth YOY
        eps_q.sort_values(by=["company_id", "report_date"], inplace=True)

        eps_q['report_year'] = eps_q['report_date'].dt.year
        eps_q['report_quarter'] = eps_q['report_date'].dt.to_period('Q')

        # Find last report quarter of each company
        last_report_quarter = eps_q.groupby('company_id')['report_quarter'].transform('max')
        eps_q['is_last_quarter'] = eps_q['report_quarter'] == last_report_quarter

        # Filter rows to only keep the last report per quarter
        eps_q = eps_q.groupby(['company_id', 'report_quarter']).tail(1)

        eps_q['previous_EPS'] = eps_q.groupby(['company_id'])['EPS'].shift(4)
        #print(f"{eps_q.columns=}, {eps_q.values=}")
        eps_q["yoy_growth"] = np.where(
            eps_q["previous_EPS"] == 0,
            np.where(
                eps_q["EPS"] > 0,
                np.inf,  # Infinite growth if previous EPS is 0 and current EPS is positive
                np.nan    # NaN if both are 0
            ),
            (eps_q["EPS"] - eps_q["previous_EPS"]) / abs(eps_q["previous_EPS"]) # Use abs() for previous EPS
        )

        # Prepare data for bulk update for earningsgrowth
        for company_id, group in eps_q.groupby('company_id'):
            last_row = group.iloc[-1]
            company_update = {
                "id": company_id,
                "earningsgrowth_quarterly_yoy": float(last_row['yoy_growth']),
                "last_eps_report_date": last_row['report_date']
            }
            update_data.append(company_update)

        # 3-Year EPS CAGR
        eps_a.sort_values(by=["company_id", "report_date"], inplace=True)

        # Group by company_id and get the last and the third to last EPS
        eps_a_grouped = eps_a.groupby("company_id").agg(
            last_eps=('EPS', 'last'),
            three_years_ago_eps=('EPS', lambda x: x.iloc[-3] if len(x) >= 3 else np.nan),
            last_report_date=('report_date', 'last')
        ).reset_index()

        eps_a_grouped["eps_cagr"] = (
            (eps_a_grouped["last_eps"] / eps_a_grouped["three_years_ago_eps"]) ** (1 / 3)
        ) - 1
        eps_a_grouped.dropna(subset=["eps_cagr"], inplace=True)

        # Merge CAGR data into update_data
        for index, row in eps_a_grouped.iterrows():
            company_id = row['company_id']
            # Find the company_update in update_data
            company_update = next((item for item in update_data if item["id"] == company_id), None)
            if company_update:
                company_update["eps_cagr_3year"] = float(row['eps_cagr'])
            else:
                # Add a new entry if the company is not in update_data
                update_data.append({
                    "id": company_id,
                    "eps_cagr_3year": float(row['eps_cagr']),
                    "earningsgrowth_quarterly_yoy": None,
                    "last_eps_report_date": row['last_report_date']
                })

    # Perform bulk update
    if update_data:
        #print(f"{update_data=}")
        db.execute(update(Company), update_data)
        db.commit()

def calculate_return_on_equity(db: Session, company_ids: list[int], batch_size=200):
    """Calculates Return on Equity (ROE) and updates the Company table.

    Args:
        db: The database session.
        company_ids: A list of company IDs to process.
    """
    logger.info("running calculate_return_on_equity...")
    update_data = []
    for i in range(0, len(company_ids), batch_size):
        batch_ids = company_ids[i:i + batch_size]
        logger.info(f"Processing batch {i // batch_size + 1} of companies: {len(batch_ids)} companies")

        # Get the last Net Income and Total Equity for each company
        income_values = db.query(Financials).filter(
            Financials.company_id.in_(batch_ids),
            Financials.type == "annual_income_statement",
            Financials.index.in_(index_name_mapping["Net Income"]),
        ).all()

        balance_values = db.query(Financials).filter(
            Financials.company_id.in_(batch_ids),
            Financials.type == "annual_balance_sheet",
            Financials.index.in_(index_name_mapping["Total Equity"]),
        ).all()

        # Create DataFrames
        income_df = pd.DataFrame([
            {
                'company_id': i.company_id,
                'report_date': i.report_date,
                'index': i.index,
                'Net Income': i.value,
            } for i in income_values
        ])

        balance_df = pd.DataFrame([
            {
                'company_id': b.company_id,
                'report_date': b.report_date,
                'index': b.index,
                'Total Equity': b.value,
            } for b in balance_values
        ])

        # Early exit if no data
        if income_df.empty or balance_df.empty:
            logger.warning(f"No data found for companies: {batch_ids}")
            continue

        # Standardize the index
        income_df = standardize_index_names(income_df, index_name_mapping)
        balance_df = standardize_index_names(balance_df, index_name_mapping)

        # Merge dataframes
        merged_df = pd.merge(income_df, balance_df, on=['company_id', 'report_date'], how='inner')

        # Calculate ROE for each company
        def calculate_roe(group):
            """Calculates the Return on Equity (ROE) for a single company (group)."""
            # Find the most recent fiscal year
            max_date = group['report_date'].max()
            last_data = group[group['report_date'] == max_date]

            if last_data.empty:
                return pd.Series({'ROE': np.nan, 'last_roe_report_date': max_date})

            # Manage cases where the current year have more than one rows: take the last one
            last_data = last_data.iloc[-1]

            net_income = last_data['Net Income']
            stockholders_equity = last_data['Total Equity']

            # Manage the case where Total Equity is equal to 0
            if stockholders_equity == 0:
                return pd.Series({'ROE': np.nan, 'last_roe_report_date': max_date})

            return pd.Series({'ROE': (net_income / stockholders_equity), 'last_roe_report_date': max_date})

        roe_df = merged_df.groupby('company_id').apply(calculate_roe).reset_index()

        # Prepare data for bulk update
        for index, row in roe_df.iterrows():
            update_data.append({
                "id": row['company_id'],
                "returnonequity": float(row['ROE']),
                "last_roe_report_date": row['last_roe_report_date']
            })

    # Update Company table with bulk update
    if update_data:
        db.execute(update(Company), update_data)
        db.commit()

def calculate_debt_to_equity_ratio(db: Session, company_ids: list[int], batch_size=200):
    """Calculates Debt-to-Equity Ratio and updates the Company table.

    Args:
        db: The database session.
        company_ids: A list of company IDs to process.
        batch_size: The number of companies to process in each batch.
    """
    logger.info("running calculate_debt_to_equity_ratio...")
    update_data = []
    for i in range(0, len(company_ids), batch_size):
        batch_ids = company_ids[i:i + batch_size]
        logger.info(f"Processing batch {i // batch_size + 1} of companies: {len(batch_ids)} companies")

        # Get the last Short Term Debt, Long Term Debt, and Total Equity for each company
        financial_data = db.query(Financials).filter(
            Financials.company_id.in_(batch_ids),
            Financials.type == "annual_balance_sheet",
            Financials.index.in_(
                index_name_mapping["Short Term Debt"]
                + index_name_mapping["Long Term Debt"]
                + index_name_mapping["Total Equity"]
            ),
        ).all()

        if not financial_data:
            logger.warning(f"No data found for companies: {batch_ids}")
            continue

        financial_df = pd.DataFrame([
            {
                'company_id': f.company_id,
                'report_date': f.report_date,
                'index': f.index,
                'value': f.value,
            } for f in financial_data
        ])

        financial_df = standardize_index_names(financial_df, index_name_mapping)

        # Pivot the DataFrame
        financial_df = financial_df.pivot_table(
            index=["company_id", "report_date"],
            columns="index",
            values="value",
            fill_value=0,
        ).reset_index()

        # Calculate DTE for each company
        for company_id in batch_ids:
            company_data = financial_df[financial_df["company_id"] == company_id].copy()
            if company_data.empty:
                continue

            # Find the most recent fiscal year
            max_year = company_data['report_date'].max()
            last_data = company_data[company_data['report_date'] == max_year]

            if last_data.empty:
                continue

            # Manage cases where the current year have more than one rows: take the last one
            last_data = last_data.iloc[-1]

            # Check if Total Equity is 0
            total_equity = last_data.get('Total Equity', 0)
            if total_equity == 0 or pd.isna(total_equity):
                debttoequity = None
            else:
                # Handle nan in short and long term debt
                short_term_debt = last_data.get('Short Term Debt', 0) if not pd.isna(last_data.get('Short Term Debt', np.nan)) else 0
                long_term_debt = last_data.get('Long Term Debt', 0) if not pd.isna(last_data.get('Long Term Debt', np.nan)) else 0
                debttoequity = (short_term_debt + long_term_debt) / total_equity

            update_data.append({
                "id": company_id,
                "debttoequity": float(debttoequity * 100) if debttoequity is not None else debttoequity,
                "last_dte_report_date": max_year
            })

    # Update Company table with bulk update
    if update_data:
        db.execute(update(Company), update_data)
        db.commit()

def calculate_average_daily_volume(db: Session, company_ids: list[int], batch_size=200):
    """Calculates the Average Daily Volume for each company and updates the Company table.

    Args:
        db: The database session.
        company_ids: A list of company IDs to process.
        batch_size: The number of companies to process in each batch.
    """
    logger.info("running calculate_average_daily_volume...")
    update_data = []
    for i in range(0, len(company_ids), batch_size):
        batch_ids = company_ids[i:i + batch_size]
        logger.info(f"Processing batch {i // batch_size + 1} of companies: {len(batch_ids)} companies")

        # Get the last 200 days of volume for each company
        volume_data = db.query(PriceHistory).filter(
            PriceHistory.company_id.in_(batch_ids)
        ).order_by(PriceHistory.timestamp.desc()).limit(200 * len(batch_ids)).all()

        if not volume_data:
            logger.warning(f"No data found for companies: {batch_ids}")
            continue

        volume_df = pd.DataFrame([
            {
                'company_id': v.company_id,
                'volume': v.volume,
            } for v in volume_data
        ])

        # Calculate the average daily volume for each company
        avg_volume_df = volume_df.groupby('company_id')['volume'].mean().reset_index()

        # Prepare data for bulk update
        for index, row in avg_volume_df.iterrows():
            update_data.append({
                "id": int(row['company_id']),
                "averagevolume": int(row['volume'])  # Store as integer
            })

    # Update Company table with bulk update
    if update_data:
        db.execute(update(Company), update_data)
        db.commit()

def calculate_shares_outstanding(db: Session, company_ids: list[int], batch_size=200):
    """Calculates the latest Shares Outstanding for each company and updates the Company table.

    Args:
        db: The database session.
        company_ids: A list of company IDs to process.
        batch_size: The number of companies to process in each batch.
    """
    logger.info("running calculate_shares_outstanding...")
    update_data = []
    for i in range(0, len(company_ids), batch_size):
        batch_ids = company_ids[i:i + batch_size]
        logger.info(f"Processing batch {i // batch_size + 1} of companies: {len(batch_ids)} companies")

        # Efficiently get shares outstanding using a subquery and JOIN
        subquery = (
            db.query(Financials.company_id, func.max(Financials.report_date).label("max_report_date"))
            .filter(
                Financials.company_id.in_(batch_ids),
                Financials.type == "annual_income_statement",
                Financials.index.in_(index_name_mapping["Shares (Diluted)"]),
            )
            .group_by(Financials.company_id)
            .subquery()
        )

        shares_outstanding_data = (
            db.query(Financials.company_id, Financials.value.label("Shares Outstanding"), Financials.report_date) # Get the shares outstanding and report_date
            .join(subquery, (Financials.company_id == subquery.c.company_id) & (Financials.report_date == subquery.c.max_report_date))
            .filter(
                Financials.type == "annual_income_statement",
                Financials.index.in_(index_name_mapping["Shares (Diluted)"]),
            )
        )

        # Prepare data for bulk update
        for company_id, shares_outstanding, report_date in shares_outstanding_data:
            if shares_outstanding:  
                update_data.append({
                    "id": company_id,
                    "sharesoutstanding": int(shares_outstanding)  # Store as integer
                })

    # Update Company table with bulk update
    if update_data:
        db.execute(update(Company), update_data)
        db.commit()

def calculate_price_relative_to_52week_high(db: Session, company_ids: list[int], batch_size=200):
    """Calculates the Price Relative to 52-Week High (%) for each company and updates the Company table.

    Optimized to reduce redundant queries and improve efficiency.
    """
    logger.info("running calculate_price_relative_to_52week_high...")
    update_data = []
    for i in range(0, len(company_ids), batch_size):
        batch_ids = company_ids[i:i + batch_size]
        logger.info(f"Processing batch {i // batch_size + 1} of companies: {len(batch_ids)} companies")

        # --- Get the last price and 52-week high for all companies in one go ---
        subquery_last_price = (
            db.query(
                PriceHistory.company_id,
                func.max(PriceHistory.timestamp).label("max_date"),
            )
            .filter(PriceHistory.company_id.in_(batch_ids))
            .group_by(PriceHistory.company_id)
            .subquery()
        )

        last_prices = (
            db.query(PriceHistory.company_id, PriceHistory.adjclose, PriceHistory.volume, PriceHistory.timestamp)
            .join(
                subquery_last_price,
                (PriceHistory.company_id == subquery_last_price.c.company_id)
                & (PriceHistory.timestamp == subquery_last_price.c.max_date),
            )
            .subquery()
        )

        a_year_ago_date = (datetime.now() - timedelta(days=365)).date()
        fiftytwoweekhighs = (
            db.query(
                PriceHistory.company_id,
                func.max(PriceHistory.adjclose).label("fiftytwoweekhigh")
            )
            .filter(
                PriceHistory.company_id.in_(batch_ids),
                PriceHistory.timestamp >= a_year_ago_date
            )
            .group_by(PriceHistory.company_id)
            .subquery()
        )

        # --- Join the subqueries to get the required data ---
        combined_data = (
            db.query(
                last_prices.c.company_id,
                last_prices.c.adjclose,
                last_prices.c.volume,
                fiftytwoweekhighs.c.fiftytwoweekhigh
            )
            .join(
                fiftytwoweekhighs,
                last_prices.c.company_id == fiftytwoweekhighs.c.company_id
            )
            .all()
        )

        # --- Update the Company table ---
        for company_id, last_price, volume, fiftytwoweekhigh in combined_data:
            if fiftytwoweekhigh is not None and last_price is not None:
                price_relative_to_high = (last_price / fiftytwoweekhigh) * 100
                update_data.append({
                    "id": company_id,
                    "regularmarketpreviousclose": round(last_price, 2),
                    "regularmarketvolume": volume,
                    "price_relative_to_52week_high": price_relative_to_high,
                    "fiftytwoweekhigh": fiftytwoweekhigh
                })
            else:
                update_data.append({
                    "id": company_id,
                    "regularmarketpreviousclose": round(last_price, 2),
                    "regularmarketvolume": volume,
                    "fiftytwoweekhigh": fiftytwoweekhigh
                })
    if update_data:
        db.execute(update(Company), update_data)
        db.commit()

def calculate_pe_ratio_and_eps_trailing_twelve_months(db: Session, company_ids: list[int], batch_size=200):
    """Calculates Trailing P/E Ratio and EPS Trailing Twelve Months and updates the Company table.

    Args:
        db: The database session.
        company_ids: A list of company IDs to process.
        batch_size: The number of companies to process in each batch.
    """
    logger.info("running calculate_pe_ratio_and_eps_trailing_twelve_months...")
    update_data = []
    for i in range(0, len(company_ids), batch_size):
        batch_ids = company_ids[i:i + batch_size]
        logger.info(f"Processing batch {i // batch_size + 1} of companies: {len(batch_ids)} companies")

        # Get the last price for the current batch of companies
        companies_data = db.query(Company.id, Company.regularmarketpreviousclose).filter(Company.id.in_(batch_ids)).all()
        companies_df = pd.DataFrame([
            {
                "company_id": c[0],
                "regularmarketpreviousclose": c[1],
            }
            for c in companies_data
        ])

        # Get EPS for the last four quarters for the current batch of companies
        eps_q_values = db.query(Financials).filter(
            Financials.company_id.in_(batch_ids),
            Financials.type == "quarterly_income_statement",
            Financials.index.in_(
                index_name_mapping["Net Income"]
                + index_name_mapping["Shares (Diluted)"]
            ),
        ).all() 

        if not eps_q_values:
            logger.warning(f"No data found for companies: {batch_ids}")
            continue

        eps_q_df = pd.DataFrame([
            {
                "company_id": e.company_id,
                "report_date": e.report_date,
                "index": e.index,
                "value": e.value,
            }
            for e in eps_q_values
        ])

        eps_q_df = standardize_index_names(eps_q_df, index_name_mapping)
        eps_q_df = eps_q_df.pivot_table(
            index=["company_id", "report_date"],
            columns="index",
            values="value",
            fill_value=0,
        ).reset_index()

        if "Net Income" not in eps_q_df.columns or "Shares (Diluted)" not in eps_q_df.columns:
            logger.warning(f"Missing Net Income or Shares (Diluted) in companies: {batch_ids}")
            continue

        eps_q_df["EPS"] = eps_q_df["Net Income"] / eps_q_df["Shares (Diluted)"]
        eps_q_df.sort_values(by=["company_id", "report_date"], inplace=True)

        # Aggregate the 4 quarters EPS
        eps_q_df = eps_q_df.groupby("company_id").tail(4)
        eps_q_df = eps_q_df.groupby("company_id")["EPS"].sum().reset_index()

        # Merge the two dataframes
        merged_df = pd.merge(eps_q_df, companies_df, on="company_id", how="left")

        # Calculate PE
        merged_df["trailingpe"] = np.where(merged_df["EPS"] != 0, merged_df["regularmarketpreviousclose"] / merged_df["EPS"], None)

        # Prepare data for bulk update
        for index, row in merged_df.iterrows():
            update_data.append({
                "id": int(row["company_id"]),
                "trailingpe": float(row["trailingpe"]) if not pd.isna(row["trailingpe"]) else None,
                "epstrailingtwelvemonths": float(row["EPS"]) if not pd.isna(row["EPS"]) else None
            })

    # Update Company table with bulk update
    if update_data:
        db.execute(update(Company), update_data)
        db.commit()

def calculate_relative_strength_percentile(db: Session, company_ids: list[int], benchmark_symbol="SPY", batch_size=200):
    """Calculates the Relative Strength (Percentile) for each company and updates the Company table.

    Args:
        db: The database session.
        company_ids: A list of company IDs to process.
        benchmark_symbol: The ticker symbol of the benchmark index (default: SPY).
        batch_size: The number of companies to process in each batch.
    """
    logger.info("running calculate_relative_strength_percentile...")
    # Define time periods
    time_periods = {
        "relative_strength_percentile_252": 252,  # 1 Year
        "relative_strength_percentile_126": 126,  # 6 Months
        "relative_strength_percentile_63": 63,    # 3 Months
    }

    # Get the benchmark data for the maximum date range
    today = pd.Timestamp.today(tz="UTC")
    past_date = today - pd.Timedelta(days=max(time_periods.values()))

    benchmark_data = db.query(PriceHistory.adjclose, PriceHistory.timestamp).filter(
        PriceHistory.company_id == db.query(Company.id).filter(Company.symbol == benchmark_symbol).scalar_subquery(),
        PriceHistory.timestamp >= past_date,
        PriceHistory.timestamp <= today
    ).all()

    if not benchmark_data:
        logger.warning(f"No benchmark data found for {benchmark_symbol}.")
        return

    benchmark_df = pd.DataFrame([{"adjclose": p[0], "date": p[1]} for p in benchmark_data])
    benchmark_df['date'] = pd.to_datetime(benchmark_df['date'], utc=True)

    def get_price_change(price_df, days=252):
        """Calculates the percentage price change for a company in the last X days."""
        if price_df.empty:
            return np.nan
        df = price_df.copy()
        
        # Define the past date
        past_date_period = df['date'].max() - pd.Timedelta(days=days)

        # Filter rows for the date range
        df = df[df['date'] >= past_date_period]

        # if no data for this period, skip
        if df.empty:
            return np.nan

        df = df.sort_values(by='date')
        current_price = df['adjclose'].iloc[-1]
        past_price = df['adjclose'].iloc[0]

        if past_price == 0:
            return np.nan
        price_change = (current_price - past_price) / past_price
        return price_change * 100

    # Calculate price change for the benchmark in the different period
    benchmark_price_changes = {
        name: get_price_change(benchmark_df, days)
        for name, days in time_periods.items()
    }

    all_rs_values = {}
    # Process companies in batches
    for i in range(0, len(company_ids), batch_size):
        batch_ids = company_ids[i:i + batch_size]
        logger.info(f"Processing batch {i // batch_size + 1} of companies: {len(batch_ids)} companies")

        # Get all prices for the current batch of companies
        price_data = db.query(PriceHistory.company_id, PriceHistory.adjclose, PriceHistory.timestamp).filter(
            PriceHistory.company_id.in_(batch_ids),
            PriceHistory.timestamp >= past_date,
            PriceHistory.timestamp <= today
        ).all()

        if not price_data:
            continue

        price_df = pd.DataFrame([{"company_id": p[0], "adjclose": p[1], "date": p[2]} for p in price_data])
        #price_df['date'] = pd.to_datetime(price_df['date'], utc=True)

        for company_id in batch_ids:
            # Filter the data for the current company
            company_price_df = price_df[price_df["company_id"] == company_id].copy()
            if company_price_df.empty:
                continue  # Skip companies without any price data

            all_rs_values.setdefault(company_id, {})
            for name, days in time_periods.items():
                # Calculate the stock's price change over the past X days
                stock_price_change = get_price_change(company_price_df, days)
                if pd.isna(stock_price_change):
                    continue
                # Calculate the Relative Strength (RS) ratio
                rs_ratio = stock_price_change / benchmark_price_changes[name]
                # Add to list
                all_rs_values[company_id][name] = rs_ratio

    all_rs_df = pd.DataFrame.from_dict(all_rs_values, orient="index").reset_index().rename(columns={'index': 'company_id'})
    #print(f"{all_rs_df[all_rs_df["company_id"]==14333]=}")

    # Calculate percentiles across all companies
    for name in time_periods.keys():
        all_values = all_rs_df[name]
        if not all_values.empty:
            # Correctly handle NaN values during ranking
            percentile_ranks = all_values.rank(pct=True, na_option='keep') * 100
            # Invert rank if benchmark price change is negative
            if benchmark_price_changes[name] < 0:
                percentile_ranks = 100 - percentile_ranks  
            all_rs_df[f"{name}_ratio"] = all_rs_df[name] # save the ratio
            # Use .loc to avoid SettingWithCopyWarning and ensure correct assignment
            all_rs_df.loc[all_rs_df.index, name] = percentile_ranks.values
        else:
            all_rs_df[name] = None
            all_rs_df[f"{name}_ratio"] = None
    #print(f"{all_rs_df[all_rs_df["company_id"]==14333].values=}")

    # Prepare data for bulk update
    update_data = []
    for index, row in all_rs_df.iterrows():
        company_id = int(row["company_id"])
        company_update = {"id": company_id}
        for name in time_periods.keys():
            company_update[name] = float(row[name]) if not pd.isna(row[name]) else None
        # Add the rs ratio to the update
        company_update["rsratio"] = float(row[f"relative_strength_percentile_252_ratio"]) if not pd.isna(row[f"relative_strength_percentile_252_ratio"]) else None
        update_data.append(company_update)

    # Update Company table with bulk update
    if update_data:
        db.execute(update(Company), update_data)
        db.commit()

    return all_rs_df

def calculate_expanding_volume(db: Session, company_ids: list[int], batch_size=200):
    """
    Calculates expanding volume, 10-day average volume, and 3-month average volume,
    and updates the Company table.
    Expanding volume is true if the 10-day average is greater than the 3-month average.

    Args:
        db: The database session.
        company_ids: A list of company IDs to process.
        batch_size: The number of companies to process in each batch.
    """
    logger.info("running calculate_expanding_volume...")
    # Need at least 60 trading days for 3-month average. Fetching ~100 calendar days as a buffer.
    today = pd.Timestamp.today(tz="UTC")
    past_date = today - pd.Timedelta(days=100)

    def get_volume_metrics(df_group):
        """Calculates expanding volume and average volumes for a single ticker (group)."""
        if df_group.empty or len(df_group) < 60:  # Need at least 60 days for 3-month average
            return pd.Series({
                'expanding_volume': False,
                'averagedailyvolume10day': np.nan,
                'averagedailyvolume3month': np.nan
            })

        df_group = df_group.sort_values(by='date')

        # Calculate average volumes using the last available data point
        avg_vol_10d = df_group['volume'].rolling(window=10).mean().iloc[-1]
        avg_vol_3m = df_group['volume'].rolling(window=60).mean().iloc[-1] # Approx 3 months

        # Determine if volume is expanding
        is_expanding = bool(avg_vol_10d > avg_vol_3m)

        return pd.Series({
            'expanding_volume': is_expanding,
            'averagedailyvolume10day': avg_vol_10d,
            'averagedailyvolume3month': avg_vol_3m
        })

    update_data = []
    # Process companies in batches
    for i in range(0, len(company_ids), batch_size):
        batch_ids = company_ids[i:i + batch_size]
        logger.info(f"Processing volume metrics for batch {i // batch_size + 1} of companies: {len(batch_ids)} companies")

        # Get price and volume data for the current batch of companies
        volume_data = db.query(PriceHistory.company_id, PriceHistory.timestamp, PriceHistory.volume).filter(
            PriceHistory.company_id.in_(batch_ids),
            PriceHistory.timestamp >= past_date
        ).all()

        if not volume_data:
            logger.warning(f"No volume data found for companies in batch: {batch_ids}")
            continue

        # Create dataframe
        volume_df = pd.DataFrame(volume_data, columns=["company_id", "date", "volume"])
        volume_df['date'] = pd.to_datetime(volume_df['date'], utc=True)

        # Group by company_id and calculate metrics
        metrics_df = volume_df.groupby('company_id').apply(get_volume_metrics)

        # Prepare data for bulk update
        for company_id, row in metrics_df.iterrows():
            if not pd.isna(row['averagedailyvolume10day']): # Only update if calculations were successful
                update_data.append({
                    "id": company_id,
                    "expanding_volume": row['expanding_volume'],
                    "averagedailyvolume10day": int(row['averagedailyvolume10day']),
                    "averagedailyvolume3month": int(row['averagedailyvolume3month'])
                })

    # Update Company table with bulk update
    if update_data:
        db.execute(update(Company), update_data)
        db.commit()
        logger.info(f"Successfully updated volume metrics for {len(update_data)} companies.")

def calculate_and_save_other_ratios(db: Session, company_ids: list[int], batch_size=200):
    """
    Calculates other financial ratios, including net income growth, and saves them to the Company table.

    Args:
        db: The database session.
        company_ids: A list of company IDs to process.
        batch_size: The number of companies to process in each batch.
    """
    logger.info("running calculate_and_save_other_ratios...")
    for i in range(0, len(company_ids), batch_size):
        batch_ids = company_ids[i:i + batch_size]
        logger.info(f"Processing batch {i // batch_size + 1} of companies: {len(batch_ids)} companies")

        # Get the required data for the current batch of companies
        financial_data = db.query(Company.id, Company.sharesoutstanding, Company.totalcash, Company.totaldebt, Company.payoutratio).filter(Company.id.in_(batch_ids)).all()
        financial_df = pd.DataFrame(financial_data, columns=['company_id', 'sharesoutstanding', 'totalcash', 'totaldebt', 'payoutratio'])

        # Get the last Net Income, Total Equity, Total Debt, Total Assets, EBIT, Interest Expense, Operating Cash Flow, Total Revenue, Inventory, Cost Of Revenue, Accounts Payable, Receivables for each company
        income_values = db.query(Financials).filter(
            Financials.company_id.in_(batch_ids),
            Financials.type == "annual_income_statement",
            Financials.index.in_(
                index_name_mapping["Net Income"]
                + index_name_mapping["Operating Income"]
                + index_name_mapping["Total Revenue"]
                + index_name_mapping["EBIT"]
                + index_name_mapping["Interest Expense"]
                + index_name_mapping["Cost Of Revenue"]
            ),
        ).all()

        balance_values = db.query(Financials).filter(
            Financials.company_id.in_(batch_ids),
            Financials.type == "annual_balance_sheet",
            Financials.index.in_(
                index_name_mapping["Total Equity"]
                + index_name_mapping["Total Liabilities"]
                + index_name_mapping["Total Assets"]
                + index_name_mapping["Inventory"]
                + index_name_mapping["Payables"]
                + index_name_mapping["Receivables"]
                + index_name_mapping["Current Assets"]
                + index_name_mapping["Current Liabilities"]
                + index_name_mapping["Cash And Equivalent"]
            ),
        ).all()

        cash_flow_values = db.query(Financials).filter(
            Financials.company_id.in_(batch_ids),
            Financials.type == "annual_cash_flow",
            Financials.index.in_(
                index_name_mapping["Operating Cash Flow"]
                + index_name_mapping["Dividends Paid"]
            ),
        ).all()

        # Create DataFrames
        income_df = pd.DataFrame([
            {
                'company_id': i.company_id,
                'report_date': i.report_date,
                'index': i.index,
                'value': i.value,
            } for i in income_values
        ])

        balance_df = pd.DataFrame([
            {
                'company_id': b.company_id,
                'report_date': b.report_date,
                'index': b.index,
                'value': b.value,
            } for b in balance_values
        ])

        cash_flow_df = pd.DataFrame([
            {
                'company_id': c.company_id,
                'report_date': c.report_date,
                'index': c.index,
                'value': c.value,
            } for c in cash_flow_values
        ])

        # Early exit if no data
        if income_df.empty or balance_df.empty or cash_flow_df.empty:
            logger.warning(f"No data found for companies: {batch_ids}")
            continue

        # Standardize the index
        income_df = standardize_index_names(income_df, index_name_mapping)
        balance_df = standardize_index_names(balance_df, index_name_mapping)
        cash_flow_df = standardize_index_names(cash_flow_df, index_name_mapping)

        # Pivot the DataFrames
        income_df = income_df.pivot_table(
            index=["company_id", "report_date"],
            columns="index",
            values="value",
            fill_value=0,
        ).reset_index()

        balance_df = balance_df.pivot_table(
            index=["company_id", "report_date"],
            columns="index",
            values="value",
            fill_value=0,
        ).reset_index()

        cash_flow_df = cash_flow_df.pivot_table(
            index=["company_id", "report_date"],
            columns="index",
            values="value",
            fill_value=0,
        ).reset_index()

        # Merge dataframes
        merged_df = pd.merge(income_df, balance_df, on=['company_id', 'report_date'], how='inner')
        merged_df = pd.merge(merged_df, cash_flow_df, on=['company_id', 'report_date'], how='inner')
        merged_df = pd.merge(merged_df, financial_df, on=['company_id'], how='inner')

        # Calculate the ratios
        update_data = []
        for company_id in batch_ids:
            company_data = merged_df[merged_df["company_id"] == company_id].copy()
            if company_data.empty:
                continue

            # Find the most recent fiscal year
            company_data.sort_values(by='report_date', ascending=True, inplace=True)
            last_data = company_data.iloc[-1]

            # --- Calculate the ratios ---
            # Profitability
            net_income = last_data.get('Net Income', 0)
            total_revenue = last_data.get('Total Revenue', 0)
            ebit = last_data.get('EBIT', 0)
            operating_income = last_data.get('Operating Income', 0)
            cost_of_revenue = last_data.get('Cost Of Revenue', 0)

            # Solvency
            total_equity = last_data.get('Total Equity', 0)
            total_debt = last_data.get('Total Liabilities', 0)
            total_assets = last_data.get('Total Assets', 0)
            interest_expense = last_data.get('Interest Expense', 0)
            operating_cf = last_data.get('Operating Cash Flow', 0)

            # Efficiency
            inventory = last_data.get('Inventory', 0)
            accounts_payable = last_data.get('Payables', 0)
            accounts_receivable = last_data.get('Receivables', 0)

            # Liquidity
            shares_outstanding = last_data.get('sharesoutstanding', 0)
            current_assets = last_data.get('Current Assets', 0)
            current_liabilities = last_data.get('Current Liabilities', 0)
            cash_and_equivalent = last_data.get('Cash And Equivalent', 0)
            total_cash = last_data.get('totalcash', 0)
            total_debt_from_company = last_data.get('totaldebt', 0)

            # --- Calculate the ratios ---
            debt_to_assets_ratio = float('inf') if total_assets == 0 else total_debt / total_assets
            dividend_payout_ratio = last_data.get('payoutratio', 0)  #float('inf') if net_income == 0 else abs(last_data.get('Dividends Paid', 0)) / net_income
            interest_coverage_ratio = float('inf') if interest_expense == 0 else ebit / interest_expense
            cash_flow_to_debt_ratio = float('inf') if total_debt == 0 else operating_cf / total_debt
            asset_turnover = float('inf') if total_assets == 0 else total_revenue / total_assets
            inventory_turnover_ratio = float('inf') if inventory == 0 else cost_of_revenue / inventory
            days_payable_outstanding = float('inf') if cost_of_revenue == 0 else (accounts_payable / cost_of_revenue) * 365
            days_sales_outstanding = float('inf') if total_revenue == 0 else (accounts_receivable / total_revenue) * 365
            cash_ratio = float('inf') if current_liabilities == 0 else cash_and_equivalent / current_liabilities
            debt_to_cash_ratio = float('inf') if total_cash == 0 else total_debt_from_company / total_cash

            # --- Calculate Net Income Growth ---
            # Get the previous year's net income
            previous_year_data = income_df[income_df['company_id'] == company_id].copy()
            previous_year_data.sort_values(by='report_date', ascending=False, inplace=True)
            previous_year_data = previous_year_data.iloc[1] if len(previous_year_data) > 1 else None
            previous_net_income = previous_year_data.get('Net Income', 0) if previous_year_data is not None else 0

            # Calculate net income growth
            net_income_growth = float('inf') if previous_net_income == 0 else (net_income - previous_net_income) / abs(previous_net_income)

            # The following values are already available from yfinance, no need to recalculate
            # gross_profit_margin = (total_revenue - cost_of_revenue) / total_revenue if total_revenue != 0 else None
            # operating_margin = operating_income / total_revenue if total_revenue != 0 else None
            # profit_margin = net_income / total_revenue if total_revenue != 0 else None
            # roe = net_income / total_equity if total_equity != 0 else None
            # roa = net_income / total_assets if total_assets != 0 else None
            # eps = net_income / shares_outstanding if shares_outstanding != 0 else None
            # current_ratio = current_assets / current_liabilities if current_liabilities != 0 else None
            # quick_ratio = (current_assets - inventory) / current_liabilities if current_liabilities != 0 else None
            # debt_to_equity_ratio = total_debt / total_equity if total_equity != 0 else None

            # Prepare data for bulk update
            update_data.append({
                "id": company_id,
                "netincomegrowth": float(net_income_growth) if not pd.isna(net_income_growth) else None, # Add net income growth
                "debttoassetsratio": float(debt_to_assets_ratio) if not pd.isna(debt_to_assets_ratio) else None,
                "interestcoverageratio": float(interest_coverage_ratio) if not pd.isna(interest_coverage_ratio) else None,
                "cashflowtodebtratio": float(cash_flow_to_debt_ratio) if not pd.isna(cash_flow_to_debt_ratio) else None,
                "assetturnover": float(asset_turnover) if not pd.isna(asset_turnover) else None,
                "inventoryturnoverratio": float(inventory_turnover_ratio) if not pd.isna(inventory_turnover_ratio) else None,
                "dayspayableoutstanding": float(days_payable_outstanding) if not pd.isna(days_payable_outstanding) else None,
                "dayssalesoutstanding": float(days_sales_outstanding) if not pd.isna(days_sales_outstanding) else None,
                "dividendpayoutratio": float(dividend_payout_ratio) if not pd.isna(dividend_payout_ratio) else None,
                "cashratio": float(cash_ratio) if not pd.isna(cash_ratio) else None,
                "debttocash": float(debt_to_cash_ratio) if not pd.isna(debt_to_cash_ratio) else None,
                "last_other_ratios_report_date": last_data['report_date']
            })

        # Update Company table with bulk update
        if update_data:
            db.execute(update(Company), update_data)
            db.commit()

def calculate_quarterly_trends(db: Session, company_ids: list[int], batch_size=200):
    """
    Calculates various quarterly trend metrics (consecutive growth, acceleration)
    for Revenue, EPS, Operating Margin, FCF, and Share Count. Updates the Company table.
    """
    logger.info("Running calculate_quarterly_trends...")
    all_update_data = {} # Use a dictionary keyed by company_id for easier merging

    for i in range(0, len(company_ids), batch_size):
        batch_ids = company_ids[i:i + batch_size]
        logger.info(f"Processing quarterly trends batch {i // batch_size + 1} for {len(batch_ids)} companies")

        # --- Fetch Necessary Quarterly Data ---
        # Adjust index_names based on what's needed for Revenue, EPS, OpMargin, FCF, Shares
        required_indices = (
            index_name_mapping["Total Revenue"]
            + index_name_mapping["Net Income"]
            + index_name_mapping["Operating Income"] # For Operating Margin
            + index_name_mapping["Shares (Diluted)"]
            + index_name_mapping["Operating Cash Flow"] # For FCF
            + index_name_mapping["Capital Expenditures"] # For FCF
        )
        # Fetch more history if needed for acceleration (e.g., 9 quarters for YoY diff)
        # Fetch enough history for TTM FCF (e.g., 8 quarters)
        limit_per_company = 10 # Fetch last 10 quarters to be safe

        financial_data = db.query(Financials).filter(
            Financials.company_id.in_(batch_ids),
            Financials.type == "quarterly_income_statement",
            Financials.index.in_(required_indices),
        ).order_by(Financials.company_id, Financials.report_date.desc()).all()

        cash_flow_data = db.query(Financials).filter(
            Financials.company_id.in_(batch_ids),
            Financials.type == "quarterly_cash_flow",
            Financials.index.in_(index_name_mapping["Operating Cash Flow"] + index_name_mapping["Capital Expenditures"]),
        ).order_by(Financials.company_id, Financials.report_date.desc()).all()

        if not financial_data:
            logger.warning(f"No quarterly income data found for batch {i // batch_size + 1}")
            # Continue to next batch, but maybe FCF data exists
            # continue # Skip if no income data

        # --- Prepare DataFrame ---
        df_income = pd.DataFrame([object_as_dict(f) for f in financial_data])
        df_cf = pd.DataFrame([object_as_dict(f) for f in cash_flow_data])

        if df_income.empty and df_cf.empty:
             logger.warning(f"No quarterly income or CF data found for batch {i // batch_size + 1}")
             continue

        # Standardize and Pivot Income Data
        if not df_income.empty:
            df_income = standardize_index_names(df_income, index_name_mapping)
            df_income = df_income.pivot_table(
                index=["company_id", "report_date"], columns="index", values="value", fill_value=np.nan
            ).reset_index()
            df_income["report_date"] = pd.to_datetime(df_income["report_date"])
            df_income.sort_values(by=["company_id", "report_date"], inplace=True)

            # --- Fill missing "Shares (Diluted)" ---
            if "Shares (Diluted)" in df_income.columns:
                df_income["Shares (Diluted)"] = df_income.groupby("company_id")["Shares (Diluted)"].ffill().bfill()
            else:
                 logger.warning(f"Missing 'Shares (Diluted)' column in income data for batch {i // batch_size + 1}")
                 df_income["Shares (Diluted)"] = np.nan # Add column if missing

            # Calculate EPS and Operating Margin
            if "Net Income" in df_income.columns and "Shares (Diluted)" in df_income.columns:
                 df_income["EPS"] = df_income["Net Income"] / df_income["Shares (Diluted)"]
            else:
                 logger.warning(f"Missing 'Net Income' or 'Shares (Diluted)' for EPS calculation in batch {i // batch_size + 1}")
                 df_income["EPS"] = np.nan

            if "Operating Income" in df_income.columns and "Total Revenue" in df_income.columns:
                 df_income["Operating Margin"] = np.where(
                     df_income["Total Revenue"] == 0, np.nan, df_income["Operating Income"] / df_income["Total Revenue"]
                 )
            else:
                 logger.warning(f"Missing 'Operating Income' or 'Total Revenue' for OpMargin calculation in batch {i // batch_size + 1}")
                 df_income["Operating Margin"] = np.nan

        # Standardize and Pivot Cash Flow Data
        if not df_cf.empty:
            df_cf = standardize_index_names(df_cf, index_name_mapping)
            df_cf = df_cf.pivot_table(
                index=["company_id", "report_date"], columns="index", values="value", fill_value=np.nan
            ).reset_index()
            df_cf["report_date"] = pd.to_datetime(df_cf["report_date"])
            df_cf.sort_values(by=["company_id", "report_date"], inplace=True)

            # Calculate FCF
            if "Operating Cash Flow" in df_cf.columns and "Capital Expenditures" in df_cf.columns:
                 # Capital Expenditures are often negative, ensure correct calculation
                 df_cf["FCF"] = df_cf["Operating Cash Flow"] + df_cf["Capital Expenditures"].fillna(0) # Add because CapEx is usually negative
            else:
                 logger.warning(f"Missing 'Operating Cash Flow' or 'Capital Expenditures' for FCF calculation in batch {i // batch_size + 1}")
                 df_cf["FCF"] = np.nan

        # --- Merge DataFrames ---
        if not df_income.empty and not df_cf.empty:
            df = pd.merge(df_income, df_cf, on=["company_id", "report_date"], how="outer")
        elif not df_income.empty:
            df = df_income
        elif not df_cf.empty:
            df = df_cf
        else:
            continue # Should not happen based on earlier check

        df.sort_values(by=["company_id", "report_date"], inplace=True)

        # --- Calculate QoQ Changes ---
        metrics_qoq = ["Total Revenue", "EPS", "Operating Margin", "FCF", "Shares (Diluted)"]
        for metric in metrics_qoq:
            if metric in df.columns:
                # Use transform instead of apply for index alignment
                # Add fill_method=None to address the FutureWarning
                df[f'{metric}_qoq_change'] = df.groupby('company_id')[metric].transform(
                    lambda x: x.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
                )
            else:
                 df[f'{metric}_qoq_change'] = np.nan

        # --- Calculate YOY Changes (for Acceleration) ---
        metrics_yoy = ["Total Revenue", "EPS", "Operating Margin", "FCF"] # Shares YOY doesn't make sense for acceleration
        for metric in metrics_yoy:
            if metric in df.columns:
                df[f'{metric}_yoy_change'] = df.groupby('company_id')[metric].transform(
                     lambda x: x.pct_change(periods=4, fill_method=None).replace([np.inf, -np.inf], np.nan)
                )
            else:
                 df[f'{metric}_yoy_change'] = np.nan

        # --- Calculate TTM FCF ---
        #print(f"{df.columns=}, {df['FCF']=}")
        if "FCF" in df.columns:
            df['FCF_ttm'] = df.groupby('company_id')['FCF'].transform(lambda x: x.rolling(window=4, min_periods=4).sum())
            df['FCF_ttm_yoy_change'] = df.groupby('company_id')['FCF_ttm'].transform(
                lambda x: x.pct_change(periods=4, fill_method=None).replace([np.inf, -np.inf], np.nan)
            )
            #print(f"{df['FCF_ttm']=}, {df['FCF_ttm_yoy_change']=}")
        else:
            df['FCF_ttm_yoy_change'] = np.nan

        # --- Calculate Trends per Company ---
        for company_id, group in df.groupby('company_id'):
            if company_id not in all_update_data:
                 all_update_data[company_id] = {"id": company_id} # Initialize dict for the company

            last_row = group.iloc[-1] if not group.empty else None
            prev_row = group.iloc[-2] if len(group) > 1 else None

            # Consecutive Quarters Growth/Improvement
            consecutive_rev_growth = 0
            for change in group['Total Revenue_qoq_change'].iloc[::-1]:
                if pd.notna(change) and change > 0: consecutive_rev_growth += 1
                else: break
            all_update_data[company_id]["consecutive_quarters_revenue_growth"] = consecutive_rev_growth

            consecutive_eps_growth = 0
            for change in group['EPS_qoq_change'].iloc[::-1]:
                if pd.notna(change) and change > 0: consecutive_eps_growth += 1
                else: break
            all_update_data[company_id]["consecutive_quarters_eps_growth"] = consecutive_eps_growth

            consecutive_opmargin_improvement = 0
            # For margins, improvement means increase (positive change)
            for change in group['Operating Margin_qoq_change'].iloc[::-1]:
                if pd.notna(change) and change > 0: consecutive_opmargin_improvement += 1
                else: break
            all_update_data[company_id]["consecutive_quarters_opmargin_improvement"] = consecutive_opmargin_improvement

            # Growth Acceleration (Latest YOY change - Previous Quarter's YOY change)
            if last_row is not None and prev_row is not None:
                 all_update_data[company_id]["revenue_growth_acceleration_qoq_yoy"] = float(last_row.get('Total Revenue_yoy_change', np.nan) - prev_row.get('Total Revenue_yoy_change', np.nan))
                 all_update_data[company_id]["eps_growth_acceleration_qoq_yoy"] = float(last_row.get('EPS_yoy_change', np.nan) - prev_row.get('EPS_yoy_change', np.nan))
                 all_update_data[company_id]["opmargin_improvement_acceleration_qoq_yoy"] = float(last_row.get('Operating Margin_yoy_change', np.nan) - prev_row.get('Operating Margin_yoy_change', np.nan))
            else:
                 all_update_data[company_id]["revenue_growth_acceleration_qoq_yoy"] = None
                 all_update_data[company_id]["eps_growth_acceleration_qoq_yoy"] = None
                 all_update_data[company_id]["opmargin_improvement_acceleration_qoq_yoy"] = None

            # FCF Growth
            all_update_data[company_id]["fcf_growth_qoq"] = float(last_row.get('FCF_qoq_change', np.nan)) if last_row is not None else None
            all_update_data[company_id]["fcf_growth_ttm_yoy"] = float(last_row.get('FCF_ttm_yoy_change', np.nan)) if last_row is not None else None

            # Share Buybacks (Reduction Rate & Consecutive Quarters)
            share_change_rate = float(last_row.get('Shares (Diluted)_qoq_change', np.nan)) if last_row is not None else None
            # Ensure it's negative for reduction, store as is (negative means reduction)
            all_update_data[company_id]["share_change_rate_qoq"] = share_change_rate

            consecutive_share_reduction = 0
            for change in group['Shares (Diluted)_qoq_change'].iloc[::-1]:
                # Count if change is negative (shares decreased)
                if pd.notna(change) and change < 0: consecutive_share_reduction += 1
                else: break
            all_update_data[company_id]["consecutive_quarters_share_reduction"] = consecutive_share_reduction

            # Clean up NaN values before update
            for key, value in all_update_data[company_id].items():
                if pd.isna(value):
                    all_update_data[company_id][key] = None

    # --- Perform Bulk Update ---
    final_update_list = list(all_update_data.values())
    if final_update_list:
        try:
            # print(f"Sample update data: {final_update_list[0]}") # Debug print
            db.execute(update(Company), final_update_list)
            db.commit()
            logger.info(f"Successfully updated quarterly trends for {len(final_update_list)} companies.")
        except Exception as e:
            db.rollback()
            logger.error(f"Error during bulk update for quarterly trends: {e}")
            traceback.print_exc()


def find_relative_strength_percentile(db: Session, company_id: int, benchmark_symbol: str = "SPY") -> dict:
    """
    Calculates the Relative Strength Percentile for a single company by comparing its
    RS Ratio against all other companies in its market. This is a more statistically
    sound approach than borrowing percentiles from peers.
    """
    results = {}
    company = db.query(Company).filter(Company.id == company_id).first()
    if not company:
        logger.warning(f"Company with id {company_id} not found.")
        return results

    if company.relative_strength_percentile_252 is not None:
        results["Relative Strength Percentile (1 year)"] = company.relative_strength_percentile_252
        results["Relative Strength Percentile (6 months)"] = company.relative_strength_percentile_126
        results["Relative Strength Percentile (3 months)"] = company.relative_strength_percentile_63
        return results

    # 1. Ensure the company has an RS Ratio. If not, it must be calculated first.
    if company.rsratio is None:
        logger.warning(f"RS Ratio for {company.symbol} is not pre-calculated. Run `calculate_relative_strength_percentile` in batch mode first.")
        return {}

    # 2. Get the distribution of RS Ratios for the entire market to compare against.
    market_companies = db.query(Company.rsratio).filter(
        Company.market == company.market,
        Company.rsratio.isnot(None)
    ).all()
    
    if not market_companies:
        logger.warning(f"No companies with RS Ratio found in market '{company.market}' to create a distribution.")
        return {}

    market_rs_ratios = [r[0] for r in market_companies]

    # 3. Calculate the percentile of the company's score within the market distribution.
    # 'kind=rank' handles ties by assigning the average rank.
    percentile = stats.percentileofscore(market_rs_ratios, company.rsratio, kind='rank')

    # 4. Update the company record with the calculated percentile.
    # Note: This only calculates the main (1-year) percentile. The batch function calculates all periods.
    update_values = {"relative_strength_percentile_252": percentile}
    try:
        db.query(Company).filter(Company.id == company_id).update(update_values)
        db.commit()
        logger.info(f"Updated RS Percentile for {company.symbol} to {percentile:.2f}")
        results["Relative Strength Percentile (1 year)"] = percentile
        return results
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to update RS percentile for company_id {company_id}: {e}", exc_info=True)
        return {}


def find_strongest_stocks_in_strongest_industries(market="us", top_n_industries=5, top_n_stocks_per_industry=5):
    """
    Finds the strongest stocks within the strongest industries based on Relative Strength (RS) data.

    Args:
        market: The market to scan.
        top_n_industries: The number of top industries to consider.
        top_n_stocks_per_industry: The number of top stocks to select from each industry.

    Returns:
        A list of dictionaries, where each dictionary contains an industry and its mean RS, and a list of the top stocks with symbol, name and RS data.
    """
    # --- 1. Get all companies in a market ---
    db = next(get_db())
    companies = db.query(Company).filter(Company.isactive == True, Company.exchange.in_(db.query(Exchange.exchange_code).filter(Exchange.country_code == market))).all()
    company_ids = [c.id for c in companies]
    logger.info(f"Total companies to scan: {len(company_ids)}")
    if not company_ids:
        return []

    # --- 2. Get Company Industry Data ---
    company_industry_data = db.query(Company.id, Company.symbol, Company.longname, Company.industry, Company.rsratio).filter(
        Company.id.in_(company_ids), Company.industry != None, Company.rsratio != None, 
        Company.marketcap > 10000000, Company.averagevolume > 100000, 
    ).all()
    company_industry_df = pd.DataFrame(company_industry_data, columns=['company_id', 'symbol', 'longname', 'industry', 'rsratio'])
    industry_counts = company_industry_df['industry'].value_counts()
    valid_industries = industry_counts[industry_counts >= 30].index.tolist()
    company_industry_df = company_industry_df[company_industry_df['industry'].isin(valid_industries)]

    # --- 3. Calculate Average RS per Industry ---
    # Calculate the average RS for each industry
    industry_rs = company_industry_df.groupby('industry')['rsratio'].mean()
    # Sort by average RS
    industry_rs = industry_rs.sort_values(ascending=False)

    # --- 4. Select Top Industries ---
    top_industries = industry_rs.head(top_n_industries).index.tolist()

    # --- 5. Filter for Top Industries ---
    top_industry_companies = company_industry_df[company_industry_df['industry'].isin(top_industries)]

    # --- 6. Select Top Stocks per Industry ---
    strongest_industries = []
    strongest_stocks = []
    for industry in top_industries:
        industry_df = top_industry_companies[top_industry_companies['industry'] == industry].copy()
        # Sort by rsratio
        industry_df = industry_df.sort_values('rsratio', ascending=False)
        # Select the top N stocks
        top_stocks = industry_df.head(top_n_stocks_per_industry)

        # Get the industry info
        industry_rs = industry_df['rsratio'].mean()
        industry_info = {
            "industry": industry,
            "rsratio_mean": float(industry_rs) if not pd.isna(industry_rs) else None,
        }

        # Append the top stocks to the result list
        for index, row in top_stocks.iterrows():
            # find the company in company_industry_df
            company_row = company_industry_df[company_industry_df['company_id'] == row['company_id']]
            strongest_stocks.append({
                "symbol": company_row['symbol'].values[0],
                "longname": company_row['longname'].values[0],
                "rsratio": float(company_row['rsratio'].values[0]) if not pd.isna(company_row['rsratio'].values[0]) else None,
            })

        industry_info["strongest_stocks"] = strongest_stocks
        strongest_industries.append(industry_info)
        strongest_stocks = []

    return strongest_industries


class FundamentalScoreCalculator:
    """
    Calculates a proprietary fundamental score for companies based on a weighted average
    of various financial metrics.

    The tool uses a default set of weights and also supports sector-specific weights
    to tailor the scoring to different industry characteristics. The final score is
    also ranked to produce a percentile score across the market and within sectors.
    """
    def __init__(self):
        # Define default weights and allow for sector-specific overrides
        self.default_weights = {
            "Revenue Growth": 0.2,
            "EPS Growth": 0.2,
            "3-Year EPS CAGR": 0.15,
            "ROE": 0.15,
            "Profit Margin": 0.1,
            "Gross Profit Margin": 0.1,
            "Debt-to-Equity Ratio": -0.05,
            "Current Ratio": 0.05,
            "ROA": 0.05,  # Return on Assets
            "Free Cash Flow": 0.05,
            "Debt-to-Assets Ratio": -0.05,
            "Inventory Turnover": 0.02,
            "Days Sales Outstanding (DSO)": -0.02,
            "Days Payable Outstanding (DPO)": 0.02,
            "Dividend Yield": 0.05, # Add Dividend Yield
        }

        self.sector_keys = ["financial-services", "healthcare", "utilities", "technology", "consumer-cyclical", "industrials", "consumer-defensive", "basic-materials", "energy", "real-estate", "communication-services"]
        self.sector_weights = {
            "technology": {
                "Revenue Growth": 0.3,
                "EPS Growth": 0.25,
                "3-Year EPS CAGR": 0.15,
                "ROE": 0.1,
                "Profit Margin": 0.05,
                "Gross Profit Margin": 0.05,
                "Debt-to-Equity Ratio": -0.05,
                "Current Ratio": 0.1,
                "ROA": 0.05,
                "Free Cash Flow": 0.05,
                "Debt-to-Assets Ratio": -0.05,
                "Inventory Turnover": 0.02,
                "Days Sales Outstanding (DSO)": -0.02,
                "Days Payable Outstanding (DPO)": 0.02,
                "Dividend Yield": 0.05,
            },
            "financial-services": {
                "Revenue Growth": 0.1,
                "EPS Growth": 0.15,
                "3-Year EPS CAGR": 0.1,
                "ROE": 0.25,
                "Profit Margin": 0.1,
                "Gross Profit Margin": 0.1,
                "Debt-to-Equity Ratio": -0.1,
                "Current Ratio": 0.1,
                "ROA": 0.1,
                "Free Cash Flow": 0.1,
                "Debt-to-Assets Ratio": -0.1,
                "Inventory Turnover": 0.05,
                "Days Sales Outstanding (DSO)": -0.05,
                "Days Payable Outstanding (DPO)": 0.05,
                "Dividend Yield": 0.05,
            },
            "utilities": {
                "Revenue Growth": 0.1,
                "EPS Growth": 0.1,
                "3-Year EPS CAGR": 0.05,
                "ROE": 0.15,
                "Profit Margin": 0.15,
                "Gross Profit Margin": 0.15,
                "Debt-to-Equity Ratio": -0.1,
                "Current Ratio": 0.1,
                "ROA": 0.1,
                "Free Cash Flow": 0.2,
                "Debt-to-Assets Ratio": -0.1,
                "Inventory Turnover": 0.05,
                "Days Sales Outstanding (DSO)": -0.05,
                "Days Payable Outstanding (DPO)": 0.05,
                "Dividend Yield": 0.1,
            },
            "basic-materials": {
                "Revenue Growth": 0.15,
                "EPS Growth": 0.1,
                "3-Year EPS CAGR": 0.05,
                "ROE": 0.25,
                "Profit Margin": 0.2,
                "Gross Profit Margin": 0.15,
                "Debt-to-Equity Ratio": -0.1,
                "Current Ratio": 0.05,
                "ROA": 0.05,
                "Free Cash Flow": 0.1,
                "Debt-to-Assets Ratio": -0.1,
                "Inventory Turnover": 0.05,
                "Days Sales Outstanding (DSO)": -0.05,
                "Days Payable Outstanding (DPO)": 0.05,
                "Dividend Yield": 0.05,
            },
            "materials": self.default_weights,
            "industrials": self.default_weights,
            "consumer-discretionary": self.default_weights,
            "consumer-staples": self.default_weights,
            "health-care": self.default_weights,
            "financials": self.default_weights,
            "information-technology": self.default_weights,
            "communication-services": self.default_weights,
            "energy": self.default_weights,
            "real-estate": self.default_weights,
            "consumer-cyclical": self.default_weights,
            "consumer-defensive": self.default_weights,
            "healthcare": self.default_weights,
            "utilities": self.default_weights,
        }

    def calculate_fundamental_score_and_percentile(self, db: Session, company_ids: list[int]):
        """
        Calculates a weighted score and percentile for companies based on provided weights, using vectorized operations.  Includes more metrics.

        Args:
            db: The database session.
            company_ids: A list of company IDs to process.
        """
        logger.info("running calculate_fundamental_score_and_percentile...")
        try:
            # Efficiently retrieve data from the database using a single query - add more columns as needed
            company_data = db.query(
                Company.id,
                Company.sectorkey, 
                Company.revenuegrowth_quarterly_yoy,
                Company.earningsgrowth_quarterly_yoy,
                Company.eps_cagr_3year,
                Company.returnonequity,
                Company.debttoequity,
                Company.currentratio,
                Company.quickratio,
                Company.grossmargins,
                Company.operatingmargins,
                Company.profitmargins,
                Company.freecashflow,
                Company.debttoassetsratio,
                Company.inventoryturnoverratio,
                Company.dayssalesoutstanding,
                Company.dayspayableoutstanding,
                Company.dividendpayoutratio,
                Company.assetturnover,
                Company.dividendyield,
                Company.returnonassets,
            ).filter(Company.id.in_(company_ids), Company.sectorkey != None).all()

            all_company_data = pd.DataFrame(company_data)

            # Handle cases where all_company_data is empty
            if all_company_data.empty:
                logger.warning(f"No data found for company_ids: {company_ids}")
                return

            #Rename columns to match weights keys
            all_company_data = all_company_data.rename(columns={
                "revenuegrowth_quarterly_yoy": "Revenue Growth",
                "earningsgrowth_quarterly_yoy": "EPS Growth",
                "eps_cagr_3year": "3-Year EPS CAGR",
                "returnonequity": "ROE",
                "debttoequity": "Debt-to-Equity Ratio",
                "currentratio": "Current Ratio",
                "quickratio": "Quick Ratio",
                "grossmargins": "Gross Profit Margin",
                "operatingmargins": "Operating Margin",
                "profitmargins": "Profit Margin",
                "freecashflow": "Free Cash Flow",
                "debttoassetsratio": "Debt-to-Assets Ratio",
                "inventoryturnoverratio": "Inventory Turnover",
                "dayssalesoutstanding": "Days Sales Outstanding (DSO)",
                "dayspayableoutstanding": "Days Payable Outstanding (DPO)",
                "dividendpayoutratio": "Dividend Payout Ratio",
                "assetturnover": "Asset Turnover",
                "dividendyield": "Dividend Yield",
                "returnonassets": "ROA",
            })

            #Check that all required columns are present
            required_columns = list(self.default_weights.keys())
            missing_columns = set(required_columns) - set(all_company_data.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns for fundamental score calculation: {missing_columns}")

            # Vectorized score calculation using the full formula
            #all_company_data = all_company_data.fillna(0) # Fill NaN values with 0 before calculation

            # Imputation using a statistical method: Replaces NaNs with the median of the column
            # Select only numeric columns before imputation
            numeric_cols = all_company_data.select_dtypes(include=np.number).columns
            numeric_data = all_company_data[numeric_cols].copy() # Make a copy explicitly
            numeric_data.replace([np.inf, -np.inf], np.nan, inplace=True)

            # Apply imputation to the numeric data only
            imputer = SimpleImputer(strategy="median")
            imputed_data = imputer.fit_transform(numeric_data)

            # Reconstruct the DataFrame with imputed numeric data and non-numeric columns
            all_company_data = pd.concat([pd.DataFrame(imputed_data, columns=numeric_cols), all_company_data.select_dtypes(exclude=np.number)], axis=1)

            # Vectorized percentile calculation
            logger.info("Calculating fundamental score and percentile...")
            all_company_data['fundamental_score'] = all_company_data.apply(lambda row: self._calculate_score(row), axis=1)
            all_company_data['fundamental_score_percentile'] = all_company_data['fundamental_score'].rank(pct=True) * 100

            # Calculate sector percentiles
            logger.info("Calculating sector percentiles...")
            all_company_data['fundamental_score_sector'] = all_company_data.apply(lambda row: self._calculate_score(row, row['sectorkey']), axis=1)
            all_company_data['fundamental_score_sector_percentile'] = all_company_data.groupby('sectorkey')['fundamental_score_sector'].transform(lambda x: x.rank(pct=True) * 100)

            # Reset index to get company_id as a column
            all_company_data = all_company_data.reset_index()

            # Save to database
            logger.info("Saving scores to database...")
            self._save_scores_to_db(db, all_company_data)

        except Exception as e:
            logger.error(f"An error occurred in calculate_fundamental_score_and_percentile: {e}")
            traceback.print_exc()

    def _calculate_score(self, row: pd.Series, sector: str = None) -> float:
        """Calculates the weighted fundamental score for a company."""
        if sector:
            weights = self.sector_weights.get(sector, self.default_weights)
        else:
            weights = self.default_weights
            
        # Check for missing keys in data
        missing_keys = set(weights.keys()) - set(row.index)
        if missing_keys:
            raise ValueError(f"Missing data for metrics: {missing_keys}")

        # Handle potential division by zero errors
        score = sum(weights[metric] * row[metric] for metric in weights if metric != "Debt-to-Equity Ratio")
        if row["Debt-to-Equity Ratio"] != 0:
            score += weights["Debt-to-Equity Ratio"] * (1 / row["Debt-to-Equity Ratio"])
        else:
            score += weights["Debt-to-Equity Ratio"] * 1e9 # Large penalty for zero debt-to-equity

        return score

    def _save_scores_to_db(self, db: Session, scores_df: pd.DataFrame):
        """Saves the calculated scores and percentiles to the database using a batch update."""
        try:
            # Prepare data for bulk update
            update_data = []
            for _, row in scores_df.iterrows():
                update_data.append({
                    "id": row['id'],
                    "fundamental_score": row['fundamental_score'],
                    "fundamental_score_percentile": row['fundamental_score_percentile'],
                    "fundamental_score_sector_percentile": row['fundamental_score_sector_percentile']
                })

            # Perform bulk update
            if update_data:
                db.execute(update(Company), update_data) 
                db.commit()
                logger.info("Scores and percentiles saved to the database successfully.")
            else:
                logger.warning("No data to update.")

        except Exception as e:
            db.rollback()
            logger.error(f"Error saving scores to database: {e}")


def get_candidate_companies(db: Session = None, market: str = 'us', min_avg_volume: int = 50000, min_fundamental_score_pct: float = 0.0, limit: int = 10000):
    """
    Retrieves a list of candidate companies from the database based on basic filtering criteria.

    This function is often used as a preliminary step for more detailed scans to narrow
    down the universe of stocks to a manageable size.

    Args:
        db: The database session. If None, a new session will be created and closed.
        market: The market to scan (e.g., 'us', 'ca').
        min_avg_volume: The minimum average daily trading volume.
        min_fundamental_score_pct: The minimum fundamental score percentile.
        limit: The maximum number of companies to return.

    Returns:
        A list of Company objects that meet the criteria.
    """
    if db is None:
        db = next(get_db())
        try:
            candidates_query = db.query(Company).filter(
                Company.isactive == True,
                Company.exchange.in_(db.query(Exchange.exchange_code).filter(Exchange.country_code == market)),
                Company.regularmarketpreviousclose > 0.5,
                Company.averagedailyvolume10day > min_avg_volume,
                Company.fundamental_score_percentile > min_fundamental_score_pct
            ).order_by(
                desc(Company.fundamental_score_percentile),
                desc(Company.averagedailyvolume10day)
            ).limit(limit)

            candidates = candidates_query.all()
            logger.info(f"Found {len(candidates)} initial candidates to check for BB Breakout.")
            return candidates
        except Exception as e:
            logger.error(f"An error occurred during the BB Breakout scan: {e}", exc_info=True)
            return []
        finally:
            db.close()
    else:
        candidates_query = db.query(Company).filter(
            Company.exchange.in_(db.query(Exchange.exchange_code).filter(Exchange.country_code == market)),
            Company.regularmarketpreviousclose > 0.5,
            Company.averagedailyvolume10day > min_avg_volume,
            Company.fundamental_score_percentile > min_fundamental_score_pct
        ).order_by(
            desc(Company.fundamental_score_percentile),
            desc(Company.averagedailyvolume10day)
        ).limit(limit)

        candidates = candidates_query.all()
        logger.info(f"Found {len(candidates)} initial candidates to check for BB Breakout.")
        return candidates
