"""
A suite of tools for interacting with the yfinance library and the database.

This module is the primary interface for downloading financial data, including company
information, price history, financial statements, and analyst estimates. It handles
the logic for fetching data from the yfinance API, processing it, and saving it
to the PostgreSQL database using SQLAlchemy.

Key functions include:
- `save_or_update_company_data`: The main orchestrator for downloading and updating data for entire markets.
- `load_ticker_data`: Loads all data for a single ticker, fetching from the database or downloading if needed.
- Various helper functions for saving specific data types (e.g., financials, holdings, transactions) to the database.
"""
import logging
import random
import re
from typing import Union
import json
from functools import wraps, lru_cache
import yfinance as yf
from yfinance.exceptions import YFRateLimitError
import pandas as pd
import os, time
from datetime import datetime, timedelta
import pandas_market_calendars as mcal
import pytz
import numpy as np # Ensure numpy is imported
from bs4 import BeautifulSoup
from curl_cffi import requests as cffi_requests
from curl_cffi.requests.exceptions import SSLError, RequestException, HTTPError
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from core.db import get_db
from tools.file_wrapper import convert_to_json_serializable
from core.model import (
    object_as_dict, Company, Exchange, PriceHistory, Financials, CompanyOfficer, UpgradeDowngrade, 
    InstitutionalHolding, InsiderTransaction, InsiderRoster,   
    AnalystEarningsEstimate, AnalystRevenueEstimate, AnalystGrowthEstimate, AnalystEarningsHistory, AnalystEpsTrend, AnalystEpsRevisions
)

logger = logging.getLogger(__name__)

def ttl_cache(ttl_seconds: int, maxsize: int = 128):
    """
    A decorator to add a time-to-live (TTL) cache to a function.

    The cache stores the function's result and expires after `ttl_seconds`.
    It uses an LRU (Least Recently Used) policy to evict old items if the
    cache exceeds `maxsize`.

    Args:
        ttl_seconds: The time-to-live for the cache in seconds.
        maxsize: The maximum size of the cache.
    """
    def decorator(func):
        cache = {}
        # Use a list to track the order of keys for LRU eviction
        lru_keys = []

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create a hashable key from the function's arguments
            key = (args, tuple(sorted(kwargs.items())))
            now = time.time()

            # Check if the key exists and is still valid
            if key in cache and (now - cache[key]['time']) < ttl_seconds:
                # Move key to the end to signify it was recently used
                lru_keys.remove(key)
                lru_keys.append(key)
                return cache[key]['value']

            result = func(*args, **kwargs)

            # Evict the least recently used item if the cache is full
            if len(lru_keys) >= maxsize:
                oldest_key = lru_keys.pop(0)
                del cache[oldest_key]

            cache[key] = {'value': result, 'time': now}
            lru_keys.append(key)
            return result
        return wrapper
    return decorator

# --- Global Constants ---
VOLATILITY_BENCHMARK_TICKER = '^VIX'
FULL_HISTORY_START_DATE = "2000-01-01" # A sensible default for the earliest date for most stock data.
TIMEFRAME_OPTIONS = ["1d", "1h", "30m", "15m"]

def yfinance_retry_handler(retries=5, backoff_factor=5, base_sleep_time=60):
    """
    A decorator factory to handle yfinance API calls with retries and exponential backoff
    for common errors like rate limiting and SSL issues.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Assumes the first positional argument is the ticker string for logging.
            ticker = args[0] if args and isinstance(args[0], str) else 'Unknown Ticker'

            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except (YFRateLimitError, SSLError) as e:
                    error_type = type(e).__name__
                    if attempt < retries - 1:
                        sleep_time = base_sleep_time + backoff_factor ** (attempt + 1) + random.uniform(0, base_sleep_time / 2)
                        logger.warning(f"{error_type} on {func.__name__} for {ticker}. Retrying in {sleep_time:.2f} seconds...")
                        time.sleep(sleep_time)
                        continue
                    else:
                        logger.error(f"Final attempt failed for {func.__name__} on {ticker} due to {error_type}: {e}")
                        return {"error": f"{error_type} after {retries} retries."}
                except Exception as e:
                    logger.error(f"An unexpected error occurred in {func.__name__} for {ticker}: {e}", exc_info=True)
                    return {"error": f"An unexpected error occurred: {e}"}
            
            return {"error": f"Function {func.__name__} failed for {ticker} after all retries."}
        return wrapper
    return decorator

@yfinance_retry_handler()
def is_ticker_active(ticker: str) -> Union[bool, dict]:
    """Check if a ticker is active."""
    logger.info(f"Checking active status for {ticker}")
    company_data = yf.Ticker(ticker)
    if company_data:
        price_history = company_data.history(period='7d', interval='1d')
        return len(price_history) > 0
    return False

def get_yf_competitors(ticker: str) -> Union[list, dict]:
    """Scrapes competitor tickers from Yahoo Finance."""
    retries = 5  # Number of retries
    backoff_factor = 5  # Exponential backoff factor

    for attempt in range(retries):
        try:
            url = f"https://finance.yahoo.com/quote/{ticker}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
            }
            response = cffi_requests.get(url, headers=headers, impersonate="chrome110", timeout=30)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

            soup = BeautifulSoup(response.content, "html.parser")

            competitor_elements = soup.find_all("section", {"data-testid": "compare-to"})
            if not competitor_elements:
                return []

            competitors = []
            for section in competitor_elements:
                # Find all the <a> tags within the section
                links = section.find_all("a")
                for link in links:
                    href = link.get("href")
                    if href and "/quote/" in href : # Check if href exists and contains /quote/
                        match = re.search(r"/quote/([A-Z\.]+)", href) # use [A-Z\.]+
                        if match:
                            competitors.append(match.group(1))

            competitors = list(set(competitors))   # remove duplicates
            if ticker in competitors:
                competitors.remove(ticker)

            return competitors

        except HTTPError as e:
            if e.response.status_code == 429:  # Rate-limited
                if attempt < retries - 1:
                    sleep_time = backoff_factor ** (attempt + 1) # Exponential backoff
                    logger.warning(f"Rate limited. Retrying after {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                else:
                    return {"error": f"Rate limited after multiple retries: {e}"}
            else:  # Other HTTP errors
                return {"error": f"HTTP error: {e}"}
        except RequestException as e:
            return {"error": f"Error fetching data: {e}"}
        except Exception as e:
            return {"error": f"An unexpected error occurred: {e}"}

def load_price_data(ticker: str, start_date: str = None, end_date: str = None, timeframe: str = '1d') -> pd.DataFrame:
    """
    Loads historical price data for the ticker from the database.
    """
    if not start_date:
        start_date = FULL_HISTORY_START_DATE
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')

    db_session = next(get_db())
    try:
        company = db_session.query(Company).filter(Company.symbol == ticker).first()

        query = db_session.query(PriceHistory).filter(PriceHistory.company_id == company.id, PriceHistory.timeframe == timeframe)
        if start_date:
            query = query.filter(PriceHistory.timestamp >= start_date)
        if end_date:
            query = query.filter(PriceHistory.timestamp <= end_date)
        query = query.order_by(PriceHistory.timestamp.asc()) # Ensure data is sorted chronologically

        price_data = pd.read_sql(query.statement, db_session.bind) 
        
        if price_data.empty:
            # logger.warning(f"No {timeframe} price data found for {ticker} between {start_date} and {end_date}.")
            # return pd.DataFrame()
            data_dict = load_ticker_data(ticker, start_date, end_date, timeframe) 
            data_df = data_dict.get('shareprices', pd.DataFrame()) if data_dict else pd.DataFrame()
            data_df.columns = [col.lower().replace(' ', '') for col in data_df.columns]
            if 'date' in data_df.columns:
                data_df['date'] = pd.to_datetime(data_df['date'])
                data_df.set_index('date', inplace=True)
            if data_df.index.tz is not None:
                data_df.index = data_df.index.tz_localize(None)
            if 'timestamp' in data_df.columns:
                data_df.rename(columns={'timestamp': 'date'}, inplace=True)
            return data_df

        # Ensure essential columns are present
        for col in ['open', 'high', 'low', 'close', 'adjclose', 'volume']:
            if col not in price_data.columns:
                logger.error(f"Essential column '{col}' missing from price data for {ticker}.")
                return pd.DataFrame()
        
        # Rename 'timestamp' to 'date' for backward compatibility within this module
        price_data.rename(columns={'timestamp': 'date'}, inplace=True)
        price_data['date'] = pd.to_datetime(price_data['date'])
        try:
            # --- Ensure timezone is removed after loading from DB ---
            if price_data['date'].dt.tz is not None:
                price_data['date'] = price_data['date'].dt.tz_localize(None)
            price_data.set_index('date', inplace=True)
            price_data.drop(columns=['timeframe'], inplace=True, errors='ignore')
        except Exception as e:
            logger.error(f"Error setting 'date' as index in load_price_data: {e}")
            logger.error(f"DataFrame info before error: {price_data.info()}")
            return pd.DataFrame()

        logger.info(f"Loaded {len(price_data)} {timeframe} data points for {ticker}.")
        return price_data
    except Exception as e:
        logger.error(f"Error loading price data for {ticker}: {e}")
        return pd.DataFrame()
    finally:
        if db_session and db_session.is_active:
            db_session.close()


def save_company_to_db(db: Session, company_info: dict):
    """Saves or updates company information in the database."""

    # Remove companyOfficers, executiveTeam and corporateActions, that are not in company table
    keys = list(company_info.keys())
    for key in keys:
        if key != "52WeekChange" and key.lower() not in Company.__table__.columns.keys():
            company_info.pop(key)

    # --- Convert Yahoo Finance keys to lowercase (with only underscore if necessary) ---
    db_ready_info = {}
    for key, value in company_info.items():
        if value is not None: # handle None values
            new_value = value

            # Handle timestamp conversion
            if key in ["earningsTimestamp", "earningsTimestampStart", "earningsTimestampEnd", "earningsCallTimestampStart", "earningsCallTimestampEnd", "postMarketTime", "regularMarketTime"]: 
                if isinstance(value, (int, float)) and value > 0: #Check that it's a int or a float and positive
                    new_value = datetime.utcfromtimestamp(value)
                elif isinstance(value, datetime): # if it's already a datetime object, don't convert
                    pass
                else:
                    new_value = None # Set to None if the value is not usable
            
            # Handle date conversion
            if key in ["governanceEpochDate", "compensationAsOfEpochDate", "sharesShortPreviousMonthDate", "dateShortInterest", "lastFiscalYearEnd", "nextFiscalYearEnd", "mostRecentQuarter", "lastDividendDate", "dividendDate","exDividendDate", "nameChangeDate", "ipoExpectedDate", "lastSplitDate", "firstTradeDateMilliseconds"]:
                if isinstance(value, str):
                    try:
                        new_value = datetime.strptime(value, '%Y-%m-%d').date()
                    except ValueError:
                        new_value = None
                elif isinstance(value, datetime):
                    new_value = value.date() # convert to date
                elif isinstance(value, (int, float)) and value > 0:
                    if key == "firstTradeDateMilliseconds" : # handle epoch date in milliseconds
                        new_value = datetime.utcfromtimestamp(value/1000).date()
                    else:
                        new_value = datetime.utcfromtimestamp(value).date()
                else:
                    new_value = None

            # Handle other cases
            if key == "52WeekChange":
                new_key = "_52weekchange"
            else:
                new_key = key.lower()
            db_ready_info[new_key] = new_value

    # Convert debttoequity from percentage to ratio
    if "debttoequity" in db_ready_info and db_ready_info["debttoequity"] > 1:
        db_ready_info["debttoequity"] /= 100

    # Get the symbol (it is always present)
    company_symbol = db_ready_info["symbol"]

    existing_company = db.query(Company).filter(Company.symbol == company_symbol).first()

    if existing_company:
        # Update existing company
        for key, value in db_ready_info.items():
            if hasattr(existing_company, key):
                setattr(existing_company, key, value)
    else:
        # Create new company
        company = Company(**db_ready_info)
        db.add(company)

    db.commit()
    return db.query(Company).filter(Company.symbol == company_symbol).first()

def save_analyst_earnings_estimates_to_db(db: Session, company_id: int, estimates_df: pd.DataFrame):
    if estimates_df is None or estimates_df.empty:
        logger.info(f"No analyst earnings estimates for company_id {company_id}.")
        return
    try:
        records_to_insert = []
        # yfinance earnings_estimate has columns like: '0q', '+1q', '0y', '+1y' as index
        # and 'No. of Analysts', 'Avg. Estimate', 'Low Estimate', 'High Estimate', 'Year Ago EPS', 'EPS Growth', 'Revenue Growth' as columns
        for period, row in estimates_df.iterrows(): # iterrows() is fine for small DFs like this
            record = {
                "company_id": company_id,
                "period_label": period, # This is '0q', '+1q', etc.
                "num_analysts": int(row.get("numberOfAnalysts", 0)) if pd.notna(row.get("numberOfAnalysts")) else None,
                "avg_estimate": float(row.get("avg", 0.0)) if pd.notna(row.get("avg")) else None,
                "low_estimate": float(row.get("low", 0.0)) if pd.notna(row.get("low")) else None,
                "high_estimate": float(row.get("high", 0.0)) if pd.notna(row.get("high")) else None,
                "year_ago_eps": float(row.get("yearAgoEps", 0.0)) if pd.notna(row.get("yearAgoEps")) else None,
                # 'EPS Growth' and 'Revenue Growth' are sometimes in earnings_estimate, sometimes separate
                "eps_growth_percent": float(row.get("growth", 0.0)) if pd.notna(row.get("growth")) else None,
                "revenue_growth_percent": float(row.get("Revenue Growth", 0.0)) if pd.notna(row.get("Revenue Growth")) else None,
            }
            records_to_insert.append(record)

        if records_to_insert:
            stmt = pg_insert(AnalystEarningsEstimate).values(records_to_insert)
            on_conflict_stmt = stmt.on_conflict_do_update(
                index_elements=['company_id', 'period_label'],
                set_={col: getattr(stmt.excluded, col) for col in AnalystEarningsEstimate.__table__.columns.keys() if col not in ['id', 'company_id', 'period_label', 'last_updated']}
            )
            db.execute(on_conflict_stmt)
            db.commit()
            logger.info(f"Saved/Updated {len(records_to_insert)} analyst earnings estimates for company_id {company_id}.")
    except Exception as e:
        db.rollback()
        logger.error(f"Error saving analyst earnings estimates for company_id {company_id}: {e}")

def save_analyst_revenue_estimates_to_db(db: Session, company_id: int, estimates_df: pd.DataFrame):
    if estimates_df is None or estimates_df.empty:
        logger.info(f"No analyst revenue estimates for company_id {company_id}.")
        return
    try:
        records_to_insert = []
        # yfinance revenue_estimate has columns like: '0q', '+1q', '0y', '+1y' as index
        # and 'No. of Analysts', 'Avg. Estimate', 'Low Estimate', 'High Estimate', 'Year Ago Revenue', 'Growth' as columns
        for period, row in estimates_df.iterrows():
            record = {
                "company_id": company_id,
                "period_label": period,
                "num_analysts": int(row.get("numberOfAnalysts", 0)) if pd.notna(row.get("numberOfAnalysts")) else None,
                "avg_estimate": int(row.get("avg", 0)) if pd.notna(row.get("avg")) else None, # Revenue is large
                "low_estimate": int(row.get("low", 0)) if pd.notna(row.get("low")) else None,
                "high_estimate": int(row.get("high", 0)) if pd.notna(row.get("high")) else None,
                "year_ago_revenue": int(row.get("yearAgoRevenue", 0)) if pd.notna(row.get("yearAgoRevenue")) else None,
                "revenue_growth_percent": float(row.get("growth", 0.0)) if pd.notna(row.get("growth")) else None, # yfinance uses 'Growth'
            }
            records_to_insert.append(record)

        if records_to_insert:
            stmt = pg_insert(AnalystRevenueEstimate).values(records_to_insert)
            on_conflict_stmt = stmt.on_conflict_do_update(
                index_elements=['company_id', 'period_label'],
                set_={col: getattr(stmt.excluded, col) for col in AnalystRevenueEstimate.__table__.columns.keys() if col not in ['id', 'company_id', 'period_label', 'last_updated']}
            )
            db.execute(on_conflict_stmt)
            db.commit()
            logger.info(f"Saved/Updated {len(records_to_insert)} analyst revenue estimates for company_id {company_id}.")
    except Exception as e:
        db.rollback()
        logger.error(f"Error saving analyst revenue estimates for company_id {company_id}: {e}")

def save_analyst_growth_estimates_to_db(db: Session, company_id: int, estimates_df: pd.DataFrame):
    # yfinance .growth_estimates is often a Series, convert to DataFrame if so
    if isinstance(estimates_df, pd.Series):
        estimates_df = estimates_df.to_frame(name="Value") # Convert Series to DataFrame

    if estimates_df is None or estimates_df.empty:
        logger.info(f"No analyst growth estimates for company_id {company_id}.")
        return
    try:
        records_to_insert = []
        # The DataFrame might have 'Growth Estimates - TICKER' as column name or just 'Value'
        # Or it might have the ticker symbol as the column name.
        value_col_name = estimates_df.columns[0] # Assume the first column has the value

        for period, row in estimates_df.iterrows(): # period is the index e.g. "Next 5 Years (per annum)"
            record = {
                "company_id": company_id,
                "period_label": period,
                "growth_value_text": str(row.get(value_col_name)) if pd.notna(row.get(value_col_name)) else None,
            }
            records_to_insert.append(record)

        if records_to_insert:
            stmt = pg_insert(AnalystGrowthEstimate).values(records_to_insert)
            on_conflict_stmt = stmt.on_conflict_do_update(
                index_elements=['company_id', 'period_label'],
                set_={"growth_value_text": stmt.excluded.growth_value_text, "last_updated": datetime.utcnow()}
            )
            db.execute(on_conflict_stmt)
            db.commit()
            logger.info(f"Saved/Updated {len(records_to_insert)} analyst growth estimates for company_id {company_id}.")
    except Exception as e:
        db.rollback()
        logger.error(f"Error saving analyst growth estimates for company_id {company_id}: {e}")

def save_analyst_earnings_history_to_db(db: Session, company_id: int, history_df: pd.DataFrame):
    if history_df is None or history_df.empty:
        logger.info(f"No analyst earnings history for company_id {company_id}.")
        return
    try:
        records_to_insert = []
        # Index is 'Report Date', columns are 'EPS Estimate', 'EPS Actual', 'Difference', 'Surprise %'
        for index_date, row in history_df.iterrows(): # index_date is the actual report date from the DataFrame's index
            record = {
                "company_id": company_id,
                "report_date": index_date.date() if isinstance(index_date, pd.Timestamp) else pd.to_datetime(index_date).date(),
                "eps_estimate": float(row.get("epsEstimate", 0.0)) if pd.notna(row.get("epsEstimate")) else None,
                "eps_actual": float(row.get("epsActual", 0.0)) if pd.notna(row.get("epsActual")) else None,
                "eps_difference": float(row.get("epsDifference", 0.0)) if pd.notna(row.get("epsDifference")) else None,
                "surprise_percent": float(row.get("surprisePercent", 0.0)) if pd.notna(row.get("surprisePercent")) else None,
            }
            records_to_insert.append(record)

        if records_to_insert:
            stmt = pg_insert(AnalystEarningsHistory).values(records_to_insert)
            # Assuming each report_date is unique for a company's earnings history
            on_conflict_stmt = stmt.on_conflict_do_nothing(index_elements=['company_id', 'report_date'])
            db.execute(on_conflict_stmt)
            db.commit()
            logger.info(f"Saved/Updated {len(records_to_insert)} analyst earnings history entries for company_id {company_id}.")
    except Exception as e:
        db.rollback()
        logger.error(f"Error saving analyst earnings history for company_id {company_id}: {e}")

def save_analyst_trend_or_revisions_to_db(db: Session, company_id: int, df: pd.DataFrame, model_class, column_mapping: dict, table_name: str):
    if df is None or df.empty:
        logger.info(f"No {table_name} data for company_id {company_id}.")
        return
    try:
        records_to_insert = []
        # Index is '0q', '+1q', etc. Columns vary.
        for period, row in df.iterrows():
            record = {"company_id": company_id, "period_label": period}
            for db_col, df_col in column_mapping.items():
                value = row.get(df_col)
                # Ensure value is float or int if notna, else None
                record[db_col] = float(value) if pd.notna(value) and isinstance(value, (int, float, np.number)) else (int(value) if pd.notna(value) and isinstance(value, (int, np.number)) else None)
            records_to_insert.append(record)

        if records_to_insert:
            stmt = pg_insert(model_class).values(records_to_insert)
            on_conflict_stmt = stmt.on_conflict_do_update(
                index_elements=['company_id', 'period_label'],
                set_={col: getattr(stmt.excluded, col) for col in model_class.__table__.columns.keys() if col not in ['id', 'company_id', 'period_label', 'last_updated']}
            )
            db.execute(on_conflict_stmt)
            db.commit()
            logger.info(f"Saved/Updated {len(records_to_insert)} {table_name} entries for company_id {company_id}.")
    except Exception as e:
        db.rollback()
        logger.error(f"Error saving {table_name} for company_id {company_id}: {e}")

def save_upgrades_downgrades_to_db(db: Session, company_id: int, upgrades_downgrades_df: pd.DataFrame):
    """Saves or updates analyst upgrades/downgrades in the database."""
    if upgrades_downgrades_df is None or upgrades_downgrades_df.empty:
        logger.info(f"No upgrades/downgrades data for company_id {company_id}.")
        return
    try:
        records_to_insert = []
        for index, row in upgrades_downgrades_df.iterrows():
            # Ensure index (date) is converted to Python date
            report_date = index.to_pydatetime().date() if isinstance(index, pd.Timestamp) else pd.to_datetime(index).date()

            record = {
                "company_id": company_id,
                "date": report_date,
                "firm": row.get("Firm"),
                "to_grade": row.get("ToGrade"),
                "from_grade": row.get("FromGrade"),
                "action": row.get("Action")
            }
            records_to_insert.append(record)

        if records_to_insert:
            stmt = pg_insert(UpgradeDowngrade).values(records_to_insert)
            on_conflict_stmt = stmt.on_conflict_do_nothing(
                index_elements=['company_id', 'date', 'firm', 'to_grade', 'action']
            )
            db.execute(on_conflict_stmt)
            db.commit()
            logger.info(f"Saved/Updated {len(records_to_insert)} upgrades/downgrades for company_id {company_id}.")
    except Exception as e:
        db.rollback()
        logger.error(f"Error saving upgrades/downgrades for company_id {company_id}: {e}")

def save_institutional_holdings_to_db(db: Session, company_id: int, holdings_df: pd.DataFrame, holder_type: str):
    """Saves or updates institutional or mutual fund holdings in the database."""
    if holdings_df is None or holdings_df.empty:
        logger.info(f"No {holder_type} holdings data for company_id {company_id}.")
        return
    try:
        records_to_insert = []
        for index, row in holdings_df.iterrows():
            date_reported = pd.to_datetime(row.get("Date Reported")).date() if pd.notna(row.get("Date Reported")) else None
            if date_reported is None: # Skip if no date reported
                continue
            record = {
                "company_id": company_id,
                "holder_name": row.get("Holder"),
                "shares": int(row.get("Shares",0)),
                "date_reported": date_reported,
                "percent_out": float(row.get("% Out", 0.0)),
                "value": int(row.get("Value", 0)),
                "holder_type": holder_type
            }
            records_to_insert.append(record)

        if records_to_insert:
            stmt = pg_insert(InstitutionalHolding).values(records_to_insert)
            on_conflict_stmt = stmt.on_conflict_do_update(
                index_elements=['company_id', 'holder_name', 'date_reported', 'holder_type'],
                set_={
                    "shares": stmt.excluded.shares,
                    "percent_out": stmt.excluded.percent_out,
                    "value": stmt.excluded.value,
                }
            )
            db.execute(on_conflict_stmt)
            db.commit()
            logger.info(f"Saved/Updated {len(records_to_insert)} {holder_type} holdings for company_id {company_id}.")
    except Exception as e:
        db.rollback()
        logger.error(f"Error saving {holder_type} holdings for company_id {company_id}: {e}")

def save_insider_transactions_to_db(db: Session, company_id: int, transactions_df: pd.DataFrame):
    """Saves or updates insider transactions in the database."""
    if transactions_df is None or transactions_df.empty:
        logger.info(f"No insider transactions data for company_id {company_id}.")
        return
    try:
        records_to_insert = []
        # Ensure 'Start Date' is datetime
        transactions_df['Start Date'] = pd.to_datetime(transactions_df['Start Date'])
        for index, row in transactions_df.iterrows():
            record = {
                "company_id": company_id,
                "insider_name": row.get("Insider"),
                "shares": int(row.get("Shares", 0)),
                "transaction_type": row.get("Transaction"), # From 'Transaction' column
                "transaction_code": row.get("Ownership"), # From 'Ownership' column (e.g. D for Direct)
                "start_date": row.get("Start Date").date(), # Convert to date
                "value": int(row.get("Value", 0)) if pd.notna(row.get("Value")) else None,
            }
            records_to_insert.append(record)

        if records_to_insert:
            stmt = pg_insert(InsiderTransaction).values(records_to_insert)
            on_conflict_stmt = stmt.on_conflict_do_nothing( # Assuming transactions are unique by this combination
                index_elements=['company_id', 'insider_name', 'transaction_type', 'start_date', 'shares']
            )
            db.execute(on_conflict_stmt)
            db.commit()
            logger.info(f"Saved/Updated {len(records_to_insert)} insider transactions for company_id {company_id}.")
    except Exception as e:
        db.rollback()
        logger.error(f"Error saving insider transactions for company_id {company_id}: {e}")

def save_insider_roster_to_db(db: Session, company_id: int, roster_df: pd.DataFrame):
    """Saves or updates insider roster in the database."""
    if roster_df is None or roster_df.empty:
        logger.info(f"No insider roster data for company_id {company_id}.")
        return
    try:
        records_to_insert = []
        # yfinance provides 'Latest Transaction Date' as a datetime object.
        # If it's not already datetime, this line would ensure it is, but it's usually redundant.
        # The key is using the correct column name 'Latest Transaction Date'.
        if 'Latest Transaction Date' in roster_df.columns and not pd.api.types.is_datetime64_any_dtype(roster_df['Latest Transaction Date']):
            roster_df['Latest Transaction Date'] = pd.to_datetime(roster_df['Latest Transaction Date'])
        for index, row in roster_df.iterrows():
            record = {
                "company_id": company_id,
                "name": row.get("Name"),
                "position": row.get("Position"),
                "most_recent_transaction": row.get("Most Recent Transaction"),
                "most_recent_transaction_date": row.get("Latest Transaction Date").date() if pd.notna(row.get("Latest Transaction Date")) else None,
                "shares_owned_directly": int(row.get("Shares Owned Directly", 0)) if pd.notna(row.get("Shares Owned Directly")) else None,
                "shares_owned_indirectly": int(row.get("Shares Owned Indirectly", 0)) if pd.notna(row.get("Shares Owned Indirectly")) else None,
            }
            records_to_insert.append(record)

        if records_to_insert:
            stmt = pg_insert(InsiderRoster).values(records_to_insert)
            on_conflict_stmt = stmt.on_conflict_do_update(
                index_elements=['company_id', 'name', 'position'],
                set_={
                    "most_recent_transaction": stmt.excluded.most_recent_transaction,
                    "most_recent_transaction_date": stmt.excluded.most_recent_transaction_date,
                    "shares_owned_directly": stmt.excluded.shares_owned_directly,
                    "shares_owned_indirectly": stmt.excluded.shares_owned_indirectly,
                }
            )
            db.execute(on_conflict_stmt)
            db.commit()
            logger.info(f"Saved/Updated {len(records_to_insert)} insider roster entries for company_id {company_id}.")
    except Exception as e:
        db.rollback()
        logger.error(f"Error saving insider roster for company_id {company_id}: {e}")
        
def save_or_update_batch_price_data(db: Session, batch_price_data: dict, timeframe: str = '1d'):
    """
    Saves or updates price history for a batch of companies in the database, using PostgreSQL's ON CONFLICT for efficiency.

    Args:
        db: The SQLAlchemy database session.
        batch_price_data: A dictionary containing price history data for multiple companies.
        timeframe: The timeframe of the price data (e.g., '1d', '1h').
    """
    try:
        records_to_insert = []
        for company_id, price_history_df in batch_price_data.items():
            price_history_df = price_history_df.dropna()
            for index, row in price_history_df.iterrows():
                try:
                    price_data = {
                        "company_id": company_id,
                        "timestamp": index.to_pydatetime(), # Keep as datetime object
                        "timeframe": timeframe,
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "close": float(row["Close"]),
                        "adjclose": float(row["Adj Close"]),
                        "volume": int(row["Volume"]),
                        "dividend_amount": float(row.get("Dividends", 0.0)),  # Use .get() to handle missing column
                        "split_coefficient": float(row.get("Stock Splits", 0.0))  # Use .get() to handle missing column
                    }
                except Exception as e:
                    logger.warning(f"Skipped price data for company {company_id} because of error: {e}")
                    continue
                
                #print(price_data)
                records_to_insert.append(price_data)

        if records_to_insert:
            # Use upsert for all entries
            stmt = pg_insert(PriceHistory).values(records_to_insert)
            on_conflict_stmt = stmt.on_conflict_do_update(
                index_elements=[PriceHistory.company_id, PriceHistory.timeframe, PriceHistory.timestamp],
                set_={
                    "open": stmt.excluded.open,
                    "high": stmt.excluded.high,
                    "low": stmt.excluded.low,
                    "close": stmt.excluded.close,
                    "adjclose": stmt.excluded.adjclose,
                    "volume": stmt.excluded.volume,
                    "dividend_amount": stmt.excluded.dividend_amount,
                    "split_coefficient": stmt.excluded.split_coefficient
                }
            )
            db.execute(on_conflict_stmt)
            logger.info(f"Upserted {timeframe} price history for {len(batch_price_data)} companies with {len(records_to_insert)} entries.")

        db.commit()  # Commit all changes in a single transaction
    except Exception as e:
        db.rollback()
        logger.error(f"An error occurred while saving price history: {e}")

def save_or_update_financials(db: Session, company_id: int, financial_statements: dict):
    """
    Saves or updates financial data for a given company in the database,
    handling multiple financial statements efficiently using PostgreSQL's ON CONFLICT.
    """
    try:
        all_records_to_insert = []
        for statement_type, financial_data in financial_statements.items():
            if financial_data is None or financial_data.empty:
                logger.info(f"No {statement_type} found for company {company_id}.")
                continue

            for report_date, row in financial_data.transpose().iterrows():
                for index, value in row.items():
                    # Create a dictionary of the data
                    new_value = convert_to_json_serializable(value)
                    financial_entry = {
                        "company_id": company_id,
                        "report_date": report_date,
                        "type": statement_type,
                        "index": index,
                        "value": new_value
                    }
                    all_records_to_insert.append(financial_entry)

        # Use upsert for all entries
        if all_records_to_insert:
            # Create an insert statement with on_conflict_do_update
            stmt = pg_insert(Financials).values(all_records_to_insert)
            on_conflict_stmt = stmt.on_conflict_do_update(
                index_elements=[Financials.company_id, Financials.report_date, Financials.type, Financials.index],
                set_={"value": stmt.excluded.value}
            )
            db.execute(on_conflict_stmt)
            logger.info(f"Upserted financials data for company {company_id} with {len(all_records_to_insert)} entries.")
        db.commit()  # Commit all changes in a single transaction
    except Exception as e:
        db.rollback()
        logger.error(f"An error occurred while saving financials for company {company_id}: {e}")

def save_company_officers_to_db(db: Session, company_id: int, officers_list: list):
    """Saves company officers data to the database."""
    if not officers_list:
      return
    for officer_info in officers_list:
        officer_data = {
            "company_id": company_id,
            "maxAge": officer_info.get("maxAge"),
            "name": officer_info.get("name"),
            "age": officer_info.get("age"),
            "title": officer_info.get("title"),
            "yearBorn": officer_info.get("yearBorn"),
            "fiscalYear": officer_info.get("fiscalYear"),
            "totalPay": officer_info.get("totalPay"),
            "exercisedValue": officer_info.get("exercisedValue"),
            "unexercisedValue": officer_info.get("unexercisedValue")
        }
        
        existing_record = db.query(CompanyOfficer).filter(
            CompanyOfficer.company_id == company_id,
            CompanyOfficer.name == officer_data["name"]
        ).first()
        if existing_record:
            # Update existing record
            for key, value in officer_data.items():
                setattr(existing_record, key, value)
        else:
            # Create new record
            officer_record = CompanyOfficer(**officer_data)
            db.add(officer_record)

    db.commit()

def save_extra_company_data_to_db(db, company_id, data, info=None):
    if info:
        save_company_officers_to_db(db, company_id, info.get("companyOfficers"))
    else:
        save_company_officers_to_db(db, company_id, data.get_info().get("companyOfficers"))

    eps_trend_mapping = {"current_estimate": "current", "seven_days_ago": "7daysAgo", 
                            "thirty_days_ago": "30daysAgo", "sixty_days_ago": "60daysAgo", 
                            "ninety_days_ago": "90daysAgo"}
    eps_revisions_mapping = {"up_last_7_days": "upLast7days", "up_last_30_days": "upLast30days", 
                                "down_last_7_days": "downLast7Days", "down_last_30_days": "downLast30days"}

    #try to save extra data if available
    try:
        save_upgrades_downgrades_to_db(db, company_id, data.upgrades_downgrades)
    except Exception as e:
        logger.warning(f"save_upgrades_downgrades_to_db error for company {company_id}: {e}")
    
    try:
        save_institutional_holdings_to_db(db, company_id, data.institutional_holders, 'institutional')
        save_institutional_holdings_to_db(db, company_id, data.mutualfund_holders, 'mutualfund')
    except Exception as e:
        logger.warning(f"save_institutional_holdings_to_db error for company {company_id}: {e}")
    
    try:
        save_insider_transactions_to_db(db, company_id, data.insider_transactions)
    except Exception as e:
        logger.warning(f"save_insider_transactions_to_db error for company {company_id}: {e}")
    
    try:
        save_insider_roster_to_db(db, company_id, data.insider_roster_holders)
    except Exception as e:
        logger.warning(f"save_insider_roster_to_db error for company {company_id}: {e}")
    
    try:
        save_analyst_earnings_estimates_to_db(db, company_id, data.earnings_estimate)
    except Exception as e:
        logger.warning(f"save_analyst_earnings_estimates_to_db error for company {company_id}: {e}")
    
    try:
        save_analyst_revenue_estimates_to_db(db, company_id, data.revenue_estimate)
    except Exception as e:
        logger.warning(f"save_analyst_revenue_estimates_to_db error for company {company_id}: {e}")
    
    try:
        save_analyst_growth_estimates_to_db(db, company_id, data.growth_estimates)
    except Exception as e:
        logger.warning(f"save_analyst_growth_estimates_to_db error for company {company_id}: {e}")
    
    try:
        save_analyst_earnings_history_to_db(db, company_id, data.earnings_history)
    except Exception as e:
        logger.warning(f"save_analyst_earnings_history_to_db error for company {company_id}: {e}")
    
    try:
        save_analyst_trend_or_revisions_to_db(db, company_id, data.eps_trend, AnalystEpsTrend, eps_trend_mapping, "analyst_eps_trend")
        save_analyst_trend_or_revisions_to_db(db, company_id, data.eps_revisions, AnalystEpsRevisions, eps_revisions_mapping, "analyst_eps_revisions")
    except Exception as e:
        logger.warning(f"save_analyst_trend_or_revisions_to_db error for company {company_id}: {e}")

    try:
        # Get financial statements first, as they are needed to map earnings dates
        financial_statements = {
            "quarterly_income_statement": data.get_income_stmt(freq="quarterly"),
        }
        quarterly_financials = financial_statements.get("quarterly_income_statement")
        if quarterly_financials is None:
            quarterly_financials = pd.DataFrame() # Ensure it's a DataFrame
    except Exception as e:
        logger.warning(f"save_earnings_dates_to_db error for company {company_id}: {e}")
    

    # Get financial statements
    financial_statements = {
        "annual_balance_sheet": data.get_balance_sheet(freq="yearly"),
        "annual_income_statement": data.get_income_stmt(freq="yearly"),
        "annual_cash_flow": data.get_cash_flow(freq="yearly"),
        "quarterly_balance_sheet": data.get_balance_sheet(freq="quarterly"),
        "quarterly_income_statement": data.get_income_stmt(freq="quarterly"),
        "quarterly_cash_flow": data.get_cash_flow(freq="quarterly"),
    }
    save_or_update_financials(db, company_id, financial_statements)
    return financial_statements

@yfinance_retry_handler()
def _download_and_save_single_ticker_data(ticker: str, db: Session, start_date: str, end_date: str, timeframe: str, price_only: bool = False) -> dict:
    """
    Downloads all data for a single ticker and saves to DB. Wrapped with retry logic.
    This is a helper function for load_ticker_data.
    """
    result = {}
    data = yf.Ticker(ticker)
    company = save_company_to_db(db, data.info)
    if not price_only:
        result['company'] = object_as_dict(company)

        financial_statements = save_extra_company_data_to_db(db, company.id, data)
        result.update(financial_statements)
    
    price_data = data.history(start=start_date, end=end_date, auto_adjust=False, period="max", interval=timeframe)
    if price_data is not None:
        if isinstance(price_data.columns, pd.MultiIndex):
            price_data.columns = [col[1] if col[0].lower().startswith('adj') else col[0] for col in price_data.columns]
        save_or_update_batch_price_data(db, {company.id: price_data})
        price_data = price_data.reset_index()
        result['shareprices'] = price_data

    return result

def load_ticker_data(ticker: str, start_date: str = None, end_date: str = None, refresh: bool = False, timeframe: str = '1d', price_only: bool = True) -> dict:
    """
    Loads data for a given ticker, either from the database or by downloading and saving it.

    Args:
        ticker: The stock ticker symbol.
        start_date: Start date for historical data (if applicable).
        end_date: End date for historical data (if applicable).
        refresh: Whether or not download and update the existing data.
        timeframe: The timeframe of the price data (e.g., '1d', '1h').
        price_only: Whether or not return only price data.

    Returns:
        A dictionary containing the data for the ticker, or None if there is an error.
        Example:
        {
            'company': { ... },
            'shareprices': pd.DataFrame(...),
            'annual_balance_sheet': pd.DataFrame(...),
            ...
        }
    """
    db = next(get_db())  # Get database session
    company = db.query(Company).filter(Company.symbol == ticker).first()
    try:
        if company and not refresh:
            # Data in database
            logger.info(f"Loading {ticker} data from database.")
            result = {}
            result['company'] = object_as_dict(company)

            # Get price history filtered by company_id, start_date and end_date if present
            query = db.query(PriceHistory).filter(PriceHistory.company_id == company.id, PriceHistory.timeframe == timeframe)
            if start_date:
                query = query.filter(PriceHistory.timestamp >= start_date)
            if end_date:
                query = query.filter(PriceHistory.timestamp <= end_date)
            price_history = query.all()

            df = pd.DataFrame([{
                "Date": record.timestamp, # Keep the full timestamp to preserve intraday data
                "Open": record.open,
                "High": record.high,
                "Low": record.low,
                "Close": record.close,
                "Adj Close": record.adjclose,
                "Volume": record.volume,
                "Dividends": record.dividend_amount,
                "Stock Splits": record.split_coefficient
            } for record in price_history])
            if not df.empty:
                df = df.set_index("Date")
                df.sort_index(inplace=True) # CRITICAL FIX: Ensure data is sorted chronologically.
                result['shareprices'] = df

            if not price_only:
                financials = db.query(Financials).filter(Financials.company_id == company.id).all()
                statements = {}
                for record in financials:
                    if record.type not in statements:
                        statements[record.type] = {}
                    if record.report_date not in statements[record.type]:
                        statements[record.type][record.report_date] = {}
                    statements[record.type][record.report_date][record.index] = record.value

                # Convert to pandas DataFrames
                for statement_type, report_data in statements.items():
                    statement_list = []
                    for report_date, index_value in report_data.items():
                        statement_list.append(dict({'Date': report_date}, **index_value))
                    
                    df_statement = pd.DataFrame(statement_list)
                    if not df_statement.empty:
                        result[statement_type] = df_statement.set_index('Date')

            return result

        # Data not in db or needs refresh. Download and save
        logger.info(f"Downloading and saving data for {ticker}")
        download_result = _download_and_save_single_ticker_data(ticker, db, start_date, end_date, timeframe, price_only)
        if isinstance(download_result, dict) and "error" in download_result:
            logger.error(f"Failed to download and save data for {ticker}: {download_result['error']}")
            return None
        return download_result

    except Exception as e:  # Handle exceptions and print or log errors
        logger.error(f"Error loading or downloading {ticker}: {e}", exc_info=True)
        return None  # or return an error message
    finally:
        db.close()


@yfinance_retry_handler()
def _fetch_ticker_history(ticker: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
    """Helper to fetch yfinance history for a single ticker, wrapped with retry logic."""
    logger.info(f"Downloading price history for {ticker}")
    return yf.Ticker(ticker).history(
        start=start_date, end=end_date, interval=interval, auto_adjust=False, period="max"
    )

def download_multiple_tickers_data(tickers: list, start_date: str = None, end_date: str = None, interval: str = "1d") -> pd.DataFrame:
    """
    Downloads price history data for multiple tickers, handling rate limits and retries.
    """
    all_data = {}

    # For multiple tickers, we build a dictionary and concatenate.
    for ticker in tickers:
        ticker_data = _fetch_ticker_history(ticker, start_date, end_date, interval)
        if isinstance(ticker_data, dict) and 'error' in ticker_data:
            logger.error(f"Failed to download data for {ticker}: {ticker_data['error']}")
            all_data[ticker] = pd.DataFrame()
        elif not ticker_data.empty:
            all_data[ticker] = ticker_data

    # Concatenate multiple tickers into a single DataFrame with a MultiIndex
    if all_data:
        result = pd.concat(all_data, axis=1, names=["Ticker", "Attributes"])
        result.columns = [f'{col[1]}' if col[0] == "Adj Close" else f'{col[0]}-{col[1]}' for col in result.columns] 
        return result
    else:
        return pd.DataFrame()

@yfinance_retry_handler()
def _fetch_ticker_info_and_data_object(ticker: str) -> list:
    """Helper to get yfinance Ticker object and info, with retry logic."""
    logger.info(f"Downloading company info for {ticker}")
    company_data = yf.Ticker(ticker)
    if company_data:
        info = company_data.get_info()
        if info:
            return [company_data, info]
    return None

def get_company_info_batch(tickers: list) -> dict:
    """Retrieves company info for multiple tickers, one by one."""
    company_info_batch = {}
    for ticker in tickers:
        result = _fetch_ticker_info_and_data_object(ticker)
        if result and not (isinstance(result, dict) and 'error' in result):
            company_info_batch[ticker] = result
        else:
            logger.error(f"Failed to get info for {ticker}")
    return company_info_batch

@yfinance_retry_handler()
def get_analyst_upgrades_downgrades(ticker: str, refresh: bool = False) -> Union[pd.DataFrame, dict]:
    """Retrieves analyst upgrades/downgrades for a ticker, prioritizing the database."""
    db = next(get_db())
    try:
        company = db.query(Company).filter(Company.symbol == ticker).first()
        if not company:
            return {"error": f"Company {ticker} not found in DB."}

        if not refresh:
            records = db.query(UpgradeDowngrade).filter(UpgradeDowngrade.company_id == company.id).order_by(UpgradeDowngrade.date.desc()).all()
            if records:
                logger.debug(f"Loaded {len(records)} upgrades/downgrades from DB for {ticker}.")
                df = pd.DataFrame([object_as_dict(r) for r in records])
                df = df.drop(columns=['id', 'company_id'])
                df['date'] = pd.to_datetime(df['date'])
                return df.set_index('date')

        logger.info(f"Fetching analyst upgrades/downgrades from yfinance for {ticker}")
        company_data = yf.Ticker(ticker)
        if company_data:
            data = company_data.upgrades_downgrades
            if data is not None and not data.empty:
                save_upgrades_downgrades_to_db(db, company.id, data)
                return data
            return pd.DataFrame()
        return {"error": f"Could not retrieve ticker data for {ticker}"}
    finally:
        db.close()

@yfinance_retry_handler()
def _get_holdings_data(ticker: str, holder_type: str, refresh: bool = False) -> Union[pd.DataFrame, dict]:
    """Helper to retrieve institutional or mutual fund holdings, prioritizing the database."""
    db = next(get_db())
    try:
        company = db.query(Company).filter(Company.symbol == ticker).first()
        if not company:
            return {"error": f"Company {ticker} not found in DB."}

        if not refresh:
            records = db.query(InstitutionalHolding).filter(
                InstitutionalHolding.company_id == company.id,
                InstitutionalHolding.holder_type == holder_type
            ).order_by(InstitutionalHolding.date_reported.desc()).all()
            if records:
                logger.debug(f"Loaded {len(records)} {holder_type} holdings from DB for {ticker}.")
                df = pd.DataFrame([object_as_dict(r) for r in records])
                return df.drop(columns=['id', 'company_id', 'holder_type'])

        logger.info(f"Downloading {holder_type} holdings for {ticker}")
        company_data = yf.Ticker(ticker)
        if not company_data:
            return {"error": f"Could not retrieve ticker data for {ticker}"}

        if holder_type == 'institutional':
            data = company_data.institutional_holders
        elif holder_type == 'mutualfund':
            data = company_data.mutualfund_holders
        else:
            return {"error": "Invalid holder type specified."}
        
        if data is not None and not data.empty:
            save_institutional_holdings_to_db(db, company.id, data, holder_type)
            return data
        else:
            return pd.DataFrame()
    finally:
        db.close()

def get_institutional_holdings(ticker, refresh: bool = False):
    """Retrieves institutional holdings for a ticker."""
    return _get_holdings_data(ticker, 'institutional', refresh)

def get_mutualfund_holdings(ticker, refresh: bool = False):
    """Retrieves mutual fund holdings for a ticker."""
    return _get_holdings_data(ticker, 'mutualfund', refresh)

@yfinance_retry_handler()
def get_insider_transactions(ticker: str, refresh: bool = False) -> Union[pd.DataFrame, dict]:
    """Retrieves insider transactions for a ticker, prioritizing the database."""
    db = next(get_db())
    try:
        company = db.query(Company).filter(Company.symbol == ticker).first()
        if not company:
            return {"error": f"Company {ticker} not found in DB."}

        if not refresh:
            records = db.query(InsiderTransaction).filter(InsiderTransaction.company_id == company.id).order_by(InsiderTransaction.start_date.desc()).all()
            if records:
                logger.debug(f"Loaded {len(records)} insider transactions from DB for {ticker}.")
                df = pd.DataFrame([object_as_dict(r) for r in records])
                return df.drop(columns=['id', 'company_id'])

        logger.info(f"Downloading insider transactions for {ticker}")
        company_data = yf.Ticker(ticker)
        if company_data:
            data = company_data.insider_transactions
            if data is not None and not data.empty:
                save_insider_transactions_to_db(db, company.id, data)
                return data
            else:
                return pd.DataFrame()
        return {"error": f"Could not retrieve ticker data for {ticker}"}
    finally:
        db.close()

@yfinance_retry_handler()
def get_insider_roster(ticker: str, refresh: bool = False) -> Union[pd.DataFrame, dict]:
    """Retrieves insider roster for a ticker, prioritizing the database."""
    db = next(get_db())
    try:
        company = db.query(Company).filter(Company.symbol == ticker).first()
        if not company:
            return {"error": f"Company {ticker} not found in DB."}

        if not refresh:
            records = db.query(InsiderRoster).filter(InsiderRoster.company_id == company.id).all()
            if records:
                logger.debug(f"Loaded {len(records)} insider roster entries from DB for {ticker}.")
                df = pd.DataFrame([object_as_dict(r) for r in records])
                return df.drop(columns=['id', 'company_id'])

        logger.info(f"Downloading insider roster for {ticker}")
        company_data = yf.Ticker(ticker)
        if company_data:
            data = company_data.insider_roster_holders
            if data is not None and not data.empty:
                save_insider_roster_to_db(db, company.id, data)
                return data
            else:
                return pd.DataFrame()
        return {"error": f"Could not retrieve ticker data for {ticker}"}
    finally:
        db.close()

@yfinance_retry_handler()
def _get_analyst_df_data(ticker: str, df_name: str, model_class, save_fn, refresh: bool = False) -> Union[pd.DataFrame, dict]:
    """
    Generic helper to retrieve various analyst estimate DataFrames, prioritizing the database.
    """
    db = next(get_db())
    try:
        company = db.query(Company).filter(Company.symbol == ticker).first()
        if not company:
            return {"error": f"Company {ticker} not found in DB."}

        if not refresh:
            records = db.query(model_class).filter(model_class.company_id == company.id).all()
            if records:
                logger.debug(f"Loaded {len(records)} {df_name} records from DB for {ticker}.")
                df = pd.DataFrame([object_as_dict(r) for r in records])
                # Set index based on what yfinance would do, for consistency
                if 'period_label' in df.columns:
                    df = df.set_index('period_label')
                elif 'report_date' in df.columns:
                    df['report_date'] = pd.to_datetime(df['report_date'])
                    df = df.set_index('report_date')
                return df.drop(columns=['id', 'company_id', 'last_updated'], errors='ignore')

        logger.info(f"Downloading {df_name} for {ticker}")
        company_data = yf.Ticker(ticker)
        if not company_data:
            return {"error": f"Could not retrieve ticker data for {ticker}"}

        data_attr = getattr(company_data, df_name, None)
        if data_attr is None or data_attr.empty:
            return pd.DataFrame()

        # Call the specific save function
        save_fn(db, company.id, data_attr)
        
        return data_attr
    finally:
        db.close()

def get_analyst_earnings_estimates(ticker, refresh: bool = False):
    """Retrieves analyst earnings estimates for a ticker."""
    return _get_analyst_df_data(ticker, 'earnings_estimate', AnalystEarningsEstimate, save_analyst_earnings_estimates_to_db, refresh)

def get_analyst_revenue_estimates(ticker, refresh: bool = False):
    """Retrieves analyst revenue estimates for a ticker."""
    return _get_analyst_df_data(ticker, 'revenue_estimate', AnalystRevenueEstimate, save_analyst_revenue_estimates_to_db, refresh)

def get_analyst_growth_estimates(ticker, refresh: bool = False):
    """Retrieves analyst growth estimates for a ticker."""
    return _get_analyst_df_data(ticker, 'growth_estimates', AnalystGrowthEstimate, save_analyst_growth_estimates_to_db, refresh)

def get_analyst_earnings_history(ticker, refresh: bool = False):
    """Retrieves analyst earnings history for a ticker."""
    return _get_analyst_df_data(ticker, 'earnings_history', AnalystEarningsHistory, save_analyst_earnings_history_to_db, refresh)

def get_analyst_eps_trend(ticker, refresh: bool = False):
    """Retrieves analyst EPS trend for a ticker."""
    eps_trend_mapping = {"current_estimate": "current", "seven_days_ago": "7daysAgo", 
                         "thirty_days_ago": "30daysAgo", "sixty_days_ago": "60daysAgo", 
                         "ninety_days_ago": "90daysAgo"}
    save_fn = lambda db, cid, df: save_analyst_trend_or_revisions_to_db(db, cid, df, AnalystEpsTrend, eps_trend_mapping, "analyst_eps_trend")
    return _get_analyst_df_data(ticker, 'eps_trend', AnalystEpsTrend, save_fn, refresh)

def get_analyst_eps_revisions(ticker, refresh: bool = False):
    """Retrieves analyst EPS revisions for a ticker."""
    eps_revisions_mapping = {"up_last_7_days": "upLast7days", "up_last_30_days": "upLast30days", 
                             "down_last_7_days": "downLast7Days", "down_last_30_days": "downLast30days"}
    save_fn = lambda db, cid, df: save_analyst_trend_or_revisions_to_db(db, cid, df, AnalystEpsRevisions, eps_revisions_mapping, "analyst_eps_revisions")
    return _get_analyst_df_data(ticker, 'eps_revisions', AnalystEpsRevisions, save_fn, refresh)

@yfinance_retry_handler()
def get_analyst_recommendations_summary(ticker: str, refresh: bool = False) -> Union[pd.DataFrame, dict]:
    """Retrieves analyst recommendations summary, prioritizing the database."""
    db = next(get_db())
    try:
        company = db.query(Company).filter(Company.symbol == ticker).first()
        if not company:
            return {"error": f"Company {ticker} not found in DB."}

        # The most important recommendation data is in the Company table.
        # We can construct a summary DataFrame from this.
        if not refresh and company.recommendationkey:
            logger.debug(f"Loaded recommendations summary from DB for {ticker}.")
            summary_data = {
                'period': ['0m'],
                'strongBuy': [None], 'buy': [None], 'hold': [None], 'sell': [None], 'strongSell': [None],
                'rating': [company.recommendationkey]
            }
            return pd.DataFrame(summary_data)

        logger.info(f"Fetching analyst recommendations summary from yfinance for {ticker}")
        company_data = yf.Ticker(ticker)
        if company_data:
            data = company_data.recommendations_summary
            # The save_company_to_db function already saves the relevant fields.
            # This function now primarily serves to get the full breakdown if needed.
            return data if data is not None else pd.DataFrame()
        return {"error": "Could not retrieve ticker data."}
    finally:
        db.close()

@yfinance_retry_handler()
def get_earnings_dates(ticker: str) -> Union[pd.DataFrame, dict]:
    """Retrieves earnings dates for a ticker."""
    logger.info(f"Downloading earnings dates for {ticker}")
    company_data = yf.Ticker(ticker)
    if company_data:
        return company_data.get_earnings_dates(limit=24)
    return {"error": "Could not retrieve ticker data."}

def _get_tickers_for_update(db: Session, market: str, exchange: str, ticker_file: str, existing_tickers_action: str) -> list:
    """Helper to get the list of tickers to process based on the specified action."""
    if existing_tickers_action == "only":
        logger.info(f"Fetching existing tickers from DB for market '{market}' and exchange '{exchange or 'any'}'.")
        query = db.query(Company.symbol).filter(Company.isactive == True)
        if exchange:
            query = query.filter(Company.exchange == exchange)
        else:
            query = query.filter(Company.exchange.in_(db.query(Exchange.exchange_code).filter(Exchange.country_code == market)))
        return [c[0] for c in query.all()]
    else:
        all_tickers = get_all_tickers_in_market(market=market, exchange=exchange, ticker_file=ticker_file)
        if isinstance(all_tickers, dict) and "error" in all_tickers:
            logger.error(f"Error retrieving tickers: {all_tickers['error']}")
            return []
        logger.info(f"Found {len(all_tickers)} total tickers in source for market '{market}'.")

        if existing_tickers_action == "skip":
            existing_db_tickers = {c[0] for c in db.query(Company.symbol).filter(Company.isactive == True).all()}
            tickers_to_process = [t for t in all_tickers if t not in existing_db_tickers]
            logger.info(f"Skipped {len(existing_db_tickers)} existing tickers. Remaining: {len(tickers_to_process)}")
            return tickers_to_process
        return all_tickers

def _process_company_info_batches(db: Session, tickers: list, batch_size: int, quote_types: list) -> list:
    """Downloads and saves company info for a list of tickers."""
    processed_tickers = []
    total_tickers = len(tickers)
    for i in range(0, len(tickers), batch_size):
        sub_batch = tickers[i:i + batch_size]
        logger.info(f"Processing company info batch {i // batch_size + 1}/{total_tickers // batch_size + 1} (tickers {i} to {min(i + batch_size, total_tickers)})")
        company_info_batch = get_company_info_batch(sub_batch)
        for ticker, company_data in company_info_batch.items():
            if company_data and company_data[1] and (not quote_types or company_data[1].get("quoteType") in quote_types):
                company = save_company_to_db(db, company_data[1])
                if company and company.quotetype in ["EQUITY", "ETF"]:
                    save_extra_company_data_to_db(db, company.id, company_data[0], company_data[1])
                processed_tickers.append(ticker)
    return processed_tickers

def _process_price_history_batches(db: Session, tickers: list, batch_size: int, start_date: str, end_date: str, interval: str):
    """Downloads and saves price history for a list of tickers."""
    total_tickers = len(tickers)
    inactive_tickers = []
    for i in range(0, total_tickers, batch_size):
        batch = tickers[i:i + batch_size]
        logger.info(f"Processing share price batch {i // batch_size + 1}/{total_tickers // batch_size + 1} (tickers {i} to {min(i + batch_size, total_tickers)})")
        
        share_price_data = download_multiple_tickers_data(batch, start_date, end_date, interval)
        
        # --- Track found tickers to correctly identify inactive ones ---
        found_tickers_in_batch = set()

        if share_price_data is not None and not share_price_data.empty:
            batch_price_data = {}
            for ticker in batch:
                company = db.query(Company).filter(Company.symbol == ticker).first()
                if not company: continue

                # --- Robustly find columns for the current ticker ---
                # This handles cases where a ticker in the batch might not have returned any data.
                # The prefix is always "TICKER-", so we check for that.
                ticker_prefix = f"{ticker}-"
                ticker_columns = [c for c in share_price_data.columns if c.startswith(ticker_prefix)]

                if ticker_columns:
                    price_data = share_price_data[ticker_columns].copy()
                    # Robustly remove the ticker prefix, which might be "TICKER-" or just "TICKER".
                    # This handles both US stocks (e.g., "AAPL-Open") and international stocks (e.g., "ZUP.TO-Open").
                    prefix_to_remove = f"{ticker}-"
                    price_data.columns = [col.replace(prefix_to_remove, "") for col in price_data.columns]
                    
                    # --- Only mark as found if there is actual data after dropping NaNs ---
                    # This ensures that tickers with all-NaN rows (due to concat alignment) are treated as missing.
                    valid_price_data = price_data.dropna()
                    if not valid_price_data.empty:
                        # Check if the data is up-to-date relative to the requested end date
                        is_up_to_date = True
                        last_date = valid_price_data.index.max()
                        if last_date.tzinfo:
                            last_date = last_date.tz_localize(None)
                        
                        target_date = datetime.now()
                        if end_date:
                            target_date = datetime.strptime(end_date, '%Y-%m-%d')
                        
                        # Use business days to account for weekends and holidays.
                        # np.busday_count handles weekends automatically.
                        try:
                            days_diff = np.busday_count(last_date.date().isoformat(), target_date.date().isoformat())
                        except Exception:
                            days_diff = (target_date - last_date).days

                        # Allow for 3 missing business days (covers long weekends + 1-2 extra holidays)
                        if days_diff > 3:
                            is_up_to_date = False
                            logger.warning(f"[{ticker}] Data appears stale (Last: {last_date.date()}, Target: {target_date.date()}). Marking as inactive.")

                        if is_up_to_date:
                            found_tickers_in_batch.add(ticker)
                        
                        batch_price_data[company.id] = price_data
            
            if batch_price_data:
                save_or_update_batch_price_data(db, batch_price_data, timeframe=interval)
                
                # --- After saving, check if any of the updated tickers in THIS BATCH had a split ---
                if start_date != FULL_HISTORY_START_DATE:
                    # Use the IDs of companies we just successfully updated
                    company_ids_in_batch = list(batch_price_data.keys())
                    
                    start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
                    tickers_to_refresh = find_tickers_with_splits_in_db(db, company_ids_in_batch, start_date_dt)
                    
                    if tickers_to_refresh:
                        logger.warning(f"Newly downloaded data contains splits for: {[t.symbol for t in tickers_to_refresh]}. Triggering full history refresh.")
                        ticker_map = {t.symbol: t.id for t in tickers_to_refresh}
                        refresh_split_tickers(db, ticker_map, batch_size)
        
        # --- Mark tickers as inactive if they returned no data ---
        batch_inactive_tickers = []
        for ticker in batch:
            if ticker not in found_tickers_in_batch:
                batch_inactive_tickers.append(ticker)
                inactive_tickers.append(ticker)

        if batch_inactive_tickers:
            logger.info(f"No data returned for {len(batch_inactive_tickers)} tickers in this batch. Marking as inactive: {batch_inactive_tickers}")
            db.query(Company).filter(Company.symbol.in_(batch_inactive_tickers)).update({"isactive": False}, synchronize_session=False)
            db.commit()

    if inactive_tickers:
        logger.info(f"Total inactive tickers identified in this run: {len(inactive_tickers)}")

def save_or_update_company_data(market="us", exchange=None, quote_types=["EQUITY", "ETF", "CRYPTOCURRENCY"], ticker_file="yhallsym.json", ticker_list: list = None, interval="1d",
        batch_size=50, start_date=FULL_HISTORY_START_DATE, end_date=None, existing_tickers_action='skip', update_prices_action='yes', **kwargs):
    """
    Loads tickers, downloads company data in batches, and saves or updates data in the database for specified quote types.

    Args:
        market: The market to filter tickers by (e.g., "us", "jp").
        exchange: The exchange to filter tickers by (e.g., "NAS", "NYQ").
        quote_types: A list of quote types to include (e.g., ["EQUITY", "ETF"]).
        ticker_file: The file containing the list of tickers.
        ticker_list: A direct list of tickers to process, which overrides market/file discovery.
        interval: The price data interval to download (e.g., "1d", "1h").
        batch_size: The number of tickers to process in each batch.
        start_date: Start date for historical data download (if required). If None, default values are used.
        end_date: End date for historical data download (if required). If None, default values are used.
        existing_tickers_action: 
        - 'skip': Skip tickers that already exist in the database.
        - 'only': Only process tickers that already exist in the database.
        - 'all': Process all tickers, regardless of whether they exist in the database.
        update_prices_action: 
        - 'yes': Update price history using the given start and end date.
        - 'no': Don't update price history.
        - 'only': Only update price history using the given start and end date.
        - 'last_day': Only update price history from the last day.

    Returns:
        None
    """
    db = next(get_db())
    try:
        # Step 1: Determine which tickers to process
        if ticker_list:
            logger.info(f"Processing a direct list of {len(ticker_list)} tickers.")
            tickers_to_process = ticker_list
        else:
            tickers_to_process = _get_tickers_for_update(db, market, exchange, ticker_file, existing_tickers_action)

        if not tickers_to_process:
            logger.info("No tickers to process based on the specified criteria.")
            return

        # Step 2: Process company info if not 'only' prices
        if existing_tickers_action != 'only' or update_prices_action in ['yes', 'no']:
            tickers_to_process = _process_company_info_batches(db, tickers_to_process, batch_size, quote_types)
            logger.info(f"Found {len(tickers_to_process)} tickers matching quote types {quote_types if quote_types else 'any'} to process.")

        # Step 3: Process price history, but only if requested.
        if update_prices_action != "no":
            if update_prices_action == 'last_day':
                # Download the last day's data
                start_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            
            # If end_date is today or in the future, set it to None to ensure today's data is included.
            # yfinance's end_date is exclusive.
            if end_date and datetime.strptime(end_date, '%Y-%m-%d').date() >= datetime.now().date():
                logger.info("end_date is today or later. Setting to None to include today's price data.")
                end_date = None

            _process_price_history_batches(db, tickers_to_process, batch_size, start_date, end_date, interval)

    except Exception as e:
        logger.error(f"An error occurred during the data update process: {e}", exc_info=True)
    finally:
        db.close()

def find_tickers_with_splits_in_db(db: Session, company_ids: list, start_date: datetime) -> list:
    """
    Queries the database to find companies from a given list that have had a stock split
    since a specified start date.
    """
    if not company_ids:
        return []
    
    return db.query(Company).join(
        PriceHistory, Company.id == PriceHistory.company_id
    ).filter(
        Company.id.in_(company_ids),
        PriceHistory.timestamp >= start_date,
        PriceHistory.split_coefficient != 0,
        PriceHistory.split_coefficient != 1.0
    ).distinct().all()

def refresh_split_tickers(db: Session, ticker_map: dict, batch_size: int):
    """
    Downloads and saves the full price history for a given list of tickers.

    Args:
        db: The database session.
        ticker_map: A dictionary mapping ticker symbols to their company IDs.
        batch_size: The number of tickers to process in each batch.
    """
    tickers_to_process = list(ticker_map.keys())
    total_tickers = len(tickers_to_process)

    for i in range(0, total_tickers, batch_size):
        batch_tickers = tickers_to_process[i:i + batch_size]
        logger.info(f"Refreshing full history for batch {i // batch_size + 1}/{total_tickers // batch_size + 1} (tickers {i} to {min(i + batch_size, total_tickers)})")
        
        # Download full history for the batch
        full_history_data = download_multiple_tickers_data(batch_tickers, start_date=FULL_HISTORY_START_DATE, end_date=None, interval="1d")
        
        # Save the downloaded data
        if full_history_data is not None and not full_history_data.empty:
            batch_data_to_save = {}
            for t in batch_tickers:
                # Select columns for the current ticker
                ticker_cols = [c for c in full_history_data.columns if c.startswith(f"{t}-")]
                if ticker_cols:
                    ticker_df = full_history_data[ticker_cols].copy()
                    ticker_df.columns = [col.replace(f"{t}-", "") for col in ticker_cols]
                    batch_data_to_save[ticker_map[t]] = ticker_df
            save_or_update_batch_price_data(db, batch_data_to_save, timeframe='1d')

def load_ticker_price_data(ticker: str, timeframe: str, start_date: str = None, end_date: str = None, refresh: bool = False) -> dict:
    """
    Loads price data for a given ticker and timeframe, either from the database or by downloading and saving it. It automatically checks for stale data and downloads if necessary.

    Args:
        ticker: The stock ticker symbol.
        timeframe: The desired intraday timeframe (e.g., '1h', '30m', '15m').
        start_date: Start date for historical data (if applicable).
        end_date: End date for historical data (if applicable).
        refresh: If True, forces a re-download regardless of data staleness.

    Returns:
        A dictionary containing the data for the ticker and timeframe, or None if there is an error.
    """
    db = next(get_db())
    try:
        company = db.query(Company).filter(Company.symbol == ticker).first()
        if not company:
            return {"error": f"Company with ticker {ticker} not found in database."}

        # Helper to fetch from DB
        def fetch_from_db():
            query = db.query(PriceHistory).filter(
                PriceHistory.company_id == company.id,
                PriceHistory.timeframe == timeframe
            )
            if start_date:
                query = query.filter(PriceHistory.timestamp >= start_date)
            if end_date:
                query = query.filter(PriceHistory.timestamp <= end_date)
            return query.order_by(PriceHistory.timestamp.asc()).all()

        price_history = fetch_from_db()

        # --- Determine if we need to download new data ---
        should_download = False
        if refresh:
            should_download = True
            logger.info(f"Forcing refresh for {ticker} on {timeframe} timeframe.")
        elif not price_history:
            should_download = True
            logger.info(f"No existing {timeframe} data for {ticker}. Downloading.")
        else:
            # Data exists, check if it's stale
            last_timestamp = price_history[-1].timestamp
            is_stale = False
            if timeframe == '1d':
                # For daily data, check if it's stale based on market close + 2 hours.
                exchange_info = db.query(Exchange).filter(Exchange.exchange_code == company.exchange).first()
                calendar_map = {
                    'NYQ': 'NYSE', 'NMS': 'NASDAQ', 'PCX': 'NYSE', 'NGM': 'NASDAQ', 'NCM': 'NASDAQ', 'BTS': 'NYSE', 'ASE': 'NYSE',
                    'TOR': 'TSX', 'VAN': 'TSXV', 'LSE': 'LSE', 'LON': 'LSE', 'PAR': 'EUREX', 'FRA': 'EUREX', 'GER': 'EUREX',
                    'HKG': 'HKEX', 'TYO': 'JPX', 'ASX': 'ASX', 'SSE': 'SSE',
                }
                calendar_name = calendar_map.get(company.exchange)

                if calendar_name:
                    try:
                        market_cal = mcal.get_calendar(calendar_name)
                        market_tz = market_cal.tz
                        now_market_time = datetime.now(market_tz)
                        today_str = now_market_time.strftime('%Y-%m-%d')
                        schedule = market_cal.schedule(start_date=today_str, end_date=today_str)

                        if not schedule.empty:
                            market_close_dt = schedule.iloc[0].market_close
                            safe_download_time = market_close_dt + timedelta(hours=2)
                            if now_market_time > safe_download_time and last_timestamp.date() < now_market_time.date():
                                is_stale = True
                    except Exception as e:
                        logger.warning(f"Could not get market calendar for {calendar_name}. Falling back to simple delta. Error: {e}")
                        if datetime.utcnow() - last_timestamp > timedelta(days=1): is_stale = True
                else: # Fallback for exchanges not in our map
                    if datetime.utcnow() - last_timestamp > timedelta(days=1): is_stale = True
            else: # For intraday timeframes
                if timeframe.endswith('m'): delta = timedelta(minutes=int(timeframe[:-1]))
                elif timeframe.endswith('h'): delta = timedelta(hours=int(timeframe[:-1]))
                if datetime.utcnow() - last_timestamp > delta: is_stale = True

            if is_stale:
                is_requesting_recent_data = not end_date or datetime.strptime(end_date, '%Y-%m-%d').date() >= datetime.utcnow().date()
                if is_requesting_recent_data:
                    should_download = True
                    logger.info(f"Data for {ticker} {timeframe} is stale (last bar: {last_timestamp}). Triggering update.")

        if should_download:
            logger.info(f"Downloading {timeframe} data for {ticker}...")
            yf_ticker = yf.Ticker(ticker)
            
            # yfinance has limits on how far back intraday data can be fetched.
            # 'max' for 1h is 730d. For minute data, it's shorter (e.g., 60d for 1m-30m).
            # We use a safe period that covers most use cases without being excessive.
            period_for_yf = '730d' if 'h' in timeframe else '60d'
            
            try:
                new_price_history = yf_ticker.history(period=period_for_yf, interval=timeframe, auto_adjust=False)
                if not new_price_history.empty:
                    # Handle MultiIndex columns if present
                    if isinstance(new_price_history.columns, pd.MultiIndex):
                        new_price_history.columns = [col[0] for col in new_price_history.columns]
                    
                    save_or_update_batch_price_data(db, {company.id: new_price_history}, timeframe=timeframe)
                    # Re-fetch from DB to get the potentially updated data
                    price_history = fetch_from_db()
                else:
                    logger.warning(f"No new {timeframe} data returned from yfinance for {ticker}. Market may be closed or data unavailable.")
            except Exception as e:
                logger.error(f"Failed to download {timeframe} data for {ticker}: {e}")

        if price_history:
            df = pd.DataFrame([{"Date": record.timestamp, "Open": record.open, "High": record.high, "Low": record.low, "Close": record.close, "Adj Close": record.adjclose, "Volume": record.volume} for record in price_history])
            df = df.set_index("Date")
            return {'shareprices': df}
        else:
             return {'shareprices': pd.DataFrame()}
    finally:
        db.close()

@ttl_cache(ttl_seconds=28800) # Cache for 8 hours (8 * 60 * 60)
def download_risk_free_rate() -> float:
    """
    Fetches the annualized risk-free rate using yfinance (^IRX).

    The result is cached with a time-to-live (TTL) to avoid excessive API calls
    for a value that changes infrequently.
    """
    logger.info("Fetching risk-free rate from yfinance...")
    try:
        # ^IRX is the 13-week T-bill, a common proxy for the short-term RFR.
        # Fetching a short period to get the latest closing yield.
        ticker_data = yf.download('^IRX', period="5d", progress=False)
        risk_free_rate_percent = ticker_data['Close'].iloc[-1].iloc[0]
    except Exception as e:
        logger.warning(f"Could not fetch risk-free rate from yfinance: {e}. Using default of 2.0.")
        return 2.0 # Use a reasonable default estimate if fetching fails
    return float(risk_free_rate_percent)
  

def get_all_tickers_in_market(market: str = None, exchange: str = None, ticker_file: str = "yhallsym.json") -> Union[list, dict]:
    """
    Loads tickers from a file and filters them by market or exchange.

    This function can read from CSV, JSON, or TXT files. For accurate filtering,
    it is recommended to use a CSV file with 'symbol' and 'exchange' columns.
    When using TXT or JSON, filtering relies on ticker suffixes, which is less precise
    for US-based exchanges.

    Args:
        market: The market to filter by (e.g., "us", "ca").
        exchange: The exchange code to filter by (e.g., "NYQ", "TOR").
        ticker_file: The name of the ticker file in the 'yfinance_symbols' directory.

    Returns:
        A list of ticker symbols that match the criteria, or a dict with an error message.
    """
    # Construct a robust path to the 'yfinance_symbols' directory, assuming it's at the project root.
    # The script is in 'tools/', so we go up one level to find the project root.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(project_root, "yfinance_symbols", ticker_file)

    try:
        tickers_source = []
        if ticker_file.lower().endswith(".csv"):
            df = pd.read_csv(file_path)
            for index, row in df.iterrows():
                tickers_source.append((row["symbol"], row["exchange"]))
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                tickers_source = list(data.keys())

        if not (market or exchange):
            return [t[0] if isinstance(t, tuple) else t for t in tickers_source]

        logger.info(f"Filtering tickers for market='{market}' and exchange='{exchange}'")
        filtered_tickers = []
        for item in tickers_source:
            ticker_symbol, ticker_exchange_code, ticker_market_code = "", None, None

            if isinstance(item, tuple):
                ticker_symbol, ticker_exchange_code = item
                if ticker_exchange_code:
                    ticker_market_code = EXCHANGE_MARKET_MAP.get(ticker_exchange_code)
            else:
                ticker_symbol = item
                if ticker_symbol.startswith("^"): continue
                for suffix, exch in EXCHANGE_SUFFIX_MAP.items():
                    if ticker_symbol.endswith(suffix):
                        ticker_exchange_code = exch
                        ticker_market_code = EXCHANGE_MARKET_MAP.get(exch)
                        break
                if not ticker_exchange_code:
                    ticker_market_code = "us"

            if market and ticker_market_code != market: continue
            if exchange and ticker_exchange_code != exchange: continue
            
            filtered_tickers.append(ticker_symbol)
        return filtered_tickers

    except FileNotFoundError:
        logger.error(f"Ticker file not found: {file_path}", exc_info=True)
        return {"error": f"Ticker file not found: {file_path}"}  
    except Exception as e:
        logger.error(f"An unexpected error occurred in get_all_tickers_in_market: {e}", exc_info=True)
        return {"error": str(e)}

# Mapping from database sector names to their corresponding SPDR Select Sector ETFs.
# The secondary benchmark is always the S&P 500.
PRIMARY_BENCHMARK = '^SPX'

SECTOR_TO_ETF_MAP = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Consumer Cyclical": "XLY",      # Maps to Consumer Discretionary
    "Consumer Defensive": "XLP",     # Maps to Consumer Staples
    "Utilities": "XLU",
    "Basic Materials": "XLB",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}

BENCHMARK_SYMBOLS = [PRIMARY_BENCHMARK] + list(SECTOR_TO_ETF_MAP.values())

def get_benchmark_ticker_for_asset(ticker: str) -> tuple[str, str]:
    """
    Dynamically determines the most appropriate market benchmark ETF for a given stock ticker
    by looking up its sector. It provides a primary (sector-specific) and a secondary
    (broad market) benchmark for fallback.

    Args:
        ticker: The stock ticker symbol.

    Returns:
        A tuple containing the primary benchmark ticker and the secondary benchmark ticker.
    """
    primary_benchmark = secondary_benchmark = PRIMARY_BENCHMARK
    db_session = next(get_db())
    try:
        company = db_session.query(Company).filter(Company.symbol == ticker).first()
        if company and company.sector in SECTOR_TO_ETF_MAP:
            # If a sector ETF is found, it becomes the primary benchmark.
            primary_benchmark = SECTOR_TO_ETF_MAP[company.sector]
    finally:
        db_session.close()
    
    return primary_benchmark, secondary_benchmark


# Create a dictionary to map ticker suffixes to exchanges (expand as needed)
EXCHANGE_SUFFIX_MAP = {
    ".TO": "TOR",  # Toronto Stock Exchange
    ".V": "VAN",   # TSX Venture Exchange
    ".CN": "CSE0",  # Canadian Securities Exchange
    ".NE": "NEO",  # NEO Exchange
    ".SA": "BSP",  # Sao Paolo Stock Exchange
    ".SN": "SAN",  # Santiago Stock Exchange
    ".CR": "CCS",  # Caracas Stock Exchange
    ".BA": "BUE",  # Buenos Aires Stock Exchange
    ".MX": "MEX",  # Mexico Stock Exchange
    ".NZ": "NZX",  # New Zealand Stock Exchange
    ".AX": "ASX",  # Australian Stock Exchange
    ".S": "SPK",   # Sapporo Stock Exchange  (Note: You'll rarely see this suffix)
    ".T": "TYO",   # Tokyo Stock Exchange   (Note: You'll rarely see this suffix)
    ".KQ": "KOQ",  # KOSDAQ
    ".KS": "KOR",  # Korea Stock Exchange
    ".TW": "TWN",  # Taiwan Stock Exchange
    ".TWO": "TWO", # Taiwan OTC Exchange
    ".SI": "SGX",  # Singapore Stock Exchange (Use SGX, more common)
    ".KL": "KLS",  # Kuala Lumpur Stock Exchange
    ".SS": "SSE",  # Shanghai Stock Exchange
    ".SZ": "SZSE", # Shenzhen Stock Exchange
    ".HK": "HKG",  # Hong Kong Stock Exchange
    ".JK": "IDX",  # Indonesia Stock Exchange
    ".BK": "BKK",  # Stock Exchange of Thailand
    ".BO": "BSE",  # India Bombay Stock Exchange
    ".NS": "NSE",  # India National Stock Exchange
    ".CM": "CSE",  # Colombo Stock Exchange
    ".QA": "DOH",  # Qatar Stock Exchange
    ".JO": "JNB",  # Johannesburg Stock Exchange
    ".TA": "TLV",  # Tel Aviv stock exchange
    ".ME": "MCX",  # Moscow Exchange
    ".SR": "SAU",  # Saudi Stock Exchange
    ".IS": "IST",  # Borsa Istanbul
    ".CA": "EGX",  # Egyptian Stock Exchange
    ".VI": "VIE",  # Vienna Stock Exchange
    ".BE": "BER",  # Berlin Stock Exchange  (Ensure difference with Belgium!)
    ".DE": "GER",  # Xetra Stock Exchange  (Using GER for Xetra and avoiding .DE, which is ambiguous)
    ".DU": "DUS",  # Dusseldorf Stock Exchange
    ".F": "FRA",   # Frankfurt Stock Exchange
    ".HM": "HAM",  # Hamburg Stock Exchange
    ".HA": "HAN",  # Hanover Stock Exchange
    ".MU": "MUN",  # Munich Stock Exchange
    ".SG": "STU",  # Euronext Stuttgart
    ".BR": "BRU",  # Euronext Brussels
    ".PR": "PRG",  # Prague Stock Exchange
    ".CO": "CPH",  # Nasdaq OMX Copenhagen
    ".TL": "TAL",  # Nasdaq OMX Tallinn
    ".HE": "HEL",  # Nasdaq OMX Helsinki
    ".PA": "PAR",  # Euronext Paris
    ".BD": "BUD",  # Budapest Stock Exchange
    ".IR": "ISE",  # Euronext Dublin  (Using ISE for Irish Stock Exchange. .IR conflicts with other symbols)
    ".TI": "ETL",  # EuroTLX 
    ".MI": "MIL",  # Borsa Italiana
    ".RG": "RIG",  # Nasdaq OMX Riga
    ".VS": "VNO",  # Nasdaq OMX Vilnius
    ".AS": "AMS",  # Euronext Amsterdam
    ".OL": "OSL",  # Oslo Stock Exchange
    ".LS": "LIS",  # Euronext Lisbon
    ".MC": "MCE",  # Bolsas y Mercados EspaÃ±oles
    ".ST": "STO",  # Nasdaq OMX Stockholm
    ".SW": "SWX",  # SIX Swiss Exchange
    ".Z": "SWX",  # SIX Swiss Exchange
    ".L": "LON",   # London Stock Exchange
    ".IL": "IOB",  # International Order Book (using IOB)
    ".AT": "ATH",  # Athens Stock Exchange (Use ATH for Athens to distinguish from Austria)
    ".IC": "ICE",  # Nasdaq OMX Iceland
}

# Create a dictionary to map exchanges to markets
EXCHANGE_MARKET_MAP = {  # Expand as needed
    # US Exchanges
    "NMS": "us",   # NASDAQ
    "NYQ": "us",   # NYSE
    "PCX": "us",   # NYSE Arca
    "ASE": "us",   # NYSE American
    # Canada
    "TOR": "ca",   # Toronto Stock Exchange -> Canada
    "VAN": "ca",   # TSX Venture Exchange  -> Canada
    "CSE0": "ca",  # Canadian Securities Exchange -> Canada
    "NEO": "ca",   # NEO Exchange -> Canada
    # Others
    "BSP": "br",   # Sao Paolo Stock Exchange -> Brazil
    "SAN": "cl",   # Santiago Stock Exchange -> Chile
    "CCS": "ve",   # Caracas Stock Exchange -> Venezuela
    "BUE": "ar",   # Buenos Aires Stock Exchange -> Argentina
    "MEX": "mx",   # Mexico Stock Exchange -> Mexico
    "NZX": "nz",   # New Zealand Stock Exchange -> New Zealand
    "ASX": "au",   # Australian Stock Exchange  -> Australia
    "SPK": "jp",   # Sapporo Stock Exchange -> Japan (Rarely used)
    "TYO": "jp",   # Tokyo Stock Exchange   -> Japan (Rarely used)
    "KOQ": "kr",   # KOSDAQ -> South Korea
    "KOR": "kr",   # Korea Stock Exchange -> South Korea
    "TWN": "tw",   # Taiwan Stock Exchange -> Taiwan
    "TWO": "tw",   # Taiwan OTC Exchange
    "SGX": "sg",   # Singapore Stock Exchange -> Singapore
    "KLS": "my",   # Kuala Lumpur Stock Exchange  -> Malaysia
    "SSE": "cn",   # Shanghai Stock Exchange -> China
    "SZSE": "cn",  # Shenzhen Stock Exchange
    "HKG": "hk",   # Hong Kong Stock Exchange  -> Hong Kong
    "IDX": "id",   # Indonesia Stock Exchange -> Indonesia
    "BKK": "th",   # Stock Exchange of Thailand -> Thailand
    "BSE": "in",   # India Bombay Stock Exchange -> India
    "NSE": "in",   # India National Stock Exchange -> India
    "CSE": "lk",   # Colombo Stock Exchange -> Sri Lanka
    "DOH": "qa",   # Qatar Stock Exchange -> Qatar
    "JNB": "za",   # Johannesburg Stock Exchange -> South Africa
    "TLV": "il",   # Tel Aviv Stock Exchange -> Israel
    "MCX": "ru",   # Moscow Exchange -> Russia
    "SAU": "sa",   # Saudi Stock Exchange -> Saudi Arabia
    "IST": "tr",   # Borsa Istanbul -> Turkey
    "EGX": "eg",   # Egyptian Stock Exchange -> Egypt
    "VIE": "at",   # Vienna Stock Exchange -> Austria
    "BER": "de",   # Berlin Stock Exchange -> Germany
    "GER": "de",   # Xetra Stock Exchange -> Germany
    "DUS": "de",   # Dusseldorf Stock Exchange -> Germany
    "FRA": "de",   # Frankfurt Stock Exchange -> Germany
    "HAM": "de",   # Hamburg Stock Exchange -> Germany
    "HAN": "de",   # Hanover Stock Exchange -> Germany
    "MUN": "de",   # Munich Stock Exchange -> Germany
    "STU": "de",   # Euronext Stuttgart -> Germany
    "BRU": "be",   # Euronext Brussels -> Belgium
    "PRG": "cz",   # Prague Stock Exchange -> Czech Republic
    "CPH": "dk",   # Nasdaq OMX Copenhagen -> Denmark
    "TAL": "ee",   # Nasdaq OMX Tallinn -> Estonia
    "HEL": "fi",   # Nasdaq OMX Helsinki -> Finland
    "PAR": "fr",   # Euronext Paris -> France
    "BUD": "hu",   # Budapest Stock Exchange -> Hungary
    "ISE": "ie",   # Euronext Dublin -> Ireland
    "ETL": "lt",   # EuroTLX -> Lithuania
    "MIL": "it",   # Borsa Italiana -> Italy
    "RIG": "lv",   # Nasdaq OMX Riga -> Latvia
    "VNO": "lt",   # Nasdaq OMX Vilnius -> Lithuania
    "AMS": "nl",   # Euronext Amsterdam -> Netherlands
    "OSL": "no",   # Oslo Stock Exchange -> Norway
    "LIS": "pt",   # Euronext Lisbon -> Portugal
    "MCE": "es",   # Bolsas y Mercados EspaÃ±oles -> Spain
    "STO": "se",   # Nasdaq OMX Stockholm -> Sweden
    "SWX": "ch",   # SIX Swiss Exchange -> Switzerland
    "LON": "gb",   # London Stock Exchange -> United Kingdom
    "IOB": "gb",   # International Order Book -> United Kingdom
    "ATH": "gr",   # Athens Stock Exchange -> Greece
    "ICE": "is",   # Nasdaq OMX Iceland -> Iceland
}
