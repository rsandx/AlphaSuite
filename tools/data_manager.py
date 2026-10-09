import json
import logging
import os
import time
import random
from typing import Optional, Union, List, Dict, Any
import numpy as np
import pandas as pd
from functools import wraps
from datetime import datetime, timedelta

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sqlalchemy import BigInteger, Float, Integer
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.inspection import inspect

from core.db import get_db
from core.data_providers.factory import DataProviderFactory
from core.model import (
    AnalystEarningsEstimate, AnalystEarningsHistory, AnalystEpsRevisions, 
    AnalystEpsTrend, AnalystGrowthEstimate, AnalystRevenueEstimate, Company, 
    CompanyOfficer, Financials, InsiderRoster, InsiderTransaction, 
    InstitutionalHolding, PriceHistory, UpgradeDowngrade, object_as_dict
)


# Constants moved from yfinance_tool
FULL_HISTORY_START_DATE = "2000-01-01"
EXCHANGE_SUFFIX_MAP = {
    ".TO": "TOR", ".V": "VAN", ".CN": "CSE0", ".NE": "NEO", ".SA": "BSP", ".SN": "SAN",
    ".CR": "CCS", ".BA": "BUE", ".MX": "MEX", ".NZ": "NZX", ".AX": "ASX", ".S": "SPK",
    ".T": "TYO", ".KQ": "KOQ", ".KS": "KOR", ".TW": "TWN", ".TWO": "TWO", ".SI": "SGX",
    ".KL": "KLS", ".SS": "SSE", ".SZ": "SZSE", ".HK": "HKG", ".JK": "IDX", ".BK": "BKK",
    ".BO": "BSE", ".NS": "NSE", ".CM": "CSE", ".QA": "DOH", ".JO": "JNB", ".TA": "TLV",
    ".ME": "MCX", ".SR": "SAU", ".IS": "IST", ".CA": "EGX", ".VI": "VIE", ".BE": "BER",
    ".DE": "GER", ".DU": "DUS", ".F": "FRA", ".HM": "HAM", ".HA": "HAN", ".MU": "MUN",
    ".SG": "STU", ".BR": "BRU", ".PR": "PRG", ".CO": "CPH", ".TL": "TAL", ".HE": "HEL",
    ".PA": "PAR", ".BD": "BUD", ".IR": "ISE", ".TI": "ETL", ".MI": "MIL", ".RG": "RIG",
    ".VS": "VNO", ".AS": "AMS", ".OL": "OSL", ".LS": "LIS", ".MC": "MCE", ".ST": "STO",
    ".SW": "SWX", ".Z": "SWX", ".L": "LON", ".IOB": "IOB", ".ATH": "ATH", ".ICE": "ICE",
}

EXCHANGE_MARKET_MAP = {
    "NMS": "us", "NYQ": "us", "PCX": "us", "ASE": "us", "TOR": "ca", "VAN": "ca",
    "CSE0": "ca", "NEO": "ca", "BSP": "br", "SAN": "cl", "CCS": "ve", "BUE": "ar",
    "MEX": "mx", "NZX": "nz", "ASX": "au", "SPK": "jp", "TYO": "jp", "KOQ": "kr",
    "KOR": "kr", "TWN": "tw", "TWO": "tw", "SGX": "sg", "KLS": "my", "SSE": "cn",
    "SZSE": "cn", "HKG": "hk", "IDX": "id", "BKK": "th", "BSE": "in", "NSE": "in",
    "CSE": "lk", "DOH": "qa", "JNB": "za", "TLV": "il", "MCX": "ru", "SAU": "sa",
    "IST": "tr", "EGX": "eg", "VIE": "at", "BER": "de", "GER": "de", "DUS": "de",
    "FRA": "de", "HAM": "de", "HAN": "de", "MUN": "mun", "STU": "stu", "BRU": "bru",
    "PRG": "prg", "CPH": "cph", "TAL": "tal", "HEL": "hel", "PAR": "par", "BUD": "bud",
    "ISE": "ise", "ETL": "etl", "MIL": "mil", "RIG": "rig", "VNO": "vno", "AMS": "ams",
    "OSL": "osl", "LIS": "lis", "MCE": "mce", "STO": "sto", "SWX": "swx", "Z": "swx",
    "LON": "lon", "IOB": "iob", "ATH": "ath", "ICE": "ice",
}

# --- SECTOR NORMALIZATION (GICS Standard) ---
SECTOR_MAP = {
    "technology": "Information Technology",
    "information technology": "Information Technology",
    "financial services": "Financials",
    "financials": "Financials",
    "healthcare": "Health Care",
    "health care": "Health Care",
    "consumer cyclical": "Consumer Discretionary",
    "consumer discretionary": "Consumer Discretionary",
    "consumer defensive": "Consumer Staples",
    "consumer staples": "Consumer Staples",
    "communication services": "Communication Services",
    "telecommunications": "Communication Services",
    "industrials": "Industrials",
    "basic materials": "Materials",
    "materials": "Materials",
    "energy": "Energy",
    "utilities": "Utilities",
    "real estate": "Real Estate",
}

# --- COUNTRY NORMALIZATION (ISO 3166-1 Alpha-2) ---
COUNTRY_MAP = {
    "united states": "US",
    "united states of america": "US",
    "usa": "US",
    "us": "US",
    "canada": "CA",
    "can": "CA",
    "ca": "CA",
    "united kingdom": "GB",
    "uk": "GB",
    "gbr": "GB",
    "germany": "DE",
    "deu": "DE",
    "japan": "JP",
    "jpn": "JP",
    "australia": "AU",
    "aus": "AU",
}

# --- EXCHANGE ALIASES (Supplementing EXCHANGE_SUFFIX_MAP) ---
# Maps raw provider strings to your standardized internal Exchange Codes (e.g., 'NMS' -> 'NAS')
EXCHANGE_NAME_TO_CODE = {
    # US Exchanges
    "nasdaq": "NMS",
    "nasdaqgs": "NMS",
    "nasdaqgm": "NMS",
    "nasdaqcm": "NMS",
    "nyse": "NYQ",
    "new york stock exchange": "NYQ",
    "nyse american": "ASE",
    "american stock exchange": "ASE",
    "amex": "ASE",
    "pcx": "PCX",
    
    # Canadian Exchanges
    "toronto": "TOR",
    "tsx": "TOR",
    "toronto stock exchange": "TOR",
    "vancouver": "VAN",
    "tsv": "VAN",
    "tsx venture": "VAN",
    "neo": "NEO",
    "neo exchange": "NEO",
    
    # European / Global Exchanges
    "london": "LON",
    "lse": "LON",
    "london stock exchange": "LON",
    "xetra": "GER",
    "frankfurt": "FRA",
    "paris": "PAR",
    "amsterdam": "AMS",
    "tokyo": "TYO",
    "hong kong": "HKG",
    "australian": "ASX",
}

# Raw market string overrides & yfinance cleanups
MARKET_CLEANUP_MAP = {
    "us_market": "us",
    "ca_market": "ca",
    "gb_market": "lon",
    "de_market": "de",
    "jp_market": "jp",
    "au_market": "au",
    "hk_market": "hk",
    "fr_market": "par",
    "nl_market": "ams",
}

# Extend EXCHANGE_MARKET_MAP to cover common raw exchange strings from providers
RAW_EXCHANGE_TO_MARKET_MAP = {
    "NASDAQ": "us", "NMS": "us", "NYQ": "us", "NYSE": "us", 
    "PCX": "us", "ASE": "us", "AMEX": "us", "OTC": "us", "PINK": "us",
    "TOR": "ca", "TSX": "ca", "VAN": "ca", "TSXV": "ca", "CSE0": "ca", "NEO": "ca",
    "LON": "lon", "LSE": "lon", "GER": "de", "FRA": "de", "PAR": "par", 
    "AMS": "ams", "TYO": "jp", "HKG": "hk", "ASX": "au"
}

# Fallback: Infer market from ISO Country Code if exchange lookup yields nothing
COUNTRY_TO_MARKET_MAP = {
    "US": "us",
    "CA": "ca",
    "GB": "lon",
    "DE": "de",
    "JP": "jp",
    "AU": "au",
    "HK": "hk",
    "FR": "par",
    "NL": "ams",
    "CH": "swx",
    "CN": "cn",
    "IN": "in",
}

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

logger = logging.getLogger(__name__)


# --- Decorators for DataManager ---

def data_manager_cache(ttl_seconds: int):
    """
    A TTL cache decorator specifically for DataManager operations.
    Prevents redundant database/API hits for the same ticker within the TTL.
    """
    cache = {}
    def decorator(func):
        @wraps(func)
        def wrapper(self, ticker: str, *args, **kwargs):
            timeframe = kwargs.get('timeframe', '1d')
            key = (ticker, timeframe)
            now = time.time()

            if key in cache and (now - cache[key]['time']) < ttl_seconds:
                logger.info(f"DataManager Cache: Returning cached result for {ticker} ({timeframe})")
                return cache[key]['value']

            result = func(self, ticker, *args, **kwargs)
            
            if not isinstance(result, dict) or "error" in result:
                return result # Don't cache errors

            cache[key] = {'value': result, 'time': now}
            return result
        return wrapper
    return decorator

def data_manager_retry(retries=3, backoff_factor=2):
    """
    A decorator for DataManager to handle orchestration-level failures 
    (e.g., Database locks or transient I/O errors).
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            last_exception = None
            for attempt in range(retries):
                try:
                    return func(self, *args, **kwargs)
                except Exception as e:
                    last_exception = e
                    sleep_time = backoff_factor ** attempt + random.uniform(0, 1)
                    logger.warning(f"DataManager Retry: Attempt {attempt+1} failed for {args}. Retrying in {sleep_time:.2f}s... Error: {e}")
                    time.sleep(sleep_time)
            logger.error(f"DataManager Retry: All {retries} attempts failed.")
            return {"status": "error", "message": f"Retry failed: {str(last_exception)}"}
        return wrapper
    return decorator


class DataManager:
    def __init__(self):
        self.provider = DataProviderFactory.get_provider()

    def resolve_symbol(self, raw_symbol: str) -> str:
        """
        Translates a raw ticker (e.g., 'AAPL') into the 
        canonical format stored in the DB (e.g., 'AAPL.US').
        """
        return self.provider.normalize_to_canonical(raw_symbol)

    def get_company_by_symbol(self, raw_symbol: str) -> Optional[Company]:
        """
        The safe way to find a company.
        """
        canonical = self.resolve_symbol(raw_symbol)
        db = next(get_db())
        try:
            return db.query(Company).filter(Company.symbol == canonical).first()
        finally:
            db.close()

    def resolve_symbols_batch(self, raw_symbols: list[str]) -> list[str]:
        """
        Transforms a list of raw symbols into their canonical DB counterparts.
        """
        if not raw_symbols:
            return []
        return [self.provider.normalize_to_canonical(s) for s in raw_symbols]

    def get_companies_by_symbols(self, raw_symbols: list[str]) -> list[Company]:
        """
        The safe way to perform bulk lookups.
        """
        canonical_symbols = self.resolve_symbols_batch(raw_symbols)
        if not canonical_symbols:
            return []

        db = next(get_db())
        try:
            return db.query(Company).filter(Company.symbol.in_(canonical_symbols)).all()
        finally:
            db.close()

    def _normalize_company_fields(self, info: Dict, symbol: str) -> Dict:
        """
        In-memory normalization of sector, country, exchange, AND market
        supporting both EODHD and yfinance format patterns (e.g. 'us_market').
        """
        normalized = info.copy()

        # 1. Sector Normalization
        raw_sector = str(info.get("sector") or info.get("GicSector") or "").strip().lower()
        if raw_sector in SECTOR_MAP:
            normalized["sector"] = SECTOR_MAP[raw_sector]

        # 2. Country Normalization (ISO Alpha-2)
        raw_country = str(info.get("country") or info.get("CountryName") or "").strip().lower()
        norm_country = COUNTRY_MAP.get(raw_country)
        if norm_country:
            normalized["country"] = norm_country

        # 3. Exchange Normalization
        resolved_exchange_code = None
        
        # Method A: Infer exchange code from symbol suffix
        if symbol:
            for suffix, exch_code in EXCHANGE_SUFFIX_MAP.items():
                if symbol.upper().endswith(suffix):
                    resolved_exchange_code = exch_code
                    break

        # Method B: Lookup raw exchange alias
        if not resolved_exchange_code and info.get("exchange"):
            raw_exch = str(info["exchange"]).strip().lower()
            resolved_exchange_code = EXCHANGE_NAME_TO_CODE.get(raw_exch)

        # Method C: Standard US defaults
        if not resolved_exchange_code:
            raw_exch = str(info.get("exchange") or "").strip().upper()
            if raw_exch in ["NASDAQ", "NMS", "NYQ", "NYSE", "ASE", "AMEX"]:
                resolved_exchange_code = raw_exch
            else:
                resolved_exchange_code = "NMS"

        normalized["exchange"] = resolved_exchange_code

        # 4. Market Normalization (with yfinance 'us_market' handling)
        resolved_market = None
        raw_provider_market = str(info.get("market") or "").strip().lower()

        # Step A: Check if info['market'] contains yfinance pattern (e.g., 'us_market' -> 'us')
        if raw_provider_market in MARKET_CLEANUP_MAP:
            resolved_market = MARKET_CLEANUP_MAP[raw_provider_market]
        elif raw_provider_market.endswith("_market"):
            resolved_market = raw_provider_market.replace("_market", "")

        # Step B: Priority lookup via EXCHANGE_MARKET_MAP
        if not resolved_market and resolved_exchange_code in EXCHANGE_MARKET_MAP:
            resolved_market = EXCHANGE_MARKET_MAP[resolved_exchange_code]

        # Step C: Fallback lookup via raw exchange strings
        if not resolved_market and info.get("exchange"):
            raw_exch_upper = str(info["exchange"]).strip().upper()
            resolved_market = RAW_EXCHANGE_TO_MARKET_MAP.get(raw_exch_upper)

        # Step D: Infer market from ISO country code
        if not resolved_market and normalized.get("country"):
            resolved_market = COUNTRY_TO_MARKET_MAP.get(normalized["country"])

        # Step E: Default fallback
        if not resolved_market:
            resolved_market = "us"

        normalized["market"] = resolved_market.lower()

        return normalized

    @data_manager_cache(ttl_seconds=3600)
    @data_manager_retry(retries=3)
    def refresh_ticker_data(self, ticker: str, timeframe: str = '1d'):
        """
        Refreshes company data and price history from the active data provider.
        """
        canonical_ticker = self.provider.normalize_to_canonical(ticker)
        api_ticker = self.provider.normalize_from_canonical(canonical_ticker)
        
        logger.info(f"DataManager: Refreshing data for {canonical_ticker} using {type(self.provider).__name__}")
        
        db = next(get_db())
        try:
            price_df = self.provider.get_historical_data(api_ticker, interval=timeframe)
            
            if not price_df.empty:
                company_info = self.provider.get_company_info(api_ticker)
                if company_info:
                    company = self._internal_save_company(db, company_info)
                    if company:
                        self.save_or_update_batch_price_data(db, {company.id: price_df}, timeframe=timeframe)
                
                return {"status": "success", "ticker": canonical_ticker}
            else:
                return {"status": "error", "message": "No price data returned from provider"}

        except Exception as e:
            db.rollback()
            logger.error(f"DataManager failed to refresh {canonical_ticker}: {e}")
            raise e
        finally:
            db.close()

    def get_all_tickers_in_market(self, market: str = None, exchange: str = None, ticker_file: str = "yhallsym.json") -> Union[list, dict]:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(project_root, "yfinance_symbols", ticker_file)

        try:
            tickers_source = []
            if ticker_file.lower().endswith(".csv"):
                df = pd.read_csv(file_path)
                for _, row in df.iterrows():
                    tickers_source.append((row["symbol"], row["exchange"]))
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    tickers_source = list(data.keys())

            if not (market or exchange):
                return [t[0] if isinstance(t, tuple) else t for t in tickers_source]

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
        except Exception as e:
            return {"error": str(e)}

    def download_multiple_tickers_data(self, tickers: list, start_date: str = None, end_date: str = None, interval: str = "1d") -> pd.DataFrame:
        all_data = {}
        for ticker in tickers:
            try:
                api_ticker = self.provider.normalize_from_canonical(self.provider.normalize_to_canonical(ticker))
                data = self.provider.get_historical_data(api_ticker, start_date=start_date, end_date=end_date, interval=interval)
                if not data.empty:
                    all_data[ticker] = data
            except Exception as e:
                logger.error(f"Error downloading {ticker}: {e}")
        
        if all_data:
            result = pd.concat(all_data, axis=1, names=["Ticker", "Attributes"])
            result.columns = [f'{col[1]}' if col[0] == "Adj Close" else f'{col[0]}-{col[1]}' for col in result.columns] 
            return result
        return pd.DataFrame()

    def get_company_info_batch(self, tickers: list) -> dict:
        batch = {}
        for ticker in tickers:
            try:
                api_ticker = self.provider.normalize_from_canonical(self.provider.normalize_to_canonical(ticker))
                info = self.provider.get_company_info(api_ticker)
                if info:
                    batch[ticker] = [api_ticker, info]
            except Exception as e:
                logger.error(f"Error fetching info for {ticker}: {e}")
        return batch

    @data_manager_cache(ttl_seconds=3600)
    @data_manager_retry(retries=3)
    def load_ticker_data(self, 
                         ticker: str, 
                         start_date: str = None, 
                         end_date: str = None, 
                         refresh: bool = False, 
                         timeframe: str = '1d', 
                         price_only: bool = True) -> Optional[Dict[str, Any]]:
        """
        The single source of truth for loading data. Completely self-contained.
        """
        db: Session = next(get_db())
        try:
            canonical_ticker = self.provider.normalize_to_canonical(ticker)
            company = db.query(Company).filter(Company.symbol == canonical_ticker).first()
            
            if company and not refresh:
                logger.info(f"Loading {canonical_ticker} from database.")
                return self._build_result_from_db(db, company, timeframe, start_date, end_date, price_only)

            logger.info(f"Fetching and saving data for {canonical_ticker} (Refresh={refresh})")
            
            data_payload = self.provider.get_ticker_data(
                canonical_ticker, 
                start_date=start_date, 
                end_date=end_date, 
                timeframe=timeframe
            )
            
            if "error" in data_payload:
                return {"error": data_payload["error"]}

            return self._persist_payload_to_db(db, canonical_ticker, data_payload, price_only, timeframe)

        except Exception as e:
            db.rollback()
            logger.error(f"DataManager.load_ticker_data error for {ticker}: {e}", exc_info=True)
            return None
        finally:
            db.close()

    def _persist_payload_to_db(self, db: Session, ticker: str, payload: Dict, price_only: bool, timeframe: str) -> Dict:
        """
        Maps the standardized payload to the Database schema within an atomic transaction block.
        """
        result = {}
        try:
            # 1. Handle Company & Fundamentals
            if not price_only and 'info' in payload:
                info_payload = payload['info'].copy()
                
                # Ensure root payload officers are preserved
                if 'officers' in payload and 'companyOfficers' not in info_payload:
                    info_payload['companyOfficers'] = payload['officers']

                company = self._internal_save_company(db, info_payload)
                result['company'] = object_as_dict(company)

                # Financials
                if 'financials' in payload:
                    self._internal_save_financials(db, company.id, payload['financials'])
                    result.update(payload['financials'])
                
                # Analyst Estimates
                if 'analyst_estimates' in payload:
                    self._internal_save_analyst_estimates(db, company.id, payload['analyst_estimates'])
                
                # Insider Data
                if 'insider' in payload:
                    self._internal_save_insider_data(db, company.id, payload['insider'])

                # Institutional/Mutual Fund Holdings
                if 'institutional_holdings' in payload:
                    self._internal_save_institutional_holdings(db, company.id, payload['institutional_holdings'])
                if 'mutualfund_holdings' in payload:
                    self._internal_save_institutional_holdings(db, company.id, payload['mutualfund_holdings'], 'mutualfund')

                # Analyst Sentiment & Price Targets/Recommendations
                sentiment_payload = payload.get('analyst_sentiment', {})
                if 'upgrades_downgrades' in payload:
                    sentiment_payload['upgrades_downgrades'] = payload['upgrades_downgrades']
                elif 'recommendations' in payload:
                    sentiment_payload['upgrades_downgrades'] = payload['recommendations']                    
                if 'price_targets' in payload:
                    sentiment_payload['price_targets'] = payload['price_targets']

                if sentiment_payload:
                    self._internal_save_analyst_sentiment(db, company.id, sentiment_payload)

            # 2. Handle Price History
            if 'price_history' in payload:
                price_df = payload['price_history']
                if not price_df.empty:
                    if not isinstance(price_df.index, pd.DatetimeIndex):
                        price_df.index = pd.to_datetime(price_df.index)
                    
                    if price_df.index.tz is not None:
                        price_df.index = price_df.index.tz_localize(None)
                    
                    price_df = price_df.sort_index(ascending=True)

                    company = db.query(Company).filter(Company.symbol == ticker).first()
                    if company:
                        self.save_price_history_flow(db, {company.id: price_df}, timeframe)
                        
                        df_out = price_df.copy()
                        df_out.columns = [col.lower().replace(' ', '') for col in df_out.columns]
                        result['shareprices'] = df_out

                    if 'shareprices' in result and not result['shareprices'].empty:
                        logger.info(f"{ticker} Daily range: {result['shareprices'].index.min()} to {result['shareprices'].index.max()}")

            db.commit()
            return result
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to persist payload to DB for {ticker}: {e}")
            raise e

    # --- INTERNAL PERSISTENCE METHODS ---

    def _internal_save_company(self, db: Session, info: Dict) -> Company:
        """Standardized Upsert for Company Table with in-memory metadata normalization."""
        
        # 1. Resolve Canonical Symbol
        raw_symbol = info.get("symbol")
        canonical_symbol = self.resolve_symbol(raw_symbol) if raw_symbol else None

        # 2. Normalize metadata fields in-memory
        normalized_info = self._normalize_company_fields(info, canonical_symbol)

        db_ready_info = {}
        mapper = inspect(Company).mapper
        bigint_cols = {c.key for c in mapper.column_attrs if isinstance(c.columns[0].type, BigInteger)}
        float_cols = {c.key for c in mapper.column_attrs if isinstance(c.columns[0].type, Float)}
        int_cols = {c.key for c in mapper.column_attrs if isinstance(c.columns[0].type, Integer)}

        for key, value in normalized_info.items():
            if value is None: 
                continue
            new_key = "_52weekchange" if key == "52WeekChange" else key.lower()

            # Sanitize numeric string types
            if isinstance(value, str):
                if new_key in bigint_cols or new_key in int_cols:
                    try:
                        value = int(float(value))
                    except (ValueError, TypeError):
                        value = None
                elif new_key in float_cols:
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        value = None

            if value is not None:
                db_ready_info[new_key] = value

        valid_company_keys = {c.key for c in mapper.column_attrs}
        filtered_db_info = {k: v for k, v in db_ready_info.items() if k in valid_company_keys}

        if canonical_symbol:
            filtered_db_info["symbol"] = canonical_symbol

        existing = db.query(Company).filter(Company.symbol == canonical_symbol).first() if canonical_symbol else None
        
        if existing:
            for k, v in filtered_db_info.items():
                setattr(existing, k, v)
        else:
            existing = Company(**filtered_db_info)
            db.add(existing)

        db.flush()
        
        # Process officers separately
        if info.get("companyOfficers"):
            for officer_info in info["companyOfficers"]:
                officer_data = {
                    "company_id": existing.id,
                    "maxage": officer_info.get("maxAge"),
                    "name": officer_info.get("name"),
                    "age": officer_info.get("age"),
                    "title": officer_info.get("title"),
                    "yearborn": officer_info.get("yearBorn"),
                    "fiscalyear": officer_info.get("fiscalYear"),
                    "totalpay": officer_info.get("totalPay"),
                    "exercisedvalue": officer_info.get("exercisedValue"),
                    "unexercisedvalue": officer_info.get("unexercisedValue")
                }
                
                existing_record = db.query(CompanyOfficer).filter(
                    CompanyOfficer.company_id == existing.id,
                    CompanyOfficer.name == officer_data["name"]
                ).first()
                if existing_record:
                    for key, value in officer_data.items():
                        setattr(existing_record, key, value)
                else:
                    officer_record = CompanyOfficer(**officer_data)
                    db.add(officer_record)

        return existing

    def _internal_save_financials(self, db: Session, company_id: int, financials: Dict):
        """Standardized Upsert for Financials Table."""
        for stmt_type, df in financials.items():
            if df.empty: continue
            records = []
            for report_date, row in df.transpose().iterrows():
                for index, value in row.items():
                    if value is None or value == "":
                        continue

                    # Skip non-numeric metadata index items like 'Filing Date'
                    if str(index).strip().lower() in ['filing date', 'filingdate', 'currency symbol', 'currencysymbol']:
                        continue

                    # Safely convert to float
                    try:
                        numeric_value = float(value)
                    except (ValueError, TypeError):
                        # Skip values that cannot be represented as float (e.g. date strings)
                        continue
                    
                    records.append({
                        "company_id": company_id,
                        "report_date": report_date,
                        "type": stmt_type,
                        "index": index,
                        "value": json.dumps(value) if isinstance(value, (dict, list)) else str(value) if value is not None else None
                    })
            if records:
                stmt = pg_insert(Financials).values(records)
                on_conflict_stmt = stmt.on_conflict_do_update(
                    index_elements=['company_id', 'report_date', 'type', 'index'],
                    set_={'value': stmt.excluded['value']}
                )
                db.execute(on_conflict_stmt)

    def _internal_save_analyst_estimates(self, db: Session, company_id: int, estimates: Dict):
        """Standardized Upsert for all Analyst Estimate tables."""
        mappings = {
            'earnings': (AnalystEarningsEstimate, ['period_label', 'avg_estimate', 'low_estimate', 'high_estimate']),
            'revenue': (AnalystRevenueEstimate, ['period_label', 'avg_estimate', 'low_estimate', 'high_estimate']),
            'growth': (AnalystGrowthEstimate, ['period_label', 'growth_value_text'])
        }
        
        for key, (model_class, fields) in mappings.items():
            df = estimates.get(key)
            if df is None or df.empty: continue
            
            records = []
            for period, row in df.iterrows():
                record = {"company_id": company_id, "period_label": period}
                for field in fields:
                    record[field] = row.get(field)
                records.append(record)
            
            if records:
                stmt = pg_insert(model_class).values(records)
                on_conflict_stmt = stmt.on_conflict_do_update(
                    index_elements=['company_id', 'period_label'],
                    set_={f: stmt.excluded[f] for f in fields}
                )
                db.execute(on_conflict_stmt)

    def _internal_save_insider_data(self, db: Session, company_id: int, insider_payload: dict):
        """Handles saving InsiderRoster and InsiderTransaction."""
        if not insider_payload:
            return

        roster_data = insider_payload.get('roster')
        if roster_data is not None and not roster_data.empty:
            records = []
            for i, r in roster_data.iterrows():
                records.append({
                    "company_id": company_id,
                    "name": r.get("name"),
                    "position": r.get("position"),
                    "most_recent_transaction": r.get("most_recent_transaction"),
                    "most_recent_transaction_date": r.get("most_recent_transaction_date"),
                    "shares_owned_directly": r.get("shares_owned_directly"),
                    "shares_owned_indirectly": r.get("shares_owned_indirectly")
                })
            if records:
                stmt = pg_insert(InsiderRoster).values(records)
                on_conflict_stmt = stmt.on_conflict_do_update(
                    index_elements=['company_id', 'name', 'position'],
                    set_={k: stmt.excluded[k] for k in ['most_recent_transaction', 'most_recent_transaction_date', 'shares_owned_directly', 'shares_owned_indirectly']}
                )
                db.execute(on_conflict_stmt)

        transactions_data = insider_payload.get('transactions')
        if transactions_data is not None and not transactions_data.empty:
            records = []
            for i, r in transactions_data.iterrows():
                records.append({
                    "company_id": company_id,
                    "insider_name": r.get("insider_name"),
                    "shares": r.get("shares"),
                    "transaction_type": r.get("transaction_type"),
                    "transaction_code": r.get("transaction_code"),
                    "start_date": r.get("start_date"),
                    "value": r.get("value")
                })
            if records:
                stmt = pg_insert(InsiderTransaction).values(records)
                stmt = stmt.on_conflict_do_nothing(index_elements=['company_id', 'insider_name', 'transaction_type', 'start_date', 'shares'])
                db.execute(stmt)

    def _internal_save_institutional_holdings(self, db: Session, company_id: int, holdings_data: pd.DataFrame, holder_type: str = 'institutional'):
        """Handles Institutional/Mutual Fund holders."""
        if holdings_data is None or holdings_data.empty: return
        records = []
        for i, h in holdings_data.iterrows():
            records.append({
                "company_id": company_id,
                "holder_name": h.get("holder_name") or h.get("Holder"),
                "shares": h.get("shares") or h.get("Shares"),
                "date_reported": h.get("date_reported") or h.get("Date Reported"),
                "percent_out": h.get("percent_out") or h.get("% Out"),
                "value": h.get("value") or h.get("Value"),
                "holder_type": holder_type
            })

        if records:
            stmt = pg_insert(InstitutionalHolding).values(records)
            on_conflict_stmt = stmt.on_conflict_do_update(
                index_elements=['company_id', 'holder_name', 'date_reported', 'holder_type'],
                set_={k: stmt.excluded[k] for k in ['shares', 'percent_out', 'value']}
            )
            db.execute(on_conflict_stmt)

    def _internal_save_analyst_sentiment(self, db: Session, company_id: int, sentiment_payload: dict):
        """Handles Upgrades, Earnings History, and EPS Trends."""
        upgrades = sentiment_payload.get('upgrades_downgrades')
        if upgrades is not None and not upgrades.empty:
            records = []
            for i, r in upgrades.iterrows():
                 records.append({
                    "company_id": company_id,
                    "date": r.get("date"),
                    "firm": r.get("firm"),
                    "to_grade": r.get("to_grade"),
                    "from_grade": r.get("from_grade"),
                    "action": r.get("action")
                })
            if records:
                stmt = pg_insert(UpgradeDowngrade).values(records)
                stmt = stmt.on_conflict_do_nothing(index_elements=['company_id', 'date', 'firm', 'to_grade', 'action'])
                db.execute(stmt)

        hist = sentiment_payload.get('earnings_history')
        if hist is not None and not hist.empty:
            records = []
            for i, r in hist.iterrows():
                records.append({
                    "company_id": company_id,
                    "report_date": r.get("report_date"),
                    "eps_estimate": r.get("eps_estimate"),
                    "eps_actual": r.get("eps_actual"),
                    "eps_difference": r.get("eps_difference"),
                    "surprise_percent": r.get("surprise_percent")
                })
            if records:
                stmt = pg_insert(AnalystEarningsHistory).values(records)
                stmt = stmt.on_conflict_do_nothing(index_elements=['company_id', 'report_date'])
                db.execute(stmt)
            
        for trend_key in ['eps_trend', 'eps_revisions']:
            data = sentiment_payload.get(trend_key)
            if data is not None and not data.empty: 
                model_class = AnalystEpsTrend if trend_key == 'eps_trend' else AnalystEpsRevisions
                records = []
                for i, r in data.iterrows():
                    rec = {"company_id": company_id, "period_label": r.get("period_label")}
                    for col in model_class.__table__.columns.keys():
                        if col in ['id', 'company_id', 'period_label', 'last_updated']: continue
                        if col in r: rec[col] = r[col]
                    records.append(rec)
                
                if records:
                    stmt = pg_insert(model_class).values(records)
                    on_conflict_stmt = stmt.on_conflict_do_update(
                        index_elements=['company_id', 'period_label'],
                        set_={c: stmt.excluded[c] for c in model_class.__table__.columns.keys() if c not in ['id', 'company_id', 'period_label', 'last_updated']}
                    )
                    db.execute(on_conflict_stmt)

    def _build_result_from_db(self, db: Session, company: Company, timeframe: str, start_date: str, end_date: str, price_only: bool) -> Dict:
        result = {'company': object_as_dict(company)}
        
        query = db.query(PriceHistory).filter(
            PriceHistory.company_id == company.id, 
            PriceHistory.timeframe == timeframe
        )
        if start_date: query = query.filter(PriceHistory.timestamp >= start_date)
        if end_date: query = query.filter(PriceHistory.timestamp <= end_date)
        
        prices = query.order_by(PriceHistory.timestamp.asc()).all()
        if prices:
            df = pd.DataFrame([{
                "date": p.timestamp, "open": p.open, "high": p.high, 
                "low": p.low, "close": p.close, "adjclose": p.adjclose, "volume": p.volume
            } for p in prices])
            if not df.empty:
                df.set_index("date", inplace=True)
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)             
            result['shareprices'] = df

        if not price_only:
            fin_records = db.query(Financials).filter(Financials.company_id == company.id).all()
            if fin_records:
                statements = {}
                for record in fin_records:
                    if record.type not in statements:
                        statements[record.type] = {}
                    if record.report_date not in statements[record.type]:
                        statements[record.type][record.report_date] = {}
                    statements[record.type][record.report_date][record.index] = record.value

                for statement_type, report_data in statements.items():
                    statement_list = []
                    for report_date, index_value in report_data.items():
                        statement_list.append(dict({'Date': report_date}, **index_value))
                    
                    df_statement = pd.DataFrame(statement_list)
                    if not df_statement.empty:
                        result[statement_type] = df_statement.set_index('Date')
            
        return result
    
    def load_price_data(self, ticker: str, start_date: str = None, end_date: str = None, timeframe: str = '1d', refresh: bool = False) -> Dict[str, Any]:
        """
        Loads historical price data for a ticker, checking DB first, then Provider.
        """
        canonical_ticker = self.provider.normalize_to_canonical(ticker)
        api_ticker = self.provider.normalize_from_canonical(canonical_ticker)
        
        db = next(get_db())
        try:
            company = db.query(Company).filter(Company.symbol == canonical_ticker).first()
            if not company:
                return {"error": f"Company {canonical_ticker} not found in database."}

            query = db.query(PriceHistory).filter(
                PriceHistory.company_id == company.id, 
                PriceHistory.timeframe == timeframe
            )
            if start_date:
                query = query.filter(PriceHistory.timestamp >= start_date)
            if end_date:
                query = query.filter(PriceHistory.timestamp <= end_date)
            
            price_history = query.order_by(PriceHistory.timestamp.asc()).all()
            
            if not price_history or refresh:
                logger.info(f"DataManager: Fetching fresh price data for {api_ticker}")
                price_df = self.provider.get_historical_data(api_ticker, interval=timeframe)
                
                if not price_df.empty:
                    self.save_price_history_flow(db, {company.id: price_df}, timeframe)
                    price_history = query.order_by(PriceHistory.timestamp.asc()).all()
                else:
                    return {"error": "No data returned from provider"}

            data_list = []
            for p in price_history:
                data_list.append({
                    "date": p.timestamp, "open": p.open, "high": p.high, 
                    "low": p.low, "close": p.close, "adjclose": p.adjclose, "volume": p.volume
                })
            
            df = pd.DataFrame(data_list)
            if not df.empty:
                df.set_index("date", inplace=True)
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None) 
            return {"status": "success", "shareprices": df}

        except Exception as e:
            logger.error(f"DataManager load_price_data error: {e}")
            return {"error": str(e)}
        finally:
            db.close()

    @data_manager_cache(ttl_seconds=3600)
    def get_analyst_data(self, ticker: str, data_type: str):
        canonical_ticker = self.provider.normalize_to_canonical(ticker)
        api_ticker = self.provider.normalize_from_canonical(canonical_ticker)
        return self.provider.get_analyst_data(api_ticker, data_type)

    @data_manager_cache(ttl_seconds=3600)
    def get_extra_competitors(self, ticker: str):
        canonical_ticker = self.provider.normalize_to_canonical(ticker)
        api_ticker = self.provider.normalize_from_canonical(canonical_ticker)
        return self.provider.get_extra_competitors(api_ticker)

    @data_manager_cache(ttl_seconds=3600)
    def get_news_content(self, ticker: str):
        canonical_ticker = self.provider.normalize_to_canonical(ticker)
        api_ticker = self.provider.normalize_from_canonical(canonical_ticker)
        return self.provider.get_news_content(api_ticker)

    def save_price_history_flow(self, db: Session, batch_price_data: dict, timeframe: str):
        if batch_price_data:
            self.save_or_update_batch_price_data(db, batch_price_data, timeframe=timeframe)
            return True
        return False

    def save_or_update_company_data(self, market="us", exchange=None, quote_types=["EQUITY", "ETF", "CRYPTOCURRENCY"], ticker_file="yhallsym.json", ticker_list: list = None, interval="1d", batch_size=50, start_date=FULL_HISTORY_START_DATE, end_date=None, existing_tickers_action='skip', update_prices_action='yes', **kwargs):
        """Batch processing logic for bulk ticker ingest and price synchronization."""
        db = next(get_db())
        try:
            if ticker_list:
                tickers_to_process = ticker_list
            else:
                tickers_to_process = self.get_all_tickers_in_market(market=market, exchange=exchange, ticker_file=ticker_file)

            if not tickers_to_process:
                return

            if existing_tickers_action != 'only' or update_prices_action in ['yes', 'no']:
                tickers_to_process = self._process_company_info_batches(db, tickers_to_process, batch_size, quote_types)

            if update_prices_action != "no":
                if update_prices_action == 'last_day':
                    start_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
                if end_date and datetime.strptime(end_date, '%Y-%m-%d').date() >= datetime.now().date():
                    end_date = None
                self._process_price_history_batches(db, tickers_to_process, batch_size, start_date, end_date, interval)
        except Exception as e:
            logger.error(f"DataManager batch error: {e}")
        finally:
            db.close()

    # --- Helper Methods ---

    def _process_company_info_batches(self, db: Session, tickers: list, batch_size: int, quote_types: list):
        processed = []
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            for ticker in batch:
                api_ticker = self.provider.normalize_from_canonical(self.provider.normalize_to_canonical(ticker))
                info = self.provider.get_company_info(api_ticker)
                if info and (not quote_types or info.get('quoteType') in quote_types):
                    company = self._internal_save_company(db, info)
                    if company: processed.append(ticker)
            db.commit()
        return processed

    def _process_price_history_batches(self, db: Session, tickers: list, batch_size: int, start_date: str, end_date: str, interval: str):
        total_tickers = len(tickers)
        inactive_tickers = []
        
        for i in range(0, total_tickers, batch_size):
            batch = tickers[i:i + batch_size]
            logger.info(f"DataManager: Processing price batch {i // batch_size + 1}/{total_tickers // batch_size + 1} ({len(batch)} tickers)")
            
            share_price_data = self.download_multiple_tickers_data(batch, start_date, end_date, interval)
            found_tickers_in_batch = set()

            if share_price_data is not None and not share_price_data.empty:
                batch_price_data = {}
                
                # Fetch companies in bulk for efficiency
                companies = db.query(Company).filter(Company.symbol.in_(batch)).all()
                company_map = {c.symbol: c for c in companies}

                for ticker in batch:
                    company = company_map.get(ticker)
                    if not company: 
                        continue

                    ticker_prefix = f"{ticker}-"
                    ticker_columns = [c for c in share_price_data.columns if c.startswith(ticker_prefix)]

                    if ticker_columns:
                        price_data = share_price_data[ticker_columns].copy()
                        price_data.columns = [col.replace(ticker_prefix, "") for col in price_data.columns]
                        
                        valid_price_data = price_data.dropna()
                        if not valid_price_data.empty:
                            is_up_to_date = True
                            last_date = valid_price_data.index.max()
                            if last_date.tzinfo:
                                last_date = last_date.tz_localize(None)
                            
                            target_date = datetime.now()
                            if end_date:
                                target_date = datetime.strptime(end_date, '%Y-%m-%d')
                            
                            try:
                                days_diff = np.busday_count(last_date.date().isoformat(), target_date.date().isoformat())
                                if days_diff > 3:
                                    is_up_to_date = False
                                    logger.warning(f"[{ticker}] Data is stale (Last: {last_date.date()}).")
                            except Exception:
                                if (target_date - last_date).days > 3:
                                    is_up_to_date = False

                            if is_up_to_date:
                                found_tickers_in_batch.add(ticker)
                            
                            batch_price_data[company.id] = price_data
            
                if batch_price_data:
                    self.save_price_history_flow(db, batch_price_data, timeframe=interval)
                    
                    if start_date != FULL_HISTORY_START_DATE:
                        company_ids_in_batch = list(batch_price_data.keys())
                        start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
                        
                        tickers_to_refresh = self.find_tickers_with_splits_in_db(db, company_ids_in_batch, start_date_dt)
                        if tickers_to_refresh:
                            logger.warning(f"Splits detected for: {[t.symbol for t in tickers_to_refresh]}. Triggering refresh.")
                            ticker_map = {t.symbol: t.id for t in tickers_to_refresh}
                            self.refresh_split_tickers(db, ticker_map, batch_size)
            
            batch_inactive_tickers = [t for t in batch if t not in found_tickers_in_batch]
            if batch_inactive_tickers:
                logger.info(f"Marking inactive: {batch_inactive_tickers}")
                db.query(Company).filter(Company.symbol.in_(batch_inactive_tickers)).update({"isactive": False}, synchronize_session=False)
                db.commit()

            inactive_tickers.extend(batch_inactive_tickers)

        if inactive_tickers:
            logger.info(f"Total inactive tickers identified: {len(inactive_tickers)}")

    def find_tickers_with_splits_in_db(self, db: Session, company_ids: list, start_date: datetime) -> list:
        if not company_ids:
            return []
        
        try:
            return db.query(Company).join(PriceHistory).filter(
                Company.id.in_(company_ids),
                PriceHistory.timestamp >= start_date,
                PriceHistory.split_coefficient != 0,
                PriceHistory.split_coefficient != 1.0
            ).distinct().all()
        except Exception as e:
            logger.error(f"Error finding tickers with splits: {e}")
            return []

    def refresh_split_tickers(self, db: Session, ticker_map: dict, batch_size: int):
        tickers_to_process = list(ticker_map.keys())
        total_tickers = len(tickers_to_process)

        for i in range(0, total_tickers, batch_size):
            batch = tickers_to_process[i:i + batch_size]
            logger.info(f"DataManager: Refreshing full history for batch {i // batch_size + 1}/{total_tickers // batch_size + 1} ({len(batch)} tickers)")
            
            full_history_data = self.download_multiple_tickers_data(
                batch, 
                start_date=FULL_HISTORY_START_DATE, 
                end_date=None, 
                interval="1d"
            )
            
            if full_history_data is not None and not full_history_data.empty:
                batch_data_to_save = {}
                for t in batch:
                    ticker_prefix = f"{t}-"
                    ticker_columns = [c for c in full_history_data.columns if c.startswith(ticker_prefix)]
                    
                    if ticker_columns:
                        ticker_df = full_history_data[ticker_columns].copy()
                        ticker_df.columns = [col.replace(ticker_prefix, "") for col in ticker_df.columns]
                        
                        if t in ticker_map:
                            batch_data_to_save[ticker_map[t]] = ticker_df
                
                if batch_data_to_save:
                    self.save_price_history_flow(db, batch_data_to_save, timeframe='1d')
            
        db.commit()

    def get_benchmark_ticker_for_asset(self, ticker: str) -> tuple[str, str]:
        primary_benchmark = secondary_benchmark = PRIMARY_BENCHMARK
        db_session = next(get_db())
        try:
            company = db_session.query(Company).filter(Company.symbol == ticker).first()
            if company and company.sector in SECTOR_TO_ETF_MAP:
                primary_benchmark = SECTOR_TO_ETF_MAP[company.sector]
        finally:
            db_session.close()
        
        return primary_benchmark, secondary_benchmark

    def save_or_update_batch_price_data(self, db: Session, batch_price_data: dict, timeframe: str = '1d'):
        """Saves or updates price history for a batch of companies in the database using PostgreSQL ON CONFLICT."""
        try:
            records_to_insert = []
            for company_id, price_history_df in batch_price_data.items():
                price_history_df = price_history_df.dropna()
                for index, row in price_history_df.iterrows():
                    try:
                        price_data = {
                            "company_id": company_id,
                            "timestamp": index.to_pydatetime(),
                            "timeframe": timeframe,
                            "open": float(row["open"]),
                            "high": float(row["high"]),
                            "low": float(row["low"]),
                            "close": float(row["close"]),
                            "adjclose": float(row["adjclose"]),
                            "volume": int(row["volume"]),
                            "dividend_amount": float(row.get("dividends", 0.0)),
                            "split_coefficient": float(row.get("stocksplits", 0.0))
                        }
                    except Exception as e:
                        logger.warning(f"Skipped price data for company {company_id} because of error: {e}", exc_info=True)
                        continue
                    
                    records_to_insert.append(price_data)

            if records_to_insert:
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

            db.commit()
        except Exception as e:
            db.rollback()
            logger.error(f"An error occurred while saving price history: {e}")

    def find_top_competitors(self, ticker: str, num_competitors: int = 5, extra_competitors: list[str] = []) -> list[dict]:
        """Finds top competitors using TF-IDF text features combined with scaled financial metrics via KNN."""
        symbol = self.provider.normalize_to_canonical(ticker)
        db = next(get_db())
        try:
            company = db.query(Company).filter(Company.symbol == symbol).first()
            if not company:
                logger.warning(f"Company with symbol {symbol} not found.")
                return []

            if not company.market or not company.industry:
                logger.warning(f"Company {symbol} is missing market or industry information.")
                return []

            extra_competitors_data = []
            if extra_competitors:
                extra_competitors_data = db.query(
                    Company.id, Company.symbol, Company.longbusinesssummary,
                    Company.marketcap, Company.totalrevenue, Company.enterprisevalue, Company.website
                ).filter(
                    Company.id != company.id,
                    Company.website != company.website,
                    Company.market == company.market,
                    Company.exchange == company.exchange,
                    Company.industry == company.industry,
                    Company.longbusinesssummary != None,
                    Company.symbol.in_(extra_competitors)
                ).distinct(Company.website).all()

            num_extra_competitors = len(extra_competitors_data)
            num_additional_competitors = max(0, num_competitors * 3 - num_extra_competitors)

            if num_additional_competitors > 0:
                company_industry_data = db.query(
                    Company.id, Company.symbol, Company.longbusinesssummary,
                    Company.marketcap, Company.totalrevenue, Company.enterprisevalue, Company.website
                ).filter(
                    Company.id != company.id,
                    Company.website != company.website,
                    Company.market == company.market,
                    Company.exchange == company.exchange,
                    Company.industry == company.industry,
                    Company.longbusinesssummary != None,
                    Company.symbol.notin_(extra_competitors)
                ).distinct(Company.website).all()

                extra_competitors_data.extend(company_industry_data)

            if not extra_competitors_data:
                logger.warning(f"No competitors found for {symbol} in the same market and industry.")
                return []

            df = pd.DataFrame(
                extra_competitors_data,
                columns=['company_id', 'symbol', 'longbusinesssummary', 'marketcap', 'totalrevenue', 'enterprisevalue', 'website']
            )

            vectorizer = TfidfVectorizer(stop_words='english')
            text_features = vectorizer.fit_transform(df['longbusinesssummary']).toarray()
            text_features_df = pd.DataFrame(text_features, index=df.index, columns=[f"tfidf_{i}" for i in range(text_features.shape[1])])

            numerical_cols = ['marketcap', 'totalrevenue', 'enterprisevalue']
            df[numerical_cols] = df[numerical_cols].fillna(0)

            imputer = SimpleImputer(strategy="mean")
            df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

            scaler = StandardScaler()
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

            df = df.drop(columns=['longbusinesssummary'])
            df = pd.concat([df, text_features_df], axis=1)

            knn_data = df.drop(columns=['symbol', 'company_id', 'website']).copy()
            if knn_data.isnull().values.any():
                logger.error("NaN values found in knn_data before fitting KNN.")
                numeric_cols_knn = knn_data.select_dtypes(include=np.number).columns
                imputer_knn = SimpleImputer(strategy="mean")
                knn_data[numeric_cols_knn] = imputer_knn.fit_transform(knn_data[numeric_cols_knn])
                
                if knn_data.isnull().values.any():
                    raise ValueError("NaN values persist in knn_data after imputation.")

            knn = NearestNeighbors(n_neighbors=min(num_competitors * 5, len(df)), metric='euclidean')
            knn.fit(knn_data)

            target_company_df = pd.DataFrame([{
                'marketcap': company.marketcap,
                'totalrevenue': company.totalrevenue,
                'enterprisevalue': company.enterprisevalue
            }])

            target_company_df[numerical_cols] = imputer.transform(target_company_df[numerical_cols])
            target_company_df[numerical_cols] = scaler.transform(target_company_df[numerical_cols])

            if company.longbusinesssummary:
                target_text_features = vectorizer.transform([company.longbusinesssummary]).toarray()
            else:
                target_text_features = np.zeros((1, text_features.shape[1]))
            target_text_features_df = pd.DataFrame(target_text_features, columns=[f"tfidf_{i}" for i in range(text_features.shape[1])])

            target_company_data = pd.concat([target_company_df, target_text_features_df], axis=1)

            distances, indices = knn.kneighbors(target_company_data)

            competitors = []
            for i, dist in zip(indices[0], distances[0]):
                competitor_symbol = df.iloc[i]['symbol']
                is_yf_competitor = competitor_symbol in extra_competitors

                priority_weight = 0.5 if is_yf_competitor else 1.0

                penalty = 0
                for col in numerical_cols:
                    diff = np.std(abs(df.iloc[i][col] - target_company_df[col].iloc[0]))
                    if diff > 1:
                        penalty += diff * 0.5

                competitors.append({"ticker": competitor_symbol, "distance": float(dist * priority_weight + penalty)})

            competitors.sort(key=lambda x: x['distance'])
            return competitors[:num_competitors]

        except Exception as e:
            logger.error(f"An error occurred in find_top_competitors: {e}")
            return []
        finally:
            db.close()
            