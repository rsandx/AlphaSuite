import logging
import random
import re
import time
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Dict, Union

import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup
from curl_cffi import requests as cffi_requests
from requests import HTTPError, RequestException

from .base import BaseDataProvider

try:
    from yfinance.exceptions import YFRateLimitError
except ImportError:
    class YFRateLimitError(Exception):
        pass

logger = logging.getLogger(__name__)


def provider_retry_handler(retries=3, backoff_factor=2):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            last_exception = None
            for attempt in range(retries):
                try:
                    return func(self, *args, **kwargs)
                except Exception as e:
                    last_exception = e
                    is_rate_limit = "RateLimit" in type(e).__name__ or "429" in str(e)
                    sleep_time = (backoff_factor ** attempt) + (random.uniform(1, 5) if is_rate_limit else 0)
                    logger.warning(f"Provider Retry: {type(e).__name__} on {args}. Attempt {attempt+1}/{retries}. Sleeping {sleep_time:.2f}s")
                    time.sleep(sleep_time)
            logger.error(f"Provider Retry: All {retries} attempts failed for {args}.")
            raise last_exception
        return wrapper
    return decorator


class YFinanceProvider(BaseDataProvider):
    def __init__(self):
        pass

    def normalize_to_canonical(self, provider_symbol: str) -> str:
        if "." not in provider_symbol:
            return f"{provider_symbol.upper()}.US"
        return provider_symbol.upper()

    def normalize_from_canonical(self, canonical_symbol: str) -> str:
        if ".US" in canonical_symbol.upper():
            return canonical_symbol.upper().replace(".US", "")
        return canonical_symbol

    def _safe_get_dataframe(self, func) -> pd.DataFrame:
        """Helper to safely execute yfinance methods returning DataFrames."""
        try:
            res = func()
            if isinstance(res, pd.DataFrame):
                return res
            return pd.DataFrame()
        except Exception as e:
            logger.debug(f"Error calling {func}: {e}")
            return pd.DataFrame()

    @provider_retry_handler()
    def get_ticker_data(self, ticker: str, start_date: str = None, end_date: str = None, timeframe: str = '1d') -> Dict[str, Any]:
        """
        Master Method: Fetches price histories, fundamental info, officers,
        financials, and analyst metrics matching canonical schema standards.
        """
        api_ticker = self.normalize_from_canonical(ticker)
        logger.info(f"Downloading data for {api_ticker} using yfinance")
        yt = yf.Ticker(api_ticker)

        payload = {
            "info": {},
            "price_history": pd.DataFrame(),
            "financials": {},
            "officers": [],
            "analyst_estimates": {
                "earnings": pd.DataFrame(),
                "revenue": pd.DataFrame(),
                "growth": pd.DataFrame(),
            },
            "analyst_sentiment": {
                "earnings_history": pd.DataFrame(),
                "eps_trend": pd.DataFrame(),
                "eps_revisions": pd.DataFrame(),
                "upgrades_downgrades": pd.DataFrame(),
            },
            "insider": {
                "transactions": pd.DataFrame(),
                "roster": pd.DataFrame(),
            },
            "institutional_holders": pd.DataFrame(),
            "mutualfund_holders": pd.DataFrame(),
        }

        # ----------------------------------------------------
        # 1. Company Info & Officer Parsing
        # ----------------------------------------------------
        info = yt.info or {}
        if info:
            info = dict(info)  # Copy to modify safely

            # Perform metric scale standardizations
            if info.get("debtToEquity") is not None and info["debtToEquity"] > 1:
                info["debtToEquity"] /= 100.0

            # Handle Unix timestamp conversions safely
            ts_keys = [
                "earningsTimestamp", "earningsTimestampStart", "earningsTimestampEnd",
                "earningsCallTimestampStart", "earningsCallTimestampEnd",
                "postMarketTime", "regularMarketTime"
            ]
            for key in ts_keys:
                val = info.get(key)
                if isinstance(val, (int, float)) and val > 0:
                    info[key] = datetime.fromtimestamp(val, tz=timezone.utc).replace(tzinfo=None)

            # Handle date field conversions
            date_keys = [
                "governanceEpochDate", "compensationAsOfEpochDate", "sharesShortPreviousMonthDate",
                "dateShortInterest", "lastFiscalYearEnd", "nextFiscalYearEnd", "mostRecentQuarter",
                "lastDividendDate", "dividendDate", "exDividendDate", "nameChangeDate",
                "ipoExpectedDate", "lastSplitDate", "firstTradeDateMilliseconds", "firstTradeDateEpochUtc"
            ]
            for key in date_keys:
                val = info.get(key)
                if val:
                    if isinstance(val, str):
                        try:
                            info[key] = datetime.strptime(val, '%Y-%m-%d').date()
                        except ValueError:
                            info[key] = None
                    elif isinstance(val, datetime):
                        info[key] = val.date()
                    elif isinstance(val, (int, float)) and val > 0:
                        if key == "firstTradeDateMilliseconds":
                            info[key] = datetime.fromtimestamp(val / 1000.0, tz=timezone.utc).date()
                        else:
                            info[key] = datetime.fromtimestamp(val, tz=timezone.utc).date()

            # Extract Company Officers -> CompanyOfficer entity
            officers_raw = info.get("companyOfficers", [])
            officers_list = []
            if isinstance(officers_raw, list):
                current_year = datetime.now().year
                for off in officers_raw:
                    if isinstance(off, dict):
                        year_born = off.get("yearBorn")
                        age = off.get("age")
                        if not age and year_born and str(year_born).isdigit():
                            age = current_year - int(year_born)

                        officers_list.append({
                            "name": off.get("name"),
                            "title": off.get("title"),
                            "yearborn": year_born,
                            "age": age,
                            "total_pay": off.get("totalPay"),
                            "exercised_value": off.get("valueExercised"),
                            "unexercised_value": off.get("unexercisedValue")
                        })
            payload["officers"] = officers_list
            payload["info"] = info

        # ----------------------------------------------------
        # 2. Price History
        # ----------------------------------------------------
        price_df = yt.history(start=start_date, end=end_date, interval=timeframe, auto_adjust=False, period="max")
        if isinstance(price_df, pd.DataFrame) and not price_df.empty:
            price_df.columns = [col.lower().replace(' ', '') for col in price_df.columns]
            payload["price_history"] = price_df

        # ----------------------------------------------------
        # 3. Financial Statements
        # ----------------------------------------------------
        payload["financials"]["annual_balance_sheet"] = self._safe_get_dataframe(lambda: yt.get_balance_sheet(freq="yearly"))
        payload["financials"]["annual_income_statement"] = self._safe_get_dataframe(lambda: yt.get_income_stmt(freq="yearly"))
        payload["financials"]["annual_cash_flow"] = self._safe_get_dataframe(lambda: yt.get_cash_flow(freq="yearly"))
        payload["financials"]["quarterly_balance_sheet"] = self._safe_get_dataframe(lambda: yt.get_balance_sheet(freq="quarterly"))
        payload["financials"]["quarterly_income_statement"] = self._safe_get_dataframe(lambda: yt.get_income_stmt(freq="quarterly"))
        payload["financials"]["quarterly_cash_flow"] = self._safe_get_dataframe(lambda: yt.get_cash_flow(freq="quarterly"))

        # ----------------------------------------------------
        # 4. Analyst & Insider Entities
        # ----------------------------------------------------
        payload["analyst_estimates"]["earnings"] = self._safe_get_dataframe(yt.get_earnings_estimate)
        payload["analyst_estimates"]["revenue"] = self._safe_get_dataframe(yt.get_revenue_estimate)
        payload["analyst_estimates"]["growth"] = self._safe_get_dataframe(yt.get_growth_estimates)

        payload["analyst_sentiment"]["upgrades_downgrades"] = self._safe_get_dataframe(yt.get_upgrades_downgrades)
        payload["analyst_sentiment"]["earnings_history"] = self._safe_get_dataframe(yt.get_earnings_history)
        payload["analyst_sentiment"]["eps_trend"] = self._safe_get_dataframe(yt.get_eps_trend)
        payload["analyst_sentiment"]["eps_revisions"] = self._safe_get_dataframe(yt.get_eps_revisions)

        payload["insider"]["transactions"] = self._safe_get_dataframe(yt.get_insider_transactions)
        payload["insider"]["roster"] = self._safe_get_dataframe(yt.get_insider_roster_holders)

        payload["institutional_holders"] = self._safe_get_dataframe(yt.get_institutional_holders)
        payload["mutualfund_holders"] = self._safe_get_dataframe(yt.get_mutualfund_holders)

        return payload

    @provider_retry_handler()
    def get_historical_data(self, ticker: str, start_date: str = None, end_date: str = None, interval: str = '1d') -> pd.DataFrame:
        api_ticker = self.normalize_from_canonical(ticker)
        price_df = yf.Ticker(api_ticker).history(start=start_date, end=end_date, interval=interval, auto_adjust=False, period="max")
        if isinstance(price_df, pd.DataFrame) and not price_df.empty:
            price_df.columns = [col.lower().replace(' ', '') for col in price_df.columns]
        return price_df

    @provider_retry_handler()
    def get_company_info(self, ticker: str) -> Dict[str, Any]:
        api_ticker = self.normalize_from_canonical(ticker)
        return yf.Ticker(api_ticker).info or {}

    @provider_retry_handler()
    def get_financial_statements(self, ticker: str, statement_type: str, frequency: str) -> pd.DataFrame:
        yt = yf.Ticker(ticker)

        if statement_type == "income_statement":
            return yt.quarterly_income_stmt if frequency == "quarterly" else yt.income_stmt
        elif statement_type == "balance_sheet":
            return yt.quarterly_balance_sheet if frequency == "quarterly" else yt.balance_sheet
        elif statement_type == "cash_flow":
            return yt.quarterly_cashflow if frequency == "quarterly" else yt.cashflow
        return pd.DataFrame()

    @provider_retry_handler()
    def get_analyst_data(self, ticker: str, data_type: str) -> Union[pd.DataFrame, Dict[str, Any]]:
        yt = yf.Ticker(ticker)

        if data_type == "recommendations_summary":
            return getattr(yt, "recommendations_summary", pd.DataFrame())
        elif data_type == "earnings_dates":
            return self._safe_get_dataframe(yt.get_earnings_dates)
        return pd.DataFrame()

    def get_extra_competitors(self, ticker: str) -> Union[list, dict]:
        """Scrapes competitor tickers from Yahoo Finance cleanly."""
        retries = 5
        backoff_factor = 5

        for attempt in range(retries):
            try:
                url = f"https://finance.yahoo.com/quote/{ticker}"
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
                }
                response = cffi_requests.get(url, headers=headers, impersonate="chrome110", timeout=30)
                response.raise_for_status()

                soup = BeautifulSoup(response.content, "html.parser")
                competitor_elements = soup.find_all("section", {"data-testid": "compare-to"})
                if not competitor_elements:
                    return []

                competitors = []
                for section in competitor_elements:
                    links = section.find_all("a")
                    for link in links:
                        href = link.get("href")
                        if href and "/quote/" in href:
                            match = re.search(r"/quote/([A-Z\.]+)", href)
                            if match:
                                competitors.append(match.group(1))

                competitors = list(set(competitors))
                if ticker in competitors:
                    competitors.remove(ticker)

                return competitors

            except HTTPError as e:
                if e.response and e.response.status_code == 429:
                    if attempt < retries - 1:
                        sleep_time = backoff_factor ** (attempt + 1)
                        logger.warning(f"Rate limited scraping competitors. Retrying after {sleep_time:.2f} seconds...")
                        time.sleep(sleep_time)
                    else:
                        return {"error": f"Rate limited after multiple retries: {e}"}
                else:
                    return {"error": f"HTTP error: {e}"}
            except RequestException as e:
                return {"error": f"Error fetching data: {e}"}
            except Exception as e:
                return {"error": f"An unexpected error occurred: {e}"}

    def get_news_content(self, ticker: str) -> str:
        try:
            yt = yf.Ticker(ticker)
            news_items = yt.news
            logger.info(f"{news_items=}")
            text = ""
            for item in news_items:
                content = item["content"] if item.get("content") else item
                if "pubDate" in content: 
                    text += f"{content.get('pubDate')}: "
                if "title" in content: # News articles
                    text += f"{content.get('title', '')}. {content.get('description', '')}. {content.get('summary', '')}\n"
                elif "text" in content: # Tweets
                    text += f"{content.get('text', '')}\n"
                else:
                    logger.warning(f"Could not extract text from news item: {item}")

            return text
        except Exception as e:
            logger.error(f"Error getting news content for {ticker}: {e}", exc_info=True)
            return {"error": str(e)}

            