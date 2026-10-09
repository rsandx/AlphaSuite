import logging
import re
from datetime import datetime
import pandas as pd
import requests
from typing import Dict, Any, Union
from .base import BaseDataProvider

logger = logging.getLogger(__name__)

def parse_date(date_str):
    """Safely converts string/timestamp to python date object."""
    if not date_str or date_str in ["0000-00-00", "None"]:
        return None
    try:
        return pd.to_datetime(date_str).date()
    except Exception:
        return None

def parse_number(val, to_type=int):
    """Safely converts string numeric values to int or float."""
    if val is None or val == "" or val == "None":
        return None
    try:
        # Convert string to float first to safely handle decimal strings like "35873000000.00"
        num = float(val)
        return int(num) if to_type is int else num
    except (ValueError, TypeError):
        return None

class EODHDProvider(BaseDataProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://eodhd.com/api"
        self.session = requests.Session()
        self.session.params = {'api_token': self.api_key}

    def normalize_to_canonical(self, provider_symbol: str) -> str:
        if "." not in provider_symbol:
            return f"{provider_symbol}.US"
        return provider_symbol

    def normalize_from_canonical(self, canonical_symbol: str) -> str:
        return canonical_symbol

    def _process_eodhd_statement(self, statement_data):
        """Transforms an EODHD statement object (yearly/quarterly dict) into a pandas DataFrame."""
        if not statement_data or not isinstance(statement_data, dict):
            return pd.DataFrame()
            
        def to_yf_key(k):
            s = re.sub(r'(?<!^)(?=[A-Z])', ' ', k)
            return s.replace('_', ' ').replace('  ', ' ').title()

        formatted_statement = {}
        for date_str, metrics in statement_data.items():
            if not isinstance(metrics, dict):
                continue

            try:
                timestamp = pd.to_datetime(date_str)
            except Exception:
                timestamp = date_str
                
            for eod_key, value in metrics.items():
                # Filter out string metadata keys from being treated as financial row metrics
                if eod_key.lower() in ['date', 'filingdate', 'currency_symbol', 'currencysymbol']:
                    continue
                    
                yf_key = to_yf_key(eod_key)
                if yf_key not in formatted_statement:
                    formatted_statement[yf_key] = {}
                
                try:
                    formatted_statement[yf_key][timestamp] = float(value) if value is not None else None
                except (ValueError, TypeError):
                    formatted_statement[yf_key][timestamp] = value

        df = pd.DataFrame.from_dict(formatted_statement, orient='index')
        if not df.empty:
            df = df.reindex(sorted(df.columns, reverse=True), axis=1)
        return df

    def get_ticker_data(self, ticker: str, start_date: str = None, end_date: str = None, timeframe: str = '1d') -> Dict[str, Any]:
        api_ticker = self.normalize_from_canonical(ticker)
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

        try:
            # ----------------------------------------------------
            # 1. Fetch Price Data
            # ----------------------------------------------------
            if timeframe == '1d':
                price_url = f"{self.base_url}/eod/{api_ticker}"
                query_str = "?fmt=json"
            elif timeframe in ['1m', '5m', '1h']:
                price_url = f"{self.base_url}/intraday/{api_ticker}"
                query_str = f"?fmt=json&interval={timeframe}"
            else:
                raise ValueError(f"Unsupported timeframe {timeframe}")

            if start_date and end_date:
                query_str += f"&from={start_date}&to={end_date}"
            elif start_date:
                query_str += f"&from={start_date}"
            elif end_date:
                query_str += f"&to={end_date}"

            response = self.session.get(f"{price_url}{query_str}", timeout=30)
            response.raise_for_status()
            price_data = response.json()

            if isinstance(price_data, list) and len(price_data) > 0:
                df = pd.DataFrame(price_data)
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                df.columns = [c.lower().replace(' ', '') for c in df.columns]
                df.rename(columns={'adjusted_close': 'adjclose'}, inplace=True)
                payload["price_history"] = df

            # ----------------------------------------------------
            # 2. Fetch Fundamental Data
            # ----------------------------------------------------
            fund_url = f"{self.base_url}/v1.1/fundamentals/{api_ticker}"
            response_fund = self.session.get(fund_url, timeout=30)
            response_fund.raise_for_status()
            fund_data = response_fund.json()

            if not fund_data or not isinstance(fund_data, dict):
                return {"error": "Could not fetch valid fundamental data from EODHD"}

            gen = fund_data.get("General", {}) or {}
            addr = gen.get("AddressData", {}) or {}
            high = fund_data.get("Highlights", {}) or {}
            val = fund_data.get("Valuation", {}) or {}
            shares = fund_data.get("SharesStats", {}) or {}
            tech = fund_data.get("Technicals", {}) or {}
            splits_div = fund_data.get("SplitsDividends", {}) or {}
            fin = fund_data.get("Financials", {}) or {}

            # Financial extracts
            income_stmt = fin.get("Income_Statement", {}).get("yearly", {})
            balance_sheet = fin.get("Balance_Sheet", {}).get("yearly", {})
            cash_flow = fin.get("Cash_Flow", {}).get("yearly", {})

            latest_income = next(iter(income_stmt.values()), {}) if isinstance(income_stmt, dict) and income_stmt else {}
            latest_balance = next(iter(balance_sheet.values()), {}) if isinstance(balance_sheet, dict) and balance_sheet else {}
            latest_cash = next(iter(cash_flow.values()), {}) if isinstance(cash_flow, dict) and cash_flow else {}

            company_info = {
                "symbol": api_ticker,
                "isactive": not gen.get("IsDelisted", False),
                "longname": gen.get("Name"),
                "shortname": gen.get("Name"),
                "displayname": gen.get("Name"),
                "exchange": gen.get("Exchange"),
                "currency": gen.get("CurrencyCode"),
                "financialcurrency": gen.get("CurrencyCode"),
                "country": gen.get("CountryName"),
                "industry": gen.get("Industry"),
                "sector": gen.get("Sector"),
                "website": gen.get("WebURL"),
                "phone": gen.get("Phone"),
                "fulltimeemployees": gen.get("FullTimeEmployees"),
                "longbusinesssummary": gen.get("Description"),
                "ipoexpecteddate": parse_date(gen.get("IPODate")),
                "typedisp": gen.get("Type"),
                "address1": addr.get("Street") or gen.get("Address"),
                "city": addr.get("City"),
                "state": addr.get("State"),
                "zip": addr.get("ZIP"),
                "marketcap": high.get("MarketCapitalization"),
                "ebitda": high.get("EBITDA"),
                "trailingpe": val.get("TrailingPE") or high.get("PERatio"),
                "forwardpe": val.get("ForwardPE"),
                "trailingpegratio": high.get("PEGRatio"),
                "targetmeanprice": high.get("WallStreetTargetPrice"),
                "bookvalue": high.get("BookValue"),
                "pricetobook": val.get("PriceBookMRQ"),
                "pricetosalestrailing12months": val.get("PriceSalesTTM"),
                "enterprisevalue": val.get("EnterpriseValue"),
                "enterprisetorevenue": val.get("EnterpriseValueRevenue"),
                "enterprisetoebitda": val.get("EnterpriseValueEbitda"),
                "profitmargins": high.get("ProfitMargin"),
                "operatingmargins": high.get("OperatingMarginTTM"),
                "returnonassets": high.get("ReturnOnAssetsTTM"),
                "returnonequity": high.get("ReturnOnEquityTTM"),
                "revenuepershare": high.get("RevenuePerShareTTM"),
                "revenuegrowth": high.get("QuarterlyRevenueGrowthYOY"),
                "revenuegrowth_quarterly_yoy": high.get("QuarterlyRevenueGrowthYOY"),
                "epstrailingtwelvemonths": high.get("DilutedEpsTTM") or high.get("EarningsShare"),
                "earningsgrowth": high.get("QuarterlyEarningsGrowthYOY"),
                "earningsgrowth_quarterly_yoy": high.get("QuarterlyEarningsGrowthYOY"),
                "sharesoutstanding": shares.get("SharesOutstanding"),
                "floatshares": shares.get("SharesFloat"),
                "heldpercentinsiders": shares.get("PercentInsiders"),
                "heldpercentinstitutions": shares.get("PercentInstitutions"),
                "sharesshort": shares.get("SharesShort"),
                "sharesshortpriormonth": shares.get("SharesShortPriorMonth"),
                "shortratio": shares.get("ShortRatio"),
                "shortpercentoffloat": shares.get("ShortPercentFloat") or shares.get("ShortPercent"),
                "sharespercentsharesout": shares.get("ShortPercentOutstanding"),
                "beta": tech.get("Beta"),
                "fiftytwoweekhigh": tech.get("52WeekHigh"),
                "fiftytwoweeklow": tech.get("52WeekLow"),
                "fiftydayaverage": tech.get("50DayMA"),
                "twohundreddayaverage": tech.get("200DayMA"),
                "dividendrate": splits_div.get("ForwardAnnualDividendRate"),
                "dividendyield": splits_div.get("ForwardAnnualDividendYield"),
                "payoutratio": splits_div.get("PayoutRatio"),
                "dividendpayoutratio": splits_div.get("PayoutRatio"),
                "dividenddate": parse_date(splits_div.get("DividendDate")),
                "exdividenddate": parse_date(splits_div.get("ExDividendDate")),
                "lastdividenddate": parse_date(splits_div.get("DividendDate")),
                "lastdividendvalue": high.get("DividendShare"),
                "lastsplitfactor": splits_div.get("LastSplitFactor"),
                "lastsplitdate": parse_date(splits_div.get("LastSplitDate")),
                "mostrecentquarter": parse_date(high.get("MostRecentQuarter")),
                "epscurrentyear": high.get("EPSEstimateCurrentYear"),
                "epsforward": high.get("EPSEstimateNextYear"),
                # Financial statement extracts mapped to int/BigInteger
                "totalrevenue": parse_number(high.get("RevenueTTM") or latest_income.get("totalRevenue"), int),
                "grossprofits": parse_number(high.get("GrossProfitTTM") or latest_income.get("grossProfit"), int),
                "freecashflow": parse_number(latest_cash.get("freeCashFlow"), int),
                "operatingcashflow": parse_number(latest_cash.get("totalCashFromOperatingActivities"), int),
                "totaldebt": parse_number(latest_balance.get("netDebt") or latest_balance.get("longTermDebt"), int),
                "totalcash": parse_number(latest_balance.get("cash"), int),            }
            payload["info"] = {k: v for k, v in company_info.items() if v is not None}

            # ----------------------------------------------------
            # 3. Company Officers Mapping -> CompanyOfficer
            # ----------------------------------------------------
            officers_data = gen.get("Officers", {})
            if isinstance(officers_data, dict):
                current_year = datetime.now().year
                officer_list = []
                for _, off in officers_data.items():
                    if isinstance(off, dict):
                        year_born = off.get("YearBorn")
                        age = (current_year - int(year_born)) if year_born and str(year_born).isdigit() else None
                        officer_list.append({
                            "name": off.get("Name"),
                            "title": off.get("Title"),
                            "yearborn": year_born,
                            "age": age
                        })
                payload["officers"] = officer_list

            # ----------------------------------------------------
            # 4. Financial Statements
            # ----------------------------------------------------
            if fin:
                payload_mapping = {
                    "annual_balance_sheet": ("Balance_Sheet", "yearly"),
                    "annual_income_statement": ("Income_Statement", "yearly"),
                    "annual_cash_flow": ("Cash_Flow", "yearly"),
                    "quarterly_balance_sheet": ("Balance_Sheet", "quarterly"),
                    "quarterly_income_statement": ("Income_Statement", "quarterly"),
                    "quarterly_cash_flow": ("Cash_Flow", "quarterly")
                }
                for payload_key, (stmt_type, period) in payload_mapping.items():
                    statement_source = fin.get(stmt_type, {}).get(period, {})
                    payload["financials"][payload_key] = self._process_eodhd_statement(statement_source)

            # ----------------------------------------------------
            # 5. Holders Mapping
            # ----------------------------------------------------
            holders = fund_data.get("Holders", {})
            for h_type in ['Institutions', 'Funds']:
                h_data = holders.get(h_type, {})
                if isinstance(h_data, dict):
                    rows = []
                    for _, val in h_data.items():
                        if isinstance(val, dict):
                            rows.append({
                                "holder_name": val.get("name"),
                                "date_reported": val.get("date"),
                                "shares": val.get("currentShares") or val.get("totalShares"),
                                "percent_out": val.get("change_p"),
                                "holder_type": "institutional" if h_type == "Institutions" else "mutualfund"
                            })
                    if h_type == 'Institutions':
                        payload['institutional_holders'] = pd.DataFrame(rows)
                    else:
                        payload['mutualfund_holders'] = pd.DataFrame(rows)

            # ----------------------------------------------------
            # 6. Earnings, Analyst Estimates & EPS Trends
            # ----------------------------------------------------
            earnings = fund_data.get("Earnings", {})
            if isinstance(earnings, dict):
                # AnalystEarningsHistory
                hist = earnings.get("History", {})
                if isinstance(hist, dict):
                    hist_list = []
                    for date_str, metrics in hist.items():
                        if isinstance(metrics, dict):
                            hist_list.append({
                                "report_date": date_str,
                                "eps_actual": metrics.get("epsActual"),
                                "eps_estimate": metrics.get("epsEstimate"),
                                "eps_difference": metrics.get("epsDifference"),
                                "surprise_percent": metrics.get("surprisePercent")
                            })
                    payload["analyst_sentiment"]["earnings_history"] = pd.DataFrame(hist_list)

                # AnalystEarningsEstimate, AnalystRevenueEstimate, AnalystEpsTrend, AnalystEpsRevisions
                trend = earnings.get("Trend", {})
                if isinstance(trend, dict):
                    ee_rows, re_rows, trend_rows, rev_rows = [], [], [], []
                    for period_lbl, t_data in trend.items():
                        if isinstance(t_data, dict):
                            # AnalystEarningsEstimate
                            ee_rows.append({
                                "period_label": period_lbl,
                                "num_analysts": t_data.get("earningsEstimateNumberOfAnalysts"),
                                "avg_estimate": t_data.get("earningsEstimateAvg"),
                                "low_estimate": t_data.get("earningsEstimateLow"),
                                "high_estimate": t_data.get("earningsEstimateHigh"),
                                "year_ago_eps": t_data.get("earningsEstimateYearAgoEps"),
                                "eps_growth_percent": t_data.get("earningsEstimateGrowth")
                            })
                            # AnalystRevenueEstimate
                            re_rows.append({
                                "period_label": period_lbl,
                                "num_analysts": t_data.get("revenueEstimateNumberOfAnalysts"),
                                "avg_estimate": t_data.get("revenueEstimateAvg"),
                                "low_estimate": t_data.get("revenueEstimateLow"),
                                "high_estimate": t_data.get("revenueEstimateHigh"),
                                "year_ago_revenue": t_data.get("revenueEstimateYearAgoRevenue"),
                                "revenue_growth_percent": t_data.get("revenueEstimateGrowth")
                            })
                            # AnalystEpsTrend
                            trend_rows.append({
                                "period_label": period_lbl,
                                "current_estimate": t_data.get("earningsEstimateAvg"),
                                "seven_days_ago": t_data.get("epsTrend7daysAgo"),
                                "thirty_days_ago": t_data.get("epsTrend30daysAgo"),
                                "sixty_days_ago": t_data.get("epsTrend60daysAgo"),
                                "ninety_days_ago": t_data.get("epsTrend90daysAgo")
                            })
                            # AnalystEpsRevisions
                            rev_rows.append({
                                "period_label": period_lbl,
                                "up_last_7_days": t_data.get("epsRevisionsUpLast7days"),
                                "up_last_30_days": t_data.get("epsRevisionsUpLast30days"),
                                "down_last_7_days": t_data.get("epsRevisionsDownLast7days"),
                                "down_last_30_days": t_data.get("epsRevisionsDownLast30days")
                            })

                    payload["analyst_estimates"]["earnings"] = pd.DataFrame(ee_rows)
                    payload["analyst_estimates"]["revenue"] = pd.DataFrame(re_rows)
                    payload["analyst_sentiment"]["eps_trend"] = pd.DataFrame(trend_rows)
                    payload["analyst_sentiment"]["eps_revisions"] = pd.DataFrame(rev_rows)

            # ----------------------------------------------------
            # 7. Insider Transactions & Roster Generation
            # ----------------------------------------------------
            insider_transactions = fund_data.get("InsiderTransactions", {})
            if isinstance(insider_transactions, dict):
                tx_list = []
                roster_map = {}

                for _, val in insider_transactions.items():
                    if isinstance(val, dict):
                        owner_name = val.get("ownerName")
                        tx_date = parse_date(val.get("transactionDate"))
                        tx_type = val.get("transactionAcquiredDisposed")
                        amount = val.get("transactionAmount") or 0
                        price = val.get("transactionPrice") or 0

                        # Transaction Record
                        tx_list.append({
                            "insider_name": owner_name,
                            "shares": amount,
                            "transaction_type": tx_type,
                            "transaction_code": val.get("transactionCode"),
                            "start_date": tx_date,
                            "value": amount * price,
                        })

                        # Track Latest Transaction Date per Insider for InsiderRoster
                        if owner_name:
                            if owner_name not in roster_map or (tx_date and roster_map[owner_name]["most_recent_transaction_date"] is None) or (tx_date and tx_date > roster_map[owner_name]["most_recent_transaction_date"]):                                roster_map[owner_name] = {
                                    "name": owner_name,
                                    "position": val.get("ownerTitle") or "Insider",
                                    "most_recent_transaction": tx_type,
                                    "most_recent_transaction_date": tx_date,
                                    "shares_owned_directly": val.get("postTransactionAmount")
                                }

                payload["insider"]["transactions"] = pd.DataFrame(tx_list)
                payload["insider"]["roster"] = pd.DataFrame(list(roster_map.values()))

            return payload

        except Exception as e:
            logger.error(f"EODHD Provider Error for {api_ticker}: {e}", exc_info=True)
            return {"error": str(e)}

    def get_historical_data(self, ticker: str, start_date: str = None, end_date: str = None, interval: str = '1d') -> pd.DataFrame:
        return self.get_ticker_data(ticker, start_date, end_date, interval).get("price_history", pd.DataFrame())

    def get_company_info(self, ticker: str) -> Dict[str, Any]:
        return self.get_ticker_data(ticker).get("info", {})

    def get_financial_statements(self, ticker: str, statement_type: str, frequency: str) -> pd.DataFrame:
        full_payload = self.get_ticker_data(ticker)
        key = f"{frequency}_{statement_type}"
        return full_payload.get("financials", {}).get(key, pd.DataFrame())

    def get_extra_competitors(self, ticker: str) -> Union[list, dict]:
        logger.info(f"Fetching competitors for {ticker} from EODHD")
        related_url = f"{self.base_url}/related/{ticker}"
        try:
            response = self.session.get(related_url, timeout=30)
            response.raise_for_status()
            related_data = response.json()
            if isinstance(related_data, list):
                return related_data
            return []
        except Exception as e:
            logger.error(f"EODHD Competitor Fetch Error for {ticker}: {e}")
            return {"error": f"Could_not_fetch_competitors: {str(e)}"}

    def get_news_content(self, ticker: str) -> str:
        logger.info(f"Fetching news for {ticker} from EODHD")
        news_url = f"{self.base_url}/news?s={ticker}&limit=10"
        try:
            response = self.session.get(news_url, timeout=30)
            response.raise_for_status()
            news_data = response.json()
            text = ""
            if isinstance(news_data, list):
                for item in news_data:
                    text += f"{item['date']}: {item.get('title', '')}. {item.get('content', '')}\n"
            return text
        except Exception as e:
            logger.error(f"EODHD News Fetch Error for {ticker}: {e}")
            return {"error": f"Could_not_fetch_news: {str(e)}"}
        