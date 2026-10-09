"""
This module contains the FinancialAnalysisTool class for performing fundamental financial analysis,
calculating a wide range of metrics, and generating comprehensive reports.
"""
import logging
from typing import Union
import numpy as np
from sqlalchemy import func
import pandas as pd
from langchain_core.prompts import PromptTemplate
from markdown_pdf import MarkdownPdf, Section

from core.db import get_db
from core.model import Company, object_as_dict
from tools.file_wrapper import generate_filename
from tools.canslim_analysis_tool import get_stock_trend
from tools.scanner_tool import calculate_and_save_common_values, find_relative_strength_percentile, find_top_competitors
from tools.sentiment_tool import analyze_sentiment
from tools.data_manager import DataManager

logger = logging.getLogger(__name__)

# --- Constants for Metric Mapping and Selection ---

METRIC_TO_DB_COLUMN_MAP = {
    "Revenue Growth (YOY)": ("revenuegrowth_quarterly_yoy", 100),
    "EPS Growth (YOY)": ("earningsgrowth_quarterly_yoy", 100),
    "3-Year EPS CAGR": ("eps_cagr_3year", 100),
    "Return on Equity (ROE)": ("returnonequity", 100),
    "Debt-to-Equity Ratio": ("debttoequity", 1),
    "P/E Ratio": ("trailingpe", 1),
    "P/B Ratio": ("pricetobook", 1),
    "P/S Ratio": ("pricetosalestrailing12months", 1),
    "Institutional Ownership": ("heldpercentinstitutions", 100),
    "Average Daily Volume": ("averagevolume", 1),
    "Shares Outstanding": ("sharesoutstanding", 1),
    "Current Ratio": ("currentratio", 1),
    "Quick Ratio": ("quickratio", 1),
    "Gross Margin": ("grossmargins", 100),
    "Operating Margin": ("operatingmargins", 100),
    "Profit Margin": ("profitmargins", 100),
    "Free Cash Flow": ("freecashflow", 1),
    "Total Debt": ("totaldebt", 1),
    "Total Cash": ("totalcash", 1),
    "Revenue Per Share": ("revenuepershare", 1),
    "Return on Assets": ("returnonassets", 100),
    "Forward P/E Ratio": ("forwardpe", 1),
    "Trailing EPS": ("trailingeps", 1),
    "Forward EPS": ("forwardeps", 1),
    "Enterprise Value": ("enterprisevalue", 1),
    "Enterprise to Revenue": ("enterprisetorevenue", 1),
    "Enterprise to EBITDA": ("enterprisetoebitda", 1),
    "Dividend Yield": ("dividendyield", 1),
    "52 Week Change (%)": ("_52weekchange", 100),
    "S&P 52 Week Change (%)": ("sandp52weekchange", 100),
    "Expanding Volume": ("expanding_volume", 1),
    "Price Relative to 52-Week High (%)": ("price_relative_to_52week_high", 1),
    "Consecutive Quarters Revenue Growth": ("consecutive_quarters_revenue_growth", 1),
    "Consecutive Quarters EPS Growth": ("consecutive_quarters_eps_growth", 1),
    "Consecutive Quarters OpMargin Improvement": ("consecutive_quarters_opmargin_improvement", 1),
    "Revenue Growth Acceleration (QoQ YOY)": ("revenue_growth_acceleration_qoq_yoy", 100),
    "EPS Growth Acceleration (QoQ YOY)": ("eps_growth_acceleration_qoq_yoy", 100),
    "OpMargin Improvement Acceleration (QoQ YOY)": ("opmargin_improvement_acceleration_qoq_yoy", 100),
    "FCF Growth (QoQ)": ("fcf_growth_qoq", 100),
    "Share Change Rate (QoQ)": ("share_change_rate_qoq", 100),
    "Consecutive Quarters Share Reduction": ("consecutive_quarters_share_reduction", 1),
}

class FinancialAnalysisTool:
    def __init__(self):
        self.fundamental_criteria = {
            "Revenue Growth (YOY)": {
                "min": 10,
                "explanation": "Year-over-year revenue growth is {value:.2f}%. A growth rate above 10% is generally considered healthy.",
            },
            "EPS Growth (YOY)": {
                "min": 10,
                "explanation": "Year-over-year earnings per share (EPS) growth is {value:.2f}%. A growth rate above 10% is generally considered healthy.",
            },
            "3-Year EPS CAGR": {
                "min": 10,
                "explanation": "The 3-year compound annual growth rate (CAGR) of EPS is {value:.2f}%. A growth rate above 10% indicates consistent earnings growth.",
            },
            "Return on Equity (ROE)": {
                "min": 15,
                "explanation": "Return on equity (ROE) is {value:.2f}%. An ROE above 15% is generally considered good.",
            },
            "Return on Assets (ROA)": {
                "min": 5,
                "explanation": "Return on assets (ROA) is {value:.2f}%. An ROA above 5% is generally considered good.",
            },
            "Debt-to-Equity Ratio": {
                "max": 1.5,
                "explanation": "The debt-to-equity ratio is {value:.2f}. A ratio below 1.5 indicates a reasonable level of debt.",
            },
            "Current Ratio": {
                "min": 1.5,
                "explanation": "The current ratio is {value:.2f}. A ratio above 1.5 indicates good short-term liquidity.",
            },
            "Quick Ratio": {
                "min": 1,
                "explanation": "The quick ratio is {value:.2f}. A ratio above 1 indicates good short-term liquidity, even without inventory.",
            },
            "Gross Margin": {
                "min": 40,
                "explanation": "The gross margin is {value:.2f}%. A higher gross margin indicates better profitability.",
            },
            "Operating Margin": {
                "min": 10,
                "explanation": "The operating margin is {value:.2f}%. A higher operating margin indicates better operational efficiency.",
            },
            "Profit Margin": {
                "min": 10,
                "explanation": "The profit margin is {value:.2f}%. A higher profit margin indicates better overall profitability.",
            },
            "P/E Ratio": {
                "min": 0,
                "max": 30,
                "explanation": "The price-to-earnings (P/E) ratio is {value:.2f}. A P/E ratio between 0 and 30 is generally considered acceptable, but it should be compared to industry peers.",
            },
            "Forward P/E Ratio": {
                "min": 0,
                "max": 30,
                "explanation": "The forward price-to-earnings (P/E) ratio is {value:.2f}. A forward P/E ratio between 0 and 30 is generally considered acceptable, but it should be compared to industry peers.",
            },
            "P/B Ratio": {
                "min": 0,
                "max": 5,
                "explanation": "The price-to-book (P/B) ratio is {value:.2f}. A P/B ratio between 0 and 5 is generally considered acceptable, but it should be compared to industry peers.",
            },
            "P/S Ratio": {
                "min": 0,
                "max": 10,
                "explanation": "The price-to-sales (P/S) ratio is {value:.2f}. A P/S ratio between 0 and 10 is generally considered acceptable, but it should be compared to industry peers.",
            },
            "Enterprise to Revenue": {
                "min": 0.1,
                "max": 15,
                "explanation": "The enterprise value to revenue ratio is {value:.2f}. A ratio between 0.1 and 15 is generally considered acceptable, but it should be compared to industry peers.",
            },
            "Enterprise to EBITDA": {
                "min": 0.1,
                "max": 25,
                "explanation": "The enterprise value to EBITDA ratio is {value:.2f}. A ratio between 0.1 and 25 is generally considered acceptable, but it should be compared to industry peers.",
            },
            "Dividend Yield": {
                "min": 2,
                "explanation": "The dividend yield is {value:.2f}%. A higher dividend yield may indicate a good income investment, but it should be compared to industry peers.",
            },
            "Free Cash Flow": {
                "min": 0,
                "explanation": "The free cash flow is {value:.2f}. Positive free cash flow is generally a good sign.",
            },
            "Total Debt": {
                "max": 10000000000,
                "explanation": "The total debt is {value:.2f}. Lower debt is generally better, but it depends on the company's size and industry.",
            },
            "Total Cash": {
                "min": 0,
                "explanation": "The total cash is {value:.2f}. Higher cash reserves are generally better.",
            },
            "Revenue Per Share": {
                "min": 0,
                "explanation": "The revenue per share is {value:.2f}. Higher revenue per share is generally better.",
            },
            "Institutional Ownership": {
                "min": 15,
                "max": 70,
                "explanation": "Institutional investors own {value:.2f}% of the company's shares. This is {comparison} the ideal range of 15% to 70%."
            },
            "Expanding Volume": {
                True: "Yes",
                False: "No",
                "explanation": "Recent trading volume is {comparison} showing expansion compared to longer-term averages."
            },
            "Average Daily Volume": {
                "min": 100000,
                "explanation": "The average daily trading volume is {value:,.0f} shares, {comparison} the minimum of 100,000 shares."
            },
            "Shares Outstanding": {
                "explanation": "There are {value:,.0f} shares outstanding."
            },
            "Sentiment": {
                "explanation": "The overall sentiment towards the stock is {value}."
            },
            "Market Posture (SPY Trend)": {
                "explanation": "The current market posture, based on the SPY trend, is {value}."
            },
            "52 Week Change (%)": {
                "min": -50,
                "explanation": "The stock's 52-week change is {value:.2f}%."
            },
            "S&P 52 Week Change (%)": {
                "min": -50,
                "explanation": "The S&P 500's 52-week change is {value:.2f}%."
            },
            "Price Relative to 52-Week High (%)": {
                "min": 50,
                "explanation": "The stock's price is {value:.2f}% of its 52-week high."
            },
            "Relative Strength Percentile (1 year)": {
                "min": 70,
                "explanation": "The stock's Relative Strength Percentile over the past year is {value:.2f}%, {comparison} the target of 70% or higher."
            },
            "Relative Strength Percentile (6 months)": {
                "min": 70,
                "explanation": "The stock's Relative Strength Percentile over the past 6 months is {value:.2f}%, {comparison} the target of 70% or higher."
            },
            "Relative Strength Percentile (3 months)": {
                "min": 70,
                "explanation": "The stock's Relative Strength Percentile over the past 3 months is {value:.2f}%, {comparison} the target of 70% or higher."
            },

            # --- NEW TREND METRICS ---
            "Consecutive Quarters Revenue Growth": {
                "min": 1, # At least 1 quarter of growth
                "explanation": "Revenue has grown consecutively for {value:.0f} quarter(s). Consistency {comparison} the minimum of 1 quarter.",
            },
            "Consecutive Quarters EPS Growth": {
                "min": 1, # At least 1 quarter of growth
                "explanation": "EPS has grown consecutively for {value:.0f} quarter(s). Consistency {comparison} the minimum of 1 quarter.",
            },
            "Consecutive Quarters OpMargin Improvement": {
                "min": 1, # At least 1 quarter of improvement
                "explanation": "Operating Margin has improved consecutively for {value:.0f} quarter(s). Consistency {comparison} the minimum of 1 quarter.",
            },
            "Revenue Growth Acceleration (QoQ YOY)": {
                "min": 0, # Growth rate is accelerating or stable
                "explanation": "Quarterly YOY revenue growth acceleration is {value:.2f}%. Positive value indicates acceleration.",
            },
            "EPS Growth Acceleration (QoQ YOY)": {
                "min": 0, # Growth rate is accelerating or stable
                "explanation": "Quarterly YOY EPS growth acceleration is {value:.2f}%. Positive value indicates acceleration.",
            },
            "Operating Margin Improvement Acceleration (QoQ YOY)": {
                "min": 0, # Improvement rate is accelerating or stable
                "explanation": "Quarterly YOY Operating Margin improvement acceleration is {value:.2f}%. Positive value indicates acceleration.",
            },
            "FCF Growth (QoQ)": {
                "min": 0, # Positive FCF growth quarter-over-quarter
                "explanation": "Quarter-over-quarter Free Cash Flow growth is {value:.2f}%. Positive growth is desirable.",
            },
            # NOTE: Commented out metrics needing at least 8 consecutive quarters' data, but yfinance only has 5 quarterly and 4 annual financial data
            # "FCF Growth (TTM YOY)": {
            #     "min": 0, # Positive TTM FCF growth year-over-year
            #     "explanation": "Trailing Twelve Month Free Cash Flow growth YOY is {value:.2f}%. Positive growth is desirable.",
            "Share Change Rate (QoQ)": {
                "max": 0, # Negative value indicates reduction (buyback)
                "explanation": "Quarter-over-quarter change in shares outstanding is {value:.2f}%. A negative value indicates share buybacks.",
            },
            "Consecutive Quarters Share Reduction": {
                "min": 1, # At least 1 quarter of reduction
                "explanation": "Shares outstanding have been reduced consecutively for {value:.0f} quarter(s). Consistency {comparison} the minimum of 1 quarter.",
            },
        }


    def calculate_fundamental_metrics_from_db(self, ticker):
        """
        Calculates a broader set of fundamental metrics for a stock using data from the database.

        Args:
            ticker: The company ticker.

        Returns:
            A dictionary containing the fundamental analysis results, or an error message.
        """
        db = next(get_db())
        try:
            results = {}

            result = DataManager().load_ticker_data(ticker, refresh=False)
            if not result:
                raise ValueError(f"No data found for {ticker}")

            company = result['company']
            if not company['last_price_date']:  # scan not yet run for this company
                calculate_and_save_common_values(db, [company['id']], False)
                find_relative_strength_percentile(db, company['id'])
                db.query(Company).filter(Company.id == company['id']).update({Company.last_price_date: func.current_date()}, synchronize_session=False)
                db.commit()
                company = object_as_dict(db.query(Company).filter(Company.id == company['id']).first())

            # --- Refactored Metric Calculation ---
            for metric_name, (db_column, multiplier) in METRIC_TO_DB_COLUMN_MAP.items():
                value = company.get(db_column)
                if value is not None:
                    results[metric_name] = value * multiplier
                else:
                    results[metric_name] = float('nan')

            relative_strength_percentile = find_relative_strength_percentile(db, company['id'])
            results.update(relative_strength_percentile)

            return results
        except Exception as e:
            logger.error(f"Error calculating fundamental metrics for {ticker}: {e}", exc_info=True)
            return {"error": f"An unexpected error occurred: {str(e)}"}
        finally:
            db.close()


    def calculate_financial_ratios_from_db(self, ticker: str) -> dict:
        """
        Calculates key financial ratios for a company using data from the database.

        Args:
            ticker (str): The company's ticker symbol.

        Returns:
            dict: A dictionary containing the calculated financial ratios, or an error message.
        """
        # Define the keys that belong to financial ratios
        FINANCIAL_RATIOS_KEYS = [
            "Gross Margin", "Operating Margin", "Return on Equity (ROE)", "Return on Assets",
            "Profit Margin", "Trailing EPS", "Dividend Payout Ratio", "Debt-to-Equity Ratio",
            "Debt-to-Assets Ratio", "Interest Coverage Ratio", "Cash Flow to Debt Ratio",
            "Asset Turnover", "Inventory Turnover Ratio", "Days Payable Outstanding (DPO)",
            "Days Sales Outstanding (DSO)", "Current Ratio", "Quick Ratio", "Revenue Growth (YOY)",
            "Net Income Growth", "EPS Growth (YOY)", "Forward P/E Ratio", "P/E Ratio",
            "P/S Ratio", "P/B Ratio", "Dividend Yield", "Enterprise Value",
            "Enterprise to Revenue", "Enterprise to EBITDA"
        ]
        try:
            results = self.calculate_fundamental_metrics_from_db(ticker)
            if isinstance(results, dict) and "error" in results:
                return results

            # Filter the results to include only the specified ratio keys
            ratios = {key: results.get(key) for key in FINANCIAL_RATIOS_KEYS if key in results}
            # Rename for consistency if needed, e.g., Gross Profit Margin
            if "Gross Margin" in ratios:
                ratios["Gross Profit Margin"] = ratios.pop("Gross Margin")

            return ratios
        except Exception as e:
            logger.error(f"Error calculating financial ratios for {ticker}: {e}", exc_info=True)
            return {"error": f"Error calculating ratios from DB: {e}"}


    def analyze_financial_health_from_db(self, ticker: str) -> dict:
        """
        Analyzes the overall financial health of a company based on calculated ratios from the database.

        Args:
            ticker (str): The company's ticker symbol.

        Returns:
            dict: A dictionary containing the financial health assessment, or an error message.
        """
        try:
            ratios = self.calculate_financial_ratios_from_db(ticker)
            if "error" in ratios:
                return ratios  # Return error if ratios calculation failed

            health_assessment = {}

            # Liquidity
            if ratios.get("Current Ratio", 0) >= 1.5:
                health_assessment["Current Ratio"] = "Healthy"
            elif ratios.get("Current Ratio", 0) >= 1:
                health_assessment["Current Ratio"] = "Acceptable"
            else:
                health_assessment["Current Ratio"] = "Concerning"

            if ratios.get("Quick Ratio", 0) >= 1:
                health_assessment["Quick Ratio"] = "Healthy"
            else:
                health_assessment["Quick Ratio"] = "Concerning"

            # Solvency
            if ratios.get("Debt-to-Equity Ratio", 0) <= 1:
                health_assessment["Debt-to-Equity Ratio"] = "Healthy"
            elif ratios.get("Debt-to-Equity Ratio", 0) <= 2:
                health_assessment["Debt-to-Equity Ratio"] = "Acceptable"
            else:
                health_assessment["Debt-to-Equity Ratio"] = "Concerning"

            if ratios.get("Interest Coverage Ratio", 0) >= 3:
                health_assessment["Interest Coverage Ratio"] = "Healthy"
            elif ratios.get("Interest Coverage Ratio", 0) >= 1.5:
                health_assessment["Interest Coverage Ratio"] = "Acceptable"
            else:
                health_assessment["Interest Coverage Ratio"] = "Concerning"

            # Profitability
            if ratios.get("Return on Equity (ROE)", 0) >= 15:
                health_assessment["Return on Equity (ROE)"] = "Healthy"
            elif ratios.get("Return on Equity (ROE)", 0) >= 10:
                health_assessment["Return on Equity (ROE)"] = "Acceptable"
            else:
                health_assessment["Return on Equity (ROE)"] = "Concerning"

            if ratios.get("Profit Margin", 0) >= 10:
                health_assessment["Profit Margin"] = "Healthy"
            elif ratios.get("Profit Margin", 0) >= 5:
                health_assessment["Profit Margin"] = "Acceptable"
            else:
                health_assessment["Profit Margin"] = "Concerning"

            # Growth
            if ratios.get("Revenue Growth", 0) >= 10:
                health_assessment["Revenue Growth"] = "Healthy"
            elif ratios.get("Revenue Growth", 0) >= 5:
                health_assessment["Revenue Growth"] = "Acceptable"
            else:
                health_assessment["Revenue Growth"] = "Concerning"

            if ratios.get("EPS Growth", 0) >= 10:
                health_assessment["EPS Growth"] = "Healthy"
            elif ratios.get("EPS Growth", 0) >= 5:
                health_assessment["EPS Growth"] = "Acceptable"
            else:
                health_assessment["EPS Growth"] = "Concerning"

            return health_assessment

        except Exception as e:
            logger.error(f"Error analyzing financial health for {ticker}: {e}", exc_info=True)
            return {"error": f"Error analyzing financial health from DB: {e}"}


    def calculate_valuation_metrics_from_db(self, ticker: str) -> dict:
        """
        Calculates key valuation metrics for a company using data from the database.

        Args:
            ticker (str): The company's ticker symbol.

        Returns:
            dict: A dictionary containing the calculated valuation metrics, or an error message.
        """
        VALUATION_METRICS_KEYS = [
            "P/E Ratio", "Forward P/E Ratio", "P/B Ratio", "P/S Ratio",
            "Enterprise Value", "Enterprise to Revenue", "Enterprise to EBITDA",
            "Dividend Yield"
        ]
        try:
            results = self.calculate_fundamental_metrics_from_db(ticker)
            if isinstance(results, dict) and "error" in results:
                return results

            # Filter the results to include only the specified valuation keys
            valuation_metrics = {key: results.get(key) for key in VALUATION_METRICS_KEYS if key in results}

            return valuation_metrics
        except Exception as e:
            logger.error(f"Error calculating valuation metrics for {ticker}: {e}", exc_info=True)
            return {"error": f"Error calculating valuation metrics from DB: {e}"}


    def generate_fundamental_ratios_table(self, metrics: dict):
        """Generates a table for fundamental ratios with classifications and explanations, skipping None/NaN/Error values."""
        table_data = {
            "Metric": [],
            "Value": [],
            "Pass": [],
            "Explanation": [],
        }

        # Use a copy of the criteria keys to maintain order if desired, or sort them
        # metric_order = sorted(self.fundamental_criteria.keys()) # Sort alphabetically

        for metric in metrics:
            if metric in self.fundamental_criteria: # Check if the metric exists in the selected criteria
                value = metrics[metric]

                # --- Skip if value is None or NaN ---
                if value is None or pd.isna(value):
                    continue # Skip this metric entirely

                criteria = self.fundamental_criteria[metric]
                # Initialize placeholders, but don't append metric name yet
                value_str = "N/A"
                passed = "N/A"
                explanation = "N/A"

                try:
                    # --- Process the value ---
                    if isinstance(value, bool):
                        value_str = "Yes" if value else "No"
                        if True in criteria and False in criteria:
                             passed = "Yes" if value == criteria[True] else "No"
                        else:
                             passed = "Yes" if value else "No"
                    elif isinstance(value, (int, float)):
                        if isinstance(value, int):
                             value_str = "{:,.0f}".format(value)
                        else:
                            value_str = "{:,.2f}".format(value)

                        # Determine pass/fail based on min/max criteria
                        if "min" in criteria and "max" in criteria:
                            passed = "Yes" if criteria["min"] <= value <= criteria["max"] else "No"
                        elif "min" in criteria:
                            passed = "Yes" if value >= criteria["min"] else "No"
                        elif "max" in criteria:
                            passed = "Yes" if value <= criteria["max"] else "No"
                        else:
                            passed = "N/A"
                    else: # Handle other types like strings
                        value_str = str(value)
                        passed = "N/A"

                    # --- Generate explanation ---
                    if "explanation" in criteria:
                        if passed != "N/A":
                            if "max" in criteria and "min" in criteria:
                                comparison = "within" if passed == "Yes" else "outside"
                            elif "max" in criteria:
                                comparison = "below" if passed == "Yes" else "above"
                            elif "min" in criteria:
                                comparison = "above" if passed == "Yes" else "below"
                            elif metric == "Expanding Volume":
                                comparison = "indeed" if value else "not"
                            elif isinstance(value, bool):
                                comparison = "meets" if passed == "Yes" else "does not meet"
                            else:
                                comparison = "meets" if passed == "Yes" else "does not meet"

                            try:
                                explanation = criteria["explanation"].format(value=value, comparison=comparison)
                            except (ValueError, KeyError, IndexError, TypeError) as fmt_e:
                                logger.error(f"Error in explanation string formatting for {metric}: {fmt_e}. Using raw explanation.")
                                explanation = criteria["explanation"]
                            except Exception as fmt_e:
                                logger.error(f"Unexpected error in explanation string formatting for {metric}: {fmt_e}")
                                explanation = criteria["explanation"]
                        else: # If pass is N/A
                            try:
                                explanation = criteria["explanation"].format(value=value)
                            except Exception as fmt_e:
                                explanation = criteria.get("explanation", "No explanation available.").split("{")[0].strip()

                    # --- If processing succeeds, append all data for this metric ---
                    table_data["Metric"].append(metric) # Append metric name *here*
                    table_data["Value"].append(value_str)
                    table_data["Pass"].append(passed)
                    table_data["Explanation"].append(explanation)

                except (ValueError, TypeError) as e:
                    # --- Skip if ValueError or TypeError occurs during processing ---
                    logger.error(f"Skipping metric {metric} due to processing error: {e}. Value: {value}")
                    continue # Skip appending anything for this metric

                # except Exception as e: # Optional: Catch other unexpected errors if needed
                #     logger.error(f"Skipping metric {metric} due to unexpected error: {e}. Value: {value}")
                #     continue

            # else: # Optional: Log if a metric from criteria is missing in calculated metrics
            #     logger.debug(f"Metric '{metric}' not found in calculated metrics for this ticker.")

        # Create DataFrame only with successfully processed metrics
        df = pd.DataFrame(table_data)
        return df.to_markdown(index=False)



    def build_fundamental_report_llm_input(self, ticker: str) -> dict:
        """
        Builds the input for the LLM-based fundamental analysis report.
        
        Args:
            ticker (str): The company's ticker symbol.

        Returns:
            dict: The input data for the LLM.
        """
        # 1. Get Fundamental Metrics from DB
        fundamental_data = self.calculate_fundamental_metrics_from_db(ticker)
        if isinstance(fundamental_data, dict) and "error" in fundamental_data:
            logger.error(f"Error getting fundamental data for {ticker}: {fundamental_data['error']}")
            return fundamental_data  # Return error if any

        # 2. Get Financial Health Assessment from DB
        financial_health = self.analyze_financial_health_from_db(ticker)
        if isinstance(financial_health, dict) and "error" in financial_health:
            logger.error(f"Error getting financial health for {ticker}: {financial_health['error']}")
            return financial_health  # Return error if any

        # 3. Get Valuation Metrics from DB
        valuation_metrics = self.calculate_valuation_metrics_from_db(ticker)
        if isinstance(valuation_metrics, dict) and "error" in valuation_metrics:
            logger.error(f"Error getting valuation metrics for {ticker}: {valuation_metrics['error']}")
            return valuation_metrics  # Return error if any

        # 4. Get Competitor Data (Refactored for clarity)
        competitor_data = {}
        # Start with the main ticker's data
        main_ticker_metrics = fundamental_data.copy() # Use a copy
        main_ticker_metrics.update(valuation_metrics)
        competitor_data[ticker] = main_ticker_metrics

        extra_competitors = DataManager().get_extra_competitors(ticker)
        if isinstance(extra_competitors, dict) and "error" in extra_competitors:
            logger.warning(f"Error getting competitors for {ticker}: {extra_competitors['error']}")
            extra_competitors = []

        competitors = DataManager().find_top_competitors(ticker, extra_competitors=extra_competitors)
        if competitors:
            for competitor_info in competitors:
                cticker = competitor_info["ticker"]
                company_info = DataManager().load_ticker_data(cticker, refresh=False)
                if not company_info:
                    logger.error(f"No data found for {cticker}")
                    continue

                competitor_metrics = self.calculate_fundamental_metrics_from_db(cticker)
                if isinstance(competitor_metrics, dict) and "error" in competitor_metrics:
                    logger.warning(f"Error getting competitor data for {cticker}: {competitor_metrics['error']}")
                    continue
                competitor_metrics.update(self.calculate_valuation_metrics_from_db(cticker))
                competitor_data[cticker] = competitor_metrics

        # 5. Get Sentiment Analysis
        sentiment = analyze_sentiment(ticker)
        if isinstance(sentiment, dict) and "error" in sentiment:
            logger.warning(f"Error getting sentiment for {ticker}: {sentiment['error']}")

        # 6. Prepare Data for LLM
        fundamental_data["Sentiment"] = sentiment["sentiment"]
        market_posture = get_stock_trend()
        if isinstance(market_posture, dict) and "error" in market_posture:
            logger.warning(f"Error getting market posture: {market_posture['error']}")
        fundamental_data["Market Posture (SPY Trend)"] = market_posture
        fundamental_table = self.generate_fundamental_ratios_table(fundamental_data)
        
        llm_input = {
            "ticker": ticker,
            "fundamental_data": fundamental_data,
            "financial_health": financial_health,
            "valuation_metrics": valuation_metrics,
            "competitor_data": competitor_data,
            "sentiment": sentiment,
            "fundamental_criteria": fundamental_table,
        }
        return llm_input

    def generate_fundamental_report(self, ticker: str, llm) -> Union[str, dict]:
        """
        Generates a comprehensive fundamental analysis report for a given ticker using an LLM.

        Args:
            ticker (str): The company's ticker symbol.
            llm: The language model to use for report generation.

        Returns:
            Union[str, dict]: The path to the generated report file (PDF), or an error message.
        """
        try:
            llm_input = self.build_fundamental_report_llm_input(ticker)
            if isinstance(llm_input, dict) and "error" in llm_input:
                logger.error(f"Error preparing LLM input for {ticker}: {llm_input['error']}")
                return llm_input  # Return error if any

            # Generate Report with LLM
            report_content = self.generate_report_with_llm(llm_input, llm)
            if isinstance(report_content, dict) and "error" in report_content:
                logger.error(f"Error generating report with LLM for {ticker}: {report_content['error']}")
                return report_content

            # Add the table to the report content if not present
            if "Year-over-year earnings per share (EPS) growth" not in report_content:
                fundamental_table = llm_input["fundamental_criteria"]
                if "**Fundamental Criteria Evaluation**" in report_content:
                    report_content = report_content.replace(
                        "**Fundamental Criteria Evaluation**",
                        f"**Fundamental Criteria Evaluation**\n\n{fundamental_table}\n\n"
                    )
                else:
                    report_content = report_content.replace(
                        "Fundamental Criteria Evaluation",
                        f"Fundamental Criteria Evaluation\n\n{fundamental_table}\n\n"
                    )

            # 8. Save Report to File (PDF)
            report_filename = generate_filename(ticker, "fundamental_report", "pdf")
            pdf = MarkdownPdf()
            pdf.meta["title"] = f'Fundamental Analysis for {ticker}'
            pdf.add_section(Section(report_content, toc=False))
            pdf.save(report_filename)

            return report_filename

        except Exception as e:
            logger.error(f"An unexpected error occurred while generating the fundamental report for {ticker}: {e}", exc_info=True)
            return {"error": f"Error generating fundamental report: {e}"}

    def generate_report_with_llm(self, llm_input: dict, llm) -> str:
        """
        Generates the fundamental analysis report content using an LLM.

        Args:
            llm_input (dict): The input data for the LLM.
            llm: The language model to use.

        Returns:
            str: The generated report content.
        """
        try:
            # Replace NaN values with "Not Available"
            for key, value in llm_input.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if pd.isna(sub_value):
                            value[sub_key] = "Not Available"
                elif pd.isna(value):
                    llm_input[key] = "Not Available"

            prompt = PromptTemplate(
                input_variables=llm_input.keys(),
                template=prompt_template,
            )

            chain = prompt | self.llm
            report_content = chain.invoke(llm_input)
            if hasattr(report_content, 'content'):
                return report_content.content

            return report_content

        except Exception as e:
            logger.error(f"Error generating report with LLM: {e}", exc_info=True)
            return f"Error generating report with LLM: {e}"


prompt_template = """
You are an expert financial analyst tasked with creating a comprehensive fundamental analysis report for {ticker}.

**Your Task:**
1.  **Analyze the company's financial health.** Based on the provided financial health assessment, discuss the company's liquidity, solvency, profitability, and growth.
    *   Consider the company's current ratio, quick ratio, debt-to-equity ratio, interest coverage ratio, return on equity, and profit margin.
    *   Analyze trends in these metrics over time, if possible.
    *   Discuss the implications of high debt or low liquidity.
2.  **Evaluate the company's valuation.** Analyze the valuation metrics (P/E, P/S, P/B, etc.) and determine if the company appears overvalued, undervalued, or fairly valued.
    *   Consider both absolute valuation (e.g., is the P/E ratio high or low in general?) and relative valuation (how does it compare to peers?).
    *   Discuss whether a high P/E ratio is justified by high growth or if it might indicate overvaluation.
    *   Consider the company's industry when analyzing its valuation.
3.  **Compare the company to its competitors.** If competitor data is available, compare the company's performance and valuation to its peers.
    *   Focus on the most relevant metrics for the industry.
    *   Highlight key differences in profitability, growth, and valuation.
4.  **Assess the overall sentiment.** Discuss the sentiment analysis results and how they might impact the company's stock price.
5.  **Provide an overall investment recommendation.** Based on your analysis, provide a concise investment recommendation (e.g., Buy, Hold, Sell) and justify your recommendation.
6. **Evaluate the company based on the fundamental criteria.** The table contains the metric, the value, if it pass the criteria, and an explanation. Highlight some criteria of significance.
7. **Conclude with a concise and confident investment summary, highlighting key strengths, weaknesses, and potential opportunities or risks.  Focus on actionable insights based on the analysis.**

**Report Structure:**
Structure your report into clear sections:
- Executive Summary
- Financial Health Analysis
- Valuation Analysis
- Competitor Comparison (if applicable)
- Sentiment Analysis
- Overall Recommendation
- Fundamental Criteria Evaluation

**Important Considerations:**
- Be concise, data-driven, and avoid making predictions about future stock prices. Focus on the current fundamental picture.
- If data for a specific metric is missing, clearly state that and explain why it might be missing (e.g., not applicable for this industry, data not reported).
- Consider the company's industry when analyzing its financial health and valuation. What are the typical ratios for this industry?
- Analyze trends in the company's financial metrics over time, if possible.
- If the company has high debt, discuss the implications for its financial health and future growth.

Here's the data you have:

**Fundamental Data:**
{fundamental_data}

**Financial Health Assessment:**
{financial_health}

**Valuation Metrics:**
{valuation_metrics}

**Competitor Data (if available):**
{competitor_data}

**Sentiment Analysis:**
{sentiment}

**Fundamental Criteria Evaluation:**
{fundamental_criteria}

"""