from datetime import datetime
import logging
import os
import traceback
from typing import Any, Dict, List, Tuple

from langchain_core.prompts import PromptTemplate
from markdown_pdf import MarkdownPdf, Section
from sqlalchemy import func

from core.db import get_db
from core.model import Company, object_as_dict
from load_cfg import WORKING_DIRECTORY
from tools.scanner_tool import (
    calculate_eps_growth_yoy_and_cagr,
    calculate_expanding_volume,
    calculate_price_relative_to_52week_high,
    calculate_revenue_growth_yoy,
    find_relative_strength_percentile,
)
from tools.sentiment_tool import analyze_sentiment
from tools.technical_analysis_tools import TechnicalAnalysisTool, get_stock_trend
from tools.data_manager import DataManager

logger = logging.getLogger(__name__)

class CanslimReportGenerator:
    """
    A class to generate a comprehensive CANSLIM analysis report for a given stock ticker.
    It encapsulates data fetching, metric calculation, LLM-based analysis, and PDF report generation.
    """

    _CANSLIM_CRITERIA = {
        "Revenue Growth (YOY)": {"min": 25, "explanation": "Revenue increased by {value:.2f}% YoY, {comparison} the 25% target."},
        "EPS Growth (YOY)": {"min": 20, "explanation": "EPS increased by {value:.2f}% YoY, {comparison} the 20% target."},
        "3-Year EPS CAGR": {"min": 25, "explanation": "3-Year EPS CAGR is {value:.2f}%, {comparison} the 25% target."},
        "Return on Equity (ROE)": {"min": 17, "explanation": "ROE is {value:.2f}%, {comparison} the 17% target."},
        "Debt-to-Equity Ratio": {"max": 1.5, "explanation": "Debt-to-Equity is {value:.2f}, {comparison} the target of <= 1.5."},
        "Institutional Ownership": {"min": 15, "max": 70, "explanation": "Institutional ownership is {value:.2f}%, {comparison} the 15-70% ideal range."},
        "Average Daily Volume": {"min": 100000, "explanation": "Avg Daily Volume is {value:,.0f}, {comparison} the 100k minimum."},
        "Price Relative to 52-Week High (%)": {"min": 70, "max": 110, "explanation": "Price is {value:.2f}% of its 52-week high, {comparison} the 70-110% range."},
        "Relative Strength Percentile (1 year)": {"min": 70, "explanation": "1-year RS Percentile is {value:.2f}%, {comparison} the 70% target."},
        "Expanding Volume": {True: "Yes", False: "No", "explanation": "Recent volume is {comparison} showing expansion."},
    }

    _INDUSTRY_ANALYSIS_PROMPT_TEMPLATE = """
    You are a financial analyst. Based on the CANSLIM metrics for {ticker} and its competitors, provide an industry analysis.

    **Main Company:** {company_info}
    **Metrics:**
    {main_ticker_metrics_md}

    **Competitors:**
    {competitor_metrics_md}

    **Analysis Task:**
    1.  **Industry Leadership:** Based on the metrics, which company appears to be the leader in this industry group? Justify your answer by comparing key metrics like revenue/EPS growth, ROE, and relative strength.
    2.  **Competitive Position:** How does {ticker} stack up against its main competitors? Is it a leader, a laggard, or in the middle of the pack?
    3.  **Overall Industry Health:** Does this industry group appear to be strong and in favor, based on the collective performance of these companies?

    Provide a concise summary.
    """

    _LLM_PROMPT_TEMPLATE = """
    You are a financial analyst specializing in the CANSLIM investment strategy. Analyze the provided data for {ticker}.

    **Ticker:** {ticker}
    **Company Information:** {company_info}
    **Recent News Snippets:** {company_news}
    **Overall Market Posture (SPY Trend):** {market_posture}
    **Competitive Landscape Summary:**
    {industry_analysis_summary}
    **CANSLIM Metrics Table:**
    {canslim_metrics_md}

    **Analysis Task:**
    Provide a concise and confident analysis based on the data.

    1.  **CANSLIM Summary:** Briefly evaluate the stock against the key CANSLIM criteria (Earnings, Financial Strength, Relative Strength, Volume).
    2.  **"N" Factor (New Things):** Discuss any new products, management changes, or significant news that could act as a catalyst.
    3.  **Investment Thesis:** Conclude with a clear Bull vs. Bear thesis, considering its position within its industry. What are the primary reasons to be bullish or bearish on this stock right now?

    Present your analysis in a structured, easy-to-read format.
    """

    def __init__(self, llm: Any):
        self.llm = llm
        self.tech_analyzer = TechnicalAnalysisTool()

    def generate_report(self, ticker: str) -> str:
        """
        The main public method to generate the full CANSLIM report PDF for a ticker.

        Args:
            ticker: The stock ticker symbol.

        Returns:
            The file path to the generated PDF report.
        """
        report_filename = os.path.join(WORKING_DIRECTORY, f"{ticker}_canslim_report_{datetime.now().strftime('%Y%m%d')}.pdf")
        if os.path.exists(report_filename):
            return report_filename

        chart_files_to_clean = []
        try:
            # 1. Gather all data and metrics
            all_metrics = self._analyze_with_competitors(ticker)
            if "error" in all_metrics.get(ticker, {}):
                raise ValueError(f"Error getting metrics for {ticker}: {all_metrics[ticker]['error']}")

            # 2. Build the markdown content for the report
            report_md, chart_files_to_clean = self._build_markdown_report(ticker, all_metrics)

            # 3. Convert markdown to PDF
            pdf = MarkdownPdf(toc_level=2)
            pdf.meta["title"] = f'CANSLIM Analysis for {ticker}'
            pdf.add_section(Section(report_md, toc=False))
            pdf.save(report_filename)

            return report_filename

        except Exception as e:
            logger.error(f"Failed to generate CANSLIM report for {ticker}: {e}", exc_info=True)
            raise  # Re-raise the exception to be caught by the UI
        finally:
            # 4. Clean up temporary chart files
            if chart_files_to_clean:
                for chart_file in chart_files_to_clean:
                    if chart_file and os.path.exists(chart_file):
                        try:
                            os.remove(chart_file)
                            logger.info(f"Cleaned up temporary file: {chart_file}")
                        except OSError as e:
                            logger.error(f"Error removing temporary file {chart_file}: {e}")

    def _calculate_metrics(self, ticker: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Calculates CANSLIM metrics for a single stock and returns the metrics and company info.

        Returns:
            A tuple containing:
            - A dictionary of calculated metrics.
            - A dictionary of the company's core information.
        """
        db = next(get_db())
        try:
            result = DataManager().load_ticker_data(ticker, refresh=True, price_only=False)
            if isinstance(result, dict) and "error" in result:
                raise ValueError(result['error'])

            company = result.get('company')
            if not company:
                raise ValueError(f"No company data found for {ticker} in the database.")

            # If scan data is stale, recalculate it on the fly
            if not company.get('last_price_date'):
                logger.info(f"Stale data for {ticker}, recalculating metrics...")
                company_id = company['id']
                calculate_revenue_growth_yoy(db, [company_id])
                calculate_eps_growth_yoy_and_cagr(db, [company_id])
                calculate_price_relative_to_52week_high(db, [company_id])
                calculate_expanding_volume(db, [company_id])
                find_relative_strength_percentile(db, company_id)
                db.query(Company).filter(Company.id == company_id).update(
                    {Company.last_price_date: func.current_date()}, synchronize_session=False
                )
                db.commit()
                company = object_as_dict(db.query(Company).filter(Company.id == company_id).first())

            data_unavailable = "N/A"
            metrics = {
                "name": company.get('longname'),
                "Revenue Growth (YOY)": company.get('revenuegrowth_quarterly_yoy', 0) * 100 if company.get('revenuegrowth_quarterly_yoy') is not None else data_unavailable,
                "EPS Growth (YOY)": company.get('earningsgrowth_quarterly_yoy', 0) * 100 if company.get('earningsgrowth_quarterly_yoy') is not None else data_unavailable,
                "3-Year EPS CAGR": company.get('eps_cagr_3year', 0) * 100 if company.get('eps_cagr_3year') is not None else data_unavailable,
                "Return on Equity (ROE)": company.get('returnonequity', 0) * 100 if company.get('returnonequity') is not None else data_unavailable,
                "Debt-to-Equity Ratio": company.get('debttoequity', 0) / 100 if company.get('debttoequity') is not None else data_unavailable,
                "Institutional Ownership": company.get('heldpercentinstitutions', 0) * 100 if company.get('heldpercentinstitutions') is not None else data_unavailable,
                "Average Daily Volume": company.get('averagevolume'),
                "Price Relative to 52-Week High (%)": company.get('price_relative_to_52week_high'),
                "Relative Strength Percentile (1 year)": company.get('relative_strength_percentile_252'),
                "Expanding Volume": company.get('expanding_volume'),
                "Sentiment": analyze_sentiment(ticker),
                "Market Posture (SPY Trend)": get_stock_trend(),
            }
            return metrics, company
        finally:
            db.close()

    def _analyze_with_competitors(self, ticker: str) -> Dict[str, Any]:
        """
        Analyzes the main ticker and its top competitors, returning a structured dictionary.

        Returns:
            A dictionary containing the main ticker's data, a list of competitors,
            and data for each competitor.
            Example: {'BNS.TO': {'metrics': {...}, 'company_info': {...}}, 'competitors': [...], ...}
        """
        metrics, company_info = self._calculate_metrics(ticker)
        results = {ticker: {"metrics": metrics, "company_info": company_info}}
        try:
            competitors = DataManager().find_top_competitors(ticker)
            if competitors:
                results["competitors"] = competitors
                for competitor in competitors:
                    cticker = competitor["ticker"]
                    try:
                        cmetrics, c_info = self._calculate_metrics(cticker)
                        results[cticker] = {"metrics": cmetrics, "company_info": c_info}
                    except Exception as e:
                        logger.warning(f"Could not calculate metrics for competitor {cticker}: {e}")
                        results[cticker] = {"metrics": {"error": str(e)}, "company_info": {}}
        except Exception as e:
            logger.warning(f"Could not analyze competitors for {ticker}: {e}")
        return results

    def _evaluate_metric(self, metric: str, value: Any) -> Tuple[str, str, str]:
        """Evaluates a single metric against its criteria."""
        criteria = self._CANSLIM_CRITERIA.get(metric, {})
        passed = "N/A"
        explanation = "N/A"
        formatted_value = "N/A"

        if value == "N/A" or value is None:
            return "N/A", "N/A", "Data not available."

        try:
            if isinstance(value, bool):
                formatted_value = "Yes" if value else "No"
                passed = "Yes" if value == criteria.get(True) else "No"
                comparison = "indeed" if value else "not"
                explanation = criteria.get("explanation", "").format(comparison=comparison)
            elif isinstance(value, (int, float)):
                formatted_value = f"{value:,.2f}"
                is_pass = True
                if "min" in criteria and value < criteria["min"]: is_pass = False
                if "max" in criteria and value > criteria["max"]: is_pass = False
                passed = "Yes" if is_pass else "No"

                if "max" in criteria and "min" in criteria:
                    comparison = "within" if passed == "Yes" else "outside"
                elif "min" in criteria:
                    comparison = "above" if passed == "Yes" else "below"
                elif "max" in criteria:
                    comparison = "below" if passed == "Yes" else "above"
                else:
                    comparison = ""
                explanation = criteria.get("explanation", "").format(value=value, comparison=comparison)
        except (ValueError, TypeError, KeyError) as e:
            logger.error(f"Error evaluating metric '{metric}' with value '{value}': {e}")
            passed = "Error"
            explanation = "Evaluation error."
            formatted_value = str(value)

        return formatted_value, passed, explanation

    def _create_metrics_table_md(self, metrics: Dict[str, Any]) -> str:
        """Creates a markdown table from the calculated metrics."""
        table_header = "| Metric | Value | Pass | Explanation |\n|---|---|---|---|\n"
        table_rows = []
        for metric, value in metrics.items():
            if metric not in self._CANSLIM_CRITERIA:
                continue
            if isinstance(value, dict) and "error" in value:
                table_rows.append(f"| {metric} | Error | N/A | {value['error']} |")
                continue

            formatted_value, passed, explanation = self._evaluate_metric(metric, value)
            table_rows.append(f"| {metric} | {formatted_value} | {passed} | {explanation} |")
        return table_header + "\n".join(table_rows)

    def _generate_industry_analysis(self, ticker: str, company_info: Dict, main_metrics_md: str, all_data: Dict[str, Any]) -> Tuple[str, str]:
        """Generates a qualitative analysis of the industry and competitive landscape."""
        if "competitors" not in all_data or not all_data["competitors"]:
            return "No competitor data available for industry analysis.", ""

        competitor_md_parts = []
        for competitor in all_data["competitors"]:
            cticker = competitor["ticker"]
            competitor_data = all_data.get(cticker)
            if competitor_data and "error" not in competitor_data.get("metrics", {}):
                cmetrics = competitor_data["metrics"]
                cinfo = competitor_data["company_info"]
                competitor_md_parts.append(f"**{cticker} ({cinfo.get('longname', cticker)})**\n")
                competitor_md_parts.append(self._create_metrics_table_md(cmetrics))
            elif competitor_data:
                competitor_md_parts.append(f"**{cticker}**\nCould not retrieve metrics.\n")
        
        if not competitor_md_parts:
            return "Could not generate metrics for competitors.", ""

        competitor_metrics_md = "\n\n".join(competitor_md_parts)

        try:
            prompt = PromptTemplate(
                template=self._INDUSTRY_ANALYSIS_PROMPT_TEMPLATE,
                input_variables=["ticker", "company_info", "main_ticker_metrics_md", "competitor_metrics_md"]
            )
            chain = prompt | self.llm
            analysis = chain.invoke({
                "ticker": ticker,
                "company_info": str(company_info),
                "main_ticker_metrics_md": main_metrics_md,
                "competitor_metrics_md": competitor_metrics_md
            })
            if hasattr(analysis, 'content'):
                analysis = analysis.content
            return analysis, competitor_metrics_md
        except Exception as e:
            logger.error(f"Error generating industry analysis for {ticker}: {e}")
            traceback.print_exc()
            return "Error: Could not generate industry analysis.", ""

    def _generate_llm_summary(self, ticker: str, company_info: Dict, metrics: Dict, metrics_md: str, industry_analysis_summary: str) -> str:
        """Generates the main CANSLIM summary using the language model."""
        try:
            company_news = DataManager().get_news_content(ticker)
            company_news_str = "Recent news not available."
            if isinstance(company_news, list):
                news_items = [f"- {n['title']}" for n in company_news[:5] if isinstance(n, dict) and 'title' in n]
                if news_items:
                    company_news_str = "\n".join(news_items)
            elif isinstance(company_news, dict) and "error" in company_news:
                logger.warning(f"Could not get news for {ticker}: {company_news['error']}")
            else:
                company_news_str = str(company_news)

            prompt = PromptTemplate(
                template=self._LLM_PROMPT_TEMPLATE,
                input_variables=["ticker", "company_info", "company_news", "market_posture", "industry_analysis_summary", "canslim_metrics_md"]
            )
            chain = prompt | self.llm

            summary = chain.invoke({
                "ticker": ticker,
                "company_info": str(company_info),
                "company_news": company_news_str,
                "market_posture": metrics.get("Market Posture (SPY Trend)", "N/A"),
                "industry_analysis_summary": industry_analysis_summary,
                "canslim_metrics_md": metrics_md
            })
            if hasattr(summary, 'content'):
                summary = summary.content
            return summary
        except Exception as e:
            logger.error(f"Error generating LLM summary for {ticker}: {e}")
            traceback.print_exc()
            return "Error: Could not generate AI-powered summary."

    def _build_markdown_report(self, ticker: str, all_data: Dict[str, Any]) -> Tuple[str, List[str]]:
        """
        Builds the full markdown string for the report, including tables, summaries, and charts.

        Returns:
            A tuple containing the markdown string and a list of temporary chart file paths.
        """
        report_parts = []
        chart_files = []

        today = datetime.now().strftime('%Y-%m-%d')
        report_parts.append(f"# CANSLIM Analysis for {ticker} ({today})\n")

        # --- Main Ticker Analysis ---
        main_ticker_data = all_data[ticker]
        main_metrics = main_ticker_data["metrics"]
        company_info = main_ticker_data["company_info"]
        metrics_md = self._create_metrics_table_md(main_metrics)

        # 1. Generate Industry Analysis first to feed into the main summary
        industry_analysis, competitor_metrics_md = self._generate_industry_analysis(ticker, company_info, metrics_md, all_data)

        # 2. LLM Summary
        llm_summary = self._generate_llm_summary(ticker, company_info, main_metrics, metrics_md, industry_analysis)
        report_parts.append("## AI-Powered Analysis\n")
        report_parts.append(llm_summary)

        # 3. Detailed Metrics Table
        report_parts.append("\n## Detailed CANSLIM Metrics\n")
        report_parts.append(metrics_md)

        # 4. Competitive Landscape Section
        report_parts.append("\n## Competitive Landscape\n")
        report_parts.append(industry_analysis)

        # --- Add the competitor tables below the analysis for reference ---
        if competitor_metrics_md:
            report_parts.append("\n### Competitor Metric Details\n")
            report_parts.append(f"\n{competitor_metrics_md}\n")

        # 5. Sentiment and Market Posture
        report_parts.append("\n## Market Context\n")
        if "Sentiment" in main_metrics and isinstance(main_metrics["Sentiment"], dict):
            sentiment = main_metrics['Sentiment']
            report_parts.append(f"**Sentiment:** {sentiment.get('sentiment', 'N/A')} (Polarity: {sentiment.get('polarity', 0):.2f}, Subjectivity: {sentiment.get('subjectivity', 0):.2f})\n")
        report_parts.append(f"**Overall Market Posture (SPY Trend):** {main_metrics.get('Market Posture (SPY Trend)', 'N/A')}\n")

        # 6. Technical Analysis Charts
        report_parts.append("\n## Technical Charts\n")
        try:
            # Use the unified chart generation method from TechnicalAnalysisTool
            price_indicators, chart_files_dict = self.tech_analyzer.get_price_indicators_and_charts_from_db(ticker)
            if isinstance(chart_files_dict, dict) and 'error' not in chart_files_dict:
                if chart_files_dict.get('short_term'):
                    report_parts.append(f"![Short Term Chart]({chart_files_dict['short_term']})\n")
                    chart_files.append(chart_files_dict['short_term'])
                if chart_files_dict.get('intermediate_term'):
                    report_parts.append(f"![Intermediate Term Chart]({chart_files_dict['intermediate_term']})\n")
                    chart_files.append(chart_files_dict['intermediate_term'])
            else:
                error_msg = chart_files_dict.get('error', 'Unknown error') if isinstance(chart_files_dict, dict) else "Invalid chart data"
                logger.error(f"Could not generate charts for {ticker}: {error_msg}")
                report_parts.append("_(Could not generate technical charts.)_\n")
        except Exception as e:
            logger.error(f"Error generating charts for {ticker}: {e}", exc_info=True)
            report_parts.append("_(Error generating technical charts.)_\n")

        return "\n".join(report_parts), chart_files
