"""
A suite of tools for performing technical analysis on stock price data.

This module provides:
- A standalone function `get_stock_trend` to determine the overall market trend.
- A `TechnicalAnalysisTool` class that can:
  - Calculate a wide range of technical indicators (SMA, EMA, RSI, MACD, etc.) for multiple timeframes.
  - Generate textual summaries of the technical picture.
  - Use an LLM to create a narrative technical analysis report.
"""
from typing import Union
import pandas as pd
import talib as ta  
from langchain_core.prompts import PromptTemplate
import numpy as np
import logging

from tools.charting_tool import ChartingTool
from tools.file_wrapper import generate_filename
from tools.data_manager import DataManager

logger = logging.getLogger(__name__)


def get_stock_trend(stock_symbol: str = "SPY") -> Union[str, dict]:
    """
    Determines the trend (Bullish, Bearish, Neutral) for a given stock symbol
    based on 50-day and 200-day SMAs and the slope of the 200-day SMA.

    Args:
        stock_symbol: The stock ticker symbol.

    Returns:
        A string indicating the trend ("Bullish", "Bearish", "Neutral")
        or a dictionary with an "error" key if an issue occurs.
    """
    try:
        price_history = DataManager().load_price_data(stock_symbol)
        if "error" in price_history or price_history["shareprices"].empty:
            return None
        
        stock_df = price_history["shareprices"]
        if len(stock_df) < 200:
            return {"error": f"Not enough data for {stock_symbol} to calculate 200-day SMA."}

        stock_df['SMA50'] = stock_df['adjclose'].rolling(window=50).mean()
        stock_df['SMA200'] = stock_df['adjclose'].rolling(window=200).mean()

        # Ensure enough data for slope calculation after SMA calculation
        sma200_series = stock_df['SMA200'].dropna()
        if len(sma200_series) < 20: # Need at least a few points for a meaningful slope
            return {"error": f"Not enough data for {stock_symbol} to calculate SMA200 slope."}

        sma_values = sma200_series.iloc[-min(200, len(sma200_series)):].values # Use available or last 200 points
        x = np.arange(len(sma_values))
        slope = np.polyfit(x, sma_values, 1)[0]  # Linear regression

        latest_indicators = stock_df.iloc[-1]
        if pd.isna(latest_indicators['SMA50']) or pd.isna(latest_indicators['SMA200']):
            return {"error": f"SMA values are NaN for {stock_symbol} on the latest date."}

        if latest_indicators['SMA50'] > latest_indicators['SMA200'] and slope > 0.01:
            stock_trend = "Bullish"
        elif latest_indicators['SMA50'] < latest_indicators['SMA200'] and slope < -0.01:
            stock_trend = "Bearish"
        else:
            stock_trend = "Neutral"

        return stock_trend
    except Exception as e:
        logger.error(f"Error in Stock Trend ({stock_symbol}) analysis: {e}", exc_info=True)
        return {"error": f"Error in Stock Trend ({stock_symbol}) analysis: {e}"}
 

class TechnicalAnalysisTool:
    """
    A tool for performing technical analysis on stock data.

    This class encapsulates methods for calculating technical indicators,
    summarizing technical data, and generating analysis reports using an LLM.
    It is designed to work with price data fetched from the application's database.
    """
    def calculate_technical_indicators(self, price_history_data: pd.DataFrame) -> Union[pd.DataFrame, dict]:  
        """
        Calculates daily, weekly, and monthly technical indicators from a daily price history DataFrame.

        This method orchestrates the calculation by resampling the daily data into
        weekly and monthly timeframes and then applying a standard set of technical
        indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR) to each timeframe.

        The parameters for the indicators (e.g., period lengths) are adjusted
        automatically for each timeframe to be more conventional for that scale
        (e.g., shorter periods for weekly/monthly analysis).

        Args:
            price_history_data: A DataFrame containing the daily price history data.

        Returns:
            A DataFrame containing the technical indicators, or a dictionary containing an error message if a problem occurs.
        """
        try:
            timeframes = {
                "daily": price_history_data,
                # Use 'closed=right' and 'label=right' to ensure the latest 
                # timestamp (e.g. 2026-09-18) falls into the correct current bucket.
                "weekly": price_history_data.resample('W', closed='right', label='right').agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Adj Close': 'last', 'Volume': 'sum'
                }),
                "monthly": price_history_data.resample('ME', closed='right', label='right').agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Adj Close': 'last', 'Volume': 'sum'
                }),
            }

            results = {}
            for timeframe, df in timeframes.items():  # Iterate through timeframes
                if df.empty:
                    return {"error": f"Error resampling data for {timeframe} timeframe."}
                
                df_with_indicators = self._calculate_indicators_for_timeframe(df.copy(), timeframe)
                if isinstance(df_with_indicators, dict) and 'error' in df_with_indicators:
                    return df_with_indicators # Propagate error

                df_with_indicators.dropna(inplace=True)
                results[timeframe] = df_with_indicators.iloc[-600:].reset_index().to_dict('records') # to json

            return results
        except Exception as e:
            logger.error(f"An unexpected error occurred in calculate_technical_indicators: {e}", exc_info=True)
            return {"error": f"An unexpected error occurred: {str(e)}"}

    def calculate_technical_indicators_from_db(self, ticker: str) -> Union[pd.DataFrame, dict]:
        """
        Fetches price history from the database for a ticker and calculates technical indicators.

        Args:
            ticker: The stock ticker symbol.

        Returns:
            A dictionary containing the calculated technical indicators for daily, weekly,
            and monthly timeframes, or a dictionary with an "error" key.
        """
        try:
            price_history = DataManager().load_price_data(ticker)
            if "error" in price_history or price_history["shareprices"].empty:
                return None
            
            df = price_history["shareprices"].rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "adjclose": "Adj Close", "volume": "Volume"})
            results = self.calculate_technical_indicators(df)
            return results
        except Exception as e:
            logger.error(f"An unexpected error occurred in calculate_technical_indicators_from_db: {e}", exc_info=True)
            return {"error": f"An unexpected error occurred: {str(e)}"}

    def summarize_technical_data(self, df: pd.DataFrame, timeframe: str = "daily"):
        """
        Calculates a dictionary of key summary statistics from a DataFrame of technical indicators.

        This function distills a time-series DataFrame of indicators into a single
        dictionary of the latest values and trends, which can be easily consumed by an LLM.
        It includes logic for trend detection on SMAs and ATR.

        Args:
            df: A DataFrame containing price data and pre-calculated technical indicators.
                The column names are expected to follow the convention set by
                `calculate_technical_indicators`.
            timeframe: The timeframe of the data ("daily", "weekly", "monthly"). This
                       determines which indicator parameters to look for.

        Returns:
            A dictionary summarizing the latest technical state (e.g., last price,
            SMA trend, RSI level, MACD signal).
        """
        if df.empty:
            return "No data available"
        #logger.debug(f"{df.columns=}")
        summary = {}
        if timeframe == "daily":
            timeframe_str = "day"
            period_sma_ema = 20
            period_rsi = 14
        elif timeframe == "weekly":
            timeframe_str = "week"
            period_sma_ema = 10
            period_rsi = 10
        elif timeframe == "monthly":
            timeframe_str = "month"
            period_sma_ema = 9
            period_rsi = 9
            
        summary["Last price"] = df["Adj Close"].iloc[-1] if "Adj Close" in df.columns else "N/A"
        summary["Last price change"] = (df["Adj Close"].iloc[-1] - df["Adj Close"].iloc[-2]) if "Adj Close" in df.columns and df.shape[0]>1 else "N/A"
        
        # Improved Trend Detection for SMA
        if f"SMA ({period_sma_ema}-{timeframe_str})" in df.columns:
            summary[f"Last SMA ({period_sma_ema}-{timeframe_str})"] = df[f"SMA ({period_sma_ema}-{timeframe_str})"].iloc[-1]
            summary[f"SMA ({period_sma_ema}-{timeframe_str}) Trend"] = self._get_series_trend(df[f"SMA ({period_sma_ema}-{timeframe_str})"])
            
            summary[f"SMA ({period_sma_ema}-{timeframe_str}) vs Price"] = "Above" if summary["Last price"] > summary[f"Last SMA ({period_sma_ema}-{timeframe_str})"] else "Below" if summary["Last price"] < summary[f"Last SMA ({period_sma_ema}-{timeframe_str})"] else "Neutral"
            if timeframe == "daily":
                if "SMA (200 day)" in df.columns:
                    summary[f"SMA ({period_sma_ema}-{timeframe_str}) crossed SMA200"] = "Yes" if df[f"SMA ({period_sma_ema}-{timeframe_str})"].iloc[-2] <= df["SMA (200 day)"].iloc[-2] and summary[f"Last SMA ({period_sma_ema}-{timeframe_str})"] > df["SMA (200 day)"].iloc[-1] else "No" if df[f"SMA ({period_sma_ema}-{timeframe_str})"].iloc[-2] >= df["SMA (200 day)"].iloc[-2] and summary[f"Last SMA ({period_sma_ema}-{timeframe_str})"] < df["SMA (200 day)"].iloc[-1] else "No"
                    summary[f"SMA ({period_sma_ema}-{timeframe_str}) vs SMA200"] = "Above" if summary[f"Last SMA ({period_sma_ema}-{timeframe_str})"] > df["SMA (200 day)"].iloc[-1] else "Below" if summary[f"Last SMA ({period_sma_ema}-{timeframe_str})"] < df["SMA (200 day)"].iloc[-1] else "Equal"

        # Improved Trend Detection for SMA 200
        if "SMA (200 day)" in df.columns and timeframe == "daily":
            summary[f"Last SMA (200 day)"] = df["SMA (200 day)"].iloc[-1]
            summary[f"SMA (200 day) Trend"] = self._get_series_trend(df["SMA (200 day)"])

            summary[f"SMA (200 day) vs Price"] = "Above" if summary["Last price"] > summary[f"Last SMA (200 day)"] else "Below" if summary["Last price"] < summary[f"Last SMA (200 day)"] else "Neutral"

        summary[f"Price vs SMA ({period_sma_ema}-{timeframe_str})"] = "Above" if summary["Last price"] > summary.get(f"Last SMA ({period_sma_ema}-{timeframe_str})",float('-inf')) else "Below" if summary["Last price"] < summary.get(f"Last SMA ({period_sma_ema}-{timeframe_str})",float('inf')) else "Equal"
        if "SMA (200 day)" in df.columns and timeframe == "daily":
            summary[f"Price vs SMA (200 day)"] = "Above" if summary["Last price"] > summary.get(f"Last SMA (200 day)",float('-inf')) else "Below" if summary["Last price"] < summary.get(f"Last SMA (200 day)",float('inf')) else "Equal"
        if f"RSI ({period_rsi}-{timeframe_str})" in df.columns:
            summary[f"Last RSI ({period_rsi}-{timeframe_str})"] = df[f"RSI ({period_rsi}-{timeframe_str})"].iloc[-1]
            summary[f"RSI ({period_rsi}-{timeframe_str}) Overbought/Oversold"] = "Overbought" if summary[f"Last RSI ({period_rsi}-{timeframe_str})"] > 70 else "Oversold" if summary[f"Last RSI ({period_rsi}-{timeframe_str})"] < 30 else "Neither"
        if f"MACD(12, 26, 9)" in df.columns or f"MACD(5, 10, 4)" in df.columns or f"MACD(4, 9, 3)" in df.columns:
            macd_column = [col for col in df.columns if "MACD(" in col][0]
            summary[f"Last {macd_column}"] = df[macd_column].iloc[-1]
            summary["Last MACD_Signal"] = df[f"MACD_Signal"].iloc[-1]
            summary[f"MACD Signal"] = "Bullish" if df[macd_column].iloc[-1] > df["MACD_Signal"].iloc[-1] else "Bearish" if df[macd_column].iloc[-1] < df["MACD_Signal"].iloc[-1] else "Neutral"
            summary[f"MACD Histogram Positive/Negative"] = "Positive" if df["MACD_Hist"].iloc[-1] > 0 else "Negative" if df["MACD_Hist"].iloc[-1] < 0 else "Zero"

        distance_to_BB_Upper = None
        distance_to_BB_Lower = None
        if f"BB_Upper ({period_sma_ema}-{timeframe_str})" in df.columns:
            summary[f"Last Bollinger Upper ({period_sma_ema}-{timeframe_str})"] = df[f"BB_Upper ({period_sma_ema}-{timeframe_str})"].iloc[-1]
            distance_to_BB_Upper = summary["Last price"] - summary[f"Last Bollinger Upper ({period_sma_ema}-{timeframe_str})"]
        else:
            summary[f"Bollinger Upper ({period_sma_ema}-{timeframe_str}) vs Price"] = "N/A"

        if f"BB_Lower ({period_sma_ema}-{timeframe_str})" in df.columns:
            summary[f"Last Bollinger Lower ({period_sma_ema}-{timeframe_str})"] = df[f"BB_Lower ({period_sma_ema}-{timeframe_str})"].iloc[-1]
            distance_to_BB_Lower = summary["Last price"] - summary[f"Last Bollinger Lower ({period_sma_ema}-{timeframe_str})"]
        else:
            summary[f"Bollinger Lower ({period_sma_ema}-{timeframe_str}) vs Price"] = "N/A"

        if distance_to_BB_Upper is not None and distance_to_BB_Lower is not None:
            if abs(distance_to_BB_Upper) > abs(distance_to_BB_Lower) * 2:
                summary[f"Bollinger Upper ({period_sma_ema}-{timeframe_str}) vs Price"] = "Far"
                summary[f"Bollinger Lower ({period_sma_ema}-{timeframe_str}) vs Price"] = "Near"
            if abs(distance_to_BB_Upper) < abs(distance_to_BB_Lower) / 2:
                summary[f"Bollinger Upper ({period_sma_ema}-{timeframe_str}) vs Price"] = "Near"
                summary[f"Bollinger Lower ({period_sma_ema}-{timeframe_str}) vs Price"] = "Far"

        # Bollinger Band Squeeze Detection
        if f"BB_Upper ({period_sma_ema}-{timeframe_str})" in df.columns and f"BB_Lower ({period_sma_ema}-{timeframe_str})" in df.columns:
            # Calculate Bollinger Band width
            bb_width = df[f"BB_Upper ({period_sma_ema}-{timeframe_str})"] - df[f"BB_Lower ({period_sma_ema}-{timeframe_str})"]

            # Calculate the percentage change in BB width over the last 'n' periods (e.g., 20)
            lookback_periods_bb = min(20, len(bb_width))
            if lookback_periods_bb > 1:
                bb_width_change_pct = bb_width.pct_change(periods=lookback_periods_bb).iloc[-1]
                # Determine squeeze condition (e.g., significant contraction)
                squeeze_threshold = -0.2 # Adjust the threshold if necessary
                summary[f"Bollinger Band Squeeze ({period_sma_ema}-{timeframe_str})"] = "Yes" if bb_width_change_pct < squeeze_threshold else "No"
            else:
                summary[f"Bollinger Band Squeeze ({period_sma_ema}-{timeframe_str})"] = "Not enough Data"

            # Bollinger Band Breakout Detection
            last_price = df["Adj Close"].iloc[-1]
            last_bb_upper = df[f"BB_Upper ({period_sma_ema}-{timeframe_str})"].iloc[-1]
            last_bb_lower = df[f"BB_Lower ({period_sma_ema}-{timeframe_str})"].iloc[-1]

            if last_price > last_bb_upper:
                summary[f"Bollinger Band Breakout ({period_sma_ema}-{timeframe_str})"] = "Above Upper Band"
            elif last_price < last_bb_lower:
                summary[f"Bollinger Band Breakout ({period_sma_ema}-{timeframe_str})"] = "Below Lower Band"
            else:
                summary[f"Bollinger Band Breakout ({period_sma_ema}-{timeframe_str})"] = "No"

        # Improved Trend Detection for ATR
        if f"ATR ({period_sma_ema}-{timeframe_str})" in df.columns:
            summary[f"Last ATR ({period_sma_ema}-{timeframe_str})"] = df[f"ATR ({period_sma_ema}-{timeframe_str})"].iloc[-1]
            summary[f"ATR ({period_sma_ema}-{timeframe_str}) Trend"] = self._get_series_trend(df[f"ATR ({period_sma_ema}-{timeframe_str})"])
        
        return summary

    def perform_technical_analysis(self, ticker: str, llm):
        """
        Performs a multi-timeframe technical analysis for a given stock ticker.

        Args:
            ticker: The stock ticker symbol.
            llm: The language model to use for analysis.

        Returns:
            A string containing the technical analysis results, or a dictionary with an error message.
        """

        try:
            price_indicators = self.calculate_technical_indicators_from_db(ticker)
            if isinstance(price_indicators, dict) and 'error' in price_indicators:
                return price_indicators
            
            daily_df = pd.DataFrame(price_indicators.get("daily", []))
            weekly_df = pd.DataFrame(price_indicators.get("weekly", []))
            monthly_df = pd.DataFrame(price_indicators.get("monthly", []))

            if daily_df.empty: # Handle the empty dataframe case
                raise ValueError("No daily data available for technical analysis.") 
            else:
                if pd.api.types.is_datetime64_any_dtype(daily_df['Date']): # type check
                    analysis_date = daily_df['Date'].max().strftime('%Y-%m-%d')
                else:
                    daily_df['Date'] = pd.to_datetime(daily_df['Date'], errors='coerce')
                    analysis_date = daily_df['Date'].max().strftime('%Y-%m-%d')
                    if analysis_date is pd.NaT:
                        raise ValueError("Invalid date format in daily data.") 

            daily_summary = self.summarize_technical_data(daily_df, "daily")
            weekly_summary = self.summarize_technical_data(weekly_df, "weekly")
            monthly_summary = self.summarize_technical_data(monthly_df, "monthly")

            technical_analysis_prompt = """
Perform a multi-timeframe technical analysis for {ticker} as of {analysis_date} using the provided indicator and price data.

Daily Data (Most Recent 600 Data Points):
{daily_data}

Weekly Data (Most Recent 600 Data Points):
{weekly_data}

Monthly Data (Most Recent 600 Data Points):
{monthly_data}

Analyze trends and signals across daily, weekly, and monthly timeframes. Identify confirmations and divergences between timeframes.  Note instances where timeframes support or contradict each other.

Indicators and Price Action Analysis:

* Price trends, recent swing highs/lows, support/resistance levels, breakouts/breakdowns.
* SMA/EMA trends, moving average crossovers (e.g., 50-day crossing above 200-day), price relative to SMA/EMA.
* RSI overbought/oversold conditions, divergences with price, RSI trend.
* MACD crossovers, divergences, MACD histogram slope, MACD sign.
* Bollinger Bands volatility, band width, squeezes, breakouts, price relative to bands.
* ATR volatility changes, confirmation/contradiction with other signals, ATR trend.
* Percentage price changes, acceleration/deceleration in price changes.


Technical Analysis Summary (as of {analysis_date}):

* Overall trend assessment (bullish, bearish, neutral): Justify with indicator and price action evidence.
* Key support and resistance levels: Identify significant levels based on price action and indicator confluence.
* Divergences or confirmations between timeframes: Highlight divergences or confirmations across timeframes.
* Volatility assessment: Assess volatility based on ATR and Bollinger Bands.  Specify if increasing, decreasing, or stable.
* Momentum assessment: Assess momentum based on RSI and MACD. Specify if increasing, decreasing, or stable.


Analysis should be data-driven and avoid predictions of future price movements. Focus on the current technical picture.  If data is missing or insufficient, state this explicitly.
            """

            prompt_template = PromptTemplate(
                input_variables=["ticker", "analysis_date", "daily_data", "weekly_data", "monthly_data"], 
                template=technical_analysis_prompt,
            )

            chain = prompt_template | self.llm
            technical_analysis_results = chain.invoke(
                ticker=ticker,
                analysis_date=analysis_date,  
                daily_data=daily_summary,
                weekly_data=weekly_summary,
                monthly_data=monthly_summary,
            )

            if hasattr(technical_analysis_results, 'content'):
                return technical_analysis_results.content
            return technical_analysis_results

        except Exception as e:
            return {"error": f"Error in technical analysis: {e}"}

    def get_price_indicators_and_charts_from_db(self, ticker: str):
        """
        Fetches indicators from the database and generates short and intermediate term charts.

        This is a convenience method that combines fetching indicator data and calling
        the ChartingTool to produce standard analysis charts.

        Args:
            ticker: The stock ticker symbol.

        Returns:
            A tuple containing:
            - A dictionary of the price indicator data.
            - A dictionary of the generated chart file paths ('short_term', 'intermediate_term').
        """
        chart_files = {}
        price_indicators = TechnicalAnalysisTool().calculate_technical_indicators_from_db(ticker)
        if isinstance(price_indicators, dict) and 'error' in price_indicators:
            return price_indicators, chart_files
        
        chart_type = "candlestick"

        title = "Short Term Price Chart with Indicators"
        period = "daily"
        chart_file1 = generate_filename(ticker, f"{period}_chart", "jpg")
        chart_specifications = {
            "ticker": ticker,
            "title": title,
            "x_axis_label": "Date",
            "y_axis_label": "Price",
            "output_file_path": chart_file1,
            "chart_type": chart_type,
            "period": period,
        }
        result = ChartingTool().create_chart_for_indicator(price_indicators, chart_specifications)
        if isinstance(result, dict) and 'error' not in result:
            chart_files['short_term'] = chart_file1

        title = "Intermediate Term Price Chart with Indicators"
        period = "weekly"
        chart_file2 = generate_filename(ticker, f"{period}_chart", "jpg")
        chart_specifications["title"] = title
        chart_specifications["period"] = period
        chart_specifications["output_file_path"] = chart_file2
        result = ChartingTool().create_chart_for_indicator(price_indicators, chart_specifications)
        if isinstance(result, dict) and 'error' not in result:
            chart_files['intermediate_term'] = chart_file2

        return price_indicators, chart_files
    
    def _calculate_indicators_for_timeframe(self, df: pd.DataFrame, timeframe: str) -> Union[pd.DataFrame, dict]:
        """Helper to calculate a standard set of indicators for a given timeframe DataFrame."""
        if timeframe == "daily":
            timeframe_str, period_sma_ema, period_rsi, macd_fast, macd_slow, macd_signal = "day", 20, 14, 12, 26, 9
        elif timeframe == "weekly":
            timeframe_str, period_sma_ema, period_rsi, macd_fast, macd_slow, macd_signal = "week", 10, 10, 5, 10, 4
        elif timeframe == "monthly":
            timeframe_str, period_sma_ema, period_rsi, macd_fast, macd_slow, macd_signal = "month", 9, 9, 4, 9, 3
        else:
            return {"error": f"Unknown timeframe: {timeframe}"}

        min_data_points = max(period_sma_ema, period_rsi, macd_slow)
        if len(df) < min_data_points:
            return {"error": f"Not enough data for {timeframe} calculations. Need {min_data_points}, have {len(df)}."}

        df.loc[:, f'SMA ({period_sma_ema}-{timeframe_str})'] = ta.SMA(df['Adj Close'], timeperiod=period_sma_ema)
        df.loc[:, f'EMA ({period_sma_ema}-{timeframe_str})'] = ta.EMA(df['Adj Close'], timeperiod=period_sma_ema)
        df.loc[:, f'RSI ({period_rsi}-{timeframe_str})'] = ta.RSI(df['Adj Close'], timeperiod=period_rsi)
        if timeframe == "daily":
            df.loc[:, 'SMA (50 day)'] = ta.SMA(df['Adj Close'], timeperiod=50)
            df.loc[:, 'SMA (200 day)'] = ta.SMA(df['Adj Close'], timeperiod=200)

        macd, macdsignal, macdhist = ta.MACD(df['Adj Close'], fastperiod=macd_fast, slowperiod=macd_slow, signalperiod=macd_signal)
        df.loc[:, f'MACD({macd_fast}, {macd_slow}, {macd_signal})'] = macd
        df.loc[:, f'MACD_Signal'] = macdsignal
        df.loc[:, f'MACD_Hist'] = macdhist

        upper, middle, lower = ta.BBANDS(df['Adj Close'], timeperiod=period_sma_ema, nbdevup=2, nbdevdn=2, matype=0)
        df.loc[:, f'BB_Upper ({period_sma_ema}-{timeframe_str})'] = upper
        df.loc[:, f'BB_Lower ({period_sma_ema}-{timeframe_str})'] = lower

        df.loc[:, f'ATR ({period_sma_ema}-{timeframe_str})'] = ta.ATR(df['High'], df['Low'], df['Close'], timeperiod=period_sma_ema)
        df.loc[:, f'Close_Pct_Change'] = df['Close'].pct_change() * 100
        df.loc[:, f'High_Pct_Change'] = df['High'].pct_change() * 100
        df.loc[:, f'Low_Pct_Change'] = df['Low'].pct_change() * 100
        
        return df

    def _get_series_trend(self, series: pd.Series, lookback_period: int = 10) -> str:
        """Calculates the trend of a pandas Series using linear regression."""
        lookback_period = min(lookback_period, len(series))
        if lookback_period < 3:
            return "Not enough Data"

        values = series.iloc[-lookback_period:].values
        if np.all(values == values[0]):
            return "Sideways"
        if np.isnan(values).any() or np.isinf(values).any() or not np.issubdtype(values.dtype, np.number):
            return "Not enough Data"

        x = np.arange(len(values))
        try:
            slope = np.polyfit(x, values, 1)[0]
            if slope > 0.01 * np.mean(values): # Slope relative to mean value
                return "Up"
            elif slope < -0.01 * np.mean(values):
                return "Down"
            else:
                return "Sideways"
        except (np.linalg.LinAlgError, ValueError):
            return "Not enough Data"


pattern_functions = {   # Dictionary to hold pattern recognition functions
    "CDL2CROWS": ta.CDL2CROWS,
    "CDL3BLACKCROWS": ta.CDL3BLACKCROWS,
    "CDL3INSIDE": ta.CDL3INSIDE,
    "CDL3LINESTRIKE": ta.CDL3LINESTRIKE,
    "CDL3OUTSIDE": ta.CDL3OUTSIDE,
    "CDL3STARSINSOUTH": ta.CDL3STARSINSOUTH,
    "CDL3WHITESOLDIERS": ta.CDL3WHITESOLDIERS,
    "CDLABANDONEDBABY": ta.CDLABANDONEDBABY,
    "CDLADVANCEBLOCK": ta.CDLADVANCEBLOCK,
    "CDLBELTHOLD": ta.CDLBELTHOLD,
    "CDLBREAKAWAY": ta.CDLBREAKAWAY,
    "CDLCLOSINGMARUBOZU": ta.CDLCLOSINGMARUBOZU,
    "CDLCONCEALBABYSWALL": ta.CDLCONCEALBABYSWALL,
    "CDLCOUNTERATTACK": ta.CDLCOUNTERATTACK,
    "CDLDARKCLOUDCOVER": ta.CDLDARKCLOUDCOVER,
    "CDLDOJI": ta.CDLDOJI,
    "CDLDOJISTAR": ta.CDLDOJISTAR,
    "CDLDRAGONFLYDOJI": ta.CDLDRAGONFLYDOJI,
    "CDLENGULFING": ta.CDLENGULFING,
    "CDLEVENINGDOJISTAR": ta.CDLEVENINGDOJISTAR,
    "CDLEVENINGSTAR": ta.CDLEVENINGSTAR,
    "CDLGAPSIDESIDEWHITE": ta.CDLGAPSIDESIDEWHITE,
    "CDLGRAVESTONEDOJI": ta.CDLGRAVESTONEDOJI,
    "CDLHAMMER": ta.CDLHAMMER,
    "CDLHANGINGMAN": ta.CDLHANGINGMAN,
    "CDLHARAMI": ta.CDLHARAMI,
    "CDLHARAMICROSS": ta.CDLHARAMICROSS,
    "CDLHIGHWAVE": ta.CDLHIGHWAVE,
    "CDLHIKKAKE": ta.CDLHIKKAKE,
    "CDLHIKKAKEMOD": ta.CDLHIKKAKEMOD,
    "CDLHOMINGPIGEON": ta.CDLHOMINGPIGEON,
    "CDLIDENTICAL3CROWS": ta.CDLIDENTICAL3CROWS,
    "CDLINNECK": ta.CDLINNECK,
    "CDLINVERTEDHAMMER": ta.CDLINVERTEDHAMMER,
    "CDLKICKING": ta.CDLKICKING,
    "CDLKICKINGBYLENGTH": ta.CDLKICKINGBYLENGTH,
    "CDLLADDERBOTTOM": ta.CDLLADDERBOTTOM,
    "CDLLONGLEGGEDDOJI": ta.CDLLONGLEGGEDDOJI,
    "CDLLONGLINE": ta.CDLLONGLINE,
    "CDLMARUBOZU": ta.CDLMARUBOZU,
    "CDLMATCHINGLOW": ta.CDLMATCHINGLOW,
    "CDLMATHOLD": ta.CDLMATHOLD,
    "CDLMORNINGDOJISTAR": ta.CDLMORNINGDOJISTAR,
    "CDLMORNINGSTAR": ta.CDLMORNINGSTAR,
    "CDLONNECK": ta.CDLONNECK,
    "CDLPIERCING": ta.CDLPIERCING,
    "CDLRICKSHAWMAN": ta.CDLRICKSHAWMAN,
    "CDLRISEFALL3METHODS": ta.CDLRISEFALL3METHODS,
    "CDLSEPARATINGLINES": ta.CDLSEPARATINGLINES,
    "CDLSHOOTINGSTAR": ta.CDLSHOOTINGSTAR,
    "CDLSHORTLINE": ta.CDLSHORTLINE,
    "CDLSPINNINGTOP": ta.CDLSPINNINGTOP,
    "CDLSTALLEDPATTERN": ta.CDLSTALLEDPATTERN,
    "CDLSTICKSANDWICH": ta.CDLSTICKSANDWICH,
    "CDLTAKURI": ta.CDLTAKURI,
    "CDLTASUKIGAP": ta.CDLTASUKIGAP,
    "CDLTHRUSTING": ta.CDLTHRUSTING,
    "CDLTRISTAR": ta.CDLTRISTAR,
    "CDLUNIQUE3RIVER": ta.CDLUNIQUE3RIVER,
    "CDLUPSIDEGAP2CROWS": ta.CDLUPSIDEGAP2CROWS,
    "CDLXSIDEGAP3METHODS": ta.CDLXSIDEGAP3METHODS,
}
