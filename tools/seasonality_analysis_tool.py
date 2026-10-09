"""
A tool for conducting seasonality analysis on financial instruments.

This module provides the SeasonalityAnalysisTool class, which analyzes historical
price data to identify recurring patterns and tendencies tied to calendar events.
It calculates statistics like win rate and average returns for various seasonal
effects, such as monthly performance, day-of-the-week effects, and holiday-
related patterns.
"""
import logging
import pandas as pd
import numpy as np

from tools.data_manager import DataManager

logger = logging.getLogger(__name__)

class SeasonalityAnalysisTool:
    """
    Analyzes historical stock price data for seasonal patterns.
    """

    def __init__(self, ticker: str, start_date: str = "2000-01-01", end_date: str = None):
        """
        Initializes the tool and loads the necessary data.

        Args:
            ticker (str): The stock ticker symbol to analyze.
            start_date (str): The start date for the historical data (YYYY-MM-DD).
            end_date (str): The end date for the historical data (YYYY-MM-DD).
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = self._load_and_prepare_data()

    def _load_and_prepare_data(self) -> pd.DataFrame:
        """
        Loads price data and enriches it with calendar-based features.
        """
        logger.info(f"Loading and preparing data for {self.ticker}...")
        data_dict = DataManager().load_price_data(self.ticker, self.start_date, self.end_date)
        if not data_dict or 'shareprices' not in data_dict or data_dict['shareprices'].empty:
            raise ValueError(f"Could not load price data for {self.ticker}.")

        df = data_dict['shareprices'][['adjclose']].copy()
        df.rename(columns={'adjclose': 'price'}, inplace=True)

        # Ensure the index is a DatetimeIndex before accessing date properties
        df.index = pd.to_datetime(df.index)

        # --- Feature Engineering ---
        df['return'] = df['price'].pct_change()
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['day_of_week'] = df.index.dayofweek # Monday=0, Sunday=6
        df['day_of_month'] = df.index.day
        df['is_turn_of_month'] = (df['day_of_month'] >= 28) | (df['day_of_month'] <= 4)

        df.dropna(inplace=True)
        return df

    def _calculate_stats(self, grouped_data: pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
        """
        Calculates win rate, average return, and other stats for grouped data.
        """
        stats = grouped_data['return'].agg(
            avg_return=lambda x: x.mean() * 100,
            median_return=lambda x: x.median() * 100,
            std_dev_return=lambda x: x.std() * 100,
            win_rate=lambda x: (x > 0).sum() / x.count() if x.count() > 0 else 0,
            periods_analyzed='count'
        ).reset_index()

        stats['win_rate'] *= 100

        return stats.round(4)

    def analyze_monthly_performance(self) -> pd.DataFrame:
        """
        Analyzes performance for each calendar month.
        """
        logger.info("Analyzing monthly performance...")
        # Calculate total return for each month in each year
        monthly_returns = self.data.groupby(['year', 'month'])['return'].apply(lambda x: (1 + x).prod() - 1)
        monthly_returns_df = monthly_returns.reset_index()
        monthly_returns_df.rename(columns={'return': 'monthly_return'}, inplace=True)

        # Now, group by month to calculate stats over the years
        grouped_by_month = monthly_returns_df.groupby('month')

        # Calculate stats based on the monthly returns
        monthly_stats = grouped_by_month['monthly_return'].agg(
            avg_return=lambda x: x.mean() * 100,
            median_return=lambda x: x.median() * 100,
            std_dev_return=lambda x: x.std() * 100,
            win_rate=lambda x: (x > 0).sum() / x.count() * 100 if x.count() > 0 else 0,
            periods_analyzed='count'
        ).reset_index().round(4)

        # Sort by month number to ensure chronological order before converting to month name
        monthly_stats = monthly_stats.sort_values(by='month')

        monthly_stats['month'] = monthly_stats['month'].apply(lambda x: pd.to_datetime(str(x), format='%m').strftime('%B'))
        monthly_stats.set_index('month', inplace=True)

        # --- FIX: Ensure chronological month order for charting ---
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        monthly_stats.index = pd.CategoricalIndex(monthly_stats.index, categories=month_order, ordered=True)
        return monthly_stats.sort_index()

    def analyze_day_of_week_performance(self) -> pd.DataFrame:
        """
        Analyzes performance for each day of the week.
        """
        logger.info("Analyzing day-of-week performance...")
        # Ensure we only group by days that exist in the data
        dow_stats = self._calculate_stats(self.data.groupby('day_of_week'))
        day_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
        dow_stats['day_of_week'] = dow_stats['day_of_week'].map(day_map)
        dow_stats.set_index('day_of_week', inplace=True)

        # --- FIX: Ensure chronological day-of-week order for charting ---
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_stats.index = pd.CategoricalIndex(dow_stats.index, categories=day_order, ordered=True)
        return dow_stats.sort_index()

    def analyze_turn_of_month_effect(self) -> pd.DataFrame:
        """
        Analyzes performance for the turn-of-the-month period vs. other days.
        """
        logger.info("Analyzing turn-of-the-month effect...")
        grouped = self.data.groupby('is_turn_of_month')
        tom_stats = self._calculate_stats(grouped)
        tom_stats['is_turn_of_month'] = tom_stats['is_turn_of_month'].map({True: 'Turn of Month Days', False: 'Other Days'})
        return tom_stats.set_index('is_turn_of_month')

    def analyze_santa_claus_rally(self) -> pd.DataFrame:
        """
        Analyzes the "Santa Claus Rally" period (last 5 trading days of December
        and first 2 of January).
        """
        logger.info("Analyzing Santa Claus Rally...")
        rally_returns = []
        for year in range(self.data['year'].min(), self.data['year'].max() + 1):
            # Last 5 trading days of December
            dec_days = self.data[(self.data['year'] == year) & (self.data['month'] == 12)]
            if len(dec_days) < 5: continue
            
            # First 2 trading days of January
            jan_days = self.data[(self.data['year'] == year + 1) & (self.data['month'] == 1)]
            if len(jan_days) < 2: continue

            rally_period = pd.concat([dec_days.tail(5), jan_days.head(2)])
            
            # Calculate the total return over this 7-day period
            period_return = (1 + rally_period['return']).prod() - 1
            rally_returns.append({'year': year, 'return': period_return})

        if not rally_returns:
            return pd.DataFrame()

        rally_df = pd.DataFrame(rally_returns)
        win_rate = (rally_df['return'] > 0).sum() / len(rally_df) * 100
        avg_return = rally_df['return'].mean() * 100
        median_return = rally_df['return'].median() * 100

        stats_df = pd.DataFrame([{
            'Pattern': 'Santa Claus Rally',
            'avg_return': avg_return,
            'median_return': median_return,
            'win_rate': win_rate,
            'periods_analyzed': len(rally_df)
        }]).round(4)
        
        return stats_df.set_index('Pattern')

    def analyze_january_effect(self) -> pd.DataFrame:
        """
        Analyzes the "January Effect", particularly after a losing year.
        """
        logger.info("Analyzing January Effect...")
        yearly_returns = self.data.groupby('year')['return'].apply(lambda x: (1 + x).prod() - 1)
        
        losing_years = yearly_returns[yearly_returns < 0].index.tolist()
        
        jan_returns_after_loss = []
        for year in losing_years:
            # Get January returns for the *following* year
            next_year = year + 1
            jan_data = self.data[(self.data['year'] == next_year) & (self.data['month'] == 1)]
            if not jan_data.empty:
                month_return = (1 + jan_data['return']).prod() - 1
                jan_returns_after_loss.append(month_return)

        if not jan_returns_after_loss:
            return pd.DataFrame()

        jan_series = pd.Series(jan_returns_after_loss)
        win_rate = (jan_series > 0).sum() / len(jan_series) * 100
        avg_return = jan_series.mean() * 100
        median_return = jan_series.median() * 100

        stats_df = pd.DataFrame([{
            'Pattern': 'January (After Losing Year)',
            'avg_return': avg_return,
            'median_return': median_return,
            'win_rate': win_rate,
            'periods_analyzed': len(jan_series)
        }]).round(4)

        return stats_df.set_index('Pattern')

    def analyze_sell_in_may(self) -> pd.DataFrame:
        """
        Analyzes the "Sell in May and Go Away" effect by comparing returns
        from November-April to May-October.
        """
        logger.info("Analyzing 'Sell in May and Go Away'...")
        df = self.data.copy()
        
        # Define the two periods
        df['period'] = np.where(df['month'].isin([11, 12, 1, 2, 3, 4]), 'Nov-Apr', 'May-Oct')

        # Calculate total return for each period in each year
        period_returns = df.groupby(['year', 'period'])['return'].apply(lambda x: (1 + x).prod() - 1)
        period_returns_df = period_returns.reset_index()
        period_returns_df.rename(columns={'return': 'period_return'}, inplace=True)

        # Now, group by period to calculate stats over the years
        grouped_by_period = period_returns_df.groupby('period')

        # Calculate stats based on the period returns
        period_stats = self._calculate_period_stats(grouped_by_period, 'period_return')
        
        return period_stats.set_index('period')

    def analyze_sector_comparison(self) -> pd.DataFrame:
        """
        Compares the ticker's monthly performance against its sector benchmark.
        """
        logger.info(f"Analyzing sector comparison for {self.ticker}...")
        
        # 1. Get the sector benchmark ETF for the current ticker
        primary_benchmark, _ = DataManager().get_benchmark_ticker_for_asset(self.ticker)
        if primary_benchmark == '^SPX': # Default if no specific sector found
            logger.warning(f"No specific sector ETF found for {self.ticker}. Skipping sector comparison.")
            return pd.DataFrame()

        # 2. Get the monthly performance for the stock and the benchmark
        stock_monthly_perf = self.analyze_monthly_performance()
        
        try:
            benchmark_tool = SeasonalityAnalysisTool(ticker=primary_benchmark, start_date=self.start_date, end_date=self.end_date)
            benchmark_monthly_perf = benchmark_tool.analyze_monthly_performance()
        except ValueError as e:
            logger.error(f"Could not load data for benchmark {primary_benchmark}: {e}")
            return pd.DataFrame()

        # 3. Combine the results and calculate the spread
        comparison_df = stock_monthly_perf[['avg_return']].copy()
        comparison_df.rename(columns={'avg_return': f'{self.ticker}_avg_return'}, inplace=True)
        
        comparison_df = comparison_df.join(benchmark_monthly_perf[['avg_return']], how='inner')
        comparison_df.rename(columns={'avg_return': f'{primary_benchmark}_avg_return'}, inplace=True)

        comparison_df['performance_spread'] = comparison_df[f'{self.ticker}_avg_return'] - comparison_df[f'{primary_benchmark}_avg_return']
        comparison_df.index.name = "month"

        # --- FIX: Ensure chronological month order for charting ---
        # Convert month name index to a categorical type with the correct order.
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        comparison_df.index = pd.CategoricalIndex(comparison_df.index, categories=month_order, ordered=True)
        comparison_df = comparison_df.sort_index()

        return comparison_df.round(4)

    def run_full_analysis(self) -> dict:
        """
        Runs all seasonality analyses and returns a compiled dictionary of results.
        """
        logger.info(f"--- Starting Full Seasonality Analysis for {self.ticker} ---")
        results = {}
        try:
            results["monthly_performance"] = self.analyze_monthly_performance()
            results["day_of_week_performance"] = self.analyze_day_of_week_performance()
            results["turn_of_month_effect"] = self.analyze_turn_of_month_effect()
            results["sector_comparison"] = self.analyze_sector_comparison()
            
            # Specific calendar patterns
            santa_rally = self.analyze_santa_claus_rally()
            jan_effect = self.analyze_january_effect()
            sell_in_may = self.analyze_sell_in_may()
            
            # Combine specific patterns into a single DataFrame
            special_patterns = pd.concat([santa_rally, jan_effect, sell_in_may])
            results["special_patterns"] = special_patterns

            logger.info("--- Seasonality Analysis Complete ---")
            return results
        except Exception as e:
            logger.error(f"An error occurred during seasonality analysis: {e}", exc_info=True)
            return {"error": str(e)}

    def _calculate_period_stats(self, grouped_data: pd.core.groupby.DataFrameGroupBy, column_name: str) -> pd.DataFrame:
        """
        A more generic helper to calculate stats for pre-calculated period returns.
        """
        stats = grouped_data[column_name].agg(
            avg_return=lambda x: x.mean() * 100,
            median_return=lambda x: x.median() * 100,
            std_dev_return=lambda x: x.std() * 100,
            win_rate=lambda x: (x > 0).sum() / x.count() * 100 if x.count() > 0 else 0,
            periods_analyzed='count'
        ).reset_index().round(4)

        return stats