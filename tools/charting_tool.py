import logging
from typing import Union
import pandas as pd
import json
import traceback
import mplfinance as mpf
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class ChartingTool():
    def create_chart_for_indicator(self, data_source: Union[dict, str], chart_specifications: dict) -> dict:
        """
        Creates a chart from the given data and specifications.

        This function acts as a dispatcher, calling the appropriate charting method based on `chart_type`.

        Args:
            data_source (Union[dict, str]): A dictionary or a path to a JSON file containing the indicator data.
            chart_specifications (dict): A dictionary containing the chart specifications.
                Required keys: "output_file_path".
                Optional keys: "chart_type", "ticker", "title", "x_axis_label", "y_axis_label", "period",
                               "y_columns" (for line), "y_column" (for bar), "x_column" (for scatter).

        Returns:
            A dictionary containing the file path and a brief interpretation of the chart. Or an error message if any issues occur during chart creation.
        """
        try:
            if isinstance(data_source, dict):
                data = data_source
            else:
                with open(data_source, 'r') as f:
                    data = json.load(f)

            period = chart_specifications.get("period", "daily")
            df = pd.DataFrame(data[period])
            if not isinstance(df.index, pd.DatetimeIndex):
                df["Date"] = pd.to_datetime(df.index)
                df = df.set_index('Date')
                df = df.sort_index(ascending=True) 
            
            chart_type = chart_specifications.get("chart_type", "line")

            if chart_type == "candlestick":
                return self._create_candlestick_chart(df, chart_specifications)

            # --- Logic for other chart types (line, bar, scatter) ---
            ticker = chart_specifications.get("ticker", "")
            title = chart_specifications.get("title", "Chart")
            if ticker:
                title += f" for {ticker.upper()}"
            title = f"{title} ({period})"
            x_axis_label = chart_specifications.get("x_axis_label", "Date")
            y_axis_label = chart_specifications.get("y_axis_label", "Value")
            output_file_path = chart_specifications["output_file_path"]

            plt.figure(figsize=(12, 7))

            if chart_type == "line":
                # By default, plot all columns except the OHLCV data
                y_columns = chart_specifications.get("y_columns", [c for c in df.columns if c not in ['Open', 'High', 'Low', 'Close', 'Volume']])
                for column in y_columns:
                    if column in df.columns:
                        plt.plot(df.index, df[column], label=column)
            elif chart_type == "bar":
                y_column = chart_specifications.get("y_column")
                if not y_column or y_column not in df.columns:
                    raise ValueError("`y_column` must be specified in chart_specifications for bar charts.")
                plt.bar(df.index, df[y_column], label=y_column)
            elif chart_type == "scatter":
                x_column = chart_specifications.get("x_column")
                y_column = chart_specifications.get("y_column")
                if not x_column or not y_column or x_column not in df.columns or y_column not in df.columns:
                    raise ValueError("`x_column` and `y_column` must be specified for scatter charts.")
                plt.scatter(df[x_column], df[y_column])
            else:
                raise ValueError(f"Unsupported chart type: {chart_type}")

            plt.title(title)
            plt.xlabel(x_axis_label)
            plt.ylabel(y_axis_label)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_file_path)
            plt.close()

            return {"file_path": output_file_path, "interpretation": f"A {chart_type} chart of {title} has been created."}

        except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Error creating chart: {e}\n{traceback.format_exc()}")
            plt.close('all') # CRITICAL: Close all figures to prevent memory leaks on error
            return {"error": f"Error creating chart: {e}"}

    def _create_candlestick_chart(self, df: pd.DataFrame, chart_specifications: dict) -> dict:
        """Helper function to create a complex candlestick chart with mplfinance."""
        output_file_path = chart_specifications["output_file_path"]
        ticker = chart_specifications.get("ticker", "")
        title = chart_specifications.get("title", f"Candlestick Chart for {ticker.upper()}")
        period = chart_specifications.get("period", "daily")
        title = f"{title} ({period})"

        # Default configuration to replicate and improve upon the original hardcoded logic.
        # This makes adding/modifying indicators trivial without changing the plotting code.
        indicator_config = {
            # Panel 3 (MACD) - Most specific keys first to ensure correct matching
            'MACD_Hist':   {'panel': 3, 'color': 'gray',   'type': 'bar'},
            'MACD_Signal': {'panel': 3, 'color': 'red',    'type': 'line'},
            'MACD':        {'panel': 3, 'color': 'blue',   'type': 'line'},
            # Panel 2 (RSI)
            'RSI':         {'panel': 2, 'color': 'purple', 'type': 'line'},
            # Panel 0 (Price)
            'BB_Upper':    {'panel': 0, 'color': 'orange', 'type': 'line'},
            'BB_Lower':    {'panel': 0, 'color': 'orange', 'type': 'line'},
            'BB_Middle':   {'panel': 0, 'color': 'orange', 'linestyle': '--', 'type': 'line'},
            'SMA_200':     {'panel': 0, 'color': 'purple', 'type': 'line'},
            'SMA_50':      {'panel': 0, 'color': 'green',  'type': 'line'},
            'EMA':         {'panel': 0, 'color': 'red',    'type': 'line'}, # Generic EMA catch-all
        }

        max_points = chart_specifications.get("max_points", 1000)
        df_chart = df.iloc[-max_points:] 
        addplots = []
        ignore_cols = {'ATR', 'Pct_Change', 'CDL', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'}

        for col in df_chart.columns:
            # Check if the column is in our ignore list
            if col in ignore_cols:
                continue

            config = None
            # Find the most specific configuration that matches the column name
            for key, value in indicator_config.items():
                if col.startswith(key):
                    config = value
                    break
            
            if config:
                plot_kwargs = {'color': config.get('color'), 'panel': config.get('panel'), 'label': col}
                if config.get('linestyle'):
                    plot_kwargs['linestyle'] = config.get('linestyle')
                
                if config['type'] == 'bar':
                    addplots.append(mpf.make_addplot(df_chart[col], type='bar', width=0.7, **plot_kwargs))
                else: # Default to line
                    addplots.append(mpf.make_addplot(df_chart[col], **plot_kwargs))

        fig, _ = mpf.plot(df_chart, type='candle', volume=True, style='yahoo', figsize=(20, 16),
                          addplot=addplots, returnfig=True, tight_layout=True,
                          title=f'\n\n{title}') # Add newlines to make space for suptitle

        fig.savefig(output_file_path)
        plt.close(fig)
        return {"file_path": output_file_path, "interpretation": f"A candlestick chart of {title} with overlaid indicators has been created."}
