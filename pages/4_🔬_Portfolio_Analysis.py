import streamlit as st
import pandas as pd
from datetime import datetime

from load_cfg import DEMO_MODE
from pybroker_trainer.strategy_loader import get_strategy_class_map
from quant_engine import (
    run_pre_scan_universe,
    run_pybroker_portfolio_backtest,
    get_default_tickers,
    plot_performance_vs_benchmark
)
from tools.yfinance_tool import TIMEFRAME_OPTIONS

st.set_page_config(page_title="Portfolio Analysis", layout="wide")
st.title("🔬 Portfolio Analysis")

if DEMO_MODE:
    st.warning(
        "**Demo Mode Active:** All portfolio analysis and backtesting operations are disabled. "
        "To enable these features, set `DEMO_MODE = False` in `load_cfg.py`.",
        icon="🔒"
    )

st.markdown("""
This page provides tools to analyze strategies across a universe of stocks.
- **Pre-Scan Universe:** Find which stocks from a list have a minimum number of historical trade setups for a given strategy. This helps build a viable candidate list for a portfolio backtest.
- **Portfolio Backtest:** Run a single walk-forward backtest on a portfolio of tickers to see how the strategy performs with a fixed number of open positions.
""")

strategy_options = list(get_strategy_class_map().keys())

if 'prescan_results' not in st.session_state:
    st.session_state.prescan_results = ""

scan_tab, backtest_tab = st.tabs(["Pre-Scan Universe", "Portfolio Backtest"])

with scan_tab:
    st.header("Find Tradable Tickers")
    with st.form("prescan_form"):
        default_tickers_str = ",".join(get_default_tickers(source=2, limit=100))
        tickers_prescan = st.text_area("Tickers to Scan (comma-separated)", value=default_tickers_str, height=150)
        
        c1, c2, c3 = st.columns(3)
        strategy_type_prescan = c1.selectbox("Strategy Type", strategy_options, key="prescan_strat")
        timeframe_prescan = c2.selectbox("Timeframe", TIMEFRAME_OPTIONS, key="prescan_tf")
        min_setups_prescan = c3.number_input("Minimum Historical Setups", min_value=5, value=60)

        c1, c2 = st.columns(2)
        start_date_prescan = c1.date_input("Start Date", datetime(2000, 1, 1), key="prescan_start")
        end_date_prescan = c2.date_input("End Date", datetime.now(), key="prescan_end")

        run_prescan = st.form_submit_button("Run Pre-Scan", width='stretch', disabled=DEMO_MODE)

    if run_prescan:
        progress_bar = st.progress(0)
        progress_text = st.empty()

        def update_progress(progress, text):
            progress_bar.progress(progress)
            progress_text.text(text)

        ticker_list = [t.strip().upper() for t in tickers_prescan.split(',') if t.strip()]
        
        with st.spinner("Scanning universe for setups..."):
            valid_tickers, counts = run_pre_scan_universe(
                tickers=",".join(ticker_list),
                strategy_type=strategy_type_prescan,
                timeframe_override=timeframe_prescan,
                min_setups=min_setups_prescan,
                start_date=start_date_prescan.strftime('%Y-%m-%d'),
                end_date=end_date_prescan.strftime('%Y-%m-%d'),
                progress_callback=update_progress
            )
        st.success("Pre-scan complete!")
        
        st.session_state.prescan_results = ",".join(valid_tickers)

        if counts:
            st.subheader("All Ticker Counts")
            # Sort by count descending for better readability
            all_counts_str = "\n".join([f"{ticker}: {count}" for ticker, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)])
            st.text_area("Full Scan Counts (Ticker: Setup Count)", value=all_counts_str, height=250)

        if valid_tickers:
            st.subheader("Tickers with Sufficient Setups")
            st.write(f"Found {len(valid_tickers)} tickers with at least {min_setups_prescan} setups:")
            
            # Create a DataFrame for display
            display_df = pd.DataFrame({
                'Ticker': valid_tickers,
                'Setup Count': [counts[t] for t in valid_tickers]
            })
            st.dataframe(display_df, width='stretch', hide_index=True)
            
            st.info(f"You can copy these tickers for the Portfolio Backtest tab: `{st.session_state.prescan_results}`")
        else:
            st.warning("No tickers found with sufficient setups for the given criteria.")

with backtest_tab:
    st.header("Run Portfolio Backtest")
    with st.form("portfolio_backtest_form"):
        # Pre-populate with results from pre-scan if available
        tickers_portfolio = st.text_area("Tickers for Portfolio (comma-separated)", value=st.session_state.prescan_results, height=150, key="portfolio_tickers")
        
        c1, c2, c3, c4 = st.columns(4)
        strategy_type_portfolio = c1.selectbox("Strategy Type", strategy_options, key="portfolio_strat")
        timeframe_portfolio = c2.selectbox("Timeframe", TIMEFRAME_OPTIONS, key="portfolio_tf")
        max_open_positions = c3.number_input("Max Open Positions", min_value=1, value=5)
        commission_portfolio = c4.number_input("Commission ($ per share)", value=0.0, format="%.4f", key="portfolio_comm")

        c1, c2 = st.columns(2)
        start_date_portfolio = c1.date_input("Start Date", datetime(2000, 1, 1), key="portfolio_start")
        end_date_portfolio = c2.date_input("End Date", datetime.now(), key="portfolio_end")

        use_tuned_params_portfolio = st.checkbox("Use Tuned Strategy Params?", value=True, key="portfolio_tuned_params")

        run_portfolio_backtest = st.form_submit_button("Run Portfolio Backtest", width='stretch', disabled=DEMO_MODE)

    if run_portfolio_backtest:
        if not tickers_portfolio:
            st.error("Please enter tickers for the portfolio backtest.")
        else:
            ticker_list_portfolio = [t.strip().upper() for t in tickers_portfolio.split(',') if t.strip()]
            if not ticker_list_portfolio:
                st.error("No valid tickers entered for portfolio backtest.")
            else:
                st.info(f"Starting portfolio backtest for {len(ticker_list_portfolio)} tickers with {strategy_type_portfolio}...")
                log_container = st.expander("Portfolio Backtest Log", expanded=True)
                log_area = log_container.empty()
                log_messages = []

                # Re-purpose the log callback for portfolio backtest
                import logging
                root_logger = logging.getLogger()
                
                class UI_Handler(logging.Handler):
                    def emit(self, record):
                        log_messages.append(self.format(record))
                        log_area.code("\n".join(log_messages[-100:]))
                
                ui_handler = UI_Handler()
                formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt='%H:%M:%S')
                ui_handler.setFormatter(formatter)
                root_logger.addHandler(ui_handler)

                with st.spinner("Running portfolio walk-forward analysis... This will take a while."):
                    result = run_pybroker_portfolio_backtest(
                        tickers=ticker_list_portfolio,
                        strategy_type=strategy_type_portfolio,
                        timeframe_override=timeframe_portfolio,
                        start_date=start_date_portfolio.strftime('%Y-%m-%d'),
                        end_date=end_date_portfolio.strftime('%Y-%m-%d'),
                        max_open_positions=max_open_positions,
                        commission_cost=commission_portfolio,
                        use_tuned_strategy_params=use_tuned_params_portfolio,
                    )
                
                root_logger.removeHandler(ui_handler) # Clean up handler
                st.success("Portfolio backtest complete!")

                if result:
                    st.subheader("Portfolio Performance Metrics")
                    metrics_display_df = result.metrics_df.astype(str)
                    st.dataframe(metrics_display_df, width='stretch')

                    st.subheader("Portfolio Equity Curve")
                    fig_equity = plot_performance_vs_benchmark(result, f'Portfolio Equity for {strategy_type_portfolio}')
                    if fig_equity:
                        st.pyplot(fig_equity)
                else:
                    st.warning("No backtest results to display.")