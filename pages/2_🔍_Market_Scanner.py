import streamlit as st
import pandas as pd
import os
import traceback
import json
from datetime import datetime

from core.model import Company
from load_cfg import DEMO_MODE, WORKING_DIRECTORY
from pybroker_trainer.strategy_loader import get_strategy_class_map, STRATEGY_CLASS_MAP
from quant_engine import run_scan, get_default_tickers
from core.db import get_db
from scanners.scanner_loader import SCANNER_CLASS_MAP, load_scanner_class

if DEMO_MODE:
    st.warning(
        "**Demo Mode Active:** Running new interactive scans is disabled. "
        "To enable this feature, set `DEMO_MODE = False` in `load_cfg.py`.",
        icon="🔒"
    )

st.set_page_config(page_title="Market Scanner", layout="wide")

st.title("🔍 Market Scanner")

if DEMO_MODE:
    st.warning(
        "**Demo Mode Active:** Running new scans is disabled. "
        "To enable this feature, set `DEMO_MODE = False` in `load_cfg.py`.",
        icon="🔒"
    )

@st.cache_data
def _get_default_tickers(limit=100):
    """Cached function to get default tickers."""
    return get_default_tickers(limit=limit)


def get_signal_scanner_content():
    st.header("Strategy Signal Scanner")
    st.markdown("This scanner runs pre-defined strategies (from the `strategies` directory) against a list of tickers to find active trading signals.")

    # --- Session State ---
    if 'signal_screener_results_df' not in st.session_state:
        st.session_state.signal_screener_results_df = pd.DataFrame()
    if 'last_scan_time' not in st.session_state:
        st.session_state.last_scan_time = None
    if 'scan_source' not in st.session_state:
        st.session_state.scan_source = "file" # 'file' or 'interactive'

    SCAN_RESULTS_FILE = os.path.join(WORKING_DIRECTORY, 'scan_results.json')

    # --- Helper to load from file ---
    def load_from_file():
        if os.path.exists(SCAN_RESULTS_FILE):
            try:
                last_modified_time = datetime.fromtimestamp(os.path.getmtime(SCAN_RESULTS_FILE))
                with open(SCAN_RESULTS_FILE, 'r') as f:
                    data = json.load(f)
                st.session_state.signal_screener_results_df = pd.DataFrame(data) if data else pd.DataFrame()
                st.session_state.last_scan_time = last_modified_time
                st.session_state.scan_source = "file"
            except (json.JSONDecodeError, IOError) as e:
                st.error(f"Error loading cached scan results file: {e}")
                st.session_state.signal_screener_results_df = pd.DataFrame()
                st.session_state.last_scan_time = None
        else:
            st.session_state.signal_screener_results_df = pd.DataFrame()
            st.session_state.last_scan_time = None

    # --- UI for Interactive Scan ---
    with st.expander("🔬 Run a New Signal Scan", expanded=False):
        with st.form("scan_form"):
            strategy_options = list(get_strategy_class_map().keys())
            default_tickers_str = ",".join(_get_default_tickers(limit=100))
            
            tickers_input = st.text_area("Tickers (comma-separated)", value=default_tickers_str, height=100)
            strategies_input = st.multiselect("Strategies to Scan", options=strategy_options, default=strategy_options)
            
            run_interactive_scan = st.form_submit_button("Run Interactive Scan", disabled=DEMO_MODE)

    if run_interactive_scan:
        if not tickers_input:
            st.error("Please enter at least one ticker.")
        elif not strategies_input:
            st.error("Please select at least one strategy.")
        else:
            tickers_list = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
            
            progress_bar = st.progress(0)
            progress_text = st.empty()

            def update_progress(progress, text):
                progress_bar.progress(progress)
                progress_text.text(text)

            try:
                with st.spinner("Running interactive scan..."):
                    interactive_results = run_scan(
                        ticker_list=tickers_list,       
                        strategy_list=strategies_input, 
                        progress_callback=update_progress
                    )
                st.session_state.signal_screener_results_df = pd.DataFrame(interactive_results) if interactive_results else pd.DataFrame()
                st.session_state.last_scan_time = datetime.now()
                st.session_state.scan_source = "interactive"
                st.success("Interactive scan complete!")
            except Exception as e:
                st.error(f"Error during interactive scan: {e}")
                st.session_state.signal_screener_results_df = pd.DataFrame()
                st.session_state.last_scan_time = None

    if st.button("Refresh from File", help="Load the latest scan results from the saved file."):
        st.session_state.signal_screener_results_df = pd.DataFrame()
        st.session_state.last_scan_time = None
        st.session_state.scan_source = "file"
        st.rerun()

    if st.session_state.signal_screener_results_df.empty and st.session_state.last_scan_time is None:
        load_from_file()

    results_df = st.session_state.signal_screener_results_df
    last_scan_time = st.session_state.last_scan_time

    if last_scan_time:
        st.info(f"Scan results last updated: **{last_scan_time.strftime('%Y-%m-%d %H:%M:%S')}**")
    else:
        st.warning("Scan results file not found. Please run the scanner first: `python quant_engine.py scan`")

    if results_df is not None and not results_df.empty:
        if results_df.empty:
            st.success("✅ No active trading signals found in the latest scan.")
        else:
            st.subheader("Active Trading Signals")

            def format_probs(p):
                if isinstance(p, list) and len(p) > 1:
                    return p[1]
                return p

            results_df['probability'] = results_df['probabilities'].apply(format_probs)
            results_df['risk_pct'] = results_df['risk_per_trade_pct'] * 100 if 'risk_per_trade_pct' in results_df.columns else None
            results_df['stockcharts'] = "https://stockcharts.com/sc3/ui/?s=" + results_df['ticker']
            results_df['yahoo_finance'] = "https://finance.yahoo.com/quote/" + results_df['ticker']
            results_df = results_df.sort_values(by='probability', ascending=False)

            # Define the columns to display
            display_columns = [
                'ticker', 'strategy', 'date', 'close', 'probability', 
                'risk_pct', 'stop_loss_price', 'take_profit_price', 'reward_risk_ratio', 
                'stockcharts', 'yahoo_finance'
            ]

            # Filter display_columns to only include those present in results_df
            existing_display_columns = [col for col in display_columns if col in results_df.columns]

            # --- Display DataFrame with existing columns ---
            st.dataframe(
                results_df[existing_display_columns],
                column_config={
                    "ticker": st.column_config.TextColumn("Ticker"),
                    "strategy": st.column_config.TextColumn("Strategy"),
                    "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
                    "close": st.column_config.NumberColumn("Close Price", format="$%.2f"),
                    "probability": st.column_config.ProgressColumn(
                        "Win Prob.",
                        help="Model's predicted probability of a winning trade.",
                        format="%.2f", min_value=0, max_value=1,
                    ),
                    "risk_pct": st.column_config.NumberColumn(
                        "Risk %",
                        help="Suggested risk for this trade as a percentage of portfolio equity.",
                        format="%.2f%%"
                    ),
                    "stop_loss_price": st.column_config.NumberColumn(
                        "Stop Loss",
                        help="Suggested stop-loss price for this trade.",
                        format="$%.2f"
                    ),
                    "take_profit_price": st.column_config.NumberColumn(
                        "Take Profit",
                        help="Suggested take-profit price for this trade.",
                        format="$%.2f"
                    ),
                    "reward_risk_ratio": st.column_config.NumberColumn(
                        "R:R",
                        help="Calculated Reward to Risk Ratio for this trade setup.",
                        format="%.2f"
                    ),
                    "stockcharts": st.column_config.LinkColumn(
                        "StockCharts",
                        help="Link to the ticker's StockCharts page.",
                        display_text="📈 Chart"
                    ),
                    "yahoo_finance": st.column_config.LinkColumn(
                        "Yahoo Finance",
                        help="Link to the ticker's Yahoo Finance page.",
                        display_text="ℹ️ Info"
                    )
                },
                width='stretch', hide_index=True
            )

            st.subheader("Market Regime Analysis")
            st.markdown("""
            This analysis counts the number of signals generated by each strategy in the latest scan.
            The prevalence of certain strategy types can provide insights into the current market's character.
            """)
            strategy_counts = results_df['strategy'].value_counts()
            strategy_interpretations = {
                s: c.description if hasattr(c, 'description') else "No description available." for s, c in STRATEGY_CLASS_MAP.items()
            }

            c1, c2 = st.columns(2)
            with c1:
                st.write("##### Signal Counts by Strategy")
                st.bar_chart(strategy_counts)
            with c2:
                st.write("##### Strategy Descriptions")
                interpretation_df = pd.DataFrame(
                    [(strat, strategy_interpretations.get(strat, "No description available.")) for strat in strategy_counts.index],
                    columns=['Strategy', 'Description']
                )
                st.dataframe(interpretation_df, hide_index=True, width='stretch')


def get_generic_scanner_content():
    st.header("Generic Stock Scanner")
    st.markdown("Select a pre-defined scanner or build a custom screen to find stocks matching your criteria.")

    st.caption(
        "ℹ️ **Note on Data & Indicators:** All calculations are based on historical data from Yahoo Finance. "
        "Minor discrepancies may exist when comparing with other platforms due to differences in data sources, "
        "adjustment methodologies, and indicator calculation nuances. Technical indicators are calculated using the industry-standard `talib` library."
    )

    # --- Session State for Generic Screener ---
    if 'generic_scanner_results_df' not in st.session_state:
        st.session_state.generic_scanner_results_df = pd.DataFrame()
    if 'generic_filters' not in st.session_state:
        st.session_state.generic_filters = []
    if 'last_scanner_name' not in st.session_state:
        st.session_state.last_scanner_name = ""
    if 'generic_scan_time' not in st.session_state:
        st.session_state.generic_scan_time = None

    # --- Screener UI ---
    # Move the scanner selection outside the form to allow dynamic parameter updates.
    c1, c2 = st.columns(2)
    # Prepend the Generic Screener to the list of scanners
    scanner_options = ["generic_screener"] + sorted([k for k in SCANNER_CLASS_MAP.keys() if k != 'generic_screener'])
    scanner_name = c1.selectbox("Select Scanner", options=scanner_options, key="scanner_name_select")
    market = c2.selectbox("Market", options=['us', 'ca'])

    # Clear results if the screener selection changes
    if scanner_name != st.session_state.last_scanner_name:
        st.session_state.generic_scanner_results_df = pd.DataFrame()
        st.session_state.generic_filters = [] # Also clear filters for the generic scanner
        st.session_state.last_scanner_name = scanner_name

    if scanner_name == 'generic_screener':
        st.markdown("---")
        st.subheader("Build Your Custom Screen")
        st.caption("Add filters to narrow down the universe of stocks. The screener will combine all active filters.")

        # --- UI for building filters, wrapped in a form ---
        with st.form("add_filter_form"):
            st.markdown("##### Configure and Add Filters")
            desc_tab, fund_tab, tech_tab = st.tabs(["Descriptive", "Fundamental", "Technical"])

            with desc_tab:
                st.markdown("###### Market Capitalization")
                # Define market cap ranges
                mc_options = ["Any", "Mega (>200B)", "Large (10B-200B)", "Mid (2B-10B)", "Small (300M-2B)", "Micro (<300M)"]
                mc_selection = st.selectbox("Select Market Cap Range", options=mc_options, index=0, key="mc_select")

                st.markdown("###### Sector")
                # Fetch distinct sectors from the database to populate the multiselect
                db = next(get_db())
                try:
                    sectors = [s[0] for s in db.query(Company.sectorkey).filter(Company.sectorkey.isnot(None)).distinct().order_by(Company.sectorkey).all()]
                finally:
                    db.close()
                
                sector_selection = st.multiselect("Select Sectors", options=sectors, key="sector_select")

                st.markdown("###### Volume & Liquidity")
                c1, c2 = st.columns(2)
                min_avg_volume = c1.number_input("Min. Avg. Volume", min_value=0, value=100000, key="gs_min_vol")
                volume_lookback = c2.number_input("Avg. Volume Lookback", min_value=5, max_value=200, value=50, key="gs_vol_lookback")

            with fund_tab:
                st.markdown("###### Valuation")
                pe_min, pe_max = st.slider("P/E Ratio", min_value=0.0, max_value=100.0, value=(0.0, 100.0), step=0.5, key="pe_slider")

                st.markdown("###### Profitability")
                c1, c2 = st.columns(2)
                roe_min = c1.slider("Min. Return on Equity (ROE %)", min_value=-50.0, max_value=100.0, value=-50.0, step=1.0, key="roe_slider")
                profit_margin_min = c2.slider("Min. Profit Margin (%)", min_value=-50.0, max_value=100.0, value=-50.0, step=1.0, key="pm_slider")

                st.markdown("###### Financial Health")
                c1, c2 = st.columns(2)
                pb_max = c1.slider("Max. Price-to-Book (P/B)", min_value=0.0, max_value=20.0, value=20.0, step=0.1, key="pb_slider")
                de_max = c2.slider("Max. Debt-to-Equity", min_value=0.0, max_value=5.0, value=5.0, step=0.1, key="de_slider")

            with tech_tab:
                st.markdown("###### Relative Strength")
                c1, c2, c3 = st.columns(3)
                rs_min_1y = c1.slider("RS Percentile (1-Year)", min_value=0, max_value=100, value=0, key="rs1y_slider")
                rs_min_6m = c2.slider("RS Percentile (6-Month)", min_value=0, max_value=100, value=0, key="rs6m_slider")
                rs_min_3m = c3.slider("RS Percentile (3-Month)", min_value=0, max_value=100, value=0, key="rs3m_slider")

                st.markdown("###### Performance")
                perf_52w_min = st.slider("52-Week Performance (%)", min_value=-100, max_value=500, value=-100, key="perf52w_slider")

                st.markdown("---")
                st.markdown("###### Moving Averages")
                c1, c2 = st.columns(2)
                with c1:
                    sma_period = st.number_input("SMA Period", min_value=5, max_value=200, value=20, key="sma_period")
                    sma_op = st.selectbox("Price vs SMA", options=["Any", "Above", "Below"], index=0, key="sma_op")
                with c2:
                    ema_period = st.number_input("EMA Period", min_value=5, max_value=200, value=20, key="ema_period")
                    ema_op = st.selectbox("Price vs EMA", options=["Any", "Above", "Below"], index=0, key="ema_op")

                st.markdown("---")
                st.markdown("###### Oscillators & Momentum")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("###### MACD")
                    macd_op = st.selectbox("MACD Signal", options=["Any", "MACD crosses above Signal", "MACD crosses below Signal"], index=0, key="macd_op")
                    with st.expander("MACD Parameters"):
                        macd_fast = st.number_input("Fast Period", value=12, min_value=1, key="macd_fast")
                        macd_slow = st.number_input("Slow Period", value=26, min_value=1, key="macd_slow")
                        macd_signal = st.number_input("Signal Period", value=9, min_value=1, key="macd_signal")
                with c2:
                    st.markdown("###### Stochastic")
                    stoch_op = st.selectbox("Stochastic Signal", options=["Any", "%K crosses above %D", "%K crosses below %D", "%K is Overbought (>80)", "%K is Oversold (<20)"], index=0, key="stoch_op")
                    with st.expander("Stochastic Parameters"):
                        stoch_fastk = st.number_input("Fast %K Period", value=14, min_value=1, key="stoch_fastk")
                        stoch_slowk = st.number_input("Slow %K Period", value=3, min_value=1, key="stoch_slowk")
                        stoch_slowd = st.number_input("Slow %D Period", value=3, min_value=1, key="stoch_slowd")

                st.markdown("---")
                st.markdown("###### Volatility & Trend Strength")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("###### Bollinger Bands")
                    bb_op = st.selectbox("Bollinger Bands Signal", options=["Any", "Price crosses above Upper Band", "Price crosses below Lower Band"], index=0, key="bb_op")
                    with st.expander("Bollinger Bands Parameters"):
                        bb_period = st.number_input("BBands Period", value=20, min_value=1, key="bb_period")
                        bb_std_up = st.number_input("Std. Dev. Up", value=2.0, min_value=0.1, step=0.1, key="bb_std_up")
                        bb_std_dn = st.number_input("Std. Dev. Down", value=2.0, min_value=0.1, step=0.1, key="bb_std_dn")

            add_filters_button = st.form_submit_button("Add Configured Filters", width='stretch')

        if add_filters_button:
            # Clear filters before adding new ones to prevent duplicates from multiple clicks
            st.session_state.generic_filters = []

            # Descriptive
            mc_map = {
                "Mega (>200B)": (200_000_000_000, None), "Large (10B-200B)": (10_000_000_000, 200_000_000_000),
                "Mid (2B-10B)": (2_000_000_000, 10_000_000_000), "Small (300M-2B)": (300_000_000, 2_000_000_000),
                "Micro (<300M)": (None, 300_000_000),
            }
            if mc_selection != "Any":
                min_val, max_val = mc_map[mc_selection]
                if min_val is not None: st.session_state.generic_filters.append({"display": f"Market Cap > {mc_selection}", "name": "marketcap", "op": ">=", "value": min_val})
                if max_val is not None: st.session_state.generic_filters.append({"display": f"Market Cap < {mc_selection}", "name": "marketcap", "op": "<", "value": max_val})
            if sector_selection:
                st.session_state.generic_filters.append({"display": f"Sector in [{', '.join(sector_selection)}]", "name": "sector", "op": "in", "value": sector_selection})
            
            # Add default volume filters, which can be removed by the user
            st.session_state.generic_filters.append({"display": f"Min. Avg. Volume > {min_avg_volume:,}", "name": "min_avg_volume", "op": ">", "value": min_avg_volume})
            st.session_state.generic_filters.append({"display": f"Avg. Volume Lookback = {volume_lookback} days", "name": "volume_lookback_days", "op": "=", "value": volume_lookback})

            # Fundamental
            # Check if the slider values have been changed from their defaults
            if pe_min > 0.0 or pe_max < 100.0:
                # Create a single display text for the range
                display_text = ""
                if pe_min > 0.0 and pe_max < 100.0:
                    display_text = f"{pe_min:.1f} < P/E < {pe_max:.1f}"
                elif pe_min > 0.0:
                    display_text = f"P/E > {pe_min:.1f}"
                else: # pe_max < 100.0
                    display_text = f"P/E < {pe_max:.1f}"
                
                st.session_state.generic_filters.append({"display": display_text, "name": "trailingpe", "op": ">", "value": pe_min})
                st.session_state.generic_filters.append({"display": "P/E Range Max", "name": "trailingpe", "op": "<", "value": pe_max, "is_display_only": True})

            if roe_min > -50.0: st.session_state.generic_filters.append({"display": f"ROE > {roe_min:.1f}%", "name": "returnonequity", "op": ">", "value": roe_min})
            if profit_margin_min > -50.0: st.session_state.generic_filters.append({"display": f"Profit Margin > {profit_margin_min:.1f}%", "name": "profitmargins", "op": ">", "value": profit_margin_min})
            if pb_max < 20.0: st.session_state.generic_filters.append({"display": f"P/B < {pb_max:.1f}", "name": "pricetobook", "op": "<", "value": pb_max})
            if de_max < 5.0: st.session_state.generic_filters.append({"display": f"Debt/Equity < {de_max:.1f}", "name": "debttoequity", "op": "<", "value": de_max})

            # Technical
            if rs_min_1y > 0: st.session_state.generic_filters.append({"display": f"RS (1Y) > {rs_min_1y}", "name": "relative_strength_percentile_252", "op": ">", "value": rs_min_1y})
            if rs_min_6m > 0: st.session_state.generic_filters.append({"display": f"RS (6M) > {rs_min_6m}", "name": "relative_strength_percentile_126", "op": ">", "value": rs_min_6m})
            if rs_min_3m > 0: st.session_state.generic_filters.append({"display": f"RS (3M) > {rs_min_3m}", "name": "relative_strength_percentile_63", "op": ">", "value": rs_min_3m})
            if perf_52w_min > -100: st.session_state.generic_filters.append({"display": f"52W Perf > {perf_52w_min}%", "name": "_52weekchange", "op": ">", "value": perf_52w_min})
            if sma_op != "Any":
                op_symbol = ">" if sma_op == "Above" else "<"
                st.session_state.generic_filters.append({"display": f"Price {op_symbol} SMA({sma_period})", "name": "sma", "op": op_symbol, "value": {"period": sma_period}})
            if ema_op != "Any":
                op_symbol = ">" if ema_op == "Above" else "<"
                st.session_state.generic_filters.append({"display": f"Price {op_symbol} EMA({ema_period})", "name": "ema", "op": op_symbol, "value": {"period": ema_period}})
            if macd_op != "Any":
                op_val = "cross_above" if "above" in macd_op else "cross_below"
                st.session_state.generic_filters.append({"display": f"MACD({macd_fast},{macd_slow},{macd_signal}) crosses {'above' if op_val == 'cross_above' else 'below'} Signal", "name": "macd", "op": op_val, "value": {"fastperiod": macd_fast, "slowperiod": macd_slow, "signalperiod": macd_signal}})
            if stoch_op != "Any":
                if "crosses above" in stoch_op: op_val, val = "cross_above", {}
                elif "crosses below" in stoch_op: op_val, val = "cross_below", {}
                elif "Overbought" in stoch_op: op_val, val = "above", {"value": 80}
                else: op_val, val = "below", {"value": 20}
                val.update({"fastk_period": stoch_fastk, "slowk_period": stoch_slowk, "slowd_period": stoch_slowd})
                st.session_state.generic_filters.append({"display": f"Stoch({stoch_fastk},{stoch_slowk},{stoch_slowd}): {stoch_op}", "name": "stoch", "op": op_val, "value": val})
            if bb_op != "Any":
                op_val = "cross_above_upper" if "above" in bb_op else "cross_below_lower"
                st.session_state.generic_filters.append({"display": f"BBands({bb_period},{bb_std_up}): {bb_op}", "name": "bbands", "op": op_val, "value": {"period": bb_period, "nbdevup": bb_std_up, "nbdevdn": bb_std_dn}})

        # --- Display active filters and allow removal ---
        if st.session_state.generic_filters:
            st.markdown("##### Active Filters")
            # Create a copy to iterate over, so we can modify the original list
            for i, f in enumerate(list(st.session_state.generic_filters)):
                # Skip displaying secondary parts of a range filter
                if f.get("is_display_only"):
                    continue
                cols = st.columns([5, 1])
                cols[0].info(f['display'])
                if cols[1].button("Remove", key=f"remove_{i}", width='stretch'):
                    st.session_state.generic_filters.pop(i)
                    st.rerun()

        run_scanner = st.button("Run Generic Screener", width='stretch', type="primary", disabled=DEMO_MODE)
        # Pass the list of filters and define which columns to show in the output
        params = {
            'filters': st.session_state.generic_filters,
            'output_columns': ['symbol', 'longname', 'sector', 'industry', 'marketcap', 'trailingpe', 'dividendyield', 'returnonequity', 'relative_strength_percentile_252', '_52weekchange', 'stockcharts', 'yahoo_finance']
        } 
        
        if run_scanner:
            if not params['filters']:
                st.warning("Please add at least one filter to run the scan.")
            else:
                scanner_class = load_scanner_class('generic_screener')
                if scanner_class:
                    with st.spinner("Running generic scanner..."):
                        db = next(get_db())
                        try:
                            # Extract volume params from the filter list and pass them to the scanner instance
                            # This keeps the UI consistent while still using the BaseScanner's dynamic volume calculation
                            filters = params.get('filters', [])
                            min_vol = next((f['value'] for f in filters if f['name'] == 'min_avg_volume'), 100000)
                            vol_lookback = next((f['value'] for f in filters if f['name'] == 'volume_lookback_days'), 50)

                            # Remove them from the filter list that goes to the query builder
                            params['filters'] = [f for f in filters if f['name'] not in ['min_avg_volume', 'volume_lookback_days']]
                            
                            full_params = {'market': market, 'min_avg_volume': min_vol, 'volume_lookback_days': vol_lookback, **params}
                            scanner_instance = scanner_class(params=full_params)
                            results_df = scanner_instance.run_scan(db)
                            st.session_state.generic_scanner_results_df = results_df
                            st.session_state.generic_scan_time = datetime.now()
                            st.success(f"Scan complete! Found {len(results_df)} results.")
                        except Exception as e:
                            st.error(f"An error occurred while running the scanner: {e}")
                            st.code(traceback.format_exc())
                        finally:
                            db.close()
    else: # UI for pre-defined scanners
        # Display scanner description outside the form
        scanner_class = load_scanner_class(scanner_name)
        if scanner_class:
            description = scanner_class.get_description()
            st.info(description)

        # Use the scanner_name as the key for the form. This forces Streamlit to
        # re-render the form and its widgets with default values when the scanner changes.
        with st.form(key=f"form_for_{scanner_name}"):
            # Dynamically generate parameter inputs
            params = {}
            scanner_class = load_scanner_class(scanner_name)
            if scanner_class:
                param_definitions = scanner_class.define_parameters()
                # st.markdown("---")
                st.subheader("Scanner Parameters")
                cols = st.columns(len(param_definitions))
                for i, p_def in enumerate(param_definitions):
                    with cols[i]:
                        if p_def['type'] == 'int':
                            # Use a dynamic step for integer inputs
                            step_value = 1 if p_def.get('default', 0) < 1000 else 10000
                            params[p_def['name']] = st.number_input(
                                label=p_def['label'], 
                                value=p_def['default'], 
                                step=step_value)
                        elif p_def['type'] == 'float':
                            params[p_def['name']] = st.number_input(
                                label=p_def['label'], value=p_def['default'], format="%.2f")
                        elif p_def['type'] == 'select':
                            params[p_def['name']] = st.selectbox(
                                label=p_def['label'], 
                                options=p_def.get('options', []))
            
            run_scanner = st.form_submit_button("Run Scanner", width='stretch', disabled=DEMO_MODE)
            
            if run_scanner:
                if scanner_class:
                    with st.spinner(f"Running '{scanner_name}' scanner..."):
                        db = next(get_db())
                        try:
                            # Combine market with other dynamic params
                            full_params = {'market': market, **params}
                            scanner_instance = scanner_class(params=full_params)
                            results_df = scanner_instance.run_scan(db)
                            st.session_state.generic_scanner_results_df = results_df
                            st.session_state.generic_scan_time = datetime.now()
                            st.success(f"Scan complete! Found {len(results_df)} results.")
                        except Exception as e:
                            st.error(f"An error occurred while running the scanner: {e}")
                            st.code(traceback.format_exc())
                            st.session_state.generic_scanner_results_df = pd.DataFrame()
                        finally:
                            db.close()

    # --- Results Display ---
    if not st.session_state.generic_scanner_results_df.empty:
        results_df = st.session_state.generic_scanner_results_df
        if st.session_state.generic_scan_time:
            st.subheader(f"Scanner Results ({st.session_state.generic_scan_time.strftime('%Y-%m-%d %H:%M:%S')})")
        else:
            st.subheader("Scanner Results")
        if results_df.empty:
            st.success("✅ No stocks matched your criteria.")
        else:
            # Add links to external sites
            results_df['stockcharts'] = "https://stockcharts.com/sc3/ui/?s=" + results_df['symbol']
            results_df['yahoo_finance'] = "https://finance.yahoo.com/quote/" + results_df['symbol']

            st.dataframe(
                results_df, 
                width='stretch', 
                hide_index=True,
                column_config={
                    "stockcharts": st.column_config.LinkColumn(
                        "StockCharts",
                        display_text="📈 Chart"
                    ),
                    "yahoo_finance": st.column_config.LinkColumn(
                        "Yahoo Finance",
                        display_text="ℹ️ Info"
                    )
                }
            )


# --- Main App Logic ---
signal_tab, scanner_tab = st.tabs(["Signal Scanner", "Generic Scanner"])

with signal_tab:
    get_signal_scanner_content()

with scanner_tab:
    get_generic_scanner_content()