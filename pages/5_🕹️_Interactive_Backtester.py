import streamlit as st
from datetime import datetime
import traceback

# Import the necessary functions from your backtesting engine
from quant_engine import (
    find_model_version_dirs,
    get_available_backtest_runs,
    run_pybroker_full_backtest,
    plot_performance_vs_benchmark,
    plot_trades_on_chart
)

st.set_page_config(page_title="Interactive Backtester", layout="wide")

st.title("🕹️ Interactive Strategy Backtester")

st.markdown("""
This tool allows you to run a backtest on a single ticker using a pre-trained model and visualize its performance.

**Important:** This is an **in-sample** backtest. It's useful for visualizing how a saved model behaves over different time periods, but it is **not** a substitute for the out-of-sample results from the `train` command.

**Prerequisite:** You must first train a model for the ticker and strategy you wish to test using the `train` command (e.g., `python quant_engine.py train --ticker SPY --strategy-type uptrend_pullback`).
""")

# --- Session State Initialization ---
if 'bt_artifacts' not in st.session_state:
    st.session_state.bt_artifacts = None

# --- UI Controls ---
st.subheader("Backtest Configuration")

def clear_bt_results():
    """Callback to clear backtest results when the selection changes."""
    if 'bt_artifacts' in st.session_state:
        st.session_state.bt_artifacts = None

available_models = get_available_backtest_runs()

col11, col12 = st.columns([2, 2])

with col11:
    selected_model = st.selectbox(
        "Select a Trained Model (Ticker & Strategy):",
        options=available_models,
        help="This list is populated from completed backtest runs found in the database.",
        key="selected_model_bt",
        on_change=clear_bt_results
    )

if selected_model:
    parts = selected_model.split(' - ')
    if len(parts) == 3:
        ticker_input, strategy_select, timeframe = parts
    else:
        ticker_input, strategy_select = parts[0], parts[1]
        timeframe = '1d' # Default fallback
else:
    ticker_input, strategy_select, timeframe = None, None, None

versions = []
if ticker_input and strategy_select:
    full_version_dirs = find_model_version_dirs(ticker_input, strategy_select, timeframe)
    if full_version_dirs:
        prefix_new = f"{ticker_input}_{strategy_select}_{timeframe}_"
        prefix_legacy = f"{ticker_input}_{strategy_select}_"
        for d in full_version_dirs:
            if d.startswith(prefix_new):
                versions.append(d.replace(prefix_new, ''))
            elif d.startswith(prefix_legacy):
                versions.append(d.replace(prefix_legacy, ''))
        versions.sort(reverse=True) # Sort in descending order

with col12:
    selected_version = st.selectbox(
        "Select Model Version:",
        options=versions,
        help="Select the timestamped version of the model to view. This list is populated from the artifacts directory.",
        key="selected_model_version_bt",
        on_change=clear_bt_results,
        disabled=not versions
    )

col21, col22 = st.columns([2, 2])

with col21:
    start_date_input = st.date_input(
        "Start Date",
        value=datetime(2000, 1, 1),
        key="start_date_bt",
        on_change=clear_bt_results
    )

with col22:
    end_date_input = st.date_input(
        "End Date", value=datetime.now(), key="end_date_bt", on_change=clear_bt_results
    )

c31, c32 = st.columns([2, 2])
with c31:
    commission_input = st.number_input("Commission ($ per share)", value=0.0, format="%.4f", key="commission_bt", on_change=clear_bt_results, help="Override the commission used for this specific backtest run.")

with c32:
    st.write("") # Spacer
    run_button = st.button("🚀 Run Backtest", width='stretch', disabled=not selected_version)

# --- Main Logic ---
if run_button:
    if not selected_model or not selected_version:
        st.warning("Please select a model and a version to run the backtest.")
        st.session_state.bt_artifacts = None
    else:
        with st.spinner(f"Running backtest for {ticker_input} with {strategy_select} strategy..."):
            try:
                backtest_artifacts = run_pybroker_full_backtest(
                    ticker=ticker_input,
                    strategy_type=strategy_select,
                    model_version=selected_version,
                    start_date=start_date_input.strftime('%Y-%m-%d'),
                    end_date=end_date_input.strftime('%Y-%m-%d'),
                    commission_cost=commission_input,
                    timeframe_override=timeframe
                )
                st.session_state.bt_artifacts = backtest_artifacts
                st.session_state.bt_ticker = ticker_input # Save for display
                st.session_state.bt_strategy = strategy_select # Save for display
                if backtest_artifacts is None:
                    st.error(f"Backtest failed. This usually means a model for '{ticker_input}' with strategy '{strategy_select}' has not been trained yet or failed to train.")
            except Exception as e:
                st.error(f"An unexpected error occurred during the backtest: {e}")
                st.code(traceback.format_exc())
                st.session_state.bt_artifacts = None

# --- Display Results ---
if st.session_state.bt_artifacts:
    result = st.session_state.bt_artifacts['result']
    ticker = st.session_state.get('bt_ticker', 'N/A')
    strategy = st.session_state.get('bt_strategy', 'N/A')

    st.markdown("---")
    st.header(f"Backtest Results for {ticker} ({strategy})")

    tab1, tab2, tab3 = st.tabs(["📊 Summary Metrics", "📈 Equity Curve", "📉 Trade Chart"])

    with tab1:
        # --- Convert metrics DataFrame to be Arrow-compatible ---
        # The metrics_df from pybroker can have mixed types in its 'value' column
        # (e.g., numbers and datetimes), which causes pyarrow serialization errors.
        # We convert the entire DataFrame to strings for robust display.
        metrics_display_df = result.metrics_df.astype(str)
        st.dataframe(metrics_display_df, width='stretch')

    with tab2:
        # --- Use a standard if/else block to prevent printing the DeltaGenerator object ---
        fig_equity = plot_performance_vs_benchmark(result, title=f"Equity Curve for {ticker} ({strategy})", timeframe=timeframe)
        if fig_equity:
            st.pyplot(fig_equity)
        else:
            st.warning("Could not generate equity curve plot.")

    with tab3:
        # --- Use a standard if/else block to prevent printing the DeltaGenerator object ---
        fig_trades = plot_trades_on_chart(result, ticker, title=f"Trades for {ticker} ({strategy})", timeframe=timeframe)
        if fig_trades:
            st.pyplot(fig_trades)
        else:
            st.warning("Could not generate trade chart.")