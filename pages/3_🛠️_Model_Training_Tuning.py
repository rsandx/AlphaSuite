import os
import pickle
import streamlit as st
import pandas as pd
import logging
import logging.handlers
from datetime import datetime
import threading
import queue
import time

from load_cfg import DEMO_MODE
from pybroker_trainer.strategy_loader import get_strategy_class_map, get_strategy_defaults, load_strategy_class
from quant_engine import (
    find_model_version_dirs,
    get_available_backtest_runs,
    run_pybroker_walkforward,
    run_tune_strategy,
    run_visualize_model,
    run_quick_test
)
from tools.yfinance_tool import TIMEFRAME_OPTIONS

st.set_page_config(page_title="Model Training & Tuning", layout="wide")
st.title("🛠️ Model Training & Tuning")

if DEMO_MODE:
    st.warning(
        "**Demo Mode Active:** All model tuning and training operations are disabled. "
        "To enable these features, set `DEMO_MODE = False` in `load_cfg.py`.",
        icon="🔒"
    )

# --- Add session state and stop checker for long-running processes ---
if 'tuning_in_progress' not in st.session_state:
    st.session_state.tuning_in_progress = False
if 'training_in_progress' not in st.session_state:
    st.session_state.training_in_progress = False
if 'quick_test_in_progress' not in st.session_state:
    st.session_state.quick_test_in_progress = False
if 'tuning_thread' not in st.session_state:
    st.session_state.tuning_thread = None
if 'training_thread' not in st.session_state:
    st.session_state.training_thread = None
if 'quick_test_thread' not in st.session_state:
    st.session_state.quick_test_thread = None
if 'log_queue' not in st.session_state:
    st.session_state.log_queue = None
if 'progress_queue' not in st.session_state:
    st.session_state.progress_queue = None
if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []
if 'stop_event' not in st.session_state:
    st.session_state.stop_event = None
if 'completion_message_tune' not in st.session_state:
    st.session_state.completion_message_tune = None
if 'completion_message_train' not in st.session_state:
    st.session_state.completion_message_train = None
if 'quick_test_results' not in st.session_state:
    st.session_state.quick_test_results = None

def run_tuning_in_thread(params, log_q, progress_q, stop_event):
    """Wrapper to run the tuning process in a separate thread and communicate via queues."""
    # Setup QueueHandler to route logs from the thread to the main UI
    qh = logging.handlers.QueueHandler(log_q)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt='%H:%M:%S')
    qh.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.addHandler(qh)

    def progress_callback(progress, text):
        progress_q.put((progress, text))

    try:
        # The stop_event_checker is a simple lambda that checks the thread-safe Event object.
        run_tune_strategy(
            ticker=params['ticker'],
            strategy_type=params['strategy_type'],
            n_calls=params['n_calls'],
            start_date=params['start_date'].strftime('%Y-%m-%d'),
            end_date=params['end_date'].strftime('%Y-%m-%d'),
            commission=params['commission'],
            max_drawdown=params['max_drawdown'],
            min_trades=params['min_trades'],
            min_win_rate=params['min_win_rate'],
            timeframe_override=params['timeframe'],
            progress_callback=progress_callback,
            stop_event_checker=lambda: stop_event.is_set()
        )
    finally:
        root_logger.removeHandler(qh)

st.markdown("""
This page provides a user interface for the core backend commands.
- **Quick Test:** Run a single, non-walk-forward backtest with custom parameters to quickly test strategy rules. For ML strategies, this uses a pass-through model to validate setups.
- **Tune Strategy:** Use Bayesian Optimization to find the best high-level parameters for a strategy (e.g., stop-loss multipliers, indicator periods).
- **Train Model:** Run a full walk-forward backtest. This process trains and tests a model on different time windows and saves the final trained model and its performance metrics.
- **Visualize Model Performance:** Load and analyze the results of a completed training run, including out-of-sample equity curves, trade charts, and feature importances.
""")

strategy_options = list(get_strategy_class_map().keys())

quick_test_tab, tune_tab, train_tab, visualize_tab = st.tabs(["Quick Test", "Tune Strategy", "Train Model", "Visualize Model Performance"])

with tune_tab:
    st.header("Tune Strategy Parameters")

    # Display completion message if it exists from a previous run
    if st.session_state.get('completion_message_tune'):
        message, msg_type = st.session_state.completion_message_tune
        if msg_type == "success":
            st.success(message)
        elif msg_type == "warning":
            st.warning(message)
        st.session_state.completion_message_tune = None # Clear the message

    # Show form only if tuning is not in progress
    if not st.session_state.tuning_in_progress:
        with st.form("tune_form"):
            c1, c2, c3, c4 = st.columns(4)
            ticker_tune = c1.text_input("Ticker", "SPY").upper()
            strategy_type_tune = c2.selectbox("Strategy Type", strategy_options, key="tune_strat")
            timeframe_tune = c3.selectbox("Timeframe", TIMEFRAME_OPTIONS, key="tune_tf")
            n_calls_tune = c4.number_input("Tuning Iterations (n_calls)", min_value=10, max_value=200, value=50)

            c1, c2 = st.columns(2)
            start_date_tune = c1.date_input("Start Date", datetime(2000, 1, 1), key="tune_start")
            end_date_tune = c2.date_input("End Date", datetime.now(), key="tune_end")

            st.subheader("Tuning Constraints")
            c1, c2, c3, c4 = st.columns(4)
            commission_tune = c1.number_input("Commission ($ per share)", value=0.0, format="%.4f")
            max_drawdown_tune = c2.number_input("Max Drawdown (%)", value=40.0)
            min_trades_tune = c3.number_input("Min Trades", value=20)
            min_win_rate_tune = c4.number_input("Min Win Rate (%)", value=40.0)

            run_tuning = st.form_submit_button("Run Tuning", width='stretch', disabled=DEMO_MODE)

            if run_tuning:
                st.session_state.tuning_in_progress = True
                # Store params in session state to use them after the rerun
                st.session_state.tune_params = {
                    "ticker": ticker_tune, "strategy_type": strategy_type_tune, "timeframe": timeframe_tune, "n_calls": n_calls_tune,
                    "start_date": start_date_tune, "end_date": end_date_tune, "commission": commission_tune,
                    "max_drawdown": max_drawdown_tune, "min_trades": min_trades_tune, "min_win_rate": min_win_rate_tune,
                }
                st.rerun()

    # This block runs when tuning_in_progress is True
    if st.session_state.tuning_in_progress:
        # If the thread object doesn't exist, it means we need to start it.
        if st.session_state.tuning_thread is None:
            st.session_state.log_queue = queue.Queue()
            st.session_state.progress_queue = queue.Queue()
            st.session_state.log_messages = [] # Clear old logs
            st.session_state.stop_event = threading.Event()

            thread = threading.Thread(
                target=run_tuning_in_thread,
                args=(st.session_state.tune_params, st.session_state.log_queue, st.session_state.progress_queue, st.session_state.stop_event),
                daemon=True # This is crucial to allow the main app to exit even if the thread is running
            )
            st.session_state.tuning_thread = thread
            thread.start()

        if st.button("Stop Tuning Process", width='stretch', type="primary"):
            if st.session_state.stop_event:
                st.session_state.stop_event.set()
            st.warning("Stop signal sent. The process will halt after the current iteration.")

        # --- UI Monitoring Section ---
        params = st.session_state.tune_params
        st.info(f"Tuning in progress for {params['ticker']} with {params['strategy_type']}...")
        log_container = st.expander("Tuning Log", expanded=True)
        log_area = log_container.empty()
        progress_bar = st.progress(0, "Initializing...")

        # Update UI from queues
        while not st.session_state.log_queue.empty():
            st.session_state.log_messages.append(st.session_state.log_queue.get().getMessage())
        log_area.code("\n".join(st.session_state.log_messages[-150:]))

        progress_val, progress_text = 0, "Running..."
        while not st.session_state.progress_queue.empty():
            progress_val, progress_text = st.session_state.progress_queue.get()
        progress_bar.progress(progress_val, text=progress_text)

        # Check thread status
        if st.session_state.tuning_thread and st.session_state.tuning_thread.is_alive():
            time.sleep(1) # Poll every second
            st.rerun()
        else:
            # Thread is finished, set message, clean up, and rerun
            was_stopped = st.session_state.stop_event and st.session_state.stop_event.is_set()
            if was_stopped:
                st.session_state.completion_message_tune = ("Tuning process was stopped by the user.", "success")
            else:
                st.session_state.completion_message_tune = ("Tuning complete! Best parameters have been saved.", "success")

            # Cleanup
            st.session_state.tuning_in_progress = False
            st.session_state.tuning_thread = None
            st.session_state.log_queue = None
            st.session_state.progress_queue = None
            st.session_state.stop_event = None
            st.session_state.log_messages = []
            st.rerun()

def run_training_in_thread(params, log_q, stop_event):
    """Wrapper to run the training process in a separate thread and log via a queue."""
    # Setup QueueHandler to route logs from the thread to the main UI
    qh = logging.handlers.QueueHandler(log_q)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt='%H:%M:%S')
    qh.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.addHandler(qh)
    try:
        result, _, _ = run_pybroker_walkforward(
            ticker=params['ticker'],
            strategy_type=params['strategy_type'],
            timeframe_override=params['timeframe'],
            model_n_calls=params['model_n_calls'],
            start_date=params['start_date'].strftime('%Y-%m-%d'),
            end_date=params['end_date'].strftime('%Y-%m-%d'),
            tune_hyperparameters=params['tune_hyperparams'],
            save_assets=True,
            use_tuned_strategy_params=params['use_tuned_strategy_params'],
            model_version=params['model_version'],
            commission_cost=params['commission'],
            stop_event_checker=lambda: stop_event.is_set(),
        )
    finally:
        # Important to remove the handler to avoid duplicate logs on subsequent runs
        root_logger.removeHandler(qh)

with train_tab:
    st.header("Train Final Model")

    # Display completion message if it exists from a previous run
    if st.session_state.get('completion_message_train'):
        message, msg_type = st.session_state.completion_message_train
        if msg_type == "success":
            st.success(message)
        elif msg_type == "warning":
            st.warning(message)
        st.session_state.completion_message_train = None # Clear the message

    ticker, strat, timeframe = None, None, None

    # Show form only if training is not in progress
    if not st.session_state.training_in_progress:
        c1, c2, c3, c4 = st.columns([1,1,1,1])
        ticker_train = c1.text_input("Ticker", ticker if ticker else "SPY", key="train_ticker").upper()
        strategy_type_train = c2.selectbox("Strategy Type", strategy_options, index=strategy_options.index(strat) if strat else 0, key="train_strat")
        timeframe_train = c3.selectbox("Timeframe", TIMEFRAME_OPTIONS, index=TIMEFRAME_OPTIONS.index(timeframe) if timeframe else 0, key="train_tf")
        model_n_calls_train = c4.number_input(
            "Model Tuning Iterations",
            min_value=10, max_value=100, value=32,
            help="Number of iterations for model hyperparameter tuning. Only used if 'Tune Model Hyperparameters?' is checked."
        )

        c1, c2, c3 = st.columns(3)
        start_date_train = c1.date_input("Start Date", datetime(2000, 1, 1), key="train_start")
        end_date_train = c2.date_input("End Date", datetime.now(), key="train_end")
        commission_train = c3.number_input("Commission ($ per share)", value=0.0, format="%.4f", key="train_comm")

        st.markdown("---")
        st.markdown("##### Parameter & Versioning Configuration")
        c1, c2 = st.columns([1,2])
        with c1:
            use_tuned_strategy_params_train = st.checkbox(
                "Use Tuned Strategy Params?",
                value=True,
                help="Use the best high-level parameters found during the 'Tune Strategy' step."
            )
            tune_model_hyperparams = st.checkbox(
                "Tune Model Hyperparameters?",
                value=True,
                help="Only applicable for Machine Learning-based strategies (e.g., those using LGBM)."
            )

        versions_train = []
        show_version_selector = use_tuned_strategy_params_train
        if show_version_selector:
            full_version_dirs = find_model_version_dirs(ticker_train, strategy_type_train, timeframe_train)
            if full_version_dirs:
                prefix_new = f"{ticker_train}_{strategy_type_train}_{timeframe_train}_"
                prefix_legacy = f"{ticker_train}_{strategy_type_train}_"
                for d in full_version_dirs:
                    if d.startswith(prefix_new):
                        versions_train.append(d.replace(prefix_new, ''))
                    elif d.startswith(prefix_legacy):
                        versions_train.append(d.replace(prefix_legacy, ''))
                versions_train.sort(reverse=True) # Sort in descending order
    
        version_index = 0 # Default to the latest
        with c2:
            selected_version_train = st.selectbox(
                "Select Parameter Version (optional)",
                options=versions_train,
                index=version_index if versions_train else None,
                help="Select a specific version to load parameters from. If left blank, the latest version will be used.",
                disabled=not show_version_selector
            )

        run_training = st.button("Run Training", width='stretch', disabled=DEMO_MODE)

        if run_training:
            st.session_state.training_in_progress = True
            st.session_state.train_params = {
                "ticker": ticker_train, "strategy_type": strategy_type_train, "timeframe": timeframe_train, "model_n_calls": model_n_calls_train,
                "start_date": start_date_train, "end_date": end_date_train, "commission": commission_train,
                "use_tuned_strategy_params": use_tuned_strategy_params_train, "tune_hyperparams": tune_model_hyperparams, 
                "model_version": selected_version_train if show_version_selector else None
            }
            st.rerun()
    
    # This block runs when training_in_progress is True
    if st.session_state.training_in_progress:
        # If the thread object doesn't exist, it means we need to start it.
        if st.session_state.training_thread is None:
            st.session_state.log_queue = queue.Queue()
            st.session_state.log_messages = [] # Clear old logs
            st.session_state.stop_event = threading.Event()

            thread = threading.Thread(
                target=run_training_in_thread,
                args=(st.session_state.train_params, st.session_state.log_queue, st.session_state.stop_event),
                daemon=True # This is crucial to allow the main app to exit even if the thread is running
            )
            st.session_state.training_thread = thread
            thread.start()

        if st.button("Stop Training Process", width='stretch', type="primary"):
            if st.session_state.stop_event:
                st.session_state.stop_event.set()
            st.warning("Stop signal sent. The process will halt at the next available checkpoint.")

        # --- UI Monitoring Section ---
        params = st.session_state.train_params
        st.info(f"Training in progress for {params['ticker']} with {params['strategy_type']}...")
        log_container = st.expander("Training Log", expanded=True)
        log_area = log_container.empty()

        # Update UI from queue
        while not st.session_state.log_queue.empty():
            st.session_state.log_messages.append(st.session_state.log_queue.get().getMessage())
        log_area.code("\n".join(st.session_state.log_messages[-150:]))

        # Check thread status
        if st.session_state.training_thread and st.session_state.training_thread.is_alive():
            time.sleep(1) # Poll every second
            st.rerun()
        else:
            # Thread is finished, set message, clean up, and rerun
            was_stopped = st.session_state.stop_event and st.session_state.stop_event.is_set()
            if was_stopped:
                st.session_state.completion_message_train = ("Training process was stopped by the user.", "success")
            else:
                st.session_state.completion_message_train = ("Training complete! Check logs for status. You can now visualize the model in the next tab.", "success")

            # Cleanup
            st.session_state.training_in_progress = False
            st.session_state.training_thread = None
            st.session_state.log_queue = None
            st.session_state.stop_event = None
            st.session_state.log_messages = []
            st.rerun()

def run_quick_test_in_thread(params, log_q, stop_event):
    """Wrapper to run the quick test process in a separate thread."""
    qh = logging.handlers.QueueHandler(log_q)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt='%H:%M:%S')
    qh.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.addHandler(qh)
    try:
        results = run_quick_test(
            ticker=params['ticker'],
            strategy_type=params['strategy_type'],
            timeframe_override=params['timeframe'],
            start_date=params['start_date'].strftime('%Y-%m-%d'),
            end_date=params['end_date'].strftime('%Y-%m-%d'),
            strategy_params=params['strategy_params'],
            commission_cost=params['commission'],
            stop_event_checker=lambda: stop_event.is_set()
        )
        # Use the log queue to pass back the result
        log_q.put(results)
    finally:
        root_logger.removeHandler(qh)

with quick_test_tab:
    st.header("Quick Test Strategy Rules")

    if st.session_state.quick_test_results:
        st.success("Quick Test complete!")
        results = st.session_state.quick_test_results
        st.subheader("Backtest Performance Metrics")
        st.dataframe(results["metrics_df"])
        st.subheader("Equity Curve")
        if results.get("performance_fig"): st.pyplot(results["performance_fig"])
        st.subheader("Trade Executions")
        if results.get("trades_fig"): st.pyplot(results["trades_fig"])
        with st.expander("View Detailed Trades"):
            st.dataframe(results["trades_df"])
        
        if results.get("noise_test_results"):
            st.subheader("🛡️ Noise & Leakage Test Results")
            for i, res in enumerate(results["noise_test_results"]):
                with st.expander(f"Fold {i+1} Analysis ({res.get('status', 'N/A')})", expanded=True):
                    if res.get('status') == 'PASS':
                        st.success(res.get('message'))
                    elif res.get('status') == 'FAIL':
                        st.error(res.get('message'))
                    else:
                        st.warning(res.get('message'))
                    
                    c1, c2 = st.columns(2)
                    c1.metric("Baseline Accuracy", f"{res.get('baseline_score', 0):.4f}")
                    c2.metric("Noise Accuracy", f"{res.get('noise_score', 0):.4f}", delta=f"{res.get('noise_score', 0) - res.get('baseline_score', 0):.4f}", delta_color="inverse")
                    
                    if res.get('permutation_importances'):
                        st.write("**Top Features (Permutation Importance):**")
                        fi_df = pd.DataFrame(res['permutation_importances'])
                        st.dataframe(fi_df, hide_index=True)

        if st.button("Run New Quick Test", width='stretch'):
            st.session_state.quick_test_results = None
            st.rerun()

    elif not st.session_state.quick_test_in_progress:
        strategy_type_qt = st.selectbox(
            "Select Strategy", strategy_options, key="qt_strat_select"
        )

        # Calculate the default params based on the selection above.
        default_params_json = "{}"
        if strategy_type_qt:
            import json
            strategy_class = load_strategy_class(strategy_type_qt)
            if strategy_class:
                default_params = get_strategy_defaults(strategy_class)
                default_params_json = json.dumps(default_params, indent=4)

        with st.form("quick_test_form"):
            c1, c2, c3 = st.columns(3)
            ticker_qt = c1.text_input("Ticker", "SPY", key="qt_ticker").upper()
            commission_qt = c2.number_input("Commission ($ per share)", value=0.0, format="%.4f", key="qt_comm")
            timeframe_qt = c3.selectbox("Timeframe", TIMEFRAME_OPTIONS, key="qt_tf")

            c1, c2 = st.columns(2)
            start_date_qt = c1.date_input("Start Date", datetime(2000, 1, 1), key="qt_start")
            end_date_qt = c2.date_input("End Date", datetime.now(), key="qt_end")

            st.markdown("###### Override Strategy Parameters (JSON format)")
            params_override_text = st.text_area("Parameters", value=default_params_json, height=250, help="Enter a JSON object of parameters to override strategy defaults.")

            run_qt = st.form_submit_button("Run Quick Test", width='stretch')

            if run_qt:
                import json
                try:
                    params_override = json.loads(params_override_text)
                    st.session_state.quick_test_in_progress = True
                    st.session_state.quick_test_params = {
                        "ticker": ticker_qt, "strategy_type": st.session_state.qt_strat_select, "commission": commission_qt,
                        "start_date": start_date_qt, "end_date": end_date_qt, "timeframe": timeframe_qt,
                        "strategy_params": params_override
                    }
                    st.rerun()
                except json.JSONDecodeError:
                    st.error("Invalid JSON in parameters override text area.")

    if st.session_state.quick_test_in_progress:
        if st.session_state.quick_test_thread is None:
            st.session_state.log_queue = queue.Queue()
            st.session_state.log_messages = []
            st.session_state.stop_event = threading.Event()
            thread = threading.Thread(target=run_quick_test_in_thread, args=(st.session_state.quick_test_params, st.session_state.log_queue, st.session_state.stop_event), daemon=True)
            st.session_state.quick_test_thread = thread
            thread.start()

        if st.button("Stop Quick Test", width='stretch', type="primary"):
            if st.session_state.stop_event: st.session_state.stop_event.set()
            st.warning("Stop signal sent. The process will halt.")

        params = st.session_state.quick_test_params
        st.info(f"Quick Test in progress for {params['ticker']} with {params['strategy_type']}...")
        log_container = st.expander("Live Log", expanded=True)
        log_area = log_container.empty()

        final_result = None
        while not st.session_state.log_queue.empty():
            msg = st.session_state.log_queue.get()
            if isinstance(msg, dict): final_result = msg
            # Handle LogRecord objects, ignoring others (like None)
            elif isinstance(msg, logging.LogRecord):
                st.session_state.log_messages.append(msg.getMessage())
        log_area.code("\n".join(st.session_state.log_messages[-150:]))

        if st.session_state.quick_test_thread and st.session_state.quick_test_thread.is_alive():
            time.sleep(1); st.rerun()
        else:
            st.session_state.quick_test_results = final_result
            st.session_state.quick_test_in_progress = False
            st.session_state.quick_test_thread = None
            st.session_state.log_queue = None
            st.session_state.stop_event = None
            st.session_state.log_messages = []
            st.rerun()

with visualize_tab:
    st.header("Visualize Trained Model Performance")
    st.markdown("Load the saved artifacts from a `train` run to visualize its out-of-sample performance, trade executions, and feature importances.")

    if 'viz_results' not in st.session_state:
        st.session_state.viz_results = None

    def clear_viz_results():
        """Callback to clear visualization results when the selection changes."""
        if 'viz_results' in st.session_state:
            st.session_state.viz_results = None

    # Get a list of all trained models from the pybroker_trainer artifacts directory
    trained_models = get_available_backtest_runs()

    if not trained_models:
        st.warning("No trained models found in the pybroker_trainer artifacts directory. Please run the 'train' command first.")
        selected_model = None
        selected_version_viz = None
    else:
        c1, c2 = st.columns(2)
        with c1:
            selected_model = st.selectbox(
                "Select a Trained Model to Visualize",
                options=trained_models,
                key="selected_model_viz",
                on_change=clear_viz_results
            )
        
        ticker_viz, strategy_type_viz, timeframe = selected_model.split(' - ', 2) if selected_model else (None, None, None)
        versions_viz = []
        if ticker_viz and strategy_type_viz:
            full_version_dirs = find_model_version_dirs(ticker_viz, strategy_type_viz, timeframe)
            if full_version_dirs:
                prefix_new = f"{ticker_viz}_{strategy_type_viz}_{timeframe}_"
                prefix_legacy = f"{ticker_viz}_{strategy_type_viz}_"
                for d in full_version_dirs:
                    if d.startswith(prefix_new):
                        versions_viz.append(d.replace(prefix_new, ''))
                    elif d.startswith(prefix_legacy):
                        versions_viz.append(d.replace(prefix_legacy, ''))
                versions_viz.sort(reverse=True) # Sort in descending order

        with c2:
            selected_version_viz = st.selectbox(
                "Select Model Version:",
                options=versions_viz,
                help="Select the timestamped version of the model to view.",
                key="selected_model_version_viz",
                on_change=clear_viz_results,
                disabled=not versions_viz
            )

    visualize_button = st.button("Load and Visualize", width='stretch', disabled=(not selected_version_viz))

    if visualize_button and selected_model and selected_version_viz:
        with st.spinner(f"Loading artifacts for {ticker_viz} - {strategy_type_viz} (version {selected_version_viz})..."):
            try:
                viz_assets = run_visualize_model(ticker_viz, strategy_type_viz, timeframe, selected_version_viz)
                st.session_state.viz_results = viz_assets
                if viz_assets is None:
                    st.error(f"Could not load visualization assets for {ticker} - {strategy_type_viz}. Check the logs. A common cause is missing artifact files (e.g., _results.pkl).")
                else:
                    st.success("Visualization assets loaded successfully.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.session_state.viz_results = None

    if st.session_state.viz_results:
        # Add a button to clear results and go back to the selection form
        if st.button("Clear Visualization & Select New Model", width='stretch'):
            st.session_state.viz_results = None
            st.rerun()

        results = st.session_state.viz_results

        st.subheader("Best Parameters Found")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Strategy Parameters (from `tune-strategy`)")
            if results.get("best_strategy_params"):
                st.json(results["best_strategy_params"])
            else:
                st.info("No tuned strategy parameters file (`_best_strategy_params.json`) found.")
        with col2:
            st.markdown("##### Model Hyperparameters (from `train`)")
            if results.get("best_model_params"):
                st.json(results["best_model_params"])
            else:
                st.info("No tuned model hyperparameters file (`_best_params.json`) found (tuning may have been disabled).")

        st.subheader("Walk-Forward Performance Metrics")
        st.dataframe(results["metrics_df"])

        st.subheader("Walk-Forward Equity Curve")
        if results.get("performance_fig"): st.pyplot(results["performance_fig"])

        st.subheader("Trade Executions on Price Chart")
        if results.get("trades_fig"): st.pyplot(results["trades_fig"])

        with st.expander("View Detailed Trades Data"):
            trades_df = results.get("trades_df")
            if trades_df is not None and not trades_df.empty:
                st.dataframe(trades_df)
            else:
                st.info("No trades were executed in this backtest, or trade data is unavailable.")

        st.subheader("Feature Importances (Averaged Across Folds)")
        if results.get("importance_fig"): st.pyplot(results["importance_fig"])
