import subprocess
import streamlit as st
import pandas as pd
from datetime import datetime
import sys
import time
import json

from core.process_utils import run_command_async, terminate_process_tree
from load_cfg import DEMO_MODE

st.set_page_config(page_title="Data Management", layout="wide")
st.title("⚙️ Data Management")

if DEMO_MODE:
    st.warning(
        "**Demo Mode Active:** All operations that modify data or run long processes are disabled. "
        "To enable these features, set `DEMO_MODE = False` in `load_cfg.py`.",
        icon="🔒"
    )

st.markdown("""
This page provides a user interface for managing the application's core data. 
You can run the full daily pipeline, perform ad-hoc data downloads, or run specific calculations and scanners.
""")

tab1, tab2, tab3, tab4 = st.tabs(["Daily Pipeline", "Full Download/Update", "Range Download", "Ad-hoc Tasks"])

with tab1:
    st.header("Run Daily Pipeline")
    st.markdown("""
    Runs the complete daily process:
    1.  Updates prices for CA and US markets for the last day.
    2.  Recalculates common scanner values (e.g., ATR, volume averages).
    """)
    st.info("**Note:** If you miss a daily run, data gaps will occur. Use the **Range Download** tab to manually download data for the missed dates.", icon="ℹ️")

    # Initialize session state for the pipeline tab
    if 'pipeline_process' not in st.session_state:
        st.session_state.pipeline_process = None
    if 'pipeline_finished_message' not in st.session_state:
        st.session_state.pipeline_finished_message = None
    if 'pipeline_logs' not in st.session_state:
        st.session_state.pipeline_logs = []

    is_running_pipeline = st.session_state.pipeline_process is not None

    if st.session_state.pipeline_finished_message:
        st.success(st.session_state.pipeline_finished_message)

    if st.button("Start Daily Pipeline", width='stretch', disabled=is_running_pipeline or DEMO_MODE):
        st.session_state.pipeline_finished_message = None # Clear previous message
        st.session_state.pipeline_logs = [f"[{datetime.now().strftime('%H:%M:%S')}] Starting daily pipeline process..."]

        cmd = [sys.executable, "download_data.py", "pipeline"]
        process, log_queue = run_command_async(cmd)

        st.session_state.pipeline_process = process
        st.session_state.pipeline_queue = log_queue
        st.rerun()

    if st.session_state.pipeline_process:
        if st.button("Stop Pipeline", key="stop_pipeline"):
            terminate_process_tree(st.session_state.pipeline_process)
            st.session_state.pipeline_process = None
            st.session_state.pipeline_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] --- PROCESS STOPPED BY USER ---")
            st.warning("Pipeline process stopped.")
            st.rerun()

        log_container = st.expander("Logs", expanded=True)
        log_area = log_container.empty()

        queue = st.session_state.get('pipeline_queue')
        if queue:
            while not queue.empty():
                message = queue.get()
                st.session_state.pipeline_logs.append(message)

        log_area.code("\n".join(st.session_state.pipeline_logs[-200:]))

        if st.session_state.pipeline_process and st.session_state.pipeline_process.poll() is None:
            time.sleep(1)
            st.rerun()
        elif st.session_state.pipeline_process is not None: # Process just finished
            st.session_state.pipeline_finished_message = "Daily pipeline finished successfully."
            st.session_state.pipeline_process = None
            st.rerun()

with tab2:
    st.header("Full Data Download / Update")
    st.markdown("Use this for initial data population or large-scale updates. This can be very time-consuming.")

    # Initialize session state for this tab
    if 'full_download_process' not in st.session_state:
        st.session_state.full_download_process = None
    if 'full_download_finished_message' not in st.session_state:
        st.session_state.full_download_finished_message = None
    if 'full_download_logs' not in st.session_state:
        st.session_state.full_download_logs = []

    is_running = st.session_state.full_download_process is not None

    if st.session_state.full_download_finished_message:
        st.success(st.session_state.full_download_finished_message)

    with st.form("full_download_form"):
        c1, c2, c3 = st.columns(3)
        market = c1.selectbox("Market", ["us", "ca"], disabled=is_running)
        batch_size = c2.number_input("Batch Size", min_value=10, max_value=1000, value=50, disabled=is_running)
        exchange = c3.text_input("Exchange (optional)", placeholder="e.g., NMS, NCM, NGS", disabled=is_running)

        c1, c2 = st.columns(2)
        existing_tickers_action = c1.selectbox("Action for Existing Tickers", ["skip", "only", "include"], help="`skip`: skip existing tickers. `only`: only process existing tickers. `include`: process all tickers.", disabled=is_running)
        update_prices_action = c2.selectbox("Update Prices Action", ["yes", "no", "last_day", "only"], help="`yes`: update both company info and prices. `no`: skip price updates. `last_day`: update only the last day's prices. `only`: only update prices, skip company info.", disabled=is_running)

        run_download = st.form_submit_button("Run Download/Update", width='stretch', disabled=is_running or DEMO_MODE)

    if run_download and not st.session_state.full_download_process:
        st.session_state.full_download_finished_message = None
        st.session_state.full_download_logs = [f"[{datetime.now().strftime('%H:%M:%S')}] Starting full download process..."]
        
        cmd = [
            sys.executable, "download_data.py", "download",
            "--market", market,
            "--batch_size", str(batch_size),
            "--existing_tickers_action", existing_tickers_action,
            "--update_prices_action", update_prices_action,
            "--start_date", "2000-01-01"
        ]
        if exchange:
            cmd.extend(["--exchange", exchange])

        process, log_queue = run_command_async(cmd)
        st.session_state.full_download_process = process
        st.session_state.full_download_queue = log_queue
        st.rerun()

    if st.session_state.full_download_process:
        if st.button("Stop Download", key="stop_full_download"):
            terminate_process_tree(st.session_state.full_download_process)
            st.session_state.full_download_process = None
            st.session_state.full_download_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] --- PROCESS STOPPED BY USER ---")
            st.warning("Download process stopped.")
            st.rerun()

        log_container = st.expander("Logs", expanded=True)
        log_area = log_container.empty()

        queue = st.session_state.get('full_download_queue')
        if queue:
            while not queue.empty():
                message = queue.get()
                st.session_state.full_download_logs.append(message)

        log_area.code("\n".join(st.session_state.full_download_logs[-200:]))

        if st.session_state.full_download_process and st.session_state.full_download_process.poll() is None:
            time.sleep(1)
            st.rerun()
        else:
            if st.session_state.full_download_process is not None:
                st.session_state.full_download_finished_message = "Download process finished."
            st.session_state.full_download_process = None
            st.rerun()

with tab3:
    st.header("Download Price Data for a Date Range")
    st.markdown("Use this to fill in data for specific periods, for example, if the daily pipeline was missed for a few days. This will only update prices for tickers already in the database.")

    if 'range_download_process' not in st.session_state:
        st.session_state.range_download_process = None
    if 'range_download_logs' not in st.session_state:
        st.session_state.range_download_logs = []
    if 'range_download_finished_message' not in st.session_state:
        st.session_state.range_download_finished_message = None
    if 'range_start_date' not in st.session_state:
        st.session_state.range_start_date = datetime.today()

    is_running_range = st.session_state.range_download_process is not None

    if st.session_state.range_download_finished_message:
        st.success(st.session_state.range_download_finished_message)

    with st.form("range_download_form"):
        c1, c2 = st.columns(2)
        range_market = c1.selectbox("Market", ["us", "ca"], key="range_market", disabled=is_running_range)
        if st.form_submit_button("Find Smart Start Date", disabled=is_running_range or DEMO_MODE):
            with st.spinner(f"Finding oldest last-updated date for '{range_market.upper()}' market..."):
                try:
                    cmd = [sys.executable, "download_data.py", "find-start-date", "--market", range_market]
                    # The script might output logging info first. The date is the last line.
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=90)
                    # Split the output by newlines and take the last non-empty line.
                    date_str = result.stdout.strip().split('\n')[-1]

                    if date_str and date_str != 'None':
                        found_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                        st.session_state.range_start_date = found_date
                        st.success(f"Suggested start date: {found_date.strftime('%Y-%m-%d')}")
                    else:
                        st.warning("Could not determine a start date. No price data found.")
                except Exception as e:
                    st.error(f"Failed to find start date: {e}")
                    st.error(f"Output: {e.stderr if hasattr(e, 'stderr') else 'N/A'}")

        update_prices_action_range = c2.selectbox(
            "Update Action",
            ["only", "yes", "no"],
            index=0,
            help="`only`: Update prices only (fastest for filling gaps). `yes`: Update company info and prices. `no`: Skip price updates.",
            disabled=is_running_range
        )

        today = datetime.today()
        c1, c2 = st.columns(2)
        start_date = c1.date_input("Start Date", value=st.session_state.range_start_date, disabled=is_running_range)
        end_date = c2.date_input("End Date", value=today, disabled=is_running_range)

        run_range_download_button = st.form_submit_button("Run Range Download", width='stretch', disabled=is_running_range or DEMO_MODE)

    if run_range_download_button and not st.session_state.range_download_process:
        st.session_state.range_download_finished_message = None
        if start_date > end_date:
            st.error("Error: Start date cannot be after end date.")
        else:
            # Dynamically set batch size based on the update action for efficiency.
            if update_prices_action_range == 'yes':
                batch_size = 50
                batch_size_info = f"Using smaller batch size ({batch_size}) for company info update."
            else:  # 'only' or 'no' are much faster and can use a larger batch.
                batch_size = 1000
                batch_size_info = f"Using larger batch size ({batch_size}) for price-only update."

            st.session_state.range_download_logs = [
                f"[{datetime.now().strftime('%H:%M:%S')}] Starting range download process...",
                f"[{datetime.now().strftime('%H:%M:%S')}] {batch_size_info}"
            ]
            cmd = [
                sys.executable, "download_data.py", "range_download",
                "--market", range_market,
                "--batch_size", str(batch_size),
                "--start_date", start_date.strftime('%Y-%m-%d'),
                "--end_date", end_date.strftime('%Y-%m-%d'),
                "--existing_tickers_action", "only",
                "--update_prices_action", update_prices_action_range
            ]
            process, log_queue = run_command_async(cmd)
            st.session_state.range_download_process = process
            st.session_state.range_download_queue = log_queue
            st.rerun()

    if st.session_state.range_download_process:
        if st.button("Stop Download", key="stop_range_download"):
            terminate_process_tree(st.session_state.range_download_process)
            st.session_state.range_download_process = None
            st.session_state.range_download_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] --- PROCESS STOPPED BY USER ---")
            st.warning("Range download process stopped.")
            st.rerun()

        log_container = st.expander("Logs", expanded=True)
        log_area = log_container.empty()

        queue = st.session_state.get('range_download_queue')
        if queue:
            while not queue.empty():
                message = queue.get()
                st.session_state.range_download_logs.append(message)

        log_area.code("\n".join(st.session_state.range_download_logs[-200:]))

        if st.session_state.range_download_process and st.session_state.range_download_process.poll() is None:
            time.sleep(1)
            st.rerun()
        else:
            if st.session_state.range_download_process is not None:
                st.session_state.range_download_finished_message = "Range download process finished."
            st.session_state.range_download_process = None
            st.rerun()

with tab4:
    st.header("Ad-hoc Tasks")

    # --- Unified State Management for Ad-hoc Tasks ---
    if 'adhoc_task_name' not in st.session_state:
        st.session_state.adhoc_task_name = None # Can be 'calc', 'scanner', or None
    if 'adhoc_process' not in st.session_state:
        st.session_state.adhoc_process = None
    if 'adhoc_logs' not in st.session_state:
        st.session_state.adhoc_logs = []
    if 'adhoc_queue' not in st.session_state:
        st.session_state.adhoc_queue = None
    if 'adhoc_finished_message' not in st.session_state:
        st.session_state.adhoc_finished_message = None
    if 'scanner_results' not in st.session_state:
        st.session_state.scanner_results = None

    is_any_running = st.session_state.adhoc_task_name is not None

    # --- Render Forms (these are always visible but disabled when a task is running) ---
    with st.form("calculation_form"):
        st.subheader("Recalculate Common Values")
        c1, c2 = st.columns([1, 2])
        calc_market = c1.selectbox("Market", ["us", "ca"], key="calc_market", disabled=is_any_running)
        calc_ticker = c2.text_input("Specific Ticker (optional)", placeholder="e.g., SPY. Leave blank for whole market.", disabled=is_any_running)
        run_calc = st.form_submit_button("Run Calculation", disabled=is_any_running or DEMO_MODE)

    with st.form("scanner_form"):
        st.subheader("Run a Specific Scanner")
        c1, c2, c3 = st.columns(3)
        scanner_name = c1.selectbox("Scanner Name", ["strongest_industries"], disabled=is_any_running)
        scan_market = c2.selectbox("Market", ["us", "ca"], key="scan_market", disabled=is_any_running)
        min_avg_volume = c3.number_input("Min Avg Volume", value=50000, disabled=is_any_running)
        run_scan_task = st.form_submit_button("Run Scanner", disabled=is_any_running or DEMO_MODE)

    with st.form("fix_splits_form"):
        st.subheader("Fix Historical Split Data")
        st.markdown("Scans the database for tickers with recent splits and refreshes their entire price history to ensure data consistency. This is a one-time fix for past data issues.")
        c1, c2 = st.columns(2)
        fix_splits_market = c1.selectbox("Market", ["all", "us", "ca"], key="fix_splits_market", disabled=is_any_running)
        fix_splits_batch_size = c2.number_input("Batch Size", min_value=10, max_value=200, value=50, disabled=is_any_running)
        run_fix_splits_task = st.form_submit_button("Run Split Fix", disabled=is_any_running or DEMO_MODE)

    # --- Handle Actions (start processes if buttons are clicked) ---
    if run_calc:
        st.session_state.adhoc_task_name = 'calc'
        st.session_state.adhoc_finished_message = None
        st.session_state.scanner_results = None # Clear previous results
        st.session_state.adhoc_logs = [f"[{datetime.now().strftime('%H:%M:%S')}] Starting calculation process..."]
        cmd = [sys.executable, "download_data.py", "calculate", "--market", calc_market]
        if calc_ticker:
            cmd.extend(["--tickers", calc_ticker.upper()])
        process, log_queue = run_command_async(cmd)
        st.session_state.adhoc_process = process
        st.session_state.adhoc_queue = log_queue
        st.rerun()

    if run_scan_task:
        st.session_state.adhoc_task_name = 'scanner'
        st.session_state.adhoc_finished_message = None
        st.session_state.scanner_results = None # Clear previous results
        st.session_state.adhoc_logs = [f"[{datetime.now().strftime('%H:%M:%S')}] Starting scanner process..."]
        cmd = [
            sys.executable, "download_data.py", "scan",
            "--scanner_name", scanner_name, "--market", scan_market,
            "--min_avg_volume", str(min_avg_volume)
        ]
        process, log_queue = run_command_async(cmd)
        st.session_state.adhoc_process = process
        st.session_state.adhoc_queue = log_queue
        st.rerun()

    if run_fix_splits_task:
        st.session_state.adhoc_task_name = 'fix_splits'
        st.session_state.adhoc_finished_message = None
        st.session_state.scanner_results = None # Clear previous results
        st.session_state.adhoc_logs = [f"[{datetime.now().strftime('%H:%M:%S')}] Starting historical split fix process..."]
        cmd = [sys.executable, "download_data.py", "fix-splits", "--batch_size", str(fix_splits_batch_size)]
        if fix_splits_market != "all":
            cmd.extend(["--market", fix_splits_market])
        
        process, log_queue = run_command_async(cmd)
        st.session_state.adhoc_process = process
        st.session_state.adhoc_queue = log_queue
        st.rerun()

    # Create a container to hold the results. This allows us to explicitly
    # clear the results area by leaving the container empty when a new task is running.
    results_container = st.empty()

    # --- Display UI based on State ---
    if st.session_state.adhoc_task_name:
        task_name = st.session_state.adhoc_task_name.capitalize()
        if st.button(f"Stop {task_name}", key=f"stop_{st.session_state.adhoc_task_name}"):
            terminate_process_tree(st.session_state.adhoc_process)
            st.session_state.adhoc_process = None
            st.session_state.adhoc_task_name = None
            st.session_state.adhoc_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] --- PROCESS STOPPED BY USER ---")
            st.warning("Calculation process stopped.")
            st.rerun()

        log_container = st.expander(f"{task_name} Logs", expanded=True)
        log_area = log_container.empty()
        queue = st.session_state.get('adhoc_queue')
        if queue:
            while not queue.empty():
                raw_message = queue.get()
                # Special handling for scanner results
                if st.session_state.adhoc_task_name == 'scanner':
                    log_line = raw_message.split("] ", 1)[-1]
                    if log_line.startswith("SCANNER_RESULT_JSON:"):
                        json_str = log_line.replace("SCANNER_RESULT_JSON:", "")
                        try:
                            st.session_state.scanner_results = json.loads(json_str)
                        except json.JSONDecodeError:
                            st.session_state.adhoc_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Error: Could not parse scanner results.")
                        continue
                st.session_state.adhoc_logs.append(raw_message)
        log_area.code("\n".join(st.session_state.adhoc_logs[-200:]))

        if st.session_state.adhoc_process and st.session_state.adhoc_process.poll() is None:
            time.sleep(1)
            st.rerun()
        else: # Process finished
            st.session_state.adhoc_finished_message = f"{task_name} finished."
            st.session_state.adhoc_process = None
            st.session_state.adhoc_task_name = None
            st.rerun()

    else:
        with results_container.container():
            # State 3: Idle. Show finished messages and results from the last run.
            if st.session_state.adhoc_finished_message:
                st.success(st.session_state.adhoc_finished_message)

            if st.session_state.scanner_results is not None:
                results = st.session_state.scanner_results
                st.subheader("Scanner Results")
                if not results:
                    st.info("No matching stocks or industries found for the selected criteria.")
                else:
                    try:
                        is_nested = (isinstance(results, list) and isinstance(results[0], dict) and 'strongest_stocks' in results[0])
                        if is_nested:
                            st.success(f"Found {len(results)} passing industries.")
                            for item in results:
                                st.subheader(f"Industry: {item.get('industry', 'N/A')}")
                                st.metric("Average RS Ratio", f"{item.get('rsratio_mean', 0):.2f}")
                                strong_stocks = item.get('strongest_stocks', [])
                                if strong_stocks:
                                    df = pd.DataFrame(strong_stocks)
                                    st.dataframe(df, width='stretch', hide_index=True)
                        else:
                            st.success(f"Found {len(results)} passing stocks.")
                            st.dataframe(pd.DataFrame(results), width='stretch', hide_index=True)
                    except (IndexError, KeyError, TypeError):
                            st.warning("Scanner ran but returned results in an unexpected format.")
                            st.write(results)