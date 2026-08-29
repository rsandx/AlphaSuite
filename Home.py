import streamlit as st
import os
import logging
from dotenv import load_dotenv
from core.db import get_db, initialize_database_schema
from core.logging_config import setup_logging
from core.model import Company
from tools.yfinance_tool import load_ticker_data

# Load environment variables from .env file at the start of the application.
# This ensures that all modules have access to them when they are imported.
load_dotenv()

# --- Logging Configuration ---
# Centralize logging setup by calling the function from the core module.
setup_logging('home.log')
logger = logging.getLogger(__name__)  # Get the logger for Home.py

def initialize_app():
    """
    Performs initial application setup on the first run of a session.
    1. Initializes the database schema (creates tables if they don't exist).
    2. Ensures baseline data for 'SPY' is loaded if it doesn't exist.
    """
    # This is the primary entry point for the web app. We initialize the DB schema
    # here to ensure it's ready before any operations are attempted. This function
    # is idempotent and safe to run every time.
    initialize_database_schema()

    session = next(get_db())
    try:
        # Always check for SPY data to ensure the app has a baseline dataset.
        # This provides a good out-of-the-box experience for all users.
        ticker = 'SPY'
        spy_exists = session.query(Company).filter(Company.symbol == ticker).first()

        if not spy_exists:
            logger.info(f"APP_INIT: {ticker} data not found. Initializing baseline dataset...")
            with st.spinner(f"First-time setup: Downloading baseline data for {ticker}, please wait..."):
                load_ticker_data(ticker, refresh=True)
            
            logger.info(f"APP_INIT: Successfully downloaded data for {ticker}.")
            st.success(f"Baseline data for {ticker} has been loaded. The app is ready.")
            st.rerun()
    finally:
        session.close()

st.set_page_config(
    page_title="AlphaSuite - Home",
    layout="wide"
)

try:
    # --- Initialize App on first run ---
    # Use session state to ensure this block only runs once per session.
    if 'app_initialized' not in st.session_state:
        initialize_app()
        st.session_state['app_initialized'] = True
except Exception as e:
    # Catch common database connection errors and provide a helpful message.
    if "database" in str(e).lower() and "does not exist" in str(e).lower() or "connection refused" in str(e).lower():
        st.error(
            f"""
            **Database Connection Error**

            Could not connect to the database. Please ensure:
            1.  Your PostgreSQL server is running.
            2.  The database `alphasuite` has been created. You can create it with the command: `CREATE DATABASE alphasuite;`
            3.  Your `.env` file has the correct `DATABASE_URL`.

            **Error details:** {e}
            """
        )
        st.stop() # Stop the app from running further
    raise e # Reraise other exceptions

st.title("🏠 Welcome to AlphaSuite")

st.markdown("""
This application provides a suite of tools for financial analysis, model training, backtesting, and trade management.
The source code for this project is available on [GitHub](https://github.com/rsandx/AlphaSuite).

Use the sidebar to navigate between the different tools.

### Available Tools:

*   **Data Management:** Control the data pipeline, from downloading market data to running rule-based scanners.
*   **Market Scanner:** Scan the market for trading signals based on pre-trained models or run an interactive scan on-demand.
*   **Model Training & Tuning:** Fine-tune strategy parameters using Bayesian optimization and train your final models with walk-forward analysis, and visualize the tuned model's out-of-sample performance, trade executions, and feature importances.
*   **Portfolio Analysis:** Discover which stocks are suitable for a strategy and run portfolio-level backtests to validate your ideas.
*   **Interactive Backtester:** Visualize the in-sample performance of a saved model on historical data.
*   **Portfolio Manager:** Manually add, view, and manage your open trading positions.
*   **Stock Report:** Generate a comprehensive fundamental and technical analysis report or CANSLIM analysis for any stock.
*   **News Intelligence:** Scans recent news, generates a detailed market briefing, and analyzes it against economic risk profiles to identify potential market-moving "red flags." 

### Getting Started

1.  **Populate Data:** On the first run, baseline data for the **SPY** ticker is loaded automatically. To add more data, go to the **Data Management** page and run a "Full Download".
2.  **Scan for Signals:** Use the **Market Scanner** to find live trading signals based on pre-built rules or your own trained models.
3.  **Tune & Train:** To build custom models, navigate to the **Model Training & Tuning** page. First, "Tune Strategy" for a specific ticker to find optimal parameters, then "Train Model" to create and save the final model artifacts.
4.  **Analyze & Backtest:** Use the **Portfolio Analysis** page to discover which stocks are suitable for a strategy and run portfolio-level backtests to validate your ideas.
5.  **Deep Research:** Use the **Stock Report** page to generate a comprehensive report for a specific stock.
6.  **Explore:** Select any tool from the sidebar to continue your analysis.
""")