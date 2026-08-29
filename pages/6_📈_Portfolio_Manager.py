"""Streamlit page for managing open trading positions."""

import streamlit as st
import pandas as pd
import os
from datetime import datetime
import traceback
import io
import contextlib
import yfinance as yf
from typing import List, Optional

from load_cfg import DEMO_MODE, WORKING_DIRECTORY
from pybroker_trainer.portfolio.manager import TradeManager

# --- Page Configuration ---
st.set_page_config(page_title="Portfolio Manager", layout="wide")

st.title("📈 Portfolio Manager")

if DEMO_MODE:
    st.warning(
        "**Demo Mode Active:** All portfolio management actions (add, close, manage) are disabled. "
        "To enable these features, set `DEMO_MODE = False` in `load_cfg.py`.",
        icon="🔒"
    )

st.markdown("""
This page allows you to view and manage your open trading positions.
Positions are stored in `open_positions.json`.
""")

# --- Constants and Initialization ---
POSITIONS_FILE = os.path.join(WORKING_DIRECTORY, "open_positions.json")

manager = TradeManager(POSITIONS_FILE)

# --- Helper Functions ---
@st.cache_data(ttl=60)
def get_latest_price(ticker: str) -> Optional[float]:
    """
    Fetches the latest available price for a given ticker.
    Tries to get pre-market/post-market price if available, otherwise regular market price.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        if 'currentPrice' in info: return info['currentPrice']
        if 'preMarketPrice' in info and info['preMarketPrice'] is not None: return info['preMarketPrice']
        if 'postMarketPrice' in info and info['postMarketPrice'] is not None: return info['postMarketPrice']
        if 'regularMarketPrice' in info: return info['regularMarketPrice']
        if 'previousClose' in info: return info['previousClose']
        
        hist = stock.history(period="1d")
        if not hist.empty: return hist['Close'].iloc[-1]
        return None
    except Exception:
        return None

@st.cache_data(ttl=60)
def load_and_enrich_positions():
    """Loads positions and enriches them with live market data."""
    positions = manager.load_positions()
    if not positions:
        return pd.DataFrame()

    enriched_data = []
    for ticker, pos in positions.items():
        pos_dict = pos.__dict__
        latest_price = get_latest_price(ticker)
        pos_dict['latest_price'] = latest_price
        if latest_price and pos.entry_price > 0:
            pos_dict['unrealized_pnl'] = (latest_price - pos.entry_price) * pos.shares
            pos_dict['unrealized_pnl_pct'] = ((latest_price / pos.entry_price) - 1) * 100
        else:
            pos_dict['unrealized_pnl'] = 0
            pos_dict['unrealized_pnl_pct'] = 0
        enriched_data.append(pos_dict)
    
    if not enriched_data:
        return pd.DataFrame()
        
    return pd.DataFrame(enriched_data)

# --- UI: Main Display ---
st.subheader("Open Positions")

if st.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

positions_df = load_and_enrich_positions()

if positions_df.empty:
    st.info("No open positions found.")
else:
    # Define desired columns and their configurations
    COLUMN_ORDER: List[str] = [
        'ticker', 'strategy', 'entry_date', 'shares', 'entry_price', 
        'latest_price', 'unrealized_pnl', 'unrealized_pnl_pct', 'current_stop_loss'
    ]
    
    # Filter dataframe to only include columns that exist, preventing errors
    display_df = positions_df[[col for col in COLUMN_ORDER if col in positions_df.columns]]

    st.dataframe(
        display_df,
        column_config={
            "ticker": "Ticker",
            "strategy": "Strategy",
            "entry_date": st.column_config.DateColumn("Entry Date", format="YYYY-MM-DD"),
            "shares": st.column_config.NumberColumn("Shares", format="%.4f"),
            "entry_price": st.column_config.NumberColumn("Entry Price", format="$%.2f"),
            "latest_price": st.column_config.NumberColumn("Latest Price", format="$%.2f"),
            "unrealized_pnl": st.column_config.NumberColumn("Unrealized P/L", format="$%.2f"),
            "unrealized_pnl_pct": st.column_config.ProgressColumn(
                "Unrealized P/L %", format="%.2f%%", min_value=-50, max_value=50,
            ),
            "current_stop_loss": st.column_config.NumberColumn("Current Stop", format="$%.2f"),
        },
        width='stretch', hide_index=True
    )

st.markdown("---")

# --- UI: Actions ---
st.subheader("Manage Positions")

with st.expander("➕ Add New Position"):
    with st.form("add_position_form", clear_on_submit=True):
        ticker = st.text_input("Ticker").upper()
        entry_price = st.number_input("Entry Price", min_value=0.0, format="%.4f")
        shares = st.number_input("Number of Shares", min_value=0.0, format="%.4f")
        strategy = st.text_input("Strategy Name")
        entry_date = st.date_input("Entry Date", value=datetime.now())
        
        submitted = st.form_submit_button("Add Position", disabled=DEMO_MODE)
        if submitted:
            if not all([ticker, entry_price > 0, shares > 0, strategy]):
                st.error("Please fill in all fields.")
            else:
                try:
                    manager.add_position(
                        ticker=ticker, entry_price=entry_price,
                        entry_date=entry_date.strftime('%Y-%m-%d'),
                        shares=shares, strategy=strategy
                    )
                    st.success(f"Successfully added position for {ticker}.")
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to add position: {e}")
                    st.code(traceback.format_exc())

if not positions_df.empty and not DEMO_MODE:
    with st.expander("➖ Close Position", expanded=True):
        with st.form("close_position_form", clear_on_submit=True):
            position_to_close = st.selectbox(
                "Select Ticker to Close",
                options=positions_df['ticker'].unique()
            )
            exit_price = st.number_input("Exit Price", min_value=0.0, format="%.4f")
            
            close_submitted = st.form_submit_button("Close Position")
            if close_submitted:
                if not all([position_to_close, exit_price > 0]):
                    st.error("Please select a ticker and provide a valid exit price.")
                else:
                    try:
                        manager.close_position(position_to_close, exit_price)
                        st.success(f"Successfully closed position for {position_to_close}.")
                        st.cache_data.clear()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to close position: {e}")

if st.button("⚙️ Check & Manage All Positions", width='stretch', disabled=DEMO_MODE):
    with st.spinner("Checking positions..."):
        try:
            string_io = io.StringIO()
            with contextlib.redirect_stdout(string_io):
                manager.check_positions()
            
            management_log = string_io.getvalue()
            
            st.success("Position check complete.")
            st.code(management_log, language='log')
            st.cache_data.clear()
            st.rerun()
        except Exception as e:
            st.error(f"An error occurred during position management: {e}")
            st.code(traceback.format_exc())