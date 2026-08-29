import base64
import os
import streamlit as st

from load_cfg import LLM_PROVIDER

st.set_page_config(page_title="News Intelligence", layout="wide")
st.title("📰 News Intelligence & Risk Monitor")

st.markdown(f"""
This tool leverages an LLM to act as a real-time risk analyst. It scans recent news from financial, tech, and AI sources, then analyzes it against specific economic risk profiles to identify potential "red flags" that could signal market corrections.

The currently configured LLM provider is: **{LLM_PROVIDER.upper()}**
""")

# --- Define Risk Profiles ---
RISK_PROFILES = {
    "Credit & Real Estate Crisis": {
        "name": "Credit & Real Estate Crisis",
        "prompt_details": """
        Focus on signs of stress in credit markets and real estate. Look for:
        - News about rising corporate or consumer loan defaults.
        - Failures or significant distress in regional banks, especially related to Commercial Real Estate (CRE) loans.
        - Reports of falling commercial property values or rising office vacancies.
        - Warnings from credit rating agencies about specific sectors or companies.
        - A sudden freeze in the high-yield ("junk") bond market.
        """
    },
    "Inflation & Fed Policy Shock": {
        "name": "Inflation & Fed Policy Shock",
        "prompt_details": """
        Focus on news that contradicts the "inflation is cooling" narrative. Look for:
        - Unexpectedly high inflation reports (CPI, PPI).
        - Spikes in key commodity prices (e.g., oil, copper).
        - Reports of stubbornly high wage growth.
        - Statements from Federal Reserve officials that are more hawkish than expected, suggesting rate cuts are off the table or more hikes are possible.
        - Disruptions in global supply chains that could lead to price increases.
        """
    },
    "Geopolitical & Supply Chain Disruption": {
        "name": "Geopolitical & Supply Chain Disruption",
        "prompt_details": """
        Focus on geopolitical events that could disrupt global trade and supply chains. Look for:
        - Escalations in major conflicts (e.g., involving China/Taiwan, Russia/Ukraine, Middle East).
        - New, significant tariffs or trade restrictions being imposed between major economies.
        - Physical disruptions to key shipping lanes or manufacturing hubs.
        - News suggesting a "decoupling" between Western economies and China is accelerating.
        """
    },
    "Tech Sector Health & Concentration Risk": {
        "name": "Tech Sector Health & Concentration Risk",
        "prompt_details": """
        Focus on news that could undermine the health and valuation of the mega-cap tech stocks (the "Magnificent Seven"). Look for:
        - Negative earnings revisions or guidance from one of the major tech leaders.
        - Signs that the AI revenue boom is slowing down or not meeting expectations.
        - Significant product failures or a lukewarm reception to a major new product launch.
        - Serious new antitrust lawsuits or regulatory actions that threaten a core business model (e.g., App Store, Search advertising).
        - A notable failure or downgrade of a key supplier (e.g., a major chip manufacturer).
        """
    }
}

tab1, tab2 = st.tabs(["Market Briefing", "Risk Monitor"])

with tab1:
    st.header("Generate a Detailed Market Briefing")
    st.markdown("This tool synthesizes the most impactful news into a coherent narrative, identifying key themes and providing a market outlook.")

    if 'market_briefing_results' not in st.session_state:
        st.session_state.market_briefing_results = None

    if st.button("Generate Latest Market Briefing", width='stretch', key="briefing_button"):
        with st.spinner("Fetching news and generating market briefing... This may take a while."):
            st.info("Displaying a sample Market Briefing Report. This feature is for demonstration purposes.")
            # Construct path to the sample report in the 'samples' directory at the project root
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            report_file = os.path.join(project_root, "samples", "market_briefing_20250820_161653.pdf")
            if not os.path.exists(report_file):
                st.error("Sample report file not found. Please ensure 'samples/market_briefing_20250820_161653.pdf' exists.")
                st.session_state.market_briefing_results = None
            else:
                st.session_state.market_briefing_results = {"pdf_path": report_file}

    # --- Display Briefing Results ---
    if st.session_state.market_briefing_results:
        results = st.session_state.market_briefing_results
        pdf_path = results.get("pdf_path", None)
        if pdf_path and os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()

            st.download_button(
                label="Download Report",
                data=pdf_bytes,
                file_name=os.path.basename(pdf_path),
                mime="application/pdf"
            )

            st.subheader("Generated Report Preview")
            try:
                base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Could not display PDF preview: {e}")

with tab2:
    st.header("Analyze News for Specific Risk Profiles")
    # --- UI ---
    selected_profile_names = st.multiselect(
        "Select Risk Profiles to Analyze",
        options=list(RISK_PROFILES.keys()),
        default=list(RISK_PROFILES.keys()) # Default to all selected
    )

    if 'risk_analysis_results' not in st.session_state:
        st.session_state.risk_analysis_results = None

    if st.button("Analyze News for Red Flags", width='stretch', key="risk_button"):
        if not selected_profile_names:
            st.warning("Please select at least one risk profile to analyze.")
            st.stop()

        # Get the full profile dictionaries for the selected names
        selected_profiles = [RISK_PROFILES[name] for name in selected_profile_names]
        
        # Generate a user-friendly string for the spinner
        profile_names_str = ", ".join(selected_profile_names)
        with st.spinner(f"Fetching recent news and analyzing for: {profile_names_str}..."):
            st.info("Displaying a sample Risk Analysis Report. This feature is for demonstration purposes.")
            # Construct path to the sample report in the 'samples' directory at the project root
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            report_file = os.path.join(project_root, "samples", "risk_analysis_20250820_162249.pdf")
            if not os.path.exists(report_file):
                st.error("Sample report file not found. Please ensure 'samples/risk_analysis_20250820_162249.pdf' exists.")
                st.session_state.risk_analysis_results = None
            else:
                st.session_state.risk_analysis_results = {"pdf_path": report_file}

    # --- Display Results ---
    if st.session_state.risk_analysis_results:
        all_results = st.session_state.risk_analysis_results
        pdf_path = all_results.get("pdf_path", None)
        if pdf_path and os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()

            st.download_button(
                label="Download Report",
                data=pdf_bytes,
                file_name=os.path.basename(pdf_path),
                mime="application/pdf"
            )

            st.subheader("Generated Report Preview")
            try:
                base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Could not display PDF preview: {e}")

