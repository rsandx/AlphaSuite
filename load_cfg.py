# load_cfg.py
# This file contains configuration settings for the AlphaSuite application.

import os
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

WORKING_DIRECTORY = os.getenv('WORKING_DIRECTORY', './work/')
DATABASE_URL = os.getenv('DATABASE_URL')

# --- Model Configuration ---
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'gemini').lower()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-3-flash-preview')
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3') # IMPORTANT: Make sure you have pulled this model in Ollama
EODHD_API_KEY = os.getenv('EODHD_API_KEY')

def get_llm():
    """
    Initializes and returns the appropriate LangChain LLM instance based on the
    configuration in .env.

    This function acts as a factory for LLM providers.

    Supported `LLM_PROVIDER` values:
    - 'gemini': Uses Google's Generative AI. Requires `GEMINI_API_KEY`.
    - 'ollama': Uses a local Ollama instance. Requires `OLLAMA_URL` and `OLLAMA_MODEL`.

    Returns:
        An instance of a LangChain compatible LLM.

    Raises:
        ValueError: If the configured `LLM_PROVIDER` is not supported or if
                    required environment variables for the provider are missing.
    """
    if LLM_PROVIDER == 'gemini':
        from langchain_google_genai import GoogleGenerativeAI
        import google.generativeai as genai

        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set in the environment for the 'gemini' provider.")
        genai.configure(api_key=GEMINI_API_KEY)
        return GoogleGenerativeAI(model=GEMINI_MODEL)
    elif LLM_PROVIDER == 'ollama':
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_URL)

    raise ValueError(f"Unsupported LLM_PROVIDER: '{LLM_PROVIDER}'. Please use 'gemini' or 'ollama'.")

# --- Demo Mode ---
# If set to True, the Streamlit application runs in "demo" or "read-only" mode.
# This disables UI elements that trigger long-running processes, database updates,
# or model training. Set the environment variable DEMO_MODE=True to enable it.
# Defaults to False for a fully functional, live application.
DEMO_MODE = os.getenv('DEMO_MODE', 'False').lower() in ('true', '1', 't')
