import os

from core.data_providers.eodhd_provider import EODHDProvider
from load_cfg import EODHD_API_KEY
from .yfinance_provider import YFinanceProvider
from .base import BaseDataProvider

class DataProviderFactory:
    @staticmethod
    def get_provider() -> BaseDataProvider:
        """
        Returns the appropriate provider based on the DATA_SOURCE environment variable.
        Defaults to YFinance.
        """
        source = os.getenv("DATA_SOURCE", "YFINANCE").upper()
        
        if source == "YFINANCE":
            return YFinanceProvider()
        elif source == "EODHD":
            # In production, fetch API key from environment variables
            return EODHDProvider(api_key=EODHD_API_KEY)
        else:
            raise ValueError(f"Unknown data source: {source}")