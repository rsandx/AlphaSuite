from abc import ABC, abstractmethod
import pandas as pd
from typing import Union, Dict, Any

class BaseDataProvider(ABC):
    """
    Abstract Base Class for all financial data providers.
    """

    @abstractmethod
    def normalize_to_canonical(self, provider_symbol: str) -> str:
        pass

    @abstractmethod
    def normalize_from_canonical(self, canonical_symbol: str) -> str:
        pass

    @abstractmethod
    def get_ticker_data(self, ticker: str, start_date: str = None, end_date: str = None, timeframe: str = '1d') -> Dict[str, Any]:
        """
        The Master Method: Fetches the full standardized payload.
        This is what DataManager.load_ticker_data calls.
        """
        pass

    @abstractmethod
    def get_historical_data(self, ticker: str, start_date: str = None, end_date: str = None, interval: str = '1d') -> pd.DataFrame:
        pass

    @abstractmethod
    def get_company_info(self, ticker: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_financial_statements(self, ticker: str, statement_type: str, frequency: str) -> pd.DataFrame:
        pass
        