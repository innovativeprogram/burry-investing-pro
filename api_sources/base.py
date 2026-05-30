# api_sources/base.py
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

import pandas as pd

logger = logging.getLogger("VQuantPro")

class BaseAPISource(ABC):
    """Classe base per le fonti API (FMP, Alpha Vantage, etc.)"""
    
    def __init__(self, name: str, api_key: Optional[str] = None):
        self.name = name
        self.api_key = api_key
    
    @abstractmethod
    def get_fundamentals(self, symbol: str) -> Optional[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def get_technical(self, symbol: str) -> Optional[pd.DataFrame]:
        pass
    
    def _request(self, url: str, params: Optional[Dict] = None):
        import requests
        from rate_limiter import get_rate_limiter
        limiter = get_rate_limiter()
        limiter.wait_and_consume(self.name)
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"{self.name} request failed for {url}: {e}")
            return None