# scrapers/base.py
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

import pandas as pd

logger = logging.getLogger("VQuantPro")

class BaseScraper(ABC):
    """
    Classe astratta per tutti gli scraper.
    Ogni scraper deve implementare get_fundamentals e get_technical.
    """
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def get_fundamentals(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Recupera i dati fondamentali (info, financials, balance sheet, cashflow).
        Restituisce un dict con la stessa struttura di get_fundamental_data.
        """
        pass
    
    @abstractmethod
    def get_technical(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Recupera i dati tecnici (OHLCV giornalieri) come DataFrame.
        """
        pass
    
    def _make_request(self, url: str, headers: Optional[Dict] = None, timeout: int = 10):
        """Helper per fare request con logging e rate limiting (da implementare nei figli)."""
        # Le sottoclassi useranno il rate limiter globale
        import requests
        from rate_limiter import get_rate_limiter
        limiter = get_rate_limiter()
        limiter.wait_and_consume(self.name)
        try:
            resp = requests.get(url, headers=headers or {}, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as e:
            logger.warning(f"{self.name} request failed for {url}: {e}")
            return None