# api_sources/fmp.py
import logging
from typing import Optional, Dict, Any

import pandas as pd

from api_sources.base import BaseAPISource
from burry_ai_prompts import safe_get_secret

logger = logging.getLogger("VQuantPro")

class FMPSource(BaseAPISource):
    """
    Financial Modeling Prep API (https://site.financialmodelingprep.com/)
    Richiede chiave API gratuita (250 richieste/giorno).
    """
    
    BASE_URL = "https://financialmodelingprep.com/api/v3"
    
    def __init__(self):
        api_key = safe_get_secret("FMP_API_KEY")
        super().__init__("fmp", api_key)
    
    def get_fundamentals(self, symbol: str) -> Optional[Dict[str, Any]]:
        if not self.api_key:
            logger.warning("FMP API key mancante")
            return None
        
        symbol = symbol.upper()
        
        # 1. Profilo azienda
        profile_url = f"{self.BASE_URL}/profile/{symbol}"
        profile_data = self._request(profile_url, {'apikey': self.api_key})
        if not profile_data or not isinstance(profile_data, list) or len(profile_data) == 0:
            return None
        profile = profile_data[0]
        
        # 2. Ratio (TTM)
        ratios_url = f"{self.BASE_URL}/ratios-ttm/{symbol}"
        ratios_data = self._request(ratios_url, {'apikey': self.api_key})
        if ratios_data and isinstance(ratios_data, list) and len(ratios_data) > 0:
            ratios = ratios_data[0]
        else:
            ratios = {}
        
        # 3. DCF (per stime)
        dcf_url = f"{self.BASE_URL}/discounted-cash-flow/{symbol}"
        dcf_data = self._request(dcf_url, {'apikey': self.api_key})
        if dcf_data and isinstance(dcf_data, list) and len(dcf_data) > 0:
            dcf = dcf_data[0]
        else:
            dcf = {}
        
        # Costruisci info
        info = {
            'symbol': symbol,
            'shortName': profile.get('companyName', symbol),
            'longName': profile.get('companyName', symbol),
            'regularMarketPrice': profile.get('price'),
            'currency': profile.get('currency', 'USD'),
            'marketCap': profile.get('mktCap'),
            'trailingPE': profile.get('pe'),
            'pegRatio': ratios.get('pegRatio'),
            'priceToBook': ratios.get('priceToBookRatio'),
            'priceToSalesTrailing12Months': ratios.get('priceToSalesRatio'),
            'revenueGrowth': ratios.get('revenueGrowthTTM'),
            'netMargin': ratios.get('netProfitMarginTTM'),
            'roic': ratios.get('roicTTM'),
            'debtToEquity': ratios.get('debtToEquityTTM'),
            'currentRatio': ratios.get('currentRatioTTM'),
            'quickRatio': ratios.get('quickRatioTTM'),
            'roe': ratios.get('roeTTM'),
            'beta': profile.get('beta'),
            'dcf': dcf.get('dcf'),
        }
        
        # Financials (income, balance, cashflow) – per semplicità non scarichiamo tutti i dati storici
        # perché FMP ha limiti; li lasceremo vuoti, useremo le metriche sopra
        return {
            "info": info,
            "financials": pd.DataFrame(),
            "balance_sheet": pd.DataFrame(),
            "cashflow": pd.DataFrame(),
            "symbol": symbol,
            "source": "FMP"
        }
    
    def get_technical(self, symbol: str) -> Optional[pd.DataFrame]:
        """FMP ha anche dati storici, ma per non complicare, restituiamo None."""
        # Opzionale: implementare /historical-price-full
        return None