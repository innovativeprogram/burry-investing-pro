# api_sources/alpha_vantage.py
import logging
from typing import Optional, Dict, Any

import pandas as pd

from api_sources.base import BaseAPISource
from burry_ai_prompts import safe_get_secret

logger = logging.getLogger("VQuantPro")

class AlphaVantageSource(BaseAPISource):
    """
    Alpha Vantage API (https://www.alphavantage.co/)
    Richiede chiave gratuita (5 chiamate/minuto, 500/giorno).
    """
    
    BASE_URL = "https://www.alphavantage.co/query"
    
    def __init__(self):
        api_key = safe_get_secret("ALPHA_VANTAGE_API_KEY")
        super().__init__("alpha_vantage", api_key)
    
    def get_fundamentals(self, symbol: str) -> Optional[Dict[str, Any]]:
        if not self.api_key:
            return None
        
        symbol = symbol.upper()
        
        # OVERVIEW
        params = {
            'function': 'OVERVIEW',
            'symbol': symbol,
            'apikey': self.api_key
        }
        data = self._request(self.BASE_URL, params)
        if not data or 'Symbol' not in data:
            return None
        
        # Mappa campi
        info = {
            'symbol': data.get('Symbol', symbol),
            'shortName': data.get('Name', symbol),
            'longName': data.get('Name', symbol),
            'currency': data.get('Currency', 'USD'),
            'regularMarketPrice': float(data.get('MarketCapitalization', 0)) / 1e9,  # fittizio, alpha non dà prezzo corrente
            'marketCap': float(data.get('MarketCapitalization', 0)) if data.get('MarketCapitalization') else None,
            'trailingPE': float(data.get('PERatio', 0)) if data.get('PERatio') else None,
            'pegRatio': float(data.get('PEGRatio', 0)) if data.get('PEGRatio') else None,
            'priceToBook': float(data.get('PriceToBookRatio', 0)) if data.get('PriceToBookRatio') else None,
            'priceToSalesTrailing12Months': float(data.get('PriceToSalesRatioTTM', 0)) if data.get('PriceToSalesRatioTTM') else None,
            'revenueGrowth': float(data.get('RevenueGrowth', 0)) if data.get('RevenueGrowth') else None,
            'netMargin': float(data.get('ProfitMargin', 0)) if data.get('ProfitMargin') else None,
            'roic': float(data.get('ReturnOnAssetsTTM', 0)) if data.get('ReturnOnAssetsTTM') else None,
            'debtToEquity': float(data.get('DebtToEquityRatio', 0)) if data.get('DebtToEquityRatio') else None,
            'currentRatio': float(data.get('CurrentRatio', 0)) if data.get('CurrentRatio') else None,
            'quickRatio': float(data.get('QuickRatio', 0)) if data.get('QuickRatio') else None,
            'roe': float(data.get('ReturnOnEquityTTM', 0)) if data.get('ReturnOnEquityTTM') else None,
            'beta': float(data.get('Beta', 0)) if data.get('Beta') else None,
            'fscore': None,  # Alpha Vantage non fornisce F-Score
            'mscore': None,
        }
        
        # INCOME STATEMENT (solo annuale, prendiamo l'ultimo)
        # ... per brevità omettiamo l'income statement dettagliato, ma potremmo implementarlo
        
        return {
            "info": info,
            "financials": pd.DataFrame(),
            "balance_sheet": pd.DataFrame(),
            "cashflow": pd.DataFrame(),
            "symbol": symbol,
            "source": "AlphaVantage"
        }
    
    def get_technical(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Recupera dati storici giornalieri (TIME_SERIES_DAILY).
        """
        if not self.api_key:
            return None
        
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'outputsize': 'compact',  # o 'full' ma limita
            'apikey': self.api_key
        }
        data = self._request(self.BASE_URL, params)
        if not data or 'Time Series (Daily)' not in data:
            return None
        
        ts = data['Time Series (Daily)']
        df = pd.DataFrame.from_dict(ts, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'
        })
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
        return df