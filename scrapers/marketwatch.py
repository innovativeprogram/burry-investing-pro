# scrapers/marketwatch.py
import re
import logging
from typing import Optional, Dict, Any

import pandas as pd
from bs4 import BeautifulSoup

from scrapers.base import BaseScraper

logger = logging.getLogger("VQuantPro")

class MarketWatchScraper(BaseScraper):
    """
    Scraper per MarketWatch (https://www.marketwatch.com/investing/stock/...)
    Utile per dati fondamentali e forse tecnici.
    """
    
    def __init__(self):
        super().__init__("marketwatch")
    
    def get_fundamentals(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Recupera fondamentali da MarketWatch."""
        url = f"https://www.marketwatch.com/investing/stock/{symbol.lower()}"
        resp = self._make_request(url)
        if not resp:
            return None
        
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # Estrai nome azienda
        company_name = soup.find('h1', class_='company__name')
        company_name = company_name.text.strip() if company_name else symbol
        
        # Estrai prezzo
        price_elem = soup.find('meta', {'name': 'price'})
        price = float(price_elem['content']) if price_elem else None
        
        # Estrai metriche dalla tabella "Key Stats"
        key_stats = {}
        stat_table = soup.find('div', class_='kv__cell')
        if stat_table:
            rows = stat_table.find_all('li')
            for row in rows:
                label_elem = row.find('small')
                value_elem = row.find('span', class_='primary')
                if label_elem and value_elem:
                    label = label_elem.text.strip().lower().replace(' ', '_')
                    value = value_elem.text.strip()
                    # Converti in numero se possibile
                    try:
                        # rimuovi %, $, virgole
                        clean = re.sub(r'[^\d.-]', '', value)
                        if clean:
                            key_stats[label] = float(clean)
                    except:
                        key_stats[label] = value
        
        # Costruisci il dizionario nel formato atteso da V-Quant
        info = {
            'symbol': symbol,
            'shortName': company_name,
            'longName': company_name,
            'regularMarketPrice': price,
            'currency': 'USD',  # MarketWatch è principalmente US
            'quoteType': 'EQUITY',
        }
        
        # Aggiungi metriche trovate
        mapping = {
            'p/e_ratio': 'trailingPE',
            'market_cap': 'marketCap',
            'revenue_growth': 'revenueGrowth',
            'net_margin': 'netMargin',
            'debt_to_equity': 'debtToEquity',
            'roic': 'roic',
            'peg_ratio': 'pegRatio',
        }
        for mw_key, our_key in mapping.items():
            if mw_key in key_stats:
                info[our_key] = key_stats[mw_key]
        
        # Non abbiamo financials, balance sheet, cashflow strutturati, ma possiamo restituire solo info
        return {
            "info": info,
            "financials": pd.DataFrame(),
            "balance_sheet": pd.DataFrame(),
            "cashflow": pd.DataFrame(),
            "symbol": symbol,
            "source": "MarketWatch"
        }
    
    def get_technical(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        MarketWatch non fornisce facilmente dati storici OHLCV via scraping semplice.
        Restituiamo None per delegare ad altre fonti.
        """
        logger.debug(f"MarketWatch non supporta dati tecnici per {symbol}")
        return None