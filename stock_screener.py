import yfinance as yf
import pandas as pd
import streamlit as st
from typing import Dict, Any, List

# Lista dei ticker S&P 500 (primi 50 per velocità, puoi ampliare)
SP500_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "UNH", "JNJ", "V",
    "PG", "JPM", "HD", "MA", "CVX", "MRK", "ABBV", "PEP", "KO", "COST",
    "TMO", "AVGO", "ADBE", "CRM", "NFLX", "CSCO", "ACN", "LIN", "TMUS", "AMD",
    "PM", "NKE", "UPS", "T", "SCHW", "LOW", "SPGI", "BLK", "SBUX", "BA",
    "GE", "GS", "MS", "BKNG", "RTX", "HON", "UNP", "LMT", "MDT", "QCOM"
]

@st.cache_data(ttl=3600)
def get_screened_tickers(filters: Dict[str, Any], ticker_list: List[str] = None) -> pd.DataFrame:
    """
    Applica filtri ai ticker e restituisce DataFrame con metriche.
    Filtri possibili: min_roic, max_peg, min_fcf_yield, max_pe, min_mcap (in miliardi), sector (non implementato)
    """
    if ticker_list is None:
        ticker_list = SP500_TICKERS
    results = []
    for ticker in ticker_list[:50]:  # limitiamo a 50 per performance
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            roic = info.get('returnOnInvestedCapital')
            peg = info.get('pegRatio')
            fcf_yield = info.get('freeCashflowYield')  # in decimal
            pe = info.get('trailingPE')
            market_cap = info.get('marketCap')
            
            # Applica filtri
            if filters.get('min_roic') and roic and roic < filters['min_roic']:
                continue
            if filters.get('max_peg') and peg and peg > filters['max_peg']:
                continue
            if filters.get('min_fcf_yield') and fcf_yield and fcf_yield < filters['min_fcf_yield']:
                continue
            if filters.get('max_pe') and pe and pe > filters['max_pe']:
                continue
            if filters.get('min_mcap') and market_cap and (market_cap / 1e9) < filters['min_mcap']:
                continue
                
            results.append({
                'Ticker': ticker,
                'Company': info.get('longName', ticker),
                'ROIC %': round(roic * 100, 2) if roic else None,
                'PEG': round(peg, 2) if peg else None,
                'FCF Yield %': round(fcf_yield * 100, 2) if fcf_yield else None,
                'P/E': round(pe, 2) if pe else None,
                'Market Cap (B)': round(market_cap / 1e9, 2) if market_cap else None,
            })
        except Exception:
            continue
    return pd.DataFrame(results)