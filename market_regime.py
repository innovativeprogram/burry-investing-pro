import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Any

@st.cache_data(ttl=3600)
def get_vix_level() -> float:
    """Restituisce il valore corrente del VIX."""
    try:
        vix = yf.Ticker("^VIX").history(period="1mo")['Close']
        return vix.iloc[-1]
    except:
        return 20.0

@st.cache_data(ttl=3600)
def get_breadth(ticker_list: List[str] = None) -> float:
    """Percentuale di ticker sopra la SMA 50."""
    if ticker_list is None:
        from stock_screener import SP500_TICKERS
        ticker_list = SP500_TICKERS[:50]  # limit
    above = 0
    for t in ticker_list:
        try:
            df = yf.Ticker(t).history(period="1y")
            if len(df) >= 50:
                sma50 = df['Close'].rolling(50).mean().iloc[-1]
                if df['Close'].iloc[-1] > sma50:
                    above += 1
        except:
            continue
    return (above / len(ticker_list)) * 100 if ticker_list else 50.0

def detect_market_regime(vix: float, breadth: float) -> str:
    """Restituisce il regime in base a VIX e breadth."""
    if vix > 25:
        return "🔴 Alto stress / Ribassista"
    elif vix < 15:
        return "🟢 Tranquillo / Rialzista"
    else:
        if breadth > 60:
            return "🟡 Neutro con forza"
        elif breadth < 40:
            return "🟠 Neutro debole"
        else:
            return "🟡 Neutro"