import yfinance as yf
import pandas as pd
import streamlit as st
from typing import Dict, Any, Optional

@st.cache_data(ttl=300)
def get_option_chain(ticker: str):
    """Restituisce la catena delle opzioni per la prima scadenza disponibile."""
    stock = yf.Ticker(ticker)
    expirations = stock.options
    if not expirations:
        return None
    opt_chain = stock.option_chain(expirations[0])
    return opt_chain

def calculate_option_greeks(option_chain: pd.DataFrame, underlying_price: float, risk_free: float = 0.04) -> pd.DataFrame:
    """
    Calcola Delta approssimato usando Black-Scholes semplificato.
    Richiede py_vollib (opzionale). Se non installato, restituisce solo prezzo e strike.
    """
    try:
        from py_vollib.black_scholes import black_scholes as bs
        from py_vollib.black_scholes.greeks.analytical import delta
        has_vollib = True
    except ImportError:
        has_vollib = False
        st.info("Per calcoli greci avanzati installa 'py_vollib'")
    
    df = option_chain.copy()
    if has_vollib:
        deltas = []
        for idx, row in df.iterrows():
            S = underlying_price
            K = row['strike']
            T = 30 / 365  # approssimazione a 30 giorni
            r = risk_free
            sigma = 0.3  # volatilità implicita fissa (approssimativa)
            option_type = 'c' if 'call' in str(row) else 'p'  # adattare
            try:
                d = delta(option_type, S, K, T, r, sigma)
                deltas.append(d)
            except:
                deltas.append(None)
        df['Delta'] = deltas
    return df