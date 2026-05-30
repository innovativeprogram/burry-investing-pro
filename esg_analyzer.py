import yfinance as yf
import streamlit as st

@st.cache_data(ttl=3600)
def get_esg_score(ticker: str) -> dict:
    """Restituisce punteggio ESG e altri dati se disponibili."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        esg = info.get('totalEsg')
        esg_risk = info.get('esgRisk')
        return {
            'score': esg,
            'risk': esg_risk,
            'available': esg is not None
        }
    except:
        return {'score': None, 'risk': None, 'available': False}