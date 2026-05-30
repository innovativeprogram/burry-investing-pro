import pandas as pd
import numpy as np
import streamlit as st
import pandas_datareader.data as web
from datetime import datetime, timedelta

@st.cache_data(ttl=86400)  # 1 giorno
def get_macro_data():
    """Recupera dati macro da FRED."""
    end = datetime.now()
    start = end - timedelta(days=365*2)
    indicators = {}
    try:
        # GDP (trimestrale)
        gdp = web.DataReader('GDP', 'fred', start, end)
        indicators['gdp_latest'] = gdp.iloc[-1, 0] if not gdp.empty else None
        # Disoccupazione
        unemp = web.DataReader('UNRATE', 'fred', start, end)
        indicators['unemployment'] = unemp.iloc[-1, 0] if not unemp.empty else None
        # CPI (inflazione YoY)
        cpi = web.DataReader('CPIAUCSL', 'fred', start, end)
        if len(cpi) >= 13:
            yoy = (cpi.iloc[-1, 0] / cpi.iloc[-13, 0] - 1) * 100
            indicators['cpi_yoy'] = yoy
        # Fed Funds Rate
        fed = web.DataReader('FEDFUNDS', 'fred', start, end)
        indicators['fed_rate'] = fed.iloc[-1, 0] if not fed.empty else None
    except Exception as e:
        st.warning(f"Macro data error: {e}")
    return indicators