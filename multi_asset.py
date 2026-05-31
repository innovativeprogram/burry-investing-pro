"""
multi_asset.py
Gestione multi-asset: materie prime, valute, bond.
"""

import pandas as pd
import streamlit as st
from typing import Dict, Any

ASSET_LIST = {
    "Oro": ["GC=F", "GLD"],
    "Petrolio": ["CL=F", "USO"],
    "EUR/USD": ["EURUSD=X"],
    "US Treasury 10Y": ["^TNX", "TLT"],
    "Bitcoin": ["BTC-USD"]
}

def get_multi_asset_data(get_technical_data_func) -> Dict[str, Dict[str, Any]]:
    results = {}
    for name, symbols in ASSET_LIST.items():
        for sym in symbols:
            df = get_technical_data_func(sym)
            if df is not None and not df.empty:
                close = df['Close']
                results[name] = {
                    "symbol": sym,
                    "price": close.iloc[-1],
                    "change_1d": close.pct_change().iloc[-1] * 100,
                    "volatility": close.pct_change().std() * 100
                }
                break
    return results

def render_multi_asset_tab(get_technical_data):
    st.subheader("🌍 Asset Macro")
    data = get_multi_asset_data(get_technical_data)
    if not data:
        st.info("Nessun dato multi-asset disponibile")
        return
    df = pd.DataFrame(data).T
    st.dataframe(df, use_container_width=True)