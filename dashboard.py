"""
dashboard.py
Nuova interfaccia principale: mostra IQ Score in alto, poi tabs compatti.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def render_dashboard(
    ticker: str,
    row: pd.Series,
    standalone_raw_data: dict,
    batch_results: pd.DataFrame,
    # funzioni di rendering originali (passate dal main)
    render_fundamental_tab,
    render_technical_tab,
    render_quant_tab,
    render_verdict_tab,
    render_portfolio_tab,
    # funzioni di calcolo
    compute_iq_score_func,
    get_technical_data,
    get_macro_indicators,
    get_unified_verdict,
    ask_gemini_chat,
    # altri parametri UI
    ui_config: dict = None
):
    if ticker is None or row is None:
        st.info("Seleziona un ticker dalla barra laterale o dalla ricerca rapida.")
        return
    
    # Recupera dati tecnici e metriche per IQ Score
    df_tech = get_technical_data(ticker)
    timing_score = 0
    qm = {}
    risk = {}
    if df_tech is not None:
        # Import delle funzioni tecniche (se non già disponibili)
        try:
            from technical_indicators import calculate_technical_indicators, calculate_timing_score
            df_calc = calculate_technical_indicators(df_tech)
            timing_score, _ = calculate_timing_score(df_calc, df_calc['Close'].iloc[-1])
            from quant_metrics import calculate_quant_metrics, calculate_risk_metrics
            qm = calculate_quant_metrics(df_tech, row.get('_raw_data', standalone_raw_data))
            risk = calculate_risk_metrics(df_tech)
        except ImportError:
            st.warning("Moduli tecnici non disponibili – IQ Score parziale")
    
    # Calcola IQ Score (con sentiment base 0.5)
    iq = compute_iq_score_func(
        ticker, row, timing_score, qm, risk, sentiment_score=0.5,
        get_unified_verdict=get_unified_verdict,
        get_macro=get_macro_indicators,
        get_technical_data=get_technical_data
    )
    
    # Header
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        st.subheader(f"{ticker} – {row.get('Company Name', ticker)}")
    with col2:
        st.metric("Prezzo", f"{row.get('Price', 0):.2f} {row.get('Currency', 'USD')}")
    with col3:
        st.metric("IQ Score", f"{iq['IQ_Score']:.1f}/100", delta=f"{iq['verdict']}")
    with col4:
        st.metric("Verdetto", f"{iq['emoji']} {iq['verdict']}")
    
    st.progress(int(iq['IQ_Score']))
    
    # Componenti in piccolo
    with st.expander("Componenti IQ Score", expanded=False):
        comp = iq['components']
        cols = st.columns(5)
        cols[0].metric("FQS", f"{comp['FQS']:.0f}")
        cols[1].metric("VAS", f"{comp['VAS']:.0f}")
        cols[2].metric("TMS", f"{comp['TMS']:.0f}")
        cols[3].metric("QRS", f"{comp['QRS']:.0f}")
        cols[4].metric("Momentum", f"{comp['Momentum']:.0f}")
        st.caption(f"Fattore sentiment: {comp['SentimentFactor']:.2f}")
    
    # Tabs compatti (riutilizzo le tue funzioni originali)
    tabs = st.tabs(["📊 Fondamentali", "📈 Tecnico", "⚛️ Quant", "⚖️ Verdetto", "📁 Portafoglio", "🤖 AI Assistant"])
    
    with tabs[0]:
        render_fundamental_tab(row, batch_results, 'dashboard', ticker)
    with tabs[1]:
        render_technical_tab(row, ticker)
    with tabs[2]:
        render_quant_tab(row, ticker, standalone_raw_data)
    with tabs[3]:
        render_verdict_tab(row, ticker, standalone_raw_data)
    with tabs[4]:
        render_portfolio_tab(ui_config or {})
    with tabs[5]:
        # Chat AI contestuale
        st.markdown("#### Chiedi a VqAi")
        ctx = f"""
        Ticker: {ticker}
        IQ Score: {iq['IQ_Score']:.1f}/100
        Componenti: {iq['components']}
        Verdetto: {iq['verdict']}
        Dati fondamentali: {row.to_dict()}
        """
        user_q = st.chat_input("Domanda sul ticker...")
        if user_q:
            reply = ask_gemini_chat(ctx, user_q, mode="Unificato")
            st.chat_message("assistant").write(reply)