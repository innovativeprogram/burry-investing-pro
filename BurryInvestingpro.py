import streamlit as st
import pandas as pd
import plotly.express as px

# Configurazione Pagina
st.set_page_config(page_title="Burry Investing Pro", layout="wide")

st.title("📊 Burry Investing Pro - Portfolio Dashboard")
st.write("Analisi degli investimenti in stile Berkshire Hathaway.")

# Dati del Portafoglio (da informazioni salvate)
portfolio_data = {
    'ETF': ['VWCE', 'VGGF', 'IS3S', 'XDWU'],
    'ISIN': ['IE00BK5BQT80', 'IE000B1A2798', 'IE00BP3QZB59', 'IE00BM67HQ30'],
    'DAC (€)': [145.41, 5.12, 56.09, 43.36],
    'Quote': [8.788929, 19.935399, 0.392206, 1.49916],
    'TER (%)': [0.19, 0.10, 0.25, 0.25]
}

df = pd.DataFrame(portfolio_data)
df['Investimento Totale (€)'] = df['DAC (€)'] * df['Quote']

# Visualizzazione Dati - Utilizzo del nuovo parametro width='stretch'
st.subheader("Riepilogo Posizioni")
st.dataframe(df, width='stretch') # Sostituito use_container_width=True

# Grafico Distribuzione
st.subheader("Allocazione del Capitale")
fig = px.pie(df, values='Investimento Totale (€)', names='ETF', hole=0.4)
st.plotly_chart(fig, width='stretch') # Sostituito use_container_width=True

# Analisi Costi
st.subheader("Efficienza dei Costi (TER)")
st.bar_chart(df.set_index('ETF')['TER (%)'], width='stretch') # Sostituito use_container_width=True

st.info("Nota: I grafici ora utilizzano width='stretch' per garantire la compatibilità post-2025.")
