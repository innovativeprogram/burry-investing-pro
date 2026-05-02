import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import logging
import concurrent.futures
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple, List
from sklearn.linear_model import LinearRegression


import streamlit as st

# 1. Configurazione Pagina (Deve essere la prima istruzione)
st.set_page_config(
    page_title="Burry Investing Pro",
    page_icon="📈",
    layout="wide"
)

# 2. Link al Manifesto PWA (Tutto su una riga per evitare errori)
st.markdown('<link rel="manifest" href="https://raw.githubusercontent.com/Innovativeprogram/burry-investing-pro/main/manifest.json">', unsafe_allow_html=True)

# 3. CSS per nascondere l'interfaccia Streamlit
hide_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.block-container {padding-top: 2rem;}
</style>
"""
st.markdown(hide_style, unsafe_allow_html=True)

# =================================================================
# DA QUI INIZIA IL TUO CODICE ORIGINALE
# =================================================================



# ==========================================



# ==========================================
# 0. SETUP LOGGING & COSTANTI GLOBALI
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


DEFAULT_TAX_RATE = 0.21
SAFE_INTEREST_COVERAGE = 100.0
TRADING_DAYS_YEAR = 252
MAX_CSV_ROWS = 100
MAX_WORKERS = 10
RISK_FREE_RATE = 0.04


# ==========================================
# CONFIGURAZIONE PAGINA UI
# ==========================================
st.set_page_config(
    page_title="BurryInvestingPro",
    page_icon="💎",
    layout="wide"
)


# ==========================================
# 1. MODELLI DATI (Dataclasses)
# ==========================================
@dataclass
class FundamentalMetrics:
    ticker: str
    company_name: str
    price: float
    fcf: float
    roic: float
    peg_ratio: Optional[float]
    peg_source: str
    pe_ratio: Optional[float]
    interest_coverage: float
    currency: str
    raw_data: Dict[str, Any]

    def to_ui_dict(self) -> Dict[str, Any]:
        return {
            "Ticker": self.ticker,
            "Company Name": self.company_name,
            "Price": self.price,
            "Free Cash Flow": self.fcf,
            "ROIC": self.roic,
            "PEG Ratio": self.peg_ratio,
            "PEG Source": self.peg_source,
            "P/E Ratio": self.pe_ratio,
            "Interest Coverage": self.interest_coverage,
            "Currency": self.currency,
            "_raw_data": self.raw_data
        }


# ==========================================
# 2. HELPER FUNCTIONS & VALIDAZIONE
# ==========================================
def sanitize_ticker(ticker: str) -> str:
    clean = str(ticker).strip().upper()
    if not re.match(r"^[A-Z0-9\-\.]+$", clean):
        raise ValueError(f"Ticker contiene caratteri non validi: {clean}")
    return clean


def normalize_ticker(ticker: str, suffix: str) -> str:
    clean_ticker = sanitize_ticker(ticker)
    if "-" in clean_ticker:
        return clean_ticker
    clean_suffix = str(suffix).strip().upper()
    if clean_suffix and not re.match(r"^\.[A-Z]+$", clean_suffix):
        clean_suffix = ""
    if clean_suffix and not clean_ticker.endswith(clean_suffix):
        return f"{clean_ticker}{clean_suffix}"
    return clean_ticker


# ==========================================
# 3. DATA ENGINE: ANALISI FONDAMENTALE
# ==========================================
@st.cache_data(ttl=3600, show_spinner=False)
def get_fundamental_data(symbol: str) -> Optional[Dict[str, Any]]:
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        if not info or 'symbol' not in info:
            return None
        return {
            "info": info,
            "financials": stock.financials,
            "balance_sheet": stock.balance_sheet,
            "cashflow": stock.cashflow,
            "symbol": symbol
        }

    except Exception as e:
        logger.error(f"Errore API Yahoo per {symbol}: {str(e)}")
        return None


def calculate_fundamental_metrics(raw_data: Dict[str, Any]) -> Optional[FundamentalMetrics]:
    try:
        info = raw_data["info"]
        if info.get('quoteType') == 'CRYPTOCURRENCY':
            return FundamentalMetrics(
                ticker=raw_data["symbol"],
                company_name=info.get('shortName', raw_data["symbol"]),
                price=float(info.get('regularMarketPrice') or info.get('currentPrice', 0.0)),
                fcf=0.0,
                roic=0.0,
                peg_ratio=None,
                peg_source="Crypto",
                pe_ratio=None,
                interest_coverage=SAFE_INTEREST_COVERAGE,
                currency=info.get('currency', 'USD'),
                raw_data=raw_data
            )

        fin, bs, cf = raw_data["financials"], raw_data["balance_sheet"], raw_data["cashflow"]

        op_cash = cf.loc['Operating Cash Flow'].iloc[0] if 'Operating Cash Flow' in cf.index else 0.0
        cap_ex = cf.loc['Capital Expenditure'].iloc[0] if 'Capital Expenditure' in cf.index else 0.0
        fcf = float(op_cash + cap_ex)

        total_debt = bs.loc['Total Debt'].iloc[0] if 'Total Debt' in bs.index else 0.0
        equity = bs.loc['Stockholders Equity'].iloc[0] if 'Stockholders Equity' in bs.index else np.nan
        invested_cap = total_debt + equity
        ebit = fin.loc['EBIT'].iloc[0] if 'EBIT' in fin.index else 0.0

        tax_rate = DEFAULT_TAX_RATE
        if 'Tax Provision' in fin.index and 'Pretax Income' in fin.index:
            pretax_inc = fin.loc['Pretax Income'].iloc[0]
            if pretax_inc > 0:
                tax_rate = max(0.0, min(1.0, fin.loc['Tax Provision'].iloc[0] / pretax_inc))

        roic = float((ebit * (1 - tax_rate)) / invested_cap) if (invested_cap > 0 and not np.isnan(equity)) else 0.0

        pe = info.get('trailingPE')
        growth = info.get('earningsGrowth')
        peg = info.get('pegRatio')
        peg_src = "N/A"
        if peg is not None:
            peg_src = "Official"
        elif pe and pe > 0 and growth and growth > 0:
            peg = float(pe / (growth * 100))
            peg_src = "Estimated"

        int_exp = fin.loc['Interest Expense'].iloc[0] if 'Interest Expense' in fin.index else 0.0
        int_cov = float(ebit / abs(int_exp)) if int_exp != 0 else SAFE_INTEREST_COVERAGE

        return FundamentalMetrics(
            ticker=raw_data["symbol"],
            company_name=info.get('longName', raw_data["symbol"]),
            price=float(info.get('currentPrice', 0.0)),
            fcf=fcf,
            roic=roic,
            peg_ratio=float(peg) if peg else None,
            peg_source=peg_src,
            pe_ratio=float(pe) if pe else None,
            interest_coverage=int_cov,
            currency=info.get('currency', 'USD'),
            raw_data=raw_data
        )

    except Exception as e:
        logger.error(f"Errore calcolo metriche {raw_data['symbol']}: {str(e)}")
        return None


# ==========================================
# 4. DATA ENGINE: ANALISI TECNICA
# ==========================================
@st.cache_data(ttl=900, show_spinner=False)
def get_technical_data(symbol: str) -> Optional[pd.DataFrame]:
    try:
        df = yf.download(symbol, period="2y", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df if len(df) >= 200 else None
    except Exception:
        return None


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data['SMA_50'] = ta.sma(data['Close'], length=50)
    data['SMA_200'] = ta.sma(data['Close'], length=200)
    data['RSI'] = ta.rsi(data['Close'], length=14)
    bb = ta.bbands(data['Close'], length=20, std=2)
    if bb is not None:
        data['BB_Lower'] = bb.filter(like='BBL_').iloc[:, 0]
        data['BB_Upper'] = bb.filter(like='BBU_').iloc[:, 0]
    return data


def calculate_timing_score(data: pd.DataFrame, current_price: float) -> Tuple[int, List[str]]:
    score, reasons = 0, []
    last_row = data.iloc[-1]
    if current_price > last_row['SMA_200']:
        score += 30
        reasons.append("✅ Trend Rialzista (Sopra SMA 200)")
    else:
        reasons.append("⚠️ Trend Ribassista (Sotto SMA 200)")

    rsi = last_row['RSI']
    if pd.notna(rsi):
        if rsi < 30:
            score += 30
            reasons.append("✅ Ipervenduto (RSI < 30)")
        elif rsi > 70:
            score -= 10
            reasons.append("🛑 Ipercomprato (RSI > 70)")

    if current_price <= last_row.get('BB_Lower', 0) * 1.02:
        score += 20
        reasons.append("✅ Prezzo su Banda Bollinger Inferiore")

    return score, reasons


# ==========================================
# 5. MOTORE QUANTISTICO (STATISTICA STAZIONARIA)
# ==========================================
def calculate_quant_metrics(df: pd.DataFrame, fund_data: Dict[str, Any]) -> Dict[str, Any]:
    returns = df['Close'].pct_change().dropna()
    excess_returns = returns - (RISK_FREE_RATE / TRADING_DAYS_YEAR)
    sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(TRADING_DAYS_YEAR) if returns.std() != 0 else 0
    vol = returns.std() * np.sqrt(TRADING_DAYS_YEAR)

    log_returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
    cum_log_returns = log_returns.cumsum().values.reshape(-1, 1)
    x = np.arange(len(cum_log_returns)).reshape(-1, 1)
    model = LinearRegression().fit(x, cum_log_returns)
    r_sq = model.score(x, cum_log_returns)

    z_score = "N/A"
    if fund_data['info'].get('quoteType') != 'CRYPTOCURRENCY':
        try:
            bs, fin, info = fund_data["balance_sheet"], fund_data["financials"], fund_data["info"]
            ta_val = bs.loc['Total Assets'].iloc[0]
            wc = (bs.loc['Current Assets'].iloc[0] - bs.loc['Current Liabilities'].iloc[0]) if 'Current Assets' in bs.index else 0
            re = bs.loc['Retained Earnings'].iloc[0] if 'Retained Earnings' in bs.index else 0
            ebit = fin.loc['EBIT'].iloc[0]
            mc = info.get('marketCap', 1)
            tl = bs.loc['Total Liabilities Net Minority Interest'].iloc[0]
            rev = info.get('totalRevenue', 0)
            z_score = (1.2 * (wc / ta_val)) + (1.4 * (re / ta_val)) + (3.3 * (ebit / ta_val)) + (0.6 * (mc / tl)) + (1.0 * (rev / ta_val))
        except Exception:
            pass

    return {
        "Sharpe Ratio": sharpe,
        "Annual Volatility": vol,
        "R-Squared": r_sq,
        "Altman Z-Score": z_score,
        "Price Percentile": (df['Close'] < df['Close'].iloc[-1]).mean() * 100,
        "Trend Slope": model.coef_[0][0]
    }


# ==========================================
# 5.B MODULO DI RISCHIO
# ==========================================
def calculate_risk_metrics(df: pd.DataFrame) -> Dict[str, float]:
    prices = df['Close'].dropna()
    returns = prices.pct_change().dropna()
    if returns.empty:
        return {
            "Max Drawdown": np.nan,
            "CAGR": np.nan,
            "VaR_95": np.nan,
            "CVaR_95": np.nan,
            "Skew": np.nan,
            "Kurt": np.nan
        }

    equity = (1 + returns).cumprod()

    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1.0
    max_dd = drawdown.min()

    total_return = equity.iloc[-1] - 1.0
    years = len(returns) / TRADING_DAYS_YEAR
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0 if years > 0 else np.nan

    alpha = 0.95
    var_95 = np.quantile(returns, 1 - alpha)
    cvar_95 = returns[returns <= var_95].mean() if (returns <= var_95).any() else np.nan

    skew = returns.skew()
    kurt = returns.kurt()

    return {
        "Max Drawdown": float(max_dd),
        "CAGR": float(cagr),
        "VaR_95": float(var_95),
        "CVaR_95": float(cvar_95),
        "Skew": float(skew),
        "Kurt": float(kurt)
    }


# ==========================================
# 5.C MONTE CARLO EQUITY
# ==========================================
def monte_carlo_equity(
    df: pd.DataFrame,
    n_paths: int = 1000,
    horizon_days: int = 252,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    prices = df['Close'].dropna()
    returns = prices.pct_change().dropna().values
    if returns.size == 0:
        return {
            "paths": None,
            "final_distribution": None,
            "q05": None,
            "q50": None,
            "q95": None
        }

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(returns), size=(n_paths, horizon_days))
    sampled = returns[idx]

    equity_paths = (1 + sampled).cumprod(axis=1)

    final_values = equity_paths[:, -1]
    q05 = np.quantile(equity_paths, 0.05, axis=0)
    q50 = np.quantile(equity_paths, 0.50, axis=0)
    q95 = np.quantile(equity_paths, 0.95, axis=0)

    return {
        "paths": equity_paths,
        "final_distribution": final_values,
        "q05": q05,
        "q50": q50,
        "q95": q95
    }


# ==========================================
# 5.D SMART QUANT SCORE
# ==========================================
def compute_smart_quant_score(
    row: Any,
    timing_score: int,
    qm: Dict[str, Any],
    risk: Dict[str, Any]
) -> Dict[str, Any]:

    f_score = 0.0
    roic = row.get("ROIC", 0.0)
    f_score += float(np.clip((roic - 0.10) / (0.25 - 0.10), 0, 1)) * 50.0

    peg = row.get("PEG Ratio", None)
    if peg is not None and peg > 0:
        if peg <= 1:
            f_score += 30.0
        elif peg <= 2:
            f_score += 15.0
        else:
            f_score += 0.0

    z = qm.get("Altman Z-Score", "N/A")
    if isinstance(z, (int, float, np.floating)):
        if z >= 3.0:
            f_score += 20.0
        elif z >= 1.8:
            f_score += 10.0

    f_score = float(np.clip(f_score, 0, 100))

    t_score = float(np.clip(timing_score, 0, 100))

    q_score = 0.0
    sharpe = qm.get("Sharpe Ratio", 0.0)
    max_dd = risk.get("Max Drawdown", 0.0)

    if sharpe <= 0:
        q_score += 0.0
    elif sharpe <= 1:
        q_score += 30.0 * (sharpe / 1.0)
    elif sharpe <= 2:
        q_score += 30.0 + 30.0 * ((sharpe - 1.0) / 1.0)
    else:
        q_score += 80.0

    if isinstance(max_dd, (float, np.floating)):
        if max_dd < -0.5:
            q_score -= 20.0
        elif max_dd < -0.3:
            q_score -= 10.0

    q_score = float(np.clip(q_score, 0, 100))

    smart = 0.4 * f_score + 0.3 * t_score + 0.3 * q_score
    smart = float(np.clip(smart, 0, 100))

    return {
        "SmartScore": smart,
        "FundamentalScore": f_score,
        "TechnicalScore": t_score,
        "QuantRiskScore": q_score
    }


# ==========================================
# 5.E PORTAFOGLIO: RENDIMENTI & METRICHE
# ==========================================
def get_daily_returns_for_ticker(symbol: str) -> Optional[pd.Series]:
    df = get_technical_data(symbol)
    if df is None or df.empty:
        return None
    returns = df["Close"].pct_change().dropna()
    if returns.empty:
        return None
    returns.name = symbol
    return returns


def build_portfolio_returns(
    tickers: List[str],
    weights_pct: Dict[str, float]
) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
    series_list: List[pd.Series] = []
    for t in tickers:
        r = get_daily_returns_for_ticker(t)
        if r is not None:
            series_list.append(r)

    if not series_list:
        return None

    df_rets = pd.concat(series_list, axis=1).dropna()
    if df_rets.empty:
        return None

    cols = df_rets.columns.tolist()
    w = np.array([weights_pct.get(t, 0.0) for t in cols], dtype=float) / 100.0
    if w.sum() <= 0:
        return None
    w = w / w.sum()

    port_ret = (df_rets * w).sum(axis=1)
    port_ret.name = "Portfolio"
    return df_rets, port_ret


def calculate_portfolio_metrics(port_ret: pd.Series) -> Dict[str, float]:
    if port_ret is None or port_ret.empty:
        return {
            "AnnRet": np.nan,
            "AnnVol": np.nan,
            "Sharpe": np.nan,
            "MaxDD": np.nan
        }

    mu = port_ret.mean() * TRADING_DAYS_YEAR
    sigma = port_ret.std() * np.sqrt(TRADING_DAYS_YEAR)

    excess = mu - RISK_FREE_RATE
    sharpe = excess / sigma if sigma > 0 else np.nan

    equity = (1 + port_ret).cumprod()
    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1.0
    max_dd = drawdown.min() if not drawdown.empty else np.nan

    return {
        "AnnRet": float(mu),
        "AnnVol": float(sigma),
        "Sharpe": float(sharpe),
        "MaxDD": float(max_dd)
    }


# ==========================================
# 6. UI: SIDEBAR & STYLE
# ==========================================
def setup_sidebar() -> Dict[str, Any]:
    st.sidebar.header("1. Selezione Asset")
    input_mode = st.sidebar.radio("Modalità:", ["Manuale", "Batch (CSV)"], horizontal=True)
    file, manual = None, None
    if input_mode == "Batch (CSV)":
        file = st.sidebar.file_uploader("Carica CSV", type=["csv"])
    else:
        manual = st.sidebar.text_input("Ticker", value="AAPL").upper().strip()

    st.sidebar.header("2. Mercato")
    market = st.sidebar.selectbox("Borsa:", ["USA ", "Italia (.MI)", "Germania (.DE)", "Francia (.PA)", "GB (.L)", "Crypto", "Custom"])
    suffix = ".MI" if "Italia" in market else (".DE" if "Germania" in market else "")
    analyze_btn = st.sidebar.button("🚀 Avvia Analisi", use_container_width=True)

    with st.sidebar.expander("⚙️ Parametri Fondamentali"):
        cfg = {
            "roic": st.number_input("Min ROIC %", 10.0, step=0.5),
            "fcf": st.number_input("Min FCF (Mld $)", 0.0) * 1e9,
            "peg": st.number_input("Max PEG Ratio", 1.5, step=0.1),
            "pe": st.number_input("Max P/E (Fallback)", 25.0),
            "int_cov": st.number_input("Min Int. Coverage", 3.0),
            "perfect_only": st.checkbox("🏆 Solo 'All Green'")
        }

    with st.sidebar.expander("❓ Come cercare il ticker corretto"):
        st.markdown(
            "- Azioni USA: normalmente solo ticker (es. `AAPL`, `MSFT`).\n"
            "- Azioni italiane: aggiungi `.MI` (es. `STLAM.MI` per Stellantis, `ENI.MI`, `ISP.MI`).\n"
            "- Azioni tedesche: aggiungi `.DE` (es. `BMW.DE`, `SAP.DE`).\n"
            "- Azioni francesi: aggiungi `.PA` (es. `AIR.PA` per Airbus, `OR.PA` per L'Oréal).\n"
            "- Azioni UK: aggiungi `.L` (es. `ULVR.L` per Unilever).\n"
            "- Crypto: di solito coppia con valuta, es. `BTC-USD`, `ETH-USD`.\n"
            "- Se hai dubbi, cerca prima il titolo su Yahoo Finance e copia il ticker esatto."
        )

    return {"mode": input_mode, "file": file, "manual": manual, "suffix": suffix, "btn": analyze_btn, "cfg": cfg}


# ==========================================
# 7. MAIN ORCHESTRATOR
# ==========================================
def main():
    st.title("💎 BurryInvestingPro")
    if 'batch_results' not in st.session_state:
        st.session_state.batch_results = None
    if 'selected_ticker' not in st.session_state:
        st.session_state.selected_ticker = None
    if 'portfolio_tickers' not in st.session_state:
        st.session_state.portfolio_tickers = []
    if 'holdings' not in st.session_state:
        st.session_state.holdings = {}
    if 'holdings_currency' not in st.session_state:
        st.session_state.holdings_currency = {}

    ui = setup_sidebar()
    if ui["btn"]:
        targets: List[str] = [ui["manual"]] if ui["mode"] == "Manuale" else []
        if ui["mode"] == "Batch (CSV)" and ui["file"]:
            targets = pd.read_csv(ui["file"])["Ticker"].tolist()[:MAX_CSV_ROWS]

        if targets:
            results: List[Dict[str, Any]] = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futures = [ex.submit(normalize_ticker, t, ui["suffix"]) for t in targets]
                tickers = [f.result() for f in concurrent.futures.as_completed(futures)]

            for t in tickers:
                raw = get_fundamental_data(t)
                if raw:
                    met = calculate_fundamental_metrics(raw)
                    if met:
                        results.append(met.to_ui_dict())
            st.session_state.batch_results = pd.DataFrame(results)
            if results:
                st.session_state.selected_ticker = results[0]["Ticker"]

    tab_f, tab_t, tab_q, tab_v, tab_p = st.tabs(["📊 FONDAMENTALI", "📉 TECNICO", "⚛️ QUANT", "⚖️ VERDETTO", "📁 PORTAFOGLIO"])

    ticker = st.session_state.selected_ticker
    if ticker and st.session_state.batch_results is not None:
        row = st.session_state.batch_results[st.session_state.batch_results['Ticker'] == ticker].iloc[0]

        # --- TAB FONDAMENTALI ---
        with tab_f:
            st.info("💡 **Come leggere questa sezione:** Questa tabella rappresenta il motore dell'azienda. Cerca società con un ROIC (ritorno sul capitale investito) costantemente alto e un debito gestibile. Il Free Cash Flow è il vero denaro prodotto dal business. Ricorda sempre: il prezzo è quello che paghi, il valore è quello che ottieni.")
            
            st.dataframe(st.session_state.batch_results.drop(columns=["_raw_data"]))
            
            st.markdown("---")
            st.markdown("<p style='text-align: center; color: gray;'>creato e sviluppato da Innovative Program[source: 1]</p>", unsafe_allow_html=True)

        # --- TAB TECNICO ---
        with tab_t:
            st.info("💡 **Come leggere il grafico:** Anche se preferiamo studiare il business, il prezzo ci dice come si muove il mercato. La linea blu (Media Mobile a 200 giorni) indica il trend di lungo periodo: se il prezzo è sopra, la marea è a nostro favore. Il grafico in basso (RSI) segnala se c'è troppa euforia (valori sopra 70, attenzione) o troppo pessimismo (sotto 30, possibili occasioni).")
            
            df_tech = get_technical_data(ticker)
            if df_tech is not None:
                df_calc = calculate_technical_indicators(df_tech)
                score, _ = calculate_timing_score(df_calc, df_calc['Close'].iloc[-1])
                st.metric("Timing Score", f"{score}/100")
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
                fig.add_trace(
                    go.Candlestick(
                        x=df_calc.index,
                        open=df_calc['Open'],
                        high=df_calc['High'],
                        low=df_calc['Low'],
                        close=df_calc['Close'],
                        name="Prezzo"
                    ),
                    row=1,
                    col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=df_calc.index,
                        y=df_calc['SMA_200'],
                        name="SMA 200",
                        line=dict(color='blue')
                    ),
                    row=1,
                    col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=df_calc.index,
                        y=df_calc['RSI'],
                        name="RSI",
                        line=dict(color='purple')
                    ),
                    row=2,
                    col=1
                )
                fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.markdown("<p style='text-align: center; color: gray;'>creato e sviluppato da Innovative Program[source: 1]</p>", unsafe_allow_html=True)

        # --- TAB QUANT ---
        with tab_q:
            st.info("💡 **Come interpretare i dati:** Qui lasciamo parlare la statistica. Lo **Sharpe Ratio** ci dice quanto rendimento stiamo ottenendo per ogni unità di rischio sopportata (più è alto, meglio è). L'**Altman Z-Score** è vitale per allontanarci dalle aziende a rischio bancarotta (sopra 3 è ottimo). I grafici in basso (Monte Carlo) simulano gli scenari futuri in base alla volatilità storica, mostrandoci il rischio concreto (Drawdown) di perdite permanenti.")
            
            df_tech = get_technical_data(ticker)
            if df_tech is not None:
                qm = calculate_quant_metrics(df_tech, row["_raw_data"])
                risk = calculate_risk_metrics(df_tech)

                c1, c2, c3 = st.columns(3)
                c1.metric("Sharpe Ratio (4% Rf)", f"{qm['Sharpe Ratio']:.2f}")
                c2.metric("Trend R-Squared", f"{qm['R-Squared']:.2f}")
                c3.metric("Altman Z-Score", f"{qm['Altman Z-Score']:.2f}" if isinstance(qm['Altman Z-Score'], float) else "N/A")

                c4, c5, c6 = st.columns(3)
                c4.metric("Max Drawdown", f"{risk['Max Drawdown'] * 100:.1f}%")
                c5.metric("CAGR", f"{risk['CAGR'] * 100:.1f}%")
                c6.metric("VaR 95% (giornaliero)", f"{risk['VaR_95'] * 100:.2f}%")

                smart = compute_smart_quant_score(row, score, qm, risk)
                st.metric("Smart Quant Score", f"{smart['SmartScore']:.1f}/100")

                with st.expander("📉 Distribuzione rendimenti & rischio"):
                    st.write(f"CVaR 95%: {risk['CVaR_95'] * 100:.2f}%")
                    st.write(f"Skewness: {risk['Skew']:.2f} | Kurtosis: {risk['Kurt']:.2f}")

                    returns = df_tech['Close'].pct_change().dropna()
                    fig_r = go.Figure()
                    fig_r.add_trace(go.Histogram(x=returns, nbinsx=50, name="Rendimenti giornalieri"))
                    fig_r.update_layout(template="plotly_dark", bargap=0.05)
                    st.plotly_chart(fig_r, use_container_width=True)

                with st.expander("🎲 Simulazione Monte Carlo (rendimenti storici)"):
                    col_mc1, col_mc2 = st.columns(2)
                    horizon_days = col_mc1.slider("Orizzonte (giorni trading)", 60, 756, 252, step=21)
                    n_paths = col_mc2.slider("Numero traiettorie", 100, 3000, 1000, step=100)

                    mc = monte_carlo_equity(df_tech, n_paths=n_paths, horizon_days=horizon_days)

                    if mc["paths"] is not None:
                        final_vals = mc["final_distribution"]
                        p05 = np.quantile(final_vals, 0.05)
                        p50 = np.quantile(final_vals, 0.50)

                        c7, c8, c9 = st.columns(3)
                        c7.metric("Prob. perdita > 20%", f"{(final_vals < 0.8).mean() * 100:.1f}%")
                        c8.metric("Mediana esito", f"{(p50 - 1) * 100:.1f}%")
                        c9.metric("Scenario 5° percentile", f"{(p05 - 1) * 100:.1f}%")

                        x = np.arange(1, horizon_days + 1)
                        fig_mc = go.Figure()
                        fig_mc.add_trace(go.Scatter(
                            x=x, y=mc["q50"], name="Mediana",
                            line=dict(color="cyan")
                        ))
                        fig_mc.add_trace(go.Scatter(
                            x=x, y=mc["q95"], name="95° percentile",
                            line=dict(color="green"), opacity=0.3
                        ))
                        fig_mc.add_trace(go.Scatter(
                            x=x, y=mc["q05"], name="5° percentile",
                            line=dict(color="red"), opacity=0.3,
                            fill="tonexty", fillcolor="rgba(255,0,0,0.1)"
                        ))
                        fig_mc.update_layout(
                            template="plotly_dark",
                            height=400,
                            xaxis_title="Giorni",
                            yaxis_title="Equity normalizzata"
                        )
                        st.plotly_chart(fig_mc, use_container_width=True)
            
            st.markdown("---")
            st.markdown("<p style='text-align: center; color: gray;'>creato e sviluppato da Innovative Program[source: 1]</p>", unsafe_allow_html=True)

        # --- TAB VERDETTO ---
        with tab_v:
            st.info("💡 **Come leggere il verdetto:** Questa è la nostra sintesi razionale. Combina la solidità del business (Fondamentali), il momento (Tecnico) e le probabilità statistiche (Quant). Richiedi sempre un margine di sicurezza: investi solo quando il verdetto ti suggerisce che il rischio di perdere capitale in modo permanente è bassissimo.")
            
            df_tech = get_technical_data(ticker)
            qm = calculate_quant_metrics(df_tech, row["_raw_data"]) if df_tech is not None else {}
            risk = calculate_risk_metrics(df_tech) if df_tech is not None else {
                "Max Drawdown": 0.0,
                "CAGR": 0.0
            }

            z_val = qm.get('Altman Z-Score', 0.0)
            z_safe = "-" in ticker or (isinstance(z_val, float) and z_val >= 1.8)
            fund_pts = (
                (1 if row['ROIC'] >= ui['cfg']['roic'] / 100 else 0) +
                (1 if row['PEG Ratio'] and row['PEG Ratio'] <= ui['cfg']['peg'] else 0)
            )

            if fund_pts >= 2 and z_safe and score >= 50:
                st.success("🟢 BUY: Fondamentali solidi e timing favorevole.")
            elif fund_pts >= 1 and z_safe:
                st.warning("🟡 HOLD: Azienda sicura ma attendere prezzi migliori.")
            else:
                st.error("🔴 SELL: Rischio finanziario o fondamentali scarsi.")

            if df_tech is not None and qm:
                smart_v = compute_smart_quant_score(row, score, qm, risk)
                st.metric("Smart Quant Score", f"{smart_v['SmartScore']:.1f}/100")
                st.write(
                    f"F: {smart_v['FundamentalScore']:.0f} | "
                    f"T: {smart_v['TechnicalScore']:.0f} | "
                    f"Q: {smart_v['QuantRiskScore']:.0f}"
                )

                if smart_v["SmartScore"] >= 70 and z_safe:
                    st.success("🟢 BUY (Quant): Vantaggio statistico e rischio controllato.")
                elif smart_v["SmartScore"] >= 50 and z_safe:
                    st.warning("🟡 HOLD (Quant): Setup discreto ma non eccezionale.")
                else:
                    st.error("🔴 NO TRADE (Quant): Vantaggio quantitativo debole o rischio elevato.")
            
            st.markdown("---")
            st.markdown("<p style='text-align: center; color: gray;'>creato e sviluppato da Innovative Program</p>", unsafe_allow_html=True)

        # --- TAB PORTAFOGLIO ---
        with tab_p:
            st.info("💡 **Come usare questa sezione:** La diversificazione sensata è la protezione per il nostro capitale. Qui aggreghiamo i tuoi investimenti. Il grafico ti mostra la crescita combinata, mentre la matrice di correlazione in fondo è cruciale: se le aziende che possiedi tendono a muoversi tutte nella stessa direzione contemporaneamente, non sei diversificato come pensi.")
            
            st.markdown(
                "### Portafoglio reale\n"
                "- Qui inserisci le **posizioni effettive** che hai in portafoglio.\n"
                "- Per ogni titolo, indica il **ticker corretto per il mercato** e l'**importo investito**.\n"
                "- Il sistema calcola automaticamente il **peso percentuale** di ogni posizione e le metriche "
                "quantitative del portafoglio (rendimento atteso, volatilità, Sharpe, drawdown).\n\n"
                "**Come cercare il ticker giusto:**\n"
                "- Azioni USA: normalmente solo ticker (es. `AAPL`, `MSFT`).\n"
                "- Azioni italiane: aggiungi `.MI` (es. `STLAM.MI` per Stellantis a Milano, `ENI.MI`, `ISP.MI`).\n"
                "- Azioni tedesche: aggiungi `.DE` (es. `BMW.DE`, `SAP.DE`).\n"
                "- Azioni francesi: aggiungi `.PA` (es. `AIR.PA` per Airbus, `OR.PA` per L'Oréal`).\n"
                "- Azioni UK: aggiungi `.L` (es. `ULVR.L` per Unilever).\n"
                "- Crypto: in genere coppia con valuta, es. `BTC-USD`, `ETH-USD`.\n"
            )

            all_tickers_batch: List[str] = []
            if st.session_state.batch_results is not None and not st.session_state.batch_results.empty:
                all_tickers_batch = st.session_state.batch_results["Ticker"].tolist()

            if all_tickers_batch:
                st.markdown("#### Seleziona dal batch analizzato")
                default_batch = [
                    t for t in st.session_state.portfolio_tickers
                    if t in all_tickers_batch
                ]
                selected_from_batch = st.multiselect(
                    "Titoli da includere nel portafoglio (da batch)",
                    all_tickers_batch,
                    default=default_batch
                )
            else:
                selected_from_batch = []

            st.markdown("#### Aggiungi manualmente altri ticker")
            manual_ticker = st.text_input(
                "Ticker (incluso suffisso mercato, es. STLAM.MI, ENI.MI, BMW.DE, AIR.PA, ULVR.L)",
                ""
            )

            if st.button("➕ Aggiungi ticker manuale al portafoglio"):
                if manual_ticker.strip():
                    t_clean = sanitize_ticker(manual_ticker)
                    if t_clean not in st.session_state.portfolio_tickers:
                        st.session_state.portfolio_tickers.append(t_clean)
                        st.success(f"Aggiunto {t_clean} al portafoglio.")
                else:
                    st.warning("Inserisci un ticker valido prima di aggiungere.")

            portfolio_list = sorted(set(selected_from_batch + st.session_state.portfolio_tickers))
            st.session_state.portfolio_tickers = portfolio_list

            if portfolio_list:
                st.markdown("#### Importo investito per ogni posizione")
                cols = st.columns(3)
                holdings: Dict[str, float] = st.session_state.holdings

                for i, t in enumerate(portfolio_list):
                    col = cols[i % 3]
                    default_amt = float(holdings.get(t, 0.0))
                    amt = col.number_input(
                        f"{t} - Importo investito",
                        min_value=0.0,
                        value=default_amt,
                        step=100.0,
                        key=f"holding_{t}"
                    )
                    holdings[t] = amt

                    cur_default = st.session_state.holdings_currency.get(t, "USD")
                    cur = col.selectbox(
                        f"{t} - Valuta",
                        ["USD", "EUR"],
                        index=["USD", "EUR"].index(cur_default),
                        key=f"currency_{t}"
                    )
                    st.session_state.holdings_currency[t] = cur

                    if col.button("🗑 Rimuovi", key=f"remove_{t}"):
                        if t in st.session_state.portfolio_tickers:
                            st.session_state.portfolio_tickers = [
                                x for x in st.session_state.portfolio_tickers if x != t
                            ]
                        if t in st.session_state.holdings:
                            del st.session_state.holdings[t]
                        if t in st.session_state.holdings_currency:
                            del st.session_state.holdings_currency[t]
                        st.rerun()

                st.session_state.holdings = holdings

                if st.button("📊 Calcola pesi e analisi del portafoglio"):
                    positive_holdings = {t: a for t, a in holdings.items() if a > 0}
                    if not positive_holdings:
                        st.error("Imposta un importo > 0 almeno per un titolo.")
                    else:
                        tot = sum(positive_holdings.values())
                        weights_pct = {t: a / tot * 100.0 for t, a in positive_holdings.items()}

                        built = build_portfolio_returns(list(positive_holdings.keys()), weights_pct)
                        if built is None:
                            st.error("Impossibile costruire la serie dei rendimenti (dati insufficienti per uno o più ticker).")
                        else:
                            df_rets, port_ret = built
                            pm = calculate_portfolio_metrics(port_ret)

                            st.markdown("#### Pesi percentuali del portafoglio")
                            df_weights = pd.DataFrame(
                                {
                                    "Ticker": list(weights_pct.keys()),
                                    "Importo": [positive_holdings[t] for t in weights_pct.keys()],
                                    "Peso %": [weights_pct[t] for t in weights_pct.keys()]
                                }
                            )
                            df_weights["Valuta"] = [
                                st.session_state.holdings_currency.get(t, "USD")
                                for t in weights_pct.keys()
                            ]
                            st.dataframe(df_weights)

                            cpa, cpv, cps, cpdd = st.columns(4)
                            cpa.metric("Rendimento annuo atteso", f"{pm['AnnRet'] * 100:.2f}%")
                            cpv.metric("Volatilità annua", f"{pm['AnnVol'] * 100:.2f}%")
                            cps.metric("Sharpe portafoglio", f"{pm['Sharpe']:.2f}")
                            cpdd.metric("Max Drawdown portafoglio", f"{pm['MaxDD'] * 100:.1f}%")

                            equity_p = (1 + port_ret).cumprod()
                            fig_p = go.Figure()
                            fig_p.add_trace(go.Scatter(
                                x=equity_p.index,
                                y=equity_p.values,
                                name="Equity portafoglio"
                            ))
                            fig_p.update_layout(
                                template="plotly_dark",
                                height=400,
                                xaxis_title="Data",
                                yaxis_title="Equity normalizzata"
                            )
                            st.plotly_chart(fig_p, use_container_width=True)

                            corr = df_rets.corr()
                            st.markdown("#### Correlazioni tra titoli in portafoglio")
                            st.dataframe(corr.style.background_gradient(cmap="RdYlGn", axis=None))
            else:
                st.info("Seleziona almeno un titolo dal batch o aggiungilo manualmente per costruire il portafoglio.")
                
            st.markdown("---")
            st.markdown("<p style='text-align: center; color: gray;'>creato e sviluppato da Innovative Program[source: 1]</p>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
