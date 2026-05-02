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
    page_icon="🐂“ˆ",
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

st.sidebar.header("Informazioni")
st.sidebar.info("""
**Burry Investing Pro**
Dashboard avanzata per l'analisi del valore intrinseco e il monitoraggio dei mercati globali. 
*Analisi basata su dati di mercato in tempo reale e modelli quantitativi.*
""")


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
    page_icon="ðŸ’Ž",
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
        reasons.append("âœ… Trend Rialzista (Sopra SMA 200)")
    else:
        reasons.append("âš ï¸ Trend Ribassista (Sotto SMA 200)")

    rsi = last_row['RSI']
    if pd.notna(rsi):
        if rsi < 30:
            score += 30
            reasons.append("âœ… Ipervenduto (RSI < 30)")
        elif rsi > 70:
            score -= 10
            reasons.append("ðŸ›‘ Ipercomprato (RSI > 70)")

    if current_price <= last_row.get('BB_Lower', 0) * 1.02:
        score += 20
        reasons.append("âœ… Prezzo su Banda Bollinger Inferiore")

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
    input_mode = st.sidebar.radio("ModalitÃ :", ["Manuale", "Batch (CSV)"], horizontal=True)
    file, manual = None, None
    if input_mode == "Batch (CSV)":
        file = st.sidebar.file_uploader("Carica CSV", type=["csv"])
    else:
        manual = st.sidebar.text_input("Ticker", value="AAPL").upper().strip()

    st.sidebar.header("2. Mercato")
    market = st.sidebar.selectbox("Borsa:", ["USA ", "Italia (.MI)", "Germania (.DE)", "Francia (.PA)", "GB (.L)", "Crypto", "Custom"])
    suffix = ".MI" if "Italia" in market else (".DE" if "Germania" in market else "")
    analyze_btn = st.sidebar.button("ðŸš€ Avvia Analisi", use_container_width=True)

    with st.sidebar.expander("âš™ï¸ Parametri Fondamentali"):
        cfg = {
            "roic": st.number_input("Min ROIC %", 10.0, step=0.5),
            "fcf": st.number_input("Min FCF (Mld $)", 0.0) * 1e9,
            "peg": st.number_input("Max PEG Ratio", 1.5, step=0.1),
            "pe": st.number_input("Max P/E (Fallback)", 25.0),
            "int_cov": st.number_input("Min Int. Coverage", 3.0),
            "perfect_only": st.checkbox("ðŸ† Solo 'All Green'")
        }

    with st.sidebar.expander("â“ Come cercare il ticker corretto"):
        st.markdown(
            "- Azioni USA: normalmente solo ticker (es. `AAPL`, `MSFT`).\n"
            "- Azioni italiane: aggiungi `.MI` (es. `STLAM.MI` per Stellantis, `ENI.MI`, `ISP.MI`).\n"
            "- Azioni tedesche: aggiungi `.DE` (es. `BMW.DE`, `SAP.DE`).\n"
            "- Azioni francesi: aggiungi `.PA` (es. `AIR.PA` per Airbus, `OR.PA` per L'OrÃ©al).\n"
            "- Azioni UK: aggiungi `.L` (es. `ULVR.L` per Unilever).\n"
            "- Crypto: di solito coppia con valuta, es. `BTC-USD`, `ETH-USD`.\n"
            "- Se hai dubbi, cerca prima il titolo su Yahoo Finance e copia il ticker esatto."
        )

    return {"mode": input_mode, "file": file, "manual": manual, "suffix": suffix, "btn": analyze_btn, "cfg": cfg}


# ==========================================
# 7. MAIN ORCHESTRATOR
# ==========================================
def main():
    st.title("ðŸ’Ž BurryInvestingPro")
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

    tab_f, tab_t, tab_q, tab_v, tab_p = st.tabs(["ðŸ“Š FONDAMENTALI", "ðŸ“‰ TECNICO", "âš›ï¸ QUANT", "âš–ï¸ VERDETTO", "ðŸ“ PORTAFOGLIO"])

    ticker = st.session_state.selected_ticker
    if ticker and st.session_state.batch_results is not None:
        row = st.session_state.batch_results[st.session_state.batch_results['Ticker'] == ticker].iloc[0]

        # --- TAB FONDAMENTALI ---
        with tab_f:
            st.info("ðŸ’¡ **Come leggere questa sezione:** Questa tabella rappresenta il motore dell'azienda. Cerca societÃ  con un ROIC (ritorno sul capitale investito) costantemente alto e un debito gestibile. Il Free Cash Flow Ã¨ il vero denaro prodotto dal business. Ricorda sempre: il prezzo Ã¨ quello che paghi, il valore Ã¨ quello che ottieni.")

            st.dataframe(st.session_state.batch_results.drop(columns=["_raw_data"]))

            st.markdown("---")
            st.markdown("<p style='text-align: center; color: gray;'>creato e sviluppato da Innovative Program[source: 1]</p>", unsafe_allow_html=True)

        # --- TAB TECNICO ---
        with tab_t:
            st.info("ðŸ’¡ **Come leggere il grafico:** Anche se preferiamo studiare il business, il prezzo ci dice come si muove il mercato. La linea blu (Media Mobile a 200 giorni) indica il trend di lungo periodo: se il prezzo Ã¨ sopra, la marea Ã¨ a nostro favore. Il grafico in basso (RSI) segnala se c'Ã¨ troppa euforia (valori sopra 70, attenzione) o troppo pessimismo (sotto 30, possibili occasioni).")

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
            st.info("ðŸ’¡ **Come interpretare i dati:** Qui lasciamo parlare la statistica. Lo **Sharpe Ratio** ci dice quanto rendimento stiamo ottenendo per ogni unitÃ  di rischio sopportata (piÃ¹ Ã¨ alto, meglio Ã¨). L'**Altman Z-Score** Ã¨ vitale per allontanarci dalle aziende a rischio bancarotta (sopra 3 Ã¨ ottimo). I grafici in basso (Monte Carlo) simulano gli scenari futuri in base alla volatilitÃ  storica, mostrandoci il rischio concreto (Drawdown) di perdite permanenti.")

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

                with st.expander("ðŸ“‰ Distribuzione rendimenti & rischio"):
                    st.write(f"CVaR 95%: {risk['CVaR_95'] * 100:.2f}%")
                    st.write(f"Skewness: {risk['Skew']:.2f} | Kurtosis: {risk['Kurt']:.2f}")

                    returns = df_tech['Close'].pct_change().dropna()
                    fig_r = go.Figure()
                    fig_r.add_trace(go.Histogram(x=returns, nbinsx=50, name="Rendimenti giornalieri"))
                    fig_r.update_layout(template="plotly_dark", bargap=0.05)
                    st.plotly_chart(fig_r, use_container_width=True)

                with st.expander("ðŸŽ² Simulazione Monte Carlo (rendimenti storici)"):
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
                        c9.metric("Scenario 5Â° percentile", f"{(p05 - 1) * 100:.1f}%")

                        x = np.arange(1, horizon_days + 1)
                        fig_mc = go.Figure()
                        fig_mc.add_trace(go.Scatter(
                            x=x, y=mc["q50"], name="Mediana",
                            line=dict(color="cyan")
                        ))
                        fig_mc.add_trace(go.Scatter(
                            x=x, y=mc["q95"], name="95Â° percentile",
                            line=dict(color="green"), opacity=0.3
                        ))
                        fig_mc.add_trace(go.Scatter(
                            x=x, y=mc["q05"], name="5Â° percentile",
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
            st.info("ðŸ’¡ **Come leggere il verdetto:** Questa Ã¨ la nostra sintesi razionale. Combina la soliditÃ  del business (Fondamentali), il momento (Tecnico) e le probabilitÃ  statistiche (Quant). Richiedi sempre un margine di sicurezza: investi solo quando il verdetto ti suggerisce che il rischio di perdere capitale in modo permanente Ã¨ bassissimo.")

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
                st.success("ðŸŸ¢ BUY: Fondamentali solidi e timing favorevole.")
            elif fund_pts >= 1 and z_safe:
                st.warning("ðŸŸ¡ HOLD: Azienda sicura ma attendere prezzi migliori.")
            else:
                st.error("ðŸ”´ SELL: Rischio finanziario o fondamentali scarsi.")

            if df_tech is not None and qm:
                smart_v = compute_smart_quant_score(row, score, qm, risk)
                st.metric("Smart Quant Score", f"{smart_v['SmartScore']:.1f}/100")
                st.write(
                    f"F: {smart_v['FundamentalScore']:.0f} | "
                    f"T: {smart_v['TechnicalScore']:.0f} | "
                    f"Q: {smart_v['QuantRiskScore']:.0f}"
                )

                if smart_v["SmartScore"] >= 70 and z_safe:
                    st.success("ðŸŸ¢ BUY (Quant): Vantaggio statistico e rischio controllato.")
                elif smart_v["SmartScore"] >= 50 and z_safe:
                    st.warning("ðŸŸ¡ HOLD (Quant): Setup discreto ma non eccezionale.")
                else:
                    st.error("ðŸ”´ NO TRADE (Quant): Vantaggio quantitativo debole o rischio elevato.")

            st.markdown("---")
            st.markdown("<p style='text-align: center; color: gray;'>creato e sviluppato da Innovative Program</p>", unsafe_allow_html=True)

        # --- TAB PORTAFOGLIO ---
        with tab_p:
            st.info("ðŸ’¡ **Come usare questa sezione:** La diversificazione sensata Ã¨ la protezione per il nostro capitale. Qui aggreghiamo i tuoi investimenti. Il grafico ti mostra la crescita combinata, mentre la matrice di correlazione in fondo Ã¨ cruciale: se le aziende che possiedi tendono a muoversi tutte nella stessa direzione contemporaneamente, non sei diversificato come pensi.")

            st.markdown(
                "### Portafoglio reale\n"
                "- Qui inserisci le **posizioni effettive** che hai in portafoglio.\n"
                "- Per ogni titolo, indica il **ticker corretto per il mercato** e l'**importo investito**.\n"
                "- Il sistema calcola automaticamente il **peso percentuale** di ogni posizione e le metriche "
                "quantitative del portafoglio (rendimento atteso, volatilitÃ , Sharpe, drawdown).\n\n"
                "**Come cercare il ticker giusto:**\n"
                "- Azioni USA: normalmente solo ticker (es. `AAPL`, `MSFT`).\n"
                "- Azioni italiane: aggiungi `.MI` (es. `STLAM.MI` per Stellantis a Milano, `ENI.MI`, `ISP.MI`).\n"
                "- Azioni tedesche: aggiungi `.DE` (es. `BMW.DE`, `SAP.DE`).\n"
                "- Azioni francesi: aggiungi `.PA` (es. `AIR.PA` per Airbus, `OR.PA` per L'OrÃ©al`).\n"
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

            if st.button("âž• Aggiungi ticker manuale al portafoglio"):
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

                    if col.button("ðŸ—‘ Rimuovi", key=f"remove_{t}"):
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

                if st.button("ðŸ“Š Calcola pesi e analisi del portafoglio"):
                    positive_holdings = {t: a for t, a in holdings.items() if a > 0}
                    if not positive_holdings:
                        st.error("Imposta un importo > 0 almeno per un titolo.")
                    else:
                        tot = sum(positive_holdings.values())
                        weights_pct = {t: a / tot * 100.0 for t, a in positive_holdings.items()}

                        built = build_portfolio_returns(list(positive_holdings.keys()), weights_pct)
                        if built is None:
                            st.error("Impossibile costruire la serie dei rendimenti (dati insufficienti per uno o piÃ¹ ticker).")
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
                            cpv.metric("VolatilitÃ  annua", f"{pm['AnnVol'] * 100:.2f}%")
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


# if __name__ == '__main__':
#     main()


# ==========================================
# ESTENSIONI AGGIUNTIVE V2 - SOLO APPEND
# ==========================================

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        if isinstance(value, str) and not value.strip():
            return float(default)
        v = float(value)
        if np.isnan(v) or np.isinf(v):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def get_last_price_from_symbol(symbol: str) -> Optional[float]:
    try:
        df = get_technical_data(symbol)
        if df is not None and not df.empty and 'Close' in df.columns:
            return _safe_float(df['Close'].dropna().iloc[-1], np.nan)
    except Exception:
        pass
    try:
        raw = get_fundamental_data(symbol)
        if raw and 'info' in raw:
            info = raw['info']
            p = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
            return _safe_float(p, np.nan)
    except Exception:
        pass
    return None


def discounted_cash_flow_valuation(fund: FundamentalMetrics, years: int = 10, growth: float = 0.05, discount_rate: float = 0.10, terminal_growth: float = 0.02) -> Dict[str, float]:
    info = fund.raw_data.get('info', {}) if isinstance(fund.raw_data, dict) else {}
    fcf = max(_safe_float(fund.fcf, 0.0), 0.0)
    shares = info.get('sharesOutstanding') or info.get('impliedSharesOutstanding') or 0
    shares = _safe_float(shares, 0.0)
    if shares <= 0 or fcf <= 0:
        return {'fair_value': np.nan, 'mos_pct': np.nan, 'fair_low': np.nan, 'fair_high': np.nan}

    def _dcf_once(g: float, dr: float, tg: float) -> float:
        cashflows = []
        for t in range(1, years + 1):
            fcf_t = fcf * ((1 + g) ** t)
            cashflows.append(fcf_t / ((1 + dr) ** t))
        spread = max(dr - tg, 0.0001)
        terminal_fcf = fcf * ((1 + g) ** years) * (1 + tg)
        terminal_value = terminal_fcf / spread
        terminal_pv = terminal_value / ((1 + dr) ** years)
        equity_value = sum(cashflows) + terminal_pv
        return equity_value / shares

    base = _dcf_once(growth, discount_rate, terminal_growth)
    low = _dcf_once(max(growth - 0.03, 0.0), discount_rate + 0.02, max(terminal_growth - 0.01, 0.0))
    high = _dcf_once(growth + 0.03, max(discount_rate - 0.02, terminal_growth + 0.01), terminal_growth + 0.01)
    mos = ((base - fund.price) / base * 100.0) if base and not np.isnan(base) else np.nan
    return {'fair_value': float(base), 'mos_pct': float(mos), 'fair_low': float(low), 'fair_high': float(high)}


def dividend_discount_model(info: Dict[str, Any], required_return: float = 0.10, growth: float = 0.03) -> Optional[float]:
    try:
        div_rate = info.get('dividendRate') or 0.0
        div_rate = _safe_float(div_rate, 0.0)
        if div_rate <= 0 or required_return <= growth:
            return None
        fair = (div_rate * (1 + growth)) / (required_return - growth)
        return float(fair)
    except Exception:
        return None


def build_valuation_summary_row(fund: FundamentalMetrics, dcf_res: Dict[str, float], ddm_fair: Optional[float]) -> Dict[str, Any]:
    return {
        'Ticker': fund.ticker,
        'Prezzo attuale': _safe_float(fund.price, np.nan),
        'DCF Fair Value': dcf_res.get('fair_value', np.nan),
        'DCF Range Low': dcf_res.get('fair_low', np.nan),
        'DCF Range High': dcf_res.get('fair_high', np.nan),
        'DCF MOS %': dcf_res.get('mos_pct', np.nan),
        'DDM Fair Value': ddm_fair if ddm_fair is not None else np.nan,
    }


def calculate_regime_statistics(returns: pd.Series, lookback: int = 63) -> Dict[str, Any]:
    if returns is None or returns.empty or len(returns) < lookback:
        return {'Regime': 'N/D', 'Vol ann.': np.nan, 'Skew': np.nan, 'Kurt': np.nan}
    w = returns.tail(lookback)
    vol_ann = _safe_float(w.std() * np.sqrt(TRADING_DAYS_YEAR), np.nan)
    skew = _safe_float(w.skew(), np.nan)
    kurt = _safe_float(w.kurt(), np.nan)
    regime = 'Normale'
    if (not np.isnan(vol_ann) and vol_ann > 0.30) or (not np.isnan(skew) and skew < -1.0):
        regime = 'Stress'
    elif (not np.isnan(vol_ann) and vol_ann < 0.15) and (not np.isnan(skew) and abs(skew) < 0.5):
        regime = 'Calmo'
    return {'Regime': regime, 'Vol ann.': vol_ann, 'Skew': skew, 'Kurt': kurt}


@st.cache_data(ttl=900, show_spinner=False)
def get_benchmark_returns(symbol: str = '^GSPC', period: str = '2y', interval: str = '1d') -> Optional[pd.Series]:
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        rets = df['Close'].pct_change().dropna()
        rets.name = symbol
        return rets
    except Exception:
        return None


def estimate_beta_and_alpha(asset_returns: pd.Series, benchmark_returns: pd.Series) -> Dict[str, float]:
    try:
        df = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
        if df.empty:
            return {'Beta': np.nan, 'Alpha ann.': np.nan, 'RÂ²': np.nan}
        x = df.iloc[:, 1].values.reshape(-1, 1)
        y = df.iloc[:, 0].values
        model = LinearRegression().fit(x, y)
        beta = float(model.coef_[0])
        alpha_daily = float(model.intercept_)
        alpha_ann = alpha_daily * TRADING_DAYS_YEAR
        r2 = float(model.score(x, y))
        return {'Beta': beta, 'Alpha ann.': alpha_ann, 'RÂ²': r2}
    except Exception:
        return {'Beta': np.nan, 'Alpha ann.': np.nan, 'RÂ²': np.nan}


def monte_carlo_risk_bands(mc_result: Dict[str, Any]) -> Dict[str, float]:
    final_values = None if mc_result is None else mc_result.get('final_distribution')
    if final_values is None:
        return {'Prob. loss >10%': np.nan, 'Prob. loss >20%': np.nan, 'Prob. loss >30%': np.nan, 'Mean return %': np.nan, 'CVaR 20% %': np.nan}
    fv = np.asarray(final_values)
    if fv.size == 0:
        return {'Prob. loss >10%': np.nan, 'Prob. loss >20%': np.nan, 'Prob. loss >30%': np.nan, 'Mean return %': np.nan, 'CVaR 20% %': np.nan}
    ret = fv - 1.0
    tail = ret[ret <= -0.20]
    return {
        'Prob. loss >10%': float((ret <= -0.10).mean() * 100.0),
        'Prob. loss >20%': float((ret <= -0.20).mean() * 100.0),
        'Prob. loss >30%': float((ret <= -0.30).mean() * 100.0),
        'Mean return %': float(ret.mean() * 100.0),
        'CVaR 20% %': float(tail.mean() * 100.0) if tail.size else np.nan,
    }


def compute_drawdown_distribution(returns: pd.Series) -> Dict[str, Any]:
    if returns is None or returns.empty:
        return {'drawdowns': pd.DataFrame(columns=['start', 'end', 'depth', 'duration'])}
    equity = (1 + returns).cumprod()
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    episodes = []
    in_dd = False
    start = None
    min_depth = 0.0
    for idx, value in dd.items():
        if value < 0 and not in_dd:
            in_dd = True
            start = idx
            min_depth = value
        elif value < 0 and in_dd:
            min_depth = min(min_depth, value)
        elif value >= 0 and in_dd:
            end = idx
            duration = (end - start).days if hasattr(end, 'days') else 0
            episodes.append({'start': start, 'end': end, 'depth': float(min_depth), 'duration': duration})
            in_dd = False
            start = None
            min_depth = 0.0
    if in_dd and start is not None:
        end = dd.index[-1]
        duration = (end - start).days if hasattr(end, 'days') else 0
        episodes.append({'start': start, 'end': end, 'depth': float(min_depth), 'duration': duration})
    return {'drawdowns': pd.DataFrame(episodes)}


def compute_risk_contribution(df_rets: pd.DataFrame, weights_pct: Dict[str, float]) -> Optional[pd.DataFrame]:
    if df_rets is None or df_rets.empty:
        return None
    cols = df_rets.columns.tolist()
    w = np.array([weights_pct.get(t, 0.0) for t in cols], dtype=float) / 100.0
    if w.sum() <= 0:
        return None
    w = w / w.sum()
    cov = df_rets.cov().values * TRADING_DAYS_YEAR
    port_var = float(w @ cov @ w)
    if port_var <= 0:
        return None
    port_vol = np.sqrt(port_var)
    mrc = cov @ w / port_vol
    rc = w * mrc
    rc_pct = rc / port_vol * 100.0
    return pd.DataFrame({'Ticker': cols, 'Peso %': w * 100.0, 'Risk Contribution %': rc_pct})


def optimize_portfolio_min_variance(df_rets: pd.DataFrame) -> Optional[Dict[str, float]]:
    if df_rets is None or df_rets.empty:
        return None
    cov = df_rets.cov().values * TRADING_DAYS_YEAR
    n = cov.shape[0]
    ones = np.ones(n)
    try:
        inv_cov = np.linalg.inv(cov + np.eye(n) * 1e-8)
        w = inv_cov @ ones
        w = np.clip(w, 0, None)
        if w.sum() <= 0:
            w = np.ones(n)
        w = w / w.sum()
        return {t: float(w[i] * 100.0) for i, t in enumerate(df_rets.columns)}
    except Exception:
        return None


def optimize_portfolio_max_sharpe(df_rets: pd.DataFrame, risk_free: float = RISK_FREE_RATE) -> Optional[Dict[str, float]]:
    if df_rets is None or df_rets.empty:
        return None
    mu = df_rets.mean().values * TRADING_DAYS_YEAR
    cov = df_rets.cov().values * TRADING_DAYS_YEAR
    n = len(mu)
    ones = np.ones(n)
    try:
        inv_cov = np.linalg.inv(cov + np.eye(n) * 1e-8)
        excess = mu - risk_free * ones
        w = inv_cov @ excess
        w = np.clip(w, 0, None)
        if w.sum() <= 0:
            w = np.ones(n)
        w = w / w.sum()
        return {t: float(w[i] * 100.0) for i, t in enumerate(df_rets.columns)}
    except Exception:
        return None


def stress_test_portfolio(df_rets: pd.DataFrame, weights_pct: Dict[str, float], shocks: Dict[str, float]) -> Optional[pd.DataFrame]:
    if df_rets is None or df_rets.empty:
        return None
    rows = []
    total_impact = 0.0
    for t in df_rets.columns:
        w = _safe_float(weights_pct.get(t, 0.0), 0.0)
        shock = _safe_float(shocks.get(t, shocks.get('DEFAULT', 0.0)), 0.0)
        impact = w / 100.0 * shock / 100.0
        total_impact += impact
        rows.append({'Ticker': t, 'Peso %': w, 'Shock %': shock, 'Contributo impatto %': impact * 100.0})
    rows.append({'Ticker': 'PORTAFOGLIO', 'Peso %': 100.0, 'Shock %': np.nan, 'Contributo impatto %': total_impact * 100.0})
    return pd.DataFrame(rows)


def rank_batch_by_smart_score(batch_df: pd.DataFrame) -> pd.DataFrame:
    if batch_df is None or batch_df.empty:
        return pd.DataFrame()
    rows = []
    for _, row in batch_df.iterrows():
        ticker = row.get('Ticker')
        dftech = get_technical_data(ticker)
        if dftech is None or dftech.empty:
            continue
        dfcalc = calculate_technical_indicators(dftech)
        timing_score, reasons = calculate_timing_score(dfcalc, _safe_float(dfcalc['Close'].iloc[-1], 0.0))
        qm = calculate_quant_metrics(dftech, row.get('raw_data', {}))
        risk = calculate_risk_metrics(dftech)
        smart = compute_smart_quant_score(row, timing_score, qm, risk)
        rows.append({
            'Ticker': ticker,
            'Company Name': row.get('Company Name'),
            'ROIC': row.get('ROIC'),
            'PEG Ratio': row.get('PEG Ratio'),
            'P/E Ratio': row.get('P/E Ratio'),
            'Timing Score': timing_score,
            'Sharpe Ratio': qm.get('Sharpe Ratio'),
            'Altman Z-Score': qm.get('Altman Z-Score'),
            'Max Drawdown': risk.get('Max Drawdown'),
            'SmartScore': smart.get('SmartScore'),
            'Timing Reasons': ', '.join(reasons),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values('SmartScore', ascending=False)


def apply_fundamental_filters(batch_df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    if batch_df is None or batch_df.empty:
        return pd.DataFrame()
    df = batch_df.copy()
    min_roic = _safe_float(cfg.get('roic', 10.0), 10.0) / 100.0
    min_fcf = _safe_float(cfg.get('fcf', 0.0), 0.0)
    max_peg = _safe_float(cfg.get('peg', 1.5), 1.5)
    max_pe = _safe_float(cfg.get('pe', 25.0), 25.0)
    min_intcov = _safe_float(cfg.get('int_cov', 3.0), 3.0)
    perfect_only = bool(cfg.get('perfect_only', False))

    cond_roic = df['ROIC'].fillna(-np.inf) >= min_roic
    cond_fcf = df['Free Cash Flow'].fillna(-np.inf) >= min_fcf
    cond_peg = df['PEG Ratio'].fillna(np.inf) <= max_peg
    cond_pe = df['P/E Ratio'].fillna(np.inf) <= max_pe
    cond_int = df['Interest Coverage'].fillna(-np.inf) >= min_intcov

    df = df[cond_roic & cond_fcf & cond_int & (cond_peg | cond_pe)].copy()
    if perfect_only and not df.empty:
        df = df[cond_roic & cond_fcf & cond_peg & cond_pe & cond_int]
    return df


def dataframe_to_csv_download(df: pd.DataFrame, filename: str, label: str) -> None:
    if df is None or df.empty:
        return
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    st.download_button(label=label, data=csv_bytes, file_name=filename, mime='text/csv', use_container_width=True)


def init_extended_session_state() -> None:
    defaults = {
        'holdings_mode': {},
        'holding_units': {},
        'last_prices': {},
        'portfolio_analysis_cache': None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def resolve_position_market_value(ticker: str, amount_value: float, units_value: float) -> Tuple[float, float]:
    price = get_last_price_from_symbol(ticker)
    if price is not None and not np.isnan(price):
        st.session_state.last_prices[ticker] = float(price)
    else:
        price = _safe_float(st.session_state.last_prices.get(ticker), np.nan)
    market_value = _safe_float(amount_value, 0.0)
    if _safe_float(units_value, 0.0) > 0 and price is not None and not np.isnan(price):
        market_value = float(units_value) * float(price)
    return market_value, price if price is not None else np.nan


def render_extended_modules() -> None:
    init_extended_session_state()
    if st.session_state.get('batch_results') is None or st.session_state.batch_results is None or st.session_state.batch_results.empty:
        return
    ticker = st.session_state.get('selected_ticker')
    if not ticker:
        return
    row_df = st.session_state.batch_results[st.session_state.batch_results['Ticker'] == ticker]
    if row_df.empty:
        return
    row = row_df.iloc[0]

    st.markdown('---')
    st.subheader('Estensioni V2')
    ext_tabs = st.tabs(['Valutazione', 'Screener avanzato', 'Analisi portafoglio estesa'])

    with ext_tabs[0]:
        try:
            fund = FundamentalMetrics(
                ticker=row['Ticker'],
                company_name=row['Company Name'],
                price=_safe_float(row['Price'], 0.0),
                fcf=_safe_float(row['Free Cash Flow'], 0.0),
                roic=_safe_float(row['ROIC'], 0.0),
                peg_ratio=row['PEG Ratio'],
                peg_source=row['PEG Source'],
                pe_ratio=row['P/E Ratio'],
                interest_coverage=_safe_float(row['Interest Coverage'], 0.0),
                currency=row['Currency'],
                raw_data=row['raw_data'],
            )
            dcf = discounted_cash_flow_valuation(fund)
            info = fund.raw_data.get('info', {}) if isinstance(fund.raw_data, dict) else {}
            ddm = dividend_discount_model(info)
            val_df = pd.DataFrame([build_valuation_summary_row(fund, dcf, ddm)])
            st.dataframe(val_df, use_container_width=True)
        except Exception as e:
            st.warning(f'Valutazione non disponibile: {e}')

        dftech = get_technical_data(ticker)
        if dftech is not None and not dftech.empty:
            rets = dftech['Close'].pct_change().dropna()
            reg = calculate_regime_statistics(rets)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric('Regime', reg.get('Regime', 'N/D'))
            c2.metric('Vol ann.', f"{reg.get('Vol ann.', np.nan) * 100:.1f}%" if pd.notna(reg.get('Vol ann.', np.nan)) else 'N/D')
            c3.metric('Skew', f"{reg.get('Skew', np.nan):.2f}" if pd.notna(reg.get('Skew', np.nan)) else 'N/D')
            c4.metric('Kurt', f"{reg.get('Kurt', np.nan):.2f}" if pd.notna(reg.get('Kurt', np.nan)) else 'N/D')

            bench = get_benchmark_returns()
            beta_data = estimate_beta_and_alpha(rets, bench) if bench is not None else {'Beta': np.nan, 'Alpha ann.': np.nan, 'RÂ²': np.nan}
            b1, b2, b3 = st.columns(3)
            b1.metric('Beta vs S&P500', f"{beta_data['Beta']:.2f}" if pd.notna(beta_data['Beta']) else 'N/D')
            b2.metric('Alpha ann.', f"{beta_data['Alpha ann.'] * 100:.2f}%" if pd.notna(beta_data['Alpha ann.']) else 'N/D')
            b3.metric('RÂ² benchmark', f"{beta_data['RÂ²']:.2f}" if pd.notna(beta_data['RÂ²']) else 'N/D')

    with ext_tabs[1]:
        ranked = rank_batch_by_smart_score(st.session_state.batch_results)
        filtered = apply_fundamental_filters(st.session_state.batch_results, {'roic': 10.0, 'fcf': 0.0, 'peg': 1.5, 'pe': 25.0, 'int_cov': 3.0, 'perfect_only': False})
        st.markdown('### Ranking Smart Score')
        if ranked is not None and not ranked.empty:
            st.dataframe(ranked, use_container_width=True)
            dataframe_to_csv_download(ranked, 'ranking_smart_score.csv', 'Scarica ranking CSV')
        else:
            st.info('Nessun ranking disponibile al momento.')
        st.markdown('### Filtro fondamentali')
        if filtered is not None and not filtered.empty:
            st.dataframe(filtered.drop(columns=['raw_data'], errors='ignore'), use_container_width=True)
            dataframe_to_csv_download(filtered.drop(columns=['raw_data'], errors='ignore'), 'filtro_fondamentali.csv', 'Scarica filtro CSV')
        else:
            st.info('Nessun titolo passa i filtri attuali.')

    with ext_tabs[2]:
        st.info('Questa sezione aggiunge la possibilitÃ  di inserire importo oppure numero quote/azioni, senza rimuovere la logica base del portafoglio.')
        all_tickers_batch = []
        if st.session_state.batch_results is not None and not st.session_state.batch_results.empty:
            all_tickers_batch = st.session_state.batch_results['Ticker'].tolist()
        selected = st.multiselect('Titoli batch per analisi estesa', all_tickers_batch, default=st.session_state.get('portfolio_tickers', []), key='ext_portfolio_selected')
        manual_ticker_ext = st.text_input('Aggiungi ticker manuale per analisi estesa', key='manual_ticker_ext')
        if st.button('Aggiungi ticker esteso', key='btn_add_ticker_ext'):
            if manual_ticker_ext.strip():
                try:
                    tclean = sanitize_ticker(manual_ticker_ext)
                    if tclean not in st.session_state.portfolio_tickers:
                        st.session_state.portfolio_tickers.append(tclean)
                    st.success(f'Aggiunto {tclean}')
                except Exception as e:
                    st.error(str(e))
        portfolio_list = sorted(set(selected + st.session_state.get('portfolio_tickers', [])))
        if portfolio_list:
            rows_data = []
            cols = st.columns(3)
            for i, t in enumerate(portfolio_list):
                col = cols[i % 3]
                mode_default = st.session_state.holdings_mode.get(t, 'Importo')
                mode = col.selectbox(f'{t} - ModalitÃ ', ['Importo', 'Quote/Azioni'], index=['Importo', 'Quote/Azioni'].index(mode_default) if mode_default in ['Importo', 'Quote/Azioni'] else 0, key=f'holdingmode_{t}')
                st.session_state.holdings_mode[t] = mode

                current_amount = _safe_float(st.session_state.holdings.get(t, 0.0), 0.0)
                current_units = _safe_float(st.session_state.holding_units.get(t, 0.0), 0.0)
                amount_value = current_amount
                units_value = current_units
                if mode == 'Importo':
                    amount_value = col.number_input(f'{t} - Importo investito', min_value=0.0, value=current_amount, step=100.0, key=f'ext_amount_{t}')
                else:
                    units_value = col.number_input(f'{t} - Numero quote/azioni', min_value=0.0, value=current_units, step=1.0, key=f'ext_units_{t}')

                cur_default = st.session_state.holdings_currency.get(t, 'USD')
                cur = col.selectbox(f'{t} - Valuta', ['USD', 'EUR'], index=['USD', 'EUR'].index(cur_default) if cur_default in ['USD', 'EUR'] else 0, key=f'ext_currency_{t}')
                st.session_state.holdings_currency[t] = cur
                st.session_state.holdings[t] = amount_value
                st.session_state.holding_units[t] = units_value

                mv, px = resolve_position_market_value(t, amount_value, units_value)
                rows_data.append({'Ticker': t, 'ModalitÃ ': mode, 'Importo inserito': amount_value, 'Quote/Azioni': units_value, 'Prezzo stimato': px, 'Market Value stimato': mv, 'Valuta': cur})

            df_positions = pd.DataFrame(rows_data)
            st.markdown('### Dati posizioni')
            st.dataframe(df_positions, use_container_width=True)

            if st.button('Calcola analisi portafoglio estesa', key='calc_ext_portfolio'):
                positive_positions = {r['Ticker']: _safe_float(r['Market Value stimato'], 0.0) for r in rows_data if _safe_float(r['Market Value stimato'], 0.0) > 0}
                if not positive_positions:
                    st.error('Inserisci almeno un importo o un numero quote/azioni valido.')
                else:
                    total_mv = sum(positive_positions.values())
                    weights_pct = {k: v / total_mv * 100.0 for k, v in positive_positions.items()}
                    built = build_portfolio_returns(list(positive_positions.keys()), weights_pct)
                    if built is None:
                        st.error('Impossibile costruire la serie rendimenti del portafoglio esteso.')
                    else:
                        df_rets, port_ret = built
                        pm = calculate_portfolio_metrics(port_ret)
                        rc = compute_risk_contribution(df_rets, weights_pct)
                        w_minvar = optimize_portfolio_min_variance(df_rets)
                        w_maxsh = optimize_portfolio_max_sharpe(df_rets)
                        stress = stress_test_portfolio(df_rets, weights_pct, {'DEFAULT': -15.0})
                        dd = compute_drawdown_distribution(port_ret)

                        wdf = pd.DataFrame({'Ticker': list(weights_pct.keys()), 'Peso %': list(weights_pct.values()), 'Market Value': [positive_positions[t] for t in weights_pct.keys()]})
                        st.markdown('### Pesi portafoglio esteso')
                        st.dataframe(wdf, use_container_width=True)

                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric('Rendimento annuo atteso', f"{pm['AnnRet'] * 100:.2f}%")
                        c2.metric('VolatilitÃ  annua', f"{pm['AnnVol'] * 100:.2f}%")
                        c3.metric('Sharpe', f"{pm['Sharpe']:.2f}")
                        c4.metric('Max Drawdown', f"{pm['MaxDD'] * 100:.2f}%")

                        equity_p = (1 + port_ret).cumprod()
                        figp = go.Figure()
                        figp.add_trace(go.Scatter(x=equity_p.index, y=equity_p.values, name='Equity portafoglio esteso'))
                        figp.update_layout(template='plotly_dark', height=400, xaxis_title='Data', yaxis_title='Equity normalizzata')
                        st.plotly_chart(figp, use_container_width=True)

                        if rc is not None and not rc.empty:
                            st.markdown('### Risk contribution')
                            st.dataframe(rc, use_container_width=True)
                        if w_minvar:
                            st.markdown('### Pesi suggeriti Min Variance')
                            st.dataframe(pd.DataFrame({'Ticker': list(w_minvar.keys()), 'Peso %': list(w_minvar.values())}), use_container_width=True)
                        if w_maxsh:
                            st.markdown('### Pesi suggeriti Max Sharpe')
                            st.dataframe(pd.DataFrame({'Ticker': list(w_maxsh.keys()), 'Peso %': list(w_maxsh.values())}), use_container_width=True)
                        if stress is not None and not stress.empty:
                            st.markdown('### Stress test (-15% default)')
                            st.dataframe(stress, use_container_width=True)
                        if isinstance(dd, dict) and 'drawdowns' in dd and not dd['drawdowns'].empty:
                            st.markdown('### Episodi di drawdown')
                            st.dataframe(dd['drawdowns'], use_container_width=True)
        else:
            st.info('Aggiungi o seleziona almeno un ticker per attivare l\'analisi portafoglio estesa.')


def main_v2_wrapper():
    try:
        init_extended_session_state()
    except Exception:
        pass
    try:
        main()
    except NameError:
        pass
    except Exception as e:
        st.error(f'Errore nella main originale: {e}')
    try:
        render_extended_modules()
    except Exception as e:
        st.warning(f'Estensioni V2 non completamente disponibili: {e}')


if __name__ == '__main__':
    main_v2_wrapper()