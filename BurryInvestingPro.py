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
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
from sklearn.linear_model import LinearRegression

# 1. Configurazione Pagina (deve comparire una sola volta e come prima istruzione Streamlit utile)
st.set_page_config(
    page_title="Burry Investing Pro",
    page_icon="📈",
    layout="wide"
)

# 2. Link al Manifesto PWA
st.markdown(
    '<link rel="manifest" href="https://raw.githubusercontent.com/Innovativeprogram/burry-investing-pro/main/manifest.json">',
    unsafe_allow_html=True
)

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
    if not re.match(r"^[A-Z0-9-.^=]+$", clean):
        raise ValueError(f"Ticker contiene caratteri non validi: {clean}")
    return clean

def normalize_ticker(ticker: str, suffix: str) -> str:
    clean_ticker = sanitize_ticker(ticker)
    if "-" in clean_ticker or clean_ticker.startswith("^"):
        return clean_ticker
    clean_suffix = str(suffix).strip().upper()
    if clean_suffix and not re.match(r"^.[A-Z]+$", clean_suffix):
        clean_suffix = ""
    if clean_suffix and not clean_ticker.endswith(clean_suffix):
        return f"{clean_ticker}{clean_suffix}"
    return clean_ticker

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return float(value)
    except Exception:
        return default

def format_pct(x: Any) -> str:
    try:
        if pd.isna(x):
            return "N/A"
        return f"{float(x) * 100:.2f}%"
    except Exception:
        return "N/A"

def format_pct_100(x: Any) -> str:
    try:
        if pd.isna(x):
            return "N/A"
        return f"{float(x):.2f}%"
    except Exception:
        return "N/A"

def infer_position_value(amount: float, quantity: float, avg_cost: float) -> float:
    if amount > 0:
        return float(amount)
    if quantity > 0 and avg_cost > 0:
        return float(quantity * avg_cost)
    return 0.0

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

        fin = raw_data["financials"]
        bs = raw_data["balance_sheet"]
        cf = raw_data["cashflow"]

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
            if pretax_inc and pretax_inc > 0:
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
        int_cov = float(ebit / abs(int_exp)) if int_exp not in [0, None] else SAFE_INTEREST_COVERAGE

        return FundamentalMetrics(
            ticker=raw_data["symbol"],
            company_name=info.get('longName', raw_data["symbol"]),
            price=float(info.get('currentPrice') or info.get('regularMarketPrice') or 0.0),
            fcf=fcf,
            roic=roic,
            peg_ratio=float(peg) if peg is not None else None,
            peg_source=peg_src,
            pe_ratio=float(pe) if pe is not None else None,
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
        df = yf.download(symbol, period="2y", interval="1d", progress=False, auto_adjust=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in df.columns:
                return None
        return df if len(df) >= 120 else df
    except Exception as e:
        logger.error(f"Errore get_technical_data {symbol}: {e}")
        return None

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data['SMA_50'] = ta.sma(data['Close'], length=50)
    data['SMA_200'] = ta.sma(data['Close'], length=200)
    data['RSI'] = ta.rsi(data['Close'], length=14)
    bb = ta.bbands(data['Close'], length=20, std=2)
    if bb is not None and not bb.empty:
        lower_cols = [c for c in bb.columns if "BBL_" in c]
        upper_cols = [c for c in bb.columns if "BBU_" in c]
        if lower_cols:
            data['BB_Lower'] = bb[lower_cols[0]]
        if upper_cols:
            data['BB_Upper'] = bb[upper_cols[0]]
    return data

def calculate_timing_score(data: pd.DataFrame, current_price: float) -> Tuple[int, List[str]]:
    score, reasons = 0, []
    last_row = data.iloc[-1]

    sma200 = last_row.get('SMA_200', np.nan)
    sma50 = last_row.get('SMA_50', np.nan)
    rsi = last_row.get('RSI', np.nan)
    bb_lower = last_row.get('BB_Lower', np.nan)

    if pd.notna(sma200):
        if current_price > sma200:
            score += 30
            reasons.append("✅ Trend Rialzista (Sopra SMA 200)")
        else:
            reasons.append("⚠️ Trend Ribassista (Sotto SMA 200)")

    if pd.notna(sma50) and pd.notna(sma200):
        if sma50 > sma200:
            score += 20
            reasons.append("✅ Momentum positivo (SMA 50 > SMA 200)")

    if pd.notna(rsi):
        if rsi < 30:
            score += 30
            reasons.append("✅ Ipervenduto (RSI < 30)")
        elif rsi > 70:
            score -= 10
            reasons.append("🛑 Ipercomprato (RSI > 70)")
        elif 45 <= rsi <= 60:
            score += 10
            reasons.append("✅ RSI equilibrato")

    if pd.notna(bb_lower) and bb_lower > 0:
        if current_price <= bb_lower * 1.02:
            score += 20
            reasons.append("✅ Prezzo su Banda Bollinger Inferiore")

    score = int(np.clip(score, 0, 100))
    return score, reasons

# ==========================================
# 5. MOTORE QUANTISTICO
# ==========================================
def calculate_quant_metrics(df: pd.DataFrame, fund_data: Dict[str, Any]) -> Dict[str, Any]:
    returns = df['Close'].pct_change().dropna()
    if returns.empty:
        return {
            "Sharpe Ratio": np.nan,
            "Annual Volatility": np.nan,
            "R-Squared": np.nan,
            "Altman Z-Score": "N/A",
            "Price Percentile": np.nan,
            "Trend Slope": np.nan
        }

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
            wc = (bs.loc['Current Assets'].iloc[0] - bs.loc['Current Liabilities'].iloc[0]) if (
                'Current Assets' in bs.index and 'Current Liabilities' in bs.index
            ) else 0
            re = bs.loc['Retained Earnings'].iloc[0] if 'Retained Earnings' in bs.index else 0
            ebit = fin.loc['EBIT'].iloc[0] if 'EBIT' in fin.index else 0
            mc = info.get('marketCap', 1)
            tl = bs.loc['Total Liabilities Net Minority Interest'].iloc[0] if 'Total Liabilities Net Minority Interest' in bs.index else 1
            rev = info.get('totalRevenue', 0)
            if ta_val and tl:
                z_score = (1.2 * (wc / ta_val)) + (1.4 * (re / ta_val)) + (3.3 * (ebit / ta_val)) + (0.6 * (mc / tl)) + (1.0 * (rev / ta_val))
        except Exception:
            pass

    return {
        "Sharpe Ratio": float(sharpe),
        "Annual Volatility": float(vol),
        "R-Squared": float(r_sq),
        "Altman Z-Score": z_score,
        "Price Percentile": float((df['Close'] < df['Close'].iloc[-1]).mean() * 100),
        "Trend Slope": float(model.coef_[0][0])
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

    return {
        "Max Drawdown": float(max_dd),
        "CAGR": float(cagr),
        "VaR_95": float(var_95),
        "CVaR_95": float(cvar_95),
        "Skew": float(returns.skew()),
        "Kurt": float(returns.kurt())
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
    if peg is not None and not pd.isna(peg) and peg > 0:
        if peg <= 1:
            f_score += 30.0
        elif peg <= 2:
            f_score += 15.0

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

    if pd.notna(sharpe):
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
# 5.F VALUTAZIONE INTRINSECA (DCF & DDM)
# ==========================================
def discounted_cash_flow_valuation(
    fund: FundamentalMetrics,
    years: int = 10,
    growth: float = 0.05,
    discount_rate: float = 0.10,
    terminal_growth: float = 0.02
) -> Dict[str, float]:
    info = fund.raw_data.get("info", {})
    fcf = max(float(fund.fcf), 0.0)
    shares = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding") or 0

    if shares <= 0 or fcf <= 0:
        return {
            "fair_value": np.nan,
            "mos_pct": np.nan,
            "fair_low": np.nan,
            "fair_high": np.nan,
        }

    def _dcf_once(g, dr):
        cash_flows = []
        for t in range(1, years + 1):
            fcf_t = fcf * ((1 + g) ** t)
            cash_flows.append(fcf_t / ((1 + dr) ** t))
        spread = max(dr - terminal_growth, 1e-4)
        terminal_fcf = fcf * ((1 + g) ** years) * (1 + terminal_growth)
        terminal_value = terminal_fcf / spread
        terminal_pv = terminal_value / ((1 + dr) ** years)
        equity_value = sum(cash_flows) + terminal_pv
        return equity_value / shares

    fair_base = _dcf_once(growth, discount_rate)
    fair_low = _dcf_once(max(growth - 0.03, 0.0), discount_rate + 0.02)
    fair_high = _dcf_once(growth + 0.03, max(discount_rate - 0.02, 0.03))
    mos_pct = ((fair_base - fund.price) / fair_base * 100.0) if fair_base > 0 else np.nan

    return {
        "fair_value": float(fair_base),
        "mos_pct": float(mos_pct),
        "fair_low": float(fair_low),
        "fair_high": float(fair_high),
    }

def dividend_discount_model(
    info: Dict[str, Any],
    required_return: float = 0.10,
    growth: float = 0.03
) -> Optional[float]:
    div_rate = info.get("dividendRate")
    if not div_rate or required_return <= growth:
        return None
    try:
        fair_value = div_rate * (1 + growth) / (required_return - growth)
        return float(fair_value)
    except Exception:
        return None

def build_valuation_summary_row(
    fund: FundamentalMetrics,
    dcf_res: Dict[str, float],
    ddm_fair: Optional[float]
) -> Dict[str, Any]:
    return {
        "Ticker": fund.ticker,
        "Prezzo attuale": fund.price,
        "DCF Fair Value": dcf_res.get("fair_value"),
        "DCF Range Low": dcf_res.get("fair_low"),
        "DCF Range High": dcf_res.get("fair_high"),
        "DCF MOS %": dcf_res.get("mos_pct"),
        "DDM Fair Value": ddm_fair,
    }

# ==========================================
# 5.G QUANT AVANZATO
# ==========================================
def calculate_regime_statistics(
    returns: pd.Series,
    lookback: int = 63
) -> Dict[str, Any]:
    if returns is None or returns.empty or len(returns) < min(lookback, 20):
        return {
            "Regime": "N/D",
            "Vol_ann": np.nan,
            "Skew": np.nan,
            "Kurt": np.nan,
        }

    window = returns.tail(lookback)
    vol_ann = window.std() * np.sqrt(TRADING_DAYS_YEAR)
    skew = window.skew()
    kurt = window.kurt()

    regime = "Normale"
    if vol_ann > 0.30 or skew < -1.0:
        regime = "Stress"
    elif vol_ann < 0.15 and abs(skew) < 0.5:
        regime = "Calmo"

    return {
        "Regime": regime,
        "Vol_ann": float(vol_ann),
        "Skew": float(skew),
        "Kurt": float(kurt),
    }

@st.cache_data(ttl=900, show_spinner=False)
def get_benchmark_returns(
    symbol: str = "^GSPC",
    period: str = "2y",
    interval: str = "1d"
) -> Optional[pd.Series]:
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        rets = df["Close"].pct_change().dropna()
        rets.name = symbol
        return rets
    except Exception:
        return None

def estimate_beta_and_alpha(
    asset_rets: pd.Series,
    bench_rets: pd.Series
) -> Dict[str, float]:
    df = pd.concat([asset_rets, bench_rets], axis=1).dropna()
    if df.empty:
        return {"Beta": np.nan, "Alpha_ann": np.nan, "R2": np.nan}

    x = df.iloc[:, 1].values.reshape(-1, 1)
    y = df.iloc[:, 0].values
    model = LinearRegression().fit(x, y)
    beta = float(model.coef_[0])
    alpha_daily = float(model.intercept_)
    alpha_ann = alpha_daily * TRADING_DAYS_YEAR
    r2 = float(model.score(x, y))

    return {"Beta": beta, "Alpha_ann": alpha_ann, "R2": r2}

def monte_carlo_risk_bands(final_values: np.ndarray) -> Dict[str, float]:
    if final_values is None or len(final_values) == 0:
        return {
            "Prob_loss_10": np.nan,
            "Prob_loss_20": np.nan,
            "Prob_loss_30": np.nan,
            "Mean_ret": np.nan,
            "CVaR_20": np.nan,
        }

    final_values = np.asarray(final_values)
    final_ret = final_values - 1.0

    prob_loss_10 = float((final_ret < -0.10).mean() * 100.0)
    prob_loss_20 = float((final_ret < -0.20).mean() * 100.0)
    prob_loss_30 = float((final_ret < -0.30).mean() * 100.0)
    mean_ret = float(final_ret.mean() * 100.0)

    tail = final_ret[final_ret < -0.20]
    cvar_20 = float(tail.mean() * 100.0) if tail.size > 0 else np.nan

    return {
        "Prob_loss_10": prob_loss_10,
        "Prob_loss_20": prob_loss_20,
        "Prob_loss_30": prob_loss_30,
        "Mean_ret": mean_ret,
        "CVaR_20": cvar_20,
    }

# ==========================================
# 5.H PORTAFOGLIO AVANZATO
# ==========================================
def compute_risk_contribution(
    df_rets: pd.DataFrame,
    weights_pct: Dict[str, float]
) -> Optional[pd.DataFrame]:
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

    df_rc = pd.DataFrame({
        "Ticker": cols,
        "Peso %": w * 100.0,
        "Risk Contribution %": rc_pct,
    })
    return df_rc

def optimize_portfolio_min_variance(df_rets: pd.DataFrame) -> Optional[Dict[str, float]]:
    if df_rets is None or df_rets.empty:
        return None

    cov = df_rets.cov().values * TRADING_DAYS_YEAR
    n = cov.shape[0]
    ones = np.ones(n)

    try:
        inv_cov = np.linalg.inv(cov + np.eye(n) * 1e-8)
        w_unnorm = inv_cov @ ones
        w = np.clip(w_unnorm, 0, None)
        if w.sum() == 0:
            w = np.ones(n)
        w = w / w.sum()
        return {t: float(w[i] * 100.0) for i, t in enumerate(df_rets.columns)}
    except np.linalg.LinAlgError:
        return None

def optimize_portfolio_max_sharpe(
    df_rets: pd.DataFrame,
    risk_free: float = RISK_FREE_RATE
) -> Optional[Dict[str, float]]:
    if df_rets is None or df_rets.empty:
        return None

    mu = df_rets.mean().values * TRADING_DAYS_YEAR
    cov = df_rets.cov().values * TRADING_DAYS_YEAR
    n = cov.shape[0]
    rf_vec = np.full(n, risk_free)

    try:
        inv_cov = np.linalg.inv(cov + np.eye(n) * 1e-8)
        excess = mu - rf_vec
        w_unnorm = inv_cov @ excess
        w = np.clip(w_unnorm, 0, None)
        if w.sum() == 0:
            w = np.ones(n)
        w = w / w.sum()
        return {t: float(w[i] * 100.0) for i, t in enumerate(df_rets.columns)}
    except np.linalg.LinAlgError:
        return None

def stress_test_portfolio(
    df_rets: pd.DataFrame,
    weights_pct: Dict[str, float],
    shock_pct: float = -0.20
) -> Optional[float]:
    if df_rets is None or df_rets.empty:
        return None

    cols = df_rets.columns.tolist()
    w = np.array([weights_pct.get(t, 0.0) for t in cols], dtype=float) / 100.0
    if w.sum() <= 0:
        return None
    w = w / w.sum()

    mu = df_rets.mean().values * TRADING_DAYS_YEAR
    shocked_mu = mu + shock_pct
    port_ret_shocked = float(np.dot(w, shocked_mu))
    return port_ret_shocked

# ==========================================
# 5.I BATCH SCREENER
# ==========================================
def apply_fundamental_filters(
    batch_df: pd.DataFrame,
    cfg: Dict[str, Any]
) -> pd.DataFrame:
    df = batch_df.copy()
    if "ROIC" in df.columns:
        df = df[df["ROIC"] >= cfg["roic"] / 100.0]
    if "Free Cash Flow" in df.columns:
        df = df[df["Free Cash Flow"] >= cfg["fcf"]]
    if "PEG Ratio" in df.columns and cfg["peg"] > 0:
        df = df[(df["PEG Ratio"].isna()) | (df["PEG Ratio"] <= cfg["peg"])]
    if "P/E Ratio" in df.columns and cfg["pe"] > 0:
        df = df[(df["P/E Ratio"].isna()) | (df["P/E Ratio"] <= cfg["pe"])]
    if "Interest Coverage" in df.columns:
        df = df[df["Interest Coverage"] >= cfg["int_cov"]]

    if cfg.get("perfect_only", False) and "PEG Ratio" in df.columns and "P/E Ratio" in df.columns:
        df = df[(df["PEG Ratio"].notna()) & (df["P/E Ratio"].notna())]

    return df

def rank_batch_by_smart_score(batch_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in batch_df.iterrows():
        try:
            raw = row["_raw_data"]
            df_tech = get_technical_data(row["Ticker"])
            if df_tech is None or df_tech.empty:
                continue
            df_calc = calculate_technical_indicators(df_tech)
            current_price = safe_float(df_calc["Close"].iloc[-1], 0.0)
            timing_score, _ = calculate_timing_score(df_calc, current_price)
            qm = calculate_quant_metrics(df_tech, raw)
            risk = calculate_risk_metrics(df_tech)
            smart = compute_smart_quant_score(row, timing_score, qm, risk)
            new_row = dict(row)
            new_row["SmartScore"] = smart["SmartScore"]
            new_row["FundamentalScore"] = smart["FundamentalScore"]
            new_row["TechnicalScore"] = smart["TechnicalScore"]
            new_row["QuantRiskScore"] = smart["QuantRiskScore"]
            rows.append(new_row)
        except Exception as e:
            logger.warning(f"Errore SmartScore batch su {row.get('Ticker', 'N/D')}: {e}")

    if not rows:
        return pd.DataFrame()

    df_out = pd.DataFrame(rows)
    if "SmartScore" in df_out.columns:
        df_out = df_out.sort_values("SmartScore", ascending=False)
    return df_out

# ==========================================
# 5.L FUNZIONI PORTAFOGLIO ESTESE QUANTITA'/QUOTE
# ==========================================
def compute_portfolio_position_table(
    portfolio_list: List[str],
    holdings_amount: Dict[str, float],
    holdings_qty: Dict[str, float],
    holdings_avg_cost: Dict[str, float],
    holdings_currency: Dict[str, str]
) -> pd.DataFrame:
    rows = []
    for t in portfolio_list:
        amt = safe_float(holdings_amount.get(t, 0.0))
        qty = safe_float(holdings_qty.get(t, 0.0))
        avg_cost = safe_float(holdings_avg_cost.get(t, 0.0))
        inferred_value = infer_position_value(amt, qty, avg_cost)
        rows.append({
            "Ticker": t,
            "Importo investito": amt,
            "Quantità (quote/azioni)": qty,
            "Prezzo medio di carico": avg_cost,
            "Controvalore usato": inferred_value,
            "Valuta": holdings_currency.get(t, "USD")
        })
    return pd.DataFrame(rows)

def get_latest_prices_for_portfolio(tickers: List[str]) -> Dict[str, float]:
    prices = {}
    for t in tickers:
        try:
            raw = get_fundamental_data(t)
            if raw and "info" in raw:
                info = raw["info"]
                px = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
                prices[t] = safe_float(px, np.nan)
            else:
                prices[t] = np.nan
        except Exception:
            prices[t] = np.nan
    return prices

def build_market_value_table(
    portfolio_df: pd.DataFrame,
    latest_prices: Dict[str, float]
) -> pd.DataFrame:
    df = portfolio_df.copy()
    df["Prezzo attuale"] = df["Ticker"].map(latest_prices)
    df["Valore di mercato stimato"] = np.where(
        (df["Quantità (quote/azioni)"] > 0) & df["Prezzo attuale"].notna(),
        df["Quantità (quote/azioni)"] * df["Prezzo attuale"],
        df["Controvalore usato"]
    )
    df["P/L stimato"] = np.where(
        df["Importo investito"] > 0,
        df["Valore di mercato stimato"] - df["Importo investito"],
        np.nan
    )
    df["P/L % stimato"] = np.where(
        df["Importo investito"] > 0,
        (df["Valore di mercato stimato"] / df["Importo investito"] - 1.0) * 100.0,
        np.nan
    )
    return df

# ==========================================
# 6. UI: SIDEBAR
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
    market = st.sidebar.selectbox(
        "Borsa:",
        ["USA", "Italia (.MI)", "Germania (.DE)", "Francia (.PA)", "GB (.L)", "Crypto", "Custom"]
    )

    if "Italia" in market:
        suffix = ".MI"
    elif "Germania" in market:
        suffix = ".DE"
    elif "Francia" in market:
        suffix = ".PA"
    elif "GB" in market:
        suffix = ".L"
    else:
        suffix = ""

    analyze_btn = st.sidebar.button("🚀 Avvia Analisi", use_container_width=True)

    with st.sidebar.expander("⚙️ Parametri Fondamentali"):
        cfg = {
            "roic": st.number_input("Min ROIC %", min_value=0.0, value=10.0, step=0.5),
            "fcf": st.number_input("Min FCF (Mld $)", min_value=0.0, value=0.0, step=0.1) * 1e9,
            "peg": st.number_input("Max PEG Ratio", min_value=0.0, value=1.5, step=0.1),
            "pe": st.number_input("Max P/E (Fallback)", min_value=0.0, value=25.0, step=1.0),
            "int_cov": st.number_input("Min Int. Coverage", min_value=0.0, value=3.0, step=0.5),
            "perfect_only": st.checkbox("🏆 Solo 'All Green'")
        }

    with st.sidebar.expander("❓ Come cercare il ticker corretto"):
        st.markdown(
            "- Azioni USA: normalmente solo ticker (es. `AAPL`, `MSFT`).
"
            "- Azioni italiane: aggiungi `.MI` (es. `STLAM.MI`, `ENI.MI`, `ISP.MI`).
"
            "- Azioni tedesche: aggiungi `.DE` (es. `BMW.DE`, `SAP.DE`).
"
            "- Azioni francesi: aggiungi `.PA` (es. `AIR.PA`, `OR.PA`).
"
            "- Azioni UK: aggiungi `.L` (es. `ULVR.L`).
"
            "- Crypto: di solito coppia con valuta, es. `BTC-USD`, `ETH-USD`.
"
            "- Se hai dubbi, cerca prima il titolo su Yahoo Finance e copia il ticker esatto."
        )

    return {
        "mode": input_mode,
        "file": file,
        "manual": manual,
        "suffix": suffix,
        "btn": analyze_btn,
        "cfg": cfg
    }

# ==========================================
# 7. MAIN
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
    if 'holdings_qty' not in st.session_state:
        st.session_state.holdings_qty = {}
    if 'holdings_avg_cost' not in st.session_state:
        st.session_state.holdings_avg_cost = {}

    ui = setup_sidebar()

    if ui["btn"]:
        targets: List[str] = []

        if ui["mode"] == "Manuale":
            if ui["manual"]:
                targets = [ui["manual"]]
        elif ui["mode"] == "Batch (CSV)" and ui["file"] is not None:
            try:
                df_csv = pd.read_csv(ui["file"])
                if "Ticker" not in df_csv.columns:
                    st.error("Il CSV deve contenere una colonna chiamata 'Ticker'.")
                    targets = []
                else:
                    targets = df_csv["Ticker"].dropna().astype(str).tolist()[:MAX_CSV_ROWS]
            except Exception as e:
                st.error(f"Errore lettura CSV: {e}")
                targets = []

        if targets:
            results: List[Dict[str, Any]] = []

            normalized_tickers = []
            for t in targets:
                try:
                    normalized_tickers.append(normalize_ticker(t, ui["suffix"]))
                except Exception as e:
                    logger.warning(f"Ticker scartato {t}: {e}")

            for t in normalized_tickers:
                raw = get_fundamental_data(t)
                if raw:
                    met = calculate_fundamental_metrics(raw)
                    if met:
                        results.append(met.to_ui_dict())

            st.session_state.batch_results = pd.DataFrame(results)
            if results:
                st.session_state.selected_ticker = results[0]["Ticker"]
            else:
                st.warning("Nessun risultato disponibile per i ticker inseriti.")

    tab_f, tab_t, tab_q, tab_v, tab_p = st.tabs(
        ["📊 FONDAMENTALI", "📉 TECNICO", "⚛️ QUANT", "⚖️ VERDETTO", "📁 PORTAFOGLIO"]
    )

    ticker = st.session_state.selected_ticker

    if ticker and st.session_state.batch_results is not None and not st.session_state.batch_results.empty:
        row = st.session_state.batch_results[st.session_state.batch_results['Ticker'] == ticker].iloc[0]

        df_tech_global = get_technical_data(ticker)
        df_calc_global = calculate_technical_indicators(df_tech_global) if df_tech_global is not None and not df_tech_global.empty else None
        current_price_global = safe_float(df_calc_global['Close'].iloc[-1], 0.0) if df_calc_global is not None else 0.0
        score_global, reasons_global = calculate_timing_score(df_calc_global, current_price_global) if df_calc_global is not None else (0, [])
        qm_global = calculate_quant_metrics(df_tech_global, row["_raw_data"]) if df_tech_global is not None else {}
        risk_global = calculate_risk_metrics(df_tech_global) if df_tech_global is not None else {
            "Max Drawdown": np.nan,
            "CAGR": np.nan,
            "VaR_95": np.nan,
            "CVaR_95": np.nan,
            "Skew": np.nan,
            "Kurt": np.nan
        }

        with tab_f:
            st.info("💡 **Come leggere questa sezione:** Questa tabella rappresenta il motore dell'azienda. Cerca società con un ROIC costantemente alto e debito gestibile. Il Free Cash Flow è il vero denaro prodotto dal business.")

            st.dataframe(st.session_state.batch_results.drop(columns=["_raw_data"], errors="ignore"), use_container_width=True)

            try:
                fund_obj = FundamentalMetrics(
                    ticker=row["Ticker"],
                    company_name=row["Company Name"],
                    price=row["Price"],
                    fcf=row["Free Cash Flow"],
                    roic=row["ROIC"],
                    peg_ratio=row["PEG Ratio"],
                    peg_source=row["PEG Source"],
                    pe_ratio=row["P/E Ratio"],
                    interest_coverage=row["Interest Coverage"],
                    currency=row["Currency"],
                    raw_data=row["_raw_data"],
                )

                dcf_res = discounted_cash_flow_valuation(fund_obj)
                info_raw = fund_obj.raw_data["info"]
                ddm_fair = dividend_discount_model(info_raw)

                val_row = build_valuation_summary_row(fund_obj, dcf_res, ddm_fair)
                df_val = pd.DataFrame([val_row])

                st.markdown("#### Valutazione intrinseca (stima semplificata)")
                st.dataframe(df_val, use_container_width=True)

            except Exception as e:
                st.warning(f"Impossibile calcolare la valutazione intrinseca: {e}")

            if st.session_state.batch_results is not None and not st.session_state.batch_results.empty:
                st.markdown("#### Screener su batch analizzato")
                col_sc1, col_sc2 = st.columns(2)

                if col_sc1.button("Filtra per parametri fondamentali (sidebar)"):
                    df_filt = apply_fundamental_filters(st.session_state.batch_results, ui["cfg"])
                    st.write("Risultati filtrati (fondamentali):")
                    st.dataframe(df_filt.drop(columns=["_raw_data"], errors="ignore"), use_container_width=True)

                if col_sc2.button("Calcola SmartScore per tutto il batch"):
                    with st.spinner("Calcolo SmartScore su batch..."):
                        df_rank = rank_batch_by_smart_score(st.session_state.batch_results)
                    if df_rank.empty:
                        st.warning("Impossibile calcolare SmartScore per il batch (dati insufficienti).")
                    else:
                        st.write("Batch ordinato per SmartScore (decrescente):")
                        st.dataframe(df_rank.drop(columns=["_raw_data"], errors="ignore"), use_container_width=True)
                        st.download_button(
                            "⬇️ Esporta screener in CSV",
                            df_rank.drop(columns=["_raw_data"], errors="ignore").to_csv(index=False).encode("utf-8"),
                            file_name="burry_screener.csv",
                            mime="text/csv",
                        )

            st.markdown("---")
            st.markdown("<p style='text-align: center; color: gray;'>creato e sviluppato da Innovative Program</p>", unsafe_allow_html=True)

        with tab_t:
            st.info("💡 **Come leggere il grafico:** La linea blu (SMA 200) indica il trend di lungo periodo. Il grafico RSI segnala eccessi di euforia o pessimismo.")

            if df_tech_global is not None and df_calc_global is not None:
                st.metric("Timing Score", f"{score_global}/100")
                if reasons_global:
                    st.write(" | ".join(reasons_global))

                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
                fig.add_trace(
                    go.Candlestick(
                        x=df_calc_global.index,
                        open=df_calc_global['Open'],
                        high=df_calc_global['High'],
                        low=df_calc_global['Low'],
                        close=df_calc_global['Close'],
                        name="Prezzo"
                    ),
                    row=1, col=1
                )
                if "SMA_200" in df_calc_global.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df_calc_global.index,
                            y=df_calc_global['SMA_200'],
                            name="SMA 200",
                            line=dict(color='blue')
                        ),
                        row=1, col=1
                    )
                if "RSI" in df_calc_global.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df_calc_global.index,
                            y=df_calc_global['RSI'],
                            name="RSI",
                            line=dict(color='purple')
                        ),
                        row=2, col=1
                    )

                fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Dati tecnici non disponibili.")

            st.markdown("---")
            st.markdown("<p style='text-align: center; color: gray;'>creato e sviluppato da Innovative Program</p>", unsafe_allow_html=True)

        with tab_q:
            st.info("💡 **Come interpretare i dati:** Qui lasciamo parlare la statistica. Sharpe Ratio, Altman Z-Score, rischio di drawdown e simulazioni Monte Carlo aiutano a stimare robustezza e rischio.")

            if df_tech_global is not None:
                c1, c2, c3 = st.columns(3)
                c1.metric("Sharpe Ratio (4% Rf)", f"{safe_float(qm_global.get('Sharpe Ratio', np.nan), np.nan):.2f}" if pd.notna(qm_global.get('Sharpe Ratio', np.nan)) else "N/A")
                c2.metric("Trend R-Squared", f"{safe_float(qm_global.get('R-Squared', np.nan), np.nan):.2f}" if pd.notna(qm_global.get('R-Squared', np.nan)) else "N/A")
                altman_val = qm_global.get('Altman Z-Score', "N/A")
                c3.metric("Altman Z-Score", f"{altman_val:.2f}" if isinstance(altman_val, (float, int, np.floating)) else "N/A")

                c4, c5, c6 = st.columns(3)
                c4.metric("Max Drawdown", format_pct(risk_global['Max Drawdown']))
                c5.metric("CAGR", format_pct(risk_global['CAGR']))
                c6.metric("VaR 95% (giornaliero)", format_pct(risk_global['VaR_95']))

                asset_rets = df_tech_global["Close"].pct_change().dropna()
                regime = calculate_regime_statistics(asset_rets)
                bench_rets = get_benchmark_returns("^GSPC")
                fa = estimate_beta_and_alpha(asset_rets, bench_rets) if bench_rets is not None else {
                    "Beta": np.nan, "Alpha_ann": np.nan, "R2": np.nan
                }

                c_reg1, c_reg2, c_reg3 = st.columns(3)
                c_reg1.metric("Regime", regime["Regime"])
                c_reg2.metric("Vol ann (63g)", format_pct(regime["Vol_ann"]))
                skew_val = regime['Skew']
                kurt_val = regime['Kurt']
                c_reg3.metric("Skew/Kurt (63g)", f"{skew_val:.2f} / {kurt_val:.2f}" if pd.notna(skew_val) and pd.notna(kurt_val) else "N/A")

                c_beta1, c_beta2, c_beta3 = st.columns(3)
                c_beta1.metric("Beta vs S&P 500", f"{fa['Beta']:.2f}" if pd.notna(fa['Beta']) else "N/A")
                c_beta2.metric("Alpha annuo", format_pct(fa["Alpha_ann"]))
                c_beta3.metric("R²", f"{fa['R2']:.2f}" if pd.notna(fa['R2']) else "N/A")

                smart = compute_smart_quant_score(row, score_global, qm_global, risk_global)
                st.metric("Smart Quant Score", f"{smart['SmartScore']:.1f}/100")

                with st.expander("📉 Distribuzione rendimenti & rischio"):
                    st.write(f"CVaR 95%: {format_pct(risk_global['CVaR_95'])}")
                    st.write(f"Skewness: {risk_global['Skew']:.2f} | Kurtosis: {risk_global['Kurt']:.2f}" if pd.notna(risk_global['Skew']) and pd.notna(risk_global['Kurt']) else "N/A")
                    returns = df_tech_global['Close'].pct_change().dropna()
                    fig_r = go.Figure()
                    fig_r.add_trace(go.Histogram(x=returns, nbinsx=50, name="Rendimenti giornalieri"))
                    fig_r.update_layout(template="plotly_dark", bargap=0.05)
                    st.plotly_chart(fig_r, use_container_width=True)

                with st.expander("🎲 Simulazione Monte Carlo (rendimenti storici)"):
                    col_mc1, col_mc2 = st.columns(2)
                    horizon_days = col_mc1.slider("Orizzonte (giorni trading)", 60, 756, 252, step=21)
                    n_paths = col_mc2.slider("Numero traiettorie", 100, 3000, 1000, step=100)

                    mc = monte_carlo_equity(df_tech_global, n_paths=n_paths, horizon_days=horizon_days)

                    if mc["paths"] is not None:
                        final_vals = mc["final_distribution"]
                        mc_risk = monte_carlo_risk_bands(final_vals)

                        st.write(
                            f"Prob. perdita >10%: {mc_risk['Prob_loss_10']:.1f}% | "
                            f">20%: {mc_risk['Prob_loss_20']:.1f}% | "
                            f">30%: {mc_risk['Prob_loss_30']:.1f}%"
                        )
                        st.write(
                            f"Rendimento medio simulato: {mc_risk['Mean_ret']:.1f}% | "
                            f"CVaR < -20%: {mc_risk['CVaR_20']:.1f}%"
                        )

                        p05 = np.quantile(final_vals, 0.05)
                        p50 = np.quantile(final_vals, 0.50)

                        c7, c8, c9 = st.columns(3)
                        c7.metric("Prob. perdita > 20%", f"{(final_vals < 0.8).mean() * 100:.1f}%")
                        c8.metric("Mediana esito", f"{(p50 - 1) * 100:.1f}%")
                        c9.metric("Scenario 5° percentile", f"{(p05 - 1) * 100:.1f}%")

                        x = np.arange(1, horizon_days + 1)
                        fig_mc = go.Figure()
                        fig_mc.add_trace(go.Scatter(x=x, y=mc["q50"], name="Mediana", line=dict(color="cyan")))
                        fig_mc.add_trace(go.Scatter(x=x, y=mc["q95"], name="95° percentile", line=dict(color="green"), opacity=0.3))
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
            else:
                st.warning("Dati quantitativi non disponibili.")

            st.markdown("---")
            st.markdown("<p style='text-align: center; color: gray;'>creato e sviluppato da Innovative Program</p>", unsafe_allow_html=True)

        with tab_v:
            st.info("💡 **Come leggere il verdetto:** Sintesi tra fondamentali, tecnico e rischio quantitativo. Serve sempre margine di sicurezza.")

            z_val = qm_global.get('Altman Z-Score', 0.0)
            z_safe = "-" in ticker or (isinstance(z_val, (float, int, np.floating)) and z_val >= 1.8)

            peg_ok = row['PEG Ratio'] is not None and not pd.isna(row['PEG Ratio']) and row['PEG Ratio'] <= ui['cfg']['peg']
            fund_pts = (
                (1 if row['ROIC'] >= ui['cfg']['roic'] / 100 else 0) +
                (1 if peg_ok else 0)
            )

            if fund_pts >= 2 and z_safe and score_global >= 50:
                st.success("🟢 BUY: Fondamentali solidi e timing favorevole.")
            elif fund_pts >= 1 and z_safe:
                st.warning("🟡 HOLD: Azienda sicura ma attendere prezzi migliori.")
            else:
                st.error("🔴 SELL: Rischio finanziario o fondamentali scarsi.")

            if df_tech_global is not None and qm_global:
                smart_v = compute_smart_quant_score(row, score_global, qm_global, risk_global)
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

    with tab_p:
        st.info("💡 **Come usare questa sezione:** Qui costruisci il portafoglio reale. Ora puoi inserire sia l'importo investito sia il numero quote/azioni. Se inserisci quantità e prezzo medio, il sistema ricava il controvalore.")

        st.markdown(
            "### Portafoglio reale
"
            "- Inserisci il **ticker corretto** per il mercato.
"
            "- Puoi usare **importo investito** oppure **numero quote/azioni + prezzo medio di carico**.
"
            "- Per ETF inserisci il **numero quote**; per azioni inserisci il **numero azioni**.
"
            "- Il sistema calcola pesi, metriche di portafoglio, contributo al rischio e analisi avanzata.
"
        )

        all_tickers_batch: List[str] = []
        if st.session_state.batch_results is not None and not st.session_state.batch_results.empty:
            all_tickers_batch = st.session_state.batch_results["Ticker"].tolist()

        if all_tickers_batch:
            st.markdown("#### Seleziona dal batch analizzato")
            default_batch = [t for t in st.session_state.portfolio_tickers if t in all_tickers_batch]
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
                try:
                    t_clean = sanitize_ticker(manual_ticker)
                    if t_clean not in st.session_state.portfolio_tickers:
                        st.session_state.portfolio_tickers.append(t_clean)
                        st.success(f"Aggiunto {t_clean} al portafoglio.")
                except Exception as e:
                    st.error(str(e))
            else:
                st.warning("Inserisci un ticker valido prima di aggiungere.")

        portfolio_list = sorted(set(selected_from_batch + st.session_state.portfolio_tickers))
        st.session_state.portfolio_tickers = portfolio_list

        if portfolio_list:
            st.markdown("#### Dati per ogni posizione")
            cols = st.columns(2)

            holdings_amount: Dict[str, float] = st.session_state.holdings
            holdings_currency: Dict[str, str] = st.session_state.holdings_currency
            holdings_qty: Dict[str, float] = st.session_state.holdings_qty
            holdings_avg_cost: Dict[str, float] = st.session_state.holdings_avg_cost

            for i, t in enumerate(portfolio_list):
                col = cols[i % 2]
                col.markdown(f"##### {t}")

                default_amt = float(holdings_amount.get(t, 0.0))
                amt = col.number_input(
                    f"{t} - Importo investito",
                    min_value=0.0,
                    value=default_amt,
                    step=100.0,
                    key=f"holding_amt_{t}"
                )
                holdings_amount[t] = amt

                default_qty = float(holdings_qty.get(t, 0.0))
                qty = col.number_input(
                    f"{t} - Numero quote / azioni",
                    min_value=0.0,
                    value=default_qty,
                    step=1.0,
                    key=f"holding_qty_{t}"
                )
                holdings_qty[t] = qty

                default_avg = float(holdings_avg_cost.get(t, 0.0))
                avg_cost = col.number_input(
                    f"{t} - Prezzo medio di carico",
                    min_value=0.0,
                    value=default_avg,
                    step=0.1,
                    key=f"holding_avg_{t}"
                )
                holdings_avg_cost[t] = avg_cost

                cur_default = holdings_currency.get(t, "USD")
                cur_options = ["USD", "EUR"]
                cur_index = cur_options.index(cur_default) if cur_default in cur_options else 0
                cur = col.selectbox(
                    f"{t} - Valuta",
                    cur_options,
                    index=cur_index,
                    key=f"currency_{t}"
                )
                holdings_currency[t] = cur

                inferred = infer_position_value(amt, qty, avg_cost)
                col.caption(f"Controvalore usato dal sistema: {inferred:,.2f} {cur}")

                if col.button("🗑 Rimuovi", key=f"remove_{t}"):
                    if t in st.session_state.portfolio_tickers:
                        st.session_state.portfolio_tickers = [x for x in st.session_state.portfolio_tickers if x != t]
                    if t in st.session_state.holdings:
                        del st.session_state.holdings[t]
                    if t in st.session_state.holdings_currency:
                        del st.session_state.holdings_currency[t]
                    if t in st.session_state.holdings_qty:
                        del st.session_state.holdings_qty[t]
                    if t in st.session_state.holdings_avg_cost:
                        del st.session_state.holdings_avg_cost[t]
                    st.rerun()

            st.session_state.holdings = holdings_amount
            st.session_state.holdings_currency = holdings_currency
            st.session_state.holdings_qty = holdings_qty
            st.session_state.holdings_avg_cost = holdings_avg_cost

            if st.button("📊 Calcola pesi e analisi del portafoglio"):
                portfolio_positions = compute_portfolio_position_table(
                    portfolio_list=portfolio_list,
                    holdings_amount=holdings_amount,
                    holdings_qty=holdings_qty,
                    holdings_avg_cost=holdings_avg_cost,
                    holdings_currency=holdings_currency
                )

                positive_df = portfolio_positions[portfolio_positions["Controvalore usato"] > 0].copy()
                if positive_df.empty:
                    st.error("Imposta un importo > 0 oppure quantità e prezzo medio > 0 almeno per un titolo.")
                else:
                    total_value = positive_df["Controvalore usato"].sum()
                    positive_df["Peso %"] = positive_df["Controvalore usato"] / total_value * 100.0

                    weights_pct = dict(zip(positive_df["Ticker"], positive_df["Peso %"]))
                    built = build_portfolio_returns(list(positive_df["Ticker"]), weights_pct)

                    st.markdown("#### Pesi percentuali del portafoglio")
                    st.dataframe(positive_df, use_container_width=True)

                    latest_prices = get_latest_prices_for_portfolio(list(positive_df["Ticker"]))
                    market_df = build_market_value_table(positive_df, latest_prices)

                    st.markdown("#### Valore di mercato stimato")
                    st.dataframe(market_df, use_container_width=True)

                    if built is None:
                        st.error("Impossibile costruire la serie dei rendimenti (dati insufficienti per uno o più ticker).")
                    else:
                        df_rets, port_ret = built
                        pm = calculate_portfolio_metrics(port_ret)

                        cpa, cpv, cps, cpdd = st.columns(4)
                        cpa.metric("Rendimento annuo atteso", format_pct(pm['AnnRet']))
                        cpv.metric("Volatilità annua", format_pct(pm['AnnVol']))
                        cps.metric("Sharpe portafoglio", f"{pm['Sharpe']:.2f}" if pd.notna(pm['Sharpe']) else "N/A")
                        cpdd.metric("Max Drawdown portafoglio", format_pct(pm['MaxDD']))

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
                        st.dataframe(corr, use_container_width=True)

                        st.markdown("#### Contributo al rischio")
                        rc_df = compute_risk_contribution(df_rets, weights_pct)
                        if rc_df is not None:
                            st.dataframe(rc_df, use_container_width=True)

                        st.markdown("#### Pesi suggeriti")
                        minvar = optimize_portfolio_min_variance(df_rets)
                        maxsh = optimize_portfolio_max_sharpe(df_rets)

                        sugg_rows = []
                        for t in df_rets.columns.tolist():
                            sugg_rows.append({
                                "Ticker": t,
                                "Peso attuale %": weights_pct.get(t, np.nan),
                                "Min Variance %": minvar.get(t, np.nan) if minvar else np.nan,
                                "Max Sharpe %": maxsh.get(t, np.nan) if maxsh else np.nan,
                            })
                        st.dataframe(pd.DataFrame(sugg_rows), use_container_width=True)

                        st.markdown("#### Stress test")
                        shock_sel = st.selectbox(
                            "Scenario shock",
                            [
                                ("Mild -10%", -0.10),
                                ("Bear -20%", -0.20),
                                ("Crash -35%", -0.35),
                            ],
                            format_func=lambda x: x[0]
                        )
                        stress_ret = stress_test_portfolio(df_rets, weights_pct, shock_pct=shock_sel[1])
                        if stress_ret is not None:
                            st.metric("Rendimento annuo stimato sotto shock", format_pct(stress_ret))
        else:
            st.info("Seleziona almeno un titolo dal batch o aggiungilo manualmente per costruire il portafoglio.")

        st.markdown("---")
        st.markdown("<p style='text-align: center; color: gray;'>creato e sviluppato da Innovative Program</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()