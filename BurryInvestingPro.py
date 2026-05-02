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

# ==========================================
# CONFIGURAZIONE PAGINA
# ==========================================
st.set_page_config(
    page_title="Burry Investing Pro V2",
    page_icon="ðŸ“ˆ",
    layout="wide"
)

st.markdown(
    '<link rel="manifest" href="https://raw.githubusercontent.com/Innovativeprogram/burry-investing-pro/main/manifest.json">',
    unsafe_allow_html=True
)

hide_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.block-container {padding-top: 2rem;}
</style>
"""
st.markdown(hide_style, unsafe_allow_html=True)

# ==========================================
# SETUP LOGGING & COSTANTI GLOBALI
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_TAX_RATE = 0.21
SAFE_INTEREST_COVERAGE = 100.0
TRADING_DAYS_YEAR = 252
MAX_CSV_ROWS = 100
MAX_WORKERS = 10
RISK_FREE_RATE = 0.04

st.sidebar.header("Informazioni")
st.sidebar.info(
    """
**Burry Investing Pro V2**
Dashboard avanzata per l'analisi del valore intrinseco, analisi quantitativa e gestione portafoglio multi-asset.
*Analisi basata su dati di mercato in tempo reale e modelli quantitativi.*
"""
)

# ==========================================
# MODELLI DATI
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
            "_raw_data": self.raw_data,
        }

# ==========================================
# HELPER FUNCTIONS & VALIDAZIONE
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


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default

# ==========================================
# DATA ENGINE: ANALISI FONDAMENTALE
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
            "symbol": symbol,
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
                price=safe_float(info.get('regularMarketPrice') or info.get('currentPrice'), 0.0),
                fcf=0.0,
                roic=0.0,
                peg_ratio=None,
                peg_source="Crypto",
                pe_ratio=None,
                interest_coverage=SAFE_INTEREST_COVERAGE,
                currency=info.get('currency', 'USD'),
                raw_data=raw_data,
            )

        fin, bs, cf = raw_data["financials"], raw_data["balance_sheet"], raw_data["cashflow"]

        op_cash = safe_float(cf.loc['Operating Cash Flow'].iloc[0]) if 'Operating Cash Flow' in cf.index else 0.0
        cap_ex = safe_float(cf.loc['Capital Expenditure'].iloc[0]) if 'Capital Expenditure' in cf.index else 0.0
        fcf = op_cash + cap_ex

        total_debt = safe_float(bs.loc['Total Debt'].iloc[0]) if 'Total Debt' in bs.index else 0.0
        equity = safe_float(bs.loc['Stockholders Equity'].iloc[0], np.nan) if 'Stockholders Equity' in bs.index else np.nan
        invested_cap = total_debt + equity if not np.isnan(equity) else np.nan
        ebit = safe_float(fin.loc['EBIT'].iloc[0]) if 'EBIT' in fin.index else 0.0

        tax_rate = DEFAULT_TAX_RATE
        if 'Tax Provision' in fin.index and 'Pretax Income' in fin.index:
            pretax_inc = safe_float(fin.loc['Pretax Income'].iloc[0])
            tax_prov = safe_float(fin.loc['Tax Provision'].iloc[0])
            if pretax_inc > 0:
                tax_rate = max(0.0, min(1.0, tax_prov / pretax_inc))

        roic = float((ebit * (1 - tax_rate)) / invested_cap) if (invested_cap and invested_cap > 0 and not np.isnan(invested_cap)) else 0.0

        pe = info.get('trailingPE')
        growth = info.get('earningsGrowth')
        peg = info.get('pegRatio')
        peg_src = "N/A"
        if peg is not None:
            peg_src = "Official"
        elif pe and pe > 0 and growth and growth > 0:
            peg = float(pe / (growth * 100))
            peg_src = "Estimated"

        int_exp = safe_float(fin.loc['Interest Expense'].iloc[0]) if 'Interest Expense' in fin.index else 0.0
        int_cov = float(ebit / abs(int_exp)) if int_exp != 0 else SAFE_INTEREST_COVERAGE

        return FundamentalMetrics(
            ticker=raw_data["symbol"],
            company_name=info.get('longName', raw_data["symbol"]),
            price=safe_float(info.get('currentPrice'), 0.0),
            fcf=fcf,
            roic=roic,
            peg_ratio=float(peg) if peg else None,
            peg_source=peg_src,
            pe_ratio=float(pe) if pe else None,
            interest_coverage=int_cov,
            currency=info.get('currency', 'USD'),
            raw_data=raw_data,
        )
    except Exception as e:
        logger.error(f"Errore calcolo metriche {raw_data['symbol']}: {str(e)}")
        return None

# ==========================================
# DATA ENGINE: ANALISI TECNICA
# ==========================================
@st.cache_data(ttl=900, show_spinner=False)
def get_technical_data(symbol: str) -> Optional[pd.DataFrame]:
    try:
        df = yf.download(symbol, period="2y", interval="1d", progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df is None or df.empty or len(df) < 200:
            return None
        return df.dropna()
    except Exception:
        return None


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data['SMA_50'] = ta.sma(data['Close'], length=50)
    data['SMA_200'] = ta.sma(data['Close'], length=200)
    data['RSI'] = ta.rsi(data['Close'], length=14)
    bb = ta.bbands(data['Close'], length=20, std=2)
    if bb is not None and not bb.empty:
        data['BB_Lower'] = bb.filter(like='BBL_').iloc[:, 0]
        data['BB_Upper'] = bb.filter(like='BBU_').iloc[:, 0]
    return data


def calculate_timing_score(data: pd.DataFrame, current_price: float) -> Tuple[int, List[str]]:
    score, reasons = 0, []
    last_row = data.iloc[-1]
    sma200 = last_row.get('SMA_200', np.nan)
    if pd.notna(sma200) and current_price > sma200:
        score += 30
        reasons.append("âœ… Trend Rialzista (Sopra SMA 200)")
    else:
        reasons.append("âš ï¸ Trend Ribassista (Sotto SMA 200)")

    rsi = last_row.get('RSI', np.nan)
    if pd.notna(rsi):
        if rsi < 30:
            score += 30
            reasons.append("âœ… Ipervenduto (RSI < 30)")
        elif rsi > 70:
            score -= 10
            reasons.append("ðŸ›‘ Ipercomprato (RSI > 70)")

    bb_lower = last_row.get('BB_Lower', np.nan)
    if pd.notna(bb_lower) and current_price <= bb_lower * 1.02:
        score += 20
        reasons.append("âœ… Prezzo su Banda Bollinger Inferiore")

    return int(score), reasons

# ==========================================
# MOTORE QUANTISTICO
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
            "Trend Slope": np.nan,
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
            ta_val = safe_float(bs.loc['Total Assets'].iloc[0])
            wc = (safe_float(bs.loc['Current Assets'].iloc[0]) - safe_float(bs.loc['Current Liabilities'].iloc[0])) if 'Current Assets' in bs.index and 'Current Liabilities' in bs.index else 0.0
            re_val = safe_float(bs.loc['Retained Earnings'].iloc[0]) if 'Retained Earnings' in bs.index else 0.0
            ebit = safe_float(fin.loc['EBIT'].iloc[0]) if 'EBIT' in fin.index else 0.0
            mc = safe_float(info.get('marketCap'), 1.0)
            tl = safe_float(bs.loc['Total Liabilities Net Minority Interest'].iloc[0], 1.0) if 'Total Liabilities Net Minority Interest' in bs.index else 1.0
            rev = safe_float(info.get('totalRevenue'), 0.0)
            if ta_val > 0 and tl > 0:
                z_score = (1.2 * (wc / ta_val)) + (1.4 * (re_val / ta_val)) + (3.3 * (ebit / ta_val)) + (0.6 * (mc / tl)) + (1.0 * (rev / ta_val))
        except Exception:
            pass

    return {
        "Sharpe Ratio": float(sharpe),
        "Annual Volatility": float(vol),
        "R-Squared": float(r_sq),
        "Altman Z-Score": z_score,
        "Price Percentile": float((df['Close'] < df['Close'].iloc[-1]).mean() * 100),
        "Trend Slope": float(model.coef_[0][0]),
    }


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
            "Kurt": np.nan,
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
        "Kurt": float(returns.kurt()),
    }


def monte_carlo_equity(df: pd.DataFrame, n_paths: int = 1000, horizon_days: int = 252, seed: Optional[int] = None) -> Dict[str, Any]:
    prices = df['Close'].dropna()
    returns = prices.pct_change().dropna().values
    if returns.size == 0:
        return {"paths": None, "final_distribution": None, "q05": None, "q50": None, "q95": None}

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(returns), size=(n_paths, horizon_days))
    sampled = returns[idx]
    equity_paths = (1 + sampled).cumprod(axis=1)
    final_values = equity_paths[:, -1]

    return {
        "paths": equity_paths,
        "final_distribution": final_values,
        "q05": np.quantile(equity_paths, 0.05, axis=0),
        "q50": np.quantile(equity_paths, 0.50, axis=0),
        "q95": np.quantile(equity_paths, 0.95, axis=0),
    }


def compute_smart_quant_score(row: Any, timing_score: int, qm: Dict[str, Any], risk: Dict[str, Any]) -> Dict[str, Any]:
    f_score = 0.0
    roic = row.get("ROIC", 0.0)
    f_score += float(np.clip((roic - 0.10) / (0.25 - 0.10), 0, 1)) * 50.0

    peg = row.get("PEG Ratio", None)
    if peg is not None and peg > 0:
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

    if isinstance(max_dd, (float, np.floating)) and pd.notna(max_dd):
        if max_dd < -0.5:
            q_score -= 20.0
        elif max_dd < -0.3:
            q_score -= 10.0

    q_score = float(np.clip(q_score, 0, 100))
    smart = float(np.clip(0.4 * f_score + 0.3 * t_score + 0.3 * q_score, 0, 100))

    return {
        "SmartScore": smart,
        "FundamentalScore": f_score,
        "TechnicalScore": t_score,
        "QuantRiskScore": q_score,
    }

# ==========================================
# PORTAFOGLIO: RENDIMENTI & METRICHE
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


def build_portfolio_returns(tickers: List[str], weights_pct: Dict[str, float]) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
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
        return {"AnnRet": np.nan, "AnnVol": np.nan, "Sharpe": np.nan, "MaxDD": np.nan}

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
        "MaxDD": float(max_dd),
    }

# ==========================================
# VALUTAZIONE INTRINSECA
# ==========================================
def discounted_cash_flow_valuation(fund: FundamentalMetrics, years: int = 10, growth: float = 0.05, discount_rate: float = 0.10, terminal_growth: float = 0.02) -> Dict[str, float]:
    info = fund.raw_data.get("info", {})
    fcf = max(float(fund.fcf), 0.0)
    shares = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding") or 0

    if shares <= 0 or fcf <= 0:
        return {"fair_value": np.nan, "mos_pct": np.nan, "fair_low": np.nan, "fair_high": np.nan}

    def _dcf_once(g, dr):
        cash_flows = []
        for t in range(1, years + 1):
            fcf_t = fcf * ((1 + g) ** t)
            cash_flows.append(fcf_t / ((1 + dr) ** t))
        terminal_fcf = fcf * ((1 + g) ** years) * (1 + terminal_growth)
        terminal_value = terminal_fcf / max(dr - terminal_growth, 1e-4)
        terminal_pv = terminal_value / ((1 + dr) ** years)
        equity_value = sum(cash_flows) + terminal_pv
        return equity_value / shares

    fair_base = _dcf_once(growth, discount_rate)
    fair_low = _dcf_once(max(growth - 0.03, 0.0), discount_rate + 0.02)
    fair_high = _dcf_once(growth + 0.03, max(discount_rate - 0.02, 0.02))
    mos_pct = ((fair_base - fund.price) / fair_base * 100.0) if fair_base > 0 else np.nan

    return {
        "fair_value": float(fair_base),
        "mos_pct": float(mos_pct),
        "fair_low": float(fair_low),
        "fair_high": float(fair_high),
    }


def dividend_discount_model(info: Dict[str, Any], required_return: float = 0.10, growth: float = 0.03) -> Optional[float]:
    div_rate = info.get("dividendRate")
    if not div_rate or required_return <= growth:
        return None
    try:
        return float(div_rate * (1 + growth) / (required_return - growth))
    except Exception:
        return None


def build_valuation_summary_row(fund: FundamentalMetrics, dcf_res: Dict[str, float], ddm_fair: Optional[float]) -> Dict[str, Any]:
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
# QUANT AVANZATO
# ==========================================
def calculate_regime_statistics(returns: pd.Series, lookback: int = 63) -> Dict[str, Any]:
    if returns is None or returns.empty or len(returns) < lookback:
        return {"Regime": "N/D", "Vol_ann": np.nan, "Skew": np.nan, "Kurt": np.nan}
    window = returns.tail(lookback)
    vol_ann = window.std() * np.sqrt(TRADING_DAYS_YEAR)
    skew = window.skew()
    kurt = window.kurt()
    regime = "Normale"
    if vol_ann > 0.30 or skew < -1.0:
        regime = "Stress"
    elif vol_ann < 0.15 and abs(skew) < 0.5:
        regime = "Calmo"
    return {"Regime": regime, "Vol_ann": float(vol_ann), "Skew": float(skew), "Kurt": float(kurt)}


@st.cache_data(ttl=900, show_spinner=False)
def get_benchmark_returns(symbol: str = "^GSPC", period: str = "2y", interval: str = "1d") -> Optional[pd.Series]:
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        rets = df["Close"].pct_change().dropna()
        rets.name = symbol
        return rets
    except Exception:
        return None


def estimate_beta_and_alpha(asset_rets: pd.Series, bench_rets: pd.Series) -> Dict[str, float]:
    df = pd.concat([asset_rets, bench_rets], axis=1).dropna()
    if df.empty:
        return {"Beta": np.nan, "Alpha_ann": np.nan, "R2": np.nan}
    x = df.iloc[:, 1].values.reshape(-1, 1)
    y = df.iloc[:, 0].values
    model = LinearRegression().fit(x, y)
    return {
        "Beta": float(model.coef_[0]),
        "Alpha_ann": float(model.intercept_ * TRADING_DAYS_YEAR),
        "R2": float(model.score(x, y)),
    }


def monte_carlo_risk_bands(final_values: np.ndarray) -> Dict[str, float]:
    if final_values is None or len(final_values) == 0:
        return {"Prob_loss_10": np.nan, "Prob_loss_20": np.nan, "Prob_loss_30": np.nan, "Mean_ret": np.nan, "CVaR_20": np.nan}
    final_values = np.asarray(final_values)
    final_ret = final_values - 1.0
    tail = final_ret[final_ret < -0.20]
    return {
        "Prob_loss_10": float((final_ret < -0.10).mean() * 100.0),
        "Prob_loss_20": float((final_ret < -0.20).mean() * 100.0),
        "Prob_loss_30": float((final_ret < -0.30).mean() * 100.0),
        "Mean_ret": float(final_ret.mean() * 100.0),
        "CVaR_20": float(tail.mean() * 100.0) if tail.size > 0 else np.nan,
    }

# ==========================================
# PORTAFOGLIO AVANZATO
# ==========================================
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
    return pd.DataFrame({"Ticker": cols, "Peso %": w * 100.0, "Risk Contribution %": rc_pct})


def optimize_portfolio_min_variance(df_rets: pd.DataFrame) -> Optional[Dict[str, float]]:
    if df_rets is None or df_rets.empty:
        return None
    cov = df_rets.cov().values * TRADING_DAYS_YEAR
    n = cov.shape[0]
    ones = np.ones(n)
    try:
        inv_cov = np.linalg.inv(cov + np.eye(n) * 1e-8)
        w = np.clip(inv_cov @ ones, 0, None)
        if w.sum() == 0:
            w = np.ones(n)
        w = w / w.sum()
        return {t: float(w[i] * 100.0) for i, t in enumerate(df_rets.columns)}
    except np.linalg.LinAlgError:
        return None


def optimize_portfolio_max_sharpe(df_rets: pd.DataFrame, risk_free: float = RISK_FREE_RATE) -> Optional[Dict[str, float]]:
    if df_rets is None or df_rets.empty:
        return None
    mu = df_rets.mean().values * TRADING_DAYS_YEAR
    cov = df_rets.cov().values * TRADING_DAYS_YEAR
    n = cov.shape[0]
    rf_vec = np.full(n, risk_free)
    try:
        inv_cov = np.linalg.inv(cov + np.eye(n) * 1e-8)
        w = np.clip(inv_cov @ (mu - rf_vec), 0, None)
        if w.sum() == 0:
            w = np.ones(n)
        w = w / w.sum()
        return {t: float(w[i] * 100.0) for i, t in enumerate(df_rets.columns)}
    except np.linalg.LinAlgError:
        return None


def stress_test_portfolio(df_rets: pd.DataFrame, weights_pct: Dict[str, float], shock_pct: float = -0.20) -> Optional[float]:
    if df_rets is None or df_rets.empty:
        return None
    cols = df_rets.columns.tolist()
    w = np.array([weights_pct.get(t, 0.0) for t in cols], dtype=float) / 100.0
    if w.sum() <= 0:
        return None
    w = w / w.sum()
    mu = df_rets.mean().values * TRADING_DAYS_YEAR
    return float(np.dot(w, mu + shock_pct))

# ==========================================
# BATCH SCREENER
# ==========================================
def apply_fundamental_filters(batch_df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    df = batch_df.copy()
    if df.empty:
        return df
    df = df[df["ROIC"] >= cfg["roic"] / 100.0]
    df = df[df["Free Cash Flow"] >= cfg["fcf"]]
    if cfg["peg"] > 0:
        df = df[(df["PEG Ratio"].isna()) | (df["PEG Ratio"] <= cfg["peg"])]
    if cfg["pe"] > 0:
        df = df[(df["P/E Ratio"].isna()) | (df["P/E Ratio"] <= cfg["pe"])]
    df = df[df["Interest Coverage"] >= cfg["int_cov"]]
    if cfg.get("perfect_only", False):
        df = df[(df["PEG Ratio"].notna()) & (df["P/E Ratio"].notna())]
    return df


def rank_batch_by_smart_score(batch_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in batch_df.iterrows():
        raw = row["_raw_data"]
        df_tech = get_technical_data(row["Ticker"])
        if df_tech is None:
            continue
        df_calc = calculate_technical_indicators(df_tech)
        score, _ = calculate_timing_score(df_calc, df_calc["Close"].iloc[-1])
        qm = calculate_quant_metrics(df_tech, raw)
        risk = calculate_risk_metrics(df_tech)
        smart = compute_smart_quant_score(row, score, qm, risk)
        new_row = dict(row)
        new_row.update(smart)
        rows.append(new_row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("SmartScore", ascending=False)

# ==========================================
# PORTAFOGLIO REALE: PREZZI, QUANTITA', COST BASIS
# ==========================================
def get_latest_price(symbol: str) -> Optional[float]:
    try:
        data = get_fundamental_data(symbol)
        if not data:
            return None
        info = data.get("info", {})
        px = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
        return float(px) if px is not None else None
    except Exception:
        return None


def infer_units_from_amount(amount: float, price: Optional[float]) -> Optional[float]:
    if amount is None or amount <= 0 or price is None or price <= 0:
        return None
    try:
        return float(amount / price)
    except Exception:
        return None


def build_portfolio_position_table(
    portfolio_list: List[str],
    holdings: Dict[str, float],
    holdings_currency: Dict[str, str],
    holdings_units: Dict[str, float],
    holdings_cost_basis: Dict[str, float],
) -> pd.DataFrame:
    rows = []
    for t in portfolio_list:
        amt = float(holdings.get(t, 0.0) or 0.0)
        cur = holdings_currency.get(t, "USD")
        units = holdings_units.get(t, None)
        cost_basis = holdings_cost_basis.get(t, None)
        live_price = get_latest_price(t)
        market_value_from_units = np.nan
        if units is not None and units > 0 and live_price is not None and live_price > 0:
            market_value_from_units = float(units * live_price)
        current_value = float(market_value_from_units) if pd.notna(market_value_from_units) else amt
        avg_cost = float(cost_basis / units) if (cost_basis is not None and units is not None and units > 0) else np.nan
        unrealized_pl = float(current_value - cost_basis) if (cost_basis is not None and cost_basis > 0 and current_value is not None) else np.nan
        unrealized_pl_pct = float((current_value / cost_basis - 1.0) * 100.0) if (cost_basis is not None and cost_basis > 0 and current_value is not None) else np.nan
        rows.append({
            "Ticker": t,
            "Valuta": cur,
            "QuantitÃ ": float(units) if units is not None else np.nan,
            "Prezzo attuale": live_price,
            "Valore attuale": float(current_value) if current_value is not None else np.nan,
            "Importo manuale": amt,
            "Costo totale": float(cost_basis) if cost_basis is not None else np.nan,
            "Prezzo medio carico": avg_cost,
            "P/L non realizzato": unrealized_pl,
            "P/L %": unrealized_pl_pct,
        })
    return pd.DataFrame(rows)


def compute_portfolio_weights_from_positions(position_df: pd.DataFrame) -> Dict[str, float]:
    if position_df is None or position_df.empty:
        return {}
    df = position_df.copy()
    df = df[df["Valore attuale"].fillna(0) > 0]
    if df.empty:
        return {}
    total = float(df["Valore attuale"].sum())
    return {row["Ticker"]: float(row["Valore attuale"] / total * 100.0) for _, row in df.iterrows()}

# ==========================================
# UI: SIDEBAR
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
    market = st.sidebar.selectbox("Borsa:", ["USA", "Italia (.MI)", "Germania (.DE)", "Francia (.PA)", "GB (.L)", "Crypto", "Custom"])
    suffix = ""
    if "Italia" in market:
        suffix = ".MI"
    elif "Germania" in market:
        suffix = ".DE"
    elif "Francia" in market:
        suffix = ".PA"
    elif "GB" in market:
        suffix = ".L"

    analyze_btn = st.sidebar.button("ðŸš€ Avvia Analisi", use_container_width=True)

    with st.sidebar.expander("âš™ï¸ Parametri Fondamentali"):
        cfg = {
            "roic": st.number_input("Min ROIC %", 10.0, step=0.5),
            "fcf": st.number_input("Min FCF (Mld $)", 0.0) * 1e9,
            "peg": st.number_input("Max PEG Ratio", 1.5, step=0.1),
            "pe": st.number_input("Max P/E (Fallback)", 25.0),
            "int_cov": st.number_input("Min Int. Coverage", 3.0),
            "perfect_only": st.checkbox("ðŸ† Solo 'All Green'"),
        }

    with st.sidebar.expander("â“ Come cercare il ticker corretto"):
        st.markdown(
            "- Azioni USA: normalmente solo ticker (es. `AAPL`, `MSFT`).\n"
            "- Azioni italiane: aggiungi `.MI` (es. `ENI.MI`, `ISP.MI`).\n"
            "- Azioni tedesche: aggiungi `.DE` (es. `BMW.DE`, `SAP.DE`).\n"
            "- Azioni francesi: aggiungi `.PA` (es. `AIR.PA`, `OR.PA`).\n"
            "- Azioni UK: aggiungi `.L` (es. `ULVR.L`).\n"
            "- Crypto: di solito coppia con valuta, es. `BTC-USD`, `ETH-USD`.\n"
            "- Se hai dubbi, cerca prima il titolo su Yahoo Finance e copia il ticker esatto."
        )

    return {"mode": input_mode, "file": file, "manual": manual, "suffix": suffix, "btn": analyze_btn, "cfg": cfg}

# ==========================================
# MAIN APP
# ==========================================
def main():
    st.title("ðŸ’Ž Burry Investing Pro V2")

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
    if 'holdings_units' not in st.session_state:
        st.session_state.holdings_units = {}
    if 'holdings_cost_basis' not in st.session_state:
        st.session_state.holdings_cost_basis = {}

    ui = setup_sidebar()
    if ui["btn"]:
        targets: List[str] = [ui["manual"]] if ui["mode"] == "Manuale" and ui["manual"] else []
        if ui["mode"] == "Batch (CSV)" and ui["file"] is not None:
            try:
                targets = pd.read_csv(ui["file"])["Ticker"].dropna().astype(str).tolist()[:MAX_CSV_ROWS]
            except Exception:
                st.error("CSV non valido: deve contenere una colonna 'Ticker'.")
                targets = []

        if targets:
            results: List[Dict[str, Any]] = []
            normalized = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futures = [ex.submit(normalize_ticker, t, ui["suffix"]) for t in targets]
                for f in concurrent.futures.as_completed(futures):
                    try:
                        normalized.append(f.result())
                    except Exception:
                        pass

            for t in normalized:
                raw = get_fundamental_data(t)
                if raw:
                    met = calculate_fundamental_metrics(raw)
                    if met:
                        results.append(met.to_ui_dict())

            st.session_state.batch_results = pd.DataFrame(results)
            if results:
                st.session_state.selected_ticker = results[0]["Ticker"]

    tabs = st.tabs(["ðŸ“Š FONDAMENTALI", "ðŸ“‰ TECNICO", "âš›ï¸ QUANT", "ðŸ’° VALUTAZIONE", "âš–ï¸ VERDETTO", "ðŸ“ PORTAFOGLIO", "ðŸ§® SCREENER"])
    tab_f, tab_t, tab_q, tab_val, tab_v, tab_p, tab_s = tabs

    ticker = st.session_state.selected_ticker
    batch_df = st.session_state.batch_results

    if ticker and batch_df is not None and not batch_df.empty:
        row = batch_df[batch_df['Ticker'] == ticker].iloc[0]
        raw_data = row["_raw_data"]
        df_tech = get_technical_data(ticker)

        with tab_f:
            st.info("Questa tab mostra i fondamentali aziendali. Cerca qualitÃ  del business, redditivitÃ  del capitale e sostenibilitÃ  del debito.")
            st.dataframe(batch_df.drop(columns=["_raw_data"], errors="ignore"))

        if df_tech is not None:
            df_calc = calculate_technical_indicators(df_tech)
            current_price = float(df_calc['Close'].iloc[-1])
            timing_score, reasons = calculate_timing_score(df_calc, current_price)
            qm = calculate_quant_metrics(df_tech, raw_data)
            risk = calculate_risk_metrics(df_tech)
            smart = compute_smart_quant_score(row, timing_score, qm, risk)
            returns = df_tech['Close'].pct_change().dropna()

            with tab_t:
                st.subheader(f"Analisi tecnica: {ticker}")
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
                fig.add_trace(go.Scatter(x=df_calc.index, y=df_calc['Close'], name='Close'), row=1, col=1)
                if 'SMA_50' in df_calc:
                    fig.add_trace(go.Scatter(x=df_calc.index, y=df_calc['SMA_50'], name='SMA 50'), row=1, col=1)
                if 'SMA_200' in df_calc:
                    fig.add_trace(go.Scatter(x=df_calc.index, y=df_calc['SMA_200'], name='SMA 200'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_calc.index, y=df_calc['RSI'], name='RSI'), row=2, col=1)
                fig.update_layout(template="plotly_dark", height=700)
                st.plotly_chart(fig, use_container_width=True)
                st.metric("Timing Score", timing_score)
                for reason in reasons:
                    st.write(reason)

            with tab_q:
                st.subheader(f"Analisi quantitativa: {ticker}")
                c1, c2, c3 = st.columns(3)
                c1.metric("Sharpe", f"{qm['Sharpe Ratio']:.2f}" if pd.notna(qm['Sharpe Ratio']) else "N/D")
                c2.metric("VolatilitÃ  annua", f"{qm['Annual Volatility'] * 100:.2f}%" if pd.notna(qm['Annual Volatility']) else "N/D")
                c3.metric("Altman Z", f"{qm['Altman Z-Score']:.2f}" if isinstance(qm['Altman Z-Score'], (int, float, np.floating)) else "N/D")

                regime = calculate_regime_statistics(returns)
                bench = get_benchmark_returns()
                beta_alpha = estimate_beta_and_alpha(returns, bench) if bench is not None else {"Beta": np.nan, "Alpha_ann": np.nan, "R2": np.nan}
                mc = monte_carlo_equity(df_tech)
                mc_bands = monte_carlo_risk_bands(mc["final_distribution"]) if mc["final_distribution"] is not None else {}

                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Regime", regime["Regime"])
                r2.metric("Beta", f"{beta_alpha['Beta']:.2f}" if pd.notna(beta_alpha['Beta']) else "N/D")
                r3.metric("Alpha ann.", f"{beta_alpha['Alpha_ann'] * 100:.2f}%" if pd.notna(beta_alpha['Alpha_ann']) else "N/D")
                r4.metric("RÂ²", f"{beta_alpha['R2']:.2f}" if pd.notna(beta_alpha['R2']) else "N/D")

                if mc.get("q50") is not None:
                    fig_mc = go.Figure()
                    fig_mc.add_trace(go.Scatter(y=mc["q05"], mode='lines', name='5%'))
                    fig_mc.add_trace(go.Scatter(y=mc["q50"], mode='lines', name='50%'))
                    fig_mc.add_trace(go.Scatter(y=mc["q95"], mode='lines', name='95%'))
                    fig_mc.update_layout(template='plotly_dark', height=400, title='Monte Carlo Equity Bands')
                    st.plotly_chart(fig_mc, use_container_width=True)
                    st.dataframe(pd.DataFrame([mc_bands]))

            with tab_val:
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
                ddm_fair = dividend_discount_model(raw_data.get("info", {}))
                val_row = build_valuation_summary_row(fund_obj, dcf_res, ddm_fair)
                st.dataframe(pd.DataFrame([val_row]))

            with tab_v:
                st.subheader("Verdetto integrato")
                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric("Smart Score", f"{smart['SmartScore']:.1f}/100")
                sc2.metric("Fundamental", f"{smart['FundamentalScore']:.1f}")
                sc3.metric("Technical", f"{smart['TechnicalScore']:.1f}")
                sc4.metric("Quant/Risk", f"{smart['QuantRiskScore']:.1f}")

                if smart['SmartScore'] >= 75:
                    st.success("Titolo molto interessante secondo il modello integrato.")
                elif smart['SmartScore'] >= 55:
                    st.warning("Titolo discreto, ma da contestualizzare meglio.")
                else:
                    st.error("Titolo debole secondo il modello integrato.")

        with tab_s:
            st.subheader("Screener batch")
            if batch_df is not None and not batch_df.empty:
                filtered = apply_fundamental_filters(batch_df, ui["cfg"])
                ranked = rank_batch_by_smart_score(filtered) if not filtered.empty else pd.DataFrame()
                if ranked.empty:
                    st.info("Nessun titolo disponibile dopo i filtri o dati tecnici insufficienti.")
                else:
                    st.dataframe(ranked.drop(columns=["_raw_data"], errors="ignore"))
                    st.download_button(
                        "â¬‡ï¸ Esporta screener CSV",
                        ranked.drop(columns=["_raw_data"], errors="ignore").to_csv(index=False).encode("utf-8"),
                        file_name="screener_results.csv",
                        mime="text/csv",
                    )
            else:
                st.info("Carica o analizza prima almeno un ticker.")
    else:
        with tab_f:
            st.info("Inserisci un ticker o carica un CSV per iniziare.")
        with tab_s:
            st.info("Lo screener si attiva dopo l'analisi batch o manuale.")

    with tab_p:
        st.subheader("Portafoglio")
        new_portfolio_ticker = st.text_input("Aggiungi ticker al portafoglio", value="").upper().strip()
        col_add_1, col_add_2 = st.columns([1, 1])
        with col_add_1:
            if st.button("âž• Aggiungi al portafoglio"):
                if new_portfolio_ticker:
                    try:
                        clean_t = sanitize_ticker(new_portfolio_ticker)
                        if clean_t not in st.session_state.portfolio_tickers:
                            st.session_state.portfolio_tickers.append(clean_t)
                    except Exception:
                        st.error("Ticker non valido.")
        with col_add_2:
            if st.button("ðŸ§¹ Svuota portafoglio"):
                st.session_state.portfolio_tickers = []
                st.session_state.holdings = {}
                st.session_state.holdings_currency = {}
                st.session_state.holdings_units = {}
                st.session_state.holdings_cost_basis = {}

        portfolio_list = st.session_state.portfolio_tickers
        if portfolio_list:
            st.markdown("#### Dati posizioni")
            st.caption("Puoi inserire importo investito, quantitÃ  quote/azioni e costo totale di carico.")
            cols = st.columns(3)
            holdings = st.session_state.holdings
            holdings_units = st.session_state.holdings_units
            holdings_cost_basis = st.session_state.holdings_cost_basis

            for i, t in enumerate(portfolio_list):
                col = cols[i % 3]
                live_price = get_latest_price(t)
                default_amt = float(holdings.get(t, 0.0))
                amt = col.number_input(f"{t} - Importo investito", min_value=0.0, value=default_amt, step=100.0, key=f"holding_{t}")
                holdings[t] = amt

                default_units = holdings_units.get(t, None)
                inferred_units = infer_units_from_amount(amt, live_price)
                units_value = float(default_units if default_units is not None else (inferred_units if inferred_units is not None else 0.0))
                units = col.number_input(f"{t} - Numero quote / azioni", min_value=0.0, value=units_value, step=1.0, key=f"units_{t}")
                holdings_units[t] = units

                default_cost_basis = float(holdings_cost_basis.get(t, amt if amt > 0 else 0.0))
                cost_basis = col.number_input(f"{t} - Costo totale di carico", min_value=0.0, value=default_cost_basis, step=100.0, key=f"cost_basis_{t}")
                holdings_cost_basis[t] = cost_basis

                cur_default = st.session_state.holdings_currency.get(t, "USD")
                cur = col.selectbox(f"{t} - Valuta", ["USD", "EUR"], index=["USD", "EUR"].index(cur_default), key=f"currency_{t}")
                st.session_state.holdings_currency[t] = cur

                if live_price is not None:
                    col.caption(f"Prezzo live stimato: {live_price:.2f} {cur}")
                else:
                    col.caption("Prezzo live non disponibile")

                if col.button("ðŸ—‘ Rimuovi", key=f"remove_{t}"):
                    st.session_state.portfolio_tickers = [x for x in st.session_state.portfolio_tickers if x != t]
                    st.session_state.holdings.pop(t, None)
                    st.session_state.holdings_currency.pop(t, None)
                    st.session_state.holdings_units.pop(t, None)
                    st.session_state.holdings_cost_basis.pop(t, None)
                    st.rerun()

            st.session_state.holdings = holdings
            st.session_state.holdings_units = holdings_units
            st.session_state.holdings_cost_basis = holdings_cost_basis

            if st.button("ðŸ“Š Calcola pesi e analisi del portafoglio"):
                position_df = build_portfolio_position_table(
                    portfolio_list,
                    holdings,
                    st.session_state.holdings_currency,
                    holdings_units,
                    holdings_cost_basis,
                )

                positive_holdings = {
                    row["Ticker"]: row["Valore attuale"]
                    for _, row in position_df.iterrows()
                    if pd.notna(row["Valore attuale"]) and row["Valore attuale"] > 0
                }

                if not positive_holdings:
                    st.error("Imposta un importo > 0 oppure quantitÃ  > 0 almeno per un titolo.")
                else:
                    weights_pct = compute_portfolio_weights_from_positions(position_df)
                    built = build_portfolio_returns(list(positive_holdings.keys()), weights_pct)

                    st.markdown("#### Posizioni e pesi percentuali del portafoglio")
                    df_weights = position_df.copy()
                    df_weights["Peso %"] = df_weights["Ticker"].map(weights_pct)
                    st.dataframe(df_weights)
                    st.download_button(
                        "â¬‡ï¸ Esporta posizioni portafoglio in CSV",
                        df_weights.to_csv(index=False).encode("utf-8"),
                        file_name="portfolio_positions.csv",
                        mime="text/csv",
                    )

                    if built is None:
                        st.error("Impossibile costruire la serie dei rendimenti: dati insufficienti per uno o piÃ¹ ticker.")
                    else:
                        df_rets, port_ret = built
                        pm = calculate_portfolio_metrics(port_ret)

                        cpa, cpv, cps, cpdd = st.columns(4)
                        cpa.metric("Rendimento annuo atteso", f"{pm['AnnRet'] * 100:.2f}%")
                        cpv.metric("VolatilitÃ  annua", f"{pm['AnnVol'] * 100:.2f}%")
                        cps.metric("Sharpe portafoglio", f"{pm['Sharpe']:.2f}")
                        cpdd.metric("Max Drawdown portafoglio", f"{pm['MaxDD'] * 100:.1f}%")

                        equity_p = (1 + port_ret).cumprod()
                        fig_p = go.Figure()
                        fig_p.add_trace(go.Scatter(x=equity_p.index, y=equity_p.values, name="Equity portafoglio"))
                        fig_p.update_layout(template="plotly_dark", height=400, xaxis_title="Data", yaxis_title="Equity normalizzata")
                        st.plotly_chart(fig_p, use_container_width=True)

                        st.markdown("#### Contributo al rischio")
                        df_rc = compute_risk_contribution(df_rets, weights_pct)
                        if df_rc is not None:
                            st.dataframe(df_rc)

                        st.markdown("#### Ottimizzazione portafoglio")
                        minvar = optimize_portfolio_min_variance(df_rets)
                        maxsh = optimize_portfolio_max_sharpe(df_rets)
                        opt_rows = []
                        for tk in df_rets.columns.tolist():
                            opt_rows.append({
                                "Ticker": tk,
                                "Peso attuale %": weights_pct.get(tk, np.nan),
                                "Min Variance %": minvar.get(tk, np.nan) if minvar else np.nan,
                                "Max Sharpe %": maxsh.get(tk, np.nan) if maxsh else np.nan,
                            })
                        st.dataframe(pd.DataFrame(opt_rows))

                        st.markdown("#### Stress test")
                        shock_choice = st.selectbox("Scenario shock uniforme", ["-10%", "-20%", "-30%"], index=1, key="portfolio_shock_choice")
                        shock_map = {"-10%": -0.10, "-20%": -0.20, "-30%": -0.30}
                        shock_ret = stress_test_portfolio(df_rets, weights_pct, shock_pct=shock_map[shock_choice])
                        if shock_ret is not None:
                            st.metric("Rendimento annuo stimato sotto stress", f"{shock_ret * 100:.2f}%")

                        st.markdown("#### Correlazioni tra titoli in portafoglio")
                        corr = df_rets.corr()
                        st.dataframe(corr)
        else:
            st.info("Aggiungi uno o piÃ¹ ticker al portafoglio per iniziare.")

if __name__ == "__main__":
    main()