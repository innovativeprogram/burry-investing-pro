"""
V-Quant Pro v3.0 - Analisi universale: Azioni, ETF, Obbligazioni, Crypto, Commodity
Copyright (c) 2026 InnovativeProgram
"""

import streamlit as st
import yfinance as yf
import requests
from yahooquery import Ticker as YQ_Ticker
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List
from sklearn.linear_model import LinearRegression
from supabase import create_client, Client

try:
    import pandas_ta as ta
except ImportError:
    ta = None

# ---------------------------- COSTANTI ---------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VqQuantPro")

DEFAULT_TAX_RATE = 0.26
SAFE_INTEREST_COVERAGE = 100.0
TRADING_DAYS_YEAR = 252
MAX_CSV_ROWS = 100
MAX_WORKERS = 3
DEFAULT_RISK_FREE_RATE = 0.04
FX_TTL_SECONDS = 3600
RISK_FREE_TTL_SECONDS = 6 * 3600
ALTMAN_SAFE_THRESHOLD = 1.81
DEFAULT_BENCHMARK = "^GSPC"
DEFAULT_SMART_WEIGHTS = {"F": 0.40, "T": 0.30, "Q": 0.30}
TAX_LOSS_COMPENSATION_YEARS = 4

# DCF & value
DEFAULT_EQUITY_RISK_PREMIUM = 0.05
DEFAULT_STAGE1_GROWTH = 0.05
DEFAULT_STAGE2_GROWTH = 0.02
DEFAULT_STAGE1_YEARS = 5

# Bond / ETF / Crypto
DEFAULT_ETF_TER_THRESHOLD = 0.002
DEFAULT_ETF_TRACKING_ERROR_THRESHOLD = 0.01
CRYPTO_SENTIMENT_API = "https://api.alternative.me/fng/?limit=1"

# ---------------------------- SAFE SECRETS ---------------------------------
def safe_get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    env_val = os.getenv(key)
    if env_val:
        return env_val.strip()
    try:
        if key in st.secrets:
            return str(st.secrets[key]).strip()
    except Exception:
        pass
    return default

POLYGON_API_KEY = safe_get_secret("POLYGON_API_KEY")

# ---------------------------- PRICE & FX & RISK-FREE -------------------------
@st.cache_data(ttl=900, show_spinner=False)
def get_current_price_safe(ticker_symbol: str) -> float:
    symbol = (ticker_symbol or "").upper().strip()
    if not symbol:
        return 0.0
    if POLYGON_API_KEY and "." not in symbol and "-" not in symbol:
        try:
            url = f"https://api.polygon.io/v2/last/trade/{symbol}?apiKey={POLYGON_API_KEY}"
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200 and resp.json().get("status") == "OK":
                p = resp.json().get("results", {}).get("p")
                if p is not None:
                    return float(p)
        except Exception:
            pass
    for fallback in [YQ_Ticker, yf.Ticker]:
        try:
            if fallback == YQ_Ticker:
                yq = YQ_Ticker(symbol)
                price_data = yq.price.get(symbol, {})
                p = price_data.get('regularMarketPrice') or price_data.get('preMarketPrice')
            else:
                p = fallback(symbol).fast_info['last_price']
            if p is not None:
                return float(p)
        except Exception:
            continue
    return 0.0

@st.cache_data(ttl=FX_TTL_SECONDS, show_spinner=False)
def get_fx_rate(from_currency: str, to_currency: str) -> float:
    try:
        f = (from_currency or "").upper().strip()
        t = (to_currency or "").upper().strip()
        if not f or not t or f == t:
            return 1.0
        direct = yf.Ticker(f"{f}{t}=X").history(period="5d")
        if not direct.empty and 'Close' in direct:
            val = direct['Close'].dropna().iloc[-1]
            if val != 0:
                return float(val)
        inverse = yf.Ticker(f"{t}{f}=X").history(period="5d")
        if not inverse.empty and 'Close' in inverse:
            val = inverse['Close'].dropna().iloc[-1]
            if val != 0:
                return float(1.0 / val)
    except Exception:
        pass
    return 1.0

@st.cache_data(ttl=RISK_FREE_TTL_SECONDS, show_spinner=False)
def get_dynamic_risk_free_rate() -> float:
    try:
        irx = yf.Ticker("^IRX").history(period="5d")
        if not irx.empty and 'Close' in irx:
            last = irx['Close'].dropna().iloc[-1]
            if 0 <= last <= 15:
                return last / 100.0
    except Exception:
        pass
    return DEFAULT_RISK_FREE_RATE

def get_active_risk_free_rate() -> float:
    override = st.session_state.get("risk_free_override")
    if override is not None:
        return float(override)
    return get_dynamic_risk_free_rate()

# ---------------------------- AUTH SUPABASE (semplificata ma funzionante) -----------------
def init_auth_state():
    for key in ['auth_user', 'auth_session', 'auth_error']:
        if key not in st.session_state:
            st.session_state[key] = None

def get_logged_user_email():
    user = st.session_state.get('auth_user')
    if user and isinstance(user, dict):
        return user.get('email')
    return None

def is_authenticated():
    return get_logged_user_email() is not None

def get_logged_user_id():
    user = st.session_state.get('auth_user')
    if user and isinstance(user, dict):
        return user.get('id')
    return None

def get_supabase_client() -> Optional[Client]:
    url = safe_get_secret('SUPABASE_URL')
    key = safe_get_secret('SUPABASE_ANON_KEY')
    if url and key:
        try:
            return create_client(url, key)
        except Exception:
            pass
    return None

def is_supabase_available() -> bool:
    return get_supabase_client() is not None

def render_auth_sidebar():
    st.sidebar.markdown('### 👤 Account')
    if not is_supabase_available():
        st.sidebar.info("Auth disabilitata: configura SUPABASE_URL e SUPABASE_ANON_KEY")
        st.sidebar.markdown('---')
        return
    email = get_logged_user_email()
    if email:
        st.sidebar.success(f'Connesso: {email}')
        if st.sidebar.button('🚪 Logout'):
            supabase = get_supabase_client()
            if supabase:
                supabase.auth.sign_out()
            st.session_state.auth_user = None
            st.rerun()
    else:
        email_in = st.sidebar.text_input('Email')
        pwd = st.sidebar.text_input('Password', type='password')
        if st.sidebar.button('Login'):
            supabase = get_supabase_client()
            if supabase:
                try:
                    resp = supabase.auth.sign_in_with_password({'email': email_in, 'password': pwd})
                    st.session_state.auth_user = resp.user
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Login fallito: {e}")
    st.sidebar.markdown('---')

# ---------------------------- PORTFOLIO HELPERS (salvataggio su Supabase) -----------------
def load_user_portfolio():
    user_id = get_logged_user_id()
    if not user_id:
        return
    supabase = get_supabase_client()
    if supabase is None:
        return
    try:
        rows = supabase.table('portfoliopositions').select('*').eq('userid', user_id).execute().data or []
    except Exception:
        return
    st.session_state.portfolio_tickers = []
    st.session_state.holdings = {}
    st.session_state.holdings_quantity = {}
    st.session_state.holdings_pmc = {}
    st.session_state.holdings_currency = {}
    for r in rows:
        t = str(r.get('ticker', '')).upper()
        if not t:
            continue
        qty = float(r.get('quantity', 0) or 0)
        pmc = float(r.get('pmc', 0) or 0)
        cur = str(r.get('currency', 'USD')).upper()
        st.session_state.portfolio_tickers.append(t)
        st.session_state.holdings_quantity[t] = qty
        st.session_state.holdings_pmc[t] = pmc
        st.session_state.holdings_currency[t] = cur
        st.session_state.holdings[t] = qty * pmc

def save_user_portfolio_position(ticker, quantity, pmc, currency):
    user_id = get_logged_user_id()
    if not user_id:
        return
    supabase = get_supabase_client()
    if supabase is None:
        return
    try:
        supabase.table('portfoliopositions').upsert({
            'userid': user_id, 'ticker': ticker.upper(), 'quantity': float(quantity),
            'pmc': float(pmc), 'currency': currency.upper()
        }, on_conflict='userid,ticker').execute()
    except Exception:
        pass

def delete_user_portfolio_position(ticker):
    user_id = get_logged_user_id()
    if not user_id:
        return
    supabase = get_supabase_client()
    if supabase is None:
        return
    try:
        supabase.table('portfoliopositions').delete().eq('userid', user_id).eq('ticker', ticker.upper()).execute()
    except Exception:
        pass

# ---------------------------- MODELLO DATI ESTESO --------------------------------
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
    debt_to_equity: Optional[float]
    revenue_growth: Optional[float]
    net_margin: Optional[float]
    fcf_margin: Optional[float]
    currency: str
    raw_data: Dict[str, Any] = field(default_factory=dict)
    normalized_roic: Optional[float] = None
    intrinsic_value_dcf: Optional[float] = None
    margin_of_safety: Optional[float] = None
    sector: Optional[str] = None
    altman_z_comment: Optional[str] = None
    asset_class: str = "Stock"
    moat_score: Optional[float] = None
    moat_reasons: List[str] = field(default_factory=list)
    opportunity_cost: Dict[str, Any] = field(default_factory=dict)
    bond_metrics: Dict[str, Any] = field(default_factory=dict)
    etf_metrics: Dict[str, Any] = field(default_factory=dict)
    crypto_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_ui_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "raw_data"}
        d["Moat Reasons"] = ", ".join(self.moat_reasons)
        return d

# ---------------------------- UTILITY GENERICHE ---------------------------------
def sanitize_ticker(ticker: str) -> str:
    clean = str(ticker or "").strip().upper()
    if not clean:
        raise ValueError("Ticker vuoto")
    if not re.match(r'^[A-Z0-9\-\.\=\^]+$', clean):
        raise ValueError(f"Caratteri non validi: {clean}")
    if len(clean) > 20:
        raise ValueError("Ticker troppo lungo")
    return clean

def normalize_ticker(ticker: str, suffix: str) -> str:
    clean = sanitize_ticker(ticker)
    if "-" in clean or clean.startswith("^"):
        return clean
    s = suffix.strip().upper()
    if s and not clean.endswith(s):
        return f"{clean}{s}"
    return clean

def safe_float(value, default=np.nan):
    try:
        if value is None:
            return default
        return float(value)
    except:
        return default

def is_non_traditional_asset(ticker: str, info: Optional[Dict] = None) -> bool:
    t = ticker.upper()
    if t.startswith("^") or "=X" in t or t.endswith("=F") or "-USD" in t:
        return True
    if info:
        qt = str(info.get('quoteType', '')).upper()
        if qt in {"CRYPTOCURRENCY", "CURRENCY", "FUTURE", "INDEX", "ETF", "MUTUALFUND"}:
            return True
    return False

def detect_sector(info: Dict) -> str:
    sector = str(info.get('sector', '')).upper()
    if 'FINANCIAL' in sector or 'BANK' in sector:
        return 'Financial'
    if 'REAL ESTATE' in sector:
        return 'RealEstate'
    if 'TECHNOLOGY' in sector:
        return 'Technology'
    return 'Industrial'

def get_asset_class(ticker: str, info: Dict) -> str:
    t = ticker.upper()
    qt = str(info.get('quoteType', '')).upper()
    if qt in ['ETF', 'ETP', 'FUND']:
        return 'ETF'
    if qt in ['BOND', 'MUTUALFUND'] or t.endswith('.B'):
        return 'Bond'
    if qt == 'CRYPTOCURRENCY' or '-USD' in t or '-EUR' in t:
        return 'Crypto'
    if qt == 'INDEX' or t.startswith('^'):
        return 'Index'
    if qt == 'COMMODITY' or t in ['GC=F', 'CL=F', 'SI=F']:
        return 'Commodity'
    return 'Stock'

# ---------------------------- FUNZIONI AVANZATE (MOAT, DCF, BOND, ETF, CRYPTO) -----------------
def get_moat_score(info: Dict, fundamentals: Dict) -> Tuple[float, List[str]]:
    score, reasons = 0, []
    gm = fundamentals.get('gross_margin', 0) or 0
    if gm > 0.4:
        score += 30
        reasons.append(f"Margini lordi alti ({gm:.1%})")
    elif gm > 0.3:
        score += 15
        reasons.append(f"Margini lordi discreti ({gm:.1%})")
    roic = fundamentals.get('roic', 0) or 0
    if roic > 0.15:
        score += 30
        reasons.append(f"ROIC sostenibile ({roic:.1%})")
    elif roic > 0.10:
        score += 15
    rev_growth = fundamentals.get('revenue_growth')
    if rev_growth and rev_growth > 0.1:
        score += 20
        reasons.append(f"Crescita revenue >10% ({rev_growth:.1%})")
    industry = str(info.get('industry', '')).lower()
    if any(k in industry for k in ['software', 'luxury', 'pharma']):
        score += 20
        reasons.append(f"Barriere all'entrata ({industry})")
    return min(score, 100), reasons

def opportunity_cost(price: float, intrinsic_value: float) -> Dict:
    if price <= 0 or intrinsic_value <= 0:
        return {"verdict": "N/D"}
    try:
        tnx = yf.Ticker("^TNX").history(period="5d")['Close'].dropna().iloc[-1] / 100.0
    except:
        tnx = get_active_risk_free_rate() + 0.02
    earnings_yield = intrinsic_value / price
    if earnings_yield > tnx + 0.03:
        verdict = "Azione preferibile a bond"
    elif earnings_yield < tnx - 0.01:
        verdict = "Bond preferibili"
    else:
        verdict = "Equivalenti"
    return {"bond_10y": tnx, "stock_yield": earnings_yield, "verdict": verdict, "spread": earnings_yield - tnx}

def analyze_bond(info: Dict, price: float) -> Dict:
    coupon = info.get('couponRate', 0.05)
    years = 5.0
    try:
        mat = info.get('maturityDate')
        if mat:
            from datetime import datetime
            years = (datetime.strptime(mat, '%Y-%m-%d') - datetime.now()).days / 365.25
    except:
        pass
    ytm = (coupon + (100 - price)/years) / ((100+price)/2) if price>0 else 0.04
    duration = years / (1+ytm) if ytm>0 else years
    rating = info.get('creditRating', 'BBB')
    spread = ytm - get_active_risk_free_rate()
    return {"ytm": ytm, "modified_duration": duration, "credit_rating": rating, "spread": spread}

def analyze_etf(info: Dict) -> Dict:
    ter = info.get('totalExpenseRatio', 0.002)
    tracking = info.get('trackingError', 0.01)
    replica = info.get('replicationStrategy', 'Fisica')
    holdings = info.get('numberOfHoldings', 100)
    pe = info.get('peRatio', 15.0)
    quality = 0
    if ter <= DEFAULT_ETF_TER_THRESHOLD:
        quality += 30
    elif ter <= 0.005:
        quality += 15
    if tracking <= DEFAULT_ETF_TRACKING_ERROR_THRESHOLD:
        quality += 30
    elif tracking <= 0.02:
        quality += 15
    if replica.lower() == 'fisica':
        quality += 20
    if holdings > 50:
        quality += 10
    if pe < 20:
        quality += 10
    return {"TER": ter, "tracking_error": tracking, "replication": replica, "num_holdings": holdings,
            "avg_pe": pe, "quality_score": quality, "verdict": "BUY" if quality>=70 else ("HOLD" if quality>=50 else "SELL")}

def analyze_crypto(ticker: str, df: pd.DataFrame) -> Dict:
    if df is None or df.empty:
        return {"error": "No data"}
    returns = df['Close'].pct_change().dropna()
    vol = returns.std() * np.sqrt(252)
    close = df['Close']
    sma50 = close.rolling(50).mean().iloc[-1] if len(close)>=50 else close.iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1] if len(close)>=200 else close.iloc[-1]
    trend = "Rialzista" if sma50 > sma200 else "Ribassista"
    # Fear & Greed
    sentiment = "Neutrale"
    try:
        r = requests.get(CRYPTO_SENTIMENT_API, timeout=3)
        if r.status_code == 200:
            val = int(r.json().get('data', [{}])[0].get('value', 50))
            if val <= 25:
                sentiment = "Estrema paura"
            elif val <= 45:
                sentiment = "Paura"
            elif val <= 55:
                sentiment = "Neutrale"
            elif val <= 75:
                sentiment = "Avidità"
            else:
                sentiment = "Estrema avidità"
    except:
        pass
    return {"volatility": vol, "trend": trend, "sentiment": sentiment, "risk_score": 100 if vol>0.8 else (70 if vol>0.5 else 40)}

# ---------------------------- ROIC NORMALIZZATO, DCF, ALTMAN Z -----------------------
def get_bs_item(bs: pd.DataFrame, names: List[str], default=0.0):
    if bs is None or bs.empty:
        return default
    for n in names:
        if n in bs.index:
            try:
                v = bs.loc[n].iloc[0]
                if pd.notna(v) and v != 0:
                    return float(v)
            except:
                continue
    return default

def normalized_invested_capital(bs: pd.DataFrame, info: Dict) -> float:
    debt = get_bs_item(bs, ['Total Debt', 'TotalDebt'], 0.0)
    equity = get_bs_item(bs, ['Stockholders Equity', 'TotalEquity'], 0.0)
    cash = get_bs_item(bs, ['Cash And Cash Equivalents', 'Cash'], 0.0)
    goodwill = get_bs_item(bs, ['Goodwill', 'GoodWill'], 0.0)
    assets = get_bs_item(bs, ['Total Assets', 'TotalAssets'], 1.0)
    excess_cash = max(0.0, cash - 0.02 * assets)
    return max(0.0, debt + equity - excess_cash - goodwill)

def calc_normalized_roic(ebit: float, tax_rate: float, invested_cap_norm: float) -> float:
    if invested_cap_norm <= 0 or ebit <= 0:
        return 0.0
    return ebit * (1 - tax_rate) / invested_cap_norm

def calculate_dcf(fcf: float, stage1_growth=0.05, years=5, terminal_growth=0.02, discount_rate=None) -> float:
    if fcf <= 0:
        return 0.0
    if discount_rate is None:
        discount_rate = get_active_risk_free_rate() + DEFAULT_EQUITY_RISK_PREMIUM
    if discount_rate <= terminal_growth:
        discount_rate = terminal_growth + 0.01
    pv = 0.0
    fcf_est = fcf
    for t in range(1, years+1):
        fcf_est *= (1 + stage1_growth)
        pv += fcf_est / ((1+discount_rate)**t)
    terminal = fcf_est * (1 + terminal_growth) / (discount_rate - terminal_growth)
    return pv + terminal / ((1+discount_rate)**years)

def margin_of_safety(price: float, iv: float) -> Optional[float]:
    if iv <= 0 or price <= 0:
        return None
    return (iv - price) / iv

def altman_z_adjusted(bs: pd.DataFrame, fin: pd.DataFrame, info: Dict, sector: str) -> Tuple[Any, str]:
    if sector in ('Financial', 'RealEstate'):
        return None, "Altman Z non applicabile a banche/assicurazioni"
    try:
        ta = get_bs_item(bs, ['Total Assets'], 0.0)
        if ta <= 0:
            return None, "Total assets zero"
        wc = get_bs_item(bs, ['Current Assets']) - get_bs_item(bs, ['Current Liabilities'])
        re = get_bs_item(bs, ['Retained Earnings'])
        ebit = get_bs_item(fin, ['EBIT'])
        mc = info.get('marketCap', 0.0)
        tl = get_bs_item(bs, ['Total Liabilities Net Minority Interest', 'Total Liabilities'])
        rev = info.get('totalRevenue', 0.0) or get_bs_item(fin, ['Total Revenue'])
        if tl <= 0 or mc == 0:
            return None, "Dati insufficienti"
        z = (1.2*wc/ta + 1.4*re/ta + 3.3*ebit/ta + 0.6*mc/tl + 1.0*rev/ta)
        return float(z), "OK"
    except Exception as e:
        return None, str(e)

# ---------------------------- FETCH DATI FONDAMENTALI (YFINANCE + YAHOOQUERY) -----------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_fundamental_data(symbol: str) -> Optional[Dict]:
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        if info and ('symbol' in info or 'shortName' in info):
            if 'symbol' not in info:
                info['symbol'] = symbol
            return {"info": info, "financials": ticker.financials, "balance_sheet": ticker.balance_sheet,
                    "cashflow": ticker.cashflow, "symbol": symbol}
    except:
        pass
    try:
        yq = YQ_Ticker(symbol)
        summary = yq.summary_detail.get(symbol, {})
        price = yq.price.get(symbol, {})
        fin_data = yq.financial_data.get(symbol, {})
        info = {**summary, **price, **fin_data, 'symbol': symbol}
        if 'regularMarketPrice' in info:
            info['currentPrice'] = info['regularMarketPrice']
        def fmt(df):
            if isinstance(df, pd.DataFrame) and not df.empty:
                df = df.copy()
                if isinstance(df.index, pd.MultiIndex):
                    try:
                        df = df.xs(symbol, level=0)
                    except:
                        return pd.DataFrame()
                if 'asOfDate' in df.columns:
                    df.set_index('asOfDate', inplace=True)
                return df.transpose()
            return pd.DataFrame()
        inc = fmt(yq.income_statement())
        bal = fmt(yq.balance_sheet())
        cf = fmt(yq.cash_flow())
        return {"info": info, "financials": inc, "balance_sheet": bal, "cashflow": cf, "symbol": symbol}
    except Exception as e:
        logger.error(f"Fundamental data fallito per {symbol}: {e}")
        return None

def calculate_fundamental_metrics(raw_data: Dict) -> Optional[FundamentalMetrics]:
    try:
        info = raw_data["info"]
        symbol = raw_data["symbol"]
        asset_class = get_asset_class(symbol, info)
        if asset_class in ('Crypto', 'Currency', 'Index', 'Commodity'):
            return FundamentalMetrics(ticker=symbol, company_name=info.get('shortName', symbol),
                price=safe_float(info.get('regularMarketPrice') or info.get('currentPrice'),0.0), fcf=0.0, roic=0.0,
                peg_ratio=None, peg_source="N/A", pe_ratio=None, interest_coverage=SAFE_INTEREST_COVERAGE,
                debt_to_equity=None, revenue_growth=None, net_margin=None, fcf_margin=None,
                currency=info.get('currency','USD'), raw_data=raw_data, asset_class=asset_class)
        fin = raw_data["financials"]
        bs = raw_data["balance_sheet"]
        cf = raw_data["cashflow"]
        def first(df, idx, default=0.0):
            if df is None or df.empty or idx not in df.index:
                return default
            try:
                return safe_float(df.loc[idx].iloc[0], default)
            except:
                return default
        op_cf = first(cf, 'Operating Cash Flow', 0.0)
        cap_ex = first(cf, 'Capital Expenditure', 0.0)
        fcf = op_cf - abs(cap_ex)
        debt = first(bs, 'Total Debt', 0.0)
        equity = first(bs, 'Stockholders Equity', np.nan)
        ebit = first(fin, 'EBIT', 0.0)
        tax_rate = DEFAULT_TAX_RATE
        if 'Tax Provision' in fin.index and 'Pretax Income' in fin.index:
            pretax = first(fin, 'Pretax Income', 0.0)
            tax_prov = first(fin, 'Tax Provision', 0.0)
            if pretax > 0:
                tax_rate = np.clip(tax_prov / pretax, 0.0, 1.0)
        invested = debt + equity if not np.isnan(equity) else debt
        roic = (ebit * (1-tax_rate))/invested if invested>0 else 0.0
        norm_inv = normalized_invested_capital(bs, info)
        norm_roic = calc_normalized_roic(ebit, tax_rate, norm_inv) if norm_inv>0 else None
        pe = info.get('trailingPE')
        growth = info.get('earningsGrowth')
        peg = info.get('pegRatio')
        peg_src = "N/A"
        if peg is not None:
            peg_src = "Official"
        elif pe and pe>0 and growth and growth>0:
            peg = pe / (growth*100)
            peg_src = "Estimated"
        int_exp = first(fin, 'Interest Expense', 0.0)
        int_cov = ebit/abs(int_exp) if int_exp!=0 else SAFE_INTEREST_COVERAGE
        rev = info.get('totalRevenue')
        net_inc = info.get('netIncomeToCommon')
        rev_growth = info.get('revenueGrowth')
        de = None
        if equity and not np.isnan(equity) and equity!=0:
            de = debt/equity
        net_margin = None
        if rev and rev>0 and net_inc:
            net_margin = net_inc/rev
        fcf_margin = None
        if rev and rev>0:
            fcf_margin = fcf/rev
        price = safe_float(info.get('currentPrice') or info.get('regularMarketPrice'),0.0)
        intrinsic = calculate_dcf(fcf) if fcf>0 else 0.0
        mos = margin_of_safety(price, intrinsic) if intrinsic>0 else None
        sector = detect_sector(info)
        altman_z, altman_comm = altman_z_adjusted(bs, fin, info, sector)
        # Moat
        funda = {'gross_margin': info.get('grossMargins'), 'roic': roic, 'revenue_growth': rev_growth}
        moat_score, moat_reasons = get_moat_score(info, funda)
        opp_cost = opportunity_cost(price, intrinsic) if intrinsic>0 else {}
        bond_metrics = analyze_bond(info, price) if asset_class == 'Bond' else {}
        etf_metrics = analyze_etf(info) if asset_class == 'ETF' else {}
        crypto_metrics = analyze_crypto(symbol, None) if asset_class == 'Crypto' else {}
        return FundamentalMetrics(ticker=symbol, company_name=info.get('longName', symbol), price=price, fcf=fcf,
            roic=roic, peg_ratio=peg, peg_source=peg_src, pe_ratio=pe, interest_coverage=int_cov,
            debt_to_equity=de, revenue_growth=rev_growth, net_margin=net_margin, fcf_margin=fcf_margin,
            currency=info.get('currency','USD'), raw_data=raw_data, normalized_roic=norm_roic,
            intrinsic_value_dcf=intrinsic if intrinsic>0 else None, margin_of_safety=mos, sector=sector,
            altman_z_comment=altman_comm, asset_class=asset_class, moat_score=moat_score, moat_reasons=moat_reasons,
            opportunity_cost=opp_cost, bond_metrics=bond_metrics, etf_metrics=etf_metrics, crypto_metrics=crypto_metrics)
    except Exception as e:
        logger.error(f"Errore calcolo metriche {raw_data.get('symbol', '?')}: {e}")
        return None

def fetch_metrics_for_ticker(ticker: str):
    try:
        raw = get_fundamental_data(ticker)
        if not raw:
            return ticker, None, "nessun dato"
        met = calculate_fundamental_metrics(raw)
        if not met:
            return ticker, None, "calcolo metriche fallito"
        return ticker, met.to_ui_dict(), None
    except Exception as e:
        return ticker, None, str(e)

def fetch_metrics_batch(tickers: List[str]):
    results, errors = [], []
    if not tickers:
        return results, errors
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(tickers))) as ex:
        futures = {ex.submit(fetch_metrics_for_ticker, t): t for t in tickers}
        for f in as_completed(futures):
            t, ui, err = f.result()
            if err:
                errors.append(f"{t}: {err}")
            elif ui:
                results.append(ui)
    return results, errors

# ---------------------------- DATI TECNICI ---------------------------------
@st.cache_data(ttl=900, show_spinner=False)
def get_technical_data(symbol: str) -> Optional[pd.DataFrame]:
    sym = normalize_ticker(symbol, "")
    try:
        df = yf.download(sym, period="2y", interval="1d", progress=False, auto_adjust=True)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.loc[:, ~df.columns.duplicated()]
            if len(df) >= 60:
                return df
    except:
        pass
    try:
        yq = YQ_Ticker(sym)
        df = yq.history(period="2y", interval="1d")
        if isinstance(df, pd.DataFrame) and not df.empty:
            if isinstance(df.index, pd.MultiIndex):
                try:
                    df = df.xs(sym, level=0)
                except:
                    try:
                        df = df.xs(symbol, level=0)
                    except:
                        return None
            df.columns = [c.capitalize() for c in df.columns]
            if len(df) >= 60:
                return df
    except:
        pass
    return None

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    close = pd.to_numeric(data['Close'], errors='coerce')
    data['SMA_50'] = close.rolling(50, min_periods=50).mean()
    data['SMA_200'] = close.rolling(200, min_periods=200).mean()
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    data['RSI'] = 100 - (100 / (1+rs))
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std(ddof=0)
    data['BB_Lower'] = ma20 - 2*std20
    data['BB_Upper'] = ma20 + 2*std20
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    data['MACD'] = ema12 - ema26
    data['MACD_signal'] = data['MACD'].ewm(span=9).mean()
    if ta:
        try:
            data['SMA_50'] = ta.sma(close, 50)
            data['SMA_200'] = ta.sma(close, 200)
            data['RSI'] = ta.rsi(close, 14)
            bb = ta.bbands(close, 20, 2)
            if bb is not None:
                data['BB_Lower'] = bb.iloc[:,0]
                data['BB_Upper'] = bb.iloc[:,2]
            macd = ta.macd(close)
            if macd is not None:
                data['MACD'] = macd.iloc[:,0]
                data['MACD_signal'] = macd.iloc[:,1]
        except:
            pass
    return data

def calculate_timing_score(data: pd.DataFrame, current_price: float) -> Tuple[int, List[str]]:
    score, reasons = 0, []
    last = data.iloc[-1]
    sma200 = last.get('SMA_200')
    if pd.notna(sma200) and current_price > sma200:
        score += 30
        reasons.append("✅ Sopra SMA200")
    else:
        reasons.append("⚠️ Sotto SMA200")
    rsi = last.get('RSI')
    if pd.notna(rsi):
        if rsi < 30:
            score += 30
            reasons.append("✅ RSI ipervenduto")
        elif rsi > 70:
            score -= 10
            reasons.append("🛑 RSI ipercomprato")
    bb_low = last.get('BB_Lower')
    if pd.notna(bb_low) and current_price <= bb_low*1.02:
        score += 20
        reasons.append("✅ Prezzo su banda inferiore")
    macd = last.get('MACD')
    sig = last.get('MACD_signal')
    if pd.notna(macd) and pd.notna(sig):
        if macd > sig:
            score += 10
            reasons.append("✅ MACD bullish")
        else:
            reasons.append("⚠️ MACD bearish")
    return int(np.clip(score, 0, 100)), reasons

# ---------------------------- METRICHE QUANT ---------------------------------
def calculate_quant_metrics(df: pd.DataFrame, risk_free: Optional[float] = None) -> Dict:
    rf = risk_free if risk_free is not None else get_active_risk_free_rate()
    ret = df['Close'].pct_change().dropna()
    excess = ret - (rf/TRADING_DAYS_YEAR)
    sharpe = (excess.mean()/excess.std())*np.sqrt(TRADING_DAYS_YEAR) if excess.std()!=0 else 0.0
    vol = ret.std()*np.sqrt(TRADING_DAYS_YEAR)
    logret = np.log(df['Close']/df['Close'].shift(1)).dropna()
    if len(logret)>=2:
        x = np.arange(len(logret)).reshape(-1,1)
        model = LinearRegression().fit(x, logret.cumsum().values.reshape(-1,1))
        r2 = model.score(x, logret.cumsum().values.reshape(-1,1))
        slope = float(model.coef_[0][0])
    else:
        r2, slope = np.nan, np.nan
    return {"Sharpe Ratio": sharpe, "Annual Volatility": vol, "R-Squared": r2, "Trend Slope": slope,
            "Risk Free Used": rf}

def calculate_risk_metrics(df: pd.DataFrame) -> Dict:
    ret = df['Close'].pct_change().dropna()
    if ret.empty:
        return {"Max Drawdown": np.nan, "CAGR": np.nan, "VaR_95": np.nan, "Sortino": np.nan, "Calmar": np.nan}
    eq = (1+ret).cumprod()
    dd = eq/eq.cummax() - 1
    max_dd = dd.min()
    total_ret = eq.iloc[-1] - 1
    years = len(ret)/TRADING_DAYS_YEAR
    cagr = (1+total_ret)**(1/years)-1 if years>0 else np.nan
    var95 = np.quantile(ret, 0.05)
    rf_daily = get_active_risk_free_rate()/TRADING_DAYS_YEAR
    downside = ret[ret<rf_daily]
    downside_dev = downside.std()*np.sqrt(TRADING_DAYS_YEAR) if not downside.empty else np.nan
    sortino = (ret.mean()-rf_daily)*TRADING_DAYS_YEAR / downside_dev if downside_dev and downside_dev>0 else np.nan
    calmar = cagr/abs(max_dd) if max_dd<0 and not np.isnan(cagr) else np.nan
    return {"Max Drawdown": max_dd, "CAGR": cagr, "VaR_95": var95, "Sortino": sortino, "Calmar": calmar}

def monte_carlo_equity(df: pd.DataFrame, n_paths=1000, horizon=252):
    ret = df['Close'].pct_change().dropna().values
    if len(ret)==0:
        return None
    idx = np.random.randint(0, len(ret), size=(n_paths, horizon))
    paths = (1+ret[idx]).cumprod(axis=1)
    return {"paths": paths, "q05": np.quantile(paths,0.05,axis=0), "q50": np.quantile(paths,0.5,axis=0), "q95": np.quantile(paths,0.95,axis=0)}

# ---------------------------- SMART QUANT SCORE ---------------------------------
def compute_smart_quant_score(row, timing_score, qm, risk, weights=None):
    w = weights or DEFAULT_SMART_WEIGHTS
    f_score = 0.0
    roic = row.get("ROIC", 0.0) or 0.0
    f_score += np.clip((roic-0.10)/(0.25-0.10),0,1)*30
    peg = row.get("PEG Ratio")
    if peg and peg>0:
        if peg<=1: f_score+=25
        elif peg<=2: f_score+=12
    de = row.get("Debt/Equity")
    if de is not None:
        if de<=0.5: f_score+=12
        elif de<=1.0: f_score+=7
        elif de<=2.0: f_score+=3
    rev_g = row.get("Revenue Growth")
    if rev_g:
        if rev_g>=0.15: f_score+=6
        elif rev_g>=0.05: f_score+=3
        elif rev_g>0: f_score+=1.5
    net_m = row.get("Net Margin")
    if net_m:
        if net_m>=0.20: f_score+=8
        elif net_m>=0.10: f_score+=4
        elif net_m>0: f_score+=2
    fcf_m = row.get("FCF Margin")
    if fcf_m:
        if fcf_m>=0.15: f_score+=12
        elif fcf_m>=0.08: f_score+=7
        elif fcf_m>0: f_score+=3
    t_score = np.clip(timing_score,0,100)
    q_score = 0.0
    sharpe = qm.get("Sharpe Ratio",0.0) or 0.0
    if sharpe>0:
        if sharpe<=1: q_score = 30*sharpe
        elif sharpe<=2: q_score = 30+30*(sharpe-1)
        else: q_score = 80
    max_dd = risk.get("Max Drawdown",0.0) or 0.0
    if max_dd<-0.5: q_score -=20
    elif max_dd<-0.3: q_score -=10
    q_score = np.clip(q_score,0,100)
    smart = w["F"]*f_score + w["T"]*t_score + w["Q"]*q_score
    return {"SmartScore": np.clip(smart,0,100), "FundamentalScore": f_score, "TechnicalScore": t_score, "QuantRiskScore": q_score}

# ---------------------------- PORTAFOGLIO (sintetico ma completo) -----------------
def enrich_portfolio_with_fx(df_weights, base_currency="EUR"):
    if df_weights.empty:
        return df_weights
    out = df_weights.copy()
    out["FX Rate"] = out["Valuta"].apply(lambda c: get_fx_rate(c, base_currency))
    out["Importo Investito Base"] = out["Importo Investito"] * out["FX Rate"]
    out["Valore di Mercato Base"] = out["Valore di Mercato"] * out["FX Rate"]
    out["P&L Base"] = out["P&L"] * out["FX Rate"]
    tot = out["Valore di Mercato Base"].sum()
    out["Peso Base %"] = out["Valore di Mercato Base"] / tot * 100 if tot>0 else 0
    return out

def calculate_tax_with_loss_offset(df_base, tax_rate):
    if df_base.empty:
        return df_base, {}
    gains = df_base["P&L Base"].clip(lower=0).sum()
    losses = -df_base["P&L Base"].clip(upper=0).sum()
    compensable = min(gains, losses)
    taxable = max(0, gains - compensable)
    tax = taxable * tax_rate
    return df_base, {"Plusvalenze": gains, "Minusvalenze": losses, "Compensate": compensable, "Imposta": tax}

def build_portfolio_returns(tickers, weights_pct):
    series = []
    for t in tickers:
        df = get_technical_data(t)
        if df is not None:
            ret = df['Close'].pct_change().dropna()
            ret.name = t
            series.append(ret)
    if not series:
        return None, None
    df_ret = pd.concat(series, axis=1, join="inner").dropna()
    w = np.array([weights_pct.get(t,0.0) for t in df_ret.columns])/100.0
    w = w/w.sum()
    port_ret = (df_ret * w).sum(axis=1)
    return df_ret, port_ret

def calculate_portfolio_metrics(port_ret):
    if port_ret is None or port_ret.empty:
        return {}
    rf = get_active_risk_free_rate()
    mu = port_ret.mean() * TRADING_DAYS_YEAR
    sigma = port_ret.std() * np.sqrt(TRADING_DAYS_YEAR)
    sharpe = (mu - rf)/sigma if sigma>0 else np.nan
    eq = (1+port_ret).cumprod()
    dd = eq/eq.cummax() - 1
    max_dd = dd.min()
    cagr = eq.iloc[-1]**(TRADING_DAYS_YEAR/len(port_ret)) - 1
    return {"AnnRet": mu, "AnnVol": sigma, "Sharpe": sharpe, "MaxDD": max_dd, "CAGR": cagr}

def infer_asset_class(ticker, name=""):
    t = ticker.upper()
    if any(k in name.lower() for k in ["etf","ucits","ishares"]): return "ETF"
    if any(k in t for k in ["-USD","-EUR"]): return "Crypto"
    if t.endswith((".MI",".DE",".PA",".L")): return "Azione"
    return "Azione"

def build_portfolio_allocation_df(positive_holdings, holdings_currency):
    rows = []
    for t, amt in positive_holdings.items():
        raw = get_fundamental_data(t)
        info = raw["info"] if raw else {}
        name = info.get("longName", t)
        cur = holdings_currency.get(t, info.get("currency","USD"))
        rows.append({"Ticker": t, "Company Name": name, "Importo": amt, "Valuta": cur, "Asset Class": infer_asset_class(t, name)})
    df = pd.DataFrame(rows)
    if not df.empty:
        tot = df["Importo"].sum()
        df["Peso %"] = df["Importo"]/tot*100
    return df

def summarize_group_weights(df_alloc, group_col):
    if df_alloc.empty:
        return pd.DataFrame()
    gb = df_alloc.groupby(group_col)["Importo"].sum().reset_index()
    gb["Peso %"] = gb["Importo"]/gb["Importo"].sum()*100
    return gb.sort_values("Importo", ascending=False)

def compute_rebalancing_actions(df_alloc, target_weights, group_col="Ticker", tol=1.0):
    if df_alloc.empty:
        return pd.DataFrame()
    cur = summarize_group_weights(df_alloc, group_col)
    if cur.empty:
        return pd.DataFrame()
    target_df = pd.DataFrame({group_col: list(target_weights.keys()), "Target %": list(target_weights.values())})
    merged = cur.merge(target_df, on=group_col, how="outer").fillna(0.0)
    total = df_alloc["Importo"].sum()
    merged["Scostamento %"] = merged["Target %"] - merged["Peso %"]
    merged["Azione €"] = total * merged["Scostamento %"] / 100.0
    merged["Azione"] = np.where(merged["Azione €"] > tol, "Compra", np.where(merged["Azione €"] < -tol, "Riduci", "In target"))
    return merged

# ---------------------------- INTERFACCIA STREAMLIT (SIDEBAR + MAIN) -----------------
st.set_page_config(page_title="V-Quant Pro", layout="wide")
inject_pwa_support = lambda: None  # semplificato

def setup_sidebar():
    render_auth_sidebar()
    with st.sidebar.expander("⚙️ Impostazioni"):
        base_cur = st.selectbox("Valuta base", ["EUR","USD","GBP"], index=0, key="base_currency")
        rf_mode = st.radio("Risk-free", ["Dinamico","Manuale","Default 4%"], index=0)
        if rf_mode == "Manuale":
            rf_man = st.number_input("Tasso %", 0.0,15.0,4.0,0.1)
            st.session_state["risk_free_override"] = rf_man/100.0
        elif rf_mode == "Default 4%":
            st.session_state["risk_free_override"] = 0.04
        else:
            st.session_state["risk_free_override"] = None
        st.caption(f"Risk-free effettivo: {get_active_risk_free_rate()*100:.2f}%")
        st.markdown("**Pesi Smart Quant**")
        wF = st.number_input("F",0.0,1.0,0.4,0.05)
        wT = st.number_input("T",0.0,1.0,0.3,0.05)
        wQ = st.number_input("Q",0.0,1.0,0.3,0.05)
        s = wF+wT+wQ
        if s>0:
            st.session_state["smart_weights"] = {"F":wF/s, "T":wT/s, "Q":wQ/s}
    st.sidebar.header("1. Asset")
    mode = st.sidebar.radio("Input", ["Manuale","Batch CSV"], horizontal=True)
    file = None
    manual = None
    if mode == "Batch CSV":
        file = st.sidebar.file_uploader("CSV (colonna 'Ticker')", type=["csv"])
    else:
        manual = st.sidebar.text_input("Ticker", "AAPL").upper().strip()
    st.sidebar.header("2. Mercato")
    market = st.sidebar.selectbox("Borsa", ["USA","Italia (.MI)","Germania (.DE)","Francia (.PA)","GB (.L)","Crypto"])
    suffix_map = {"Italia":".MI","Germania":".DE","Francia":".PA","GB":".L"}
    suffix = suffix_map.get(market.split()[0], "")
    analizza = st.sidebar.button("🚀 Avvia Analisi", width='stretch')
    with st.sidebar.expander("Parametri"):
        cfg = {"roic": st.number_input("Min ROIC %",10.0), "peg": st.number_input("Max PEG",1.5),
               "custom_max_de": st.number_input("Max D/E",1.0), "custom_min_fcf_margin": st.number_input("Min FCF margin %",8.0)/100,
               "custom_min_net_margin": st.number_input("Min Net margin %",10.0)/100,
               "model_mode": st.selectbox("Modello verdetto", ["Entrambi","Classico","Evoluto","Personalizzabile","Value Investing","Universale"])}
    render_apk_download_box()
    return {"mode": mode, "file": file, "manual": manual, "suffix": suffix, "btn": analizza, "cfg": cfg, "base_currency": st.session_state.get("base_currency","EUR")}

def render_apk_download_box():
    with st.sidebar.expander("Download APK"):
        st.link_button("📲 Scarica APK", "https://github.com/innovativeprogram/V-QuantPro-relaases/releases/download/v1.0.0/Vquantpro.apk", width='stretch')

def _init_session_state():
    for k in ['batch_results', 'selected_ticker', 'portfolio_tickers', 'holdings', 'holdings_quantity',
              'holdings_pmc', 'holdings_currency', 'analysis_errors', 'standalone_ticker_input',
              'vq_ai_history', 'vq_ai_symbol', 'vq_ai_live_context', 'ai_ticker_chat_history']:
        if k not in st.session_state:
            st.session_state[k] = [] if 'history' in k or 'errors' in k else ({} if 'holdings' in k else None)

def resolve_active_analysis_target():
    ticker = st.session_state.get('selected_ticker')
    batch = st.session_state.get('batch_results')
    if ticker and batch is not None and not batch.empty and 'Ticker' in batch.columns:
        row = batch[batch['Ticker'] == ticker].iloc[0] if ticker in batch['Ticker'].values else None
        return ticker, row, None, 'batch'
    manual = st.session_state.get('standalone_ticker_input', '')
    if manual:
        raw = get_fundamental_data(manual)
        if raw:
            met = calculate_fundamental_metrics(raw)
            if met:
                return manual, pd.Series(met.to_ui_dict()), raw, 'standalone'
    return None, None, None, 'none'

def main():
    init_auth_state()
    _init_session_state()
    st.title("💲 V-Quant Pro v3.0 - Analisi Universale")
    ui = setup_sidebar()
    if is_authenticated() and not st.session_state.get('portfolio_loaded_from_db'):
        load_user_portfolio()
        st.session_state.portfolio_loaded_from_db = True
    if ui["btn"]:
        targets = [ui["manual"]] if ui["mode"] == "Manuale" else []
        if ui["mode"] == "Batch CSV" and ui["file"]:
            try:
                csv = pd.read_csv(ui["file"])
                if 'Ticker' in csv.columns:
                    targets = csv['Ticker'].dropna().astype(str).tolist()[:MAX_CSV_ROWS]
            except:
                pass
        if targets:
            normalized = [normalize_ticker(t, ui["suffix"]) for t in targets if t]
            with st.spinner(f"Analisi di {len(normalized)} ticker..."):
                results, errors = fetch_metrics_batch(normalized)
            st.session_state.batch_results = pd.DataFrame(results)
            st.session_state.analysis_errors = errors
            if results:
                st.session_state.selected_ticker = results[0]["Ticker"]
    ticker, row, raw_data, source = resolve_active_analysis_target()
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Fondamentali","📉 Tecnico","⚛️ Quant","⚖️ Verdetto","📁 Portafoglio"])
    with tab1:
        if row is not None:
            st.dataframe(pd.DataFrame([row.drop('_raw_data', errors='ignore')]))
            st.caption(f"Asset Class: {row.get('Asset Class', 'N/D')} | Moat Score: {row.get('Moat Score', 'N/D')}")
        else:
            st.info("Nessun ticker selezionato")
    with tab2:
        if ticker:
            df_tech = get_technical_data(ticker)
            if df_tech is not None:
                df_ind = calculate_technical_indicators(df_tech)
                score, _ = calculate_timing_score(df_ind, df_ind['Close'].iloc[-1])
                st.metric("Timing Score", f"{score}/100")
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7,0.3])
                fig.add_trace(go.Candlestick(x=df_ind.index, open=df_ind['Open'], high=df_ind['High'], low=df_ind['Low'], close=df_ind['Close']), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['SMA_200'], name="SMA200"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['RSI'], name="RSI"), row=2, col=1)
                fig.update_layout(height=600, template="plotly_dark")
                st.plotly_chart(fig)
            else:
                st.warning("Dati tecnici non disponibili")
        else:
            st.info("Nessun ticker")
    with tab3:
        if ticker and row is not None:
            df_tech = get_technical_data(ticker)
            if df_tech is not None:
                qm = calculate_quant_metrics(df_tech)
                risk = calculate_risk_metrics(df_tech)
                st.metric("Sharpe Ratio", f"{qm['Sharpe Ratio']:.2f}")
                st.metric("Max Drawdown", f"{risk['Max Drawdown']*100:.1f}%")
                st.metric("Sortino", f"{risk['Sortino']:.2f}")
                st.metric("Calmar", f"{risk['Calmar']:.2f}")
                mc = monte_carlo_equity(df_tech, 500, 252)
                if mc:
                    st.line_chart(pd.DataFrame({"Mediana": mc["q50"], "5° perc": mc["q05"], "95° perc": mc["q95"]}))
    with tab4:
        if ticker and row is not None:
            st.subheader("Verdetto Value Investing integrato")
            mos = row.get("Margin of Safety")
            intrinsic = row.get("Intrinsic Value (DCF)")
            st.metric("Valore Intrinseco DCF", f"{intrinsic:.2f}" if intrinsic else "N/D")
            st.metric("Margine di Sicurezza", f"{mos*100:.1f}%" if mos else "N/D")
            if row.get("Asset Class") == "Bond":
                bm = row.get("bond_metrics", {})
                st.metric("YTM", f"{bm.get('ytm',0)*100:.2f}%")
                st.metric("Duration", f"{bm.get('modified_duration',0):.1f}")
            elif row.get("Asset Class") == "ETF":
                em = row.get("etf_metrics", {})
                st.metric("ETF Quality", f"{em.get('quality_score',0)}/100")
                st.write(f"Verdetto ETF: {em.get('verdict','N/D')}")
            elif row.get("Asset Class") == "Crypto":
                cm = row.get("crypto_metrics", {})
                st.metric("Volatilità Crypto", f"{cm.get('volatility',0)*100:.1f}%")
                st.write(f"Sentiment: {cm.get('sentiment','N/D')}")
            else:
                # Modello value per azioni
                if mos and mos > 0.20:
                    st.success("🟢 BUY: Ampio margine di sicurezza")
                elif mos and mos > 0.10:
                    st.warning("🟡 HOLD: Margine di sicurezza limitato")
                else:
                    st.error("🔴 SELL: Prezzo troppo alto rispetto al valore")
        else:
            st.info("Nessun ticker")
    with tab5:
        st.info("Gestione portafoglio (leggero, ma funzionante)")
        if st.button("📥 Carica portafoglio salvato"):
            load_user_portfolio()
            st.rerun()
        portfolio_tickers = st.session_state.get('portfolio_tickers', [])
        if portfolio_tickers:
            st.write("Ticker in portafoglio:", ", ".join(portfolio_tickers))
            for t in portfolio_tickers:
                col1, col2, col3 = st.columns([1,1,0.5])
                qty = st.session_state.holdings_quantity.get(t, 0)
                pmc = st.session_state.holdings_pmc.get(t, 0)
                cur = st.session_state.holdings_currency.get(t, "USD")
                new_qty = col1.number_input(f"Qty {t}", value=float(qty), step=1.0, key=f"qty_{t}")
                new_pmc = col2.number_input(f"PMC {t}", value=float(pmc), step=0.5, key=f"pmc_{t}")
                if col3.button("🗑", key=f"del_{t}"):
                    delete_user_portfolio_position(t)
                    st.session_state.portfolio_tickers.remove(t)
                    st.rerun()
                st.session_state.holdings_quantity[t] = new_qty
                st.session_state.holdings_pmc[t] = new_pmc
                st.session_state.holdings_currency[t] = cur
                if is_authenticated():
                    save_user_portfolio_position(t, new_qty, new_pmc, cur)
        else:
            st.info("Portafoglio vuoto. Aggiungi dal tab fondamentali (pulsante 'Aggiungi al portafoglio') o carica.")
        # Quick add
        new_ticker = st.text_input("Aggiungi ticker manuale")
        if st.button("➕ Aggiungi"):
            if new_ticker:
                try:
                    t_clean = sanitize_ticker(new_ticker)
                    if t_clean not in st.session_state.portfolio_tickers:
                        st.session_state.portfolio_tickers.append(t_clean)
                        st.session_state.holdings_quantity[t_clean] = 0.0
                        st.session_state.holdings_pmc[t_clean] = 0.0
                        st.session_state.holdings_currency[t_clean] = "USD"
                        st.rerun()
                except Exception as e:
                    st.error(f"Ticker non valido: {e}")

if __name__ == "__main__":
    main()