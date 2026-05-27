"""
# Copyright (c) 2026 InnovativeProgram
# Tutti i diritti riservati.
# Proprietà intellettuale di [Canio Tedesco].
# La copia o distribuzione non autorizzata è severamente vietata.
#
# V-Quant Pro - Versione Unificata
# Include: menu laterale, grafici avanzati, metriche quantitative, ML, macro, ticker resolver, cascata dati
"""

import streamlit as st
import yfinance as yf
import requests
from yahooquery import Ticker as YQ_Ticker
import pandas as pd
import numpy as np
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import logging
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List
from sklearn.linear_model import LinearRegression
from supabase import create_client, Client

# Tentativi di import di librerie aggiuntive (opzionali)
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import pandas_datareader.data as web
    PDR_AVAILABLE = True
except ImportError:
    PDR_AVAILABLE = False

try:
    from ticker_classifier.classifier import TickerClassifier
    TICKER_CLASSIFIER_AVAILABLE = True
except ImportError:
    TICKER_CLASSIFIER_AVAILABLE = False

try:
    from openbb import obb
    OPENBB_AVAILABLE = True
except ImportError:
    OPENBB_AVAILABLE = False

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False

try:
    from defeatbeta_api.data.ticker import Ticker as DefeatTicker
    DEFEATBETA_AVAILABLE = True
except ImportError:
    DEFEATBETA_AVAILABLE = False

# Import prompts AI (file separato)
from vq_ai_prompts import (
    build_ai_context_for_ticker,
    ask_gemini_ticker_chat,
    build_burry_ai_context,
)

# ==========================================================================
# 0. SETUP LOGGING & COSTANTI GLOBALI
# ==========================================================================
if os.getenv("VQ_LOG_FILE"):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=os.getenv("VQ_LOG_FILE")
    )
else:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
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
# Pesi ottimizzati per Smart Quant Score (analista value)
SMART_WEIGHTS = {"F": 0.35, "V": 0.30, "Q": 0.20, "T": 0.15}
TAX_LOSS_COMPENSATION_YEARS = 4
BENEISH_THRESHOLD = -1.78

POLYGON_RATE_LIMIT_SEC = 12.0
_last_polygon_call = 0.0
_polygon_lock = threading.Lock()

def throttle_polygon():
    global _last_polygon_call
    with _polygon_lock:
        now = time.time()
        elapsed = now - _last_polygon_call
        if elapsed < POLYGON_RATE_LIMIT_SEC:
            time.sleep(POLYGON_RATE_LIMIT_SEC - elapsed)
        _last_polygon_call = time.time()

BATCH_RATE_LIMIT_SEC = 0.3

FOOTER_HTML = """
<p style='text-align:center;color:gray;'>
    Creato e sviluppato da <a href='https://www.vquantpro.it' target='_blank'>vquantpro.it</a>
</p>
"""

# ==========================================================================
# 0.A SAFE SECRETS / CONFIG ACCESS
# ==========================================================================
def safe_get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    env_val = os.getenv(key)
    if env_val:
        return env_val.strip()
    try:
        if key in st.secrets:
            val = st.secrets[key]
            return str(val).strip() if val is not None else default
    except Exception:
        pass
    return default

POLYGON_API_KEY = safe_get_secret("POLYGON_API_KEY", default=None)
POLYGON_BASE_URL = "https://api.polygon.io"

# ==========================================================================
# 0.B PRICE / FX / RISK-FREE PROVIDERS (con cascata)
# ==========================================================================
@st.cache_data(ttl=900, show_spinner=False)
def get_current_price_safe(ticker_symbol: str) -> float:
    symbol = (ticker_symbol or "").upper().strip()
    if not symbol:
        return 0.0

    # 1. Polygon
    if POLYGON_API_KEY and "." not in symbol and "-" not in symbol:
        try:
            throttle_polygon()
            url = f"{POLYGON_BASE_URL}/v2/last/trade/{symbol}?apiKey={POLYGON_API_KEY}"
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "OK" and "results" in data:
                    p = data["results"].get("p")
                    if p is not None:
                        return float(p)
        except Exception:
            pass

    # 2. yfinance
    try:
        t = yf.Ticker(symbol)
        price = t.fast_info.get('last_price')
        if price is not None:
            return float(price)
    except Exception:
        pass

    # 3. yahooquery
    try:
        yq = YQ_Ticker(symbol)
        price_data = yq.price.get(symbol, {}) if isinstance(yq.price, dict) else {}
        if isinstance(price_data, dict):
            p = price_data.get('regularMarketPrice') or price_data.get('preMarketPrice')
            if p is not None:
                return float(p)
    except Exception:
        pass

    # 4. OpenBB (se disponibile)
    if OPENBB_AVAILABLE:
        try:
            data = obb.equity.price.quote(symbol)
            if data and hasattr(data, 'results') and data.results:
                return float(data.results[0].last_price)
        except Exception:
            pass

    return 0.0

@st.cache_data(ttl=FX_TTL_SECONDS, show_spinner=False)
def get_fx_rate(from_currency: str, to_currency: str) -> float:
    try:
        f = str(from_currency or "").upper().strip()
        t = str(to_currency or "").upper().strip()
        if not f or not t or f == t:
            return 1.0
        direct = yf.Ticker(f"{f}{t}=X").history(period="5d", interval="1d")
        if direct is not None and not direct.empty and 'Close' in direct.columns:
            s = direct['Close'].dropna()
            if not s.empty:
                return safe_float(s.iloc[-1], 1.0)
        inverse = yf.Ticker(f"{t}{f}=X").history(period="5d", interval="1d")
        if inverse is not None and not inverse.empty and 'Close' in inverse.columns:
            s = inverse['Close'].dropna()
            if not s.empty:
                inv_val = safe_float(s.iloc[-1], 1.0)
                if inv_val != 0:
                    return 1.0 / inv_val
    except Exception:
        pass
    return 1.0

@st.cache_data(ttl=RISK_FREE_TTL_SECONDS, show_spinner=False)
def get_dynamic_risk_free_rate() -> float:
    try:
        irx = yf.Ticker("^IRX").history(period="5d")
        if irx is not None and not irx.empty and 'Close' in irx.columns:
            last = irx['Close'].dropna()
            if not last.empty:
                rf_pct = safe_float(last.iloc[-1], DEFAULT_RISK_FREE_RATE * 100)
                if 0 <= rf_pct <= 15:
                    return rf_pct / 100.0
    except Exception:
        pass
    return DEFAULT_RISK_FREE_RATE

def get_active_risk_free_rate() -> float:
    # Non più override da UI, solo dinamico
    return get_dynamic_risk_free_rate()

# ==========================================================================
# 0.C RISOLUZIONE AUTOMATICA TICKER
# ==========================================================================
def auto_resolve_ticker(symbol: str) -> Tuple[str, Optional[str]]:
    """Converte ticker semplice (es. ENI) in formato completo (ENI.MI)"""
    symbol_clean = symbol.upper().strip()
    
    # Se ha già suffisso, lascia invariato
    suffixes = ('.MI', '.DE', '.PA', '.L', '.MC', '.SW', '.TO', '.T', '.HK', '.AX', '.NS')
    if any(symbol_clean.endswith(suf) for suf in suffixes):
        return symbol_clean, None
    
    # Usa ticker-classifier se disponibile
    if TICKER_CLASSIFIER_AVAILABLE:
        try:
            classifier = TickerClassifier()
            result = classifier.classify([symbol_clean])
            if result and len(result) > 0:
                r = result[0]
                category = r.get('category', '').upper()
                yahoo_lookup = r.get('yahoo_lookup')
                if category == 'EQUITY' and yahoo_lookup:
                    return yahoo_lookup, r.get('company_profile', {}).get('exchange')
        except Exception as e:
            logger.debug(f"Ticker classifier error: {e}")
    
    # Mappa manuale per ticker italiani noti
    it_map = {
        'ENI': 'ENI.MI', 'STLAM': 'STLAM.MI', 'STLAP': 'STLAP.MI', 'STM': 'STM.MI',
        'ISP': 'ISP.MI', 'UCG': 'UCG.MI', 'TIT': 'TIT.MI', 'RACE': 'RACE.MI',
        'CNHI': 'CNHI.MI', 'PIRC': 'PIRC.MI', 'BAMI': 'BAMI.MI', 'NEXI': 'NEXI.MI',
        'MONC': 'MONC.MI', 'PRY': 'PRY.MI', 'FBK': 'FBK.MI', 'BPE': 'BPE.MI'
    }
    if symbol_clean in it_map:
        return it_map[symbol_clean], "MIL"
    
    # Altrimenti restituisci così com'è
    return symbol_clean, None

# ==========================================================================
# 1. MODELLI DATI
# ==========================================================================
@dataclass
class FundamentalMetrics:
    ticker: str
    company_name: str
    price: float
    fcf: float
    roic: float
    croic: float
    peg_ratio: Optional[float]
    peg_source: str
    pe_ratio: Optional[float]
    interest_coverage: float
    debt_to_equity: Optional[float]
    revenue_growth: Optional[float]
    net_margin: Optional[float]
    fcf_margin: Optional[float]
    fcf_yield: Optional[float]
    ev_to_ebit: Optional[float]
    ev_to_ebitda: Optional[float]
    price_to_book: Optional[float]
    price_to_sales: Optional[float]
    f_score: Optional[int]
    m_score: Optional[float]
    m_score_reliable: bool = True
    currency: str = "USD"
    raw_data: Dict[str, Any] = field(default_factory=dict)

    def to_ui_dict(self) -> Dict[str, Any]:
        return {
            "Ticker": self.ticker, "Company Name": self.company_name, "Price": self.price,
            "Free Cash Flow": self.fcf, "ROIC": self.roic, "CROIC": self.croic,
            "PEG Ratio": self.peg_ratio, "PEG Source": self.peg_source,
            "P/E Ratio": self.pe_ratio, "Interest Coverage": self.interest_coverage,
            "Debt/Equity": self.debt_to_equity, "Revenue Growth": self.revenue_growth,
            "Net Margin": self.net_margin, "FCF Margin": self.fcf_margin,
            "FCF Yield": self.fcf_yield, "EV/EBIT": self.ev_to_ebit,
            "EV/EBITDA": self.ev_to_ebitda, "Price/Book": self.price_to_book,
            "Price/Sales": self.price_to_sales, "F-Score": self.f_score,
            "Beneish M-Score": self.m_score, "M-Score reliable": self.m_score_reliable,
            "Currency": self.currency, "_raw_data": self.raw_data,
        }

# ==========================================================================
# 2. HELPER & VALIDAZIONE
# ==========================================================================
def sanitize_ticker(ticker: str) -> str:
    clean = str(ticker or '').strip().upper()
    if not clean:
        raise ValueError('Ticker vuoto')
    if not re.match(r'^[A-Z0-9\-\.=^]+$', clean):
        raise ValueError(f'Ticker non valido: {clean}')
    return clean

def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default

def is_non_traditional_asset(ticker: str, raw_info: Optional[Dict[str, Any]] = None) -> bool:
    t = (ticker or "").upper()
    if t.startswith("^") or "=X" in t or t.endswith("=F"):
        return True
    parts = t.split("-")
    if len(parts) == 2:
        crypto_symbols = {"BTC", "ETH", "XRP", "LTC", "BCH", "ADA", "DOT", "LINK", "XLM", "UNI", "SOL", "AVAX", "MATIC"}
        if parts[0] in crypto_symbols and parts[1] in {"USD", "EUR", "GBP", "JPY"}:
            return True
    if raw_info:
        qt = str(raw_info.get('quoteType', '')).upper()
        if qt in {"CRYPTOCURRENCY", "CURRENCY", "FUTURE", "INDEX", "ETF", "MUTUALFUND"}:
            return True
    return False


# ==========================================================================
# 3. DATA ENGINE: FONDAMENTALI CON CASCATA MULTIPLA
# ==========================================================================
def get_polygon_fundamentals(symbol: str) -> Optional[Dict[str, Any]]:
    if not POLYGON_API_KEY:
        return None
    try:
        throttle_polygon()
        resp = requests.get(f"{POLYGON_BASE_URL}/v3/reference/tickers/{symbol}?apiKey={POLYGON_API_KEY}", timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if data.get("status") != "OK":
            return None
        details = data.get("results", {})
        info = {
            "symbol": symbol,
            "longName": details.get("name", symbol),
            "shortName": details.get("name", symbol),
            "marketCap": details.get("market_cap"),
            "currency": details.get("currency_name", "USD"),
            "quoteType": "EQUITY",
            "sector": details.get("sic_description", ""),
            "industry": details.get("sic_description", ""),
        }
        throttle_polygon()
        fin_resp = requests.get(
            f"{POLYGON_BASE_URL}/vX/reference/financials",
            params={"ticker": symbol, "timeframe": "annual", "include_sources": "false", "limit": 4, "apiKey": POLYGON_API_KEY},
            timeout=15
        )
        if fin_resp.status_code != 200:
            return None
        fin_data = fin_resp.json()
        if fin_data.get("status") != "OK":
            return None
        filings = fin_data.get("results", [])
        INC_MAP = {"Total Revenue": ["Revenues"], "EBIT": ["Operating Income (Loss)"],
                   "Interest Expense": ["Interest Expense", "Interest Expense (Income)"],
                   "Pretax Income": ["Income (Loss) from Continuing Operations before Income Taxes, Noncontrolling Interest"],
                   "Tax Provision": ["Income Tax Expense (Benefit)"],
                   "Net Income": ["Net Income (Loss) Attributable to Parent", "Net Income (Loss)"],
                   "Cost Of Goods Sold": ["Cost of Goods and Services Sold"]}
        BAL_MAP = {"Total Assets": ["Assets"], "Current Assets": ["Assets, Current"],
                   "Current Liabilities": ["Liabilities, Current"],
                   "Stockholders Equity": ["Stockholders' Equity Attributable to Parent", "Stockholders' Equity"],
                   "Retained Earnings": ["Retained Earnings (Accumulated Deficit)"],
                   "Total Debt": ["Long-term Debt, Excluding Current Maturities", "Long-term Debt"]}
        CF_MAP = {"Operating Cash Flow": ["Net Cash Provided by (Used in) Operating Activities"],
                  "Capital Expenditure": ["Payments to Acquire Property, Plant, and Equipment"]}
        def build_df(stmt_type, mapping):
            df = pd.DataFrame()
            for filing in filings:
                fy = str(filing.get("fiscal_year", ""))
                if stmt_type in filing:
                    for item in filing[stmt_type]:
                        label = item.get("label")
                        val = item.get("value")
                        if val is None:
                            continue
                        for std, cand in mapping.items():
                            if label in cand:
                                df.loc[std, fy] = float(val)
            if df.empty:
                return pd.DataFrame()
            df = df[sorted(df.columns, key=lambda x: int(x), reverse=True)]
            return df
        inc_stmt = build_df("income_statement", INC_MAP)
        bal_sheet = build_df("balance_sheet", BAL_MAP)
        cash_flow = build_df("cash_flow_statement", CF_MAP)
        return {"info": info, "financials": inc_stmt, "balance_sheet": bal_sheet, "cashflow": cash_flow, "symbol": symbol}
    except Exception as e:
        logger.debug(f"Polygon fundamentals error: {e}")
        return None

def get_fundamental_data_defeatbeta(symbol: str) -> Optional[Dict[str, Any]]:
    if not DEFEATBETA_AVAILABLE:
        return None
    try:
        ticker = DefeatTicker(symbol)
        price_data = ticker.price()
        if price_data is None or price_data.empty:
            return None
        info = {
            'symbol': symbol,
            'shortName': symbol,
            'currentPrice': float(price_data['close'].iloc[-1]) if not price_data.empty else 0.0,
            'regularMarketPrice': float(price_data['close'].iloc[-1]) if not price_data.empty else 0.0,
        }
        try:
            inc = ticker.annual_income_statement()
            if inc is not None and not inc.empty:
                info['totalRevenue'] = inc.get('TotalRevenue', {}).get(symbol, {}).get('raw', 0.0)
                info['netIncome'] = inc.get('NetIncome', {}).get(symbol, {}).get('raw', 0.0)
        except Exception:
            pass
        try:
            bs = ticker.annual_balance_sheet()
            if bs is not None and not bs.empty:
                info['totalAssets'] = bs.get('TotalAssets', {}).get(symbol, {}).get('raw', 0.0)
                info['totalDebt'] = bs.get('TotalDebt', {}).get(symbol, {}).get('raw', 0.0)
        except Exception:
            pass
        return {"info": info, "financials": pd.DataFrame(), "balance_sheet": pd.DataFrame(), "cashflow": pd.DataFrame(), "symbol": symbol}
    except Exception as e:
        logger.debug(f"defeatbeta-api error: {e}")
        return None

def get_fundamental_data_akshare(symbol: str) -> Optional[Dict[str, Any]]:
    if not AKSHARE_AVAILABLE:
        return None
    try:
        if symbol.startswith('^') or '-USD' in symbol:
            return None
        # Solo per alcuni mercati
        if '.HK' in symbol:
            clean = symbol.replace('.HK', '')
            hk_info = ak.stock_hk_individual_info_em(symbol=clean)
            if hk_info is not None and not hk_info.empty:
                info_dict = dict(zip(hk_info['item'], hk_info['value']))
                info = {
                    'symbol': symbol,
                    'shortName': info_dict.get('股票简称', symbol),
                    'marketCap': safe_float(info_dict.get('总市值', 0)),
                }
                return {"info": info, "financials": pd.DataFrame(), "balance_sheet": pd.DataFrame(), "cashflow": pd.DataFrame(), "symbol": symbol}
        else:
            stock_info = ak.stock_individual_info_em(symbol=symbol)
            if stock_info is not None and not stock_info.empty:
                info_dict = dict(zip(stock_info['item'], stock_info['value']))
                info = {
                    'symbol': symbol,
                    'shortName': symbol,
                    'marketCap': safe_float(info_dict.get('总市值', 0)),
                    'pe_ratio': safe_float(info_dict.get('市盈率-动态', 0)),
                }
                return {"info": info, "financials": pd.DataFrame(), "balance_sheet": pd.DataFrame(), "cashflow": pd.DataFrame(), "symbol": symbol}
    except Exception as e:
        logger.debug(f"akshare error: {e}")
    return None

def get_fundamental_data(symbol: str) -> Optional[Dict[str, Any]]:
    # 1. Polygon
    poly = get_polygon_fundamentals(symbol)
    if poly and (not poly["financials"].empty or not poly["balance_sheet"].empty):
        return poly
    # 2. yfinance
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        if info and ('symbol' in info or 'shortName' in info):
            if 'symbol' not in info:
                info['symbol'] = symbol
            if info.get('totalRevenue') or info.get('netIncomeToCommon') or info.get('marketCap'):
                return {"info": info, "financials": stock.financials, "balance_sheet": stock.balance_sheet, "cashflow": stock.cashflow, "symbol": symbol}
    except Exception as e:
        logger.debug(f"yfinance fundamentals failed: {e}")
    # 3. yahooquery
    try:
        yq = YQ_Ticker(symbol)
        summary = yq.summary_detail.get(symbol, {}) if isinstance(yq.summary_detail, dict) else {}
        price = yq.price.get(symbol, {}) if isinstance(yq.price, dict) else {}
        fin_data = yq.financial_data.get(symbol, {}) if isinstance(yq.financial_data, dict) else {}
        combined = {**summary, **price, **fin_data}
        combined['symbol'] = symbol
        if 'regularMarketPrice' in combined:
            combined['currentPrice'] = combined['regularMarketPrice']
        if combined.get('totalRevenue') or combined.get('netIncomeToCommon') or combined.get('marketCap'):
            def fmt(df):
                if isinstance(df, pd.DataFrame) and not df.empty:
                    if isinstance(df.index, pd.MultiIndex):
                        try:
                            df = df.xs(symbol, level=0)
                        except KeyError:
                            return pd.DataFrame()
                    if 'asOfDate' in df.columns:
                        df.set_index('asOfDate', inplace=True)
                    df_t = df.transpose()
                    try:
                        date_cols = pd.to_datetime(df_t.columns, errors='coerce')
                        df_t = df_t.iloc[:, date_cols.argsort()[::-1]]
                    except Exception:
                        pass
                    return df_t
                return pd.DataFrame()
            inc = fmt(yq.income_statement())
            bal = fmt(yq.balance_sheet())
            cf = fmt(yq.cash_flow())
            return {"info": combined, "financials": inc, "balance_sheet": bal, "cashflow": cf, "symbol": symbol}
    except Exception as e:
        logger.debug(f"yahooquery fundamentals failed: {e}")
    # 4. defeatbeta
    defeat = get_fundamental_data_defeatbeta(symbol)
    if defeat:
        return defeat
    # 5. akshare
    ak_data = get_fundamental_data_akshare(symbol)
    if ak_data:
        return ak_data
    logger.error(f"Tutte le fonti fondamentali fallite per {symbol}")
    return None

def get_first(df: pd.DataFrame, idx: str, default: float = 0.0) -> float:
    if df is None or df.empty or idx not in df.index or df.shape[1] == 0:
        return default
    try:
        return safe_float(df.loc[idx].iloc[0], default)
    except Exception:
        return default

def calculate_piotroski_fscore(raw_data: Dict[str, Any]) -> int:
    info = raw_data.get('info', {})
    bs = raw_data.get('balance_sheet')
    fin = raw_data.get('financials')
    cf = raw_data.get('cashflow')
    if bs is None or fin is None or cf is None:
        return 0
    fscore = 0
    try:
        net_income = safe_float(info.get('netIncomeToCommon'), 0.0)
        total_assets = get_first(bs, 'Total Assets', 0.0)
        if total_assets > 0:
            roa = net_income / total_assets
            if roa > 0:
                fscore += 1
        op_cash = get_first(cf, 'Operating Cash Flow', 0.0)
        if total_assets > 0 and op_cash > 0:
            fscore += 1
        if 'Net Income' in fin.index and fin.shape[1] >= 2:
            ni0 = safe_float(fin.loc['Net Income'].iloc[0], 0.0)
            ni1 = safe_float(fin.loc['Net Income'].iloc[1], 0.0)
            if total_assets > 0 and (ni0 / total_assets) > (ni1 / total_assets):
                fscore += 1
        if total_assets > 0:
            accruals = (net_income - op_cash) / total_assets
            if accruals < 0:
                fscore += 1
        if 'Total Debt' in bs.index and bs.shape[1] >= 2:
            debt0 = safe_float(bs.loc['Total Debt'].iloc[0], 0.0)
            debt1 = safe_float(bs.loc['Total Debt'].iloc[1], 0.0)
            if debt0 < debt1:
                fscore += 1
        if 'Current Assets' in bs.index and 'Current Liabilities' in bs.index and bs.shape[1] >= 2:
            ca0 = safe_float(bs.loc['Current Assets'].iloc[0], 0.0)
            cl0 = safe_float(bs.loc['Current Liabilities'].iloc[1], 1.0)
            ca1 = safe_float(bs.loc['Current Assets'].iloc[1], 0.0)
            cl1 = safe_float(bs.loc['Current Liabilities'].iloc[1], 1.0)
            cr0 = ca0 / cl0 if cl0 != 0 else 0.0
            cr1 = ca1 / cl1 if cl1 != 0 else 0.0
            if cr0 > cr1:
                fscore += 1
        fscore += 1
        if 'Total Revenue' in fin.index and fin.shape[1] >= 2:
            rev0 = safe_float(fin.loc['Total Revenue'].iloc[0], 0.0)
            rev1 = safe_float(fin.loc['Total Revenue'].iloc[1], 0.0)
            cogs0 = safe_float(fin.loc.get('Cost Of Goods Sold', pd.Series([0])).iloc[0], 0.0) if 'Cost Of Goods Sold' in fin.index else 0.0
            cogs1 = safe_float(fin.loc.get('Cost Of Goods Sold', pd.Series([0])).iloc[1], 0.0) if 'Cost Of Goods Sold' in fin.index else 0.0
            gm0 = (rev0 - cogs0) / rev0 if rev0 != 0 else 0.0
            gm1 = (rev1 - cogs1) / rev1 if rev1 != 0 else 0.0
            if gm0 > gm1:
                fscore += 1
        if total_assets > 0 and bs.shape[1] >= 2:
            ta0 = total_assets
            ta1 = safe_float(bs.loc['Total Assets'].iloc[1], 0.0)
            if ta1 > 0:
                at0 = rev0 / ta0 if rev0 else 0.0
                at1 = rev1 / ta1 if rev1 else 0.0
                if at0 > at1:
                    fscore += 1
    except Exception as e:
        logger.debug(f"Piotroski F-Score error: {e}")
        fscore = 0
    return fscore

def calculate_beneish_mscore(raw_data: Dict[str, Any]) -> Tuple[Optional[float], bool]:
    info = raw_data.get('info', {})
    bs = raw_data.get('balance_sheet')
    fin = raw_data.get('financials')
    cf = raw_data.get('cashflow')
    if bs is None or fin is None or cf is None:
        return None, False
    try:
        if bs.shape[1] < 2 or fin.shape[1] < 2 or cf.shape[1] < 2:
            return None, False
        total_assets = get_first(bs, 'Total Assets', 0.0)
        cur_assets = get_first(bs, 'Current Assets', 0.0)
        cur_liab = get_first(bs, 'Current Liabilities', 0.0)
        cash = safe_float(info.get('totalCash', 0.0), 0.0)
        total_debt = get_first(bs, 'Total Debt', 0.0)
        revenue = safe_float(info.get('totalRevenue', 0.0), 0.0)
        cogs = safe_float(fin.loc.get('Cost Of Goods Sold', pd.Series([0])).iloc[0], 0.0) if 'Cost Of Goods Sold' in fin.index else 0.0
        net_income = safe_float(info.get('netIncomeToCommon', 0.0), 0.0)
        op_cash = get_first(cf, 'Operating Cash Flow', 0.0)
        depreciation = safe_float(info.get('depreciationExpense', 0.0), 0.0)
        total_assets_prev = safe_float(bs.loc['Total Assets'].iloc[1], total_assets)
        cur_assets_prev = safe_float(bs.loc['Current Assets'].iloc[1], cur_assets)
        cur_liab_prev = safe_float(bs.loc['Current Liabilities'].iloc[1], cur_liab)
        cash_prev = 0.0
        total_debt_prev = safe_float(bs.loc['Total Debt'].iloc[1], total_debt)
        revenue_prev = safe_float(fin.loc['Total Revenue'].iloc[1] if 'Total Revenue' in fin.index else revenue, revenue)
        cogs_prev = safe_float(fin.loc.get('Cost Of Goods Sold', pd.Series([0])).iloc[1], cogs) if 'Cost Of Goods Sold' in fin.index else cogs
        op_cash_prev = safe_float(cf.loc['Operating Cash Flow'].iloc[1] if 'Operating Cash Flow' in cf.index else op_cash, op_cash)
        depreciation_prev = safe_float(info.get('depreciationExpense', depreciation), depreciation)
        def sr(num, den, d=0.0):
            return num/den if den != 0 else d
        reliable = True
        if 'Cost Of Goods Sold' not in fin.index:
            reliable = False
        if 'Total Assets' not in bs.index:
            reliable = False
        dsri = sr((cur_assets - cash) / revenue, (cur_assets_prev - cash_prev) / revenue_prev)
        gmi = sr((revenue_prev - cogs_prev)/revenue_prev, (revenue - cogs)/revenue)
        aqi = sr(1 - (cur_assets + total_assets - cur_liab - cash)/total_assets, 1 - (cur_assets_prev + total_assets_prev - cur_liab_prev - cash_prev)/total_assets_prev)
        sgi = revenue / revenue_prev if revenue_prev != 0 else 1.0
        depi = sr(depreciation_prev/(depreciation_prev + total_assets_prev), depreciation/(depreciation + total_assets))
        sgai = 1.0
        lvgi = sr(total_debt/total_assets, total_debt_prev/total_assets_prev)
        tata = (net_income - op_cash) / total_assets
        m_score = -4.84 + 0.92*dsri + 0.528*gmi + 0.404*aqi + 0.892*sgi + 0.115*depi - 0.172*sgai + 4.679*tata - 0.327*lvgi
        return float(m_score), reliable
    except Exception as e:
        logger.debug(f"Beneish M-Score error: {e}")
        return None, False

def calculate_fundamental_metrics(raw_data: Dict[str, Any]) -> Optional[FundamentalMetrics]:
    try:
        info = raw_data["info"]
        symbol = raw_data["symbol"]
        if is_non_traditional_asset(symbol, info):
            return FundamentalMetrics(
                ticker=symbol, company_name=info.get('shortName', symbol),
                price=safe_float(info.get('regularMarketPrice') or info.get('currentPrice'), 0.0),
                fcf=0.0, roic=0.0, croic=0.0,
                peg_ratio=None, peg_source="N/A (non-equity)", pe_ratio=None,
                interest_coverage=SAFE_INTEREST_COVERAGE,
                debt_to_equity=None, revenue_growth=None, net_margin=None, fcf_margin=None,
                fcf_yield=None, ev_to_ebit=None, ev_to_ebitda=None, price_to_book=None, price_to_sales=None,
                f_score=None, m_score=None, m_score_reliable=False,
                currency=info.get('currency', 'USD'), raw_data=raw_data
            )
        fin = raw_data["financials"]
        bs = raw_data["balance_sheet"]
        cf = raw_data["cashflow"]
        op_cash = get_first(cf, 'Operating Cash Flow', 0.0)
        cap_ex = get_first(cf, 'Capital Expenditure', 0.0)
        fcf = float(op_cash - cap_ex)
        total_debt = get_first(bs, 'Total Debt', 0.0)
        cash_equiv = safe_float(info.get('totalCash'), 0.0)
        net_debt = max(0.0, total_debt - cash_equiv)
        equity = get_first(bs, 'Stockholders Equity', np.nan)
        invested_cap = net_debt + equity if not np.isnan(equity) and equity is not None else 0.0
        ebit = get_first(fin, 'EBIT', 0.0)
        tax_rate = DEFAULT_TAX_RATE
        if 'Tax Provision' in fin.index and 'Pretax Income' in fin.index and not fin.empty:
            pretax_inc = get_first(fin, 'Pretax Income', 0.0)
            tax_prov = get_first(fin, 'Tax Provision', 0.0)
            if pretax_inc > 0:
                tax_rate = float(np.clip(tax_prov / pretax_inc, 0.0, 1.0))
        roic = 0.0
        croic = 0.0
        if invested_cap > 0:
            nopat = ebit * (1.0 - tax_rate)
            roic = float(nopat / invested_cap)
            croic = float(fcf / invested_cap)
        pe = info.get('trailingPE')
        growth = info.get('earningsGrowth')
        peg = info.get('pegRatio')
        peg_src = "N/A"
        if peg is not None:
            peg_src = "Official"
        elif pe and pe > 0 and growth and growth > 0:
            peg = float(pe / (growth * 100))
            peg_src = "Estimated"
        int_exp = get_first(fin, 'Interest Expense', 0.0)
        int_cov = float(ebit / abs(int_exp)) if int_exp != 0 else SAFE_INTEREST_COVERAGE
        total_revenue = info.get('totalRevenue')
        net_income = info.get('netIncomeToCommon')
        revenue_growth = info.get('revenueGrowth')
        debt_to_equity = None
        if equity is not None and not np.isnan(equity) and equity != 0:
            debt_to_equity = float(total_debt / equity)
        net_margin = None
        if total_revenue not in (None, 0) and net_income is not None:
            net_margin = safe_float(net_income, 0.0) / float(total_revenue)
        fcf_margin = None
        if total_revenue not in (None, 0):
            fcf_margin = fcf / float(total_revenue)
        mc = info.get('marketCap')
        fcf_yield = None
        if mc and mc > 0 and fcf != 0:
            fcf_yield = fcf / float(mc)
        ev = info.get('enterpriseValue')
        ebitda = info.get('ebitda')
        ev_to_ebit = None
        if ev and ebit and ebit != 0:
            ev_to_ebit = ev / ebit
        ev_to_ebitda = None
        if ev and ebitda and ebitda != 0:
            ev_to_ebitda = ev / ebitda
        pb = info.get('priceToBook')
        ps = info.get('priceToSalesTrailing12Months')
        fscore = calculate_piotroski_fscore(raw_data)
        mscore, mscore_reliable = calculate_beneish_mscore(raw_data)
        return FundamentalMetrics(
            ticker=symbol, company_name=info.get('longName', symbol),
            price=safe_float(info.get('currentPrice') or info.get('regularMarketPrice'), 0.0),
            fcf=fcf, roic=roic, croic=croic,
            peg_ratio=safe_float(peg, None) if peg is not None else None, peg_source=peg_src,
            pe_ratio=safe_float(pe, None) if pe is not None else None,
            interest_coverage=int_cov, debt_to_equity=debt_to_equity,
            revenue_growth=safe_float(revenue_growth, None) if revenue_growth is not None else None,
            net_margin=net_margin, fcf_margin=fcf_margin,
            fcf_yield=fcf_yield, ev_to_ebit=ev_to_ebit, ev_to_ebitda=ev_to_ebitda,
            price_to_book=safe_float(pb, None) if pb is not None else None,
            price_to_sales=safe_float(ps, None) if ps is not None else None,
            f_score=fscore, m_score=mscore, m_score_reliable=mscore_reliable,
            currency=info.get('currency', 'USD'), raw_data=raw_data
        )
    except Exception as e:
        logger.error(f"Errore calcolo metriche {raw_data.get('symbol', '?')}: {e}")
        return None

def fetch_metrics_for_ticker(ticker: str) -> Tuple[str, Optional[Dict[str, Any]], Optional[str]]:
    try:
        raw = get_fundamental_data(ticker)
        if not raw:
            return ticker, None, "nessun dato fondamentale disponibile"
        met = calculate_fundamental_metrics(raw)
        if not met:
            return ticker, None, "impossibile calcolare le metriche"
        return ticker, met.to_ui_dict(), None
    except Exception as e:
        return ticker, None, str(e)

def fetch_metrics_batch(tickers: List[str]) -> Tuple[List[Dict[str, Any]], List[str]]:
    results = []
    errors = []
    if not tickers:
        return results, errors
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(fetch_metrics_for_ticker, t): t for t in tickers}
        for f in as_completed(futures):
            t, ui_dict, err = f.result()
            if err:
                errors.append(f"{t}: {err}")
            elif ui_dict is not None:
                results.append(ui_dict)
            time.sleep(BATCH_RATE_LIMIT_SEC)
    return results, errors

# ==========================================================================
# 4. DATA ENGINE: TECNICO E TIMING
# ==========================================================================
@st.cache_data(ttl=900, show_spinner=True)
def get_technical_data(symbol: str) -> Optional[pd.DataFrame]:
    # 1. Polygon
    if POLYGON_API_KEY:
        try:
            throttle_polygon()
            today = pd.Timestamp.now(tz='UTC')
            two_years_ago = today - pd.DateOffset(years=2)
            url = f"{POLYGON_BASE_URL}/v2/aggs/ticker/{symbol}/range/1/day/{two_years_ago.strftime('%Y-%m-%d')}/{today.strftime('%Y-%m-%d')}?adjusted=true&sort=asc&limit=5000&apiKey={POLYGON_API_KEY}"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "OK" and data.get("resultsCount", 0) > 0:
                    df = pd.DataFrame(data["results"])
                    df['t'] = pd.to_datetime(df['t'], unit='ms')
                    df = df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume', 't': 'Date'})
                    df.set_index('Date', inplace=True)
                    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                    if len(df) >= 60:
                        return df
        except Exception as e:
            logger.info(f"Polygon tech failed: {e}")
    # 2. yfinance
    try:
        df = yf.download(symbol, period="2y", interval="1d", progress=False, auto_adjust=True)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.loc[:, ~df.columns.duplicated(keep='first')]
            if len(df) >= 60:
                return df
    except Exception as e:
        logger.info(f"yfinance tech failed: {e}")
    # 3. yahooquery
    try:
        t = YQ_Ticker(symbol)
        df_yq = t.history(period="2y", interval="1d")
        if isinstance(df_yq, pd.DataFrame) and not df_yq.empty:
            if isinstance(df_yq.index, pd.MultiIndex):
                try:
                    df_yq = df_yq.xs(symbol, level=0)
                except KeyError:
                    return None
            df_yq.columns = [str(c).capitalize() for c in df_yq.columns]
            df_yq = df_yq.loc[:, ~df_yq.columns.duplicated(keep='first')]
            if len(df_yq) >= 60:
                return df_yq
    except Exception as e:
        logger.error(f"All tech sources failed for {symbol}: {e}")
    return None

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    close = pd.to_numeric(data['Close'], errors='coerce')
    high = pd.to_numeric(data['High'], errors='coerce')
    low = pd.to_numeric(data['Low'], errors='coerce')
    data['SMA_50'] = close.rolling(50, min_periods=50).mean()
    data['SMA_200'] = close.rolling(200, min_periods=200).mean()
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    data['RSI'] = 100 - (100 / (1 + rs))
    ma20 = close.rolling(20, min_periods=20).mean()
    std20 = close.rolling(20, min_periods=20).std(ddof=0)
    data['BB_Lower'] = ma20 - 2 * std20
    data['BB_Upper'] = ma20 + 2 * std20
    macd_ok = False
    if ta is not None:
        try:
            data['SMA_50'] = ta.sma(close, length=50)
            data['SMA_200'] = ta.sma(close, length=200)
            data['RSI'] = ta.rsi(close, length=14)
            bb = ta.bbands(close, length=20, std=2)
            if bb is not None:
                lowb = bb.filter(like='BBL_')
                upb = bb.filter(like='BBU_')
                if not lowb.empty:
                    data['BB_Lower'] = lowb.iloc[:, 0]
                if not upb.empty:
                    data['BB_Upper'] = upb.iloc[:, 0]
            macd = ta.macd(close)
            if macd is not None and not macd.empty:
                m_col = macd.filter(like='MACD_').filter(regex=r'_\d+_\d+_\d+$')
                s_col = macd.filter(like='MACDs_')
                if not m_col.empty:
                    data['MACD'] = m_col.iloc[:, 0]
                    macd_ok = True
                if not s_col.empty:
                    data['MACD_signal'] = s_col.iloc[:, 0]
            adx_df = ta.adx(high=high, low=low, close=close, length=14)
            if adx_df is not None:
                adx_col = adx_df.filter(like='ADX_')
                if not adx_col.empty:
                    data['ADX'] = adx_col.iloc[:, 0]
        except Exception:
            pass
    if not macd_ok:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        data['MACD'] = ema12 - ema26
        data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    return data

def calculate_timing_score(data: pd.DataFrame, current_price: float) -> Tuple[int, List[str]]:
    score, reasons = 0, []
    last_row = data.iloc[-1]
    sma200 = last_row.get('SMA_200')
    if pd.notna(sma200) and current_price > sma200:
        score += 30
        reasons.append("✅ Trend Rialzista (Sopra SMA 200)")
    else:
        reasons.append("⚠️ Trend Ribassista (Sotto SMA 200)")
    rsi = last_row.get('RSI')
    if pd.notna(rsi):
        if rsi < 30:
            score += 30
            reasons.append("✅ Ipervenduto (RSI < 30)")
        elif rsi > 70:
            score -= 10
            reasons.append("🛑 Ipercomprato (RSI > 70)")
    bb_lower = last_row.get('BB_Lower', np.nan)
    if pd.notna(bb_lower) and current_price <= bb_lower * 1.02:
        score += 20
        reasons.append("✅ Prezzo su Banda Bollinger Inferiore")
    macd = last_row.get('MACD')
    macd_sig = last_row.get('MACD_signal')
    if pd.notna(macd) and pd.notna(macd_sig):
        if macd > macd_sig:
            score += 10
            reasons.append("✅ MACD sopra signal line")
        else:
            reasons.append("⚠️ MACD sotto signal line")
    adx = last_row.get('ADX')
    if pd.notna(adx):
        if adx > 25:
            score += 10
            reasons.append("✅ Trend forte (ADX > 25)")
        elif adx < 20:
            score -= 5
            reasons.append("⚠️ Trend debole/assente (ADX < 20)")
    score = int(np.clip(score, 0, 100))
    return score, reasons

# ==========================================================================
# 5. METRICHE QUANTITATIVE AVANZATE (Omega, Jensen, Ulcer, GARCH)
# ==========================================================================
def omega_ratio(returns: pd.Series, rf_annual: float = 0.04, threshold: float = 0.0) -> float:
    """Omega Ratio: (probabilità di rendimenti sopra soglia) / (sotto soglia)"""
    if returns.empty or len(returns) < 2:
        return np.nan
    rf_daily = rf_annual / TRADING_DAYS_YEAR
    excess = returns - rf_daily - threshold
    gains = excess[excess > 0].sum()
    losses = -excess[excess < 0].sum()
    if losses == 0:
        return np.inf if gains > 0 else np.nan
    return gains / losses

def jensens_alpha(port_ret: pd.Series, bench_ret: pd.Series, rf_annual: float = 0.04) -> float:
    """Jensen's Alpha annualizzato"""
    if len(port_ret) < 30 or len(bench_ret) < 30:
        return np.nan
    rf_daily = rf_annual / TRADING_DAYS_YEAR
    port_excess = port_ret - rf_daily
    bench_excess = bench_ret - rf_daily
    cov = np.cov(port_excess, bench_excess)[0, 1]
    var_bench = bench_excess.var()
    if var_bench == 0:
        return np.nan
    beta = cov / var_bench
    alpha_daily = port_excess.mean() - beta * bench_excess.mean()
    alpha_ann = (1 + alpha_daily) ** TRADING_DAYS_YEAR - 1
    return alpha_ann

def information_ratio(port_ret: pd.Series, bench_ret: pd.Series) -> float:
    """Information Ratio (excess return vs benchmark / tracking error)"""
    if len(port_ret) < 30 or len(bench_ret) < 30:
        return np.nan
    active = port_ret - bench_ret
    if active.std() == 0:
        return np.nan
    return (active.mean() * TRADING_DAYS_YEAR) / (active.std() * np.sqrt(TRADING_DAYS_YEAR))

def ulcer_index(returns: pd.Series) -> float:
    """Ulcer Index: radice quadrata della media dei drawdown quadrati"""
    if returns.empty:
        return np.nan
    equity = (1 + returns).cumprod()
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max
    squared_dd = drawdown ** 2
    return np.sqrt(squared_dd.mean())

def fit_garch(returns: pd.Series) -> Optional[Dict[str, float]]:
    """Stima modello GARCH(1,1) e restituisce volatilità condizionale prevista"""
    if not ARCH_AVAILABLE or len(returns) < 100:
        return None
    try:
        returns_clean = returns.dropna() * 100  # scala percentuale
        model = arch_model(returns_clean, vol='Garch', p=1, q=1, dist='normal')
        res = model.fit(update_freq=0, disp='off')
        forecast = res.forecast(horizon=1)
        cond_vol = np.sqrt(forecast.variance.iloc[-1, 0]) / 100.0  # ritorno in scala decimale
        return {
            'garch_vol': cond_vol,
            'omega': res.params.get('omega', np.nan),
            'alpha': res.params.get('alpha[1]', np.nan),
            'beta': res.params.get('beta[1]', np.nan)
        }
    except Exception as e:
        logger.debug(f"GARCH fitting failed: {e}")
        return None

# ==========================================================================
# 6. MACHINE LEARNING (Random Forest segnale probabilistico)
# ==========================================================================
def prepare_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara feature per modello ML: rendimenti passati, volumi, volatilità rolling, RSI, MACD, ecc."""
    if df is None or len(df) < 50:
        return pd.DataFrame()
    close = df['Close'].copy()
    features = pd.DataFrame(index=df.index)
    features['ret_1'] = close.pct_change(1)
    features['ret_5'] = close.pct_change(5)
    features['ret_21'] = close.pct_change(21)
    features['volatility_21'] = features['ret_1'].rolling(21).std()
    features['volume_change'] = df['Volume'].pct_change()
    features['rsi'] = calculate_technical_indicators(df)['RSI']
    features['macd'] = calculate_technical_indicators(df)['MACD']
    features['sma_200_dist'] = (close / calculate_technical_indicators(df)['SMA_200'] - 1) * 100
    features['target'] = close.shift(-21) / close - 1  # forward return 21 giorni
    return features.dropna()

def train_rf_predictor(features: pd.DataFrame) -> Tuple[Any, Any, float]:
    """Addestra Random Forest e restituisce modello, scaler e ultimo score di training"""
    if not SKLEARN_AVAILABLE or features.empty or len(features) < 100:
        return None, None, np.nan
    try:
        X = features.drop('target', axis=1)
        y = features['target']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_scaled, y)
        score = model.score(X_scaled, y)  # R^2 in-sample
        return model, scaler, score
    except Exception as e:
        logger.debug(f"RF training failed: {e}")
        return None, None, np.nan

def predict_forward_return(model, scaler, latest_features: pd.DataFrame) -> Optional[float]:
    """Predice il rendimento a 21 giorni usando il modello addestrato"""
    if model is None or scaler is None or latest_features.empty:
        return None
    try:
        X_latest = scaler.transform(latest_features)
        pred = model.predict(X_latest)[0]
        return float(pred)
    except Exception as e:
        logger.debug(f"RF prediction failed: {e}")
        return None

# ==========================================================================
# 7. MACRO INDICATORI (Treasury, CPI)
# ==========================================================================
@st.cache_data(ttl=86400, show_spinner=False)
def get_macro_indicators() -> Dict[str, float]:
    """Recupera rendimento Treasury 10Y e CPI (USA)"""
    result = {"treasury_10y": np.nan, "cpi_yoy": np.nan}
    if not PDR_AVAILABLE:
        return result
    try:
        # Treasury 10Y (^TNX o DGS10)
        treasury = web.DataReader('DGS10', 'fred', start=pd.Timestamp.now() - pd.DateOffset(days=30))
        if not treasury.empty:
            result["treasury_10y"] = safe_float(treasury.iloc[-1, 0], np.nan) / 100.0
    except Exception:
        pass
    try:
        # CPI YoY (CPIAUCSL)
        cpi = web.DataReader('CPIAUCSL', 'fred', start=pd.Timestamp.now() - pd.DateOffset(months=13))
        if len(cpi) >= 13:
            cpi_series = cpi.iloc[:, 0]
            yoy = (cpi_series.iloc[-1] / cpi_series.iloc[-13] - 1) * 100
            result["cpi_yoy"] = safe_float(yoy, np.nan)
    except Exception:
        pass
    return result

# ==========================================================================
# 8. QUANT METRICS BASE (Sharpe, Sortino, Calmar, etc.)
# ==========================================================================
def calculate_quant_metrics(df: pd.DataFrame, fund_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    rf = get_active_risk_free_rate()
    returns = df['Close'].pct_change().dropna()
    excess_returns = returns - (rf / TRADING_DAYS_YEAR)
    sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(TRADING_DAYS_YEAR) if excess_returns.std() != 0 else 0.0
    vol = returns.std() * np.sqrt(TRADING_DAYS_YEAR) if not returns.empty else np.nan
    log_returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
    if len(log_returns) >= 2:
        cum_log = log_returns.cumsum().values.reshape(-1, 1)
        x = np.arange(len(cum_log)).reshape(-1, 1)
        model = LinearRegression().fit(x, cum_log)
        r_sq = model.score(x, cum_log)
        slope = float(model.coef_[0][0])
    else:
        r_sq = np.nan
        slope = np.nan
    z_score: Any = "N/A"
    z_score2: Any = "N/A"
    if fund_data and 'info' in fund_data and not is_non_traditional_asset(fund_data.get('symbol', ''), fund_data.get('info')):
        try:
            bs = fund_data.get("balance_sheet")
            fin = fund_data.get("financials")
            info = fund_data.get("info", {})
            if isinstance(bs, pd.DataFrame) and not bs.empty and isinstance(fin, pd.DataFrame) and not fin.empty:
                ta_val = bs.loc['Total Assets'].iloc[0] if 'Total Assets' in bs.index else 0
                if ta_val and ta_val > 0:
                    wc = (bs.loc['Current Assets'].iloc[0] - bs.loc['Current Liabilities'].iloc[0]) if 'Current Assets' in bs.index and 'Current Liabilities' in bs.index else 0
                    re_val = bs.loc['Retained Earnings'].iloc[0] if 'Retained Earnings' in bs.index else 0
                    ebit_val = fin.loc['EBIT'].iloc[0] if 'EBIT' in fin.index else 0
                    mc = info.get('marketCap')
                    tl = (bs.loc['Total Liabilities Net Minority Interest'].iloc[0] if 'Total Liabilities Net Minority Interest' in bs.index else (bs.loc['Total Liabilities'].iloc[0] if 'Total Liabilities' in bs.index else 0))
                    if mc is not None and tl and tl > 0:
                        rev = info.get('totalRevenue', 0) or 0
                        z_score = float((1.2 * (wc / ta_val)) + (1.4 * (re_val / ta_val)) + (3.3 * (ebit_val / ta_val)) + (0.6 * (mc / tl)) + (1.0 * (rev / ta_val)))
                        bv_equity = bs.loc['Stockholders Equity'].iloc[0] if 'Stockholders Equity' in bs.index else 0
                        if bv_equity:
                            z_score2 = float(6.56 * (wc / ta_val) + 3.26 * (re_val / ta_val) + 6.72 * (ebit_val / ta_val) + 1.05 * (bv_equity / tl))
        except Exception as e:
            logger.debug(f"Altman Z-Score error: {e}")
    return {"Sharpe Ratio": float(sharpe) if not np.isnan(sharpe) else 0.0,
            "Annual Volatility": float(vol) if not np.isnan(vol) else np.nan,
            "R-Squared": float(r_sq) if not np.isnan(r_sq) else np.nan,
            "Altman Z-Score": z_score,
            "Altman Z''-Score": z_score2,
            "Price Percentile": float((df['Close'] < df['Close'].iloc[-1]).mean() * 100),
            "Trend Slope": slope,
            "Risk Free Used": float(rf)}

def calculate_risk_metrics(df: pd.DataFrame) -> Dict[str, float]:
    prices = df['Close'].dropna()
    returns = prices.pct_change().dropna()
    if returns.empty:
        nan = float('nan')
        return {"Max Drawdown": nan, "CAGR": nan, "VaR_95": nan, "CVaR_95": nan, "Skew": nan, "Kurt": nan, "Sortino": nan, "Calmar": nan, "Downside Deviation": nan}
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
    rf_daily = get_active_risk_free_rate() / TRADING_DAYS_YEAR
    downside = returns[returns < rf_daily]
    downside_dev = downside.std() * np.sqrt(TRADING_DAYS_YEAR) if not downside.empty else np.nan
    ann_excess_ret = (returns.mean() - rf_daily) * TRADING_DAYS_YEAR
    sortino = ann_excess_ret / downside_dev if downside_dev and not np.isnan(downside_dev) and downside_dev > 0 else np.nan
    calmar = cagr / abs(max_dd) if max_dd and max_dd < 0 and not np.isnan(cagr) else np.nan
    # Metriche aggiuntive
    omega = omega_ratio(returns, rf_annual=get_active_risk_free_rate())
    ulcer = ulcer_index(returns)
    garch = fit_garch(returns)
    return {"Max Drawdown": float(max_dd), "CAGR": float(cagr) if not np.isnan(cagr) else np.nan,
            "VaR_95": float(var_95), "CVaR_95": float(cvar_95) if not np.isnan(cvar_95) else np.nan,
            "Skew": float(skew), "Kurt": float(kurt), "Sortino": float(sortino) if not np.isnan(sortino) else np.nan,
            "Calmar": float(calmar) if not np.isnan(calmar) else np.nan,
            "Downside Deviation": float(downside_dev) if not np.isnan(downside_dev) else np.nan,
            "Omega Ratio": omega, "Ulcer Index": ulcer,
            "GARCH Vol (next)": garch['garch_vol'] if garch else np.nan}

def monte_carlo_equity(df: pd.DataFrame, n_paths: int = 1000, horizon_days: int = 252, seed: Optional[int] = 42) -> Dict[str, Any]:
    prices = df['Close'].dropna()
    returns = prices.pct_change().dropna().values
    if returns.size == 0:
        return {"paths": None, "final_distribution": None, "q05": None, "q50": None, "q95": None}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(returns), size=(n_paths, horizon_days))
    sampled = returns[idx]
    equity_paths = (1 + sampled).cumprod(axis=1)
    final_values = equity_paths[:, -1]
    q05 = np.quantile(equity_paths, 0.05, axis=0)
    q50 = np.quantile(equity_paths, 0.50, axis=0)
    q95 = np.quantile(equity_paths, 0.95, axis=0)
    return {"paths": equity_paths, "final_distribution": final_values, "q05": q05, "q50": q50, "q95": q95}

# ==========================================================================
# 9. SMART SCORE E VERDETTO UNIFICATO
# ==========================================================================
def compute_smart_quant_score(row: Any, timing_score: int, qm: Dict[str, Any], risk: Dict[str, Any]) -> Dict[str, Any]:
    w = SMART_WEIGHTS
    f_score = 0.0
    roic = row.get("ROIC", 0.0) or 0.0
    f_score += float(np.clip((roic - 0.10) / (0.25 - 0.10), 0, 1)) * 30.0
    peg = row.get("PEG Ratio", None)
    if peg is not None and peg > 0:
        if peg <= 1:
            f_score += 25.0
        elif peg <= 2:
            f_score += 12.0
    debt_to_equity = row.get("Debt/Equity", None)
    if debt_to_equity is not None:
        if debt_to_equity <= 0.5:
            f_score += 12.0
        elif debt_to_equity <= 1.0:
            f_score += 7.0
        elif debt_to_equity <= 2.0:
            f_score += 3.0
    revenue_growth = row.get("Revenue Growth", None)
    if revenue_growth is not None:
        if revenue_growth >= 0.15:
            f_score += 6.0
        elif revenue_growth >= 0.05:
            f_score += 3.0
        elif revenue_growth > 0:
            f_score += 1.5
    net_margin = row.get("Net Margin", None)
    if net_margin is not None:
        if net_margin >= 0.20:
            f_score += 8.0
        elif net_margin >= 0.10:
            f_score += 4.0
        elif net_margin > 0:
            f_score += 2.0
    fcf_margin = row.get("FCF Margin", None)
    if fcf_margin is not None:
        if fcf_margin >= 0.15:
            f_score += 12.0
        elif fcf_margin >= 0.08:
            f_score += 7.0
        elif fcf_margin > 0:
            f_score += 3.0
    fscore_val = row.get("F-Score", 0) or 0
    if fscore_val >= 7:
        f_score += 12.0
    elif fscore_val >= 4:
        f_score += 6.0
    elif fscore_val >= 2:
        f_score += 3.0
    ev_ebit = row.get("EV/EBIT", None)
    if ev_ebit is not None and ev_ebit > 0:
        if ev_ebit <= 10:
            f_score += 10.0
        elif ev_ebit <= 15:
            f_score += 5.0
    fcf_yield = row.get("FCF Yield", None)
    if fcf_yield is not None:
        if fcf_yield >= 0.10:
            f_score += 8.0
        elif fcf_yield >= 0.05:
            f_score += 4.0
    mscore = row.get("Beneish M-Score", None)
    if mscore is not None:
        if mscore > -1.78:
            f_score -= 15.0
        elif mscore > -2.22:
            f_score -= 8.0
    if roic >= 0.15:
        f_score += 6.0
    elif roic >= 0.12:
        f_score += 3.0
    z = qm.get("Altman Z-Score", "N/A")
    if isinstance(z, (int, float, np.floating)) and not isinstance(z, bool):
        if z >= 3.0:
            f_score += 7.0
        elif z >= ALTMAN_SAFE_THRESHOLD:
            f_score += 3.5
    f_score = float(np.clip(f_score, 0, 100))
    t_score = float(np.clip(timing_score, 0, 100))
    q_score = 0.0
    sharpe = qm.get("Sharpe Ratio", 0.0) or 0.0
    max_dd = risk.get("Max Drawdown", 0.0) or 0.0
    sortino = risk.get("Sortino", 0.0) or 0.0
    omega = risk.get("Omega Ratio", 0.0) or 0.0
    if sharpe <= 0:
        q_score += 0.0
    elif sharpe <= 1:
        q_score += 30.0 * sharpe
    elif sharpe <= 2:
        q_score += 30.0 + 30.0 * (sharpe - 1.0)
    else:
        q_score += 80.0
    if sortino > 0:
        q_score += min(10.0, sortino * 10)
    if omega > 1.5:
        q_score += 10.0
    elif omega > 1.0:
        q_score += 5.0
    if isinstance(max_dd, (float, np.floating)):
        if max_dd < -0.5:
            q_score -= 20.0
        elif max_dd < -0.3:
            q_score -= 10.0
    q_score = float(np.clip(q_score, 0, 100))
    # Valuation score (V)
    v_score = 0.0
    pe = row.get("P/E Ratio", None)
    if pe is not None and pe > 0:
        if pe <= 15:
            v_score += 30
        elif pe <= 20:
            v_score += 15
    if peg is not None and peg > 0:
        if peg <= 1:
            v_score += 30
        elif peg <= 1.5:
            v_score += 15
    if fcf_yield is not None:
        if fcf_yield >= 0.08:
            v_score += 20
        elif fcf_yield >= 0.05:
            v_score += 10
    if ev_ebit is not None and ev_ebit > 0:
        if ev_ebit <= 10:
            v_score += 20
        elif ev_ebit <= 15:
            v_score += 10
    v_score = np.clip(v_score, 0, 100)
    smart = w["F"] * f_score + w["V"] * v_score + w["T"] * t_score + w["Q"] * q_score
    smart = float(np.clip(smart, 0, 100))
    return {"SmartScore": smart, "FundamentalScore": f_score, "TechnicalScore": t_score, "QuantRiskScore": q_score, "ValuationScore": v_score}

def compute_unified_verdict(row: pd.Series, timing_score: int, qm: Dict[str, Any], risk: Dict[str, Any], macro: Dict[str, float]) -> Dict[str, Any]:
    weights = {"F": 0.40, "V": 0.30, "T": 0.15, "Q": 0.15}
    fqs = 0.0
    details = []
    roic = safe_float(row.get("ROIC"), 0.0)
    if roic >= 0.10:
        fqs += 20
        details.append("✅ ROIC ≥ 10%")
    elif roic > 0:
        fqs += 10
        details.append("⚠️ ROIC positivo ma basso")
    croic = safe_float(row.get("CROIC"), 0.0)
    if croic >= 0.05:
        fqs += 10
        details.append("✅ CROIC ≥ 5%")
    fcf_margin = safe_float(row.get("FCF Margin"), None)
    if fcf_margin is not None and fcf_margin >= 0.08:
        fqs += 10
        details.append("✅ FCF Margin ≥ 8%")
    net_margin = safe_float(row.get("Net Margin"), None)
    if net_margin is not None and net_margin >= 0.10:
        fqs += 10
        details.append("✅ Net Margin ≥ 10%")
    de = safe_float(row.get("Debt/Equity"), None)
    if de is not None:
        if de <= 0.5:
            fqs += 10
            details.append("✅ D/E ≤ 0.5")
        elif de <= 1.0:
            fqs += 5
            details.append("⚠️ D/E ≤ 1.0")
    int_cov = safe_float(row.get("Interest Coverage"), SAFE_INTEREST_COVERAGE)
    if int_cov >= 5:
        fqs += 10
        details.append("✅ Int.Cover. > 5x")
    elif int_cov >= 3:
        fqs += 5
        details.append("⚠️ Int.Cover. > 3x")
    fscore = safe_float(row.get("F-Score"), 0.0)
    if fscore >= 7:
        fqs += 15
        details.append("✅ F‑Score ≥ 7 (eccellente)")
    elif fscore >= 4:
        fqs += 8
        details.append("⚠️ F‑Score ≥ 4 (discreto)")
    mscore = safe_float(row.get("Beneish M-Score"), None)
    mscore_reliable = row.get("M-Score reliable", True)
    if mscore is not None:
        if mscore <= -2.22:
            fqs += 10
            details.append("✅ M‑Score ≤ -2.22")
        elif mscore <= BENEISH_THRESHOLD:
            fqs += 5
            details.append("⚠️ M‑Score accettabile")
        else:
            fqs -= 20
            details.append("🛑 M‑Score > -1.78 (sospetto)")
        if not mscore_reliable:
            details.append("⚠️ M‑Score approssimativo (dati incompleti)")
    z = qm.get("Altman Z-Score", "N/A")
    if isinstance(z, (int, float)):
        if z >= 3.0:
            fqs += 5
            details.append("✅ Z‑Score > 3")
        elif z >= ALTMAN_SAFE_THRESHOLD:
            fqs += 2
            details.append("⚠️ Z‑Score > 1.81")
    rev_growth = safe_float(row.get("Revenue Growth"), None)
    if rev_growth is not None and rev_growth > 0.05:
        fqs += 5
        details.append("✅ Crescita ricavi > 5%")
    fqs = np.clip(fqs, 0, 100)
    vas = 0.0
    peg = safe_float(row.get("PEG Ratio"), None)
    if peg is not None and peg > 0:
        if peg <= 1.0:
            vas += 30
            details.append("✅ PEG ≤ 1")
        elif peg <= 1.5:
            vas += 20
            details.append("✅ PEG ≤ 1.5")
        else:
            vas += 5
    ev_ebit = safe_float(row.get("EV/EBIT"), None)
    if ev_ebit is not None and ev_ebit > 0:
        if ev_ebit <= 10:
            vas += 25
            details.append("✅ EV/EBIT ≤ 10")
        elif ev_ebit <= 15:
            vas += 15
            details.append("⚠️ EV/EBIT ≤ 15")
    fcf_yield = safe_float(row.get("FCF Yield"), None)
    if fcf_yield is not None:
        if fcf_yield >= 0.08:
            vas += 20
            details.append("✅ FCF Yield ≥ 8%")
        elif fcf_yield >= 0.04:
            vas += 10
            details.append("⚠️ FCF Yield ≥ 4%")
    pb = safe_float(row.get("Price/Book"), None)
    if pb is not None and pb > 0:
        if pb <= 1.5:
            vas += 15
            details.append("✅ P/B ≤ 1.5")
        elif pb <= 3.0:
            vas += 8
            details.append("⚠️ P/B ≤ 3")
    vas = np.clip(vas, 0, 100)
    tms = float(np.clip(timing_score, 0, 100))
    details.append(f"📈 Timing Score: {tms:.0f}/100")
    sharpe = qm.get("Sharpe Ratio", 0.0) or 0.0
    max_dd = risk.get("Max Drawdown", 0.0) or 0.0
    sortino = risk.get("Sortino", 0.0) or 0.0
    qrs = 50
    if sharpe > 0:
        qrs += min(20, sharpe * 20)
    if sortino > 0:
        qrs += min(15, sortino * 15)
    if max_dd < -0.50:
        qrs -= 25
    elif max_dd < -0.30:
        qrs -= 10
    qrs = np.clip(qrs, 0, 100)
    details.append(f"📉 Rischio Quant (Sharpe {sharpe:.2f}, MaxDD {max_dd*100:.1f}%)")
    # Macro aggiunte
    treasury = macro.get("treasury_10y", np.nan)
    if not np.isnan(treasury):
        details.append(f"🏦 Treasury 10Y: {treasury*100:.2f}%")
    cpi = macro.get("cpi_yoy", np.nan)
    if not np.isnan(cpi):
        details.append(f"📈 CPI YoY: {cpi:.2f}%")
    final_score = weights["F"] * fqs + weights["V"] * vas + weights["T"] * tms + weights["Q"] * qrs
    if final_score >= 75:
        verdict = "Strong Buy"
        emoji = "🟢"
    elif final_score >= 60:
        verdict = "Buy"
        emoji = "🟢"
    elif final_score >= 45:
        verdict = "Hold"
        emoji = "🟡"
    elif final_score >= 30:
        verdict = "Reduce"
        emoji = "🟠"
    else:
        verdict = "Sell"
        emoji = "🔴"
    return {"FinalScore": final_score, "Verdict": verdict, "Emoji": emoji,
            "FQS": fqs, "VAS": vas, "TMS": tms, "QRS": qrs,
            "Details": details, "Weights": weights}

# ==========================================================================
# 10. PORTAFOGLIO E UTILITY
# ==========================================================================
def enrich_portfolio_with_fx(df_weights: pd.DataFrame, base_currency: str = "EUR") -> pd.DataFrame:
    if df_weights is None or df_weights.empty:
        return pd.DataFrame()
    out = df_weights.copy()
    base = str(base_currency or "EUR").upper().strip()
    out["Valuta Base"] = base
    out["FX Rate"] = out["Valuta"].apply(lambda c: float(get_fx_rate(c, base)))
    out["Importo Investito Base"] = out["Importo Investito"].astype(float) * out["FX Rate"]
    out["Valore di Mercato Base"] = out["Valore di Mercato"].astype(float) * out["FX Rate"]
    out["P&L Base"] = out["P&L"].astype(float) * out["FX Rate"]
    total_base_mv = float(out["Valore di Mercato Base"].sum())
    out["Peso Base %"] = np.where(total_base_mv > 0, out["Valore di Mercato Base"] / total_base_mv * 100.0, 0.0)
    return out

def calculate_tax_impact(df_weights: pd.DataFrame, tax_rate: float = DEFAULT_TAX_RATE) -> pd.DataFrame:
    if df_weights is None or df_weights.empty:
        return pd.DataFrame()
    df_tax = df_weights.copy()
    df_tax["Aliquota Fiscale %"] = tax_rate * 100.0
    df_tax["Plus/Minus Lorda"] = df_tax["P&L"].astype(float)
    df_tax["Imposta Teorica"] = np.where(df_tax["Plus/Minus Lorda"] > 0, df_tax["Plus/Minus Lorda"] * tax_rate, 0.0)
    df_tax["Plus/Minus Netta"] = df_tax["Plus/Minus Lorda"] - df_tax["Imposta Teorica"]
    df_tax["Valore Netto Post Imposta"] = df_tax["Valore di Mercato"] - df_tax["Imposta Teorica"]
    df_tax["Rendimento Netto %"] = np.where(df_tax["Importo Investito"] > 0, (df_tax["Plus/Minus Netta"] / df_tax["Importo Investito"]) * 100.0, 0.0)
    cols = ["Ticker", "Importo Investito", "Valore di Mercato", "P&L", "Aliquota Fiscale %", "Imposta Teorica", "Plus/Minus Netta", "Valore Netto Post Imposta", "Rendimento Netto %"]
    return df_tax[[c for c in cols if c in df_tax.columns]]

def calculate_tax_with_loss_offset(df_weights_base: pd.DataFrame, tax_rate: float = DEFAULT_TAX_RATE) -> Tuple[pd.DataFrame, Dict[str, float]]:
    if df_weights_base is None or df_weights_base.empty:
        return pd.DataFrame(), {}
    df = df_weights_base.copy()
    pl_lorda = df["P&L Base"].astype(float)
    gains = pl_lorda.clip(lower=0).sum()
    losses = (-pl_lorda.clip(upper=0)).sum()
    compensable = min(gains, losses)
    taxable_base = max(0.0, gains - compensable)
    theoretical_tax = taxable_base * tax_rate
    df["Aliquota Fiscale %"] = tax_rate * 100.0
    df["Plus/Minus Lorda Base"] = pl_lorda
    df["Imposta Teorica Base (compensata)"] = np.where(pl_lorda > 0, np.where(gains > 0, pl_lorda / gains * theoretical_tax, 0.0), 0.0)
    summary = {
        "Plusvalenze totali": float(gains),
        "Minusvalenze totali": float(losses),
        "Minusvalenze compensate": float(compensable),
        "Imponibile residuo": float(taxable_base),
        "Imposta teorica netta": float(theoretical_tax),
        "Risparmio fiscale da compensazione": float(compensable * tax_rate),
    }
    return df, summary

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
    series_list = []
    for t in tickers:
        r = get_daily_returns_for_ticker(t)
        if r is not None:
            series_list.append(r)
    if not series_list:
        return None
    df_rets = pd.concat(series_list, axis=1, join="outer").sort_index()
    df_rets = df_rets.dropna(how="all").dropna()
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
        return {"AnnRet": np.nan, "AnnVol": np.nan, "Sharpe": np.nan, "MaxDD": np.nan, "Sortino": np.nan, "Calmar": np.nan}
    rf = get_active_risk_free_rate()
    mu = port_ret.mean() * TRADING_DAYS_YEAR
    sigma = port_ret.std() * np.sqrt(TRADING_DAYS_YEAR)
    excess = mu - rf
    sharpe = excess / sigma if sigma > 0 else np.nan
    equity = (1 + port_ret).cumprod()
    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1.0
    max_dd = drawdown.min() if not drawdown.empty else np.nan
    rf_daily = rf / TRADING_DAYS_YEAR
    downside = port_ret[port_ret < rf_daily]
    downside_dev = downside.std() * np.sqrt(TRADING_DAYS_YEAR) if not downside.empty else np.nan
    sortino = excess / downside_dev if downside_dev and downside_dev > 0 else np.nan
    n_years = len(port_ret) / TRADING_DAYS_YEAR
    cagr = (equity.iloc[-1]) ** (1.0 / n_years) - 1.0 if n_years > 0 else np.nan
    calmar = cagr / abs(max_dd) if max_dd < 0 and not np.isnan(cagr) else np.nan
    return {"AnnRet": float(mu), "AnnVol": float(sigma), "Sharpe": float(sharpe) if not np.isnan(sharpe) else np.nan,
            "MaxDD": float(max_dd) if not np.isnan(max_dd) else np.nan,
            "Sortino": float(sortino) if not np.isnan(sortino) else np.nan,
            "Calmar": float(calmar) if not np.isnan(calmar) else np.nan,
            "CAGR": float(cagr) if not np.isnan(cagr) else np.nan}

def calculate_concentration_metrics(weights_pct: Dict[str, float]) -> Dict[str, float]:
    w = np.array(list(weights_pct.values()), dtype=float) / 100.0
    if w.sum() <= 0:
        return {"HHI": np.nan, "ENS": np.nan, "Top1 %": np.nan, "Top3 %": np.nan}
    w = w / w.sum()
    hhi = float((w ** 2).sum())
    ens = 1.0 / hhi if hhi > 0 else np.nan
    sorted_w = np.sort(w)[::-1]
    top1 = float(sorted_w[0] * 100.0)
    top3 = float(sorted_w[:3].sum() * 100.0)
    return {"HHI": hhi, "ENS": float(ens), "Top1 %": top1, "Top3 %": top3}

def calculate_portfolio_beta(port_ret: pd.Series, benchmark_symbol: str = DEFAULT_BENCHMARK) -> Dict[str, float]:
    try:
        bench_df = get_technical_data(benchmark_symbol)
        if bench_df is None or bench_df.empty:
            return {"Beta": np.nan, "Alpha (ann.)": np.nan, "Corr": np.nan}
        bench_ret = bench_df['Close'].pct_change().dropna()
        joined = pd.concat([port_ret, bench_ret], axis=1, join='inner').dropna()
        if joined.empty or len(joined) < 30:
            return {"Beta": np.nan, "Alpha (ann.)": np.nan, "Corr": np.nan}
        joined.columns = ['port', 'bench']
        cov = joined['port'].cov(joined['bench'])
        var_b = joined['bench'].var()
        beta = cov / var_b if var_b > 0 else np.nan
        rf_daily = get_active_risk_free_rate() / TRADING_DAYS_YEAR
        alpha_daily = (joined['port'].mean() - rf_daily) - beta * (joined['bench'].mean() - rf_daily)
        alpha_ann = alpha_daily * TRADING_DAYS_YEAR
        corr = joined['port'].corr(joined['bench'])
        return {"Beta": float(beta) if not np.isnan(beta) else np.nan,
                "Alpha (ann.)": float(alpha_ann) if not np.isnan(alpha_ann) else np.nan,
                "Corr": float(corr) if not np.isnan(corr) else np.nan}
    except Exception as e:
        logger.debug(f"Beta calculation failed: {e}")
        return {"Beta": np.nan, "Alpha (ann.)": np.nan, "Corr": np.nan}

def get_latest_price(symbol: str) -> Optional[float]:
    df = get_technical_data(symbol)
    if df is not None and not df.empty and 'Close' in df.columns:
        val = df['Close'].dropna().iloc[-1]
        return safe_float(val, None)
    raw = get_fundamental_data(symbol)
    if raw and raw.get('info'):
        info = raw['info']
        price = info.get('currentPrice') or info.get('regularMarketPrice')
        if price is not None:
            return safe_float(price, None)
    return None

def get_ticker_native_currency(symbol: str) -> Optional[str]:
    raw = get_fundamental_data(symbol)
    if raw and raw.get('info'):
        cur = raw['info'].get('currency')
        if cur:
            return str(cur).upper().strip()
    return None

def calculate_position_from_quantity(ticker: str, quantity: float, pmc: float, user_currency: Optional[str] = None, base_currency: Optional[str] = None) -> Dict[str, float]:
    current_price_native = safe_float(get_latest_price(ticker), np.nan)
    native_cur = get_ticker_native_currency(ticker) or "EUR"
    user_cur = (user_currency or native_cur).upper().strip()
    invested = float(quantity * pmc)
    if np.isnan(current_price_native):
        market_value = 0.0
        current_price_in_user_cur = np.nan
    else:
        fx_native_to_user = get_fx_rate(native_cur, user_cur) if native_cur != user_cur else 1.0
        current_price_in_user_cur = current_price_native * fx_native_to_user
        market_value = float(quantity) * current_price_in_user_cur
    pnl_value = market_value - invested
    pnl_pct = (pnl_value / invested) * 100.0 if invested > 0 else 0.0
    return {'Prezzo Attuale': current_price_in_user_cur if not np.isnan(current_price_native) else np.nan,
            'Prezzo Attuale Nativa': float(current_price_native) if not np.isnan(current_price_native) else np.nan,
            'Importo Investito': invested, 'Valore di Mercato': market_value, 'P&L': pnl_value, 'P&L %': pnl_pct,
            'Valuta Nativa': native_cur, 'FX Native->User': get_fx_rate(native_cur, user_cur) if native_cur != user_cur else 1.0}

def build_portfolio_allocation_df(positive_holdings: Dict[str, float], holdings_currency: Dict[str, str]) -> pd.DataFrame:
    rows = []
    for ticker, amount in positive_holdings.items():
        raw = get_fundamental_data(ticker)
        info = raw["info"] if raw and "info" in raw else {}
        company_name = info.get("longName", info.get("shortName", ticker))
        detected_currency = holdings_currency.get(ticker, info.get("currency", "USD"))
        rows.append({"Ticker": ticker, "Company Name": company_name, "Importo": float(amount), "Valuta": detected_currency})
    df_alloc = pd.DataFrame(rows)
    if df_alloc.empty:
        return df_alloc
    total = df_alloc["Importo"].sum()
    df_alloc["Peso %"] = np.where(total > 0, df_alloc["Importo"] / total * 100.0, 0.0)
    return df_alloc.sort_values("Peso %", ascending=False).reset_index(drop=True)

def summarize_group_weights(df_alloc: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if df_alloc.empty or group_col not in df_alloc.columns:
        return pd.DataFrame()
    out = df_alloc.groupby(group_col, dropna=False)["Importo"].sum().reset_index().sort_values("Importo", ascending=False)
    total = out["Importo"].sum()
    out["Peso %"] = np.where(total > 0, out["Importo"] / total * 100.0, 0.0)
    return out.reset_index(drop=True)

def compute_rebalancing_actions(df_alloc: pd.DataFrame, target_weights: Dict[str, float], group_col: str = "Ticker", tolerance_pct: float = 1.0) -> pd.DataFrame:
    if df_alloc.empty:
        return pd.DataFrame()
    current = summarize_group_weights(df_alloc, group_col)
    if current.empty:
        return pd.DataFrame()
    target_df = pd.DataFrame({group_col: list(target_weights.keys()), "Target %": list(target_weights.values())})
    merged = current.merge(target_df, on=group_col, how="outer").fillna(0.0)
    total_value = df_alloc["Importo"].sum()
    merged["Scostamento %"] = merged["Target %"] - merged["Peso %"]
    merged["Azione €"] = total_value * merged["Scostamento %"] / 100.0
    threshold = tolerance_pct / 100.0 * total_value
    merged["Azione"] = np.where(merged["Azione €"] > threshold, "Compra", np.where(merged["Azione €"] < -threshold, "Riduci", "In target"))
    return merged.sort_values("Azione €", ascending=False).reset_index(drop=True)

def _portfolio_export_csv(df_weights: pd.DataFrame) -> bytes:
    return df_weights.to_csv(index=False).encode("utf-8")

# ==========================================================================
# 11. AUTH E SESSION STATE
# ==========================================================================
def get_supabase_credentials() -> Tuple[Optional[str], Optional[str]]:
    url = safe_get_secret('SUPABASE_URL')
    key = safe_get_secret('SUPABASE_ANON_KEY')
    return url, key

@st.cache_resource(show_spinner=False)
def get_supabase_client() -> Optional[Client]:
    url, key = get_supabase_credentials()
    if not url or not key:
        logger.warning("Supabase non configurato.")
        return None
    try:
        return create_client(url, key)
    except Exception as e:
        logger.error(f"Errore inizializzazione Supabase: {e}")
        return None

def init_auth_state() -> None:
    if 'auth_user' not in st.session_state:
        st.session_state.auth_user = None
    if 'auth_session' not in st.session_state:
        st.session_state.auth_session = None

def get_logged_user_email() -> Optional[str]:
    user = st.session_state.get('auth_user')
    if not user:
        return None
    if isinstance(user, dict):
        return user.get('email')
    return getattr(user, 'email', None)

def is_authenticated() -> bool:
    return get_logged_user_email() is not None

def get_logged_user_id() -> Optional[str]:
    user = st.session_state.get('auth_user')
    if not user:
        return None
    if isinstance(user, dict):
        return user.get('id')
    return getattr(user, 'id', None)

def is_supabase_available() -> bool:
    return get_supabase_client() is not None

def load_user_portfolio() -> None:
    user_id = get_logged_user_id()
    if not user_id:
        return
    supabase = get_supabase_client()
    if supabase is None:
        st.warning("Supabase non configurato.")
        return
    try:
        res = supabase.table('portfoliopositions').select('*').eq('userid', user_id).execute()
        rows = getattr(res, 'data', None) or []
    except Exception as e:
        logger.warning(f'Load portfolio skipped: {e}')
        return
    st.session_state.portfolio_tickers = []
    st.session_state.holdings = {}
    st.session_state.holdings_quantity = {}
    st.session_state.holdings_pmc = {}
    st.session_state.holdings_currency = {}
    for r in rows:
        t = str(r.get('ticker', '')).upper().strip()
        if not t:
            continue
        try:
            qty = float(r.get('quantity', 0) or 0)
            pmc = float(r.get('pmc', 0) or 0)
        except (TypeError, ValueError):
            continue
        cur = str(r.get('currency', 'USD')).upper().strip() or 'USD'
        if t not in st.session_state.portfolio_tickers:
            st.session_state.portfolio_tickers.append(t)
        st.session_state.holdings_quantity[t] = qty
        st.session_state.holdings_pmc[t] = pmc
        st.session_state.holdings_currency[t] = cur
        st.session_state.holdings[t] = qty * pmc

def save_user_portfolio_position(ticker: str, quantity: float, pmc: float, currency: str) -> None:
    user_id = get_logged_user_id()
    if not user_id:
        return
    supabase = get_supabase_client()
    if supabase is None:
        st.error("Supabase non configurato.")
        return
    try:
        supabase.table('portfoliopositions').upsert({
            'userid': user_id,
            'ticker': str(ticker).upper().strip(),
            'quantity': float(quantity),
            'pmc': float(pmc),
            'currency': str(currency).upper().strip() or 'USD',
        }, on_conflict='userid,ticker').execute()
    except Exception as e:
        logger.exception("Errore salvataggio portafoglio")
        st.error(f"Errore Supabase durante il salvataggio di {ticker}: {e}")

def delete_user_portfolio_position(ticker: str) -> None:
    user_id = get_logged_user_id()
    if not user_id:
        return
    supabase = get_supabase_client()
    if supabase is None:
        return
    try:
        supabase.table('portfoliopositions').delete().eq('userid', user_id).eq('ticker', str(ticker).upper().strip()).execute()
    except Exception as e:
        logger.warning(f'Delete portfolio skipped for {ticker}: {e}')

def sign_in_with_supabase(email: str, password: str) -> Tuple[bool, str]:
    supabase = get_supabase_client()
    if supabase is None:
        return False, "Supabase non configurato."
    try:
        response = supabase.auth.sign_in_with_password({'email': email.strip(), 'password': password})
        user = getattr(response, 'user', None)
        session = getattr(response, 'session', None)
        if user is None:
            return False, 'Login non riuscito.'
        st.session_state.auth_user = user
        st.session_state.auth_session = session
        return True, 'Login eseguito.'
    except Exception as e:
        return False, f'Login fallito: {e}'

def sign_out_from_supabase() -> Tuple[bool, str]:
    supabase = get_supabase_client()
    try:
        if supabase is not None:
            try:
                supabase.auth.sign_out()
            except Exception:
                pass
        st.session_state.auth_user = None
        st.session_state.auth_session = None
        st.session_state.portfolio_loaded_from_db = False
        return True, 'Logout eseguito.'
    except Exception as e:
        return False, f'Logout fallito: {e}'

def render_auth_sidebar() -> None:
    st.sidebar.markdown('### 👤 Account')
    if not is_supabase_available():
        st.sidebar.info("Auth disabilitata: configura SUPABASE_URL e SUPABASE_ANON_KEY.")
        st.sidebar.markdown('---')
        return
    current_email = get_logged_user_email()
    if current_email:
        st.sidebar.success(f'Connesso come: {current_email}')
        if st.sidebar.button('🚪 Logout', width='stretch'):
            ok, msg = sign_out_from_supabase()
            if ok:
                st.sidebar.success(msg)
                st.rerun()
            else:
                st.sidebar.error(msg)
        st.sidebar.markdown('---')
        return
    auth_mode = st.sidebar.selectbox('Accesso', ['Login', 'Iscrizione'], key='auth_mode_select')
    email = st.sidebar.text_input('Email', key='auth_email')
    password = st.sidebar.text_input('Password', type='password', key='auth_password')
    if auth_mode == 'Iscrizione':
        password_confirm = st.sidebar.text_input('Conferma password', type='password', key='auth_password_confirm')
        if st.sidebar.button('📝 Crea account', width='stretch'):
            if not email:
                st.sidebar.error('Email non valida.')
            elif password != password_confirm:
                st.sidebar.error('Le password non coincidono.')
            else:
                supabase = get_supabase_client()
                if supabase:
                    try:
                        response = supabase.auth.sign_up({'email': email.strip(), 'password': password})
                        user = getattr(response, 'user', None)
                        if user:
                            st.sidebar.success('Registrazione completata. Controlla la tua email.')
                        else:
                            st.sidebar.error('Registrazione fallita.')
                    except Exception as e:
                        st.sidebar.error(f'Errore: {e}')
    else:
        if st.sidebar.button('🔐 Login', width='stretch'):
            if not email or not password:
                st.sidebar.error('Inserisci email e password.')
            else:
                ok, msg = sign_in_with_supabase(email, password)
                if ok:
                    st.sidebar.success(msg)
                    st.rerun()
                else:
                    st.sidebar.error(msg)
    st.sidebar.caption('Auth gestita con Supabase email/password.')
    st.sidebar.markdown('---')

# ==========================================================================
# 12. UI GRAFICI E TAB (menu laterale)
# ==========================================================================
def render_fondamentali_tab(row, batch_results, ticker):
    if row is None:
        st.info("Nessun ticker attivo. Usa il box 'Analisi rapida senza ricerca'.")
        if batch_results is not None and not batch_results.empty:
            st.dataframe(batch_results.drop(columns=['_raw_data'], errors='ignore'))
    else:
        st.info("💡 **Come leggere questa sezione:** qualita' del business prima, sostenibilita' finanziaria poi, prezzo/multipli alla fine.")
        if batch_results is not None and not batch_results.empty:
            st.dataframe(batch_results.drop(columns=['_raw_data'], errors='ignore'))
        else:
            st.success(f'Analisi standalone attiva su: {ticker}')
            st.dataframe(pd.DataFrame([dict(row)]).drop(columns=['_raw_data'], errors='ignore'))
    st.markdown("---")
    st.markdown(FOOTER_HTML, unsafe_allow_html=True)

def render_tecnico_tab(row, ticker):
    if row is None:
        st.info("Nessun ticker attivo.")
        st.markdown(FOOTER_HTML, unsafe_allow_html=True)
        return
    st.info("💡 **Come leggere il grafico:** SMA200 = trend di fondo, RSI = momentum, ADX = forza del trend.")
    df_tech = get_technical_data(ticker)
    if df_tech is not None:
        df_calc = calculate_technical_indicators(df_tech)
        score, reasons = calculate_timing_score(df_calc, df_calc['Close'].iloc[-1])
        st.metric("Timing Score", f"{score}/100")
        with st.expander("Dettaglio segnali"):
            for r in reasons:
                st.write(r)
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25])
        fig.add_trace(go.Candlestick(x=df_calc.index, open=df_calc['Open'], high=df_calc['High'], low=df_calc['Low'], close=df_calc['Close'], name="Prezzo"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_calc.index, y=df_calc['SMA_200'], name="SMA 200", line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_calc.index, y=df_calc['RSI'], name="RSI", line=dict(color='purple')), row=2, col=1)
        if 'ADX' in df_calc.columns:
            fig.add_trace(go.Scatter(x=df_calc.index, y=df_calc['ADX'], name="ADX", line=dict(color='orange')), row=3, col=1)
        fig.update_layout(height=700, xaxis_rangeslider_visible=False, template="plotly_dark")
        st.plotly_chart(fig, width='stretch')
    else:
        st.warning(f'Dati tecnici non disponibili per {ticker}.')
    st.markdown("---")
    st.markdown(FOOTER_HTML, unsafe_allow_html=True)

def render_quant_tab(row, ticker, standalone_raw_data):
    if row is None:
        st.info("Nessun ticker attivo.")
        st.markdown(FOOTER_HTML, unsafe_allow_html=True)
        return
    st.info("💡 **Quant:** Sharpe, Sortino, Omega, Jensen, Ulcer, GARCH, Monte Carlo, etc.")
    df_tech = get_technical_data(ticker)
    if df_tech is not None:
        rf_eff = get_active_risk_free_rate()
        qm = calculate_quant_metrics(df_tech, row.get('_raw_data', standalone_raw_data) if row is not None else standalone_raw_data)
        risk = calculate_risk_metrics(df_tech)
        macro = get_macro_indicators()
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Sharpe ({rf_eff*100:.1f}% Rf)", f"{qm['Sharpe Ratio']:.2f}")
        c2.metric("Trend R-Squared", f"{qm['R-Squared']:.2f}")
        z = qm['Altman Z-Score']
        c3.metric("Altman Z-Score", f"{z:.2f}" if isinstance(z, (int, float)) else "N/A")
        c4, c5, c6 = st.columns(3)
        c4.metric("Max Drawdown", f"{risk['Max Drawdown']*100:.1f}%")
        c5.metric("CAGR", f"{risk['CAGR']*100:.1f}%" if not np.isnan(risk['CAGR']) else "N/A")
        c6.metric("Omega Ratio", f"{risk['Omega Ratio']:.2f}" if not np.isnan(risk['Omega Ratio']) else "N/A")
        c7, c8, c9 = st.columns(3)
        c7.metric("Sortino", f"{risk['Sortino']:.2f}" if not np.isnan(risk['Sortino']) else "N/A")
        c8.metric("Ulcer Index", f"{risk['Ulcer Index']:.3f}" if not np.isnan(risk['Ulcer Index']) else "N/A")
        c9.metric("GARCH Vol (next)", f"{risk['GARCH Vol (next)']*100:.2f}%" if not np.isnan(risk['GARCH Vol (next)']) else "N/A")
        if not np.isnan(macro['treasury_10y']):
            st.caption(f"🏦 Treasury 10Y: {macro['treasury_10y']*100:.2f}% | CPI YoY: {macro['cpi_yoy']:.2f}%")
        df_calc_q = calculate_technical_indicators(df_tech)
        score_q, _ = calculate_timing_score(df_calc_q, df_calc_q['Close'].iloc[-1])
        smart = compute_smart_quant_score(row, score_q, qm, risk)
        st.metric("Smart Quant Score", f"{smart['SmartScore']:.1f}/100")
        st.caption(f"Pesi attivi: F={SMART_WEIGHTS['F']:.2f} | V={SMART_WEIGHTS['V']:.2f} | T={SMART_WEIGHTS['T']:.2f} | Q={SMART_WEIGHTS['Q']:.2f}")
        with st.expander("📉 Distribuzione rendimenti & rischio"):
            st.write(f"Skewness: {risk['Skew']:.2f} | Kurtosis: {risk['Kurt']:.2f}")
            returns = df_tech['Close'].pct_change().dropna()
            fig_r = go.Figure()
            fig_r.add_trace(go.Histogram(x=returns, nbinsx=50, name="Rendimenti giornalieri"))
            fig_r.update_layout(template="plotly_dark", bargap=0.05)
            st.plotly_chart(fig_r, width='stretch')
        with st.expander("🎲 Simulazione Monte Carlo"):
            col_mc1, col_mc2 = st.columns(2)
            horizon_days = col_mc1.slider("Orizzonte (giorni)", 60, 756, 252, step=21)
            n_paths = col_mc2.slider("Numero traiettorie", 100, 3000, 1000, step=100)
            mc = monte_carlo_equity(df_tech, n_paths, horizon_days)
            if mc["paths"] is not None:
                final_vals = mc["final_distribution"]
                p05 = np.quantile(final_vals, 0.05)
                p50 = np.quantile(final_vals, 0.50)
                c10, c11, c12 = st.columns(3)
                c10.metric("Prob. perdita > 20%", f"{(final_vals < 0.8).mean()*100:.1f}%")
                c11.metric("Mediana esito", f"{(p50-1)*100:.1f}%")
                c12.metric("5° percentile", f"{(p05-1)*100:.1f}%")
                x = np.arange(1, horizon_days + 1)
                fig_mc = go.Figure()
                fig_mc.add_trace(go.Scatter(x=x, y=mc["q50"], name="Mediana", line=dict(color="cyan")))
                fig_mc.add_trace(go.Scatter(x=x, y=mc["q95"], name="95° percentile", line=dict(color="green"), opacity=0.3))
                fig_mc.add_trace(go.Scatter(x=x, y=mc["q05"], name="5° percentile", line=dict(color="red"), opacity=0.3, fill="tonexty", fillcolor="rgba(255,0,0,0.1)"))
                fig_mc.update_layout(template="plotly_dark", height=400, xaxis_title="Giorni", yaxis_title="Equity")
                st.plotly_chart(fig_mc, width='stretch')
    st.markdown("---")
    st.markdown(FOOTER_HTML, unsafe_allow_html=True)

def render_verdetto_tab(row, ticker, standalone_raw_data):
    if row is None:
        st.info("Nessun ticker attivo.")
        st.markdown(FOOTER_HTML, unsafe_allow_html=True)
        return
    st.info("💡 **Modello Unificato:** qualità, valutazione, timing e rischio in un unico punteggio (0‑100).")
    df_tech = get_technical_data(ticker)
    macro = get_macro_indicators()
    qm = calculate_quant_metrics(df_tech, row.get('_raw_data', standalone_raw_data) if row is not None else standalone_raw_data) if df_tech is not None else {}
    risk = calculate_risk_metrics(df_tech) if df_tech is not None else {"Max Drawdown": 0.0, "CAGR": 0.0}
    if df_tech is not None:
        df_calc_v = calculate_technical_indicators(df_tech)
        timing_score, timing_reasons = calculate_timing_score(df_calc_v, df_calc_v['Close'].iloc[-1])
    else:
        timing_score = 0
        timing_reasons = []
    verdict = compute_unified_verdict(row=row, timing_score=timing_score, qm=qm, risk=risk, macro=macro)
    st.markdown(f"## {verdict['Emoji']} {verdict['Verdict']}  ({verdict['FinalScore']:.1f}/100)")
    st.progress(int(verdict['FinalScore']))
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Qualità", f"{verdict['FQS']:.0f}/100")
    col2.metric("Valutazione", f"{verdict['VAS']:.0f}/100")
    col3.metric("Timing", f"{verdict['TMS']:.0f}/100")
    col4.metric("Rischio", f"{verdict['QRS']:.0f}/100")
    with st.expander("🔍 Criteri analizzati"):
        for d in verdict["Details"]:
            st.write(d)
    if timing_reasons:
        with st.expander("📈 Dettaglio segnali tecnici"):
            for r in timing_reasons:
                st.write(r)
    # ML prediction (se disponibile)
    if SKLEARN_AVAILABLE and df_tech is not None and len(df_tech) > 100:
        features = prepare_ml_features(df_tech)
        if not features.empty:
            model, scaler, score = train_rf_predictor(features)
            if model is not None:
                latest = features.drop('target', axis=1).iloc[-1:].copy()
                pred_ret = predict_forward_return(model, scaler, latest)
                if pred_ret is not None:
                    st.metric("🤖 ML Predizione (21gg)", f"{pred_ret*100:.2f}%", help="Rendimento atteso previsto dal modello Random Forest")
    st.markdown('---')
    st.subheader('Spiegazione AI (VqAi)')
    ai_context = build_ai_context_for_ticker(ticker, row, qm, risk, timing_score, timing_reasons, mode='Unificato')
    st.session_state.vq_ai_live_context = st.session_state.get('vq_ai_live_context', {})
    st.session_state.vq_ai_live_context[ticker] = ai_context
    if st.session_state.get('ai_ticker_chat_last_symbol') != ticker:
        st.session_state.ai_ticker_chat_history = []
        st.session_state.ai_ticker_chat_last_symbol = ticker
    if st.button('🧠 Spiega con VqAi', key=f'ai_explain_{ticker}'):
        with st.spinner('Analisi AI in corso...'):
            ai_answer = ask_gemini_ticker_chat(ai_context, 'Spiegami questo titolo come un analista buy-side prudente, coerente con il verdetto mostrato.', mode='Unificato')
        st.session_state.ai_ticker_chat_history.append({'role': 'assistant', 'content': ai_answer})
    if st.session_state.get('ai_ticker_chat_history'):
        last_ai_msg = st.session_state.ai_ticker_chat_history[-1]
        if last_ai_msg.get('role') == 'assistant':
            st.markdown(last_ai_msg.get('content', ''))
    st.subheader('Chat AI sul ticker')
    for msg in st.session_state.get('ai_ticker_chat_history', []):
        with st.chat_message(msg.get('role', 'assistant')):
            st.markdown(msg.get('content', ''))
    ai_user_prompt = st.chat_input('Fai una domanda su questo ticker', key=f'ai_chat_input_{ticker}')
    if ai_user_prompt:
        st.session_state.ai_ticker_chat_history.append({'role': 'user', 'content': ai_user_prompt})
        conv_history = "CRONOLOGIA DELLA CONVERSAZIONE:\n"
        for m in st.session_state.ai_ticker_chat_history[:-1]:
            conv_history += f"[{m['role'].upper()}]: {m['content']}\n"
        enriched_prompt = f"{conv_history}\nDOMANDA ATTUALE: {ai_user_prompt}"
        with st.chat_message('user'):
            st.markdown(ai_user_prompt)
        with st.chat_message('assistant'):
            with st.spinner("L'AI sta rispondendo..."):
                ai_reply = ask_gemini_ticker_chat(ai_context, enriched_prompt, mode='Unificato')
            st.markdown(ai_reply)
        st.session_state.ai_ticker_chat_history.append({'role': 'assistant', 'content': ai_reply})
    st.markdown("---")
    st.markdown(FOOTER_HTML, unsafe_allow_html=True)

def render_portafoglio_tab(ui):
    st.info("💡 **Cabina di controllo del portafoglio reale.** Posizioni con quantita', PMC, valuta. Calcoli FX-aware, fiscalita' con compensazione minusvalenze, concentrazione, ribilanciamento.")
    base_currency = st.selectbox("Valuta base portafoglio", ["EUR", "USD", "GBP", "CHF"], index=0, key="base_currency_portfolio")
    st.success(f"Valuta base portafoglio: **{base_currency}** | Conversione FX reale attiva.")
    all_tickers_batch: List[str] = []
    if st.session_state.batch_results is not None and not st.session_state.batch_results.empty:
        all_tickers_batch = st.session_state.batch_results["Ticker"].tolist()
    if all_tickers_batch:
        st.markdown("#### Seleziona dal batch analizzato")
        default_batch = [t for t in st.session_state.portfolio_tickers if t in all_tickers_batch]
        selected_from_batch = st.multiselect("Titoli da includere nel portafoglio (da batch)", all_tickers_batch, default=default_batch)
    else:
        selected_from_batch = []
    st.markdown("#### Aggiungi manualmente altri ticker")
    manual_ticker = st.text_input("Ticker (incluso suffisso mercato, es. STLAM.MI, BMW.DE, AIR.PA, ULVR.L)", "")
    if st.button("➕ Aggiungi ticker manuale al portafoglio"):
        if manual_ticker.strip():
            try:
                t_clean = sanitize_ticker(manual_ticker)
                if t_clean not in st.session_state.portfolio_tickers:
                    st.session_state.portfolio_tickers.append(t_clean)
                    st.session_state.holdings_quantity.setdefault(t_clean, 0.0)
                    st.session_state.holdings_pmc.setdefault(t_clean, 0.0)
                    st.session_state.holdings_currency.setdefault(t_clean, 'USD')
                    if is_authenticated():
                        save_user_portfolio_position(t_clean, st.session_state.holdings_quantity[t_clean], st.session_state.holdings_pmc[t_clean], st.session_state.holdings_currency[t_clean])
                    st.success(f"Aggiunto {t_clean} al portafoglio.")
            except ValueError as e:
                st.error(f"Ticker non valido: {e}")
        else:
            st.warning("Inserisci un ticker valido prima di aggiungere.")
    portfolio_list = sorted(set(selected_from_batch + st.session_state.portfolio_tickers))
    st.session_state.portfolio_tickers = portfolio_list
    if portfolio_list:
        st.markdown("#### Dati posizione per ogni ticker")
        cols = st.columns(2)
        holdings_quantity = st.session_state.holdings_quantity
        holdings_pmc = st.session_state.holdings_pmc
        for i, t in enumerate(portfolio_list):
            col = cols[i % 2]
            col.markdown(f"##### {t}")
            default_qty = float(holdings_quantity.get(t, 0.0))
            default_pmc = float(holdings_pmc.get(t, 0.0))
            qty = col.number_input(f"{t} - Quantità / Quote", min_value=0.0, value=default_qty, step=0.01, format="%.4f", key=f"holding_qty_{t}")
            pmc = col.number_input(f"{t} - PMC", min_value=0.0, value=default_pmc, step=0.01, format="%.4f", key=f"holding_pmc_{t}")
            cur_default = st.session_state.holdings_currency.get(t, "USD")
            cur_options = ["USD", "EUR", "GBP", "CHF", "JPY", "CAD", "AUD"]
            if cur_default not in cur_options:
                cur_options = [cur_default] + cur_options
            cur = col.selectbox(f"{t} - Valuta posizione", cur_options, index=cur_options.index(cur_default) if cur_default in cur_options else 0, key=f"currency_{t}")
            st.session_state.holdings_currency[t] = cur
            derived = calculate_position_from_quantity(t, qty, pmc, user_currency=cur) if qty > 0 and pmc > 0 else {'Importo Investito': 0.0, 'Prezzo Attuale': np.nan, 'Valore di Mercato': 0.0, 'P&L': 0.0, 'P&L %': 0.0, 'Valuta Nativa': cur, 'FX Native->User': 1.0}
            holdings_quantity[t] = qty
            holdings_pmc[t] = pmc
            st.session_state.holdings[t] = float(derived['Importo Investito'])
            if is_authenticated():
                save_user_portfolio_position(t, qty, pmc, cur)
            price_text = "N/D" if pd.isna(derived['Prezzo Attuale']) else f"{derived['Prezzo Attuale']:.2f}"
            native_cur = derived.get('Valuta Nativa', cur)
            fx_used = derived.get('FX Native->User', 1.0)
            col.caption(f"Prezzo (in {cur}): {price_text} | Investito: {derived['Importo Investito']:.2f} | Valore: {derived['Valore di Mercato']:.2f} | P&L: {derived['P&L']:.2f} ({derived['P&L %']:.2f}%)")
            if native_cur and native_cur != cur:
                col.caption(f"⚠️ Valuta nativa: {native_cur}, FX applicato: {fx_used:.4f}")
            if col.button("🗑 Rimuovi", key=f"remove_{t}"):
                if t in st.session_state.portfolio_tickers:
                    st.session_state.portfolio_tickers = [x for x in st.session_state.portfolio_tickers if x != t]
                for d in [st.session_state.holdings, st.session_state.holdings_currency, st.session_state.holdings_quantity, st.session_state.holdings_pmc]:
                    d.pop(t, None)
                if is_authenticated():
                    delete_user_portfolio_position(t)
                st.rerun()
        st.session_state.holdings_quantity = holdings_quantity
        st.session_state.holdings_pmc = holdings_pmc
        if st.button("📊 Calcola pesi e analisi del portafoglio"):
            positive_holdings = {t: a for t, a in st.session_state.holdings.items() if a > 0}
            if not positive_holdings:
                st.error("Imposta quantita' e PMC > 0 almeno per un titolo.")
            else:
                tot = sum(positive_holdings.values())
                weights_pct = {t: a / tot * 100.0 for t, a in positive_holdings.items()}
                built = build_portfolio_returns(list(positive_holdings.keys()), weights_pct)
                if built is None:
                    st.error("Impossibile costruire la serie dei rendimenti (dati insufficienti).")
                else:
                    df_rets, port_ret = built
                    pm = calculate_portfolio_metrics(port_ret)
                    st.markdown("#### Dettaglio posizioni e pesi")
                    rows_pos = []
                    for t in weights_pct.keys():
                        qty = st.session_state.holdings_quantity.get(t, 0.0)
                        pmc_val = st.session_state.holdings_pmc.get(t, 0.0)
                        cur = st.session_state.holdings_currency.get(t, "USD")
                        derived = calculate_position_from_quantity(t, qty, pmc_val, user_currency=cur)
                        rows_pos.append({"Ticker": t, "Quantità": qty, "PMC": pmc_val, "Prezzo Attuale": derived['Prezzo Attuale'], "Importo Investito": derived['Importo Investito'], "Valore di Mercato": derived['Valore di Mercato'], "P&L": derived['P&L'], "P&L %": derived['P&L %'], "Peso %": weights_pct[t], "Valuta": cur})
                    df_weights = pd.DataFrame(rows_pos)
                    st.dataframe(df_weights, width='stretch')
                    csv_bytes = _portfolio_export_csv(df_weights)
                    st.download_button("💾 Scarica portafoglio (CSV)", data=csv_bytes, file_name="portfolio_vq.csv", mime="text/csv")
                    df_weights_base = enrich_portfolio_with_fx(df_weights, base_currency=base_currency)
                    st.markdown(f"#### Portafoglio in valuta base ({base_currency})")
                    st.dataframe(df_weights_base, width='stretch')
                    st.markdown("#### Analisi fiscale teorica")
                    tax_rate_input = st.slider("Aliquota fiscale (%)", 0.0, 50.0, value=float(DEFAULT_TAX_RATE * 100.0), step=1.0, key="portfolio_tax_rate_slider") / 100.0
                    df_tax = calculate_tax_impact(df_weights, tax_rate=tax_rate_input)
                    if not df_tax.empty:
                        st.dataframe(df_tax, width='stretch')
                        total_tax = float(df_tax["Imposta Teorica"].sum())
                        total_net_pnl = float(df_tax["Plus/Minus Netta"].sum())
                        tax_c1, tax_c2 = st.columns(2)
                        tax_c1.metric("Imposta teorica (no compensazione)", f"{total_tax:,.2f}")
                        tax_c2.metric("P&L netto", f"{total_net_pnl:,.2f}")
                    st.markdown("#### Compensazione fiscale (minusvalenze)")
                    df_tax_comp, summary_comp = calculate_tax_with_loss_offset(df_weights_base, tax_rate=tax_rate_input)
                    if summary_comp:
                        ccol1, ccol2, ccol3 = st.columns(3)
                        ccol1.metric("Plusvalenze tot.", f"{summary_comp['Plusvalenze totali']:,.2f}")
                        ccol2.metric("Minusvalenze tot.", f"{summary_comp['Minusvalenze totali']:,.2f}")
                        ccol3.metric("Risparmio fiscale", f"{summary_comp['Risparmio fiscale da compensazione']:,.2f}")
                        st.caption(f"In Italia le minusvalenze realizzate possono essere usate per ridurre l'imponibile sulle plusvalenze entro {TAX_LOSS_COMPENSATION_YEARS} anni (art. 68 TUIR). Imposta teorica netta: {summary_comp['Imposta teorica netta']:,.2f} {base_currency}.")
                    st.markdown("#### Allocazione del portafoglio")
                    df_alloc = build_portfolio_allocation_df(positive_holdings, st.session_state.holdings_currency)
                    if not df_alloc.empty:
                        st.dataframe(df_alloc, width='stretch')
                    st.markdown("#### Concentrazione e diversificazione")
                    conc = calculate_concentration_metrics(weights_pct)
                    ccc1, ccc2, ccc3, ccc4 = st.columns(4)
                    ccc1.metric("HHI", f"{conc['HHI']:.3f}")
                    ccc2.metric("Numero effettivo titoli", f"{conc['ENS']:.1f}")
                    ccc3.metric("Top 1 %", f"{conc['Top1 %']:.1f}%")
                    ccc4.metric("Top 3 %", f"{conc['Top3 %']:.1f}%")
                    st.caption("HHI < 0.10 → portafoglio molto diversificato; HHI > 0.25 → concentrazione elevata.")
                    if not df_weights_base.empty:
                        total_invested_base = float(df_weights_base["Importo Investito Base"].sum())
                        total_value_base = float(df_weights_base["Valore di Mercato Base"].sum())
                        total_pnl_base = float(df_weights_base["P&L Base"].sum())
                        total_pnl_pct_base = (total_pnl_base / total_invested_base * 100.0) if total_invested_base > 0 else 0.0
                        ctot1, ctot2, ctot3, ctot4 = st.columns(4)
                        ctot1.metric(f"Investito ({base_currency})", f"{total_invested_base:,.2f}")
                        ctot2.metric(f"Valore ({base_currency})", f"{total_value_base:,.2f}")
                        ctot3.metric(f"P&L ({base_currency})", f"{total_pnl_base:,.2f}")
                        ctot4.metric("Rend. totale", f"{total_pnl_pct_base:.2f}%")
                    cpa, cpv, cps, cpdd = st.columns(4)
                    cpa.metric("Rend. annuo atteso", f"{pm['AnnRet']*100:.2f}%")
                    cpv.metric("Volatilita' annua", f"{pm['AnnVol']*100:.2f}%")
                    cps.metric("Sharpe portafoglio", f"{pm['Sharpe']:.2f}")
                    cpdd.metric("Max Drawdown", f"{pm['MaxDD']*100:.1f}%")
                    cps2, cps3, cps4 = st.columns(3)
                    cps2.metric("Sortino", f"{pm['Sortino']:.2f}" if not np.isnan(pm['Sortino']) else "N/A")
                    cps3.metric("Calmar", f"{pm['Calmar']:.2f}" if not np.isnan(pm['Calmar']) else "N/A")
                    cps4.metric("CAGR portafoglio", f"{pm['CAGR']*100:.1f}%" if not np.isnan(pm['CAGR']) else "N/A")
                    st.markdown("#### Esposizione di mercato (vs S&P 500)")
                    beta_metrics = calculate_portfolio_beta(port_ret, benchmark_symbol=DEFAULT_BENCHMARK)
                    bcol1, bcol2, bcol3 = st.columns(3)
                    bcol1.metric("Beta", f"{beta_metrics['Beta']:.2f}" if not np.isnan(beta_metrics['Beta']) else "N/A")
                    bcol2.metric("Alpha (annuo)", f"{beta_metrics['Alpha (ann.)']*100:.2f}%" if not np.isnan(beta_metrics['Alpha (ann.)']) else "N/A")
                    bcol3.metric("Correlazione", f"{beta_metrics['Corr']:.2f}" if not np.isnan(beta_metrics['Corr']) else "N/A")
                    equity_p = (1 + port_ret).cumprod()
                    fig_p = go.Figure()
                    fig_p.add_trace(go.Scatter(x=equity_p.index, y=equity_p.values, name="Equity portafoglio"))
                    fig_p.update_layout(template="plotly_dark", height=400, xaxis_title="Data", yaxis_title="Equity normalizzata")
                    st.plotly_chart(fig_p, width='stretch')
                    corr = df_rets.corr()
                    st.markdown("#### Correlazioni tra titoli in portafoglio")
                    st.dataframe(corr.style.background_gradient(cmap="RdYlGn", axis=None))
    else:
        st.info("Seleziona almeno un titolo dal batch o aggiungilo manualmente.")
    st.markdown("---")
    st.markdown(FOOTER_HTML, unsafe_allow_html=True)

def inject_pwa_support():
    st.markdown("""
    <script>
    (function(){
      const base64Png = 'iVBORw0KGgoAAAANSUhEUgAAAMAAAADACAIAAADdvvtQAAACNklEQVR4nO3SwQ3AIBDAsNL9dz6WIEJC9gR5ZM18A6ft2wG8yQBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmAxgBjDICfiBZ0UZYAAAAASUVORK5CYII=';
      const manifest = {
        name: 'V-Quant Pro', short_name: 'V-Quant Pro', description: 'Analisi investimenti e portafoglio installabile su smartphone',
        start_url: '.', display: 'standalone', background_color: '#0e1117', theme_color: '#0e1117',
        icons: [{ src: 'data:image/png;base64,' + base64Png, sizes: '192x192', type: 'image/png' },
                { src: 'data:image/png;base64,' + base64Png, sizes: '512x512', type: 'image/png' }]
      };
      const manifestBlob = new Blob([JSON.stringify(manifest)], {type: 'application/manifest+json'});
      const manifestUrl = URL.createObjectURL(manifestBlob);
      const link = document.createElement('link'); link.rel = 'manifest'; link.href = manifestUrl; document.head.appendChild(link);
      const appleIcon = document.createElement('link'); appleIcon.rel = 'apple-touch-icon'; appleIcon.href = 'data:image/png;base64,' + base64Png; document.head.appendChild(appleIcon);
      const meta1 = document.createElement('meta'); meta1.name = 'apple-mobile-web-app-capable'; meta1.content = 'yes'; document.head.appendChild(meta1);
      const meta2 = document.createElement('meta'); meta2.name = 'apple-mobile-web-app-status-bar-style'; meta2.content = 'black-translucent'; document.head.appendChild(meta2);
      const meta3 = document.createElement('meta'); meta3.name = 'apple-mobile-web-app-title'; meta3.content = 'VqPro'; document.head.appendChild(meta3);
      const meta4 = document.createElement('meta'); meta4.name = 'theme-color'; meta4.content = '#0e1117'; document.head.appendChild(meta4);
    })();
    </script>
    """, unsafe_allow_html=True)

# ==========================================================================
# MAIN
# ==========================================================================
def main():
    init_auth_state()
    # Inizializza session state
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
    if 'holdings_quantity' not in st.session_state:
        st.session_state.holdings_quantity = {}
    if 'holdings_pmc' not in st.session_state:
        st.session_state.holdings_pmc = {}
    if 'portfolio_loaded_from_db' not in st.session_state:
        st.session_state.portfolio_loaded_from_db = False
    if 'analysis_errors' not in st.session_state:
        st.session_state.analysis_errors = []
    if 'standalone_ticker_input' not in st.session_state:
        st.session_state.standalone_ticker_input = ''
    if 'ai_ticker_chat_history' not in st.session_state:
        st.session_state.ai_ticker_chat_history = []
    if 'ai_ticker_chat_last_symbol' not in st.session_state:
        st.session_state.ai_ticker_chat_last_symbol = None
    if 'vq_ai_live_context' not in st.session_state:
        st.session_state.vq_ai_live_context = {}
    
    st.title("💲 V-Quant Pro")
    st.caption(f"Ultimo aggiornamento dati: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')} (cache 15 min)")
    inject_pwa_support()
    
    # Sidebar autenticazione
    render_auth_sidebar()
    
    # Sidebar per selezione asset e analisi
    with st.sidebar:
        st.markdown("## 📊 Analisi")
        input_mode = st.radio("Modalità", ["Manuale", "Batch CSV"], horizontal=True)
        file = None
        manual = ""
        if input_mode == "Batch CSV":
            file = st.file_uploader("Carica CSV (colonna 'Ticker' richiesta)", type=["csv"])
        else:
            manual = st.text_input("Ticker", value="").upper().strip()
        market = st.selectbox("Borsa", ["USA", "Italia (.MI)", "Germania (.DE)", "Francia (.PA)", "GB (.L)", "Spagna (.MC)", "Svizzera (.SW)", "Canada (.TO)", "Giappone (.T)", "Hong Kong (.HK)", "Australia (.AX)", "India (.NS)", "Crypto", "Custom"])
        suffix_lookup = {"Italia": ".MI", "Germania": ".DE", "Francia": ".PA", "GB": ".L", "Spagna": ".MC", "Svizzera": ".SW", "Canada": ".TO", "Giappone": ".T", "Hong Kong": ".HK", "Australia": ".AX", "India": ".NS"}
        suffix = ""
        for k, s in suffix_lookup.items():
            if k in market:
                suffix = s
                break
        analyze_btn = st.button("🚀 Avvia Analisi", width='stretch')
        
        if analyze_btn:
            targets: List[str] = []
            if input_mode == "Manuale" and manual:
                targets = [manual]
            elif input_mode == "Batch CSV" and file:
                try:
                    csv_df = pd.read_csv(file)
                    if 'Ticker' not in csv_df.columns:
                        st.error('Il CSV deve contenere una colonna Ticker.')
                    else:
                        targets = csv_df['Ticker'].dropna().astype(str).tolist()[:MAX_CSV_ROWS]
                except Exception as e:
                    st.error(f"Errore lettura CSV: {e}")
            if targets:
                normalized_targets = []
                analysis_errors = []
                for t in targets:
                    try:
                        # Risoluzione automatica ticker
                        resolved, _ = auto_resolve_ticker(t)
                        normalized_targets.append(resolved)
                    except Exception as e:
                        analysis_errors.append(f'{t}: {e}')
                with st.spinner(f"Analisi di {len(normalized_targets)} ticker in parallelo..."):
                    results, fetch_errors = fetch_metrics_batch(normalized_targets)
                analysis_errors.extend(fetch_errors)
                st.session_state.batch_results = pd.DataFrame(results)
                st.session_state.analysis_errors = analysis_errors
                if results:
                    st.session_state.selected_ticker = results[0]["Ticker"]
                else:
                    st.session_state.selected_ticker = None
    
    # Box analisi rapida
    with st.expander("🎯 Analisi rapida senza ricerca", expanded=(st.session_state.batch_results is None or st.session_state.batch_results.empty)):
        csel1, csel2, csel3 = st.columns([1.2, 1.2, 1])
        batch_options: List[str] = []
        if st.session_state.batch_results is not None and not st.session_state.batch_results.empty and 'Ticker' in st.session_state.batch_results.columns:
            batch_options = st.session_state.batch_results['Ticker'].dropna().astype(str).tolist()
        portfolio_options = sorted(st.session_state.get('portfolio_tickers', []) or [])
        if batch_options:
            sel_idx = ([''] + batch_options).index(st.session_state.selected_ticker) if st.session_state.selected_ticker in batch_options else 0
            selected_from_batch = csel1.selectbox('Ticker dai risultati caricati', [''] + batch_options, index=sel_idx, key='quick_batch_pick')
            if selected_from_batch:
                st.session_state.selected_ticker = selected_from_batch
                st.session_state.standalone_ticker_input = ''
        else:
            csel1.caption('Nessun batch attivo.')
        if portfolio_options:
            portfolio_pick = csel2.selectbox('Ticker dal portafoglio', [''] + portfolio_options, index=0, key='standalone_portfolio_pick')
            if portfolio_pick:
                st.session_state.selected_ticker = None
                st.session_state.standalone_ticker_input = portfolio_pick
        else:
            csel2.caption('Portafoglio vuoto.')
        manual_quick = csel3.text_input('Ticker libero', value=st.session_state.get('standalone_ticker_input', ''), key='quick_manual_ticker').upper().strip()
        if manual_quick:
            st.session_state.selected_ticker = None
            st.session_state.standalone_ticker_input = manual_quick
    
    # Risoluzione ticker attivo
    ticker = None
    row = None
    standalone_raw_data = None
    analysis_source = "none"
    
    if st.session_state.selected_ticker:
        ticker = st.session_state.selected_ticker
        if st.session_state.batch_results is not None and not st.session_state.batch_results.empty and 'Ticker' in st.session_state.batch_results.columns:
            match = st.session_state.batch_results[st.session_state.batch_results['Ticker'] == ticker]
            if not match.empty:
                row = match.iloc[0]
                standalone_raw_data = row.get('_raw_data') if hasattr(row, 'get') else None
                analysis_source = 'batch'
    elif st.session_state.standalone_ticker_input:
        ticker = st.session_state.standalone_ticker_input
        try:
            raw_data = get_fundamental_data(ticker)
            if raw_data:
                met = calculate_fundamental_metrics(raw_data)
                if met:
                    row = pd.Series(met.to_ui_dict())
                    standalone_raw_data = raw_data
            analysis_source = 'standalone'
        except Exception as e:
            logger.warning(f"Standalone analysis failed: {e}")
    
    # Menu laterale per navigazione tra tab
    if ticker is not None:
        st.sidebar.markdown("---")
        st.sidebar.markdown("## 📑 Navigazione")
        tab_selection = st.sidebar.radio("Vai a:", ["📊 Fondamentali", "📉 Tecnico", "⚛️ Quant", "⚖️ Verdetto", "📁 Portafoglio"])
    else:
        tab_selection = "📁 Portafoglio"  # default per portafoglio anche senza ticker
    
    # Mostra il tab selezionato
    if tab_selection == "📊 Fondamentali":
        render_fondamentali_tab(row, st.session_state.batch_results, ticker)
    elif tab_selection == "📉 Tecnico":
        render_tecnico_tab(row, ticker)
    elif tab_selection == "⚛️ Quant":
        render_quant_tab(row, ticker, standalone_raw_data)
    elif tab_selection == "⚖️ Verdetto":
        render_verdetto_tab(row, ticker, standalone_raw_data)
    elif tab_selection == "📁 Portafoglio":
        render_portafoglio_tab({})

if __name__ == "__main__":
    main()