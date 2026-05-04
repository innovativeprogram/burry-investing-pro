import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
try:
    import pandas_ta as ta
except Exception:
    ta = None
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import logging
import os
from urllib.parse import urlparse
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
from sklearn.linear_model import LinearRegression
from supabase import create_client, Client


# ==========================================
# 0. SETUP LOGGING & COSTANTI GLOBALI
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


DEFAULT_TAX_RATE = 0.26
SAFE_INTEREST_COVERAGE = 100.0
TRADING_DAYS_YEAR = 252
MAX_CSV_ROWS = 100
MAX_WORKERS = 3
RISK_FREE_RATE = 0.04
FX_TTL_SECONDS = 3600

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
                return float(s.iloc[-1])
        inverse = yf.Ticker(f"{t}{f}=X").history(period="5d", interval="1d")
        if inverse is not None and not inverse.empty and 'Close' in inverse.columns:
            s = inverse['Close'].dropna()
            if not s.empty and float(s.iloc[-1]) != 0:
                return float(1.0 / float(s.iloc[-1]))
    except Exception as e:
        logger.warning(f"FX fallback {from_currency}->{to_currency}: {e}")
    return 1.0

def enrich_portfolio_with_fx(dfweights: pd.DataFrame, base_currency: str = "EUR") -> pd.DataFrame:
    if dfweights is None or dfweights.empty:
        return pd.DataFrame()
    out = dfweights.copy()
    base = str(base_currency or "EUR").upper().strip()
    out["Valuta Base"] = base
    out["FX Rate"] = out["Valuta"].apply(lambda c: float(get_fx_rate(c, base)))
    out["Importo Investito Base"] = out["Importo Investito"].astype(float) * out["FX Rate"]
    out["Valore di Mercato Base"] = out["Valore di Mercato"].astype(float) * out["FX Rate"]
    out["P/L Base"] = out["P/L"].astype(float) * out["FX Rate"]
    total_base_mv = float(out["Valore di Mercato Base"].sum())
    out["Peso Base"] = np.where(total_base_mv > 0, out["Valore di Mercato Base"] / total_base_mv * 100.0, 0.0)
    return out

def calculate_tax_impact_df_base(dfweights_base: pd.DataFrame, tax_rate: float = DEFAULT_TAX_RATE) -> pd.DataFrame:
    if dfweights_base is None or dfweights_base.empty:
        return pd.DataFrame()
    dftax = dfweights_base.copy()
    dftax["Aliquota Fiscale %"] = tax_rate * 100.0
    dftax["PlusMinus Lorda Base"] = dftax["P/L Base"].astype(float)
    dftax["Imposta Teorica Base"] = np.where(dftax["PlusMinus Lorda Base"] > 0, dftax["PlusMinus Lorda Base"] * tax_rate, 0.0)
    dftax["PlusMinus Netta Base"] = dftax["PlusMinus Lorda Base"] - dftax["Imposta Teorica Base"]
    dftax["Valore Netto Post Imposta Base"] = dftax["Valore di Mercato Base"] - dftax["Imposta Teorica Base"]
    dftax["Rendimento Netto Base"] = np.where(dftax["Importo Investito Base"] > 0, dftax["PlusMinus Netta Base"] / dftax["Importo Investito Base"] * 100.0, 0.0)
    return dftax


# ==========================================
# CONFIGURAZIONE PAGINA UI
# ==========================================
st.set_page_config(
    page_title="BurryInvestingPro",
    page_icon="💎",
    layout="wide"
)


# ==========================================
# 0.B AUTH SUPABASE
# ==========================================
SUPABASE_PROJECT_REF = "fuupxyksbaylznlboawy"
SUPABASE_ANON_KEY_FALLBACK = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZ1dXB4eWtzYmF5bHpubGJvYXd5Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3Nzc4OTU4NzQsImV4cCI6MjA5MzQ3MTg3NH0.j6XW9WK2IZOFw0HLH-M4G-QGBl60fYLx_IJQL-nAMAY"
SUPABASE_URL_FALLBACK = f"https://{SUPABASE_PROJECT_REF}.supabase.co"

SUPABASE_PORTFOLIO_SQL = """
create extension if not exists pgcrypto;

create table if not exists portfolio_positions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null,
  ticker text not null,
  quantity numeric not null default 0,
  pmc numeric not null default 0,
  currency text not null default 'USD',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint portfolio_positions_user_ticker_unique unique (user_id, ticker)
);

create index if not exists portfolio_positions_user_id_idx
  on portfolio_positions(user_id);

create or replace function set_portfolio_positions_updated_at()
returns trigger as $$
begin
  new.updated_at = now();
  return new;
end;
$$ language plpgsql;

drop trigger if exists trg_portfolio_positions_updated_at on portfolio_positions;
create trigger trg_portfolio_positions_updated_at
before update on portfolio_positions
for each row execute function set_portfolio_positions_updated_at();

alter table portfolio_positions enable row level security;

drop policy if exists "portfolio_select_own" on portfolio_positions;
create policy "portfolio_select_own"
on portfolio_positions
for select
using (auth.uid() = user_id);

drop policy if exists "portfolio_insert_own" on portfolio_positions;
create policy "portfolio_insert_own"
on portfolio_positions
for insert
with check (auth.uid() = user_id);

drop policy if exists "portfolio_update_own" on portfolio_positions;
create policy "portfolio_update_own"
on portfolio_positions
for update
using (auth.uid() = user_id)
with check (auth.uid() = user_id);

drop policy if exists "portfolio_delete_own" on portfolio_positions;
create policy "portfolio_delete_own"
on portfolio_positions
for delete
using (auth.uid() = user_id);
"""


def get_app_base_url() -> str:
    candidates = [os.getenv('APP_BASE_URL'), os.getenv('STREAMLIT_APP_URL'), os.getenv('PUBLIC_APP_URL')]
    for c in candidates:
        if c and str(c).strip().startswith(('http://', 'https://')):
            return str(c).strip().rstrip('/')
    return 'http://localhost:8501'

def get_email_redirect_url() -> str:
    return get_app_base_url()


def get_supabase_credentials() -> Tuple[str, str]:
    url = os.getenv('SUPABASE_URL', SUPABASE_URL_FALLBACK)
    key = os.getenv('SUPABASE_ANON_KEY', SUPABASE_ANON_KEY_FALLBACK)
    try:
        if 'SUPABASE_URL' in st.secrets and st.secrets['SUPABASE_URL']:
            url = str(st.secrets['SUPABASE_URL']).strip()
        if 'SUPABASE_ANON_KEY' in st.secrets and st.secrets['SUPABASE_ANON_KEY']:
            key = str(st.secrets['SUPABASE_ANON_KEY']).strip()
    except Exception:
        pass
    return url, key

@st.cache_resource(show_spinner=False)
def get_supabase_client() -> Client:
    url, key = get_supabase_credentials()
    return create_client(url, key)

def init_auth_state() -> None:
    if 'auth_user' not in st.session_state:
        st.session_state.auth_user = None
    if 'auth_session' not in st.session_state:
        st.session_state.auth_session = None
    if 'auth_error' not in st.session_state:
        st.session_state.auth_error = None

def get_logged_user_email() -> Optional[str]:
    user = st.session_state.get('auth_user')
    if not user:
        return None
    if isinstance(user, dict):
        return user.get('email')
    return getattr(user, 'email', None)

def is_authenticated() -> bool:
    return get_logged_user_email() is not None

def _extract_auth_payload(auth_response: Any) -> Tuple[Any, Any]:
    user = getattr(auth_response, 'user', None)
    session = getattr(auth_response, 'session', None)
    return user, session

def sign_up_with_supabase(email: str, password: str) -> Tuple[bool, str]:
    try:
        supabase = get_supabase_client()
        response = supabase.auth.sign_up({
            'email': email.strip(),
            'password': password,
            'options': {'email_redirect_to': get_email_redirect_url()}
        })
        user, session = _extract_auth_payload(response)
        if user is None:
            return False, 'Registrazione non completata. Controlla eventuali restrizioni del progetto Supabase.'
        st.session_state.auth_user = user
        st.session_state.auth_session = session
        if session is None:
            return True, "Registrazione eseguita. Controlla la tua email per confermare l'account, se la conferma è attiva."
        return True, 'Registrazione completata con accesso effettuato.'
    except Exception as e:
        return False, f'Registrazione fallita: {e}'

def sign_in_with_supabase(email: str, password: str) -> Tuple[bool, str]:
    try:
        supabase = get_supabase_client()
        response = supabase.auth.sign_in_with_password({
            'email': email.strip(),
            'password': password
        })
        user, session = _extract_auth_payload(response)
        if user is None:
            return False, 'Login non riuscito. Verifica email e password.'
        st.session_state.auth_user = user
        st.session_state.auth_session = session
        return True, 'Login eseguito con successo.'
    except Exception as e:
        return False, f'Login fallito: {e}'

def sign_out_from_supabase() -> Tuple[bool, str]:
    try:
        supabase = get_supabase_client()
        try:
            supabase.auth.sign_out()
        except Exception:
            pass
        st.session_state.auth_user = None
        st.session_state.auth_session = None
        return True, 'Logout eseguito con successo.'
    except Exception as e:
        return False, f'Logout fallito: {e}'

def get_logged_user_id() -> Optional[str]:
    user = st.session_state.get("auth_user")
    if not user:
        return None
    if isinstance(user, dict):
        return user.get("id")
    return getattr(user, "id", None)

def is_guest_mode() -> bool:
    return not is_authenticated()

def ensure_portfolio_state() -> None:
    ensure_portfolio_state()
    maybe_autoload_portfolio()
    if "portfolio_loaded_for_user" not in st.session_state:
        st.session_state.portfolio_loaded_for_user = None

def clear_portfolio_session() -> None:
    st.session_state.portfolio_tickers = []
    st.session_state.holdings = {}
    st.session_state.holdings_currency = {}
    st.session_state.holdings_quantity = {}
    st.session_state.holdings_pmc = {}

def load_user_portfolio() -> Tuple[bool, str]:
    ensure_portfolio_state()
    user_id = get_logged_user_id()
    if not user_id:
        return False, "Utente non autenticato."
    try:
        supabase = get_supabase_client()
        response = supabase.table("portfolio_positions").select("*").eq("user_id", user_id).execute()
        rows = response.data or []
        clear_portfolio_session()
        for row in rows:
            ticker = str(row.get("ticker", "")).strip().upper()
            if not ticker:
                continue
            quantity = float(row.get("quantity") or 0.0)
            pmc = float(row.get("pmc") or 0.0)
            currency = str(row.get("currency") or "USD").upper().strip() or "USD"
            if ticker not in st.session_state.portfolio_tickers:
                st.session_state.portfolio_tickers.append(ticker)
            st.session_state.holdings_quantity[ticker] = quantity
            st.session_state.holdings_pmc[ticker] = pmc
            st.session_state.holdings_currency[ticker] = currency
            st.session_state.holdings[ticker] = quantity * pmc
        st.session_state.portfolio_loaded_for_user = user_id
        return True, f"Portafoglio caricato: {len(rows)} posizioni."
    except Exception as e:
        return False, f"Errore caricamento portafoglio: {e}"

def save_user_portfolio_position(ticker: str, quantity: float, pmc: float, currency: str) -> Tuple[bool, str]:
    user_id = get_logged_user_id()
    if not user_id:
        return False, "Modalità ospite: salvataggio non disponibile."
    try:
        payload = {
            "user_id": user_id,
            "ticker": str(ticker).strip().upper(),
            "quantity": float(quantity or 0.0),
            "pmc": float(pmc or 0.0),
            "currency": str(currency or "USD").strip().upper() or "USD",
        }
        supabase = get_supabase_client()
        supabase.table("portfolio_positions").upsert(payload, on_conflict="user_id,ticker").execute()
        return True, f"Posizione {payload['ticker']} salvata."
    except Exception as e:
        return False, f"Errore salvataggio posizione: {e}"

def save_all_portfolio_positions() -> Tuple[bool, str]:
    user_id = get_logged_user_id()
    if not user_id:
        return False, "Modalità ospite: salvataggio non disponibile."
    ensure_portfolio_state()
    tickers = list(st.session_state.get("portfolio_tickers", []))
    try:
        supabase = get_supabase_client()
        existing = supabase.table("portfolio_positions").select("ticker").eq("user_id", user_id).execute()
        existing_tickers = {str(x.get("ticker", "")).upper() for x in (existing.data or [])}
        current_tickers = {str(t).upper() for t in tickers}
        to_delete = existing_tickers - current_tickers
        for ticker in to_delete:
            supabase.table("portfolio_positions").delete().eq("user_id", user_id).eq("ticker", ticker).execute()
        for ticker in tickers:
            quantity = float(st.session_state.holdings_quantity.get(ticker, 0.0) or 0.0)
            pmc = float(st.session_state.holdings_pmc.get(ticker, 0.0) or 0.0)
            currency = str(st.session_state.holdings_currency.get(ticker, "USD") or "USD")
            supabase.table("portfolio_positions").upsert({
                "user_id": user_id,
                "ticker": ticker,
                "quantity": quantity,
                "pmc": pmc,
                "currency": currency.upper().strip() or "USD",
            }, on_conflict="user_id,ticker").execute()
        st.session_state.portfolio_loaded_for_user = user_id
        return True, f"Portafoglio salvato: {len(tickers)} posizioni."
    except Exception as e:
        return False, f"Errore salvataggio portafoglio: {e}"

def delete_user_portfolio_position(ticker: str) -> Tuple[bool, str]:
    user_id = get_logged_user_id()
    if not user_id:
        return False, "Modalità ospite: nessun dato remoto da eliminare."
    try:
        supabase = get_supabase_client()
        supabase.table("portfolio_positions").delete().eq("user_id", user_id).eq("ticker", str(ticker).strip().upper()).execute()
        return True, f"Posizione {ticker} eliminata dal cloud."
    except Exception as e:
        return False, f"Errore eliminazione posizione: {e}"

def maybe_autoload_portfolio() -> None:
    ensure_portfolio_state()
    user_id = get_logged_user_id()
    if not user_id:
        return
    if st.session_state.get("portfolio_loaded_for_user") == user_id:
        return
    load_user_portfolio()

def render_auth_sidebar() -> None:
    st.sidebar.markdown('### 👤 Account')
    current_email = get_logged_user_email()
    if current_email:
        st.sidebar.success(f'Connesso come: {current_email}')
        if st.sidebar.button('🚪 Logout', use_container_width=True):
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
        if st.sidebar.button('📝 Crea account', use_container_width=True):
            if not email or '@' not in email:
                st.sidebar.error('Inserisci una email valida.')
            elif len(password) < 6:
                st.sidebar.error('La password deve contenere almeno 6 caratteri.')
            elif password != password_confirm:
                st.sidebar.error('Le password non coincidono.')
            else:
                ok, msg = sign_up_with_supabase(email, password)
                if ok:
                    st.sidebar.success(msg)
                    st.rerun()
                else:
                    st.sidebar.error(msg)
    else:
        if st.sidebar.button('🔐 Login', use_container_width=True):
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

def require_login_screen() -> None:
    st.title('💎 BurryInvestingPro')
    st.info("Per usare l'app devi prima registrarti o accedere dal menu a tendina di sinistra.")
    st.stop()

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
    debt_to_equity: Optional[float]
    revenue_growth: Optional[float]
    net_margin: Optional[float]
    fcf_margin: Optional[float]
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
            "Debt/Equity": self.debt_to_equity,
            "Revenue Growth": self.revenue_growth,
            "Net Margin": self.net_margin,
            "FCF Margin": self.fcf_margin,
            "Currency": self.currency,
            "_raw_data": self.raw_data,
        }


# ==========================================
# 2. HELPER FUNCTIONS & VALIDAZIONE
# ==========================================
def sanitize_ticker(ticker: str) -> str:
    clean = str(ticker or '').strip().upper()
    if not clean:
        raise ValueError('Ticker vuoto')
    if not re.match(r'^[A-Z0-9\-\.=]+$', clean):
        raise ValueError(f'Ticker contiene caratteri non validi: {clean}')
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
        if not info:
            return None
        if 'symbol' not in info:
            info['symbol'] = symbol
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
                debt_to_equity=None,
                revenue_growth=None,
                net_margin=None,
                fcf_margin=None,
                currency=info.get('currency', 'USD'),
                raw_data=raw_data
            )

        fin, bs, cf = raw_data["financials"], raw_data["balance_sheet"], raw_data["cashflow"]

        op_cash = cf.loc['Operating Cash Flow'].iloc[0] if 'Operating Cash Flow' in cf.index else 0.0
        cap_ex = cf.loc['Capital Expenditure'].iloc[0] if 'Capital Expenditure' in cf.index else 0.0
        fcf = float(op_cash - abs(cap_ex))

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
        total_revenue = info.get('totalRevenue')
        net_income = info.get('netIncomeToCommon')
        revenue_growth = info.get('revenueGrowth')

        debt_to_equity = None
        if equity is not None and not np.isnan(equity) and equity != 0:
            debt_to_equity = float(total_debt / equity)

        net_margin = None
        if total_revenue is not None and total_revenue != 0 and net_income is not None:
            net_margin = float(net_income / total_revenue)

        fcf_margin = None
        if total_revenue is not None and total_revenue != 0:
            fcf_margin = float(fcf / total_revenue)

        return FundamentalMetrics(
            ticker=raw_data["symbol"],
            company_name=info.get('longName', raw_data["symbol"]),
            price=float(info.get('currentPrice', 0.0)),
            fcf=fcf,
            roic=roic,
            peg_ratio=float(peg) if peg is not None else None,
            peg_source=peg_src,
            pe_ratio=float(pe) if pe is not None else None,
            interest_coverage=int_cov,
            debt_to_equity=debt_to_equity,
            revenue_growth=float(revenue_growth) if revenue_growth is not None else None,
            net_margin=net_margin,
            fcf_margin=fcf_margin,
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
        df = df.loc[:, ~df.columns.duplicated(keep='first')]
        return df if len(df) >= 200 else None
    except Exception:
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
    data['RSI'] = 100 - (100 / (1 + rs))
    ma20 = close.rolling(20, min_periods=20).mean()
    std20 = close.rolling(20, min_periods=20).std(ddof=0)
    data['BB_Lower'] = ma20 - 2 * std20
    data['BB_Upper'] = ma20 + 2 * std20
    if ta is not None:
        try:
            data['SMA_50'] = ta.sma(close, length=50)
            data['SMA_200'] = ta.sma(close, length=200)
            data['RSI'] = ta.rsi(close, length=14)
            bb = ta.bbands(close, length=20, std=2)
            if bb is not None:
                low = bb.filter(like='BBL_')
                up = bb.filter(like='BBU_')
                if not low.empty:
                    data['BB_Lower'] = low.iloc[:, 0]
                if not up.empty:
                    data['BB_Upper'] = up.iloc[:, 0]
        except Exception:
            pass
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

    return score, reasons


# ==========================================
# 5. MOTORE QUANTISTICO (STATISTICA STAZIONARIA)
# ==========================================
def calculate_quant_metrics(df: pd.DataFrame, fund_data: Dict[str, Any]) -> Dict[str, Any]:
    returns = df['Close'].pct_change().dropna()
    excess_returns = returns - (RISK_FREE_RATE / TRADING_DAYS_YEAR)
    sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(TRADING_DAYS_YEAR) if excess_returns.std() != 0 else 0
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
            if ta_val == 0:
                raise ValueError("Total Assets è zero, impossibile calcolare Z-Score")
            wc = (bs.loc['Current Assets'].iloc[0] - bs.loc['Current Liabilities'].iloc[0]) if 'Current Assets' in bs.index else 0
            re = bs.loc['Retained Earnings'].iloc[0] if 'Retained Earnings' in bs.index else 0
            ebit = fin.loc['EBIT'].iloc[0]
            mc = info.get('marketCap') or 1  # Evita fallback fallaci a None
            tl = bs.loc['Total Liabilities Net Minority Interest'].iloc[0]
            if tl == 0:
                raise ValueError("Total Liabilities è zero, impossibile calcolare Z-Score")
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

    z = qm.get("Altman Z-Score", "N/A")
    if isinstance(z, (int, float, np.floating)):
        if z >= 3.0:
            f_score += 7.0
        elif z >= 1.8:
            f_score += 3.5

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
    
def calculate_tax_impact(df_weights: pd.DataFrame, tax_rate: float = DEFAULT_TAX_RATE) -> pd.DataFrame:
    if df_weights is None or df_weights.empty:
        return pd.DataFrame()

    df_tax = df_weights.copy()

    df_tax["Aliquota Fiscale %"] = tax_rate * 100.0
    df_tax["Plus/Minus Lorda"] = df_tax["P&L"].astype(float)

    df_tax["Imposta Teorica"] = np.where(
        df_tax["Plus/Minus Lorda"] > 0,
        df_tax["Plus/Minus Lorda"] * tax_rate,
        0.0
    )

    df_tax["Plus/Minus Netta"] = df_tax["Plus/Minus Lorda"] - df_tax["Imposta Teorica"]

    df_tax["Valore Netto Post Imposta"] = df_tax["Valore di Mercato"] - df_tax["Imposta Teorica"]

    df_tax["Rendimento Netto %"] = np.where(
        df_tax["Importo Investito"] > 0,
        (df_tax["Plus/Minus Netta"] / df_tax["Importo Investito"]) * 100.0,
        0.0
    )

    cols = [
        "Ticker",
        "Importo Investito",
        "Valore di Mercato",
        "P&L",
        "Aliquota Fiscale %",
        "Imposta Teorica",
        "Plus/Minus Netta",
        "Valore Netto Post Imposta",
        "Rendimento Netto %"
    ]

    existing_cols = [c for c in cols if c in df_tax.columns]
    return df_tax[existing_cols]

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
# 5.F HELPER PORTAFOGLIO AVANZATO & PWA
# ==========================================
def get_latest_price(symbol: str) -> Optional[float]:
    df = get_technical_data(symbol)
    if df is not None and not df.empty and 'Close' in df.columns:
        try:
            return float(df['Close'].dropna().iloc[-1])
        except Exception:
            pass
    raw = get_fundamental_data(symbol)
    if raw and raw.get('info'):
        info = raw['info']
        price = info.get('currentPrice') or info.get('regularMarketPrice')
        if price is not None:
            return float(price)
    return None


def inject_pwa_support():
    st.markdown("""
    <script>
    (function(){
      const base64Png = 'iVBORw0KGgoAAAANSUhEUgAAAMAAAADACAIAAADdvvtQAAACNklEQVR4nO3SwQ3AIBDAsNL9dz6WIEJC9gR5ZM18A6ft2wG8yQBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA9gBTjICfuDZUUYAAAAASUVORK5CYII=';
      const manifest = {
        name: 'BurryInvestingPro',
        short_name: 'BurryPro',
        description: 'Analisi investimenti e portafoglio installabile su smartphone',
        start_url: '.',
        display: 'standalone',
        background_color: '#0e1117',
        theme_color: '#0e1117',
        icons: [
          { src: 'data:image/png;base64,' + base64Png, sizes: '192x192', type: 'image/png' },
          { src: 'data:image/png;base64,' + base64Png, sizes: '512x512', type: 'image/png' }
        ]
      };
      const manifestBlob = new Blob([JSON.stringify(manifest)], {type: 'application/manifest+json'});
      const manifestUrl = URL.createObjectURL(manifestBlob);
      const link = document.createElement('link');
      link.rel = 'manifest';
      link.href = manifestUrl;
      document.head.appendChild(link);

      const appleIcon = document.createElement('link');
      appleIcon.rel = 'apple-touch-icon';
      appleIcon.href = 'data:image/png;base64,' + base64Png;
      document.head.appendChild(appleIcon);

      const meta1 = document.createElement('meta');
      meta1.name = 'apple-mobile-web-app-capable';
      meta1.content = 'yes';
      document.head.appendChild(meta1);

      const meta2 = document.createElement('meta');
      meta2.name = 'apple-mobile-web-app-status-bar-style';
      meta2.content = 'black-translucent';
      document.head.appendChild(meta2);

      const meta3 = document.createElement('meta');
      meta3.name = 'apple-mobile-web-app-title';
      meta3.content = 'BurryPro';
      document.head.appendChild(meta3);

      const meta4 = document.createElement('meta');
      meta4.name = 'theme-color';
      meta4.content = '#0e1117';
      document.head.appendChild(meta4);

      const swCode = `
        self.addEventListener('install', event => { self.skipWaiting(); });
        self.addEventListener('activate', event => { event.waitUntil(self.clients.claim()); });
        self.addEventListener('fetch', event => {
          if (event.request.method !== 'GET') return;
          event.respondWith(fetch(event.request).catch(() => caches.match(event.request)));
        });
      `;
      if ('serviceWorker' in navigator) {
        const swBlob = new Blob([swCode], { type: 'text/javascript' });
        const swUrl = URL.createObjectURL(swBlob);
        navigator.serviceWorker.register(swUrl).catch(() => {});
      }
    })();
    </script>
    """, unsafe_allow_html=True)

def calculate_position_from_quantity(ticker: str, quantity: float, pmc: float) -> Dict[str, float]:
    current_price = get_latest_price(ticker)
    invested = float(quantity * pmc)
    market_value = float(quantity * current_price) if current_price is not None else 0.0
    pnl_value = market_value - invested if current_price is not None else 0.0
    pnl_pct = (pnl_value / invested) * 100.0 if invested > 0 and current_price is not None else 0.0
    return {
        'Prezzo Attuale': float(current_price) if current_price is not None else np.nan,
        'Importo Investito': invested,
        'Valore di Mercato': market_value,
        'P&L': pnl_value,
        'P&L %': pnl_pct,
    }

# ==========================================
# 5.F PORTAFOGLIO: ALLOCAZIONE & RIBILANCIAMENTO
# ==========================================
def infer_asset_class(ticker: str, company_name: str = "") -> str:
    t = str(ticker).upper()
    name = str(company_name).lower()

    etf_keywords = ["etf", "ucits", "ishares", "xtrackers", "vanguard", "lyxor", "amundi", "invesco", "wisdomtree"]
    bond_keywords = ["bond", "treasury", "aggregate", "gov", "government"]
    gold_keywords = ["gold", "physical gold", "precious", "silver", "metals"]
    crypto_keywords = ["btc-", "eth-", "-usd"]

    if any(k in name for k in etf_keywords):
        if any(k in name for k in bond_keywords):
            return "ETF Obbligazionario"
        if any(k in name for k in gold_keywords):
            return "ETF/ETC Oro"
        return "ETF Azionario"

    if any(k in t for k in crypto_keywords):
        return "Crypto"

    if any(k in name for k in bond_keywords):
        return "Obbligazione/Fondo Bond"

    if any(k in name for k in gold_keywords):
        return "Oro/Metalli"

    return "Azione"


def infer_geography(ticker: str, company_name: str = "") -> str:
    t = str(ticker).upper()
    name = str(company_name).lower()

    if t.endswith(".MI"):
        return "Italia"
    if t.endswith(".DE"):
        return "Germania"
    if t.endswith(".PA"):
        return "Francia"
    if t.endswith(".L"):
        return "Regno Unito"
    if "-USD" in t:
        return "Crypto/USD"

    us_keywords = ["s&p", "nasdaq", "russell", "usa", "united states", "msci usa"]
    eu_keywords = ["europe", "stoxx", "euro stoxx", "msci europe"]
    em_keywords = ["emerging", "em", "msci em"]
    world_keywords = ["world", "all-world", "acwi", "ftse all-world"]
    japan_keywords = ["japan", "topix", "nikkei"]
    china_keywords = ["china", "csi", "hang seng"]

    if any(k in name for k in world_keywords):
        return "Globale"
    if any(k in name for k in us_keywords):
        return "USA"
    if any(k in name for k in eu_keywords):
        return "Europa"
    if any(k in name for k in em_keywords):
        return "Emergenti"
    if any(k in name for k in japan_keywords):
        return "Giappone"
    if any(k in name for k in china_keywords):
        return "Cina"

    return "Da classificare"


def build_portfolio_allocation_df(
    positive_holdings: Dict[str, float],
    holdings_currency: Dict[str, str]
) -> pd.DataFrame:
    rows = []

    for ticker, amount in positive_holdings.items():
        raw = get_fundamental_data(ticker)
        info = raw["info"] if raw and "info" in raw else {}
        company_name = info.get("longName", info.get("shortName", ticker))
        detected_currency = holdings_currency.get(ticker, info.get("currency", "USD"))

        rows.append({
            "Ticker": ticker,
            "Company Name": company_name,
            "Importo": float(amount),
            "Valuta": detected_currency,
            "Asset Class": infer_asset_class(ticker, company_name),
            "Geografia": infer_geography(ticker, company_name)
        })

    df_alloc = pd.DataFrame(rows)
    if df_alloc.empty:
        return df_alloc

    total = df_alloc["Importo"].sum()
    df_alloc["Peso %"] = np.where(total > 0, df_alloc["Importo"] / total * 100.0, 0.0)
    return df_alloc.sort_values("Peso %", ascending=False).reset_index(drop=True)


def summarize_group_weights(df_alloc: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if df_alloc.empty or group_col not in df_alloc.columns:
        return pd.DataFrame()

    out = (
        df_alloc.groupby(group_col, dropna=False)["Importo"]
        .sum()
        .reset_index()
        .sort_values("Importo", ascending=False)
    )

    total = out["Importo"].sum()
    out["Peso %"] = np.where(total > 0, out["Importo"] / total * 100.0, 0.0)
    return out.reset_index(drop=True)


def compute_rebalancing_actions(
    df_alloc: pd.DataFrame,
    target_weights: Dict[str, float],
    group_col: str = "Ticker",
    tolerance_pct: float = 1.0
) -> pd.DataFrame:
    if df_alloc.empty:
        return pd.DataFrame()

    current = summarize_group_weights(df_alloc, group_col)
    if current.empty:
        return pd.DataFrame()

    target_df = pd.DataFrame({
        group_col: list(target_weights.keys()),
        "Target %": list(target_weights.values())
    })

    merged = current.merge(target_df, on=group_col, how="outer").fillna(0.0)
    total_value = df_alloc["Importo"].sum()

    merged["Scostamento %"] = merged["Target %"] - merged["Peso %"]
    merged["Azione €"] = total_value * merged["Scostamento %"] / 100.0
    merged["Azione"] = np.where(
        merged["Azione €"] > tolerance_pct / 100.0 * total_value,
        "Compra",
        np.where(
            merged["Azione €"] < -tolerance_pct / 100.0 * total_value,
            "Riduci",
            "In target"
        )
    )

    return merged.sort_values("Azione €", ascending=False).reset_index(drop=True)

# ==========================================
# 6. UI: SIDEBAR & STYLE
# ==========================================


def render_apk_download_box() -> None:
    with st.sidebar.expander("Download App Android (APK)", expanded=False):
        st.write("Scarica l'APK ufficiale di BurryInvestingPro per installare l'app su Android.")
        st.link_button(
            "📲 Scarica Burryinvestingpro.apk",
            "https://github.com/innovativeprogram/Burryinvestingpro-release/releases/latest/download/Burryinvestingpro.apk",
            width="stretch"
        )
        st.caption("Se Android blocca l'installazione, abilita temporaneamente le origini sconosciute per il browser o il file manager usato per il download.")
def setup_sidebar() -> Dict[str, Any]:
    render_auth_sidebar()
    st.sidebar.header("1. Selezione Asset")

    input_mode = st.sidebar.radio(
        "Modalità",
        ["Manuale", "Batch CSV"],
        horizontal=True
    )

    file, manual = None, None
    if input_mode == "Batch CSV":
        file = st.sidebar.file_uploader("Carica CSV", type=["csv"])
    else:
        manual = st.sidebar.text_input("Ticker", value="AAPL").upper().strip()

    st.sidebar.header("2. Mercato")
    market = st.sidebar.selectbox(
        "Borsa",
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
            "roic": st.number_input("Min ROIC %", 10.0, step=0.5),
            "fcf": st.number_input("Min FCF (Mld)", 0.0, step=1e9),
            "peg": st.number_input("Max PEG Ratio", 1.5, step=0.1),
            "pe": st.number_input("Max PE (Fallback)", 25.0),
            "intcov": st.number_input("Min Int. Coverage", 3.0),

            "custom_max_de": st.number_input("Custom Max Debt/Equity", 1.0, step=0.1),
            "custom_min_fcf_margin": st.number_input("Custom Min FCF Margin", 0.08, step=0.01, format="%.2f"),
            "custom_min_net_margin": st.number_input("Custom Min Net Margin", 0.10, step=0.01, format="%.2f"),

            "perfectonly": st.checkbox("Solo All Green"),
            "model_mode": st.selectbox(
                "Modello verdetto",
                ["Entrambi", "Classico", "Evoluto", "Personalizzabile"],
                index=0
            ),
        }

    with st.sidebar.expander("❓ Come cercare il ticker corretto"):
        st.markdown("""
- Azioni USA normalmente solo ticker, es. AAPL, MSFT.
- Azioni italiane aggiungi .MI, es. STLAM.MI, ENI.MI, ISP.MI.
- Azioni tedesche aggiungi .DE, es. BMW.DE, SAP.DE.
- Azioni francesi aggiungi .PA, es. AIR.PA, OR.PA.
- Azioni UK aggiungi .L, es. ULVR.L.
- Crypto di solito coppia con valuta, es. BTC-USD, ETH-USD.
- Se hai dubbi, cerca prima il titolo su Yahoo Finance e copia il ticker esatto.
        """)

    render_apk_download_box()

    with st.sidebar.expander("Contatti", expanded=True):
        st.write("Per supporto tecnico, collaborazioni o richieste:")
        st.link_button(
            "📧 Scrivimi via mail",
            "mailto:innovativeprogram@proton.me?subject=Richiesta%20da%20BurryInvestingPro",
            width="stretch"
        )
        st.caption("Risposta normalmente entro 24/48 ore.")

    return {
        "mode": input_mode,
        "file": file,
        "manual": manual,
        "suffix": suffix,
        "btn": analyze_btn,
        "cfg": cfg
    }
# ==========================================
# 7. MAIN ORCHESTRATOR
# ==========================================
def main():
    init_auth_state()
    st.title("💎 BurryInvestingPro")
    inject_pwa_support()
    if 'localhost' in get_app_base_url() or '127.0.0.1' in get_app_base_url():
        st.warning("Configura APP_BASE_URL con l'URL pubblico della tua app per evitare errori nei link email di conferma.")
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
    if 'portfolio_target_mode' not in st.session_state:
        st.session_state.portfolio_target_mode = "Ticker"
    if 'portfolio_targets' not in st.session_state:
        st.session_state.portfolio_targets = {}
    if 'analysis_errors' not in st.session_state:
        st.session_state.analysis_errors = []

    ui = setup_sidebar()
    if is_guest_mode():
        st.info("Modalità ospite attiva: puoi usare analisi, tab e portafoglio locale senza registrazione. Per salvare il portafoglio in modo permanente, effettua il login.")
    if ui["btn"]:
        targets: List[str] = [ui["manual"]] if ui["mode"] == "Manuale" else []
        if ui["mode"] == "Batch CSV" and ui["file"]:
            csv_df = pd.read_csv(ui["file"])
            if 'Ticker' not in csv_df.columns:
                st.error('Il CSV deve contenere una colonna Ticker.')
                targets = []
            else:
                targets = csv_df['Ticker'].dropna().astype(str).tolist()[:MAX_CSV_ROWS]

        if targets:
            results: List[Dict[str, Any]] = []
            analysis_errors: List[str] = []
            normalized_targets: List[str] = []
            for t in targets:
                try:
                    normalized_targets.append(normalize_ticker(t, ui["suffix"]))
                except Exception as e:
                    analysis_errors.append(f'{t}: {e}')
            for t in normalized_targets:
                try:
                    raw = get_fundamental_data(t)
                    if not raw:
                        analysis_errors.append(f'{t}: nessun dato fondamentale disponibile')
                        continue
                    met = calculate_fundamental_metrics(raw)
                    if met:
                        results.append(met.to_ui_dict())
                    else:
                        analysis_errors.append(f'{t}: impossibile calcolare le metriche')
                except Exception as e:
                    analysis_errors.append(f'{t}: {e}')
            st.session_state.batch_results = pd.DataFrame(results)
            st.session_state.analysis_errors = analysis_errors
            if results:
                st.session_state.selected_ticker = results[0]["Ticker"]
            else:
                st.session_state.selected_ticker = None

    if st.session_state.get('analysis_errors'):
        with st.expander('⚠️ Diagnostica analisi', expanded=not bool(st.session_state.get('batch_results') is not None and not st.session_state.batch_results.empty)):
            for err in st.session_state.analysis_errors:
                st.write(f'- {err}')

    tab_f, tab_t, tab_q, tab_v, tab_p = st.tabs(["📊 FONDAMENTALI", "📉 TECNICO", "⚛️ QUANT", "⚖️ VERDETTO", "📁 PORTAFOGLIO"])

    ticker = st.session_state.selected_ticker
    if st.session_state.analysis_errors:
        st.warning('Alcuni ticker non sono stati caricati correttamente: ' + ' | '.join(st.session_state.analysis_errors[:5]))
    if ticker and st.session_state.batch_results is not None and not st.session_state.batch_results.empty and ticker in st.session_state.batch_results['Ticker'].values:
        row = st.session_state.batch_results[st.session_state.batch_results['Ticker'] == ticker].iloc[0]
    elif st.session_state.batch_results is None or st.session_state.batch_results.empty:
        st.info('Nessun dato disponibile. Verifica il ticker inserito, il mercato selezionato e la connessione ai dati Yahoo Finance.')
        row = None
    else:
        st.info('Ticker selezionato non presente nei risultati correnti.')
        row = None

    if row is not None:

        # --- TAB FONDAMENTALI ---
        with tab_f:
            st.info("💡 **Come leggere questa sezione:** Qui analizzi la qualità economica e finanziaria del business, non il movimento del prezzo. Le metriche principali ti aiutano a capire se l'azienda crea valore in modo efficiente, se cresce con equilibrio e se il debito resta sostenibile. **ROIC** misura quanto bene il management reinveste il capitale; **Free Cash Flow** indica il denaro realmente generato; **PEG Ratio** mette in relazione valutazione e crescita; **Interest Coverage** e **Debt/Equity** servono per controllare la solidità finanziaria. Nelle nuove aggiunte trovi anche **Revenue Growth**, **Net Margin** e **FCF Margin**: la prima misura la crescita del fatturato, la seconda la redditività finale, la terza la capacità di trasformare i ricavi in cassa vera. Questa tab va letta così: prima qualità del business, poi sostenibilità finanziaria, solo alla fine prezzo e multipli.")
            
            st.dataframe(st.session_state.batch_results.drop(columns=["_raw_data"]))
            
            st.markdown("---")
            st.markdown("<p style='text-align: center; color: gray;'>creato e sviluppato da Innovative Program </p>", unsafe_allow_html=True)

        # --- TAB TECNICO ---
        with tab_t:
            st.info("💡 **Come leggere il grafico:** Questa tab non serve a dire se un'azienda è buona, ma a capire **quando** il mercato la sta premiando o penalizzando. La candela mostra il prezzo, la **SMA 200** identifica il trend di fondo e l'**RSI** misura se il movimento recente è troppo tirato o troppo depresso. Il **Timing Score** nasce dalla combinazione delle regole tecniche del programma: premio al prezzo sopra SMA 200, premio aggiuntivo in caso di ipervenduto RSI e ulteriore supporto quando il prezzo si avvicina alla banda bassa di Bollinger. Va quindi interpretato come un indicatore di contesto: punteggio alto significa setup tecnico più favorevole, non certezza di rialzo.")
            
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
            else:
                st.warning(f'Dati tecnici non disponibili per {ticker}.')
            
            st.markdown("---")
            st.markdown("<p style='text-align: center; color: gray;'>creato e sviluppato da Innovative Program </p>", unsafe_allow_html=True)

        # --- TAB QUANT ---
        with tab_q:
            st.info("💡 **Come interpretare i dati:** In questa tab il programma misura la qualità statistica del titolo e il suo profilo di rischio-rendimento. **Sharpe Ratio** valuta quanto rendimento ottieni per unità di rischio, **R-Squared** misura quanto il trend è lineare e pulito, mentre **Altman Z-Score** aiuta a identificare aziende potenzialmente fragili sul piano patrimoniale. Le nuove aggiunte più importanti sono i **risk metrics**: **Max Drawdown** per la perdita massima storica dal picco, **CAGR** per la crescita composta annua, **VaR 95%** per la perdita giornaliera attesa in scenari normali estremi e **CVaR 95%** per la severità media delle perdite oltre quel livello. Anche **Skewness** e **Kurtosis** sono utili: la prima indica l'asimmetria dei rendimenti, la seconda segnala la presenza di code estreme. La simulazione **Monte Carlo** non prevede il futuro, ma mostra un ventaglio di esiti possibili partendo dal comportamento storico del titolo, così puoi ragionare in termini probabilistici e non emotivi.")
            
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

                df_calc_q = calculate_technical_indicators(df_tech)
                score_q, _ = calculate_timing_score(df_calc_q, df_calc_q['Close'].iloc[-1])

                smart = compute_smart_quant_score(row, score_q, qm, risk)
                st.metric("Smart Quant Score", f"{smart['SmartScore']:.1f}/100")

                with st.expander("📉 Distribuzione rendimenti & rischio"):
                    st.markdown("**Come leggere questi indicatori di rischio:** il **CVaR 95%** stima la perdita media nei giorni peggiori oltre la soglia del VaR; più è negativo, più la coda sinistra è pesante. **Skewness** negativa indica ribassi estremi più frequenti dei rialzi estremi, mentre **Kurtosis** elevata segnala distribuzioni con code più violente della normale. L'istogramma sotto ti aiuta a capire se i rendimenti sono compatti e regolari oppure instabili e pieni di eventi estremi.")
                    st.write(f"CVaR 95%: {risk['CVaR_95'] * 100:.2f}%")
                    st.write(f"Skewness: {risk['Skew']:.2f} | Kurtosis: {risk['Kurt']:.2f}")

                    returns = df_tech['Close'].pct_change().dropna()
                    fig_r = go.Figure()
                    fig_r.add_trace(go.Histogram(x=returns, nbinsx=50, name="Rendimenti giornalieri"))
                    fig_r.update_layout(template="plotly_dark", bargap=0.05)
                    st.plotly_chart(fig_r, use_container_width=True)

                with st.expander("🎲 Simulazione Monte Carlo (rendimenti storici)"):
                    st.markdown("**Come leggere la simulazione:** il modello estrae molte sequenze possibili di rendimenti sulla base della storia recente del titolo e costruisce traiettorie alternative di equity. La **mediana** rappresenta lo scenario centrale, il **5° percentile** mostra uno scenario prudente e il **95° percentile** uno scenario molto favorevole. La probabilità di perdita oltre il 20% serve a visualizzare in modo immediato quanto può essere duro il percorso dell'investimento anche quando il caso base sembra buono.")
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
            st.markdown("<p style='text-align: center; color: gray;'>creato e sviluppato da Innovative Program </p>", unsafe_allow_html=True)

        # --- TAB VERDETTO ---
        with tab_v:
            st.info("💡 **Come leggere il verdetto:** Questa tab sintetizza tutte le analisi precedenti in una decisione operativa, ma va letta con metodo. Il programma usa tre livelli: **modello Classico** con criteri essenziali, **modello Evoluto** con controlli aggiuntivi su leva e marginalità, e **modello Personalizzabile** che applica le soglie impostate nella sidebar. In parallelo viene calcolato lo **Smart Quant Score**, che unisce **Fundamental Score**, **Technical Score** e **Quant/Risk Score** per dare una misura complessiva del vantaggio statistico del setup. Il senso corretto del verdetto è questo: BUY indica coerenza forte tra qualità, rischio e timing; HOLD segnala qualità parziale o timing ancora incompleto; SELL o NO TRADE indicano che il margine di sicurezza non è sufficiente secondo il modello selezionato.")
            
            df_tech = get_technical_data(ticker)
            qm = calculate_quant_metrics(df_tech, row["_raw_data"]) if df_tech is not None else {}
            risk = calculate_risk_metrics(df_tech) if df_tech is not None else {
                "Max Drawdown": 0.0,
                "CAGR": 0.0
            }

            if df_tech is not None:
                df_calc_v = calculate_technical_indicators(df_tech)
                score, _ = calculate_timing_score(df_calc_v, df_calc_v['Close'].iloc[-1])
            else:
                score = 0

            z_val = qm.get('Altman Z-Score', 0.0)
            z_safe = "-" in ticker or (isinstance(z_val, float) and z_val >= 1.8)

            fund_pts_classic = (
                (1 if row['ROIC'] >= ui['cfg']['roic'] / 100 else 0) +
                (1 if row['PEG Ratio'] is not None and row['PEG Ratio'] <= ui['cfg']['peg'] else 0)
            )

            # Rinomina la logica precedentemente chiamata "fund_pts" in "fund_pts_evoluto" per evitare il NameError
            fund_pts_evoluto = (
                (1 if row['ROIC'] >= ui['cfg']['roic'] / 100 else 0) +
                (1 if row['PEG Ratio'] is not None and row['PEG Ratio'] <= ui['cfg']['peg'] else 0) +
                (1 if row.get('Debt/Equity') is not None and row.get('Debt/Equity') <= 1.0 else 0) +
                (1 if row.get('FCF Margin') is not None and row.get('FCF Margin') >= 0.08 else 0) +
                (1 if row.get('Net Margin') is not None and row.get('Net Margin') >= 0.10 else 0)
            )

            # Definisci la logica mancante per "fund_pts_custom" usando le opzioni UI
            fund_pts_custom = (
                (1 if row['ROIC'] >= ui['cfg']['roic'] / 100 else 0) +
                (1 if row['PEG Ratio'] is not None and row['PEG Ratio'] <= ui['cfg']['peg'] else 0) +
                (1 if row.get('Debt/Equity') is not None and row.get('Debt/Equity') <= ui['cfg']['custom_max_de'] else 0) +
                (1 if row.get('FCF Margin') is not None and row.get('FCF Margin') >= ui['cfg']['custom_min_fcf_margin'] else 0) +
                (1 if row.get('Net Margin') is not None and row.get('Net Margin') >= ui['cfg']['custom_min_net_margin'] else 0)
            )

            mode = ui['cfg']['model_mode']

            if mode == "Entrambi":
                col_m1, col_m2 = st.columns(2)
                col_m1.metric("Punti Modello Classico", f"{fund_pts_classic}/2")
                col_m2.metric("Punti Modello Evoluto", f"{fund_pts_evoluto}/5")
                
            if mode == "Personalizzabile":
                st.caption(
                    "Personalizzabile → "
                    f"ROIC: {'✅' if row['ROIC'] >= ui['cfg']['roic'] / 100 else '❌'} | "
                    f"PEG: {'✅' if row['PEG Ratio'] is not None and row['PEG Ratio'] <= ui['cfg']['peg'] else '❌'} | "
                    f"D/E: {'✅' if row.get('Debt/Equity') is not None and row.get('Debt/Equity') <= ui['cfg']['custom_max_de'] else '❌'} | "
                    f"FCF Margin: {'✅' if row.get('FCF Margin') is not None and row.get('FCF Margin') >= ui['cfg']['custom_min_fcf_margin'] else '❌'} | "
                    f"Net Margin: {'✅' if row.get('Net Margin') is not None and row.get('Net Margin') >= ui['cfg']['custom_min_net_margin'] else '❌'}"
                )                
                
            if mode in ["Entrambi", "Evoluto"]:
                st.caption(
                    "Evoluto → "
                    f"ROIC: {'✅' if row['ROIC'] >= ui['cfg']['roic'] / 100 else '❌'} | "
                    f"PEG: {'✅' if row['PEG Ratio'] is not None and row['PEG Ratio'] <= ui['cfg']['peg'] else '❌'} | "
                    f"D/E: {'✅' if row.get('Debt/Equity') is not None and row.get('Debt/Equity') <= 1.0 else '❌'} | "
                    f"FCF Margin: {'✅' if row.get('FCF Margin') is not None and row.get('FCF Margin') >= 0.08 else '❌'} | "
                    f"Net Margin: {'✅' if row.get('Net Margin') is not None and row.get('Net Margin') >= 0.10 else '❌'}"
                )                

            if mode in ["Entrambi", "Classico"]:
                st.subheader("Modello Classico")

                if fund_pts_classic >= 2 and z_safe and score >= 50:
                    st.success("🟢 BUY: Fondamentali base solidi e timing favorevole.")
                elif fund_pts_classic >= 1 and z_safe:
                    st.warning("🟡 HOLD: Azienda discreta, ma serve più margine di sicurezza.")
                else:
                    st.error("🔴 SELL: Fondamentali o sicurezza finanziaria insufficienti.")

            if mode in ["Entrambi", "Evoluto"]:
                st.subheader("Modello Evoluto")

                if fund_pts_evoluto >= 4 and z_safe and score >= 50:
                    st.success("🟢 BUY: Fondamentali robusti, buona qualità finanziaria e timing favorevole.")
                elif fund_pts_evoluto >= 3 and z_safe:
                    st.warning("🟡 HOLD: Azienda interessante, ma serve più conferma su prezzo o momentum.")
                else:
                    st.error("🔴 SELL: Fondamentali insufficienti o profilo rischio/rendimento debole.")
                    
            if mode == "Personalizzabile":
                st.subheader("Modello Personalizzabile")

                if fund_pts_custom >= 4 and z_safe and score >= 50:
                    st.success("🟢 BUY: Fondamentali coerenti con i parametri personalizzati e timing favorevole.")
                elif fund_pts_custom >= 3 and z_safe:
                    st.warning("🟡 HOLD: Setup discreto secondo i parametri personalizzati, ma non ancora abbastanza forte.")
                else:
                    st.error("🔴 SELL: Il titolo non soddisfa i criteri del modello personalizzato.")                    

            if df_tech is not None and qm:
                smart_v = compute_smart_quant_score(row, score, qm, risk)

                if mode in ["Entrambi", "Evoluto"]:
                    st.metric("Smart Quant Score", f"{smart_v['SmartScore']:.1f}/100")
                    st.write(
                        f"F: {smart_v['FundamentalScore']:.0f} | "
                        f"T: {smart_v['TechnicalScore']:.0f} | "
                        f"Q: {smart_v['QuantRiskScore']:.0f}"
                    )

                    if smart_v["SmartScore"] >= 70 and z_safe and fund_pts_evoluto >= 4:
                        st.success("🟢 BUY (Quant): Vantaggio statistico, fondamentali solidi e rischio controllato.")
                    elif smart_v["SmartScore"] >= 50 and z_safe and fund_pts_evoluto >= 3:
                        st.warning("🟡 HOLD (Quant): Setup discreto, ma non ancora abbastanza forte per un BUY pieno.")
                    else:
                        st.error("🔴 NO TRADE (Quant): Vantaggio quantitativo debole, fondamentali insufficienti o rischio elevato.")
                        
                if mode == "Personalizzabile":
                    st.metric("Smart Quant Score", f"{smart_v['SmartScore']:.1f}/100")
                    st.write(
                        f"F: {smart_v['FundamentalScore']:.0f} | "
                        f"T: {smart_v['TechnicalScore']:.0f} | "
                        f"Q: {smart_v['QuantRiskScore']:.0f}"
                    )

                    if smart_v["SmartScore"] >= 70 and z_safe and fund_pts_custom >= 4:
                        st.success("🟢 BUY (Quant): Vantaggio statistico e criteri personalizzati soddisfatti.")
                    elif smart_v["SmartScore"] >= 50 and z_safe and fund_pts_custom >= 3:
                        st.warning("🟡 HOLD (Quant): Setup discreto secondo il modello personalizzato.")
                    else:
                        st.error("🔴 NO TRADE (Quant): Score quantitativo o criteri personalizzati insufficienti.")                        
            
            st.markdown("---")
            st.markdown("<p style='text-align: center; color: gray;'>creato e sviluppato da Innovative Program </p>", unsafe_allow_html=True)

        # --- TAB PORTAFOGLIO ---
        with tab_p:
            st.info("💡 **Come usare questa sezione:** Qui costruisci il tuo **portafoglio reale** partendo dalle posizioni effettivamente detenute. Per ogni titolo inserisci **ticker**, **quantità/quote**, **PMC** e **valuta**; il sistema calcola automaticamente **importo investito**, **prezzo attuale**, **valore di mercato**, **P&L in euro o valuta locale**, **P&L %** e **peso percentuale**. Le nuove aggiunte più importanti sono la gestione di **quantità frazionate**, la **conversione FX reale** verso la valuta base del portafoglio, l'**analisi fiscale teorica**, la classificazione per **asset class**, **geografia** e **valuta**, oltre al **ribilanciamento automatico** rispetto ai target impostati. Questa tab va letta come una cabina di controllo: prima ricostruisci correttamente le posizioni, poi controlli concentrazione, rischio valutario, fiscalità teorica e scostamenti dai pesi obiettivo.")
            st.success("Versione Premium attiva: conversione FX reale integrata nei totali portafoglio.")
            st.markdown("""
### Portafoglio reale
- Qui inserisci le **posizioni effettive** che hai in portafoglio, non semplici pesi teorici.
- Per ogni titolo indica **ticker**, **quantità/quote**, **PMC** e **valuta della posizione**.
- Le **quantità frazionate** sono supportate e sono particolarmente utili per ETF, PAC e broker che consentono acquisti parziali.
- Il sistema ricostruisce in automatico **capitale investito**, **valore di mercato**, **profitto/perdita**, **rendimento %** e **peso della posizione** sul totale.
- Se la valuta del titolo è diversa dalla valuta base, entra in gioco la **conversione FX reale**, così i totali portafoglio diventano confrontabili in modo corretto.
- L'analisi successiva mostra anche **impatto fiscale teorico**, **allocazione per asset class**, **allocazione geografica**, **allocazione per valuta** e **suggerimenti di ribilanciamento**.

**Come cercare il ticker giusto:**
- Azioni USA: normalmente solo ticker (es. `AAPL`, `MSFT`).
- Azioni italiane: aggiungi `.MI` (es. `STLAM.MI`, `ENI.MI`, `ISP.MI`).
- Azioni tedesche: aggiungi `.DE` (es. `BMW.DE`, `SAP.DE`).
- Azioni francesi: aggiungi `.PA` (es. `AIR.PA`, `OR.PA`).
- Azioni UK: aggiungi `.L` (es. `ULVR.L`).
- Crypto: in genere coppia con valuta, es. `BTC-USD`, `ETH-USD`.
- Inserisci sempre il ticker esatto di Yahoo Finance, perché tutti i calcoli di prezzo, resa e conversione dipendono da quello.
""")
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
                    t_clean = sanitize_ticker(manual_ticker)
                    if t_clean not in st.session_state.portfolio_tickers:
                        st.session_state.portfolio_tickers.append(t_clean)
                        st.success(f"Aggiunto {t_clean} al portafoglio.")
                else:
                    st.warning("Inserisci un ticker valido prima di aggiungere.")

            portfolio_list = sorted(set(selected_from_batch + st.session_state.portfolio_tickers))
            st.session_state.portfolio_tickers = portfolio_list

            if portfolio_list:
                st.markdown("#### Dati posizione per ogni ticker")
                cols = st.columns(2)
                holdings: Dict[str, float] = st.session_state.holdings
                holdings_quantity: Dict[str, float] = st.session_state.holdings_quantity
                holdings_pmc: Dict[str, float] = st.session_state.holdings_pmc

                for i, t in enumerate(portfolio_list):
                    col = cols[i % 2]
                    col.markdown(f"##### {t}")
                    default_qty = float(holdings_quantity.get(t, 0.0))
                    default_pmc = float(holdings_pmc.get(t, 0.0))
                    qty = col.number_input(
                        f"{t} - Quantità / Quote",
                        min_value=0.0,
                        value=default_qty,
                        step=0.01,
                        format="%.4f",
                        key=f"holding_qty_{t}"
                    )
                    pmc = col.number_input(
                        f"{t} - PMC",
                        min_value=0.0,
                        value=default_pmc,
                        step=0.01,
                        format="%.4f",
                        key=f"holding_pmc_{t}"
                    )
                    derived = calculate_position_from_quantity(t, qty, pmc) if qty > 0 and pmc > 0 else {
                        'Importo Investito': 0.0,
                        'Prezzo Attuale': np.nan,
                        'Valore di Mercato': 0.0,
                        'P&L': 0.0,
                        'P&L %': 0.0,
                    }
                    holdings_quantity[t] = qty
                    holdings_pmc[t] = pmc
                    holdings[t] = float(derived['Importo Investito'])

                    cur_default = st.session_state.holdings_currency.get(t, "USD")
                    cur = col.selectbox(
                        f"{t} - Valuta",
                        ["USD", "EUR"],
                        index=["USD", "EUR"].index(cur_default),
                        key=f"currency_{t}"
                    )
                    st.session_state.holdings_currency[t] = cur

                    price_text = "N/D" if pd.isna(derived['Prezzo Attuale']) else f"{derived['Prezzo Attuale']:.2f}"
                    col.caption(
                        f"Prezzo attuale: {price_text} | Investito: {derived['Importo Investito']:.2f} | "
                        f"Valore: {derived['Valore di Mercato']:.2f} | P&L: {derived['P&L']:.2f} ({derived['P&L %']:.2f}%)"
                    )

                    if col.button("🗑 Rimuovi", key=f"remove_{t}"):
                        if t in st.session_state.portfolio_tickers:
                            st.session_state.portfolio_tickers = [x for x in st.session_state.portfolio_tickers if x != t]
                        if t in st.session_state.holdings:
                            del st.session_state.holdings[t]
                        if t in st.session_state.holdings_currency:
                            del st.session_state.holdings_currency[t]
                        if t in st.session_state.holdings_quantity:
                            del st.session_state.holdings_quantity[t]
                        if t in st.session_state.holdings_pmc:
                            del st.session_state.holdings_pmc[t]
                        st.rerun()

                st.session_state.holdings = holdings
                st.session_state.holdings_quantity = holdings_quantity
                st.session_state.holdings_pmc = holdings_pmc

                if st.button("📊 Calcola pesi e analisi del portafoglio"):
                    positive_holdings = {t: a for t, a in holdings.items() if a > 0}
                    if not positive_holdings:
                        st.error("Imposta quantità e PMC > 0 almeno per un titolo.")
                    else:
                        tot = sum(positive_holdings.values())
                        weights_pct = {t: a / tot * 100.0 for t, a in positive_holdings.items()}
                        built = build_portfolio_returns(list(positive_holdings.keys()), weights_pct)
                        if built is None:
                            st.error("Impossibile costruire la serie dei rendimenti (dati insufficienti per uno o più ticker).")
                        else:
                            df_rets, port_ret = built
                            pm = calculate_portfolio_metrics(port_ret)
                            st.markdown("#### Dettaglio posizioni e pesi del portafoglio")
                            rows = []
                            total_market_value = 0.0
                            for t in weights_pct.keys():
                                qty = holdings_quantity.get(t, 0.0)
                                pmc = holdings_pmc.get(t, 0.0)
                                derived = calculate_position_from_quantity(t, qty, pmc)
                                total_market_value += derived['Valore di Mercato']
                                rows.append({
                                    "Ticker": t,
                                    "Quantità": qty,
                                    "PMC": pmc,
                                    "Prezzo Attuale": derived['Prezzo Attuale'],
                                    "Importo Investito": derived['Importo Investito'],
                                    "Valore di Mercato": derived['Valore di Mercato'],
                                    "P&L": derived['P&L'],
                                    "P&L %": derived['P&L %'],
                                    "Peso %": weights_pct[t],
                                    "Valuta": st.session_state.holdings_currency.get(t, "USD")
                                })
                            df_weights = pd.DataFrame(rows)
                            st.dataframe(df_weights)
                            st.markdown("#### Analisi fiscale teorica")
                            st.caption("Questa sezione stima l'effetto fiscale potenziale sulle plusvalenze utilizzando l'aliquota selezionata. È una simulazione teorica utile per capire quanto del profitto lordo resterebbe dopo le imposte; non sostituisce il calcolo fiscale reale del broker o del commercialista.")
                            tax_rate_input = st.slider(
                                "Aliquota fiscale teorica (%)",
                                min_value=0.0,
                                max_value=50.0,
                                value=float(DEFAULT_TAX_RATE * 100.0),
                                step=1.0,
                                key="portfolio_tax_rate_slider"
                            ) / 100.0

                            df_tax = calculate_tax_impact(df_weights, tax_rate=tax_rate_input)

                            if not df_tax.empty:
                                st.dataframe(df_tax, use_container_width=True)

                                total_tax = float(df_tax["Imposta Teorica"].sum())
                                total_net_pnl = float(df_tax["Plus/Minus Netta"].sum())
                                total_net_value = float(df_tax["Valore Netto Post Imposta"].sum())

                                tax_c1, tax_c2, tax_c3 = st.columns(3)
                                tax_c1.metric("Imposta teorica totale", f"{total_tax:,.2f}")
                                tax_c2.metric("P&L netto teorico", f"{total_net_pnl:,.2f}")
                                tax_c3.metric("Valore netto post imposta", f"{total_net_value:,.2f}")

                                st.caption("Stima teorica fiscale: non sostituisce il calcolo fiscale ufficiale del broker o del commercialista.")                            
                            st.markdown("#### Allocazione del portafoglio")
                            df_alloc = build_portfolio_allocation_df(
                                positive_holdings,
                                st.session_state.holdings_currency
                            )

                            if not df_alloc.empty:
                                st.dataframe(df_alloc, use_container_width=True)

                                col_a1, col_a2, col_a3 = st.columns(3)

                                with col_a1:
                                    st.markdown("**Per Asset Class**")
                                    df_asset = summarize_group_weights(df_alloc, "Asset Class")
                                    st.dataframe(df_asset, use_container_width=True)

                                with col_a2:
                                    st.markdown("**Per Geografia**")
                                    df_geo = summarize_group_weights(df_alloc, "Geografia")
                                    st.dataframe(df_geo, use_container_width=True)

                                with col_a3:
                                    st.markdown("**Per Valuta**")
                                    df_cur = summarize_group_weights(df_alloc, "Valuta")
                                    st.dataframe(df_cur, use_container_width=True)

                                st.markdown("#### Ribilanciamento automatico")
                                st.caption("Il ribilanciamento confronta i pesi attuali con i target definiti dall'utente e calcola lo scostamento. Se la differenza supera la tolleranza impostata, il sistema suggerisce in quali aree comprare o ridurre per riportare il portafoglio verso la struttura desiderata.")
                                rebalance_mode = st.radio(
                                    "Livello target",
                                    ["Ticker", "Asset Class", "Geografia", "Valuta"],
                                    horizontal=True,
                                    key="rebalance_mode_radio"
                                )
                                st.session_state.portfolio_target_mode = rebalance_mode

                                if rebalance_mode == "Ticker":
                                    current_target_df = df_alloc[["Ticker", "Peso %"]].copy()
                                    label_col = "Ticker"
                                elif rebalance_mode == "Asset Class":
                                    current_target_df = df_asset[["Asset Class", "Peso %"]].copy()
                                    label_col = "Asset Class"
                                elif rebalance_mode == "Geografia":
                                    current_target_df = df_geo[["Geografia", "Peso %"]].copy()
                                    label_col = "Geografia"
                                else:
                                    current_target_df = df_cur[["Valuta", "Peso %"]].copy()
                                    label_col = "Valuta"

                                st.caption("Inserisci i target percentuali desiderati. Se la somma non fa 100%, il sistema la normalizza automaticamente.")

                                target_inputs = {}
                                cols_target = st.columns(3)

                                for i, rec in enumerate(current_target_df.to_dict("records")):
                                    col = cols_target[i % 3]
                                    label = rec[label_col]
                                    current_weight = float(rec["Peso %"])
                                    default_target = float(
                                        st.session_state.portfolio_targets.get(
                                            f"{rebalance_mode}::{label}",
                                            current_weight
                                        )
                                    )

                                    target_val = col.number_input(
                                        f"Target {label}",
                                        min_value=0.0,
                                        max_value=100.0,
                                        value=default_target,
                                        step=1.0,
                                        key=f"target_{rebalance_mode}_{label}"
                                    )

                                    target_inputs[label] = target_val
                                    st.session_state.portfolio_targets[f"{rebalance_mode}::{label}"] = target_val

                                tolerance_pct = st.slider(
                                    "Tolleranza ribilanciamento (%)",
                                    min_value=0.0,
                                    max_value=10.0,
                                    value=1.0,
                                    step=0.5
                                )

                                target_sum = sum(target_inputs.values())
                                if target_sum > 0:
                                    normalized_targets = {
                                        k: v / target_sum * 100.0 for k, v in target_inputs.items()
                                    }
                                else:
                                    normalized_targets = target_inputs

                                rebalance_df = compute_rebalancing_actions(
                                    df_alloc=df_alloc,
                                    target_weights=normalized_targets,
                                    group_col=label_col,
                                    tolerance_pct=tolerance_pct
                                )

                                if not rebalance_df.empty:
                                    st.markdown("##### Azioni suggerite per tornare in target")
                                    st.dataframe(rebalance_df, use_container_width=True)

                                    fig_reb = go.Figure()
                                    fig_reb.add_trace(go.Bar(
                                        x=rebalance_df[label_col],
                                        y=rebalance_df["Peso %"],
                                        name="Peso attuale %"
                                    ))
                                    fig_reb.add_trace(go.Bar(
                                        x=rebalance_df[label_col],
                                        y=rebalance_df["Target %"],
                                        name="Target %"
                                    ))
                                    fig_reb.update_layout(
                                        barmode="group",
                                        template="plotly_dark",
                                        height=420,
                                        xaxis_title=label_col,
                                        yaxis_title="Peso %"
                                    )
                                    st.plotly_chart(fig_reb, use_container_width=True)
                            total_invested = float(df_weights["Importo Investito"].sum())
                            total_pnl = float(df_weights["P&L"].sum())
                            total_pnl_pct = (total_pnl / total_invested) * 100.0 if total_invested > 0 else 0.0

                            ctot1, ctot2, ctot3, ctot4 = st.columns(4)
                            ctot1.metric("Investito totale", f"{total_invested:,.2f}")
                            ctot2.metric("Valore attuale", f"{total_market_value:,.2f}")
                            ctot3.metric("P&L totale", f"{total_pnl:,.2f}")
                            ctot4.metric("Rendimento totale", f"{total_pnl_pct:.2f}%")

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
            st.markdown("\ncreato e sviluppato da Innovative Program \n\n", unsafe_allow_html=True)

if __name__ == "__main__":
    main()


# ==============================
# IMPLEMENTAZIONE AGGIORNATA
# - Modalità ospite attiva senza login obbligatorio
# - Portfolio persistente per account tramite Supabase
# - SQL completo incluso nella costante SUPABASE_PORTFOLIO_SQL
# - Sidebar account con sezione Il mio portafoglio
# - Le tab restano accessibili anche senza analisi preventiva
# ==============================
