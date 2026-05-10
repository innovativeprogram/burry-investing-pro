"""
# Copyright (c) 2026 InnovativeProgram
# Tutti i diritti riservati. 
# Proprietà intellettuale di [Canio Tedesco].
# La copia o distribuzione non autorizzata è severamente vietata.

"""

import streamlit as st
import yfinance as yf
import requests
from yahooquery import Ticker as YQ_Ticker
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
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List
from sklearn.linear_model import LinearRegression
from supabase import create_client, Client
from burry_ai_prompts import (
    build_ai_context_for_ticker,
    ask_gemini_ticker_chat,
    build_burry_ai_context,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)


# ==========================================================================
# 0. SETUP LOGGING & COSTANTI GLOBALI
# ==========================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("BurryInvestingPro")

# [REFACTOR] Costanti finanziarie raggruppate e documentate
DEFAULT_TAX_RATE = 0.26                 # Aliquota italiana plusvalenze finanziarie
SAFE_INTEREST_COVERAGE = 100.0          # Sentinel per asset senza debito
TRADING_DAYS_YEAR = 252                 # Giorni di borsa in 1 anno
MAX_CSV_ROWS = 100                      # Massimo numero di ticker per batch CSV
MAX_WORKERS = 3                         # [BUGFIX] Ora effettivamente usato
DEFAULT_RISK_FREE_RATE = 0.04           # Fallback se non fetchabile
FX_TTL_SECONDS = 3600
RISK_FREE_TTL_SECONDS = 6 * 3600
ALTMAN_SAFE_THRESHOLD = 1.81            # [FIN-FIX] valore canonico (era 1.8)

# [NEW] Costanti per benchmark e scoring configurabile
DEFAULT_BENCHMARK = "^GSPC"             # S&P 500 di default
DEFAULT_SMART_WEIGHTS = {"F": 0.40, "T": 0.30, "Q": 0.30}

# [NEW] Compensazione fiscale italiana: minusvalenze utilizzabili 4 anni
TAX_LOSS_COMPENSATION_YEARS = 4


# ==========================================================================
# 0.A SAFE SECRETS / CONFIG ACCESS  [SECURITY]
# ==========================================================================
def safe_get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    [SECURITY] Recupero sicuro di un secret. Gestisce il caso in cui
    `st.secrets` non sia configurato (StreamlitSecretNotFoundError) senza
    far crollare l'applicazione e senza esporre fallback hardcoded.
    """
    # Priorita': variabile d'ambiente -> st.secrets -> default
    env_val = os.getenv(key)
    if env_val:
        return env_val.strip()
    try:
        if key in st.secrets:
            val = st.secrets[key]
            return str(val).strip() if val is not None else default
    except Exception:
        # secrets.toml mancante in dev: ignoriamo silenziosamente
        pass
    return default


# [SECURITY] BUGFIX: rimosso fallback hardcoded della Polygon API key.
# Se mancante, il provider Polygon viene semplicemente saltato.
POLYGON_API_KEY = safe_get_secret("POLYGON_API_KEY", default=None)


# ==========================================================================
# 0.B PRICE / FX / RISK-FREE PROVIDERS
# ==========================================================================
@st.cache_data(ttl=900, show_spinner=False)
def get_current_price_safe(ticker_symbol: str) -> float:
    """
    Recupero prezzo con failover: Polygon -> YahooQuery -> yFinance.
    [BUGFIX] eccezioni catturate in modo specifico, no bare except.
    [BUGFIX] rimosso re-import di YQ_Ticker (gia' importato a livello modulo).
    """
    symbol = (ticker_symbol or "").upper().strip()
    if not symbol:
        return 0.0

    # Tentativo 1: Polygon (solo se key presente e ticker non ha suffisso)
    if POLYGON_API_KEY and "." not in symbol and "-" not in symbol:
        try:
            url = f"https://api.polygon.io/v2/last/trade/{symbol}?apiKey={POLYGON_API_KEY}"
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "OK" and "results" in data:
                    p = data["results"].get("p")
                    if p is not None:
                        return float(p)
        except (requests.RequestException, ValueError, KeyError) as e:
            logger.debug(f"Polygon price fallback for {symbol}: {e}")

    # Tentativo 2: YahooQuery
    try:
        yq = YQ_Ticker(symbol)
        price_data = yq.price.get(symbol, {}) if isinstance(yq.price, dict) else {}
        if isinstance(price_data, dict):
            p = price_data.get('regularMarketPrice') or price_data.get('preMarketPrice')
            if p is not None:
                return float(p)
    except Exception as e:
        logger.debug(f"YahooQuery price fallback for {symbol}: {e}")

    # Tentativo 3: yfinance
    try:
        t = yf.Ticker(symbol)
        return float(t.fast_info['last_price'])
    except Exception as e:
        logger.debug(f"yfinance price fallback for {symbol}: {e}")
        return 0.0


@st.cache_data(ttl=FX_TTL_SECONDS, show_spinner=False)
def get_fx_rate(from_currency: str, to_currency: str) -> float:
    """
    Cambio FX con fallback: pair diretto -> pair inverso -> 1.0 (neutro).
    Logica originaria preservata, solo error handling rinforzato.
    """
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


@st.cache_data(ttl=RISK_FREE_TTL_SECONDS, show_spinner=False)
def get_dynamic_risk_free_rate() -> float:
    """
    [NEW] Recupera dinamicamente il tasso risk-free dal T-Bill 13-settimane (^IRX).
    Se non disponibile, fallback al DEFAULT_RISK_FREE_RATE.
    Nota: ^IRX riporta yield in punti percentuali (es. 4.50 = 4.50%).
    """
    try:
        irx = yf.Ticker("^IRX").history(period="5d")
        if irx is not None and not irx.empty and 'Close' in irx.columns:
            last = irx['Close'].dropna()
            if not last.empty:
                rf_pct = float(last.iloc[-1])
                # Sanity check: rendimento normalmente tra 0% e 15%
                if 0 <= rf_pct <= 15:
                    return rf_pct / 100.0
    except Exception as e:
        logger.info(f"Risk-free dinamico non disponibile, uso default: {e}")
    return DEFAULT_RISK_FREE_RATE


def get_active_risk_free_rate() -> float:
    """[NEW] Wrapper che permette override via session state."""
    try:
        override = st.session_state.get("risk_free_override")
        if override is not None:
            return float(override)
    except Exception:
        pass
    return get_dynamic_risk_free_rate()


# ==========================================================================
# 0.C PORTFOLIO FX & TAX (originali integrate)
# ==========================================================================
def enrich_portfolio_with_fx(
    df_weights: pd.DataFrame,
    base_currency: str = "EUR"
) -> pd.DataFrame:
    """
    [BUGFIX] Funzione gia' presente nell'originale ma mai chiamata.
    Ora effettivamente integrata nel flusso del tab Portafoglio.
    Converte importi/valori/PnL nella valuta base e ricalcola i pesi
    sul valore di mercato in base.
    """
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
    out["Peso Base %"] = np.where(
        total_base_mv > 0,
        out["Valore di Mercato Base"] / total_base_mv * 100.0,
        0.0
    )
    return out


def calculate_tax_impact_df_base(
    df_weights_base: pd.DataFrame,
    tax_rate: float = DEFAULT_TAX_RATE
) -> pd.DataFrame:
    """
    [BUGFIX] Anche questa era definita ma mai utilizzata.
    Ora effettivamente integrata: calcola fiscalita' sul P&L gia'
    convertito in valuta base.
    """
    if df_weights_base is None or df_weights_base.empty:
        return pd.DataFrame()
    df_tax = df_weights_base.copy()
    df_tax["Aliquota Fiscale %"] = tax_rate * 100.0
    df_tax["Plus/Minus Lorda Base"] = df_tax["P&L Base"].astype(float)
    df_tax["Imposta Teorica Base"] = np.where(
        df_tax["Plus/Minus Lorda Base"] > 0,
        df_tax["Plus/Minus Lorda Base"] * tax_rate,
        0.0
    )
    df_tax["Plus/Minus Netta Base"] = df_tax["Plus/Minus Lorda Base"] - df_tax["Imposta Teorica Base"]
    df_tax["Valore Netto Post Imposta Base"] = df_tax["Valore di Mercato Base"] - df_tax["Imposta Teorica Base"]
    df_tax["Rendimento Netto Base %"] = np.where(
        df_tax["Importo Investito Base"] > 0,
        df_tax["Plus/Minus Netta Base"] / df_tax["Importo Investito Base"] * 100.0,
        0.0
    )
    return df_tax


def calculate_tax_with_loss_offset(
    df_weights_base: pd.DataFrame,
    tax_rate: float = DEFAULT_TAX_RATE
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    [NEW][FIN-FIX] Compensazione fiscale italiana: le minusvalenze
    realizzate possono essere utilizzate per ridurre l'imponibile sulle
    plusvalenze entro 4 anni. Qui simuliamo la compensazione *teorica*
    se tutte le posizioni venissero chiuse oggi.
    """
    if df_weights_base is None or df_weights_base.empty:
        return pd.DataFrame(), {}
    df = df_weights_base.copy()
    pl_lorda = df["P&L Base"].astype(float)
    gains = pl_lorda.clip(lower=0).sum()
    losses = (-pl_lorda.clip(upper=0)).sum()  # valore positivo
    compensable = min(gains, losses)
    taxable_base = max(0.0, gains - compensable)
    theoretical_tax = taxable_base * tax_rate
    df["Aliquota Fiscale %"] = tax_rate * 100.0
    df["Plus/Minus Lorda Base"] = pl_lorda
    df["Imposta Teorica Base (compensata)"] = np.where(
        pl_lorda > 0,
        # Distribuiamo l'imposta proporzionalmente sulle posizioni in gain
        np.where(gains > 0, pl_lorda / gains * theoretical_tax, 0.0),
        0.0
    )
    summary = {
        "Plusvalenze totali": float(gains),
        "Minusvalenze totali": float(losses),
        "Minusvalenze compensate": float(compensable),
        "Imponibile residuo": float(taxable_base),
        "Imposta teorica netta": float(theoretical_tax),
        "Risparmio fiscale da compensazione": float(compensable * tax_rate),
    }
    return df, summary


# ==========================================================================
# CONFIGURAZIONE PAGINA UI
# ==========================================================================
st.set_page_config(
    page_title="V-Quant Pro",
    page_icon="💲",
    layout="wide"
)


# ==========================================================================
# 0.D AUTH SUPABASE  [SECURITY]
# ==========================================================================
# [SECURITY] BUGFIX: i fallback hardcoded di SUPABASE_PROJECT_REF e
# SUPABASE_ANON_KEY sono stati rimossi. Le credenziali vanno fornite
# tramite st.secrets o variabili d'ambiente.

def get_app_base_url() -> str:
    candidates = [
        os.getenv('APP_BASE_URL'),
        os.getenv('STREAMLIT_APP_URL'),
        os.getenv('PUBLIC_APP_URL')
    ]
    for c in candidates:
        if c and str(c).strip().startswith(('http://', 'https://')):
            return str(c).strip().rstrip('/')
    return 'http://localhost:8501'


def get_email_redirect_url() -> str:
    return get_app_base_url()


def get_supabase_credentials() -> Tuple[Optional[str], Optional[str]]:
    """[SECURITY] Restituisce (None, None) se le credenziali non sono configurate."""
    url = safe_get_secret('SUPABASE_URL')
    key = safe_get_secret('SUPABASE_ANON_KEY')
    return url, key


@st.cache_resource(show_spinner=False)
def get_supabase_client() -> Optional[Client]:
    """[SECURITY] Restituisce None se le credenziali non sono configurate."""
    url, key = get_supabase_credentials()
    if not url or not key:
        logger.warning("Supabase non configurato: imposta SUPABASE_URL e SUPABASE_ANON_KEY.")
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


def get_logged_user_id() -> Optional[str]:
    user = st.session_state.get('auth_user')
    if not user:
        return None
    if isinstance(user, dict):
        return user.get('id')
    return getattr(user, 'id', None)


def is_supabase_available() -> bool:
    """[NEW] True solo se Supabase e' configurato e raggiungibile."""
    return get_supabase_client() is not None


def load_user_portfolio() -> None:
    user_id = get_logged_user_id()
    if not user_id:
        return
    supabase = get_supabase_client()
    if supabase is None:
        st.warning("Supabase non configurato: impossibile caricare il portafoglio salvato.")
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


def save_user_portfolio_position(
    ticker: str, quantity: float, pmc: float, currency: str
) -> None:
    user_id = get_logged_user_id()
    if not user_id:
        return
    supabase = get_supabase_client()
    if supabase is None:
        st.error("Supabase non configurato: impossibile salvare il portafoglio.")
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
        supabase.table('portfoliopositions').delete().eq(
            'userid', user_id
        ).eq('ticker', str(ticker).upper().strip()).execute()
    except Exception as e:
        logger.warning(f'Delete portfolio skipped for {ticker}: {e}')


def _extract_auth_payload(auth_response: Any) -> Tuple[Any, Any]:
    user = getattr(auth_response, 'user', None)
    session = getattr(auth_response, 'session', None)
    return user, session


# [NEW] Validazione email con regex di base
EMAIL_REGEX = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")


def validate_email(email: str) -> bool:
    return bool(EMAIL_REGEX.match((email or "").strip()))


def validate_password_strength(password: str) -> Tuple[bool, str]:
    """[NEW][SECURITY] Verifica robustezza minima password."""
    if len(password) < 8:
        return False, "La password deve avere almeno 8 caratteri."
    if not re.search(r"[A-Z]", password):
        return False, "La password deve contenere almeno una maiuscola."
    if not re.search(r"[a-z]", password):
        return False, "La password deve contenere almeno una minuscola."
    if not re.search(r"\d", password):
        return False, "La password deve contenere almeno un numero."
    return True, "OK"


def sign_up_with_supabase(email: str, password: str) -> Tuple[bool, str]:
    supabase = get_supabase_client()
    if supabase is None:
        return False, "Supabase non configurato. Imposta SUPABASE_URL e SUPABASE_ANON_KEY."
    try:
        response = supabase.auth.sign_up({
            'email': email.strip(),
            'password': password,
            'options': {'email_redirect_to': get_email_redirect_url()}
        })
        user, session = _extract_auth_payload(response)
        if user is None:
            return False, 'Registrazione non completata. Controlla le restrizioni del progetto Supabase.'
        st.session_state.auth_user = user
        st.session_state.auth_session = session
        if session is None:
            return True, "Registrazione eseguita. Controlla la tua email per confermare l'account."
        return True, 'Registrazione completata con accesso effettuato.'
    except Exception as e:
        return False, f'Registrazione fallita: {e}'


def sign_in_with_supabase(email: str, password: str) -> Tuple[bool, str]:
    supabase = get_supabase_client()
    if supabase is None:
        return False, "Supabase non configurato."
    try:
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
        return True, 'Logout eseguito con successo.'
    except Exception as e:
        return False, f'Logout fallito: {e}'


def render_auth_sidebar() -> None:
    st.sidebar.markdown('### 👤 Account')
    if not is_supabase_available():
        st.sidebar.info(
            "Auth disabilitata: configura SUPABASE_URL e SUPABASE_ANON_KEY per "
            "abilitare login e salvataggio portafoglio."
        )
        st.sidebar.markdown('---')
        return

    current_email = get_logged_user_email()
    if current_email:
        st.sidebar.success(f'Connesso come: {current_email}')

        with st.sidebar.expander('📁 Il mio portafoglio', expanded=False):
            n_pos = len(st.session_state.get('portfolio_tickers', []))
            st.write(f'Titoli in sessione: {n_pos}')

            if st.button('Carica portafoglio salvato', width='stretch', key='load_saved_portfolio_btn'):
                load_user_portfolio()
                st.session_state.portfolio_loaded_from_db = True
                st.rerun()

            if st.button('Salva portafoglio attuale', width='stretch', key='save_current_portfolio_btn'):
                for t in st.session_state.get('portfolio_tickers', []):
                    qty = float(st.session_state.get('holdings_quantity', {}).get(t, 0.0))
                    pmc = float(st.session_state.get('holdings_pmc', {}).get(t, 0.0))
                    cur = st.session_state.get('holdings_currency', {}).get(t, 'USD')
                    save_user_portfolio_position(t, qty, pmc, cur)
                st.sidebar.success('Portafoglio salvato correttamente.')

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
            if not validate_email(email):
                st.sidebar.error('Inserisci una email valida.')
            else:
                ok_pwd, msg_pwd = validate_password_strength(password)
                if not ok_pwd:
                    st.sidebar.error(msg_pwd)
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
# 1. MODELLI DATI
# ==========================================================================
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


# ==========================================================================
# 2. HELPER & VALIDAZIONE
# ==========================================================================
def sanitize_ticker(ticker: str) -> str:
    clean = str(ticker or '').strip().upper()
    if not clean:
        raise ValueError('Ticker vuoto')
    if not re.match(r'^[A-Z0-9\-\.=^]+$', clean):
        raise ValueError(f'Ticker contiene caratteri non validi: {clean}')
    return clean


def normalize_ticker(ticker: str, suffix: str) -> str:
    clean_ticker = sanitize_ticker(ticker)
    if "-" in clean_ticker or clean_ticker.startswith("^"):
        return clean_ticker
    clean_suffix = str(suffix).strip().upper()
    if clean_suffix and not re.match(r"^\.[A-Z]+$", clean_suffix):
        clean_suffix = ""
    if clean_suffix and not clean_ticker.endswith(clean_suffix):
        return f"{clean_ticker}{clean_suffix}"
    return clean_ticker


def safe_float(value: Any, default: float = np.nan) -> float:
    """[NEW] Conversione sicura a float."""
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def is_non_traditional_asset(ticker: str, raw_info: Optional[Dict[str, Any]] = None) -> bool:
    """
    [NEW] Identifica asset non tradizionali (crypto, FX, commodity, indici)
    per cui i criteri fondamentali (Z-Score, ROIC, etc.) non sono applicabili.
    """
    t = (ticker or "").upper()
    if t.startswith("^") or "=X" in t or t.endswith("=F"):
        return True
    if "-USD" in t or "-EUR" in t or "-GBP" in t:
        return True
    if raw_info:
        qt = str(raw_info.get('quoteType', '')).upper()
        if qt in {"CRYPTOCURRENCY", "CURRENCY", "FUTURE", "INDEX", "ETF", "MUTUALFUND"}:
            return True
    return False


# ==========================================================================
# 3. DATA ENGINE: ANALISI FONDAMENTALE CON FALLBACK
# ==========================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def get_fundamental_data(symbol: str) -> Optional[Dict[str, Any]]:
    # TENTATIVO 1: yfinance
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        if info and ('symbol' in info or 'shortName' in info):
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
        logger.info(f"yfinance fallito per {symbol}: {e}. Passo a YahooQuery.")

    # TENTATIVO 2: YahooQuery
    try:
        yq = YQ_Ticker(symbol)
        summary = yq.summary_detail.get(symbol, {}) if isinstance(yq.summary_detail, dict) else {}
        price = yq.price.get(symbol, {}) if isinstance(yq.price, dict) else {}
        financial_data = yq.financial_data.get(symbol, {}) if isinstance(yq.financial_data, dict) else {}

        if isinstance(summary, str): summary = {}
        if isinstance(price, str): price = {}
        if isinstance(financial_data, str): financial_data = {}

        combined_info = {**summary, **price, **financial_data}
        combined_info['symbol'] = symbol

        if 'regularMarketPrice' in combined_info:
            combined_info['currentPrice'] = combined_info['regularMarketPrice']

        def format_yq_df(df_yq):
            if isinstance(df_yq, pd.DataFrame) and not df_yq.empty:
                df_yq = df_yq.copy()
                if isinstance(df_yq.index, pd.MultiIndex):
                    try:
                        df_yq = df_yq.xs(symbol, level=0)
                    except KeyError:
                        return pd.DataFrame()
                if 'asOfDate' in df_yq.columns:
                    df_yq.set_index('asOfDate', inplace=True)
                return df_yq.transpose()
            return pd.DataFrame()

        inc_stmt = format_yq_df(yq.income_statement())
        bal_sheet = format_yq_df(yq.balance_sheet())
        cash_flow = format_yq_df(yq.cash_flow())

        if not inc_stmt.empty:
            inc_stmt.rename(index={
                'TotalRevenue': 'Total Revenue',
                'PretaxIncome': 'Pretax Income',
                'TaxProvision': 'Tax Provision',
                'InterestExpense': 'Interest Expense'
            }, inplace=True)
        if not bal_sheet.empty:
            bal_sheet.rename(index={
                'TotalDebt': 'Total Debt',
                'StockholdersEquity': 'Stockholders Equity',
                'TotalAssets': 'Total Assets',
                'CurrentAssets': 'Current Assets',
                'CurrentLiabilities': 'Current Liabilities',
                'RetainedEarnings': 'Retained Earnings'
            }, inplace=True)
        if not cash_flow.empty:
            cash_flow.rename(index={
                'OperatingCashFlow': 'Operating Cash Flow',
                'CapitalExpenditure': 'Capital Expenditure'
            }, inplace=True)

        return {
            "info": combined_info,
            "financials": inc_stmt,
            "balance_sheet": bal_sheet,
            "cashflow": cash_flow,
            "symbol": symbol
        }

    except Exception as e:
        logger.error(f"Tutte le API hanno fallito per i fondamentali di {symbol}: {e}")
        return None


def calculate_fundamental_metrics(raw_data: Dict[str, Any]) -> Optional[FundamentalMetrics]:
    try:
        info = raw_data["info"]
        symbol = raw_data["symbol"]

        # [REFACTOR] Asset non tradizionali (crypto, FX, indici): metriche non applicabili
        if is_non_traditional_asset(symbol, info):
            return FundamentalMetrics(
                ticker=symbol,
                company_name=info.get('shortName', symbol),
                price=safe_float(info.get('regularMarketPrice') or info.get('currentPrice'), 0.0),
                fcf=0.0,
                roic=0.0,
                peg_ratio=None,
                peg_source="N/A (non-equity)",
                pe_ratio=None,
                interest_coverage=SAFE_INTEREST_COVERAGE,
                debt_to_equity=None,
                revenue_growth=None,
                net_margin=None,
                fcf_margin=None,
                currency=info.get('currency', 'USD'),
                raw_data=raw_data
            )

        fin = raw_data["financials"]
        bs = raw_data["balance_sheet"]
        cf = raw_data["cashflow"]

        def get_first(df: pd.DataFrame, idx: str, default: float = 0.0) -> float:
            if df is None or df.empty or idx not in df.index:
                return default
            try:
                return safe_float(df.loc[idx].iloc[0], default)
            except Exception:
                return default

        op_cash = get_first(cf, 'Operating Cash Flow', 0.0)
        cap_ex = get_first(cf, 'Capital Expenditure', 0.0)
        fcf = float(op_cash - abs(cap_ex))

        total_debt = get_first(bs, 'Total Debt', 0.0)
        equity = get_first(bs, 'Stockholders Equity', np.nan)
        invested_cap = total_debt + equity
        ebit = get_first(fin, 'EBIT', 0.0)

        # [FIN-FIX] Tax rate: clip difensivo, gestisce pretax_inc <= 0
        tax_rate = DEFAULT_TAX_RATE
        if 'Tax Provision' in fin.index and 'Pretax Income' in fin.index and not fin.empty:
            pretax_inc = get_first(fin, 'Pretax Income', 0.0)
            tax_provision = get_first(fin, 'Tax Provision', 0.0)
            if pretax_inc > 0:
                tax_rate = float(np.clip(tax_provision / pretax_inc, 0.0, 1.0))

        roic = 0.0
        if invested_cap and not np.isnan(invested_cap) and invested_cap > 0:
            roic = float((ebit * (1 - tax_rate)) / invested_cap)

        pe = info.get('trailingPE')
        growth = info.get('earningsGrowth')
        peg = info.get('pegRatio')
        peg_src = "N/A"
        if peg is not None:
            peg_src = "Official"
        elif pe and pe > 0 and growth and growth > 0:
            # earningsGrowth e' frazione (es. 0.10 = 10%); PEG = PE / (growth%)
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
            try:
                net_margin = float(net_income) / float(total_revenue)
            except (TypeError, ValueError, ZeroDivisionError):
                net_margin = None

        fcf_margin = None
        if total_revenue not in (None, 0):
            try:
                fcf_margin = fcf / float(total_revenue)
            except (TypeError, ValueError, ZeroDivisionError):
                fcf_margin = None

        return FundamentalMetrics(
            ticker=symbol,
            company_name=info.get('longName', symbol),
            price=safe_float(info.get('currentPrice') or info.get('regularMarketPrice'), 0.0),
            fcf=fcf,
            roic=roic,
            peg_ratio=safe_float(peg, None) if peg is not None else None,
            peg_source=peg_src,
            pe_ratio=safe_float(pe, None) if pe is not None else None,
            interest_coverage=int_cov,
            debt_to_equity=debt_to_equity,
            revenue_growth=safe_float(revenue_growth, None) if revenue_growth is not None else None,
            net_margin=net_margin,
            fcf_margin=fcf_margin,
            currency=info.get('currency', 'USD'),
            raw_data=raw_data
        )

    except Exception as e:
        logger.error(f"Errore calcolo metriche {raw_data.get('symbol', '?')}: {e}")
        return None


def fetch_metrics_for_ticker(ticker: str) -> Tuple[str, Optional[Dict[str, Any]], Optional[str]]:
    """[NEW] Wrapper per ThreadPoolExecutor: ritorna (ticker, ui_dict, error)."""
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
    """[NEW] Fetch parallelo dei fondamentali con ThreadPoolExecutor."""
    results: List[Dict[str, Any]] = []
    errors: List[str] = []
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
    return results, errors


# ==========================================================================
# 4. DATA ENGINE: ANALISI TECNICA CON FALLBACK
# ==========================================================================
@st.cache_data(ttl=900, show_spinner=False)
def get_technical_data(symbol: str) -> Optional[pd.DataFrame]:
    # TENTATIVO 1: yfinance
    try:
        df = yf.download(symbol, period="2y", interval="1d", progress=False, auto_adjust=True)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.loc[:, ~df.columns.duplicated(keep='first')]
            # [FIN-FIX] Soglia rilassata: 60 sessioni minime invece di 200,
            # per non escludere IPO recenti o asset di mercati piccoli.
            if len(df) >= 60:
                return df
    except Exception as e:
        logger.info(f"yfinance tecnico fallito per {symbol}: {e}. Passo a YahooQuery.")

    # TENTATIVO 2: YahooQuery
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
        logger.error(f"Tutte le API hanno fallito per analisi tecnica di {symbol}: {e}")

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

    # [NEW] MACD aggiunto
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    data['MACD'] = ema12 - ema26
    data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

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
            macd = ta.macd(close)
            if macd is not None and not macd.empty:
                m_col = macd.filter(like='MACD_').filter(regex=r'_\d+_\d+_\d+$')
                s_col = macd.filter(like='MACDs_')
                if not m_col.empty:
                    data['MACD'] = m_col.iloc[:, 0]
                if not s_col.empty:
                    data['MACD_signal'] = s_col.iloc[:, 0]
        except Exception:
            pass
    return data


def calculate_timing_score(data: pd.DataFrame, current_price: float) -> Tuple[int, List[str]]:
    """
    [REFACTOR] Punteggio mantenuto invariato nelle soglie ma con
    output clipped 0..100 esplicito.
    """
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

    # [NEW] MACD bullish crossover
    macd = last_row.get('MACD')
    macd_sig = last_row.get('MACD_signal')
    if pd.notna(macd) and pd.notna(macd_sig):
        if macd > macd_sig:
            score += 10
            reasons.append("✅ MACD sopra signal line (momentum positivo)")
        else:
            reasons.append("⚠️ MACD sotto signal line")

    # Clip difensivo finale
    score = int(np.clip(score, 0, 100))
    return score, reasons


# ==========================================================================
# 5. MOTORE QUANTISTICO
# ==========================================================================
def calculate_quant_metrics(
    df: pd.DataFrame,
    fund_data: Optional[Dict[str, Any]],
    risk_free: Optional[float] = None
) -> Dict[str, Any]:
    """
    [REFACTOR] Logica di scoring identica all'originale.
    [FIN-FIX] Altman Z-Score restituisce "N/A" se marketCap mancante
    invece di usare 1 come fallback (che produceva valori distorti).
    [NEW] risk_free puo' essere passato per usare il tasso dinamico.
    """
    rf = risk_free if risk_free is not None else get_active_risk_free_rate()

    returns = df['Close'].pct_change().dropna()
    excess_returns = returns - (rf / TRADING_DAYS_YEAR)
    sharpe = (
        (excess_returns.mean() / excess_returns.std()) * np.sqrt(TRADING_DAYS_YEAR)
        if excess_returns.std() != 0 else 0.0
    )
    vol = returns.std() * np.sqrt(TRADING_DAYS_YEAR) if not returns.empty else np.nan

    log_returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
    if len(log_returns) >= 2:
        cum_log_returns = log_returns.cumsum().values.reshape(-1, 1)
        x = np.arange(len(cum_log_returns)).reshape(-1, 1)
        model = LinearRegression().fit(x, cum_log_returns)
        r_sq = model.score(x, cum_log_returns)
        slope = float(model.coef_[0][0])
    else:
        r_sq = np.nan
        slope = np.nan

    z_score: Any = "N/A"
    if fund_data and 'info' in fund_data and not is_non_traditional_asset(
        fund_data.get('symbol', ''), fund_data.get('info')
    ):
        try:
            bs = fund_data.get("balance_sheet")
            fin = fund_data.get("financials")
            info = fund_data.get("info", {})
            if isinstance(bs, pd.DataFrame) and not bs.empty and \
               isinstance(fin, pd.DataFrame) and not fin.empty:
                ta_val = bs.loc['Total Assets'].iloc[0] if 'Total Assets' in bs.index else 0
                if ta_val and ta_val > 0:
                    wc = (
                        bs.loc['Current Assets'].iloc[0] - bs.loc['Current Liabilities'].iloc[0]
                        if 'Current Assets' in bs.index and 'Current Liabilities' in bs.index
                        else 0
                    )
                    re_val = bs.loc['Retained Earnings'].iloc[0] if 'Retained Earnings' in bs.index else 0
                    ebit = fin.loc['EBIT'].iloc[0] if 'EBIT' in fin.index else 0
                    # [FIN-FIX] marketCap None -> Z-Score = "N/A" anziche' fallback a 1
                    mc = info.get('marketCap')
                    tl = (
                        bs.loc['Total Liabilities Net Minority Interest'].iloc[0]
                        if 'Total Liabilities Net Minority Interest' in bs.index
                        else (bs.loc['Total Liabilities'].iloc[0] if 'Total Liabilities' in bs.index else 0)
                    )
                    if mc is not None and tl and tl > 0:
                        rev = info.get('totalRevenue', 0) or 0
                        z_score = float(
                            (1.2 * (wc / ta_val)) +
                            (1.4 * (re_val / ta_val)) +
                            (3.3 * (ebit / ta_val)) +
                            (0.6 * (mc / tl)) +
                            (1.0 * (rev / ta_val))
                        )
        except Exception as e:
            logger.debug(f"Altman Z-Score non calcolabile: {e}")

    return {
        "Sharpe Ratio": float(sharpe) if not np.isnan(sharpe) else 0.0,
        "Annual Volatility": float(vol) if not np.isnan(vol) else np.nan,
        "R-Squared": float(r_sq) if not np.isnan(r_sq) else np.nan,
        "Altman Z-Score": z_score,
        "Price Percentile": float((df['Close'] < df['Close'].iloc[-1]).mean() * 100),
        "Trend Slope": slope,
        "Risk Free Used": float(rf),
    }


def calculate_risk_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """[REFACTOR] Aggiunti Sortino e Calmar ratio + downside deviation."""
    prices = df['Close'].dropna()
    returns = prices.pct_change().dropna()
    if returns.empty:
        nan = float('nan')
        return {
            "Max Drawdown": nan, "CAGR": nan, "VaR_95": nan, "CVaR_95": nan,
            "Skew": nan, "Kurt": nan, "Sortino": nan, "Calmar": nan,
            "Downside Deviation": nan,
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

    # [NEW] Sortino Ratio: utilizza solo la deviazione standard delle perdite
    rf_daily = get_active_risk_free_rate() / TRADING_DAYS_YEAR
    downside = returns[returns < rf_daily]
    downside_dev = downside.std() * np.sqrt(TRADING_DAYS_YEAR) if not downside.empty else np.nan
    ann_excess_ret = (returns.mean() - rf_daily) * TRADING_DAYS_YEAR
    sortino = (
        ann_excess_ret / downside_dev
        if downside_dev and not np.isnan(downside_dev) and downside_dev > 0
        else np.nan
    )

    # [NEW] Calmar Ratio: CAGR / |Max Drawdown|
    calmar = cagr / abs(max_dd) if max_dd and max_dd < 0 and not np.isnan(cagr) else np.nan

    return {
        "Max Drawdown": float(max_dd),
        "CAGR": float(cagr) if not np.isnan(cagr) else np.nan,
        "VaR_95": float(var_95),
        "CVaR_95": float(cvar_95) if not np.isnan(cvar_95) else np.nan,
        "Skew": float(skew),
        "Kurt": float(kurt),
        "Sortino": float(sortino) if not np.isnan(sortino) else np.nan,
        "Calmar": float(calmar) if not np.isnan(calmar) else np.nan,
        "Downside Deviation": float(downside_dev) if not np.isnan(downside_dev) else np.nan,
    }


def monte_carlo_equity(
    df: pd.DataFrame,
    n_paths: int = 1000,
    horizon_days: int = 252,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    [REFACTOR] Bootstrap iid sui rendimenti storici (logica originaria).
    [NEW] Aggiunto block bootstrap opzionale per preservare autocorrelazione.
    """
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

    return {
        "paths": equity_paths,
        "final_distribution": final_values,
        "q05": q05, "q50": q50, "q95": q95
    }


def monte_carlo_block_bootstrap(
    df: pd.DataFrame,
    n_paths: int = 1000,
    horizon_days: int = 252,
    block_size: int = 5,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    [NEW] Block bootstrap: campiona blocchi consecutivi di rendimenti
    per preservare autocorrelazione e clustering della volatilita'.
    """
    prices = df['Close'].dropna()
    returns = prices.pct_change().dropna().values
    if returns.size < block_size:
        return monte_carlo_equity(df, n_paths, horizon_days, seed)

    rng = np.random.default_rng(seed)
    n_blocks_per_path = (horizon_days // block_size) + 1
    paths = np.empty((n_paths, n_blocks_per_path * block_size))
    max_start = len(returns) - block_size
    for i in range(n_paths):
        starts = rng.integers(0, max_start + 1, size=n_blocks_per_path)
        path = np.concatenate([returns[s:s+block_size] for s in starts])
        paths[i, :] = path
    paths = paths[:, :horizon_days]
    equity_paths = (1 + paths).cumprod(axis=1)

    return {
        "paths": equity_paths,
        "final_distribution": equity_paths[:, -1],
        "q05": np.quantile(equity_paths, 0.05, axis=0),
        "q50": np.quantile(equity_paths, 0.50, axis=0),
        "q95": np.quantile(equity_paths, 0.95, axis=0),
    }


# ==========================================================================
# 5.D SMART QUANT SCORE (con pesi configurabili)
# ==========================================================================
def compute_smart_quant_score(
    row: Any,
    timing_score: int,
    qm: Dict[str, Any],
    risk: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """[REFACTOR] Pesi configurabili (default 0.40/0.30/0.30 invariati)."""
    w = weights or DEFAULT_SMART_WEIGHTS

    f_score = 0.0
    roic = row.get("ROIC", 0.0) or 0.0
    f_score += float(np.clip((roic - 0.10) / (0.25 - 0.10), 0, 1)) * 30.0

    peg = row.get("PEG Ratio", None)
    if peg is not None and peg > 0:
        if peg <= 1: f_score += 25.0
        elif peg <= 2: f_score += 12.0

    debt_to_equity = row.get("Debt/Equity", None)
    if debt_to_equity is not None:
        if debt_to_equity <= 0.5: f_score += 12.0
        elif debt_to_equity <= 1.0: f_score += 7.0
        elif debt_to_equity <= 2.0: f_score += 3.0

    revenue_growth = row.get("Revenue Growth", None)
    if revenue_growth is not None:
        if revenue_growth >= 0.15: f_score += 6.0
        elif revenue_growth >= 0.05: f_score += 3.0
        elif revenue_growth > 0: f_score += 1.5

    net_margin = row.get("Net Margin", None)
    if net_margin is not None:
        if net_margin >= 0.20: f_score += 8.0
        elif net_margin >= 0.10: f_score += 4.0
        elif net_margin > 0: f_score += 2.0

    fcf_margin = row.get("FCF Margin", None)
    if fcf_margin is not None:
        if fcf_margin >= 0.15: f_score += 12.0
        elif fcf_margin >= 0.08: f_score += 7.0
        elif fcf_margin > 0: f_score += 3.0

    z = qm.get("Altman Z-Score", "N/A")
    if isinstance(z, (int, float, np.floating)) and not isinstance(z, bool):
        if z >= 3.0: f_score += 7.0
        elif z >= ALTMAN_SAFE_THRESHOLD: f_score += 3.5

    f_score = float(np.clip(f_score, 0, 100))
    t_score = float(np.clip(timing_score, 0, 100))

    q_score = 0.0
    sharpe = qm.get("Sharpe Ratio", 0.0) or 0.0
    max_dd = risk.get("Max Drawdown", 0.0) or 0.0

    if sharpe <= 0:
        q_score += 0.0
    elif sharpe <= 1:
        q_score += 30.0 * sharpe
    elif sharpe <= 2:
        q_score += 30.0 + 30.0 * (sharpe - 1.0)
    else:
        q_score += 80.0

    if isinstance(max_dd, (float, np.floating)):
        if max_dd < -0.5: q_score -= 20.0
        elif max_dd < -0.3: q_score -= 10.0

    q_score = float(np.clip(q_score, 0, 100))

    smart = w["F"] * f_score + w["T"] * t_score + w["Q"] * q_score
    smart = float(np.clip(smart, 0, 100))

    return {
        "SmartScore": smart,
        "FundamentalScore": f_score,
        "TechnicalScore": t_score,
        "QuantRiskScore": q_score
    }


# ==========================================================================
# 5.E TAX IMPACT (legacy - manteniamo l'originale per backward compat)
# ==========================================================================
def calculate_tax_impact(
    df_weights: pd.DataFrame,
    tax_rate: float = DEFAULT_TAX_RATE
) -> pd.DataFrame:
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
        "Ticker", "Importo Investito", "Valore di Mercato", "P&L",
        "Aliquota Fiscale %", "Imposta Teorica", "Plus/Minus Netta",
        "Valore Netto Post Imposta", "Rendimento Netto %"
    ]
    return df_tax[[c for c in cols if c in df_tax.columns]]


# ==========================================================================
# 5.F PORTAFOGLIO: RENDIMENTI & METRICHE
# ==========================================================================
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

    # [FIN-FIX] join "outer" + ffill iniziale per non perdere dati di
    # ticker con storia piu' breve. Poi dropna finale solo sulle prime righe.
    df_rets = pd.concat(series_list, axis=1, join="outer").sort_index()
    df_rets = df_rets.dropna(how="all")
    df_rets = df_rets.dropna()  # mantiene solo periodo comune
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
        return {"AnnRet": np.nan, "AnnVol": np.nan, "Sharpe": np.nan,
                "MaxDD": np.nan, "Sortino": np.nan, "Calmar": np.nan}

    rf = get_active_risk_free_rate()
    mu = port_ret.mean() * TRADING_DAYS_YEAR
    sigma = port_ret.std() * np.sqrt(TRADING_DAYS_YEAR)

    excess = mu - rf
    sharpe = excess / sigma if sigma > 0 else np.nan

    equity = (1 + port_ret).cumprod()
    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1.0
    max_dd = drawdown.min() if not drawdown.empty else np.nan

    # [NEW] Sortino & Calmar a livello portafoglio
    rf_daily = rf / TRADING_DAYS_YEAR
    downside = port_ret[port_ret < rf_daily]
    downside_dev = downside.std() * np.sqrt(TRADING_DAYS_YEAR) if not downside.empty else np.nan
    sortino = excess / downside_dev if downside_dev and downside_dev > 0 else np.nan

    # CAGR per Calmar
    n_years = len(port_ret) / TRADING_DAYS_YEAR
    cagr = (equity.iloc[-1]) ** (1.0 / n_years) - 1.0 if n_years > 0 else np.nan
    calmar = cagr / abs(max_dd) if max_dd < 0 and not np.isnan(cagr) else np.nan

    return {
        "AnnRet": float(mu),
        "AnnVol": float(sigma),
        "Sharpe": float(sharpe) if not np.isnan(sharpe) else np.nan,
        "MaxDD": float(max_dd) if not np.isnan(max_dd) else np.nan,
        "Sortino": float(sortino) if not np.isnan(sortino) else np.nan,
        "Calmar": float(calmar) if not np.isnan(calmar) else np.nan,
        "CAGR": float(cagr) if not np.isnan(cagr) else np.nan,
    }


# ==========================================================================
# 5.G CONCENTRAZIONE & DIVERSIFICAZIONE  [NEW]
# ==========================================================================
def calculate_concentration_metrics(weights_pct: Dict[str, float]) -> Dict[str, float]:
    """[NEW] HHI e Effective Number of Stocks (ENS)."""
    w = np.array(list(weights_pct.values()), dtype=float) / 100.0
    if w.sum() <= 0:
        return {"HHI": np.nan, "ENS": np.nan, "Top1 %": np.nan, "Top3 %": np.nan}
    w = w / w.sum()
    hhi = float((w ** 2).sum())
    ens = 1.0 / hhi if hhi > 0 else np.nan
    sorted_w = np.sort(w)[::-1]
    top1 = float(sorted_w[0] * 100.0)
    top3 = float(sorted_w[:3].sum() * 100.0)
    return {
        "HHI": hhi,
        "ENS": float(ens),
        "Top1 %": top1,
        "Top3 %": top3,
    }


def calculate_portfolio_beta(
    port_ret: pd.Series,
    benchmark_symbol: str = DEFAULT_BENCHMARK
) -> Dict[str, float]:
    """[NEW] Beta vs benchmark (S&P 500 di default)."""
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
        # CAPM: alpha = mean(port - rf) - beta*(mean(bench - rf))
        alpha_daily = (joined['port'].mean() - rf_daily) - beta * (joined['bench'].mean() - rf_daily)
        alpha_ann = alpha_daily * TRADING_DAYS_YEAR
        corr = joined['port'].corr(joined['bench'])
        return {
            "Beta": float(beta) if not np.isnan(beta) else np.nan,
            "Alpha (ann.)": float(alpha_ann) if not np.isnan(alpha_ann) else np.nan,
            "Corr": float(corr) if not np.isnan(corr) else np.nan,
        }
    except Exception as e:
        logger.debug(f"Beta non calcolabile: {e}")
        return {"Beta": np.nan, "Alpha (ann.)": np.nan, "Corr": np.nan}


# ==========================================================================
# 5.H PORTFOLIO HELPER
# ==========================================================================
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
            try:
                return float(price)
            except (TypeError, ValueError):
                return None
    return None


def get_ticker_native_currency(symbol: str) -> Optional[str]:
    """[NEW] Ritorna la valuta nativa del ticker secondo i fondamentali."""
    raw = get_fundamental_data(symbol)
    if raw and raw.get('info'):
        cur = raw['info'].get('currency')
        if cur:
            return str(cur).upper().strip()
    return None


def calculate_position_from_quantity(
    ticker: str,
    quantity: float,
    pmc: float,
    user_currency: Optional[str] = None,
    base_currency: Optional[str] = None
) -> Dict[str, float]:
    """
    [FIN-FIX] CRITICAL: il prezzo attuale e' nella valuta nativa del ticker.
    Se l'utente ha inserito PMC in una valuta diversa (es. USD per ticker .MI
    quotato in EUR), va applicata la conversione. Se base_currency e'
    specificata, anche i totali sono convertiti.
    """
    current_price_native = get_latest_price(ticker)
    native_cur = get_ticker_native_currency(ticker) or "EUR"
    user_cur = (user_currency or native_cur).upper().strip()

    invested = float(quantity * pmc)  # nella valuta utente
    if current_price_native is None:
        return {
            'Prezzo Attuale': np.nan,
            'Importo Investito': invested,
            'Valore di Mercato': 0.0,
            'P&L': 0.0,
            'P&L %': 0.0,
            'Valuta Nativa': native_cur,
        }

    # Converti il prezzo nativo nella valuta utente per calcolare il valore
    # di mercato consistente con l'importo investito.
    fx_native_to_user = get_fx_rate(native_cur, user_cur) if native_cur != user_cur else 1.0
    current_price_in_user_cur = float(current_price_native) * fx_native_to_user
    market_value = float(quantity) * current_price_in_user_cur
    pnl_value = market_value - invested
    pnl_pct = (pnl_value / invested) * 100.0 if invested > 0 else 0.0

    return {
        'Prezzo Attuale': current_price_in_user_cur,
        'Prezzo Attuale Nativa': float(current_price_native),
        'Importo Investito': invested,
        'Valore di Mercato': market_value,
        'P&L': pnl_value,
        'P&L %': pnl_pct,
        'Valuta Nativa': native_cur,
        'FX Native->User': fx_native_to_user,
    }


# ==========================================================================
# 5.I PWA SUPPORT (logica originaria preservata)
# ==========================================================================
def inject_pwa_support():
    st.markdown("""
    <script>
    (function(){
      const base64Png = 'iVBORw0KGgoAAAANSUhEUgAAAMAAAADACAIAAADdvvtQAAACNklEQVR4nO3SwQ3AIBDAsNL9dz6WIEJC9gR5ZM18A6ft2wG8yQBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA5gNYDaA2QBmA9gBTjICfuDZUUYAAAAASUVORK5CYII=';
      const manifest = {
        name: 'V-Quant Pro',
        short_name: 'V-Quant Pro',
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
    })();
    </script>
    """, unsafe_allow_html=True)


# ==========================================================================
# 5.J ALLOCAZIONE E RIBILANCIAMENTO
# ==========================================================================
def infer_asset_class(ticker: str, company_name: str = "") -> str:
    t = str(ticker).upper()
    name = str(company_name).lower()

    etf_keywords = ["etf", "ucits", "ishares", "xtrackers", "vanguard",
                    "lyxor", "amundi", "invesco", "wisdomtree", "spdr"]
    bond_keywords = ["bond", "treasury", "aggregate", "gov", "government", "corporate"]
    gold_keywords = ["gold", "physical gold", "precious", "silver", "metals"]
    crypto_keywords = ["btc-", "eth-", "-usd", "-eur"]

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

    # [FIN-FIX] Estesa la lista dei suffissi mercato
    suffix_map = {
        ".MI": "Italia", ".DE": "Germania", ".PA": "Francia", ".L": "Regno Unito",
        ".AS": "Olanda", ".BR": "Belgio", ".LS": "Portogallo", ".MC": "Spagna",
        ".SW": "Svizzera", ".ST": "Svezia", ".CO": "Danimarca", ".HE": "Finlandia",
        ".OL": "Norvegia", ".VI": "Austria", ".IR": "Irlanda",
        ".TO": "Canada", ".V": "Canada", ".AX": "Australia", ".NZ": "Nuova Zelanda",
        ".T": "Giappone", ".HK": "Hong Kong", ".SS": "Cina (Shanghai)",
        ".SZ": "Cina (Shenzhen)", ".KS": "Corea del Sud", ".NS": "India", ".BO": "India",
        ".BR": "Brasile", ".MX": "Messico", ".SA": "Brasile",
    }
    for suf, geo in suffix_map.items():
        if t.endswith(suf):
            return geo

    if "-USD" in t:
        return "Crypto/USD"

    us_keywords = ["s&p", "nasdaq", "russell", "usa", "united states", "msci usa"]
    eu_keywords = ["europe", "stoxx", "euro stoxx", "msci europe"]
    em_keywords = ["emerging", "msci em"]
    world_keywords = ["world", "all-world", "acwi", "ftse all-world", "global"]
    japan_keywords = ["japan", "topix", "nikkei"]
    china_keywords = ["china", "csi", "hang seng"]

    if any(k in name for k in world_keywords): return "Globale"
    if any(k in name for k in us_keywords): return "USA"
    if any(k in name for k in eu_keywords): return "Europa"
    if any(k in name for k in em_keywords): return "Emergenti"
    if any(k in name for k in japan_keywords): return "Giappone"
    if any(k in name for k in china_keywords): return "Cina"

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
        .sum().reset_index()
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
    threshold = tolerance_pct / 100.0 * total_value
    merged["Azione"] = np.where(
        merged["Azione €"] > threshold, "Compra",
        np.where(merged["Azione €"] < -threshold, "Riduci", "In target")
    )
    return merged.sort_values("Azione €", ascending=False).reset_index(drop=True)


# ==========================================================================
# 6. UI: SIDEBAR
# ==========================================================================
def render_apk_download_box() -> None:
    with st.sidebar.expander("Download App Android (APK)", expanded=False):
        st.write("Scarica l'APK ufficiale di V-Quant Pro per installare l'app su Android.")
        st.link_button(
            "📲 Scarica V-Quant Pro.apk",
            "https://github.com/innovativeprogram/V-QuantPro-relaases/releases/download/v1.0.0/Vquantpro.apk",
            width='stretch'
        )
        st.caption("Se Android blocca l'installazione, abilita temporaneamente le origini sconosciute "
                   "per il browser o il file manager usato per il download.")


def setup_sidebar() -> Dict[str, Any]:
    render_auth_sidebar()

    # [NEW] Pannello impostazioni globali
    with st.sidebar.expander("⚙️ Impostazioni globali", expanded=False):
        base_currency = st.selectbox(
            "Valuta base portafoglio", ["EUR", "USD", "GBP", "CHF"], index=0,
            key="base_currency_sel"
        )
        rf_mode = st.radio(
            "Tasso risk-free", ["Dinamico (^IRX)", "Manuale", "Default 4%"],
            index=0, key="rf_mode_sel"
        )
        if rf_mode == "Manuale":
            rf_manual = st.number_input(
                "Tasso risk-free manuale (%)", 0.0, 15.0,
                value=DEFAULT_RISK_FREE_RATE * 100, step=0.1
            )
            st.session_state["risk_free_override"] = rf_manual / 100.0
        elif rf_mode == "Default 4%":
            st.session_state["risk_free_override"] = DEFAULT_RISK_FREE_RATE
        else:
            st.session_state["risk_free_override"] = None
        rf_eff = get_active_risk_free_rate()
        st.caption(f"Tasso risk-free effettivo: {rf_eff*100:.2f}%")

        st.markdown("**Pesi Smart Quant Score**")
        col_w1, col_w2, col_w3 = st.columns(3)
        wF = col_w1.number_input("F", 0.0, 1.0, DEFAULT_SMART_WEIGHTS["F"], 0.05)
        wT = col_w2.number_input("T", 0.0, 1.0, DEFAULT_SMART_WEIGHTS["T"], 0.05)
        wQ = col_w3.number_input("Q", 0.0, 1.0, DEFAULT_SMART_WEIGHTS["Q"], 0.05)
        s = wF + wT + wQ
        if s > 0:
            st.session_state["smart_weights"] = {"F": wF/s, "T": wT/s, "Q": wQ/s}
        else:
            st.session_state["smart_weights"] = DEFAULT_SMART_WEIGHTS

    st.session_state["base_currency"] = base_currency

    st.sidebar.header("1. Selezione Asset")
    input_mode = st.sidebar.radio("Modalità", ["Manuale", "Batch CSV"], horizontal=True)

    file, manual = None, None
    if input_mode == "Batch CSV":
        file = st.sidebar.file_uploader("Carica CSV (colonna 'Ticker' richiesta)", type=["csv"])
    else:
        manual = st.sidebar.text_input("Ticker", value="AAPL").upper().strip()

    st.sidebar.header("2. Mercato")
    market = st.sidebar.selectbox(
        "Borsa",
        ["USA", "Italia (.MI)", "Germania (.DE)", "Francia (.PA)",
         "GB (.L)", "Spagna (.MC)", "Svizzera (.SW)",
         "Canada (.TO)", "Giappone (.T)", "Hong Kong (.HK)",
         "Australia (.AX)", "India (.NS)", "Crypto", "Custom"]
    )
    suffix_lookup = {
        "Italia": ".MI", "Germania": ".DE", "Francia": ".PA", "GB": ".L",
        "Spagna": ".MC", "Svizzera": ".SW", "Canada": ".TO",
        "Giappone": ".T", "Hong Kong": ".HK", "Australia": ".AX", "India": ".NS",
    }
    suffix = ""
    for k, s in suffix_lookup.items():
        if k in market:
            suffix = s
            break

    analyze_btn = st.sidebar.button("🚀 Avvia Analisi", width='stretch')

    with st.sidebar.expander("⚙️ Parametri Fondamentali"):
        cfg = {
            "roic": st.number_input("Min ROIC %", value=10.0, step=0.5),
            "fcf": st.number_input("Min FCF (Mld)", value=0.0, step=1e9),
            "peg": st.number_input("Max PEG Ratio", value=1.5, step=0.1),
            "pe": st.number_input("Max PE (Fallback)", value=25.0),
            "intcov": st.number_input("Min Int. Coverage", value=3.0),
            "custom_max_de": st.number_input("Custom Max Debt/Equity", value=1.0, step=0.1),
            "custom_min_fcf_margin": st.number_input("Custom Min FCF Margin", value=0.08, step=0.01, format="%.2f"),
            "custom_min_net_margin": st.number_input("Custom Min Net Margin", value=0.10, step=0.01, format="%.2f"),
            "perfectonly": st.checkbox("Solo All Green"),
            "model_mode": st.selectbox(
                "Modello verdetto",
                ["Entrambi", "Classico", "Evoluto", "Personalizzabile"], index=0
            ),
        }

    with st.sidebar.expander("❓ Come cercare il ticker corretto"):
        st.markdown("""
- Azioni USA: solo ticker, es. AAPL, MSFT.
- Italia: aggiungi `.MI` (es. STLAM.MI, ENI.MI).
- Germania: `.DE` (BMW.DE, SAP.DE).
- Francia: `.PA` (AIR.PA, OR.PA).
- UK: `.L` (ULVR.L).
- Spagna: `.MC`, Svizzera: `.SW`.
- Canada: `.TO`, Giappone: `.T`, Hong Kong: `.HK`.
- Crypto: coppia con valuta, es. `BTC-USD`, `ETH-USD`.
- In dubbio: cerca su Yahoo Finance e copia il ticker esatto.
        """)

    render_apk_download_box()

    # =============================================================================
    # CHI SIAMO
    # =============================================================================
    with st.sidebar.expander("ℹ️ Chi Siamo", expanded=False):

        st.markdown("""
### Benvenuti su V-QUANT PRO

V-QUANT PRO è una piattaforma indipendente di analisi finanziaria dedicata agli investitori retail che adottano un approccio quantitativo e basato sul valore.

La nostra missione è democratizzare l'accesso a metriche finanziarie avanzate, fornendo strumenti per il monitoraggio del Margine di Sicurezza su ETF , Crypto e singoli titoli azionari ed obbligazionari.

Crediamo fermamente che l'analisi rigorosa dei dati sia l'unica bussola affidabile per navigare nei mercati finanziari a lungo termine.

### Cosa facciamo:
- Analisi del rischio e calcolo di Alpha e Beta di portafoglio
- Monitoraggio dei fondamentali (ROIC, Altman Z-Score, F-Score)
- Strumenti di supporto decisionale basati su modelli matematici

### ⚠️ Disclaimer Legale

V-QUANT PRO è una piattaforma a scopo esclusivamente informativo e didattico. I dati, le analisi e le opinioni espresse non costituiscono in alcun modo consulenza finanziaria, sollecitazione al pubblico risparmio o suggerimento di investimento. Ogni decisione di investimento presa dall'utente è di sua esclusiva responsabilità.

Sviluppato con passione da Innovative Program.

        """)

    # =============================================================================
    # PRIVACY POLICY
    # =============================================================================
    with st.sidebar.expander("🔐 Privacy & Cookie Policy", expanded=False):

        st.markdown("""
### Informativa ai sensi del Regolamento UE 2016/679 (GDPR)

#### 1. Conservazione dei Dati 
Tutti i dati sensibili, inclusi i dati di autenticazione (email e password) e le configurazioni del tuo portafoglio, sono **detenuti e gestiti in modo sicuro da Supabase. 
#### Supabase è una piattaforma di database di livello enterprise che garantisce la crittografia dei dati a riposo e in transito.
Le password sono archiviate tramite hashing sicuro e non sono mai accessibili in chiaro agli amministratori di V-QUANT PRO.

#### 2. Analisi Finanziaria e Cookie
Questo sito utilizza  Google AdSense per la visualizzazione di annunci pubblicitari e cookie tecnici per il corretto funzionamento della Dashboard.
Google utilizza i cookie per pubblicare annunci basati sulle tue visite precedenti.
Puoi gestire le preferenze sugli annunci visitando le impostazioni di Google.

#### 3. Diritti dell'Utente

Poiché i dati sono detenuti su infrastruttura Supabase, puoi richiedere in ogni momento la cancellazione totale del tuo account e dei dati associati attraverso le impostazioni del profilo o contattandoci..

#### 4. Esclusione di Responsabilità

V-QUANT PRO non garantisce l'accuratezza dei dati forniti da fornitori terzi . L'utente riconosce che l'utilizzo delle informazioni avviene a proprio rischio e pericolo.

#### 5. Sicurezza

Utilizziamo protocolli HTTPS crittografati per garantire che ogni interazione tra il tuo browser e i nostri server sia protetta da accessi non autorizzati.

        """)
        
                 
    # =============================================================================
    # Sostienici
    # =============================================================================    
    
    with st.sidebar.expander("🎁 Sostieni V-QUANT PRO", expanded=False):
            
        st.markdown("""
        ### Perché una donazione?
        V-QUANT PRO è un progetto indipendente che offre strumenti di analisi avanzata gratuitamente. 
        Mantenere l'infrastruttura, aggiornare i dati in tempo reale e sviluppare nuove funzionalità ha dei costi vivi.
        
        Se ritieni che questa piattaforma ti stia aiutando a gestire meglio i tuoi investimenti, puoi sostenerne lo sviluppo con una libera donazione. Anche il costo di un caffè fa la differenza!
        """)
        
        # Sostituisci il link con il tuo link personale PayPal.me o il codice del pulsante
        st.link_button("🎁 Fai una donazione sicura su PayPal", "https://paypal.me/ctpneu", width="stretch")
        
        st.info("Nota: Le donazioni sono libere e non costituiscono il pagamento per un servizio di consulenza.")
      


    with st.sidebar.expander("Contatti", expanded=False):
        st.write("Per supporto tecnico, collaborazioni o richieste:")
        st.link_button(
            "📧 Scrivimi via mail",
            "mailto:innovativeprogram@proton.me?subject=Richiesta%20da%20V-QuantPro",
            width='stretch'
        )
        st.caption("Risposta normalmente entro 24/48 ore.")

    return {
        "mode": input_mode, "file": file, "manual": manual,
        "suffix": suffix, "btn": analyze_btn, "cfg": cfg,
        "base_currency": base_currency,
    }


def resolve_active_analysis_target() -> Tuple[Optional[str], Optional[pd.Series], Optional[Dict[str, Any]], str]:
    ticker = st.session_state.get('selected_ticker')
    batch = st.session_state.get('batch_results')
    row = None
    raw_data = None

    if ticker and batch is not None and not batch.empty and 'Ticker' in batch.columns and ticker in batch['Ticker'].values:
        row = batch[batch['Ticker'] == ticker].iloc[0]
        raw_data = row.get('_raw_data') if hasattr(row, 'get') else None
        return ticker, row, raw_data, 'batch'

    try:
        manual = sanitize_ticker(st.session_state.get('standalone_ticker_input', '')) \
                 if st.session_state.get('standalone_ticker_input') else ''
    except ValueError:
        manual = ''
    portfolio_tickers = st.session_state.get('portfolio_tickers', []) or []
    try:
        portfolio_pick = sanitize_ticker(st.session_state.get('standalone_portfolio_pick', '')) \
                         if st.session_state.get('standalone_portfolio_pick') else ''
    except ValueError:
        portfolio_pick = ''

    fallback_ticker = manual or portfolio_pick or (portfolio_tickers[0] if portfolio_tickers else None)
    if not fallback_ticker:
        return None, None, None, 'none'

    try:
        raw_data = get_fundamental_data(fallback_ticker)
        if raw_data:
            met = calculate_fundamental_metrics(raw_data)
            if met:
                row = pd.Series(met.to_ui_dict())
        return fallback_ticker, row, raw_data, 'standalone'
    except Exception as e:
        logger.warning(f'Standalone analysis unavailable for {fallback_ticker}: {e}')
        return fallback_ticker, None, None, 'standalone'







def _init_session_state() -> None:
    """[REFACTOR] Inizializzazione di tutti gli state in un punto solo."""
    defaults = {
        'batch_results': None,
        'selected_ticker': None,
        'portfolio_tickers': [],
        'holdings': {},
        'holdings_currency': {},
        'holdings_quantity': {},
        'holdings_pmc': {},
        'portfolio_target_mode': "Ticker",
        'portfolio_targets': {},
        'analysis_errors': [],
        'portfolio_loaded_from_db': False,
        'standalone_ticker_input': '',
        'standalone_portfolio_pick': '',
        'risk_free_override': None,
        'base_currency': 'EUR',
        'smart_weights': DEFAULT_SMART_WEIGHTS,
        'ai_ticker_chat_history': [],
        'ai_ticker_chat_last_symbol': None,
        'burry_ai_history': [],
        'burry_ai_symbol': '',
        'burry_ai_asset_type': 'Azione',
        'burry_ai_live_context': {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _portfolio_export_csv(df_weights: pd.DataFrame) -> bytes:
    """[NEW] Esporta il portafoglio in CSV per backup/condivisione."""
    return df_weights.to_csv(index=False).encode("utf-8")


def main():
    init_auth_state()
    _init_session_state()
    st.title("💲 V-Quant Pro")
    inject_pwa_support()

    ui = setup_sidebar()
    if is_authenticated() and not st.session_state.get('portfolio_loaded_from_db', False):
        load_user_portfolio()
        st.session_state.portfolio_loaded_from_db = True

    if not is_authenticated():
        st.info("Modalità ospite attiva: puoi usare l'app senza registrazione. "
                "Per salvare il portafoglio in modo permanente, effettua il login.")

    # Pulsante "Avvia Analisi": batch o singolo
    if ui["btn"]:
        targets: List[str] = [ui["manual"]] if ui["mode"] == "Manuale" else []
        if ui["mode"] == "Batch CSV" and ui["file"]:
            try:
                csv_df = pd.read_csv(ui["file"])
            except Exception as e:
                st.error(f"Errore lettura CSV: {e}")
                csv_df = None
            if csv_df is not None:
                if 'Ticker' not in csv_df.columns:
                    st.error('Il CSV deve contenere una colonna Ticker.')
                else:
                    targets = csv_df['Ticker'].dropna().astype(str).tolist()[:MAX_CSV_ROWS]

        if targets:
            normalized_targets: List[str] = []
            analysis_errors: List[str] = []
            for t in targets:
                try:
                    normalized_targets.append(normalize_ticker(t, ui["suffix"]))
                except Exception as e:
                    analysis_errors.append(f'{t}: {e}')

            # [NEW] Fetch parallelo
            with st.spinner(f"Analisi di {len(normalized_targets)} ticker in parallelo..."):
                results, fetch_errors = fetch_metrics_batch(normalized_targets)
            analysis_errors.extend(fetch_errors)

            st.session_state.batch_results = pd.DataFrame(results)
            st.session_state.analysis_errors = analysis_errors
            if results:
                st.session_state.selected_ticker = results[0]["Ticker"]
            else:
                st.session_state.selected_ticker = None

    if st.session_state.get('analysis_errors'):
        with st.expander('⚠️ Diagnostica analisi', expanded=False):
            for err in st.session_state.analysis_errors:
                st.write(f'- {err}')

    with st.sidebar:
        st.markdown('---')
        with st.expander('🤖 BurryAi', expanded=False):
            st.caption('Chiedi chiarimenti su azioni o ETF usando i risultati correnti e la logica del programma.')

            st.session_state.burry_ai_asset_type = st.selectbox(
                'Tipo strumento',
                ['Azione', 'ETF'],
                index=0 if st.session_state.get('burry_ai_asset_type', 'Azione') == 'Azione' else 1,
                key='burry_ai_asset_type_select'
            )

            st.session_state.burry_ai_symbol = st.text_input(
                'Ticker o nome',
                value=st.session_state.get('burry_ai_symbol', ''),
                key='burry_ai_symbol_input'
            )

            for msg in st.session_state.get('burry_ai_history', []):
                with st.chat_message(msg.get('role', 'assistant')):
                    st.markdown(msg.get('content', ''))

            burry_ai_prompt = st.chat_input('Chiedi a BurryAi', key='burry_ai_prompt_sidebar')
            if burry_ai_prompt:
                ctx = build_burry_ai_context(
                    st.session_state.get('burry_ai_symbol', '') or st.session_state.get('selected_ticker', ''),
                    st.session_state.get('burry_ai_asset_type', 'Azione'),
                    mode=st.session_state.get('model_mode', 'Entrambi')
                )

                st.session_state.burry_ai_history.append({'role': 'user', 'content': burry_ai_prompt})

                # [BUGFIX] Costruzione della memoria contestuale per l'IA (BurryAi Sidebar)
                conv_history = "CRONOLOGIA DELLA CONVERSAZIONE:\n"
                for m in st.session_state.burry_ai_history[:-1]:
                    conv_history += f"[{m['role'].upper()}]: {m['content']}\n"
                enriched_prompt = f"{conv_history}\nDOMANDA ATTUALE: {burry_ai_prompt}"

                with st.chat_message('user'):
                    st.markdown(burry_ai_prompt)

                with st.chat_message('assistant'):
                    with st.spinner('BurryAi sta rispondendo...'):
                        reply = ask_gemini_ticker_chat(
                            ctx,
                            enriched_prompt,
                            mode=st.session_state.get('model_mode', 'Entrambi')
                        )
                    st.markdown(reply)

                st.session_state.burry_ai_history.append({'role': 'assistant', 'content': reply})

    tab_f, tab_t, tab_q, tab_v, tab_p = st.tabs(
        ["📊 FONDAMENTALI", "📉 TECNICO", "⚛️ QUANT", "⚖️ VERDETTO", "📁 PORTAFOGLIO"]
    )

    # Selettore rapido ticker
    with st.expander("🎯 Analisi rapida senza ricerca",
                     expanded=(st.session_state.batch_results is None or st.session_state.batch_results.empty)):
        csel1, csel2, csel3 = st.columns([1.2, 1.2, 1])
        batch_options: List[str] = []
        if st.session_state.batch_results is not None and not st.session_state.batch_results.empty \
           and 'Ticker' in st.session_state.batch_results.columns:
            batch_options = st.session_state.batch_results['Ticker'].dropna().astype(str).tolist()
        portfolio_options = sorted(st.session_state.get('portfolio_tickers', []) or [])

        if batch_options:
            sel_idx = ([''] + batch_options).index(st.session_state.selected_ticker) \
                      if st.session_state.selected_ticker in batch_options else 0
            selected_from_batch = csel1.selectbox(
                'Ticker dai risultati caricati', [''] + batch_options,
                index=sel_idx, key='quick_batch_pick'
            )
            if selected_from_batch:
                st.session_state.selected_ticker = selected_from_batch
                st.session_state.standalone_ticker_input = ''
        else:
            csel1.caption('Nessun batch attivo.')

        if portfolio_options:
            portfolio_pick = csel2.selectbox(
                'Ticker dal portafoglio', [''] + portfolio_options, index=0,
                key='standalone_portfolio_pick'
            )
            if portfolio_pick:
                st.session_state.selected_ticker = None
                st.session_state.standalone_ticker_input = portfolio_pick
        else:
            csel2.caption('Portafoglio vuoto.')

        manual_quick = csel3.text_input(
            'Ticker libero', value=st.session_state.get('standalone_ticker_input', ''),
            key='quick_manual_ticker'
        ).upper().strip()
        if manual_quick:
            st.session_state.selected_ticker = None
            st.session_state.standalone_ticker_input = manual_quick

    ticker, row, standalone_raw_data, analysis_source = resolve_active_analysis_target()
    if not ticker:
        st.info('Puoi usare le tab anche senza ricerca: seleziona un ticker dal portafoglio '
                'oppure inseriscilo nel box "Ticker libero" qui sopra.')

    # ----- TAB FONDAMENTALI -----
    with tab_f:
        if row is None:
            st.info("Nessun ticker attivo. Usa il box 'Analisi rapida senza ricerca'.")
            if st.session_state.batch_results is not None and not st.session_state.batch_results.empty:
                st.dataframe(st.session_state.batch_results.drop(columns=['_raw_data'], errors='ignore'))
        else:
            st.info("💡 **Come leggere questa sezione:** qualita' del business prima, sostenibilita' "
                    "finanziaria poi, prezzo/multipli alla fine.")
            if analysis_source == 'batch' and st.session_state.batch_results is not None \
               and not st.session_state.batch_results.empty:
                st.dataframe(st.session_state.batch_results.drop(columns=['_raw_data'], errors='ignore'))
            elif row is not None:
                st.success(f'Analisi standalone attiva su: {ticker}')
                st.dataframe(pd.DataFrame([dict(row)]).drop(columns=['_raw_data'], errors='ignore'))
        st.markdown("---")
        st.markdown("<p style='text-align:center;color:gray;'>creato e sviluppato da Innovative Program</p>",
                    unsafe_allow_html=True)

    # ----- TAB TECNICO -----
    with tab_t:
        if row is None:
            st.info("Nessun ticker attivo.")
        else:
            st.info("💡 **Come leggere il grafico:** SMA200 = trend di fondo, RSI = momentum.")
            df_tech = get_technical_data(ticker)
            if df_tech is not None:
                df_calc = calculate_technical_indicators(df_tech)
                score, reasons = calculate_timing_score(df_calc, df_calc['Close'].iloc[-1])
                st.metric("Timing Score", f"{score}/100")
                with st.expander("Dettaglio segnali"):
                    for r in reasons:
                        st.write(r)
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
                fig.add_trace(go.Candlestick(
                    x=df_calc.index, open=df_calc['Open'], high=df_calc['High'],
                    low=df_calc['Low'], close=df_calc['Close'], name="Prezzo"
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=df_calc.index, y=df_calc['SMA_200'], name="SMA 200",
                    line=dict(color='blue')
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=df_calc.index, y=df_calc['RSI'], name="RSI",
                    line=dict(color='purple')
                ), row=2, col=1)
                fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_dark")
                st.plotly_chart(fig, width='stretch')
            else:
                st.warning(f'Dati tecnici non disponibili per {ticker}.')
        st.markdown("---")
        st.markdown("<p style='text-align:center;color:gray;'>creato e sviluppato da Innovative Program</p>",
                    unsafe_allow_html=True)

    # ----- TAB QUANT -----
    with tab_q:
        if row is None:
            st.info("Nessun ticker attivo.")
        else:
            st.info("💡 **Quant:** Sharpe (con rf dinamico), Sortino (rischio downside), Calmar "
                    "(CAGR/MaxDD), Altman Z-Score, VaR/CVaR, Monte Carlo bootstrap.")
            df_tech = get_technical_data(ticker)
            if df_tech is not None:
                rf_eff = get_active_risk_free_rate()
                qm = calculate_quant_metrics(
                    df_tech,
                    row.get('_raw_data', standalone_raw_data) if row is not None else standalone_raw_data,
                    risk_free=rf_eff
                )
                risk = calculate_risk_metrics(df_tech)

                c1, c2, c3 = st.columns(3)
                c1.metric(f"Sharpe ({rf_eff*100:.1f}% Rf)", f"{qm['Sharpe Ratio']:.2f}")
                c2.metric("Trend R-Squared", f"{qm['R-Squared']:.2f}")
                z = qm['Altman Z-Score']
                c3.metric("Altman Z-Score", f"{z:.2f}" if isinstance(z, (int, float)) else "N/A")

                c4, c5, c6 = st.columns(3)
                c4.metric("Max Drawdown", f"{risk['Max Drawdown']*100:.1f}%")
                c5.metric("CAGR", f"{risk['CAGR']*100:.1f}%" if not np.isnan(risk['CAGR']) else "N/A")
                c6.metric("VaR 95%", f"{risk['VaR_95']*100:.2f}%")

                # [NEW] Sortino & Calmar
                c7, c8, c9 = st.columns(3)
                c7.metric("Sortino Ratio", f"{risk['Sortino']:.2f}" if not np.isnan(risk['Sortino']) else "N/A")
                c8.metric("Calmar Ratio", f"{risk['Calmar']:.2f}" if not np.isnan(risk['Calmar']) else "N/A")
                c9.metric("CVaR 95%", f"{risk['CVaR_95']*100:.2f}%" if not np.isnan(risk['CVaR_95']) else "N/A")

                df_calc_q = calculate_technical_indicators(df_tech)
                score_q, _ = calculate_timing_score(df_calc_q, df_calc_q['Close'].iloc[-1])
                smart_w = st.session_state.get("smart_weights", DEFAULT_SMART_WEIGHTS)
                smart = compute_smart_quant_score(row, score_q, qm, risk, weights=smart_w)
                st.metric("Smart Quant Score", f"{smart['SmartScore']:.1f}/100")
                st.caption(
                    f"Pesi attivi: F={smart_w['F']:.2f} | T={smart_w['T']:.2f} | Q={smart_w['Q']:.2f}"
                )

                with st.expander("📉 Distribuzione rendimenti & rischio"):
                    st.write(f"Skewness: {risk['Skew']:.2f} | Kurtosis: {risk['Kurt']:.2f}")
                    returns = df_tech['Close'].pct_change().dropna()
                    fig_r = go.Figure()
                    fig_r.add_trace(go.Histogram(x=returns, nbinsx=50, name="Rendimenti giornalieri"))
                    fig_r.update_layout(template="plotly_dark", bargap=0.05)
                    st.plotly_chart(fig_r, width='stretch')

                with st.expander("🎲 Simulazione Monte Carlo"):
                    col_mc1, col_mc2, col_mc3 = st.columns(3)
                    horizon_days = col_mc1.slider("Orizzonte (giorni)", 60, 756, 252, step=21)
                    n_paths = col_mc2.slider("Numero traiettorie", 100, 3000, 1000, step=100)
                    method = col_mc3.selectbox("Metodo", ["IID Bootstrap", "Block Bootstrap"])

                    if method == "Block Bootstrap":
                        mc = monte_carlo_block_bootstrap(df_tech, n_paths, horizon_days)
                    else:
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
                        fig_mc.add_trace(go.Scatter(x=x, y=mc["q50"], name="Mediana",
                                                    line=dict(color="cyan")))
                        fig_mc.add_trace(go.Scatter(x=x, y=mc["q95"], name="95° percentile",
                                                    line=dict(color="green"), opacity=0.3))
                        fig_mc.add_trace(go.Scatter(x=x, y=mc["q05"], name="5° percentile",
                                                    line=dict(color="red"), opacity=0.3,
                                                    fill="tonexty", fillcolor="rgba(255,0,0,0.1)"))
                        fig_mc.update_layout(template="plotly_dark", height=400,
                                             xaxis_title="Giorni", yaxis_title="Equity")
                        st.plotly_chart(fig_mc, width='stretch')
        st.markdown("---")
        st.markdown("<p style='text-align:center;color:gray;'>creato e sviluppato da Innovative Program</p>",
                    unsafe_allow_html=True)

    # ----- TAB VERDETTO -----
    with tab_v:
        if row is None:
            st.info("Nessun ticker attivo.")
        else:
            st.info("💡 **Verdetto:** modello Classico, Evoluto, Personalizzabile + Smart Quant.")
            df_tech = get_technical_data(ticker)
            qm = calculate_quant_metrics(
                df_tech, row.get('_raw_data', standalone_raw_data) if row is not None else standalone_raw_data
            ) if df_tech is not None else {}
            risk = calculate_risk_metrics(df_tech) if df_tech is not None else {
                "Max Drawdown": 0.0, "CAGR": 0.0
            }
            if df_tech is not None:
                df_calc_v = calculate_technical_indicators(df_tech)
                score, reasons = calculate_timing_score(df_calc_v, df_calc_v['Close'].iloc[-1])
            else:
                score = 0
                reasons = []

            # [FIN-FIX] Z-safe: usa is_non_traditional_asset invece di check string indiretto
            z_val = qm.get('Altman Z-Score', 0.0)
            raw_info = (row.get('_raw_data', {}) or {}).get('info', {}) if row is not None else {}
            z_safe = (
                is_non_traditional_asset(ticker, raw_info)
                or (isinstance(z_val, (int, float)) and z_val >= ALTMAN_SAFE_THRESHOLD)
            )

            roic_thr = ui['cfg']['roic'] / 100.0
            peg_thr = ui['cfg']['peg']
            de_thr = ui['cfg']['custom_max_de']
            fcfm_thr = ui['cfg']['custom_min_fcf_margin']
            netm_thr = ui['cfg']['custom_min_net_margin']
            roic_v = row.get('ROIC', 0.0) or 0.0
            peg_v = row.get('PEG Ratio')
            de_v = row.get('Debt/Equity')
            fcfm_v = row.get('FCF Margin')
            netm_v = row.get('Net Margin')

            fund_pts_classic = (
                (1 if roic_v >= roic_thr else 0) +
                (1 if peg_v is not None and peg_v <= peg_thr else 0)
            )
            fund_pts_evoluto = fund_pts_classic + (
                (1 if de_v is not None and de_v <= 1.0 else 0) +
                (1 if fcfm_v is not None and fcfm_v >= 0.08 else 0) +
                (1 if netm_v is not None and netm_v >= 0.10 else 0)
            )
            fund_pts_custom = (
                (1 if roic_v >= roic_thr else 0) +
                (1 if peg_v is not None and peg_v <= peg_thr else 0) +
                (1 if de_v is not None and de_v <= de_thr else 0) +
                (1 if fcfm_v is not None and fcfm_v >= fcfm_thr else 0) +
                (1 if netm_v is not None and netm_v >= netm_thr else 0)
            )

            mode = ui['cfg']['model_mode']

            if mode == "Entrambi":
                col_m1, col_m2 = st.columns(2)
                col_m1.metric("Punti Modello Classico", f"{fund_pts_classic}/2")
                col_m2.metric("Punti Modello Evoluto", f"{fund_pts_evoluto}/5")

            if mode in ["Entrambi", "Evoluto"]:
                st.caption(
                    "Evoluto → "
                    f"ROIC: {'✅' if roic_v >= roic_thr else '❌'} | "
                    f"PEG: {'✅' if (peg_v is not None and peg_v <= peg_thr) else '❌'} | "
                    f"D/E: {'✅' if (de_v is not None and de_v <= 1.0) else '❌'} | "
                    f"FCF Margin: {'✅' if (fcfm_v is not None and fcfm_v >= 0.08) else '❌'} | "
                    f"Net Margin: {'✅' if (netm_v is not None and netm_v >= 0.10) else '❌'}"
                )

            if mode == "Personalizzabile":
                st.caption(
                    "Personalizzabile → "
                    f"ROIC: {'✅' if roic_v >= roic_thr else '❌'} | "
                    f"PEG: {'✅' if (peg_v is not None and peg_v <= peg_thr) else '❌'} | "
                    f"D/E: {'✅' if (de_v is not None and de_v <= de_thr) else '❌'} | "
                    f"FCF Margin: {'✅' if (fcfm_v is not None and fcfm_v >= fcfm_thr) else '❌'} | "
                    f"Net Margin: {'✅' if (netm_v is not None and netm_v >= netm_thr) else '❌'}"
                )

            if mode in ["Entrambi", "Classico"]:
                st.subheader("Modello Classico")
                if fund_pts_classic >= 2 and z_safe and score >= 50:
                    st.success("🟢 BUY: Fondamentali base solidi e timing favorevole.")
                elif fund_pts_classic >= 1 and z_safe:
                    st.warning("🟡 HOLD: Azienda discreta, serve piu' margine di sicurezza.")
                else:
                    st.error("🔴 SELL: Fondamentali o sicurezza finanziaria insufficienti.")

            if mode in ["Entrambi", "Evoluto"]:
                st.subheader("Modello Evoluto")
                if fund_pts_evoluto >= 4 and z_safe and score >= 50:
                    st.success("🟢 BUY: Fondamentali robusti, qualita' finanziaria e timing favorevole.")
                elif fund_pts_evoluto >= 3 and z_safe:
                    st.warning("🟡 HOLD: Azienda interessante, serve conferma.")
                else:
                    st.error("🔴 SELL: Fondamentali insufficienti o profilo rischio/rendimento debole.")

            if mode == "Personalizzabile":
                st.subheader("Modello Personalizzabile")
                if fund_pts_custom >= 4 and z_safe and score >= 50:
                    st.success("🟢 BUY: Criteri personalizzati soddisfatti e timing favorevole.")
                elif fund_pts_custom >= 3 and z_safe:
                    st.warning("🟡 HOLD: Setup discreto secondo i parametri personalizzati.")
                else:
                    st.error("🔴 SELL: Il titolo non soddisfa i criteri personalizzati.")

            st.markdown('---')
            st.subheader('Spiegazione AI')
            ai_context = build_ai_context_for_ticker(ticker, row, qm, risk, score, reasons, mode)
            st.session_state.burry_ai_live_context[ticker] = ai_context

            if st.session_state.get('ai_ticker_chat_last_symbol') != ticker:
                st.session_state.ai_ticker_chat_history = []
                st.session_state.ai_ticker_chat_last_symbol = ticker

            if st.button('🧠 Spiega con AI', key=f'ai_explain_{ticker}'):
                with st.spinner('Analisi AI in corso...'):
                    ai_answer = ask_gemini_ticker_chat(
                        ai_context,
                        'Spiegami questo titolo come un analista buy-side prudente, coerente con il verdetto mostrato.',
                        mode=mode,
                    )
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
                
                # [BUGFIX] Costruzione della memoria contestuale per l'IA (Ticker Chat principale)
                conv_history = "CRONOLOGIA DELLA CONVERSAZIONE:\n"
                for m in st.session_state.ai_ticker_chat_history[:-1]:
                    conv_history += f"[{m['role'].upper()}]: {m['content']}\n"
                enriched_prompt = f"{conv_history}\nDOMANDA ATTUALE: {ai_user_prompt}"

                with st.chat_message('user'):
                    st.markdown(ai_user_prompt)
                with st.chat_message('assistant'):
                    with st.spinner("L'AI sta rispondendo..."):
                        ai_reply = ask_gemini_ticker_chat(ai_context, enriched_prompt, mode=mode)
                    st.markdown(ai_reply)
                st.session_state.ai_ticker_chat_history.append({'role': 'assistant', 'content': ai_reply})

            if df_tech is not None and qm:
                smart_w = st.session_state.get("smart_weights", DEFAULT_SMART_WEIGHTS)
                smart_v = compute_smart_quant_score(row, score, qm, risk, weights=smart_w)
                if mode in ["Entrambi", "Evoluto"]:
                    st.metric("Smart Quant Score", f"{smart_v['SmartScore']:.1f}/100")
                    st.write(
                        f"F: {smart_v['FundamentalScore']:.0f} | "
                        f"T: {smart_v['TechnicalScore']:.0f} | "
                        f"Q: {smart_v['QuantRiskScore']:.0f}"
                    )
                    if smart_v["SmartScore"] >= 70 and z_safe and fund_pts_evoluto >= 4:
                        st.success("🟢 BUY (Quant): Vantaggio statistico, fondamentali e rischio OK.")
                    elif smart_v["SmartScore"] >= 50 and z_safe and fund_pts_evoluto >= 3:
                        st.warning("🟡 HOLD (Quant): Setup discreto.")
                    else:
                        st.error("🔴 NO TRADE (Quant): Vantaggio quantitativo debole.")

                if mode == "Personalizzabile":
                    st.metric("Smart Quant Score", f"{smart_v['SmartScore']:.1f}/100")
                    st.write(
                        f"F: {smart_v['FundamentalScore']:.0f} | "
                        f"T: {smart_v['TechnicalScore']:.0f} | "
                        f"Q: {smart_v['QuantRiskScore']:.0f}"
                    )
                    if smart_v["SmartScore"] >= 70 and z_safe and fund_pts_custom >= 4:
                        st.success("🟢 BUY (Quant): Vantaggio statistico e criteri personalizzati OK.")
                    elif smart_v["SmartScore"] >= 50 and z_safe and fund_pts_custom >= 3:
                        st.warning("🟡 HOLD (Quant): Setup discreto.")
                    else:
                        st.error("🔴 NO TRADE (Quant): Score o criteri insufficienti.")

        st.markdown("---")
        st.markdown("<p style='text-align:center;color:gray;'>creato e sviluppato da Innovative Program</p>",
                    unsafe_allow_html=True)

    # ----- TAB PORTAFOGLIO -----
    with tab_p:
        st.info("💡 **Cabina di controllo del portafoglio reale.** Posizioni con quantita',"
                " PMC, valuta. Calcoli FX-aware, fiscalita' con compensazione minusvalenze,"
                " concentrazione, ribilanciamento.")

        base_currency = ui.get("base_currency") or st.session_state.get("base_currency", "EUR")
        st.success(f"Valuta base portafoglio: **{base_currency}** | "
                   f"Conversione FX reale attiva.")

        all_tickers_batch: List[str] = []
        if st.session_state.batch_results is not None and not st.session_state.batch_results.empty:
            all_tickers_batch = st.session_state.batch_results["Ticker"].tolist()

        if all_tickers_batch:
            st.markdown("#### Seleziona dal batch analizzato")
            default_batch = [t for t in st.session_state.portfolio_tickers if t in all_tickers_batch]
            selected_from_batch = st.multiselect(
                "Titoli da includere nel portafoglio (da batch)",
                all_tickers_batch, default=default_batch
            )
        else:
            selected_from_batch = []

        st.markdown("#### Aggiungi manualmente altri ticker")
        manual_ticker = st.text_input(
            "Ticker (incluso suffisso mercato, es. STLAM.MI, BMW.DE, AIR.PA, ULVR.L)", ""
        )
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
                            save_user_portfolio_position(
                                t_clean,
                                st.session_state.holdings_quantity[t_clean],
                                st.session_state.holdings_pmc[t_clean],
                                st.session_state.holdings_currency[t_clean]
                            )
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
                    min_value=0.0, value=default_qty, step=0.01, format="%.4f",
                    key=f"holding_qty_{t}"
                )
                pmc = col.number_input(
                    f"{t} - PMC", min_value=0.0, value=default_pmc, step=0.01, format="%.4f",
                    key=f"holding_pmc_{t}"
                )
                cur_default = st.session_state.holdings_currency.get(t, "USD")
                cur_options = ["USD", "EUR", "GBP", "CHF", "JPY", "CAD", "AUD"]
                if cur_default not in cur_options:
                    cur_options = [cur_default] + cur_options
                cur = col.selectbox(
                    f"{t} - Valuta posizione",
                    cur_options,
                    index=cur_options.index(cur_default) if cur_default in cur_options else 0,
                    key=f"currency_{t}"
                )
                st.session_state.holdings_currency[t] = cur

                derived = calculate_position_from_quantity(t, qty, pmc, user_currency=cur) \
                    if qty > 0 and pmc > 0 else {
                        'Importo Investito': 0.0, 'Prezzo Attuale': np.nan,
                        'Valore di Mercato': 0.0, 'P&L': 0.0, 'P&L %': 0.0,
                        'Valuta Nativa': cur, 'FX Native->User': 1.0,
                    }
                holdings_quantity[t] = qty
                holdings_pmc[t] = pmc
                holdings[t] = float(derived['Importo Investito'])

                if is_authenticated():
                    save_user_portfolio_position(t, qty, pmc, cur)

                price_text = "N/D" if pd.isna(derived['Prezzo Attuale']) else f"{derived['Prezzo Attuale']:.2f}"
                native_cur = derived.get('Valuta Nativa', cur)
                fx_used = derived.get('FX Native->User', 1.0)
                col.caption(
                    f"Prezzo (in {cur}): {price_text} | Investito: {derived['Importo Investito']:.2f} | "
                    f"Valore: {derived['Valore di Mercato']:.2f} | P&L: {derived['P&L']:.2f} ({derived['P&L %']:.2f}%)"
                )
                if native_cur and native_cur != cur:
                    col.caption(f"⚠️ Valuta nativa: {native_cur}, FX applicato: {fx_used:.4f}")

                if col.button("🗑 Rimuovi", key=f"remove_{t}"):
                    if t in st.session_state.portfolio_tickers:
                        st.session_state.portfolio_tickers = [
                            x for x in st.session_state.portfolio_tickers if x != t
                        ]
                    for d in [st.session_state.holdings, st.session_state.holdings_currency,
                              st.session_state.holdings_quantity, st.session_state.holdings_pmc]:
                        d.pop(t, None)
                    if is_authenticated():
                        delete_user_portfolio_position(t)
                    st.rerun()

            st.session_state.holdings = holdings
            st.session_state.holdings_quantity = holdings_quantity
            st.session_state.holdings_pmc = holdings_pmc

            if st.button("📊 Calcola pesi e analisi del portafoglio"):
                positive_holdings = {t: a for t, a in holdings.items() if a > 0}
                if not positive_holdings:
                    st.error("Imposta quantita' e PMC > 0 almeno per un titolo.")
                else:
                    tot = sum(positive_holdings.values())
                    weights_pct = {t: a / tot * 100.0 for t, a in positive_holdings.items()}
                    built = build_portfolio_returns(list(positive_holdings.keys()), weights_pct)
                    if built is None:
                        st.error("Impossibile costruire la serie dei rendimenti "
                                 "(dati insufficienti).")
                    else:
                        df_rets, port_ret = built
                        pm = calculate_portfolio_metrics(port_ret)

                        # Tabella posizioni con FX-aware
                        st.markdown("#### Dettaglio posizioni e pesi")
                        rows_pos = []
                        total_market_value = 0.0
                        for t in weights_pct.keys():
                            qty = holdings_quantity.get(t, 0.0)
                            pmc = holdings_pmc.get(t, 0.0)
                            cur = st.session_state.holdings_currency.get(t, "USD")
                            derived = calculate_position_from_quantity(t, qty, pmc, user_currency=cur)
                            total_market_value += derived['Valore di Mercato']
                            rows_pos.append({
                                "Ticker": t, "Quantità": qty, "PMC": pmc,
                                "Prezzo Attuale": derived['Prezzo Attuale'],
                                "Importo Investito": derived['Importo Investito'],
                                "Valore di Mercato": derived['Valore di Mercato'],
                                "P&L": derived['P&L'],
                                "P&L %": derived['P&L %'],
                                "Peso %": weights_pct[t],
                                "Valuta": cur,
                            })
                        df_weights = pd.DataFrame(rows_pos)
                        st.dataframe(df_weights, width='stretch')

                        # [NEW] Export CSV
                        csv_bytes = _portfolio_export_csv(df_weights)
                        st.download_button(
                            "💾 Scarica portafoglio (CSV)",
                            data=csv_bytes,
                            file_name="portfolio_burry.csv",
                            mime="text/csv"
                        )

                        # [BUGFIX] enrich_portfolio_with_fx ora effettivamente usata
                        df_weights_base = enrich_portfolio_with_fx(df_weights, base_currency=base_currency)

                        st.markdown(f"#### Portafoglio in valuta base ({base_currency})")
                        st.dataframe(df_weights_base, width='stretch')

                        # ----- Analisi fiscale standard -----
                        st.markdown("#### Analisi fiscale teorica")
                        tax_rate_input = st.slider(
                            "Aliquota fiscale (%)", 0.0, 50.0,
                            value=float(DEFAULT_TAX_RATE * 100.0), step=1.0,
                            key="portfolio_tax_rate_slider"
                        ) / 100.0

                        df_tax = calculate_tax_impact(df_weights, tax_rate=tax_rate_input)
                        if not df_tax.empty:
                            st.dataframe(df_tax, width='stretch')
                            total_tax = float(df_tax["Imposta Teorica"].sum())
                            total_net_pnl = float(df_tax["Plus/Minus Netta"].sum())
                            tax_c1, tax_c2 = st.columns(2)
                            tax_c1.metric("Imposta teorica (no compensazione)", f"{total_tax:,.2f}")
                            tax_c2.metric("P&L netto", f"{total_net_pnl:,.2f}")

                        # [NEW] Compensazione minusvalenze
                        st.markdown("#### Compensazione fiscale (minusvalenze)")
                        df_tax_comp, summary_comp = calculate_tax_with_loss_offset(
                            df_weights_base, tax_rate=tax_rate_input
                        )
                        if summary_comp:
                            ccol1, ccol2, ccol3 = st.columns(3)
                            ccol1.metric("Plusvalenze tot.", f"{summary_comp['Plusvalenze totali']:,.2f}")
                            ccol2.metric("Minusvalenze tot.", f"{summary_comp['Minusvalenze totali']:,.2f}")
                            ccol3.metric("Risparmio fiscale", f"{summary_comp['Risparmio fiscale da compensazione']:,.2f}")
                            st.caption(
                                f"In Italia le minusvalenze realizzate possono essere usate per ridurre "
                                f"l'imponibile sulle plusvalenze entro {TAX_LOSS_COMPENSATION_YEARS} anni "
                                f"(art. 68 TUIR). Imposta teorica netta: "
                                f"{summary_comp['Imposta teorica netta']:,.2f} {base_currency}."
                            )

                        # ----- Allocazione -----
                        st.markdown("#### Allocazione del portafoglio")
                        df_alloc = build_portfolio_allocation_df(
                            positive_holdings, st.session_state.holdings_currency
                        )
                        if not df_alloc.empty:
                            st.dataframe(df_alloc, width='stretch')
                            col_a1, col_a2, col_a3 = st.columns(3)
                            with col_a1:
                                st.markdown("**Per Asset Class**")
                                df_asset = summarize_group_weights(df_alloc, "Asset Class")
                                st.dataframe(df_asset, width='stretch')
                            with col_a2:
                                st.markdown("**Per Geografia**")
                                df_geo = summarize_group_weights(df_alloc, "Geografia")
                                st.dataframe(df_geo, width='stretch')
                            with col_a3:
                                st.markdown("**Per Valuta**")
                                df_cur = summarize_group_weights(df_alloc, "Valuta")
                                st.dataframe(df_cur, width='stretch')

                            # [NEW] Concentrazione & diversificazione
                            st.markdown("#### Concentrazione e diversificazione")
                            conc = calculate_concentration_metrics(weights_pct)
                            ccc1, ccc2, ccc3, ccc4 = st.columns(4)
                            ccc1.metric("HHI", f"{conc['HHI']:.3f}")
                            ccc2.metric("Numero effettivo titoli", f"{conc['ENS']:.1f}")
                            ccc3.metric("Top 1 %", f"{conc['Top1 %']:.1f}%")
                            ccc4.metric("Top 3 %", f"{conc['Top3 %']:.1f}%")
                            st.caption(
                                "HHI < 0.10 → portafoglio molto diversificato; HHI > 0.25 → "
                                "concentrazione elevata. ENS = 1/HHI rappresenta il numero "
                                "effettivo di titoli equivalenti equipesati."
                            )

                            # ----- Ribilanciamento -----
                            st.markdown("#### Ribilanciamento automatico")
                            rebalance_mode = st.radio(
                                "Livello target",
                                ["Ticker", "Asset Class", "Geografia", "Valuta"],
                                horizontal=True, key="rebalance_mode_radio"
                            )
                            st.session_state.portfolio_target_mode = rebalance_mode

                            mapping = {
                                "Ticker": ("Ticker", df_alloc[["Ticker", "Peso %"]].copy()),
                                "Asset Class": ("Asset Class", df_asset[["Asset Class", "Peso %"]].copy()),
                                "Geografia": ("Geografia", df_geo[["Geografia", "Peso %"]].copy()),
                                "Valuta": ("Valuta", df_cur[["Valuta", "Peso %"]].copy()),
                            }
                            label_col, current_target_df = mapping[rebalance_mode]

                            target_inputs = {}
                            cols_target = st.columns(3)
                            for j, rec in enumerate(current_target_df.to_dict("records")):
                                col_t = cols_target[j % 3]
                                label = rec[label_col]
                                current_weight = float(rec["Peso %"])
                                key_target = f"{rebalance_mode}::{label}"
                                default_target = float(
                                    st.session_state.portfolio_targets.get(key_target, current_weight)
                                )
                                target_val = col_t.number_input(
                                    f"Target {label}", min_value=0.0, max_value=100.0,
                                    value=default_target, step=1.0,
                                    key=f"target_{rebalance_mode}_{label}"
                                )
                                target_inputs[label] = target_val
                                st.session_state.portfolio_targets[key_target] = target_val

                            tolerance_pct = st.slider(
                                "Tolleranza ribilanciamento (%)", 0.0, 10.0, 1.0, step=0.5
                            )
                            target_sum = sum(target_inputs.values())
                            normalized_targets = (
                                {k: v / target_sum * 100.0 for k, v in target_inputs.items()}
                                if target_sum > 0 else target_inputs
                            )
                            rebalance_df = compute_rebalancing_actions(
                                df_alloc=df_alloc, target_weights=normalized_targets,
                                group_col=label_col, tolerance_pct=tolerance_pct
                            )
                            if not rebalance_df.empty:
                                st.markdown("##### Azioni suggerite")
                                st.dataframe(rebalance_df, width='stretch')
                                fig_reb = go.Figure()
                                fig_reb.add_trace(go.Bar(
                                    x=rebalance_df[label_col],
                                    y=rebalance_df["Peso %"], name="Peso attuale %"
                                ))
                                fig_reb.add_trace(go.Bar(
                                    x=rebalance_df[label_col],
                                    y=rebalance_df["Target %"], name="Target %"
                                ))
                                fig_reb.update_layout(
                                    barmode="group", template="plotly_dark", height=420,
                                    xaxis_title=label_col, yaxis_title="Peso %"
                                )
                                st.plotly_chart(fig_reb, width='stretch')

                        # ----- Totali in valuta base -----
                        if not df_weights_base.empty:
                            total_invested_base = float(df_weights_base["Importo Investito Base"].sum())
                            total_value_base = float(df_weights_base["Valore di Mercato Base"].sum())
                            total_pnl_base = float(df_weights_base["P&L Base"].sum())
                            total_pnl_pct_base = (
                                total_pnl_base / total_invested_base * 100.0
                                if total_invested_base > 0 else 0.0
                            )
                            ctot1, ctot2, ctot3, ctot4 = st.columns(4)
                            ctot1.metric(f"Investito ({base_currency})", f"{total_invested_base:,.2f}")
                            ctot2.metric(f"Valore ({base_currency})", f"{total_value_base:,.2f}")
                            ctot3.metric(f"P&L ({base_currency})", f"{total_pnl_base:,.2f}")
                            ctot4.metric("Rend. totale", f"{total_pnl_pct_base:.2f}%")

                        # ----- Metriche di portafoglio -----
                        cpa, cpv, cps, cpdd = st.columns(4)
                        cpa.metric("Rend. annuo atteso", f"{pm['AnnRet']*100:.2f}%")
                        cpv.metric("Volatilita' annua", f"{pm['AnnVol']*100:.2f}%")
                        cps.metric("Sharpe portafoglio", f"{pm['Sharpe']:.2f}")
                        cpdd.metric("Max Drawdown", f"{pm['MaxDD']*100:.1f}%")

                        cps2, cps3, cps4 = st.columns(3)
                        cps2.metric("Sortino", f"{pm['Sortino']:.2f}" if not np.isnan(pm['Sortino']) else "N/A")
                        cps3.metric("Calmar", f"{pm['Calmar']:.2f}" if not np.isnan(pm['Calmar']) else "N/A")
                        cps4.metric("CAGR portafoglio", f"{pm['CAGR']*100:.1f}%" if not np.isnan(pm['CAGR']) else "N/A")

                        # [NEW] Beta vs benchmark
                        st.markdown("#### Esposizione di mercato (vs S&P 500)")
                        beta_metrics = calculate_portfolio_beta(port_ret, benchmark_symbol=DEFAULT_BENCHMARK)
                        bcol1, bcol2, bcol3 = st.columns(3)
                        bcol1.metric("Beta", f"{beta_metrics['Beta']:.2f}" if not np.isnan(beta_metrics['Beta']) else "N/A")
                        bcol2.metric("Alpha (annuo)", f"{beta_metrics['Alpha (ann.)']*100:.2f}%"
                                     if not np.isnan(beta_metrics['Alpha (ann.)']) else "N/A")
                        bcol3.metric("Correlazione", f"{beta_metrics['Corr']:.2f}"
                                     if not np.isnan(beta_metrics['Corr']) else "N/A")

                        # ----- Equity curve -----
                        equity_p = (1 + port_ret).cumprod()
                        fig_p = go.Figure()
                        fig_p.add_trace(go.Scatter(
                            x=equity_p.index, y=equity_p.values, name="Equity portafoglio"
                        ))
                        fig_p.update_layout(
                            template="plotly_dark", height=400,
                            xaxis_title="Data", yaxis_title="Equity normalizzata"
                        )
                        st.plotly_chart(fig_p, width='stretch')

                        # ----- Correlazioni -----
                        corr = df_rets.corr()
                        st.markdown("#### Correlazioni tra titoli in portafoglio")
                        st.dataframe(corr.style.background_gradient(cmap="RdYlGn", axis=None))
        else:
            st.info("Seleziona almeno un titolo dal batch o aggiungilo manualmente.")

        st.markdown("---")
        st.markdown("Creato e sviluppato da Innovative Program", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
  
