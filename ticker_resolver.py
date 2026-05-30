# ticker_resolver.py
import re
import logging
import time
from typing import Dict, Optional, Tuple

import requests
import yfinance as yf

from rate_limiter import get_rate_limiter

logger = logging.getLogger("VQuantPro")

# ------------------------------------------------------------------
# Mappe statiche per risoluzione veloce (simboli comuni italiani/europei)
# ------------------------------------------------------------------
MANUAL_TICKER_MAP = {
    "ENI": "ENI.MI",
    "STLAM": "STLAM.MI",
    "STLA": "STLAM.MI",
    "FERRARI": "RACE.MI",
    "UNICREDIT": "UCG.MI",
    "INTESA": "ISP.MI",
    "GENERALI": "G.MI",
    "ENEL": "ENEL.MI",
    "TELECOM": "TIT.MI",
    "CNH": "CNHI.MI",
    "PRADA": "1913.HK",
    "MONCLER": "MONC.MI",
    "NEXI": "NEXI.MI",
    "DIASORIN": "DIA.MI",
    "RECORDATI": "REC.MI",
    "TENARIS": "TEN.MI",
    "SAIPEM": "SPM.MI",
    "BPER": "BPE.MI",
    "BANCO BPM": "BAMI.MI",
    "POP SONDRIO": "BPSO.MI",
    "MEDIOBANCA": "MB.MI",
    "FINCANTIERI": "FCT.MI",
    "LEONARDO": "LDO.MI",
    "PIRELLI": "PIRC.MI",
    "ITALGAS": "IG.MI",
    "HERA": "HER.MI",
    "ACEA": "ACE.MI",
    "A2A": "A2A.MI",
    "IREN": "IRE.MI",
    "Terna": "TRN.MI",
    "SNAM": "SRG.MI",
    "POSTE": "PST.MI",
    "XDWU": "XDWU.DE",
    "BMW": "BMW.DE",
    "AIR": "AIR.PA",
    "VOW": "VOW3.DE",
    "DAI": "DAI.DE",
    "SIEMENS": "SIE.DE",
    "SAP": "SAP.DE",
    "ADIDAS": "ADS.DE",
    "BASF": "BAS.DE",
    "BAYER": "BAYN.DE",
    "MERCEDES": "MBG.DE",
    "VOLKSWAGEN": "VOW3.DE",
    "ALLIANZ": "ALV.DE",
    "DEUTSCHE BANK": "DBK.DE",
    "TELEKOM": "DTE.DE",
    "LVMH": "MC.PA",
    "LOREAL": "OR.PA",
    "TOTAL": "TTE.PA",
    "SANOFI": "SAN.PA",
    "AIRBUS": "AIR.PA",
    "BNP": "BNP.PA",
    "SOCIETE GENERALE": "GLE.PA",
    "UNILEVER": "ULVR.L",
    "HSBC": "HSBA.L",
    "BP": "BP.L",
    "SHELL": "SHEL.L",
    "RIO TINTO": "RIO.L",
    "ASTRAZENECA": "AZN.L",
    "GSK": "GSK.L",
    "BARCLAYS": "BARC.L",
    "LLOYDS": "LLOY.L",
    "NATIONAL GRID": "NG.L",
}

# Suffissi da provare in ordine (i più comuni per primi)
COMMON_SUFFIXES = [
    "", ".MI", ".DE", ".PA", ".L", ".TO", ".T", ".HK", ".AX", ".NS",
    ".SW", ".MC", ".BR", ".MX", ".SA", ".BO", ".KS", ".SS", ".SZ",
    "-USD", "-EUR", ".CO", ".HE", ".OL", ".VI", ".IR", ".V", ".NZ"
]

# ------------------------------------------------------------------
# Test funzioni con rate limiting
# ------------------------------------------------------------------
def _test_yfinance(symbol: str) -> bool:
    """Verifica se il simbolo è valido su yfinance (con rate limiting)."""
    try:
        limiter = get_rate_limiter()
        limiter.wait_and_consume("yfinance", timeout=5.0)
        ticker = yf.Ticker(symbol)
        info = ticker.info
        if info and ('symbol' in info or 'regularMarketPrice' in info):
            return True
        return False
    except Exception:
        return False

def _test_polygon(symbol: str, api_key: Optional[str] = None) -> bool:
    """Verifica se il simbolo è valido su Polygon (con rate limiting)."""
    import os
    from cache_manager import get_cache_manager  # evitiamo circolarità, import dinamico
    from burry_ai_prompts import safe_get_secret

    if '.' in symbol:
        return False
    api_key = api_key or safe_get_secret("POLYGON_API_KEY")
    if not api_key:
        return False
    try:
        limiter = get_rate_limiter()
        limiter.wait_and_consume("polygon", timeout=5.0)
        url = f"https://api.polygon.io/v3/reference/tickers/{symbol}?apiKey={api_key}"
        resp = requests.get(url, timeout=5)
        return resp.status_code == 200 and resp.json().get('status') == 'OK'
    except Exception:
        return False

def _test_openfigi(symbol: str) -> Optional[str]:
    """
    Tenta risoluzione tramite OpenFIGI API (gratuita, nessuna chiave).
    Restituisce il simbolo corretto se trovato, altrimenti None.
    """
    try:
        limiter = get_rate_limiter()
        limiter.wait_and_consume("openfigi", timeout=5.0)
        url = "https://api.openfigi.com/v3/mapping"
        headers = {"Content-Type": "application/json"}
        payload = [{"idType": "TICKER", "idValue": symbol.upper(), "marketSecDes": "Equity"}]
        resp = requests.post(url, json=payload, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data and len(data) > 0:
                result = data[0].get('data', [])
                if result and len(result) > 0:
                    # Prendi il primo mapping
                    figi_info = result[0]
                    ticker = figi_info.get('ticker')
                    exchange = figi_info.get('exchCode')
                    if ticker and exchange:
                        # Costruisci simbolo completo (es. "AAPL" per US, ma alcuni exchange hanno suffisso)
                        # Per semplicità restituiamo il ticker + eventuale suffix noto
                        # Mappa exchange -> suffix
                        exchange_suffix = {
                            "MI": ".MI", "DE": ".DE", "PA": ".PA", "L": ".L", "TO": ".TO",
                            "T": ".T", "HK": ".HK", "AX": ".AX", "NS": ".NS", "SW": ".SW",
                            "MC": ".MC", "BR": ".BR", "MX": ".MX", "SA": ".SA", "BO": ".BO",
                            "KS": ".KS", "SS": ".SS", "SZ": ".SZ"
                        }
                        suffix = exchange_suffix.get(exchange, "")
                        return ticker + suffix
        return None
    except Exception as e:
        logger.debug(f"OpenFIGI error: {e}")
        return None

# ------------------------------------------------------------------
# Resolver principale
# ------------------------------------------------------------------
class TickerResolver:
    """
    Risolve un ticker generico nel simbolo corretto per yfinance/Polygon/API.
    Utilizza cache in memoria (session state) e fallback multipli.
    """

    def __init__(self):
        # Cache in memoria per questa sessione (opzionale, oltre a cache_manager)
        self._cache = {}

    def resolve(self, symbol: str, force_refresh: bool = False) -> str:
        """
        Risolve il ticker. Restituisce il simbolo più probabile.
        """
        symbol_clean = symbol.upper().strip()
        if not symbol_clean:
            return symbol_clean

        # Cache in memoria
        if not force_refresh and symbol_clean in self._cache:
            return self._cache[symbol_clean]

        # 1. Mappa manuale
        if symbol_clean in MANUAL_TICKER_MAP:
            resolved = MANUAL_TICKER_MAP[symbol_clean]
            logger.info(f"Mappa manuale: {symbol_clean} -> {resolved}")
            self._cache[symbol_clean] = resolved
            return resolved

        # 2. OpenFIGI (chiamata esterna, richiede rate limiting)
        try:
            figi_resolved = _test_openfigi(symbol_clean)
            if figi_resolved:
                logger.info(f"OpenFIGI risolve {symbol_clean} -> {figi_resolved}")
                self._cache[symbol_clean] = figi_resolved
                return figi_resolved
        except Exception as e:
            logger.debug(f"OpenFIGI fallito: {e}")

        # 3. Test suffissi con yfinance e Polygon
        for suffix in COMMON_SUFFIXES:
            candidate = symbol_clean + suffix
            # Usa yfinance
            if _test_yfinance(candidate):
                logger.info(f"yfinance risolve {symbol_clean} -> {candidate}")
                self._cache[symbol_clean] = candidate
                return candidate
            # Usa Polygon per suffissi di mercati supportati (senza punto iniziale)
            if suffix in [".MI", ".DE", ".PA", ".L", ".TO", ".T", ".HK", ".AX", ".NS"]:
                if _test_polygon(candidate):
                    logger.info(f"Polygon risolve {symbol_clean} -> {candidate}")
                    self._cache[symbol_clean] = candidate
                    return candidate

        # Fallback: restituisci l'originale (forse funzionerà lo stesso)
        logger.warning(f"Nessuna risoluzione per {symbol_clean}, uso originale")
        self._cache[symbol_clean] = symbol_clean
        return symbol_clean

    def invalidate_cache(self, symbol: str):
        """Rimuove un simbolo dalla cache in memoria."""
        self._cache.pop(symbol.upper(), None)


# Istanza singleton
_resolver = None

def get_ticker_resolver() -> TickerResolver:
    global _resolver
    if _resolver is None:
        _resolver = TickerResolver()
    return _resolver


# ------------------------------------------------------------------
# Funzione helper per compatibilità con il codice esistente
# (sostituisce la vecchia auto_resolve_ticker_adaptive)
# ------------------------------------------------------------------
def auto_resolve_ticker_adaptive(symbol: str, force_refresh: bool = False) -> str:
    """
    Funzione compatibile con il vecchio nome, ma usa il nuovo resolver.
    """
    resolver = get_ticker_resolver()
    return resolver.resolve(symbol, force_refresh)