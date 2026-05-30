# rate_limiter.py
import time
import threading
from typing import Dict, Optional


class TokenBucket:
    """
    Implementazione del token bucket per rate limiting.
    Ogni bucket ha una capacità e un tasso di refill (token al secondo).
    Thread-safe.
    """

    def __init__(self, capacity: int, refill_rate: float, name: str = "default"):
        """
        Args:
            capacity: Numero massimo di token nel bucket.
            refill_rate: Token aggiunti al secondo (es. 1.0 = 1 token/sec).
            name: Nome identificativo per il bucket (utile per debug).
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.name = name
        self._tokens = capacity
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self):
        """Ricalcola i token in base al tempo passato."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        if elapsed > 0:
            new_tokens = elapsed * self.refill_rate
            self._tokens = min(self.capacity, self._tokens + new_tokens)
            self._last_refill = now

    def consume(self, tokens: int = 1) -> bool:
        """
        Consuma un numero di token.
        Restituisce True se il consumo è consentito, False altrimenti.
        """
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def wait_and_consume(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """
        Attende finché non ci sono abbastanza token, poi consuma.
        Restituisce True se riuscito, False se timeout scattato.
        """
        start = time.monotonic()
        while True:
            if self.consume(tokens):
                return True
            if timeout is not None and (time.monotonic() - start) >= timeout:
                return False
            # Attendiamo un breve intervallo prima di riprovare
            time.sleep(0.05)

    def get_available_tokens(self) -> float:
        """Restituisce il numero attuale di token (approssimativo)."""
        with self._lock:
            self._refill()
            return self._tokens


class RateLimiterRegistry:
    """
    Registry centralizzato per gestire più bucket (uno per fonte dati).
    Singleton thread-safe.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._buckets = {}
        return cls._instance

    def register_bucket(self, name: str, capacity: int, refill_rate: float) -> TokenBucket:
        """Crea e registra un nuovo bucket. Se già esiste, lo restituisce."""
        with self._lock:
            if name not in self._buckets:
                self._buckets[name] = TokenBucket(capacity, refill_rate, name)
            return self._buckets[name]

    def get_bucket(self, name: str) -> Optional[TokenBucket]:
        """Recupera un bucket esistente per nome."""
        return self._buckets.get(name)

    def consume(self, source: str, tokens: int = 1) -> bool:
        """Consuma token per una fonte registrata. Restituisce True se consentito."""
        bucket = self.get_bucket(source)
        if bucket is None:
            # Se non registrato, crea un bucket di default molto permissivo (100 req/sec)
            bucket = self.register_bucket(source, 100, 100.0)
        return bucket.consume(tokens)

    def wait_and_consume(self, source: str, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """Attende e consuma token per una fonte."""
        bucket = self.get_bucket(source)
        if bucket is None:
            bucket = self.register_bucket(source, 100, 100.0)
        return bucket.wait_and_consume(tokens, timeout)


# Istanza globale del registry (singleton)
_registry = RateLimiterRegistry()

def get_rate_limiter() -> RateLimiterRegistry:
    """Restituisce l'istanza globale del rate limiter registry."""
    return _registry


# ============================================================================
# Configurazioni predefinite per le varie fonti API (esempi)
# ============================================================================
DEFAULT_API_LIMITS = {
    "yfinance": {"capacity": 30, "refill_rate": 2.0},      # 2 richieste/sec, max 30 burst
    "polygon": {"capacity": 5, "refill_rate": 0.5},        # 0.5 req/sec (1 ogni 2 sec)
    "fmp": {"capacity": 10, "refill_rate": 1.0},           # 1 req/sec
    "alpha_vantage": {"capacity": 5, "refill_rate": 0.333}, # 1 ogni 3 secondi
    "yahooquery": {"capacity": 20, "refill_rate": 1.0},     # 1 req/sec
    "marketwatch": {"capacity": 10, "refill_rate": 0.5},    # scraping gentile
    "finviz": {"capacity": 10, "refill_rate": 0.5},
    "cboe": {"capacity": 10, "refill_rate": 0.5},
    "borsa_italiana": {"capacity": 10, "refill_rate": 0.5},
    "lse": {"capacity": 10, "refill_rate": 0.5},
    "openfigi": {"capacity": 10, "refill_rate": 1.0},
}

def register_default_limits():
    """Registra i limiti predefiniti per tutte le fonti note."""
    limiter = get_rate_limiter()
    for source, limits in DEFAULT_API_LIMITS.items():
        limiter.register_bucket(source, limits["capacity"], limits["refill_rate"])

# Opzionale: registra automaticamente all'import
register_default_limits()