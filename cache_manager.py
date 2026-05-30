# cache_manager.py
import sqlite3
import json
import threading
import time
from typing import Any, Dict, Optional
from contextlib import contextmanager

import pandas as pd

# Percorso del database (nella stessa cartella dell'app)
CACHE_DB_PATH = "vquant_cache.db"

# TTL predefiniti (secondi)
TTL_FUNDAMENTAL = 86400      # 24 ore
TTL_TECHNICAL = 900          # 15 minuti


class CacheManager:
    """Gestione cache SQLite persistente con TTL differenziati per fondamentale/tecnico."""

    def __init__(self, db_path: str = CACHE_DB_PATH):
        self.db_path = db_path
        self._local = threading.local()
        self._init_db()

    @contextmanager
    def _get_connection(self):
        """Context manager per connessione thread-safe con WAL."""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        """Crea le tabelle se non esistono."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Tabella per dati fondamentali
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS fundamental_cache (
                    symbol TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    ttl INTEGER NOT NULL
                )
            ''')
            # Tabella per dati tecnici (serie di prezzi)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS technical_cache (
                    symbol TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    ttl INTEGER NOT NULL
                )
            ''')
            conn.commit()

    def _is_expired(self, timestamp: float, ttl: int) -> bool:
        """Controlla se un record è scaduto."""
        return (time.time() - timestamp) > ttl

    def _serialize_data(self, data: Any) -> str:
        """Serializza un oggetto (dict, DataFrame, list) in JSON string."""
        if isinstance(data, pd.DataFrame):
            # Converti DataFrame in dict con orient='split' per preservare indici e colonne
            return data.to_json(date_format='iso', orient='split')
        elif isinstance(data, (dict, list)):
            return json.dumps(data, default=str)
        else:
            # Fallback: converti in stringa
            return json.dumps({"value": str(data)})

    def _deserialize_data(self, data_str: str, as_dataframe: bool = False) -> Any:
        """Deserializza una stringa JSON nell'oggetto originale."""
        try:
            if as_dataframe:
                # Prova a ricostruire DataFrame
                return pd.read_json(data_str, orient='split')
            else:
                return json.loads(data_str)
        except Exception:
            # Fallback: restituisci la stringa grezza
            return data_str

    # ----- Metodi per fondamentali -----
    def get_fundamental(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Recupera i dati fondamentali dalla cache, se validi."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT data, timestamp, ttl FROM fundamental_cache WHERE symbol = ?",
                (symbol.upper(),)
            )
            row = cursor.fetchone()
            if row and not self._is_expired(row[1], row[2]):
                return self._deserialize_data(row[0], as_dataframe=False)
        return None

    def set_fundamental(self, symbol: str, data: Dict[str, Any], ttl: int = TTL_FUNDAMENTAL):
        """Salva i dati fondamentali in cache."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO fundamental_cache (symbol, data, timestamp, ttl) VALUES (?, ?, ?, ?)",
                (symbol.upper(), self._serialize_data(data), time.time(), ttl)
            )
            conn.commit()

    # ----- Metodi per tecnici -----
    def get_technical(self, symbol: str) -> Optional[pd.DataFrame]:
        """Recupera i dati tecnici (DataFrame OHLCV) dalla cache."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT data, timestamp, ttl FROM technical_cache WHERE symbol = ?",
                (symbol.upper(),)
            )
            row = cursor.fetchone()
            if row and not self._is_expired(row[1], row[2]):
                return self._deserialize_data(row[0], as_dataframe=True)
        return None

    def set_technical(self, symbol: str, df: pd.DataFrame, ttl: int = TTL_TECHNICAL):
        """Salva i dati tecnici in cache."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO technical_cache (symbol, data, timestamp, ttl) VALUES (?, ?, ?, ?)",
                (symbol.upper(), self._serialize_data(df), time.time(), ttl)
            )
            conn.commit()

    # ----- Utilità -----
    def clear_expired(self):
        """Rimuove tutte le voci scadute da entrambe le tabelle."""
        now = time.time()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM fundamental_cache WHERE (? - timestamp) > ttl", (now,))
            cursor.execute("DELETE FROM technical_cache WHERE (? - timestamp) > ttl", (now,))
            conn.commit()

    def invalidate_symbol(self, symbol: str):
        """Invalida completamente i dati di un simbolo (sia fondamentali che tecnici)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM fundamental_cache WHERE symbol = ?", (symbol.upper(),))
            cursor.execute("DELETE FROM technical_cache WHERE symbol = ?", (symbol.upper(),))
            conn.commit()

    def get_stats(self) -> Dict[str, int]:
        """Restituisce statistiche sul numero di record in cache."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM fundamental_cache")
            fund_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM technical_cache")
            tech_count = cursor.fetchone()[0]
        return {"fundamental_entries": fund_count, "technical_entries": tech_count}


# Istanza globale (singleton)
_cache_manager = None

def get_cache_manager() -> CacheManager:
    """Restituisce l'istanza singleton del cache manager."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager