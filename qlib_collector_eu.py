# qlib_collector_eu.py
import os
import pandas as pd
import yfinance as yf
from datetime import datetime
from typing import List, Optional
import time
from pathlib import Path

class EUYahooCollector:
    """
    Collector personalizzato per ticker europei e internazionali.
    Scarica dati da yfinance e li salva in formato CSV compatibile con Qlib.
    """
    
    def __init__(self, save_dir: str = "~/.qlib/raw_data/eu", max_workers: int = 2, delay: float = 1.0):
        self.save_dir = os.path.expanduser(save_dir)
        self.max_workers = max_workers
        self.delay = delay
        os.makedirs(self.save_dir, exist_ok=True)
    
    def normalize_symbol(self, symbol: str) -> str:
        """
        Normalizza il simbolo per Qlib.
        Esempi:
        - AAPL → AAPL
        - ENI.MI → ENI.MI (mantiene il suffisso)
        - BMW.DE → BMW.DE
        - AIR.PA → AIR.PA
        - ULVR.L → ULVR.L
        """
        return symbol.upper().strip()
    
    def get_data(self, symbol: str, interval: str = "1d", 
                 start_date: Optional[str] = None, 
                 end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Scarica dati da yfinance.
        """
        ticker = yf.Ticker(symbol)
        if start_date and end_date:
            df = ticker.history(start=start_date, end=end_date, interval=interval)
        else:
            df = ticker.history(period="max", interval=interval)
        
        if df.empty:
            raise ValueError(f"Nessun dato per {symbol}")
        
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        return df
    
    def save_instrument(self, symbol: str, df: pd.DataFrame):
        """Salva i dati in CSV per un simbolo."""
        symbol_norm = self.normalize_symbol(symbol)
        file_path = os.path.join(self.save_dir, f"{symbol_norm}.csv")
        df.to_csv(file_path, index=False)
        print(f"Salvato {symbol_norm} → {file_path}")
        return file_path
    
    def collect(self, symbols: List[str], interval: str = "1d",
                start_date: Optional[str] = None, end_date: Optional[str] = None):
        """Collega dati per una lista di simboli."""
        results = {}
        for i, symbol in enumerate(symbols):
            print(f"Scaricamento {symbol} ({i+1}/{len(symbols)})...")
            try:
                df = self.get_data(symbol, interval, start_date, end_date)
                if len(df) >= 60:
                    path = self.save_instrument(symbol, df)
                    results[symbol] = {"path": path, "rows": len(df)}
                else:
                    print(f"⚠️ {symbol}: dati insufficienti ({len(df)} righe)")
            except Exception as e:
                print(f"❌ Errore per {symbol}: {e}")
            time.sleep(self.delay)
        return results