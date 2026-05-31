# qlib_data_converter.py
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import shutil

class QlibDataConverter:
    """
    Converte i dati CSV nel formato binario di Qlib.
    Struttura finale:
        ~/.qlib/qlib_data/eu/
        ├── calendars/
        │   └── day.txt (lista delle date)
        ├── features/
        │   └── (file .bin per ogni feature)
        └── instruments/
            └── all.txt (lista dei simboli)
    """
    
    def __init__(self, raw_data_dir: str = "~/.qlib/raw_data/eu",
                 output_dir: str = "~/.qlib/qlib_data/eu"):
        self.raw_dir = os.path.expanduser(raw_data_dir)
        self.output_dir = os.path.expanduser(output_dir)
        self.calendar_file = os.path.join(self.output_dir, "calendars", "day.txt")
        self.instruments_file = os.path.join(self.output_dir, "instruments", "all.txt")
        
    def prepare_structure(self):
        """Crea la struttura delle cartelle."""
        os.makedirs(os.path.join(self.output_dir, "calendars"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "features"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "instruments"), exist_ok=True)
    
    def read_raw_data(self, symbol: str) -> pd.DataFrame:
        """Legge il CSV raw per un simbolo."""
        path = os.path.join(self.raw_dir, f"{symbol}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV non trovato per {symbol}")
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df
    
    def build_calendar(self, all_dates: List[pd.Timestamp]):
        """Costruisce il file calendar (tutte le date in ordine)."""
        dates_sorted = sorted(set(all_dates))
        with open(self.calendar_file, "w") as f:
            for d in dates_sorted:
                f.write(d.strftime("%Y%m%d") + "\n")
        return len(dates_sorted)
    
    def write_bin_feature(self, data: np.ndarray, output_path: str):
        """Scrive una feature binaria nel formato Qlib (float32)."""
        data.astype("<f4").tofile(output_path)
    
    def convert(self, symbols: List[str]):
        """
        Converte i CSV raw nel formato binario Qlib.
        """
        self.prepare_structure()
        
        # Step 1: trova tutte le date
        all_dates = set()
        symbol_data = {}
        
        for symbol in symbols:
            try:
                df = self.read_raw_data(symbol)
                symbol_data[symbol] = df
                all_dates.update(df.index)
            except FileNotFoundError:
                print(f"⚠️ {symbol}: CSV non trovato")
        
        if not all_dates:
            raise ValueError("Nessun dato valido trovato")
        
        # Step 2: costruisci il calendario
        date_list = sorted(all_dates)
        date_to_idx = {d: i for i, d in enumerate(date_list)}
        num_dates = len(date_list)
        
        with open(self.calendar_file, "w") as f:
            for d in date_list:
                f.write(d.strftime("%Y%m%d") + "\n")
        print(f"✅ Calendar: {num_dates} date")
        
        # Step 3: scrivi il file degli strumenti
        with open(self.instruments_file, "w") as f:
            for symbol in symbol_data.keys():
                f.write(symbol + "\n")
        print(f"✅ Instruments: {len(symbol_data)} simboli")
        
        # Step 4: per ogni strumento, scrivi le feature binarie
        features = ["open", "high", "low", "close", "volume"]
        
        for symbol, df in symbol_data.items():
            feature_dir = os.path.join(self.output_dir, "features", symbol)
            os.makedirs(feature_dir, exist_ok=True)
            
            for feature in features:
                if feature not in df.columns:
                    continue
                # Crea array delle date
                feature_array = np.full(num_dates, np.nan, dtype=np.float32)
                for idx, (date, value) in enumerate(df[feature].items()):
                    if date in date_to_idx:
                        feature_array[date_to_idx[date]] = float(value)
                # Salva .bin
                bin_path = os.path.join(feature_dir, f"{feature}.bin")
                self.write_bin_feature(feature_array, bin_path)
            
            print(f"✅ {symbol}: convertito")
        
        print(f"\n✅ Conversione completata in {self.output_dir}")
        return self.output_dir