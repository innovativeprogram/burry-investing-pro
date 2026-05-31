# qlib_strategies.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from typing import Dict, Any, Optional, List, Tuple

# Qlib optional
try:
    import qlib
    from qlib.constant import REG_US, REG_CN
    from qlib.config import C
    from qlib.data import D
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.contrib.data.handler import Alpha158
    from qlib.contrib.model.gbdt import LGBModel
    from qlib.contrib.evaluate import risk_analysis
    from qlib.contrib.strategy import TopkDropoutStrategy
    from qlib.backtest import backtest, executor
    from qlib.contrib.evaluate import backtest as daily_backtest
    QLIB_AVAILABLE = True
except ImportError:
    QLIB_AVAILABLE = False
    print("Qlib non disponibile. Installare con: pip install pyqlib")

# Importa i moduli personalizzati
from qlib_data_converter import QlibDataConverter
from qlib_collector_eu import EUYahooCollector


class QlibPipeline:
    """
    Pipeline completa per Qlib:
    1. Scarica dati da yfinance (con collector personalizzato per EU)
    2. Converte in formato binario Qlib
    3. Addestra modello LightGBM
    4. Esegue backtest e restituisce metriche
    """
    
    def __init__(self, data_uri: str = "~/.qlib/qlib_data/eu"):
        self.data_uri = os.path.expanduser(data_uri)
        self.collector = EUYahooCollector()
        self.converter = QlibDataConverter(output_dir=self.data_uri)
        self._initialized = False
    
    def initialize_qlib(self):
        """Inizializza Qlib con i dati convertiti."""
        if not QLIB_AVAILABLE:
            raise ImportError("Qlib non installato")
        if not self._initialized:
            # Inizializza nella modalità US (perché abbiamo dati da yfinance USA/Internazionali)
            qlib.init(provider_uri=self.data_uri, region=REG_US)
            self._initialized = True
            print(f"✅ Qlib inizializzato con URI: {self.data_uri}")
    
    def prepare_data(self, symbols: List[str], start_date: str = "2018-01-01", 
                     end_date: Optional[str] = None, force_refresh: bool = False) -> bool:
        """
        Prepara i dati per Qlib:
        1. Scarica i CSV raw (se non esistono o force_refresh=True)
        2. Converte in formato binario
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        # Step 1: scarica i dati raw
        raw_required = force_refresh
        for symbol in symbols:
            csv_path = os.path.join(self.collector.save_dir, f"{self.collector.normalize_symbol(symbol)}.csv")
            if not os.path.exists(csv_path):
                raw_required = True
                break
        
        if raw_required:
            st.info(f"Scaricamento dati per {len(symbols)} simboli...")
            results = self.collector.collect(symbols, start_date=start_date, end_date=end_date)
            print(f"Dati scaricati: {results}")
        
        # Step 2: converti in formato binario
        bin_required = force_refresh
        for symbol in symbols:
            feature_dir = os.path.join(self.data_uri, "features", symbol)
            if not os.path.exists(feature_dir):
                bin_required = True
                break
        
        if bin_required:
            st.info(f"Conversione dei dati in formato Qlib ({self.data_uri})...")
            self.converter.convert(symbols)
        
        return True
    
    def train_model(self, symbols: List[str], start_train: str = "2018-01-01",
                    end_train: str = "2023-12-31", horizon: int = 5) -> Dict[str, Any]:
        """
        Addestra un modello LightGBM su Alpha158 features.
        """
        self.initialize_qlib()
        
        # Configura l'handler dei dati
        handler = Alpha158(instruments=symbols, start_time=start_train, 
                          end_time=end_train, fit_start_time=start_train,
                          fit_end_time=end_train, infer_processors=[])
        
        dataset = DatasetH(handler, segments={"train": (start_train, end_train)})
        
        # Configura il modello LightGBM
        model = LGBModel(
            loss="mse",
            colsample_bytree=0.8,
            learning_rate=0.1,
            subsample=0.8,
            lambda_l1=205,
            lambda_l2=580,
            max_depth=8,
            num_leaves=210,
            num_threads=20,
            verbosity=-1,
            early_stopping_rounds=50
        )
        
        # Addestra
        st.info(f"Addestramento modello LightGBM su {len(symbols)} simboli ({start_train} → {end_train})...")
        model.fit(dataset)
        
        # Predici
        pred = model.predict(dataset)
        
        # Crea una strategia semplice: prendi i top k ogni giorno
        strategy = TopkDropoutStrategy(
            model=model,
            dataset=dataset,
            topk=len(symbols) // 2,
            n_drop=5,
            risk_degree=0.05
        )
        
        return {
            "model": model,
            "predictions": pred,
            "strategy": strategy,
            "dataset": dataset
        }
    
    def run_backtest(self, symbols: List[str], start_backtest: str = "2024-01-01",
                     end_backtest: Optional[str] = None) -> Dict[str, Any]:
        """
        Esegue backtest su un orizzonte successivo all'addestramento.
        """
        if end_backtest is None:
            end_backtest = datetime.now().strftime("%Y-%m-%d")
        
        self.initialize_qlib()
        
        # Carica l'handler per il periodo di backtest
        handler = Alpha158(instruments=symbols, start_time=start_backtest,
                          end_time=end_backtest, infer_processors=[])
        
        dataset = DatasetH(handler)
        
        # Addestra un modello semplice per il backtest
        # (in produzione, si caricherebbe un modello pre-addestrato)
        model = LGBModel(early_stopping_rounds=50, verbosity=-1)
        model.fit(dataset)
        
        # Strategia TopK
        strategy = TopkDropoutStrategy(
            model=model,
            dataset=dataset,
            topk=len(symbols) // 3,
            n_drop=3,
            risk_degree=0.05
        )
        
        # Esegui backtest
        st.info(f"Esecuzione backtest ({start_backtest} → {end_backtest})...")
        report = daily_backtest(
            strategy=strategy,
            start_time=start_backtest,
            end_time=end_backtest,
            account=100000,
            benchmark=None,
            freq="day",
            exchange_kwargs={"limit_threshold": 0.095, "deal_price": "close"}
        )
        
        # Estrai metriche
        metrics = {}
        if report is not None and not report.empty:
            metrics["total_return"] = float(report["return"].iloc[-1]) if "return" in report else 0.0
            metrics["sharpe"] = float(report["sharpe"].iloc[-1]) if "sharpe" in report else 0.0
            metrics["max_drawdown"] = float(report["max_drawdown"].iloc[-1]) if "max_drawdown" in report else 0.0
            metrics["annual_return"] = float(report["annual_return"].iloc[-1]) if "annual_return" in report else 0.0
        
        return {"report": report, "metrics": metrics}
    
    def generate_predictions(self, symbols: List[str], lookback_days: int = 252,
                             forward_days: int = 5) -> pd.DataFrame:
        """
        Genera previsioni per i prossimi forward_days giorni.
        """
        self.initialize_qlib()
        
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        
        handler = Alpha158(instruments=symbols, start_time=start_date,
                          end_time=end_date, infer_processors=[])
        
        dataset = DatasetH(handler)
        
        # Modello veloce (addestrato su tutti i dati disponibili)
        model = LGBModel(verbosity=-1)
        model.fit(dataset)
        
        predictions = model.predict(dataset)
        
        results = []
        for i, symbol in enumerate(symbols):
            if i < len(predictions):
                results.append({
                    "symbol": symbol,
                    "prediction": float(predictions[i]),
                    "direction": "UP" if predictions[i] > 0 else "DOWN",
                    "confidence": abs(float(predictions[i]))
                })
        
        return pd.DataFrame(results)


def run_quick_backtest(tickers: List[str], lookback_years: int = 3) -> Dict[str, Any]:
    """
    Funzione wrapper per eseguire un backtest rapido con Qlib.
    """
    pipeline = QlibPipeline()
    
    # Prepara i dati
    end_date = datetime.now()
    start_date = (end_date - timedelta(days=lookback_years*365)).strftime("%Y-%m-%d")
    
    success = pipeline.prepare_data(tickers, start_date=start_date)
    if not success:
        return {"error": "Preparazione dati fallita"}
    
    # Addestra e backtest
    split_date = (end_date - timedelta(days=180)).strftime("%Y-%m-%d")
    
    try:
        pipeline.initialize_qlib()
        result = pipeline.run_backtest(tickers, start_backtest=split_date)
        return result
    except Exception as e:
        return {"error": f"Errore durante backtest: {e}"}