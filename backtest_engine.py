"""
Modulo per il backtesting di strategie di trading.
Utilizza la libreria backtesting.py
"""

import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from typing import Dict, Any, Optional, Tuple


class SmaCrossoverStrategy(Strategy):
    """
    Strategia: crossover tra due medie mobili semplici.
    Compra quando la SMA veloce incrocia sopra la SMA lenta.
    Vende quando incrocia sotto.
    """
    n1 = 50   # media veloce
    n2 = 200  # media lenta

    def init(self):
        self.sma1 = self.I(lambda x: pd.Series(x).rolling(self.n1).mean(), self.data.Close)
        self.sma2 = self.I(lambda x: pd.Series(x).rolling(self.n2).mean(), self.data.Close)

    def next(self):
        if not self.position:
            if crossover(self.sma1, self.sma2):
                self.buy()
        elif crossover(self.sma2, self.sma1):
            self.position.close()


class RsiMeanReversionStrategy(Strategy):
    """
    Strategia: acquista quando RSI < 30 (ipervenduto),
    vende quando RSI > 70 (ipercomprato).
    """
    rsi_period = 14
    overbought = 70
    oversold = 30

    def init(self):
        # Calcola RSI manualmente
        close = self.data.Close
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(self.rsi_period).mean()
        avg_loss = loss.rolling(self.rsi_period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        self.rsi = self.I(lambda: rsi)

    def next(self):
        if not self.position:
            if self.rsi[-1] < self.oversold:
                self.buy()
        else:
            if self.rsi[-1] > self.overbought:
                self.position.close()


def run_backtest(
    data: pd.DataFrame,
    strategy: str,
    cash: float = 10000.0,
    commission: float = 0.001,
) -> Dict[str, Any]:
    """
    Esegue un backtest su dati storici.
    strategy: 'sma_crossover' o 'rsi_mean_reversion'
    Restituisce un dizionario con metriche e il grafico.
    """
    if data is None or data.empty or len(data) < 200:
        return {"error": "Dati insufficienti (servono almeno 200 giorni)"}

    # Prepara i dati nel formato richiesto da backtesting.py
    df = data.copy()
    df.columns = [col.capitalize() for col in df.columns]
    if 'Open' not in df.columns or 'High' not in df.columns or 'Low' not in df.columns or 'Close' not in df.columns or 'Volume' not in df.columns:
        return {"error": "Il DataFrame deve contenere Open, High, Low, Close, Volume"}

    # Scegli la strategia
    if strategy == 'sma_crossover':
        bt = Backtest(df, SmaCrossoverStrategy, cash=cash, commission=commission)
    elif strategy == 'rsi_mean_reversion':
        bt = Backtest(df, RsiMeanReversionStrategy, cash=cash, commission=commission)
    else:
        return {"error": f"Strategia sconosciuta: {strategy}"}

    # Esegui il backtest
    results = bt.run()
    
    # Estrai metriche principali
    metrics = {
        "Start": results['Start'],
        "End": results['End'],
        "Duration": results['Duration'],
        "Return [%]": results['Return [%]'],
        "Sharpe Ratio": results['Sharpe Ratio'],
        "Max Drawdown [%]": results['Max Drawdown [%]'],
        "Win Rate [%]": results['Win Rate [%]'],
        "Total Trades": results['# Trades'],
        "Equity Final [$]": results['Equity Final [$]'],
    }
    
    return {"metrics": metrics, "results": results, "plot": bt.plot()}