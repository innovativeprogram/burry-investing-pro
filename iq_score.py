"""
iq_score.py
Calcola l'IQ Score unificato (0-100) combinando:
- Qualità fondamentale (FQS)
- Valutazione (VAS)
- Timing tecnico (TMS)
- Rischio quantitativo (QRS)
- Momentum a 20 giorni
- Sentiment (da notizie + social)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Callable

def compute_iq_score(
    ticker: str,
    row: pd.Series,
    timing_score: float,
    qm: Dict[str, Any],
    risk: Dict[str, Any],
    sentiment_score: float = 0.5,
    # funzioni esterne (iniettate dal main)
    get_unified_verdict: Optional[Callable] = None,
    get_macro: Optional[Callable] = None,
    get_technical_data: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Restituisce:
        {
            "IQ_Score": float,
            "components": {...},
            "verdict": str,
            "emoji": str
        }
    """
    if get_unified_verdict is None or get_macro is None:
        raise ValueError("Missing required callables: get_unified_verdict, get_macro")
    
    macro = get_macro()
    verdict = get_unified_verdict(row, timing_score, qm, risk, macro)
    fqs = verdict['FQS']
    vas = verdict['VAS']
    tms = verdict['TMS']
    qrs = verdict['QRS']
    
    # Momentum (rendimento vs SMA20)
    momentum_score = 0.0
    if get_technical_data is not None:
        df_tech = get_technical_data(ticker)
        if df_tech is not None and len(df_tech) >= 20:
            close = df_tech['Close']
            sma20 = close.rolling(20).mean().iloc[-1]
            last_close = close.iloc[-1]
            if sma20 and sma20 > 0:
                momentum = (last_close - sma20) / sma20
                momentum_score = np.clip((momentum + 0.10) / 0.20, 0, 1) * 10
    
    # Pesi
    weights = {'fqs': 0.35, 'vas': 0.25, 'tms': 0.20, 'qrs': 0.15, 'momentum': 0.05}
    raw = (weights['fqs'] * fqs + weights['vas'] * vas +
           weights['tms'] * tms + weights['qrs'] * qrs +
           weights['momentum'] * momentum_score)
    
    # Fattore sentiment (0.8 .. 1.2)
    sentiment_factor = 0.8 + sentiment_score * 0.4
    final = np.clip(raw * sentiment_factor, 0, 100)
    
    return {
        "IQ_Score": final,
        "components": {
            "FQS": fqs,
            "VAS": vas,
            "TMS": tms,
            "QRS": qrs,
            "Momentum": momentum_score,
            "SentimentFactor": sentiment_factor
        },
        "verdict": verdict['Verdict'],
        "emoji": verdict['Emoji']
    }