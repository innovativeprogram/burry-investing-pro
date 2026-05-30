"""
Modulo per addestrare e utilizzare un modello ML di classificazione del trend.
Il modello viene addestrato offline e salvato su disco.
In produzione, viene caricato e usato per predire il trend a 5 giorni.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from typing import Optional, Tuple, Dict, Any

MODEL_PATH = "trend_predictor_model.pkl"

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Costruisce le feature tecniche per il modello."""
    df = df.copy()
    # Calcola indicatori già presenti nel codice principale
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['SMA_200'] = df['Close'].rolling(200).mean()
    df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().clip(lower=0).ewm(alpha=1/14).mean() / 
                                 (-df['Close'].diff().clip(upper=0)).ewm(alpha=1/14).mean())))
    # MACD
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    # Volatilità rolling
    df['volatility'] = df['Close'].pct_change().rolling(20).std()
    # Target: prezzo a 5 giorni in futuro > prezzo attuale?
    df['target'] = (df['Close'].shift(-5) > df['Close']).astype(int)
    # Rimuovi NaN
    df = df.dropna()
    features = ['SMA_50', 'SMA_200', 'RSI', 'MACD', 'MACD_signal', 'volatility']
    return df[features + ['target']]

def train_model(df: pd.DataFrame) -> Tuple[RandomForestClassifier, float]:
    """Addestra un modello RandomForest e lo salva su disco."""
    data = build_features(df)
    if data.empty:
        raise ValueError("Dati insufficienti per l'addestramento")
    X = data.drop('target', axis=1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    joblib.dump(clf, MODEL_PATH)
    return clf, acc

def load_or_train_model(df: Optional[pd.DataFrame] = None) -> Optional[RandomForestClassifier]:
    """Carica il modello salvato, oppure lo addestra se non esiste e vengono forniti dati."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    elif df is not None:
        clf, acc = train_model(df)
        print(f"Modello addestrato con accuratezza: {acc:.2f}")
        return clf
    else:
        return None

def predict_trend(ticker: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Predice il trend a 5 giorni per un ticker usando il modello salvato."""
    model = load_or_train_model()
    if model is None:
        return None
    features_df = build_features(df)
    if features_df.empty:
        return None
    # Prendi l'ultima riga disponibile
    last_features = features_df.drop('target', axis=1).iloc[-1:].values
    prob_up = model.predict_proba(last_features)[0][1]  # probabilità di rialzo
    return {
        "probability_up": float(prob_up),
        "signal": "🟢 Rialzo atteso" if prob_up > 0.55 else "🔴 Ribasso atteso" if prob_up < 0.45 else "⚪ Neutro",
        "confidence": prob_up * 100
    }