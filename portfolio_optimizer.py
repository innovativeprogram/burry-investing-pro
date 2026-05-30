"""
Ottimizzazione portafoglio con Markowitz (PyPortfolioOpt)
Utilizza i rendimenti storici dei ticker per calcolare i pesi ottimali.
"""

import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier, expected_returns, risk_models
from pypfopt import plotting
from typing import List, Dict, Tuple, Optional
import plotly.graph_objects as go


def get_portfolio_weights_markowitz(
    tickers: List[str],
    returns_df: pd.DataFrame,
    method: str = "max_sharpe",
    risk_free_rate: float = 0.04,
) -> Dict[str, float]:
    """
    Calcola i pesi ottimali usando la teoria di Markowitz.

    Args:
        tickers: Lista dei ticker nel portafoglio.
        returns_df: DataFrame con i rendimenti giornalieri (colonne = ticker).
        method: "max_sharpe" o "min_volatility".
        risk_free_rate: Tasso privo di rischio annualizzato (default 4%).

    Returns:
        Dizionario {ticker: peso_ottimale (in percentuale, somma 100%)}
    """
    if returns_df.empty or len(tickers) == 0:
        return {}

    # Calcola rendimenti attesi annualizzati e matrice di covarianza
    mu = expected_returns.mean_historical_return(returns_df, frequency=252)
    S = risk_models.sample_cov(returns_df, frequency=252)

    # Ottimizzazione
    ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))  # no short selling

    if method == "max_sharpe":
        ef.max_sharpe(risk_free_rate=risk_free_rate)
    elif method == "min_volatility":
        ef.min_volatility()
    else:
        raise ValueError("method deve essere 'max_sharpe' o 'min_volatility'")

    cleaned_weights = ef.clean_weights()
    # Converti in percentuali e arrotonda
    weights_pct = {ticker: round(weight * 100, 2) for ticker, weight in cleaned_weights.items()}
    return weights_pct


def get_efficient_frontier_points(
    tickers: List[str],
    returns_df: pd.DataFrame,
    points: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcola una serie di punti sulla frontiera efficiente.
    Restituisce (volatilità, rendimento atteso).
    """
    if returns_df.empty:
        return np.array([]), np.array([])

    mu = expected_returns.mean_historical_return(returns_df, frequency=252)
    S = risk_models.sample_cov(returns_df, frequency=252)
    ef = EfficientFrontier(mu, S)

    # Genera la frontiera
    try:
        rets, vols = ef.efficient_frontier(points)
        return vols, rets
    except Exception as e:
        print(f"Errore frontiera efficiente: {e}")
        return np.array([]), np.array([])


def plot_efficient_frontier(
    tickers: List[str],
    returns_df: pd.DataFrame,
    current_weights: Optional[Dict[str, float]] = None,
    optimal_weights_max_sharpe: Optional[Dict[str, float]] = None,
) -> go.Figure:
    """
    Crea un grafico Plotly della frontiera efficiente con i punti:
    - portafoglio attuale (se fornito)
    - portafoglio max Sharpe
    - portafoglio min varianza
    """
    vols, rets = get_efficient_frontier_points(tickers, returns_df)
    if len(vols) == 0:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=vols * 100, y=rets * 100,
        mode='lines', name='Frontiera Efficiente',
        line=dict(color='blue', dash='dot')
    ))

    # Portafoglio attuale (se fornito)
    if current_weights is not None and returns_df is not None:
        # Calcola rendimento atteso e volatilità del portafoglio attuale
        w = np.array([current_weights.get(t, 0.0) for t in tickers]) / 100.0
        mu = expected_returns.mean_historical_return(returns_df, frequency=252).values
        S = risk_models.sample_cov(returns_df, frequency=252).values
        port_ret = np.dot(w, mu) * 100
        port_vol = np.sqrt(np.dot(w.T, np.dot(S, w))) * 100
        fig.add_trace(go.Scatter(
            x=[port_vol], y=[port_ret],
            mode='markers', name='Portafoglio attuale',
            marker=dict(color='orange', size=12, symbol='circle')
        ))

    # Aggiungi i pesi ottimali (max Sharpe e min vol)
    if optimal_weights_max_sharpe is not None:
        w_opt = np.array([optimal_weights_max_sharpe.get(t, 0.0) for t in tickers]) / 100.0
        mu_opt = np.dot(w_opt, mu) * 100 if 'mu' in locals() else 0
        vol_opt = np.sqrt(np.dot(w_opt.T, np.dot(S, w_opt))) * 100 if 'S' in locals() else 0
        fig.add_trace(go.Scatter(
            x=[vol_opt], y=[mu_opt],
            mode='markers', name='Max Sharpe',
            marker=dict(color='green', size=14, symbol='star')
        ))

    fig.update_layout(
        title="Frontiera Efficiente di Markowitz",
        xaxis_title="Volatilità annualizzata (%)",
        yaxis_title="Rendimento atteso annualizzato (%)",
        template="plotly_dark",
        height=500,
    )
    return fig