"""
portfolio_optimizer.py
Ottimizzazione portafoglio con Sharpe massimo, min varianza, e calcolo CADR.
"""

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import minimize

def portfolio_annualised_performance(weights, mean_returns, cov_matrix, risk_free_rate=0.04):
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe = (returns - risk_free_rate) / std
    return std, returns, sharpe

def neg_sharpe(weights, mean_returns, cov_matrix, risk_free_rate):
    return -portfolio_annualised_performance(weights, mean_returns, cov_matrix, risk_free_rate)[2]

def max_sharpe_weights(mean_returns, cov_matrix, risk_free_rate=0.04):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    result = minimize(neg_sharpe, num_assets * [1./num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def compute_cadr(returns_df, weights):
    """Correlation-Adjusted Drawdown Risk"""
    port_ret = (returns_df * pd.Series(weights)).sum(axis=1)
    equity = (1 + port_ret).cumprod()
    rolling_max = equity.expanding().max()
    drawdown = (equity - rolling_max) / rolling_max
    contributions = {}
    for ticker in weights:
        asset_ret = returns_df[ticker]
        cov = np.cov(asset_ret.dropna(), drawdown.dropna())[0,1]
        std_asset = asset_ret.std()
        if std_asset != 0:
            beta_dd = cov / std_asset**2
            contributions[ticker] = beta_dd * weights[ticker] * drawdown.mean()
    return contributions