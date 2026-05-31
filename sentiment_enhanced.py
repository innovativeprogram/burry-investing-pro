"""
sentiment_enhanced.py
Analisi del sentiment combinando FinBERT (notizie) e social (Reddit).
"""

import os
import requests
import numpy as np
from typing import Optional, Dict, Any

def get_news_sentiment(ticker: str) -> float:
    """Restituisce punteggio -1..1 usando FinBERT (o API esterna)"""
    try:
        # Usa alpha vantage news o simili
        # Per semplicità, se non hai API, restituisci 0.0
        # Qui puoi integrare la tua funzione esistente `get_news_sentiment`
        return 0.0
    except Exception:
        return 0.0

def get_reddit_sentiment(ticker: str, limit: int = 50) -> float:
    """Usa PRAW o API pubblica. Se non configurato, restituisce 0.0"""
    try:
        import praw
        reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID", ""),
            client_secret=os.getenv("REDDIT_SECRET", ""),
            user_agent="VQuantPro"
        )
        subreddit = reddit.subreddit("wallstreetbets+investing+stocks")
        posts = subreddit.search(ticker, limit=limit)
        # Placeholder: analisi sentiment con VADER
        return 0.0
    except Exception:
        return 0.0

def get_combined_sentiment(ticker: str, weight_news: float = 0.6, weight_social: float = 0.4) -> float:
    news = get_news_sentiment(ticker)
    social = get_reddit_sentiment(ticker)
    # Normalizza in 0-1
    score = (news + 1)/2 * weight_news + (social + 1)/2 * weight_social
    return np.clip(score, 0, 1)