"""
Analisi del sentiment da news (FinBERT) e social (Reddit con VADER).
"""

import requests
import feedparser
from typing import Dict, Any, List, Optional
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import praw
import time

# Configurazione Reddit (devi inserire le tue credenziali, oppure usa mock)
REDDIT_CLIENT_ID = None   # Inserisci il tuo client ID
REDDIT_CLIENT_SECRET = None
REDDIT_USER_AGENT = "VQuantPro/0.1"

# Carica modello FinBERT (una volta)
try:
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    finbert_loaded = True
except Exception as e:
    print(f"FinBERT non disponibile: {e}")
    finbert_loaded = False

vader_analyzer = SentimentIntensityAnalyzer()

def get_finbert_sentiment(text: str) -> float:
    """Restituisce un punteggio da -1 (negativo) a +1 (positivo) usando FinBERT."""
    if not finbert_loaded or not text:
        return 0.0
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    # Le etichette: 0 = negativo, 1 = neutro, 2 = positivo
    scores = probs.cpu().detach().numpy()[0]
    # Mappa da -1 a +1
    sentiment_score = (scores[2] - scores[0])  # (positivo - negativo)
    return float(sentiment_score)

def get_vader_sentiment(text: str) -> float:
    """Restituisce compound score di VADER (-1 a +1)."""
    return vader_analyzer.polarity_scores(text)['compound']

def fetch_yahoo_finance_news(ticker: str, limit: int = 5) -> List[str]:
    """Recupera i titoli delle news da Yahoo Finance RSS."""
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}"
    feed = feedparser.parse(url)
    titles = [entry.title for entry in feed.entries[:limit]]
    return titles

def fetch_reddit_posts(subreddit: str = "wallstreetbets", limit: int = 10, ticker: str = "") -> List[str]:
    """Recupera post da Reddit (richiede credenziali). Se non configurato, restituisce lista vuota."""
    if REDDIT_CLIENT_ID is None or REDDIT_CLIENT_SECRET is None:
        return []
    try:
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT
        )
        posts = []
        for submission in reddit.subreddit(subreddit).hot(limit=limit):
            if ticker.lower() in submission.title.lower() or ticker.lower() in submission.selftext.lower():
                posts.append(submission.title + " " + submission.selftext[:500])
        return posts
    except Exception as e:
        print(f"Reddit error: {e}")
        return []

def get_overall_sentiment(ticker: str) -> Dict[str, Any]:
    """Combina sentiment da news e social."""
    news_titles = fetch_yahoo_finance_news(ticker, limit=5)
    news_scores = [get_finbert_sentiment(title) for title in news_titles if title]
    avg_news = sum(news_scores) / len(news_scores) if news_scores else 0.0
    
    reddit_posts = fetch_reddit_posts(ticker=ticker, limit=10)
    reddit_scores = [get_vader_sentiment(post) for post in reddit_posts]
    avg_reddit = sum(reddit_scores) / len(reddit_scores) if reddit_scores else 0.0
    
    # Media pesata: più peso alle news (più affidabili)
    total = avg_news * 0.7 + avg_reddit * 0.3
    
    if total > 0.2:
        label = "🟢 Positivo"
    elif total < -0.2:
        label = "🔴 Negativo"
    else:
        label = "🟡 Neutro"
    
    return {
        "score": total,
        "label": label,
        "news_score": avg_news,
        "reddit_score": avg_reddit
    }