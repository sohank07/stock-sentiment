from fastapi import FastAPI, HTTPException
from typing import List
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datetime import datetime, timedelta
import numpy as np
from dotenv import load_dotenv
import os

app = FastAPI()

# Load FinBERT model and tokenizer
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
sentiment_analyzer = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

# NewsAPI settings
api_key = os.getenv('NEWS_API_KEY')

def fetch_news(stock: str, days: int = 2):
    today = datetime.now().date()
    from_date = (today - timedelta(days=days)).strftime('%Y-%m-%d')
    to_date = today.strftime('%Y-%m-%d')

    url = (
        f'https://newsapi.org/v2/everything?q={stock}'
        f'&from={from_date}&to={to_date}'
        f'&language=en'
        f'&sortBy=publishedAt&apiKey={api_key}'
    )

    response = requests.get(url)
    news_data = response.json()
    
    if news_data['status'] != 'ok':
        raise HTTPException(status_code=400, detail=news_data.get('message', 'Unknown error'))

    return news_data['articles']

def analyze_sentiment(articles):
    results = []
    for article in articles:
        text = f"{article['title']}. {article['description']}"
        sentiment = sentiment_analyzer(text)[0]
        score = sentiment['score'] if sentiment['label'] == 'positive' else -sentiment['score']
        results.append({
            'title': article['title'],
            'description': article['description'],
            'sentiment_score': score
        })
    return results

@app.get("/sentiment/{stock}", response_model=List[dict])
def get_sentiment(stock: str):
    articles = fetch_news(stock)
    sentiments = analyze_sentiment(articles)
    return sentiments

@app.get("/top_articles/{stock}", response_model=List[dict])
def get_top_articles(stock: str):
    articles = fetch_news(stock)
    sentiments = analyze_sentiment(articles)
    sorted_articles = sorted(sentiments, key=lambda x: abs(x['sentiment_score']), reverse=True)[:5]
    return sorted_articles

@app.get("/average_sentiment/{stock}")
def get_average_sentiment(stock: str):
    articles = fetch_news(stock)
    sentiments = analyze_sentiment(articles)
    scores = [article['sentiment_score'] for article in sentiments]
    average_score = np.mean(scores)
    return {"average_sentiment_score": average_score}
