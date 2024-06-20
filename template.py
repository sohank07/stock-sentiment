import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datetime import datetime, timedelta
import numpy as np
from dotenv import load_dotenv
import os

# NewsAPI settings
api_key = os.getenv('NEWS_API_KEY')
query = 'Tesla'

# Define the date range for the past week
today = datetime.now().date()
last_week = today - timedelta(days=3)
from_date = last_week.strftime('%Y-%m-%d')                          #today.strftime('%Y-%m-%d')                         
to_date = today.strftime('%Y-%m-%d')

# Construct the NewsAPI URL
url = (
    f'https://newsapi.org/v2/everything?q={query}'
    f'&from={from_date}&to={to_date}'
    f'&language=en'
    f'&sortBy=publishedAt&apiKey={api_key}'
)

# Fetch news articles
response = requests.get(url)
news_data = response.json()

print("News Details: ", news_data)

# Check if the request was successful
if news_data['status'] != 'ok':
    print("Error fetching news data:", news_data.get('message', 'Unknown error'))
    exit()

# Extract article titles and descriptions
articles = news_data['articles']
texts = [f"{article['title']}. {article['description']}" for article in articles if article['description']]

# Load FinBERT model and tokenizer
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Initialize sentiment analysis pipeline
sentiment_analyzer = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

sentiments = []
# Analyze sentiment for each article
for text in texts:
    sentiment = sentiment_analyzer(text)
    print(f"Text: {text}")
    print(f"Sentiment: {sentiment}\n")
    print()
    sentiments.append(sentiment[0]['score'] if sentiment[0]['label'] == 'positive' else -sentiment[0]['score'])

# Calculate the average sentiment
average_sentiment = np.mean(sentiments)
print(f"Average Sentiment: {average_sentiment}")