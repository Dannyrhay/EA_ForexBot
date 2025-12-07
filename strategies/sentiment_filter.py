import requests
import logging
from datetime import datetime, timedelta
from textblob import TextBlob # Assuming TextBlob is available, or we use simple keyword scoring

logger = logging.getLogger(__name__)

class SentimentFilter:
    """
    Fetches news via NewsAPI and calculates a Sentiment Score.
    Acts as a Circuit Breaker: If Sentiment is highly negative, BLOCK trades.
    """
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2/everything"
        self.cache = {}
        self.cache_duration = timedelta(minutes=30) # Cache news for 30 mins to save API calls

    def get_sentiment(self, symbol):
        """
        Returns a sentiment score between -1.0 (Negative) and 1.0 (Positive).
        Returns 0.0 if neutral or no news.
        """
        # Map symbol to search query
        query = self._map_symbol_to_query(symbol)
        if not query:
            return 0.0

        # Check Cache
        if query in self.cache:
            last_time, score = self.cache[query]
            if datetime.now() - last_time < self.cache_duration:
                return score

        # Fetch News
        try:
            params = {
                'q': query,
                'from': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
                'sortBy': 'publishedAt',
                'language': 'en',
                'apiKey': self.api_key
            }
            response = requests.get(self.base_url, params=params)
            data = response.json()

            if data.get('status') != 'ok':
                logger.error(f"NewsAPI Error: {data.get('message')}")
                return 0.0

            articles = data.get('articles', [])
            if not articles:
                return 0.0

            # Calculate Score
            total_score = 0
            count = 0

            for article in articles[:10]: # Analyze top 10 articles
                title = article.get('title', '')
                description = article.get('description', '')
                text = f"{title} {description}"

                # Simple Sentiment Analysis
                blob = TextBlob(text)
                total_score += blob.sentiment.polarity
                count += 1

            final_score = total_score / count if count > 0 else 0.0

            # Update Cache
            self.cache[query] = (datetime.now(), final_score)

            logger.info(f"Sentiment for {symbol} ({query}): {final_score:.2f}")
            return final_score

        except Exception as e:
            logger.error(f"Error fetching sentiment: {e}")
            return 0.0

    def _map_symbol_to_query(self, symbol):
        """
        Maps forex/crypto symbols to news keywords.
        """
        if "EUR" in symbol: return "Euro OR ECB OR European Economy"
        if "USD" in symbol: return "USD OR Federal Reserve OR US Economy"
        if "GBP" in symbol: return "GBP OR Bank of England OR UK Economy"
        if "JPY" in symbol: return "JPY OR Bank of Japan OR Yen"
        if "BTC" in symbol: return "Bitcoin OR Crypto OR SEC"
        if "ETH" in symbol: return "Ethereum"
        if "XAU" in symbol: return "Gold Price OR XAUUSD"
        return symbol # Fallback
