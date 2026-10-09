"""
A tool for performing sentiment analysis on recent news headlines for a given stock ticker.
It uses a combination of TextBlob and VADER for a more robust sentiment score.
"""
import logging
import re
from typing import Dict

import yfinance as yf
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from tools.data_manager import DataManager

logger = logging.getLogger(__name__)

def _clean_text(text: str) -> str:
    """Removes special characters and extra whitespace for sentiment analysis."""
    text = re.sub(r"[#|@]\S+", "", text)
    text = text.strip()
    return text

def analyze_sentiment(ticker: str) -> Dict:
    """
    Analyzes the sentiment of recent news headlines for a given stock ticker.
    It fetches news from Yahoo Finance, combines TextBlob and VADER sentiment
    scores for a more robust analysis, and returns a structured dictionary.
    Args:
        ticker: The stock ticker symbol (e.g., 'AAPL').
    Returns:
        A dictionary containing the sentiment analysis results.
    """
    news_content = DataManager().get_news_content(ticker)
    if isinstance(news_content, dict) and "error" in news_content:
        return {"sentiment": "neutral", "polarity": 0, "subjectivity": 0, "message": f"Could not fetch news: {news_content['error']}"}
    if not news_content:
        return {"sentiment": "neutral", "polarity": 0, "subjectivity": 0, "message": "No news content found to analyze."}

    try:
        cleaned_text = _clean_text(news_content)

        # TextBlob sentiment
        text_tb = TextBlob(cleaned_text)
        polarity_tb = text_tb.sentiment.polarity
        subjectivity_tb = text_tb.sentiment.subjectivity

        # VADER sentiment
        analyzer = SentimentIntensityAnalyzer()
        scores_vs = analyzer.polarity_scores(cleaned_text)
        polarity_vs = scores_vs['compound']  # Use compound score from VADER

        # Average the polarity scores for a more balanced view
        polarity = (polarity_tb + polarity_vs) / 2

        if polarity > 0.05:
            sentiment = "positive"
        elif polarity < -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return {
            "sentiment": sentiment,
            "polarity": polarity,
            "subjectivity": subjectivity_tb,
            "vader_scores": scores_vs,
            "textblob_polarity": polarity_tb,
        }

    except Exception as e:
        logger.error(f"An unexpected error occurred during sentiment analysis for {ticker}: {e}", exc_info=True)
        return {"error": f"An unexpected error occurred: {str(e)}"}
