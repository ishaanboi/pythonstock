import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def fetch_stock_data(stock_id, period):
    stock_data = yf.download(stock_id + ".NS", period=period)
    if stock_data.empty:
        print(f"Error: No data found for {stock_id}. Check the stock symbol and period.")
        return None
    return stock_data

def calculate_indicators(stock_data):
    # SMA (Simple Moving Average)
    stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['SMA_200'] = stock_data['Close'].rolling(window=200).mean()
    
    # EMA (Exponential Moving Average)
    stock_data['EMA_50'] = stock_data['Close'].ewm(span=50, adjust=False).mean()
    stock_data['EMA_200'] = stock_data['Close'].ewm(span=200, adjust=False).mean()
    
    # RSI (Relative Strength Index)
    delta = stock_data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    stock_data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    stock_data['MACD'] = stock_data['Close'].ewm(span=12, adjust=False).mean() - stock_data['Close'].ewm(span=26, adjust=False).mean()
    stock_data['Signal_Line'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()
    
    return stock_data

def analyze_trend(stock_data):
    if stock_data is None or stock_data.empty:
        return "Data Unavailable", None
    
    score = 0
    
    # SMA Trend
    if stock_data['SMA_50'].iloc[-1] > stock_data['SMA_200'].iloc[-1]:
        score += 1  # Bullish
    else:
        score -= 0.25  # Reduced penalty
    
    # EMA Trend
    if stock_data['EMA_50'].iloc[-1] > stock_data['EMA_200'].iloc[-1]:
        score += 1
    else:
        score -= 0.25  # Reduced penalty
    
    # RSI Trend
    if stock_data['RSI'].iloc[-1] > 70:
        score -= 0.5  # Overbought (potential reversal down)
    elif stock_data['RSI'].iloc[-1] < 30:
        score += 1.5  # Increased weight for oversold conditions
    
    # MACD Trend
    if stock_data['MACD'].iloc[-1] > stock_data['Signal_Line'].iloc[-1]:
        score += 1.5  # Increased weight for positive momentum
    else:
        score -= 0.5  # Less Negative momentum
    
    # Final Decision Based on Score
    if score >= 0:
        recommendation = "Buy"
    else:
        recommendation = "Not a Good Investment"
    
    return recommendation, stock_data

def plot_stock_price(stock_data, stock_id):
    plt.figure(figsize=(12,6))
    plt.plot(stock_data.index, stock_data['Close'], label='Close Price', color='blue')
    plt.plot(stock_data.index, stock_data['SMA_50'], label='50-day SMA', color='orange', linestyle='dashed')
    plt.plot(stock_data.index, stock_data['SMA_200'], label='200-day SMA', color='red', linestyle='dashed')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'{stock_id} Stock Price Over Time')
    plt.legend()
    plt.grid()
    plt.show()

def evaluate_investment(stock_id, period):
    stock_data = fetch_stock_data(stock_id, period)
    if stock_data is None:
        return "Error: No stock data available.", None
    
    stock_data = calculate_indicators(stock_data)
    recommendation, analyzed_data = analyze_trend(stock_data)
    
    plot_stock_price(stock_data, stock_id)
    
    return recommendation, analyzed_data.tail(5) if analyzed_data is not None else None

# Example Usage
stock_id = input("Enter the stock ID (e.g., RELIANCE): ").upper()
period = input("Enter the period for investment (e.g., '1y', '5y', '10y'): ")

recommendation, details = evaluate_investment(stock_id, period)

print(f"\nRecommendation: {recommendation}")
if details is not None:
    print("\nRecent Stock Data for Analysis:")
    print(details[['Close', 'SMA_50', 'SMA_200', 'EMA_50', 'EMA_200', 'RSI', 'MACD', 'Signal_Line']])
