# fetch.py
import yfinance as yf

def fetch_stock_data(stock_id, period):
    df = yf.download(f"{stock_id}.NS", period=period)
    if df.empty:
        print(f"⚠️ No data found for {stock_id}")
        return None
    return df
