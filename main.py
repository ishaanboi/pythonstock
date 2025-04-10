from fetch import fetch_stock_data
from indicators import calculate_indicators
from ml_model import train_model, ml_predict
import os
import joblib
from pathlib import Path

# 🔁 Helper to normalize user input like '3' → '3y'
def normalize_period(period):
    if period.isdigit():
        return f"{period}y"
    return period

def evaluate():
    stock_id = input("Enter stock ID (e.g., RELIANCE): ").upper()
    period = normalize_period(input("Enter period (e.g., 1y, 5y or just 3): "))

    model_path = f"{stock_id}_model.pkl"

    # 🧹 Always delete old model to avoid feature mismatch
    if os.path.exists(model_path):
        print(f"🧹 Deleting old model for {stock_id} to fix feature mismatch...")
        os.remove(model_path)

    stock_data = fetch_stock_data(stock_id, period)
    if stock_data is None:
        print("Error fetching stock data.")
        return

    stock_data = calculate_indicators(stock_data)

    print(f"🔄 Training model for {stock_id}...")
    model = train_model(stock_id)
    joblib.dump(model, model_path)

    if model:
        recommendation = ml_predict(model, stock_data)
        print(f"\n📈 ML Recommendation for {stock_id}: {recommendation}")
    else:
        print("❌ Failed to train ML model.")

if __name__ == "__main__":
    evaluate()
