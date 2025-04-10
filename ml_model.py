import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from indicators import calculate_indicators
import joblib

# Fixed feature set
FEATURES = ['SMA_50', 'SMA_200', 'EMA_50', 'EMA_200', 'RSI', 'MACD', 'Signal_Line']

def create_labeled_dataset(stock_id):
    df = yf.download(f"{stock_id}.NS", period="5y", interval="1d")
    df = calculate_indicators(df)

    # Flatten MultiIndex columns if present
    df.columns = ['_'.join([str(c) for c in col if c]).strip() for col in df.columns]

    # Get close column dynamically
    close_col = [col for col in df.columns if 'Close' in col and stock_id in col]
    if not close_col:
        close_col = [col for col in df.columns if 'Close' in col]
    close_col = close_col[0]

    # Target creation
    df['Future_Close'] = df[close_col].shift(-10)
    df.dropna(subset=[close_col, 'Future_Close'], inplace=True)
    df['Target'] = np.where(df['Future_Close'] > df[close_col] * 1.05, 1, 0)

    # Use only selected features
    features = df[FEATURES]
    target = df['Target']
    return features, target

def train_model(stock_id):
    X, y = create_labeled_dataset(stock_id)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Model Evaluation:")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, f"{stock_id}_model.pkl")
    return model

def ml_predict(model, stock_data):
    latest = stock_data[FEATURES].dropna().iloc[-1]
    latest_df = pd.DataFrame([latest.values], columns=FEATURES)

    prediction = model.predict(latest_df)[0]
    confidence = model.predict_proba(latest_df)[0][prediction]

    label = "✅ BUY" if prediction == 1 else "❌ NO BUY"
    print(f"🧠 Confidence: {confidence:.2%}")
    return label
