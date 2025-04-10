# 📈 PythonStock - Stock Trend Analyzer

PythonStock is a simple Python-based CLI tool to help investors analyze Indian stock trends using technical indicators such as SMA, EMA, RSI, and MACD. It fetches historical data using the Yahoo Finance API and provides a basic **Buy or Not a Good Investment** recommendation.

---

## 🚀 Features

- 📊 Fetches live stock data using `yfinance`
- 📈 Calculates technical indicators:
  - Simple Moving Averages (SMA 50 & SMA 200)
  - Exponential Moving Averages (EMA 50 & EMA 200)
  - Relative Strength Index (RSI)
  - MACD & Signal Line
- 🧠 Analyzes trends and provides investment recommendations
- 📉 Plots price chart with indicators for visual analysis
- ✅ Works with Indian stocks listed on NSE

---

## 🛠️ Requirements

Make sure you have Python 3 installed. Install dependencies using pip:

```bash
pip install yfinance pandas numpy matplotlib
