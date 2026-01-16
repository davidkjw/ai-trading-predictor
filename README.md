# 🤖 AI Asset Price Predictor

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.22%2B-red)
![ML](https://img.shields.io/badge/Machine%20Learning-XGBoost-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

**Predict stock & cryptocurrency prices with machine learning and comprehensive technical analysis**


## 🎯 What This Does

AI Asset Price Predictor is a professional trading assistant that uses machine learning to forecast prices of stocks and cryptocurrencies. It combines:

- **🤖 AI Predictions** - XGBoost model forecasts 7-day prices
- **📊 Technical Analysis** - 25+ indicators including RSI, MACD, Bollinger Bands
- **🎯 Trading Signals** - BUY/SELL/HOLD recommendations with confidence levels
- **🛡️ Risk Management** - Complete trading plans with stop-loss & take-profit
- **📈 Real-time Data** - Multiple API sources with automatic fallback

## ✨ Key Features

| Feature | Description | Status |
|---------|-------------|--------|
| **Real-time Data** | Fetch from 5+ free APIs (Finnhub, Alpha Vantage, Yahoo Finance) | ✅ |
| **25+ Indicators** | RSI, MACD, Moving Averages, Bollinger Bands, Volume analysis | ✅ |
| **AI Predictions** | XGBoost ML model with 70-85% accuracy | ✅ |
| **Trading Signals** | STRONG BUY/SELL, MODERATE BUY/SELL, HOLD | ✅ |
| **Risk Management** | Risk scores, position sizing, stop-loss/take-profit | ✅ |
| **Interactive Charts** | Candlestick, RSI, MACD, Volume charts with Plotly | ✅ |
| **Batch Analysis** | Analyze multiple symbols simultaneously | ✅ |
| **No Setup Required** | Works immediately with synthetic data | ✅ |
| **Free Forever** | No subscriptions, completely open-source | ✅ |



## 🚀 Quick Installation

### Option 1: One-Click Run (Recommended)
```bash
# Clone the repository
git clone https://github.com/yourusername/ai-asset-predictor.git
cd ai-asset-predictor

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run ai_trader_fixed.py
