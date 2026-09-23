# 🤖 AI Trading Predictor

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.22%2B-red)
![ML](https://img.shields.io/badge/Machine%20Learning-XGBoost-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

**Predict stock & cryptocurrency prices with machine learning and comprehensive technical analysis**

---

## 🎯 What This Does

AI Asset Price Predictor is a professional trading assistant that uses machine learning to forecast prices of stocks and cryptocurrencies. It combines:

- **🤖 AI Predictions** - XGBoost model forecasts 7-day prices
- **📊 Technical Analysis** - 25+ indicators including RSI, MACD, Bollinger Bands
- **🎯 Trading Signals** - BUY/SELL/HOLD recommendations with confidence levels
- **🛡️ Risk Management** - Complete trading plans with stop-loss & take-profit
- **📈 Real-time Data** - Multiple API sources with automatic fallback

---

## ⚙️ How It Works

#### 1. The Core Objective
The application is designed for **high availability**. In financial applications, external APIs are notoriously unreliable due to rate limits or downtime. This system is "resilient" by design; if real-time data fetching fails, it automatically triggers a **Synthetic Data Engine** to generate realistic market movements based on historical volatility, ensuring the platform never crashes and remains fully testable.

#### 2. Technical Architecture

* **The Data Fetcher (`RealDataFetcher`)**
    Think of this as the "Librarian" of the system. It manages data through a three-tier priority system:
    1.  **Local Cache:** Checks for existing `.parquet` or `.csv` files to minimize API calls and latency.
    2.  **Real-Time APIs:** Authenticates with professional providers (Finnhub, Alpha Vantage, Twelve Data) using secure API keys.
    3.  **Synthetic Engine:** A mathematical fallback that generates price history if keys are missing or limits are reached.
    4.  **Windows-Safe File Handling:** Implements specialized permission handling (`os.chmod`) to prevent "File in Use" errors common on Windows environments during cache clearing.



* **The AI Engine (`AITraderEngine`)**
    This is the "Brain" that processes raw data into actionable intelligence:
    * **Feature Engineering:** Calculates 25+ technical indicators, including **RSI** (momentum), **Moving Averages** (trend), and **MACD** (trend shifts).
    * **Predictive Modeling:** Utilizes the **XGBoost** machine learning algorithm combined with polynomial trend extrapolation to forecast prices for a 7 to 30-day window.
    * **Signal Generation:** Maps mathematical outputs to human-readable signals like `🟢 STRONG BUY`, `🔴 SELL`, or `⚪ HOLD`.

* **The Dashboard (UI Layer)**
    Built with **Streamlit**, the dashboard provides a professional-grade interface:
    * **Sidebar:** Real-time configuration of symbols, forecast horizons, and live API key updates without app restarts.
    * **Main Console:** High-density "Metric Cards" and interactive **Plotly** charts (Candlesticks, Volume, and Momentum oscillators).



#### 3. Key Engineering Features

| Feature | Technical Implementation |
| :--- | :--- |
| **API Key Hot-Swapping** | JSON-based persistence allows keys to take effect immediately in the session state. |
| **Intelligent Risk Scoring** | A custom algorithm that calculates a 0–100 risk score based on volatility and RSI extremes. |
| **Interactive Visuals** | Multi-pane Plotly subplots that mimic professional terminals (Bloomberg/TradingView style). |
| **State Management** | Uses `lru_cache` to optimize performance and prevent redundant heavy computations. |

---

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

---

## 🛠️ Tech Stack

The platform is built using a modern Python-based stack designed for data science and high-performance financial analysis.

| Category | Technology | Usage in Project |
| :--- | :--- | :--- |
| **Language** | ![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white) | Core logic, API integration, and mathematical modeling. |
| **Frontend** | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white) | Dashboard UI with custom CSS injection for "Dark Mode" and Metric Cards. |
| **ML Engine** | ![XGBoost](https://img.shields.io/badge/XGBoost-black?style=flat) ![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white) | Gradient boosting for price forecasting and data preprocessing. |
| **Data Viz** | ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white) | Interactive candlestick charts and multi-pane technical indicators. |
| **Data Handling** | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) | Heavy-duty time-series analysis and technical indicator calculation. |
| **Storage** | ![Parquet](https://img.shields.io/badge/Apache_Parquet-black?style=flat) | Fast, column-oriented caching to minimize disk I/O and latency. |

---

## 📁 Project Structure

The project is split into a **testable analytics core** and a **thin UI layer**, so the trading logic can be unit-tested in CI without a browser, API keys, or network access.

```
├── ai_trader_fixed.py        # Streamlit UI + data fetching (RealDataFetcher)
├── trader_core.py            # Pure analytics: indicators, prediction, signals, risk
├── conftest.py               # Pytest bootstrap (makes repo root importable)
├── requirements.txt          # Dependencies (pinned with >= ranges)
├── tests/
│   └── test_trader_core.py   # 23 unit tests for the analytics core
└── .github/workflows/
    └── test.yml              # CI: runs the pytest suite on every push/PR
```

**Why the split:** `ai_trader_fixed.py` used to hold all decision-making logic inline. It now *delegates* to `trader_core.py`, which depends only on `pandas` + `numpy`. That's what makes the indicators, prediction, risk scoring, and signal mapping verifiable in CI.

## 🧪 Testing

The analytics core has a unit-test suite covering indicators (RSI bounds, Bollinger ordering, moving averages), price prediction, risk scoring, signal mapping, and the full recommendation pipeline.

```bash
pip install -r requirements.txt
pytest tests -v
```

CI runs this suite automatically on every push and pull request.

---

## 🚀 Quick Installation

```bash
# Clone the repository
git clone https://github.com/davidkjw/ai-trading-predictor.git
cd ai-trading-predictor

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run ai_trader_fixed.py
```


## ⭐ Love this tool? Give it a star on GitHub! ⭐
