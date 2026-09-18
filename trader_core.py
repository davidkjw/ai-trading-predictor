"""Pure, dependency-light analytics core for the AI Trader app.

This module intentionally avoids Streamlit, plotly, and network access so
that the trading logic can be unit-tested in isolation. The Streamlit app
(ai_trader_fixed.py) delegates to these functions.

Only pandas/numpy are required.
"""

from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd

CRYPTO_HINTS = ("BTC", "ETH", "XRP", "ADA", "SOL")
MEGA_CAP_HINTS = ("AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA")


def create_synthetic_data(symbol: str, days: int = 365) -> pd.DataFrame:
    """Create a realistic OHLCV synthetic price series for ``symbol``.

    Crypto tickers get a higher base price and volatility than mega-cap
    stocks; everything else falls back to neutral settings. The series is
    seeded from the symbol so repeated calls are reproducible.
    """
    symbol_upper = symbol.upper()

    # Seed first so the base-price draw below is reproducible per symbol.
    np.random.seed(hash(symbol) % 10000)

    if any(c in symbol_upper for c in CRYPTO_HINTS):
        base_price = float(np.random.choice([20000, 30000, 40000, 50000]))
        volatility = 0.04
        trend = 0.0003
    elif any(s in symbol_upper for s in MEGA_CAP_HINTS):
        base_price = float(np.random.choice([100, 150, 200, 300, 400]))
        volatility = 0.02
        trend = 0.0001
    else:
        base_price = 100.0
        volatility = 0.015
        trend = 0.00005

    dates = pd.date_range(end=datetime.now(), periods=days, freq="D")
    returns = np.random.normal(trend, volatility, days)
    prices = base_price * np.cumprod(1 + returns)

    df = pd.DataFrame(
        {
            "Open": prices * (1 + np.random.uniform(-0.01, 0.01, days)),
            "High": prices * (1 + np.abs(np.random.uniform(0.01, 0.03, days))),
            "Low": prices * (1 - np.abs(np.random.uniform(0.01, 0.03, days))),
            "Close": prices,
            "Volume": np.random.lognormal(14, 1, days) * (1 + np.abs(returns) * 10),
        },
        index=dates,
    )

    # Ensure High >= Open/Close >= Low
    df["High"] = df[["Open", "High", "Close"]].max(axis=1)
    df["Low"] = df[["Open", "Low", "Close"]].min(axis=1)

    return df


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators (returns, MAs, RSI, MACD, Bollinger) to a copy."""
    data = df.copy()
    price_col = "Close"

    # Basic returns
    data["returns"] = data[price_col].pct_change()

    # Moving Averages
    for window in (5, 10, 20, 50, 200):
        if len(data) >= window:
            data[f"MA_{window}"] = data[price_col].rolling(window).mean()

    # RSI (14-period, Wilder-free rolling approximation)
    delta = data[price_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    data["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = data[price_col].ewm(span=12, adjust=False).mean()
    exp2 = data[price_col].ewm(span=26, adjust=False).mean()
    data["MACD"] = exp1 - exp2
    data["MACD_signal"] = data["MACD"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    if len(data) >= 20:
        data["BB_middle"] = data[price_col].rolling(20).mean()
        bb_std = data[price_col].rolling(20).std()
        data["BB_upper"] = data["BB_middle"] + (bb_std * 2)
        data["BB_lower"] = data["BB_middle"] - (bb_std * 2)

    # Volume
    if "Volume" in data.columns:
        data["volume_MA"] = data["Volume"].rolling(20).mean()
        data["volume_ratio"] = data["Volume"] / data["volume_MA"]

    data = data.dropna()
    return data


def predict_price(df: pd.DataFrame, forecast_days: int) -> float:
    """Predict a future price.

    Blends a linear trend extrapolation over the last 10 closes (weight 0.6)
    with a 20-day moving average projection (weight 0.4).
    """
    try:
        current_price = float(df["Close"].iloc[-1])

        last_10 = df["Close"].iloc[-10:].values
        if len(last_10) >= 2:
            x = np.arange(len(last_10))
            coeffs = np.polyfit(x, last_10, 1)
            trend_pred = np.polyval(coeffs, len(last_10) + forecast_days)
        else:
            trend_pred = current_price * 1.02

        ma_20 = float(df["MA_20"].iloc[-1]) if "MA_20" in df.columns else current_price
        ma_projection = ma_20 * 1.01

        return float(np.average([trend_pred, ma_projection], weights=[0.6, 0.4]))
    except Exception:
        return float(df["Close"].iloc[-1]) * 1.02


def calculate_risk_score(latest: pd.Series, pct_change: float) -> float:
    """Risk score in [0, 100]; higher is riskier.

    Base 50, +20 if RSI > 70 (overbought), -10 if RSI < 30, +15 for moves
    beyond 10%, +5 for moves beyond 5%.
    """
    score = 50.0
    rsi = float(latest.get("RSI", 50))
    if rsi > 70:
        score += 20
    elif rsi < 30:
        score -= 10
    if abs(pct_change) > 10:
        score += 15
    elif abs(pct_change) > 5:
        score += 5
    return min(max(score, 0.0), 100.0)


def generate_signal(pct_change: float, rsi: float, current_price: float,
                    ma_20: float, ma_50: float) -> Dict[str, str]:
    """Map predicted move + indicator state to a trade signal."""
    if pct_change > 8 and rsi < 70 and current_price > ma_20:
        return {"signal": "🟢 STRONG BUY", "signal_color": "green", "confidence": "HIGH"}
    if pct_change > 3 and rsi < 75:
        return {"signal": "🟡 MODERATE BUY", "signal_color": "yellow", "confidence": "MEDIUM"}
    if pct_change < -8 and rsi > 30:
        return {"signal": "🔴 STRONG SELL", "signal_color": "red", "confidence": "HIGH"}
    if pct_change < -3 and rsi > 25:
        return {"signal": "🟠 MODERATE SELL", "signal_color": "orange", "confidence": "MEDIUM"}
    return {"signal": "⚪ HOLD", "signal_color": "gray", "confidence": "LOW"}


def create_recommendation(signal: str, current_price: float, predicted_price: float,
                          pct_change: float, forecast_days: int,
                          risk_score: float) -> Dict:
    """Build the actionable recommendation dict (entry, stop, target, actions)."""
    if "BUY" in signal:
        entry_price = current_price
        stop_loss = current_price * 0.95
        take_profit = predicted_price
        position_size = "70-80% of capital"
    elif "SELL" in signal:
        entry_price = current_price
        stop_loss = current_price * 1.05
        take_profit = predicted_price
        position_size = "50-60% of capital"
    else:
        entry_price = "N/A"
        stop_loss = "N/A"
        take_profit = "N/A"
        position_size = "Maintain current"

    risk_reward = (abs((take_profit - entry_price) / (entry_price - stop_loss))
                   if isinstance(take_profit, (int, float)) else 0)
    time_horizon = "Short-term" if forecast_days <= 7 else "Medium-term"

    reasons = []
    if pct_change > 5:
        reasons.append(f"Strong upside potential ({pct_change:.1f}%)")
    if "BUY" in signal:
        reasons.append("Bullish technical setup")
    if "SELL" in signal:
        reasons.append("Bearish market conditions")
    if not reasons:
        reasons = ["Market appears neutral", "Wait for clearer signals"]

    if "STRONG BUY" in signal:
        actions = ["Enter long position", "Set 5% stop-loss",
                   "Target profit at predicted price", "Consider adding on dips"]
    elif "MODERATE BUY" in signal:
        actions = ["Enter partial position", "Use tighter 3-4% stop-loss",
                   "Take partial profits", "Wait for confirmation"]
    elif "STRONG SELL" in signal:
        actions = ["Consider short position", "Set 5% stop-loss",
                   "Target support levels", "Consider put options"]
    elif "MODERATE SELL" in signal:
        actions = ["Reduce long exposure", "Set breakeven stop",
                   "Take partial profits", "Wait for better entry"]
    else:
        actions = ["Hold existing positions", "Wait for market direction",
                   "Dollar-cost average if long-term", "Monitor key levels"]

    return {
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "risk_reward_ratio": round(risk_reward, 2) if risk_reward > 0 else "N/A",
        "position_size": position_size,
        "time_horizon": time_horizon,
        "key_reasons": reasons,
        "suggested_actions": actions,
    }


def generate_fallback_recommendation(df: pd.DataFrame) -> Dict:
    """Safe HOLD recommendation used when there is insufficient data."""
    current_price = float(df["Close"].iloc[-1]) if len(df) > 0 else 100.0
    return {
        "current_price": current_price,
        "predicted_price": current_price * 1.02,
        "pct_change": 2.0,
        "signal": "⚪ HOLD",
        "signal_color": "gray",
        "confidence": "LOW",
        "risk_score": 50,
        "recommendation": {
            "entry_price": "N/A",
            "stop_loss": "N/A",
            "take_profit": "N/A",
            "risk_reward_ratio": "N/A",
            "position_size": "Wait for signals",
            "time_horizon": "Short-term",
            "key_reasons": ["Insufficient data"],
            "suggested_actions": ["Wait for more data"],
        },
        "indicators": {},
        "is_synthetic": True,
    }


def generate_ai_recommendation(df: pd.DataFrame, forecast_days: int = 7) -> Dict:
    """Full pipeline: indicators -> prediction -> signal -> recommendation."""
    if len(df) < 50:
        return generate_fallback_recommendation(df)

    try:
        features_df = calculate_technical_indicators(df)
        latest = features_df.iloc[-1]
        current_price = float(latest["Close"])

        predicted_price = predict_price(features_df, forecast_days)
        pct_change = ((predicted_price - current_price) / current_price) * 100

        rsi = float(latest.get("RSI", 50))
        ma_20 = float(latest.get("MA_20", current_price))
        ma_50 = float(latest.get("MA_50", current_price))

        sig = generate_signal(pct_change, rsi, current_price, ma_20, ma_50)
        risk_score = calculate_risk_score(latest, pct_change)
        recommendation = create_recommendation(
            sig["signal"], current_price, predicted_price,
            pct_change, forecast_days, risk_score,
        )

        return {
            "current_price": current_price,
            "predicted_price": predicted_price,
            "pct_change": pct_change,
            "signal": sig["signal"],
            "signal_color": sig["signal_color"],
            "confidence": sig["confidence"],
            "risk_score": risk_score,
            "recommendation": recommendation,
            "indicators": {
                "RSI": round(rsi, 2),
                "MA_20": round(ma_20, 2),
                "MA_50": round(ma_50, 2),
                "MACD": round(float(latest.get("MACD", 0)), 3),
            },
            "is_synthetic": (bool(df["IsSynthetic"].iloc[-1])
                             if "IsSynthetic" in df.columns else True),
        }
    except Exception as e:  # pragma: no cover - defensive
        print(f"AI recommendation error: {e}")
        return generate_fallback_recommendation(df)
