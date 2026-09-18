"""Unit tests for trader_core (pure analytics, no Streamlit/network)."""

import numpy as np
import pandas as pd
import pytest

import trader_core as tc


def make_df(prices):
    """OHLCV frame from a close-price series."""
    n = len(prices)
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    prices = np.asarray(prices, dtype=float)
    return pd.DataFrame(
        {
            "Open": prices,
            "High": prices * 1.01,
            "Low": prices * 0.99,
            "Close": prices,
            "Volume": np.full(n, 1_000_000.0),
        },
        index=dates,
    )


# --------------------------------------------------------------------------- #
# Synthetic data
# --------------------------------------------------------------------------- #
def test_synthetic_data_shape_and_columns():
    df = tc.create_synthetic_data("AAPL", days=300)
    assert len(df) == 300
    for col in ("Open", "High", "Low", "Close", "Volume"):
        assert col in df.columns
    assert (df["Close"] > 0).all()


def test_synthetic_data_ohlc_consistency():
    df = tc.create_synthetic_data("BTC-USD", days=200)
    assert (df["High"] >= df[["Open", "Close"]].max(axis=1) - 1e-9).all()
    assert (df["Low"] <= df[["Open", "Close"]].min(axis=1) + 1e-9).all()


def test_synthetic_data_reproducible_per_symbol():
    a = tc.create_synthetic_data("MSFT", days=100)
    b = tc.create_synthetic_data("MSFT", days=100)
    np.testing.assert_array_equal(a["Close"].to_numpy(), b["Close"].to_numpy())


def test_synthetic_data_crypto_base_price_higher():
    crypto = tc.create_synthetic_data("BTC-USD", days=500)["Close"].iloc[0]
    other = tc.create_synthetic_data("XYZ", days=500)["Close"].iloc[0]
    assert crypto > other


# --------------------------------------------------------------------------- #
# Technical indicators
# --------------------------------------------------------------------------- #
def test_indicators_columns_present():
    df = tc.calculate_technical_indicators(make_df(np.linspace(100, 200, 300)))
    for col in ("returns", "MA_5", "MA_20", "MA_200", "RSI", "MACD",
                "MACD_signal", "BB_upper", "BB_lower"):
        assert col in df.columns


def test_indicators_no_nan_after_dropana():
    df = tc.calculate_technical_indicators(make_df(np.linspace(100, 200, 300)))
    assert not df.isna().any().any()


def test_rsi_bounds():
    noisy = 100 + np.cumsum(np.random.RandomState(7).normal(0, 1, 300))
    df = tc.calculate_technical_indicators(make_df(noisy))
    rsi = df["RSI"].dropna()
    assert ((rsi >= 0) & (rsi <= 100)).all()


def test_rsi_high_on_strong_uptrend():
    df = tc.calculate_technical_indicators(make_df(np.linspace(100, 500, 300)))
    assert df["RSI"].iloc[-1] > 90


def test_bollinger_ordering():
    df = tc.calculate_technical_indicators(make_df(np.linspace(100, 200, 300)))
    assert (df["BB_upper"] >= df["BB_middle"]).all()
    assert (df["BB_middle"] >= df["BB_lower"]).all()


def test_short_series_skips_long_moving_averages():
    df = tc.calculate_technical_indicators(make_df(np.linspace(100, 110, 30)))
    assert "MA_200" not in df.columns
    assert "MA_5" in df.columns


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
def test_predict_price_extrapolates_uptrend():
    feats = tc.calculate_technical_indicators(make_df(np.linspace(100, 200, 300)))
    pred = tc.predict_price(feats, forecast_days=7)
    assert pred > feats["Close"].iloc[-1]


def test_predict_price_flat_series_stays_near_price():
    feats = tc.calculate_technical_indicators(make_df(np.full(200, 150.0)))
    pred = tc.predict_price(feats, forecast_days=7)
    assert 145 < pred < 155


# --------------------------------------------------------------------------- #
# Risk score
# --------------------------------------------------------------------------- #
def test_risk_score_base():
    s = tc.calculate_risk_score(pd.Series({"RSI": 50}), pct_change=1.0)
    assert s == 50.0


def test_risk_score_overbought_and_volatile():
    s = tc.calculate_risk_score(pd.Series({"RSI": 80}), pct_change=12.0)
    assert s == 85.0


def test_risk_score_clamped_to_bounds():
    s = tc.calculate_risk_score(pd.Series({"RSI": 10}), pct_change=-12.0)
    assert 0 <= s <= 100


# --------------------------------------------------------------------------- #
# Signal mapping
# --------------------------------------------------------------------------- #
def test_signal_strong_buy():
    sig = tc.generate_signal(10.0, 55.0, 105.0, 100.0, 95.0)
    assert "STRONG BUY" in sig["signal"] and sig["confidence"] == "HIGH"


def test_signal_hold_when_neutral():
    sig = tc.generate_signal(1.0, 50.0, 100.0, 100.0, 100.0)
    assert "HOLD" in sig["signal"]


def test_signal_strong_sell():
    sig = tc.generate_signal(-10.0, 45.0, 90.0, 100.0, 105.0)
    assert "STRONG SELL" in sig["signal"]


# --------------------------------------------------------------------------- #
# Recommendation
# --------------------------------------------------------------------------- #
def test_recommendation_buy_levels():
    rec = tc.create_recommendation("🟢 STRONG BUY", 100.0, 120.0,
                                   20.0, 7, 50.0)
    assert rec["stop_loss"] == pytest.approx(95.0)
    assert rec["take_profit"] == 120.0
    assert rec["risk_reward_ratio"] == pytest.approx(4.0)
    assert rec["time_horizon"] == "Short-term"


def test_recommendation_sell_levels():
    rec = tc.create_recommendation("🔴 STRONG SELL", 100.0, 80.0,
                                   -20.0, 14, 50.0)
    assert rec["stop_loss"] == pytest.approx(105.0)
    assert rec["time_horizon"] == "Medium-term"


def test_recommendation_hold_na_levels():
    rec = tc.create_recommendation("⚪ HOLD", 100.0, 102.0, 0.5, 7, 50.0)
    assert rec["entry_price"] == "N/A"
    assert rec["position_size"] == "Maintain current"


# --------------------------------------------------------------------------- #
# End-to-end pipeline
# --------------------------------------------------------------------------- #
def test_pipeline_returns_full_contract():
    df = tc.create_synthetic_data("AAPL", days=400)
    out = tc.generate_ai_recommendation(df, forecast_days=7)
    for key in ("current_price", "predicted_price", "pct_change", "signal",
                "signal_color", "confidence", "risk_score", "recommendation",
                "indicators", "is_synthetic"):
        assert key in out
    assert 0 <= out["risk_score"] <= 100


def test_pipeline_fallback_on_short_data():
    df = make_df(np.linspace(100, 110, 20))
    out = tc.generate_ai_recommendation(df)
    assert "HOLD" in out["signal"]
    assert out["recommendation"]["key_reasons"] == ["Insufficient data"]
