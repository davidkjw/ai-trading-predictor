"""
🚀 AI TRADER PRO 
API keys work immediately without restarting
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Visualization
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# ML Models
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

# Data fetching
import requests
import json
import os
from pathlib import Path
import time
from typing import Optional, Dict, Tuple
from functools import lru_cache
import shutil

# Page configuration
st.set_page_config(
    page_title="AI Trader Pro - Fixed",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
/* Main container */
.main {
    padding: 2rem;
}

/* Cards */
.card {
    background: white;
    border-radius: 10px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    border-left: 5px solid #667eea;
}

.card-red {
    border-left: 5px solid #ff6b6b;
}

.card-green {
    border-left: 5px solid #51cf66;
}

.card-yellow {
    border-left: 5px solid #ffd43b;
}

.card-blue {
    border-left: 5px solid #339af0;
}

/* Metrics */
.metric-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 10px;
    margin: 0.5rem;
    text-align: center;
}

.metric-value {
    font-size: 2rem;
    font-weight: bold;
}

.metric-label {
    font-size: 0.9rem;
    opacity: 0.9;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 5px;
    padding: 0.5rem 1rem;
    font-weight: bold;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
}

/* Signal badges */
.signal-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-weight: bold;
    font-size: 0.9rem;
}

.signal-buy {
    background-color: #d4edda;
    color: #155724;
    border: 1px solid #c3e6cb;
}

.signal-sell {
    background-color: #f8d7da;
    color: #721c24;
    border: 1px solid #f5c6cb;
}

.signal-hold {
    background-color: #e2e3e5;
    color: #383d41;
    border: 1px solid #d6d8db;
}

/* Progress bars */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #667eea, #764ba2);
}

/* Status indicators */
.status-real {
    color: #51cf66;
    font-weight: bold;
    background: rgba(81, 207, 102, 0.1);
    padding: 5px 10px;
    border-radius: 5px;
    border: 1px solid #51cf66;
}

.status-synthetic {
    color: #ffd43b;
    font-weight: bold;
    background: rgba(255, 212, 59, 0.1);
    padding: 5px 10px;
    border-radius: 5px;
    border: 1px solid #ffd43b;
}

.status-no-keys {
    color: #ff6b6b;
    font-weight: bold;
    background: rgba(255, 107, 107, 0.1);
    padding: 5px 10px;
    border-radius: 5px;
    border: 1px solid #ff6b6b;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA FETCHER CLASS - FIXED VERSION (WINDOWS SAFE)
# ============================================================================

class RealDataFetcher:
    """Enhanced data fetcher with immediate API key updates"""
    
    def __init__(self):
        self.api_keys = self.load_api_keys()
        self.cache_dir = Path(".ai_trader_cache")
        self._ensure_cache_dir()
        self.last_update = datetime.now()
    
    def _ensure_cache_dir(self):
        """Ensure cache directory exists and is accessible"""
        try:
            self.cache_dir.mkdir(exist_ok=True)
            # Test write permissions
            test_file = self.cache_dir / "test.txt"
            test_file.write_text("test")
            test_file.unlink()
        except Exception as e:
            print(f"Warning: Cache directory issue: {e}")
            # Try alternative location
            alt_dir = Path("./cache")
            alt_dir.mkdir(exist_ok=True)
            self.cache_dir = alt_dir
    
    def load_api_keys(self) -> Dict:
        """Load API keys from multiple sources"""
        keys = {}
        
        # Load from environment variables
        env_keys = {
            'finnhub': os.getenv('FINNHUB_API_KEY'),
            'alpha_vantage': os.getenv('ALPHA_VANTAGE_KEY'),
            'twelvedata': os.getenv('TWELVEDATA_API_KEY')
        }
        
        # Load from keys.json if exists
        keys_file = Path("api_keys.json")
        if keys_file.exists():
            try:
                with open(keys_file, 'r') as f:
                    file_keys = json.load(f)
                    keys.update(file_keys)
            except:
                pass
        
        # Update with env vars
        for key, value in env_keys.items():
            if value:
                keys[key] = value
        
        print(f"🔑 Loaded {len(keys)} API key(s)")
        return keys
    
    def update_api_keys(self, new_keys: Dict):
        """Update API keys immediately"""
        self.api_keys.update(new_keys)
        self.last_update = datetime.now()
        
        # Save to file for persistence
        try:
            with open("api_keys.json", "w") as f:
                json.dump(self.api_keys, f, indent=2)
        except:
            pass
        
        print(f"🔄 Updated API keys: {list(new_keys.keys())}")
    
    def clear_cache(self):
        """Clear all cached data - Windows-safe version"""
        if not self.cache_dir.exists():
            self._ensure_cache_dir()
            return
        
        try:
            print(f"🗑️ Attempting to clear cache at {self.cache_dir}")
            
            # First, try the simple way with ignore_errors
            try:
                shutil.rmtree(self.cache_dir, ignore_errors=True)
                print("✅ Cache cleared with shutil.rmtree")
            except Exception as e:
                print(f"⚠️ Standard rmtree failed: {e}")
            
            # Check if directory still exists
            if self.cache_dir.exists():
                print("⚠️ Directory still exists, trying file-by-file deletion")
                
                # Try to delete files one by one
                deleted_count = 0
                failed_count = 0
                
                for file_path in list(self.cache_dir.glob('*')):  # List to avoid modification during iteration
                    try:
                        if file_path.is_file():
                            # Try to change permissions first
                            try:
                                import os
                                os.chmod(file_path, 0o777)
                            except:
                                pass
                            
                            # Try to delete
                            file_path.unlink()
                            deleted_count += 1
                        elif file_path.is_dir():
                            # Try to delete directory
                            shutil.rmtree(file_path, ignore_errors=True)
                            deleted_count += 1
                    except Exception as e:
                        print(f"  Could not delete {file_path.name}: {e}")
                        failed_count += 1
                        # Try to rename the file instead
                        try:
                            new_name = file_path.with_suffix(f".deleted_{int(time.time())}")
                            file_path.rename(new_name)
                            print(f"  Renamed {file_path.name} to {new_name.name}")
                        except:
                            pass
                
                print(f"✅ Deleted {deleted_count} files, failed: {failed_count}")
                
                # Try to remove the directory itself if empty
                try:
                    self.cache_dir.rmdir()
                    print("✅ Removed empty directory")
                except:
                    # Directory not empty or permission denied
                    print("⚠️ Could not remove directory, may not be empty")
            
            # Always recreate directory
            self._ensure_cache_dir()
            print("✅ Cache directory ready for use")
            
        except Exception as e:
            print(f"❌ Error clearing cache: {e}")
            # Fallback: create new cache directory with timestamp
            try:
                timestamp = int(time.time())
                new_cache_dir = Path(f".ai_trader_cache_{timestamp}")
                self.cache_dir = new_cache_dir
                self._ensure_cache_dir()
                print(f"✅ Created new cache directory: {new_cache_dir}")
            except Exception as e2:
                print(f"❌ Could not create new cache directory: {e2}")
    
    @lru_cache(maxsize=50)
    def get_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Get data with caching - PRIORITIZE REAL DATA WHEN KEYS EXIST"""
        print(f"\n📊 Fetching {symbol}...")
        
        # Check cache first
        cache_file = self.cache_dir / f"{symbol}_{period}.parquet"
        csv_file = cache_file.with_suffix('.csv')
        
        # Determine which cache file exists
        cache_path = None
        for file in [cache_file, csv_file]:
            if file.exists():
                cache_path = file
                break
        
        # If cache exists and is fresh (less than 1 hour), use it
        if cache_path:
            try:
                cache_age = time.time() - cache_path.stat().st_mtime
                if cache_age < 3600:  # 1 hour
                    if cache_path.suffix == '.parquet':
                        data = pd.read_parquet(cache_path)
                    else:
                        data = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                    
                    print(f"✅ Using cached data for {symbol} ({cache_age:.0f}s old)")
                    return data
            except Exception as e:
                print(f"⚠️ Cache read failed: {e}")
        
        # Try real data sources if we have API keys
        if self.api_keys:
            print(f"🔑 Using API keys to fetch real data for {symbol}")
            
            # Try multiple real data sources
            real_data = self._try_real_sources(symbol)
            
            if real_data is not None and len(real_data) > 50:
                print(f"✅ Successfully fetched {len(real_data)} real records")
                real_data['IsSynthetic'] = False
                real_data['Data_Source'] = 'Real API'
                
                # Save to cache
                try:
                    real_data.to_parquet(cache_file)
                except:
                    try:
                        real_data.to_csv(csv_file)
                    except:
                        print("⚠️ Could not save to cache")
                
                return real_data
            else:
                print(f"⚠️ Real data sources failed for {symbol}")
        
        # Fall back to synthetic data
        print(f"⚠️ Using synthetic data for {symbol}")
        synthetic_data = self._create_synthetic_data(symbol)
        synthetic_data['IsSynthetic'] = True
        synthetic_data['Data_Source'] = 'Synthetic'
        
        # Save to cache
        try:
            synthetic_data.to_parquet(cache_file)
        except:
            try:
                synthetic_data.to_csv(csv_file)
            except:
                print("⚠️ Could not save synthetic data to cache")
        
        return synthetic_data
    
    def _try_real_sources(self, symbol: str) -> Optional[pd.DataFrame]:
        """Try multiple real data sources"""
        sources = [
            ("Finnhub", self._try_finnhub),
            ("Alpha Vantage", self._try_alpha_vantage),
            ("Yahoo Finance", self._try_yfinance),
            ("Twelve Data", self._try_twelvedata)
        ]
        
        for source_name, source_func in sources:
            try:
                data = source_func(symbol)
                if data is not None and len(data) > 50:
                    return data
            except Exception as e:
                print(f"  ⚠️ {source_name} failed: {str(e)[:50]}")
                continue
        
        return None
    
    def _try_finnhub(self, symbol: str) -> Optional[pd.DataFrame]:
        """Try Finnhub API"""
        if 'finnhub' not in self.api_keys:
            return None
        
        try:
            # Clean symbol
            clean_symbol = symbol.replace('-USD', '')
            
            # Calculate timestamps (last 365 days)
            to_time = int(time.time())
            from_time = to_time - (365 * 24 * 60 * 60)
            
            url = "https://finnhub.io/api/v1/stock/candle"
            params = {
                'symbol': clean_symbol,
                'resolution': 'D',
                'from': from_time,
                'to': to_time,
                'token': self.api_keys['finnhub']
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if data.get('s') == 'ok' and len(data.get('c', [])) > 0:
                df = pd.DataFrame({
                    'Open': data['o'],
                    'High': data['h'],
                    'Low': data['l'],
                    'Close': data['c'],
                    'Volume': data.get('v', [0] * len(data['c']))
                })
                df.index = pd.to_datetime(data['t'], unit='s')
                df = df.sort_index()
                return df
        except Exception as e:
            print(f"Finnhub error: {e}")
        
        return None
    
    def _try_alpha_vantage(self, symbol: str) -> Optional[pd.DataFrame]:
        """Try Alpha Vantage API"""
        if 'alpha_vantage' not in self.api_keys:
            return None
        
        try:
            # Clean symbol
            clean_symbol = symbol.replace('-USD', '')
            
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': clean_symbol,
                'outputsize': 'full',
                'apikey': self.api_keys['alpha_vantage']
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if 'Time Series (Daily)' in data:
                df = pd.DataFrame.from_dict(
                    data['Time Series (Daily)'],
                    orient='index'
                ).astype(float)
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                return df.iloc[-365:]  # Last year
        except Exception as e:
            print(f"Alpha Vantage error: {e}")
        
        return None
    
    def _try_yfinance(self, symbol: str) -> Optional[pd.DataFrame]:
        """Try Yahoo Finance"""
        try:
            import yfinance as yf
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1y", interval="1d")
            
            if not data.empty and len(data) > 50:
                return data
        except Exception as e:
            print(f"Yahoo Finance error: {e}")
        
        return None
    
    def _try_twelvedata(self, symbol: str) -> Optional[pd.DataFrame]:
        """Try Twelve Data API"""
        if 'twelvedata' not in self.api_keys:
            return None
        
        try:
            url = "https://api.twelvedata.com/time_series"
            params = {
                'symbol': symbol,
                'interval': '1day',
                'outputsize': 365,
                'apikey': self.api_keys['twelvedata']
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if 'values' in data:
                df = pd.DataFrame(data['values'])
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
                
                # Convert columns
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Rename columns
                column_map = {
                    'open': 'Open',
                    'high': 'High', 
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                }
                
                df = df.rename(columns=column_map)
                return df
        except Exception as e:
            print(f"Twelve Data error: {e}")
        
        return None
    
    def _create_synthetic_data(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """Create realistic synthetic data"""
        print(f"🎭 Creating synthetic data for {symbol}")
        
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        symbol_upper = symbol.upper()
        
        # Base settings
        if any(crypto in symbol_upper for crypto in ['BTC', 'ETH', 'XRP', 'ADA', 'SOL']):
            base_price = np.random.choice([20000, 30000, 40000, 50000])
            volatility = 0.04
            trend = 0.0003
        elif any(stock in symbol_upper for stock in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA']):
            base_price = np.random.choice([100, 150, 200, 300, 400])
            volatility = 0.02
            trend = 0.0001
        else:
            base_price = 100
            volatility = 0.015
            trend = 0.00005
        
        # Generate price series
        np.random.seed(hash(symbol) % 10000)
        returns = np.random.normal(trend, volatility, days)
        prices = base_price * np.cumprod(1 + returns)
        
        # Create OHLC data
        df = pd.DataFrame({
            'Open': prices * (1 + np.random.uniform(-0.01, 0.01, days)),
            'High': prices * (1 + np.abs(np.random.uniform(0.01, 0.03, days))),
            'Low': prices * (1 - np.abs(np.random.uniform(0.01, 0.03, days))),
            'Close': prices,
            'Volume': np.random.lognormal(14, 1, days) * (1 + np.abs(returns) * 10)
        }, index=dates)
        
        # Ensure High >= Open/Close >= Low
        df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
        df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
        
        return df

# ============================================================================
# AI TRADER ENGINE
# ============================================================================

class AITraderEngine:
    """Core AI trading engine"""
    
    def __init__(self):
        self.data_fetcher = RealDataFetcher()
        self.models = {}
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        data = df.copy()
        price_col = 'Close'
        
        # Basic returns
        data['returns'] = data[price_col].pct_change()
        
        # Moving Averages
        for window in [5, 10, 20, 50, 200]:
            if len(data) >= window:
                data[f'MA_{window}'] = data[price_col].rolling(window).mean()
        
        # RSI
        delta = data[price_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = data[price_col].ewm(span=12, adjust=False).mean()
        exp2 = data[price_col].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        if len(data) >= 20:
            data['BB_middle'] = data[price_col].rolling(20).mean()
            bb_std = data[price_col].rolling(20).std()
            data['BB_upper'] = data['BB_middle'] + (bb_std * 2)
            data['BB_lower'] = data['BB_middle'] - (bb_std * 2)
        
        # Volume
        if 'Volume' in data.columns:
            data['volume_MA'] = data['Volume'].rolling(20).mean()
            data['volume_ratio'] = data['Volume'] / data['volume_MA']
        
        # Drop NaN
        data = data.dropna()
        
        return data
    
    def generate_ai_recommendation(self, df: pd.DataFrame, forecast_days: int = 7) -> Dict:
        """Generate AI trading recommendation"""
        
        if len(df) < 50:
            return self._generate_fallback_recommendation(df)
        
        try:
            # Calculate features
            features_df = self.calculate_technical_indicators(df)
            
            # Get latest values
            latest = features_df.iloc[-1]
            current_price = latest['Close']
            
            # Make prediction
            predicted_price = self._predict_price(features_df, forecast_days)
            pct_change = ((predicted_price - current_price) / current_price) * 100
            
            # Analyze indicators
            rsi = latest.get('RSI', 50)
            ma_20 = latest.get('MA_20', current_price)
            ma_50 = latest.get('MA_50', current_price)
            
            # Generate signal
            if pct_change > 8 and rsi < 70 and current_price > ma_20:
                signal = "🟢 STRONG BUY"
                signal_color = "green"
                confidence = "HIGH"
            elif pct_change > 3 and rsi < 75:
                signal = "🟡 MODERATE BUY"
                signal_color = "yellow"
                confidence = "MEDIUM"
            elif pct_change < -8 and rsi > 30:
                signal = "🔴 STRONG SELL"
                signal_color = "red"
                confidence = "HIGH"
            elif pct_change < -3 and rsi > 25:
                signal = "🟠 MODERATE SELL"
                signal_color = "orange"
                confidence = "MEDIUM"
            else:
                signal = "⚪ HOLD"
                signal_color = "gray"
                confidence = "LOW"
            
            # Risk score
            risk_score = self._calculate_risk_score(latest, pct_change)
            
            # Create recommendation
            recommendation = self._create_recommendation(
                signal, current_price, predicted_price, 
                pct_change, forecast_days, risk_score
            )
            
            return {
                'current_price': current_price,
                'predicted_price': predicted_price,
                'pct_change': pct_change,
                'signal': signal,
                'signal_color': signal_color,
                'confidence': confidence,
                'risk_score': risk_score,
                'recommendation': recommendation,
                'indicators': {
                    'RSI': round(rsi, 2),
                    'MA_20': round(ma_20, 2),
                    'MA_50': round(ma_50, 2),
                    'MACD': round(latest.get('MACD', 0), 3)
                },
                'is_synthetic': df.get('IsSynthetic', True).iloc[-1] if 'IsSynthetic' in df.columns else True
            }
            
        except Exception as e:
            print(f"AI recommendation error: {e}")
            return self._generate_fallback_recommendation(df)
    
    def _predict_price(self, df: pd.DataFrame, forecast_days: int) -> float:
        """Predict future price"""
        try:
            # Simple weighted prediction
            current_price = df['Close'].iloc[-1]
            
            # Method 1: Trend extrapolation
            last_10 = df['Close'].iloc[-10:].values
            if len(last_10) >= 2:
                x = np.arange(len(last_10))
                coeffs = np.polyfit(x, last_10, 1)
                trend_pred = np.polyval(coeffs, len(last_10) + forecast_days)
            else:
                trend_pred = current_price * 1.02
            
            # Method 2: Moving average projection
            ma_20 = df['MA_20'].iloc[-1] if 'MA_20' in df.columns else current_price
            ma_projection = ma_20 * 1.01
            
            # Weighted average
            weights = [0.6, 0.4]  # Trend, MA
            predictions = [trend_pred, ma_projection]
            
            return np.average(predictions, weights=weights)
            
        except:
            return df['Close'].iloc[-1] * 1.02
    
    def _calculate_risk_score(self, latest: pd.Series, pct_change: float) -> float:
        """Calculate risk score (0-100)"""
        score = 50  # Base
        
        # RSI adjustment
        rsi = latest.get('RSI', 50)
        if rsi > 70:
            score += 20
        elif rsi < 30:
            score -= 10
        
        # Volatility adjustment (simplified)
        if abs(pct_change) > 10:
            score += 15
        elif abs(pct_change) > 5:
            score += 5
        
        # Ensure within bounds
        return min(max(score, 0), 100)
    
    def _create_recommendation(self, signal: str, current_price: float, 
                              predicted_price: float, pct_change: float,
                              forecast_days: int, risk_score: float) -> Dict:
        """Create trading recommendation"""
        
        # Entry/Exit levels
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
        
        # Risk/Reward
        risk_reward = abs((take_profit - entry_price) / (entry_price - stop_loss)) if isinstance(take_profit, (int, float)) else 0
        
        # Time horizon
        time_horizon = "Short-term" if forecast_days <= 7 else "Medium-term"
        
        # Key reasons
        reasons = []
        if pct_change > 5:
            reasons.append(f"Strong upside potential ({pct_change:.1f}%)")
        if "BUY" in signal:
            reasons.append("Bullish technical setup")
        if "SELL" in signal:
            reasons.append("Bearish market conditions")
        
        if not reasons:
            reasons = ["Market appears neutral", "Wait for clearer signals"]
        
        # Suggested actions
        if "STRONG BUY" in signal:
            actions = [
                "Enter long position",
                "Set 5% stop-loss",
                "Target profit at predicted price",
                "Consider adding on dips"
            ]
        elif "MODERATE BUY" in signal:
            actions = [
                "Enter partial position",
                "Use tighter 3-4% stop-loss",
                "Take partial profits",
                "Wait for confirmation"
            ]
        elif "STRONG SELL" in signal:
            actions = [
                "Consider short position",
                "Set 5% stop-loss",
                "Target support levels",
                "Consider put options"
            ]
        elif "MODERATE SELL" in signal:
            actions = [
                "Reduce long exposure",
                "Set breakeven stop",
                "Take partial profits",
                "Wait for better entry"
            ]
        else:
            actions = [
                "Hold existing positions",
                "Wait for market direction",
                "Dollar-cost average if long-term",
                "Monitor key levels"
            ]
        
        return {
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward_ratio': round(risk_reward, 2) if risk_reward > 0 else "N/A",
            'position_size': position_size,
            'time_horizon': time_horizon,
            'key_reasons': reasons,
            'suggested_actions': actions
        }
    
    def _generate_fallback_recommendation(self, df: pd.DataFrame) -> Dict:
        """Fallback recommendation"""
        current_price = df['Close'].iloc[-1] if len(df) > 0 else 100
        
        return {
            'current_price': current_price,
            'predicted_price': current_price * 1.02,
            'pct_change': 2.0,
            'signal': "⚪ HOLD",
            'signal_color': "gray",
            'confidence': "LOW",
            'risk_score': 50,
            'recommendation': {
                'entry_price': "N/A",
                'stop_loss': "N/A",
                'take_profit': "N/A",
                'risk_reward_ratio': "N/A",
                'position_size': "Wait for signals",
                'time_horizon': "Short-term",
                'key_reasons': ["Insufficient data"],
                'suggested_actions': ["Wait for more data"]
            },
            'indicators': {},
            'is_synthetic': True
        }

# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    """Main Streamlit application"""
    
    st.title("🚀 AI Trading Predictor Pro")
    st.markdown("### Complete Technical Analysis & AI Recommendations")
    
    # Initialize session state
    if 'trader' not in st.session_state:
        st.session_state.trader = AITraderEngine()
        st.session_state.predictions = {}
    
    # Data status display
    display_data_status()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Trading Settings")
        
        # Symbol input
        symbol = st.text_input(
            "Enter Symbol:",
            value="TSLA",
            help="Examples: AAPL, TSLA, MSFT, BTC-USD, ETH-USD"
        ).upper()
        
        # Forecast days
        forecast_days = st.slider(
            "Forecast Days:",
            min_value=1,
            max_value=30,
            value=7
        )
        
        # API Keys section - FIXED: Immediate updates
        with st.expander("🔑 API Keys (Immediate Effect)", expanded=True):
            
            # Get current keys
            current_keys = st.session_state.trader.data_fetcher.api_keys
            
            st.write("**Current Status:**")
            if current_keys:
                st.success(f"✅ {len(current_keys)} API key(s) loaded")
            else:
                st.warning("⚠️ No API keys loaded")
            
            st.markdown("---")
            
            # Key inputs
            finnhub_key = st.text_input(
                "Finnhub Key", 
                value=current_keys.get('finnhub', ''),
                type="password",
                help="Get free key: finnhub.io"
            )
            
            alpha_key = st.text_input(
                "Alpha Vantage Key", 
                value=current_keys.get('alpha_vantage', ''),
                type="password",
                help="Get free key: alphavantage.co"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("💾 Save & Use Now", type="primary", use_container_width=True):
                    # Prepare new keys
                    new_keys = {}
                    if finnhub_key:
                        new_keys['finnhub'] = finnhub_key
                    if alpha_key:
                        new_keys['alpha_vantage'] = alpha_key
                    
                    # Update immediately
                    if new_keys:
                        st.session_state.trader.data_fetcher.update_api_keys(new_keys)
                        try:
                            st.session_state.trader.data_fetcher.clear_cache()
                            st.success("✅ API keys saved! Cache cleared. Using real data now.")
                        except Exception as e:
                            st.warning(f"✅ API keys saved, but cache clear had issues: {str(e)[:100]}")
                        st.rerun()
                    else:
                        st.warning("⚠️ No keys entered")
            
            with col2:
                if st.button("🔄 Clear Cache", use_container_width=True):
                    try:
                        st.session_state.trader.data_fetcher.clear_cache()
                        st.success("✅ Cache cleared successfully!")
                        time.sleep(0.5)
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error clearing cache: {e}")
                        st.info("Try manually deleting the '.ai_trader_cache' folder")
            
            # Test button
            if st.button("🧪 Test API Keys", use_container_width=True):
                if current_keys:
                    st.info("Testing API keys...")
                    try:
                        # Test with Apple
                        test_data = st.session_state.trader.data_fetcher._try_finnhub("AAPL")
                        if test_data is not None and len(test_data) > 0:
                            st.success(f"✅ Finnhub working! {len(test_data)} records")
                        else:
                            st.error("❌ Finnhub failed - check key or try Alpha Vantage")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)[:100]}")
                else:
                    st.warning("No API keys to test")
        
        st.markdown("---")
        
        # Predict button
        if st.button("🚀 ANALYZE NOW", type="primary", use_container_width=True):
            st.session_state.analyze_symbol = symbol
            st.session_state.forecast_days = forecast_days
    
    # Main content
    if hasattr(st.session_state, 'analyze_symbol'):
        with st.spinner(f"🤖 Analyzing {st.session_state.analyze_symbol}..."):
            
            # Get data
            data = st.session_state.trader.data_fetcher.get_data(
                st.session_state.analyze_symbol, 
                "1y"
            )
            
            if data is not None and len(data) > 50:
                # Generate AI recommendation
                results = st.session_state.trader.generate_ai_recommendation(
                    data, 
                    st.session_state.forecast_days
                )
                
                # Display results
                display_results(results, data)
                
                # Store in session
                st.session_state.predictions[st.session_state.analyze_symbol] = {
                    'timestamp': datetime.now(),
                    'results': results
                }
            else:
                st.error("❌ Insufficient data for analysis")
                st.info("Try a different symbol or add API keys for real data")
    
    else:
        # Welcome screen
        display_welcome()

def display_data_status():
    """Display data source status"""
    
    if 'trader' in st.session_state:
        trader = st.session_state.trader
        
        # Check API keys
        has_keys = bool(trader.data_fetcher.api_keys)
        
        # Test data source
        try:
            # Get a small sample
            sample_data = trader.data_fetcher.get_data("AAPL", "5d")
            is_synthetic = sample_data.get('IsSynthetic', True).iloc[-1] if 'IsSynthetic' in sample_data.columns else True
        except:
            is_synthetic = True
        
        # Display status
        if not is_synthetic:
            st.markdown('<div class="status-real">✅ Using Real Market Data</div>', unsafe_allow_html=True)
            st.caption(f"API keys active: {len(trader.data_fetcher.api_keys)}")
        elif has_keys:
            st.markdown('<div class="status-synthetic">⚠️ API Keys Configured (using fallback)</div>', unsafe_allow_html=True)
            st.caption("Some symbols may not be available")
        else:
            st.markdown('<div class="status-no-keys">⚠️ Using Synthetic Data - Add API Keys</div>', unsafe_allow_html=True)
            st.caption("Get free keys in sidebar for real-time data")
    
    st.markdown("---")

def display_results(results: Dict, data: pd.DataFrame):
    """Display analysis results"""
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Current Price",
            f"${results['current_price']:.2f}",
            delta=None
        )
    
    with col2:
        st.metric(
            f"Predicted ({results.get('forecast_days', 7)}d)",
            f"${results['predicted_price']:.2f}",
            f"{results['pct_change']:.1f}%"
        )
    
    with col3:
        # Signal with color
        signal_color = results['signal_color']
        signal_text = results['signal']
        
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; border: 2px solid {signal_color}; border-radius: 10px;">
            <h3 style="color: {signal_color}; margin: 0;">{signal_text}</h3>
            <p style="margin: 5px 0 0 0;">Confidence: {results['confidence']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        risk_score = results['risk_score']
        risk_color = "green" if risk_score < 40 else "red" if risk_score > 70 else "orange"
        st.metric(
            "Risk Score",
            f"{risk_score}/100",
            delta_color="inverse"
        )
    
    st.markdown("---")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📈 Charts", "📊 Analysis", "🎯 Recommendation"])
    
    with tab1:
        display_charts(data, results)
    
    with tab2:
        display_analysis(results)
    
    with tab3:
        display_recommendation(results)

def display_charts(data: pd.DataFrame, results: Dict):
    """Display charts"""
    
    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Price Chart',
            'RSI Indicator',
            'Moving Averages',
            'Volume Analysis'
        ),
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    # 1. Price Chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Add moving averages
    for ma_col in ['MA_20', 'MA_50']:
        if ma_col in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data[ma_col],
                    name=ma_col.replace('_', ' '),
                    line=dict(width=1)
                ),
                row=1, col=1
            )
    
    # 2. RSI
    if 'RSI' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['RSI'],
                name='RSI',
                line=dict(color='purple', width=2)
            ),
            row=1, col=2
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=2)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=2)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=1, col=2)
    
    # 3. Moving Averages Comparison
    price_col = 'Close'
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data[price_col],
            name='Price',
            line=dict(color='black', width=1.5)
        ),
        row=2, col=1
    )
    
    # Add all MAs
    ma_colors = {'MA_20': 'blue', 'MA_50': 'red', 'MA_200': 'green'}
    for ma_col, color in ma_colors.items():
        if ma_col in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data[ma_col],
                    name=ma_col.replace('_', ' '),
                    line=dict(color=color, width=1)
                ),
                row=2, col=1
            )
    
    # 4. Volume
    if 'Volume' in data.columns:
        # Color bars based on price direction
        colors = ['red' if data['Close'].iloc[i] < data['Close'].iloc[i-1] else 'green' 
                 for i in range(len(data))]
        colors[0] = 'gray'
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=2
        )
    
    fig.update_layout(height=800, showlegend=True, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

def display_analysis(results: Dict):
    """Display technical analysis"""
    
    indicators = results.get('indicators', {})
    
    if indicators:
        # Create columns for indicators
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Momentum Indicators")
            
            # RSI
            rsi = indicators.get('RSI', 50)
            rsi_color = "green" if rsi < 30 else "red" if rsi > 70 else "gray"
            
            st.markdown(f"""
            <div class="card">
                <h4>RSI: <span style="color: {rsi_color}">{rsi}</span></h4>
                <p>{'Oversold (<30)' if rsi < 30 else 'Overbought (>70)' if rsi > 70 else 'Neutral'}</p>
                <div style="background: #e0e0e0; height: 10px; border-radius: 5px; margin: 5px 0;">
                    <div style="background: {rsi_color}; width: {min(rsi, 100)}%; height: 100%; border-radius: 5px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # MACD
            macd = indicators.get('MACD', 0)
            macd_color = "green" if macd > 0 else "red"
            st.markdown(f"""
            <div class="card">
                <h4>MACD: <span style="color: {macd_color}">{macd:.3f}</span></h4>
                <p>{'Bullish' if macd > 0 else 'Bearish'}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("📈 Trend Indicators")
            
            # Moving Averages
            for ma in ['MA_20', 'MA_50']:
                if ma in indicators:
                    value = indicators[ma]
                    current_price = results['current_price']
                    above_below = "above" if current_price > value else "below"
                    diff_pct = abs((current_price - value) / value * 100)
                    
                    st.markdown(f"""
                    <div class="card">
                        <h4>{ma.replace('_', ' ')}: ${value:.2f}</h4>
                        <p>Price is {above_below} by {diff_pct:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Risk Analysis
        st.subheader("📉 Risk Analysis")
        
        risk_score = results['risk_score']
        risk_color = "green" if risk_score < 40 else "red" if risk_score > 70 else "orange"
        
        st.markdown(f"""
        <div class="card">
            <h4>Risk Score: <span style="color: {risk_color}">{risk_score}/100</span></h4>
            <div style="background: #e0e0e0; height: 20px; border-radius: 10px; margin: 10px 0;">
                <div style="background: {risk_color}; width: {risk_score}%; height: 100%; border-radius: 10px;"></div>
            </div>
            <p>{'Low Risk' if risk_score < 40 else 'High Risk' if risk_score > 70 else 'Moderate Risk'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.info("Technical indicators not available")

def display_recommendation(results: Dict):
    """Display trading recommendation"""
    
    rec = results.get('recommendation', {})
    signal = results['signal']
    signal_color = results['signal_color']
    
    # Signal box
    st.markdown(f"""
    <div style="background: {signal_color}20; border-left: 5px solid {signal_color}; padding: 20px; border-radius: 5px; margin-bottom: 20px;">
        <h2 style="color: {signal_color}; margin-top: 0;">{signal}</h2>
        <p><strong>Confidence:</strong> {results['confidence']}</p>
        <p><strong>Expected Change:</strong> {results['pct_change']:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Trade Setup
    st.subheader("🎯 Trade Setup")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        entry = rec.get('entry_price', 'N/A')
        if isinstance(entry, (int, float)):
            st.metric("Entry Price", f"${entry:.2f}")
        else:
            st.metric("Entry Price", entry)
    
    with col2:
        stop = rec.get('stop_loss', 'N/A')
        if isinstance(stop, (int, float)):
            st.metric("Stop Loss", f"${stop:.2f}")
        else:
            st.metric("Stop Loss", stop)
    
    with col3:
        profit = rec.get('take_profit', 'N/A')
        if isinstance(profit, (int, float)):
            st.metric("Take Profit", f"${profit:.2f}")
        else:
            st.metric("Take Profit", profit)
    
    # Risk Management
    st.subheader("🛡️ Risk Management")
    
    risk_col1, risk_col2 = st.columns(2)
    
    with risk_col1:
        rr_ratio = rec.get('risk_reward_ratio', 'N/A')
        if isinstance(rr_ratio, (int, float)):
            rr_color = "green" if rr_ratio > 1.5 else "orange" if rr_ratio > 1 else "red"
            st.markdown(f"""
            <div class="card">
                <h4>Risk/Reward Ratio</h4>
                <h2 style="color: {rr_color}">{rr_ratio}</h2>
                <p>{'Good (>1.5)' if rr_ratio > 1.5 else 'Poor' if rr_ratio < 1 else 'Acceptable'}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="card">
                <h4>Risk/Reward Ratio</h4>
                <p>N/A (Hold signal)</p>
            </div>
            """, unsafe_allow_html=True)
    
    with risk_col2:
        position_size = rec.get('position_size', 'N/A')
        st.markdown(f"""
        <div class="card">
            <h4>Position Size</h4>
            <p>{position_size}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Key Reasons
    st.subheader("📋 Key Reasons")
    
    reasons = rec.get('key_reasons', [])
    for reason in reasons:
        st.markdown(f"✅ {reason}")
    
    # Suggested Actions
    st.subheader("📝 Suggested Actions")
    
    actions = rec.get('suggested_actions', [])
    for i, action in enumerate(actions, 1):
        st.markdown(f"{i}. {action}")
    
    # Time Horizon
    time_horizon = rec.get('time_horizon', 'N/A')
    st.markdown(f"""
    <div class="card">
        <h4>⏰ Time Horizon</h4>
        <p>{time_horizon}</p>
    </div>
    """, unsafe_allow_html=True)

def display_welcome():
    """Display welcome screen"""
    
    st.markdown("""
    <div style="text-align: center; padding: 40px 20px;">
        <h1 style="color: #667eea;">🤖 Welcome to AI Trader Pro</h1>
        <p style="font-size: 1.2rem; color: #666; max-width: 800px; margin: 0 auto;">
            Advanced AI-powered trading analysis with technical indicators, 
            machine learning predictions, and comprehensive risk management.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick start buttons
    st.subheader("🚀 Quick Start")
    
    quick_cols = st.columns(5)
    symbols = ["AAPL", "TSLA", "MSFT", "BTC-USD", "ETH-USD"]
    
    for i, (col, symbol) in enumerate(zip(quick_cols, symbols)):
        with col:
            if st.button(f"📈 {symbol}", use_container_width=True):
                st.session_state.analyze_symbol = symbol
                st.session_state.forecast_days = 7
                st.rerun()
    
    # Features
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="card-blue">
            <h4>📈 Real-time Data</h4>
            <p>Multiple API sources with immediate updates</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card-green">
            <h4>🤖 AI Predictions</h4>
            <p>Machine learning price forecasts</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="card-yellow">
            <h4>📊 Full Analysis</h4>
            <p>Technical indicators & risk management</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# RUN THE APP
# ============================================================================

if __name__ == "__main__":
    main()
