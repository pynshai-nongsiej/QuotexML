import pandas as pd
import numpy as np
from typing import Tuple

class IndicatorEngine:
    """
    Robust technical indicator engine for high-frequency trading.
    Uses Wilder's smoothing and epsilon protection for live feeds.
    """
    
    def rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Standard RSI using Wilder's Smoothing (EWM) with epsilon protection."""
        if len(df) < period: return pd.Series(50, index=df.index)
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0))
        loss = (-delta.where(delta < 0, 0))
        
        # alpha = 1 / period is the Wilder's Smoothing alpha
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        
        # Protect against division by zero with a small epsilon
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        # Fill leading NaNs from diff()
        return rsi.fillna(50)

    def ema(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Exponential Moving Average."""
        if len(df) < 2: return df['close']
        return df['close'].ewm(span=period, adjust=False).mean()

    def bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Enhanced Bollinger Bands with better NaN handling."""
        sma = df['close'].rolling(window=period, min_periods=1).mean()
        rstd = df['close'].rolling(window=period, min_periods=1).std()
        
        upper_band = sma + (std_dev * rstd.fillna(0))
        lower_band = sma - (std_dev * rstd.fillna(0))
        return upper_band, sma, lower_band

    def adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average Directional Index (Wilder's Smoothing)."""
        if len(df) < period: return pd.Series(0, index=df.index)
        
        high_diff = df['high'].diff()
        low_diff = df['low'].diff()
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        tr = self.atr(df, period)
        
        plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / (tr + 1e-10))
        minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / (tr + 1e-10))
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        return adx.fillna(0)

    def atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Wilder's True Range ATR."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.ewm(alpha=1/period, adjust=False).mean().fillna(0)

    def zigzag(self, df: pd.DataFrame, deviation: float = 5.0) -> pd.Series:
        """Causal ZigZag Trend Direction."""
        series = df['close'].values
        trends = np.zeros(len(series))
        if len(series) == 0: return pd.Series(trends)
        
        dev_val = (deviation * 0.00001) if np.mean(series) < 100 else (deviation / 100.0) # Relative for non-forex
        trend, last_high, last_low = 0, series[0], series[0]
        
        for i, price in enumerate(series):
            if trend == 0:
                if price > last_high + dev_val: trend = 1
                elif price < last_low - dev_val: trend = -1
            elif trend == 1:
                if price > last_high: last_high = price
                elif price < last_high - dev_val: trend, last_low = -1, price
            elif trend == -1:
                if price < last_low: last_low = price
                elif price > last_low + dev_val: trend, last_high = 1, price
            trends[i] = trend
        return pd.Series(trends, index=df.index)

    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['rsi'] = self.rsi(df, period=14) 
        df['ema10'] = self.ema(df, 10)
        df['ema21'] = self.ema(df, 21)
        df['ema50'] = self.ema(df, 50)
        
        upper_ext, mid, lower_ext = self.bollinger_bands(df, period=20, std_dev=2.5)
        upper_std, _, lower_std = self.bollinger_bands(df, period=20, std_dev=1.5)
        
        df['bb_upper_ext'], df['bb_lower_ext'] = upper_ext, lower_ext
        df['bb_upper_std'], df['bb_lower_std'] = upper_std, lower_std
        df['bb_mid'] = mid
        df['bb_width'] = ((upper_ext - lower_ext) / (mid + 1e-10)).fillna(0)
        
        df['adx'] = self.adx(df, period=14)
        df['atr'] = self.atr(df, period=14)
        df['zigzag'] = self.zigzag(df)
        return df

if __name__ == "__main__":
    print("IndicatorEngine v5.0 Loaded.")
