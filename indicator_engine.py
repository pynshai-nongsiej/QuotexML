import pandas as pd
import numpy as np
from typing import Tuple

class IndicatorEngine:
    """
    Calculates technical indicators: RSI, EMA, MACD, BB, ATR.
    """
    
    @staticmethod
    def rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def ema(df: pd.DataFrame, period: int) -> pd.Series:
        return df['close'].ewm(span=period, adjust=False).mean()

    @staticmethod
    def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = IndicatorEngine.ema(df, fast)
        ema_slow = IndicatorEngine.ema(df, slow)
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        sma = df['close'].rolling(window=period).mean()
        rstd = df['close'].rolling(window=period).std()
        upper_band = sma + (std_dev * rstd)
        lower_band = sma - (std_dev * rstd)
        return upper_band, sma, lower_band

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(window=period).mean()

    @staticmethod
    def stochastic(df: pd.DataFrame, period: int = 5, smooth_k: int = 3, smooth_d: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Calculates Stochastic Oscillator. 
        Returns: %K (smoothed), %D (smoothed)
        """
        low_min = df['low'].rolling(window=period).min()
        high_max = df['high'].rolling(window=period).max()
        
        # Fast %K
        k_fast = 100 * ((df['close'] - low_min) / (high_max - low_min))
        
        # Smooth %K and %D
        k_smooth = k_fast.rolling(window=smooth_k).mean()
        d_smooth = k_smooth.rolling(window=smooth_d).mean()
        
        return k_smooth, d_smooth

    @staticmethod
    def demarker(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculates DeMarker Oscillator.
        """
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        
        demax = high_diff.where(high_diff > 0, 0)
        demin = low_diff.where(low_diff > 0, 0)
        
        demax_ma = demax.rolling(window=period).mean()
        demin_ma = demin.rolling(window=period).mean()
        
        dem = demax_ma / (demax_ma + demin_ma)
        return dem * 100

    @staticmethod
    def zigzag(df: pd.DataFrame, deviation: int = 5, depth: int = 12, backstep: int = 3) -> pd.Series:
        """
        Calculates ZigZag Trend Direction (Causal).
        Returns: 1 = Up Swing, -1 = Down Swing.
        Deviation is assumed to be points (0.00001 for Forex/Standard).
        If values are large (like crypto), assumes deviation is absolute value.
        """
        # Adjust deviation scale if it looks like forex (small numbers)
        # Main assets are like 1.0500, so 5 points = 0.00005
        # If asset is BTC (50000), 5 points = 5
        
        series = df['close'].values
        trends = np.zeros(len(series))
        
        # Heuristic for deviation scale
        price_mean = np.mean(series)
        dev_val = deviation
        if price_mean < 100: # Likely Forex
             dev_val = deviation * 0.00001
        
        trend = 0 # 0=Init, 1=Up, -1=Down
        last_high = series[0]
        last_low = series[0]
        
        for i in range(len(series)):
            price = series[i]
            
            if trend == 0:
                if price > last_high + dev_val: trend = 1
                elif price < last_low - dev_val: trend = -1
            
            if trend == 1: # Up Trend
                if price > last_high:
                    last_high = price
                elif price < last_high - dev_val:
                    trend = -1
                    last_low = price # Pivot Low? No, this is start of downswing. Pivot High was last_high.
                    
            elif trend == -1: # Down Trend
                if price < last_low:
                    last_low = price
                elif price > last_low + dev_val:
                    trend = 1
                    last_high = price
            
            trends[i] = trend
            
    @staticmethod
    def parabolic_sar(df: pd.DataFrame, step: float = 0.02, max_val: float = 0.2) -> pd.Series:
        """
        Calculates Parabolic SAR.
        """
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        open_p = df['open'].values
        sar = np.zeros(len(close))
        
        # Initial Trend (use first candle)
        trend = 1 if close[0] > open_p[0] else -1
        # Simple start: if close > open, UP.
        # But for array access, let's use logic:
        trend = 1 # 1 for Up, -1 for Down
        sar[0] = low[0]
        ep = high[0]
        af = step
        
        for i in range(1, len(close)):
            prev_sar = sar[i-1]
            
            # Calculate new SAR based on previous trend
            sar[i] = prev_sar + af * (ep - prev_sar)
            
            # Constraints
            if trend == 1:
                sar[i] = min(sar[i], low[i-1], low[i-2] if i > 1 else low[i-1])
                if low[i] < sar[i]: # Switch to Down
                    trend = -1
                    sar[i] = ep # Start at extreme point
                    ep = low[i]
                    af = step
                else: # Continue Up
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + step, max_val)
                        
            else: # Down Trend
                sar[i] = max(sar[i], high[i-1], high[i-2] if i > 1 else high[i-1])
                if high[i] > sar[i]: # Switch to Up
                    trend = 1
                    sar[i] = ep
                    ep = high[i]
                    af = step
                else:
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + step, max_val)
                        
        return pd.Series(sar, index=df.index)

    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds all required indicators to the dataframe."""
        df = df.copy()
        df['rsi'] = self.rsi(df, period=7) 
        df['ema3'] = self.ema(df, 3) 
        df['ema7'] = self.ema(df, 7)
        # Restore standard EMAs for regime check
        df['ema9'] = self.ema(df, 9)
        df['ema21'] = self.ema(df, 21)
        df['ema50'] = self.ema(df, 50)
        
        # Slopes
        df['ema9_slope'] = df['ema9'].diff()
        df['ema21_slope'] = df['ema21'].diff()
        df['ema50_slope'] = df['ema50'].diff()
        
        # Parabolic SAR (0.04 step? User said 0.04. Max default usually 0.2)
        df['psar'] = self.parabolic_sar(df, step=0.04, max_val=0.2)
        
        # Bollinger Bands (Standard 20, 2)
        upper, mid, lower = self.bollinger_bands(df)
        df['bb_upper'] = upper
        df['bb_mid'] = mid
        df['bb_lower'] = lower
        
        # MACD (Standard 12, 26, 9)
        macd_line, signal_line, hist = self.macd(df)
        df['macd_line'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_hist'] = hist

        # Stochastic (12, 5, 8) as requested
        stoch_k, stoch_d = self.stochastic(df, period=12, smooth_k=8, smooth_d=5)
        df['stoch_k'] = stoch_k
        df['stoch_d'] = stoch_d
        
        # ATR for ML Scorer
        df['atr'] = self.atr(df, period=14)
        
        # ZigZag & DeMarker (Keep for safety or remove? Keep for now to avoid other hidden deps)
        df['zigzag_trend'] = self.zigzag(df, deviation=5, depth=12, backstep=3)
        df['demarker'] = self.demarker(df, period=14)
        
        return df

if __name__ == "__main__":
    print("IndicatorEngine module loaded.")
