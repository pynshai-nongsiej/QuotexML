import pandas as pd
import numpy as np
from typing import List, Dict

class ChartEngine:
    """
    Detects market structure, S/R pivots, and candle patterns.
    """

    @staticmethod
    def get_pivots(df: pd.DataFrame, window: int = 5) -> Dict[str, List[int]]:
        """Finds local highs and lows."""
        highs = []
        lows = []
        
        for i in range(window, len(df) - window):
            is_high = True
            is_low = True
            for j in range(i - window, i + window + 1):
                if df['high'].iloc[j] > df['high'].iloc[i]:
                    is_high = False
                if df['low'].iloc[j] < df['low'].iloc[i]:
                    is_low = False
            
            if is_high:
                highs.append(i)
            if is_low:
                lows.append(i)
        
        return {"highs": highs, "lows": lows}

    @staticmethod
    def detect_market_structure(df: pd.DataFrame, pivots: Dict[str, List[int]]) -> str:
        """Determines if the structure is HH/HL (Bullish) or LL/LH (Bearish)."""
        high_indices = pivots['highs']
        low_indices = pivots['lows']
        
        if len(high_indices) < 2 or len(low_indices) < 2:
            return "neutral"
            
        last_high = df['high'].iloc[high_indices[-1]]
        prev_high = df['high'].iloc[high_indices[-2]]
        
        last_low = df['low'].iloc[low_indices[-1]]
        prev_low = df['low'].iloc[low_indices[-2]]
        
        if last_high > prev_high and last_low > prev_low:
            return "bullish" # HH and HL
        elif last_high < prev_high and last_low < prev_low:
            return "bearish" # LH and LL
        
        return "neutral"

    @staticmethod
    def is_engulfing(df: pd.DataFrame, idx: int) -> str:
        """Detects bullish or bearish engulfing patterns at index idx."""
        if idx < 1: return "none"
        
        curr = df.iloc[idx]
        prev = df.iloc[idx-1]
        
        # Bullish Engulfing
        if prev['close'] < prev['open'] and curr['close'] > curr['open']:
            if curr['close'] > prev['open'] and curr['open'] < prev['close']:
                return "bullish"
                
        # Bearish Engulfing
        if prev['close'] > prev['open'] and curr['close'] < curr['open']:
            if curr['close'] < prev['open'] and curr['open'] > prev['close']:
                return "bearish"
                
        return "none"

    @staticmethod
    def is_pin_bar(df: pd.DataFrame, idx: int) -> str:
        """Detects pin bar (hammer/shooting star) at index idx."""
        curr = df.iloc[idx]
        body = abs(curr['close'] - curr['open'])
        total_range = curr['high'] - curr['low']
        
        if total_range == 0: return "none"
        
        upper_wick = curr['high'] - max(curr['open'], curr['close'])
        lower_wick = min(curr['open'], curr['close']) - curr['low']
        
        # Bullish Pin Bar (Long lower wick)
        if lower_wick > (total_range * 0.6) and body < (total_range * 0.3):
            return "bullish"
            
        # Bearish Pin Bar (Long upper wick)
        if upper_wick > (total_range * 0.6) and body < (total_range * 0.3):
            return "bearish"
            
        return "none"

    def analyze(self, df: pd.DataFrame) -> Dict:
        """Performs full chart analysis on the latest data."""
        pivots = self.get_pivots(df)
        structure = self.detect_market_structure(df, pivots)
        
        last_idx = len(df) - 1
        engulfing = self.is_engulfing(df, last_idx)
        pinbar = self.is_pin_bar(df, last_idx)
        
        # Simple S/R detection - check if price is near recent pivots
        price = df['close'].iloc[-1]
        near_sr = False
        sr_type = "none"
        
        for h_idx in pivots['highs'][-3:]:
            if abs(price - df['high'].iloc[h_idx]) / price < 0.002: # 0.2% tolerance
                near_sr = True
                sr_type = "resistance"
                break
        
        for l_idx in pivots['lows'][-3:]:
            if abs(price - df['low'].iloc[l_idx]) / price < 0.002:
                near_sr = True
                sr_type = "support"
                break

        return {
            "structure": structure,
            "engulfing": engulfing,
            "pinbar": pinbar,
            "near_sr": near_sr,
            "sr_type": sr_type
        }

if __name__ == "__main__":
    print("ChartEngine module loaded.")
