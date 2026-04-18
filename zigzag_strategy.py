import pandas as pd
import numpy as np

class ZigZagStrategy:
    """
    Quotex 1-Minute Scalping Strategy
    Tools:
    - ZigZag (pseudo representation using Local Pivots: Depth=12)
    - DeMarker (Period=14, Overbought=70, Oversold=30)
    - Candlestick Patterns (Engulfing / Hammer / Shooting Star)
    """
    
    def __init__(self):
        # Only pattern logic required
        pass

    def get_candle_color(self, row):
        body = abs(row['close'] - row['open'])
        total_len = row['high'] - row['low']
        
        # DOJI FILTER: body is less than 15% of the total wick-to-wick length, or extremely small pure points
        if body < 1e-5 or (total_len > 0 and body / total_len < 0.15):
            return "D"
            
        return "G" if row['close'] > row['open'] else "R"

    def analyze_recent_candles(self, df: pd.DataFrame, lookback=5):
        recent = df.iloc[-lookback:]
        colors = ""
        total_body, total_wick = 0.0, 0.0
        has_doji = False
        
        for idx in range(len(recent)):
            row = recent.iloc[idx]
            color = self.get_candle_color(row)
            colors += color
            if color == "D":
                has_doji = True
                
            body = abs(row['close'] - row['open'])
            wick = (row['high'] - row['low']) - body
            total_body += body
            total_wick += wick
            
        # MARKET FLUCTUATION FILTER: if sum of wicks is double the sum of bodies over the sequence
        high_fluctuation = (total_body > 0 and (total_wick / total_body) > 2.0)
        
        return colors, has_doji, high_fluctuation

    def is_bullish_pattern(self, colors):
        # 2. GRRG / RGRG / GGRG / GGGRG
        return any(colors.endswith(pat) for pat in ["GRRG", "RGRG", "GGRG", "GGGRG"])

    def is_bearish_pattern(self, colors):
        # 2. RGGR / GRGR / RRGR / RRRGR
        return any(colors.endswith(pat) for pat in ["RGGR", "GRGR", "RRGR", "RRRGR"])

    def execute(self, df: pd.DataFrame) -> dict:
        if len(df) < 6:
            return {"decision": "WAIT", "confidence": 0.0, "reason": "Gathering Data", "confluence_score": 0.0}
            
        decision = "WAIT"
        reason = "Scanning Color Patterns"
        confidence = 0.5
        
        # Extract Sequence Patterns
        colors, has_doji, high_fluctuation = self.analyze_recent_candles(df, lookback=5)
        
        # Apply Global Filters (Rule 3 & 4)
        if has_doji:
             reason = "DOJI Detected (No Trade)"
        elif high_fluctuation:
             reason = "Market Highly Fluctuating (No Trade)"
        else:
            # BUY LOGIC
            # 1. Bullish Color Pattern Formed
            if self.is_bullish_pattern(colors):
                decision = "UP"
                reason = f"Bullish Pattern Formed: {colors}"
                confidence = 0.90
                        
            # SELL LOGIC
            # 1. Bearish Color Pattern Formed
            elif self.is_bearish_pattern(colors):
                decision = "DOWN"
                reason = f"Bearish Pattern Formed: {colors}"
                confidence = 0.90

        return {
            "decision": decision,
            "confidence": confidence,
            "reason": reason,
            "confluence_score": confidence,
            "metrics": {
                "pattern": colors
            }
        }
