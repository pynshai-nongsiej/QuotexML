import pandas as pd
import numpy as np

class AoEmaStrategy:
    """
    High-Frequency Short Timeframe Scalping Strategy (e.g. 10s candles -> 60s trades)
    Indicators:
    - EMA 3
    - EMA 7
    - Awesome Oscillator (5, 34)
    Rules:
    - EMA Crossover confirms Direction.
    - Awesome Oscillator must align in the same momentum.
    - Strong breakout candle body required to pierce EMA.
    """
    
    def __init__(self):
        self.fast_ema_period = 3
        self.slow_ema_period = 7
        self.ao_short = 5
        self.ao_long = 34
        self.trend_ema = 50
        self.sr_period = 20
        
        # Override trade setting explicitly to 5 Minute expiration (300 seconds)
        self.target_trade_duration = 300

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Calculate EMA
        df['ema3'] = df['close'].ewm(span=self.fast_ema_period, adjust=False).mean()
        df['ema7'] = df['close'].ewm(span=self.slow_ema_period, adjust=False).mean()
        
        # Calculate Awesome Oscillator
        med_price = (df['high'] + df['low']) / 2.0
        ao_sma_short = med_price.rolling(window=self.ao_short).mean()
        ao_sma_long = med_price.rolling(window=self.ao_long).mean()
        df['ao'] = ao_sma_short - ao_sma_long
        
        # Calculate recent average body size to define 'Strong Breakout'
        df['body_size'] = abs(df['close'] - df['open'])
        df['avg_body'] = df['body_size'].rolling(window=10).mean()
        
        # Calculate Trend Alignment (EMA 50)
        df['ema50'] = df['close'].ewm(span=self.trend_ema, adjust=False).mean()
        
        # Calculate Support & Resistance Zones (Rolling Donchian Channel)
        df['resistance'] = df['high'].rolling(window=self.sr_period).max()
        df['support'] = df['low'].rolling(window=self.sr_period).min()
        
        return df

    def execute(self, df: pd.DataFrame) -> dict:
        if len(df) < max(self.ao_long, self.trend_ema) + 2:
            return {
                "decision": "WAIT", 
                "confidence": 0.0, 
                "reason": f"Gathering Data ({len(df)}/{max(self.ao_long, self.trend_ema)})", 
                "confluence_score": 0.0
            }
            
        # Ensure indicators are processed
        if 'ao' not in df.columns:
            df = self.add_indicators(df)
            
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        decision = "WAIT"
        reason = "Scanning AO / EMA Clusters"
        confidence = 0.5
        
        # Track Crossings
        ema_cross_up = (prev['ema3'] <= prev['ema7']) and (curr['ema3'] > curr['ema7'])
        ema_cross_down = (prev['ema3'] >= prev['ema7']) and (curr['ema3'] < curr['ema7'])
        
        # Track AO Status (Both Line positioning and Color momentum must concur)
        ao_is_green = (curr['ao'] > 0) and (curr['ao'] > prev['ao'])
        ao_is_red = (curr['ao'] < 0) and (curr['ao'] < prev['ao'])
        
        # --- Breakout Strength Logic ---
        curr_is_strong_green = (curr['close'] > curr['open']) and (curr['body_size'] > curr['avg_body'] * 1.2)
        curr_is_strong_red = (curr['close'] < curr['open']) and (curr['body_size'] > curr['avg_body'] * 1.2)
        
        # --- Trend Alignment Filter ---
        trend_is_up = curr['close'] > curr['ema50']
        trend_is_down = curr['close'] < curr['ema50']
        
        # --- Support and Resistance (Proximity) Filter ---
        # Don't buy if the closing price is in the top 15% of the recent resistance channel.
        channel_range = curr['resistance'] - curr['support']
        safe_to_buy = curr['close'] < (curr['resistance'] - channel_range * 0.15)
        safe_to_sell = curr['close'] > (curr['support'] + channel_range * 0.15)
        
        # BUY LOGIC
        if ema_cross_up:
            if not trend_is_up:
                reason = "EMA Crossed UP, but anti-trend (Close < EMA 50)"
            elif not safe_to_buy:
                reason = "Buy rejected: Price hitting Resistance Zone"
            elif curr_is_strong_green and ao_is_green:
                decision = "UP"
                reason = "Strong Bullish Breakout + EMA3>7 Cross + AO is Green + Trend OK"
                confidence = 0.88
            else:
                reason = "EMA Crossed UP, but weak breakout or AO mismatch"
                
        # SELL LOGIC
        elif ema_cross_down:
            if not trend_is_down:
                reason = "EMA Crossed DOWN, but anti-trend (Close > EMA 50)"
            elif not safe_to_sell:
                reason = "Sell rejected: Price hitting Support Zone"
            elif curr_is_strong_red and ao_is_red:
                decision = "DOWN"
                reason = "Strong Bearish Breakout + EMA3<7 Cross + AO is Red + Trend OK"
                confidence = 0.88
            else:
                reason = "EMA Crossed DOWN, but weak breakout or AO mismatch"

        return {
            "decision": decision,
            "confidence": confidence,
            "reason": reason,
            "confluence_score": confidence,
            "duration_override": self.target_trade_duration, # Force custom length
            "metrics": {
                "ao": curr['ao'],
                "ema3": curr['ema3'],
                "ema7": curr['ema7'],
                "body_pct": (curr['body_size'] / (curr['avg_body'] + 1e-10)) * 100
            }
        }
