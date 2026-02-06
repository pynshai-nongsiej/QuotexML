import pandas as pd
import numpy as np
from indicator_engine import IndicatorEngine
from ml_scorer import MLScorer

class StrategyEngine:
    def __init__(self):
        self.indicators = IndicatorEngine()
        self.ml_scorer = MLScorer()

    def detect_patterns(self, df: pd.DataFrame):
        if len(df) < 3: return "None"
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        body = abs(curr['close'] - curr['open'])
        wick_top = curr['high'] - max(curr['close'], curr['open'])
        wick_bottom = min(curr['close'], curr['open']) - curr['low']
        
        if body > 0:
            if wick_bottom > (body * 1.3): return "REJECTION_UP"
            if wick_top > (body * 1.3): return "REJECTION_DOWN"
        if curr['close'] > curr['open'] and prev['close'] < prev['open']:
            if curr['close'] > prev['open']: return "ENGULFING_UP"
        if curr['close'] < curr['open'] and prev['close'] > prev['open']:
            if curr['close'] < prev['open']: return "ENGULFING_DOWN"
        return "None"

    def execute(self, df: pd.DataFrame) -> dict:
        """
        Multi-Flow Scalper v5.2 (High-Density & Quality)
        Tier 1: Extreme Reversal (High RR)
        Tier 2: Trend Continuation (High Win Rate)
        """
        df = self.indicators.add_all_indicators(df)
        curr = df.iloc[-1]
        
        # Core Indicators
        ema21 = curr['ema21']
        ema50 = curr['ema50']
        ema10 = curr['ema10']
        rsi = curr['rsi']
        px = curr['close']
        pattern = self.detect_patterns(df)
        
        # Bands
        bb_up_ext = curr['bb_upper_ext']
        bb_low_ext = curr['bb_lower_ext']
        bb_up_mid = curr['bb_upper_std'] # 1.5 SD
        bb_low_mid = curr['bb_lower_std'] # 1.5 SD
        
        decision = "WAIT"
        reason = "Scanning Liquidity"
        score = 0.5
        
        # --- MARKET REGIME ---
        uptrend = px > ema50 and ema10 > ema21
        downtrend = px < ema50 and ema10 < ema21
        ranging = not uptrend and not downtrend
        
        # --- SCALPER LOGIC ---
        
        # 1. UP SIGNALS (CALL)
        if decision == "WAIT":
            # REVERSAL: Extreme RSI + Extreme Band
            if px <= bb_low_ext and rsi < 35:
                decision = "UP"
                reason = "V-Snipe (Extreme)"
                score = 0.85
            # TREND SCALP: Price touches 1.5 SD Band while in Uptrend
            elif uptrend and px <= bb_low_mid and rsi < 50:
                decision = "UP"
                reason = "Trend Pullback"
                score = 0.70
            # MOMENTUM: Engulfing with Trend
            elif uptrend and pattern == "ENGULFING_UP" and rsi > 50:
                decision = "UP"
                reason = "Trend Breakout"
                score = 0.65

        # 2. DOWN SIGNALS (DISABLED for CALL-ONLY mode)
        if decision == "WAIT":
            # REVERSAL: Extreme RSI + Extreme Band
            if px >= bb_up_ext and rsi > 65:
                decision = "WAIT" # Disabled
                reason = "DOWN Signal Ignored (CALL-Only)"
                score = 0.5
            # TREND SCALP: Pullback in Downtrend
            elif downtrend and px >= bb_up_mid and rsi > 50:
                decision = "WAIT" # Disabled
                reason = "DOWN Signal Ignored (CALL-Only)"
                score = 0.5
            # MOMENTUM: Engulfing with Trend
            elif downtrend and pattern == "ENGULFING_DOWN" and rsi < 50:
                decision = "WAIT" # Disabled
                reason = "DOWN Signal Ignored (CALL-Only)"
                score = 0.5

        # --- TICK FILTER ---
        if 'ticks' in curr and curr['ticks'] < 5:
             if decision != "WAIT": 
                 # Still block it if too slow, but keep reason descriptive
                 decision = "WAIT"
                 reason = "Waiting for Volatility"

        # Export Metrics
        return {
            "decision": decision,
            "reason": reason,
            "confluence_score": score,
            "features": [rsi, curr['adx'], curr['bb_width'], ema50],
            "metrics": {
                "ema50": ema50,
                "rsi": rsi,
                "px": px,
                "bb_up": bb_up_ext,
                "bb_low": bb_low_ext,
                "pattern": pattern
            }
        }
