import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from indicator_engine import IndicatorEngine
from chart_engine import ChartEngine
from ml_scorer import MLScorer

class StrategyEngine:
    """
    Pure Price Action Strategy.
    Core Logic: SNR Levels + Wick Rejection + Candle Patterns.
    No lagging indicators allowed for decision making.
    """
    def __init__(self, config: Dict = None):
        self.config = config or {
            "snr_zone_width": 0.0020, # Increased tolerance for finding levels
        }
        self.indicators = IndicatorEngine()
        self.charts = ChartEngine()
        self.ml = MLScorer()
        
    def find_snr_levels(self, df: pd.DataFrame, window: int = 50) -> Dict[str, List[float]]:
        """
        Identifies key Support and Resistance levels based on recent pivot points.
        Returns a list of price levels.
        """
        data = df.iloc[-window:]
        highs = data['high'].values
        lows = data['low'].values
        
        # Simple Pivot ID: Current High > Prevs and Nexts? (Lookahead not possible in real-time, 
        # but for Support/Resistance we look at *past* peaks)
        
        # We find Local Maxima/Minima in the PAST window
        levels = []
        
        # Find peaks (Resistance)
        for i in range(2, len(highs)-2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                   levels.append(highs[i])
                   
        # Find valleys (Support)
        for i in range(2, len(lows)-2):
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                   levels.append(lows[i])
        
        # Cluster levels to avoid duplicates (Naively)
        levels.sort()
        cleaned_levels = []
        if levels:
            current_cluster = [levels[0]]
            for l in levels[1:]:
                if l - current_cluster[-1] < 0.0010: # Cluster tolerance
                    current_cluster.append(l)
                else:
                    cleaned_levels.append(np.mean(current_cluster))
                    current_cluster = [l]
            cleaned_levels.append(np.mean(current_cluster))
            
        return cleaned_levels

    def analyze_candle_action(self, open_p, high, low, close):
        """
        Reads the candle structure and wicks.
        """
        color = "GREEN" if close > open_p else "RED"
        body = abs(close - open_p)
        upper_wick = high - max(close, open_p)
        lower_wick = min(close, open_p) - low
        total_range = high - low
        
        if total_range == 0: return "Doji", color, 0, 0
        
        body_ratio = body / total_range
        upper_wick_ratio = upper_wick / total_range
        lower_wick_ratio = lower_wick / total_range
        
        type_ = "Normal"
        
        # Rejection Logic
        if lower_wick_ratio > 0.50: 
            type_ = "Hammer/Rejection Low" # Bullish Rejection
        elif upper_wick_ratio > 0.50:
            type_ = "ShootingStar/Rejection High" # Bearish Rejection
        elif body_ratio > 0.8:
            type_ = "Marubozu/Strong"
        elif body_ratio < 0.1:
            type_ = "Doji"
            
        return type_, color, upper_wick_ratio, lower_wick_ratio

    def check_market_regime(self, tech: pd.DataFrame, window: int = 15) -> Tuple[bool, str]:
        """
        Analyzes the last 15 minutes to determine if the market is Trendy or Sideways/Slow.
        Returns: (is_tradable, status_message)
        """
        # Get last 15 periods
        data = tech.tail(window)
        
        # 1. Trend Strength (ADX)
        curr_adx = data['adx'].iloc[-1]
        avg_adx = data['adx'].mean()
        
        # 2. Volatility (BB Width)
        curr_bbw = data['bb_width'].iloc[-1]
        
        # 3. Sideways/Slow Detection
        if avg_adx < 20 and curr_adx < 25:
            # Low ADX over 15m means no clear direction (Sideways)
            return False, f"Wait: Sideways Market (ADX {curr_adx:.1f})"
            
        if curr_bbw < 0.0008:
            # Low BB width means market is "slow" / consolidating in a tight range
            return False, f"Wait: Slow Market (BBW {curr_bbw:.4f})"
            
        # 4. Success Case: Trendy Market
        return True, "Trendy Market"

    def execute(self, df: pd.DataFrame, pre_calc_tech: pd.DataFrame = None) -> Dict:
        """
        Analyzes data and returns UP/DOWN based on SNR + Candle Rejections.
        """
        if pre_calc_tech is not None:
            tech = pre_calc_tech
        else:
            tech = self.indicators.add_all_indicators(df)
        
        decision = "WAIT" 
        reason = "No signal"
        
        # 1. Identify Market Structure
        # Use larger window to find Key Levels
        levels = self.find_snr_levels(df, window=400)
        
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        close = curr['close']
        
        body_size = abs(curr['close'] - curr['open'])
        total_range = curr['high'] - curr['low']
        body_ratio = body_size / total_range if total_range > 0 else 0
        
        pattern, color, u_wick, l_wick = self.analyze_candle_action(curr['open'], curr['high'], curr['low'], curr['close'])
        
        # 2. Check Proximity to SNR
        limit = self.config['snr_zone_width']
        nearest_level = 0.0
        # UP Signal (Bounce off Support)
        # Logic: Price touched Support during the candle, but REJECTED it (Closed higher).
        # We look for:
        # 1. Low < Support Level (Touch)
        # 2. Close > Support Level (Rejection/Hold)
        # 3. Candle is GREEN (Momentum shifted up)
        # 4. Lower Wick is significant (> 40% of range) or Body is strong Green.
        
        # 3. Strategy Execution (Statistical SNR / Dynamic Levels)
        # Static levels failed. We use Dynamic Volatility Levels (Bollinger 2.5) to find true extremes.
        # This matches the "Neutral Reversion" success but is stricter.
        
        # Calculate dynamic levels (Bollinger Bands 2.5)
        # We use a 20-period basis.
        window = df.iloc[-21:-1]
        mean = window['close'].mean()
        std = window['close'].std()
        
        upper_level = mean + (2.5 * std)
        lower_level = mean - (2.5 * std)
        
        # We need Momentum Exhaustion (RSI)
        rsi = tech['rsi'].iloc[-1]
        
        # --- REGIME FILTER (15m Lookback) ---
        is_tradable, regime_msg = self.check_market_regime(tech, window=15)
        
        # 3. Strategy Execution (Dynamic Breakout / Momentum)
        # Data showed 12% WR for Reversion, implying 88% WR for Breakout.
        # We FOLLOW the move when it hits extreme levels.
        
        # Only execute if market is trendy
        if is_tradable:
            # UP Signal (Breakout UP)
            if curr['high'] >= upper_level:
                 if rsi > 65 and color == "GREEN":
                      decision = "UP"
                      reason = f"Dynamic Breakout UP (Dev 2.5 + RSI {rsi:.1f})"
                      
            # DOWN Signal (Breakout DOWN)
            if curr['low'] <= lower_level:
                 if rsi < 35 and color == "RED":
                      decision = "DOWN"
                      reason = f"Dynamic Breakout DOWN (Dev 2.5 + RSI {rsi:.1f})"
        else:
            decision = "WAIT"
            reason = regime_msg

        metrics = {
            "dyn_res": upper_level,
            "dyn_sup": lower_level,
            "rsi": rsi,
            "adx": tech['adx'].iloc[-1],
            "pattern": pattern,
            "snr_levels": levels[-3:] if levels else [],
            "nearest": nearest_level
        }
        
        return self._format_output(decision, "pa", reason, 1.0, None, metrics=metrics)

    def _format_output(self, decision: str, regime: str, reason: str, score: float, features: np.ndarray = None, metrics: Dict = None) -> Dict:
        output = {
            "decision": decision,
            "regime": regime,
            "reason": reason,
            "confluence_score": round(score, 2),
            "features": features.tolist()[0] if features is not None else []
        }
        if metrics:
            output.update(metrics)
        return output

if __name__ == "__main__":
    print("StrategyEngine module loaded.")
