import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_data(n: int = 500, mode: str = "random_walk"):
    """
    Generates synthetic OHLCV data for testing.
    Modes: 'random_walk', 'zigzag_wave'
    """
    np.random.seed(42)
    start_time = datetime.now() - timedelta(minutes=n)
    timestamps = [start_time + timedelta(minutes=i) for i in range(n)]
    
    if mode == "zigzag_wave":
        # Generate Sine Waves to force ZigZag pivots
        # We want a cycle roughly every 50 candles
        cycles = n / 50
        x = np.linspace(0, cycles * 2 * np.pi, n)
        sine = np.sin(x) * 10 
        
        
        # Add linear trend to simulate EMA conditions
        # Rise 50 points over N -> CHANGED: Trend = 0 for SNR Testing
        trend_component = np.zeros(n) 
        
        close = 100 + sine + trend_component + np.random.normal(0, 0.5, n)
        
        # Inject specific patterns at peaks/valleys
        # Valley (Low) -> Should be Hammer
        # Peak (High) -> Should be Shooting Star
        
        open_p = close - np.random.normal(0, 0.2, n)
        high = np.maximum(open_p, close) + np.abs(np.random.normal(0, 0.5, n))
        low = np.minimum(open_p, close) - np.abs(np.random.normal(0, 0.5, n))
        
        # Manually craft Hammer/Shooting Star at extremes
        for i in range(2, n-5):
            # Trough (Local Min)
            if sine[i] < -9.5 and sine[i-1] > sine[i]: # Bottom tip
                # Create Hammer (Long lower wick, small body at top)
                close[i] = 90.5 
                open_p[i] = 90.4
                high[i] = 90.6
                low[i] = 89.0 
                
                # FORCE V-SHAPE RECOVERY (Next 2 candles strong green)
                # This ensures DeMarker spikes > 30 and ZigZag flips UP
                # Deviation is 5, so we need > 5 jump
                # i+1
                open_p[i+1] = close[i]
                close[i+1] = close[i] + 4.0 # Big jump
                high[i+1] = close[i+1]
                low[i+1] = open_p[i+1]
                # i+2
                open_p[i+2] = close[i+1]
                close[i+2] = close[i+1] + 3.0 # Total +7
                high[i+2] = close[i+2]
                low[i+2] = open_p[i+2]
                
                # SUSTAIN for i+3, i+4 (Smooth return or hold)
                # To ensure a win for entry at i+1 (expiry i+3) OR entry at i+2 (expiry i+4)
                # i+3
                close[i+3] = close[i+2] + 0.5
                open_p[i+3] = close[i+2]
                high[i+3] = close[i+3]
                low[i+3] = open_p[i+3]
                # i+4
                close[i+4] = close[i+3] + 0.5
                open_p[i+4] = close[i+3]
                high[i+4] = close[i+4]
                low[i+4] = open_p[i+4]
                
            # Peak (Local Max)
            elif sine[i] > 9.5 and sine[i-1] < sine[i]: # Top tip
                # Create Shooting Star
                close[i] = 109.5
                open_p[i] = 109.6
                high[i] = 111.0 
                low[i] = 109.4
                
                # FORCE A-SHAPE DROP
                # i+1
                open_p[i+1] = close[i]
                close[i+1] = close[i] - 4.0 # Big drop
                high[i+1] = open_p[i+1]
                low[i+1] = close[i+1]
                # i+2
                open_p[i+2] = close[i+1]
                close[i+2] = close[i+1] - 3.0 # Total -7
                high[i+2] = open_p[i+2]
                low[i+2] = close[i+2]
                
                # SUSTAIN for i+3
                close[i+3] = close[i+2] - 0.5
                open_p[i+3] = close[i+2]
                high[i+3] = open_p[i+3]
                low[i+3] = close[i+3]

                # i+4
                close[i+4] = close[i+3] - 0.5
                open_p[i+4] = close[i+3]
                high[i+4] = open_p[i+4]
                low[i+4] = close[i+4]

        # RE-INJECT SIDEWAYS ZONES
        # Every 500 candles, add 50 candles of "noise" with low volatility
        for i in range(500, n-50, 500):
            # Flat price around current value
            base_p = close[i]
            for j in range(i, i+50):
                close[j] = base_p + np.random.normal(0, 0.05)
                open_p[j] = close[j] - np.random.normal(0, 0.02)
                high[j] = max(open_p[j], close[j]) + 0.05
                low[j] = min(open_p[j], close[j]) - 0.05
    else:
        # Random Walk
        close = 100 + np.cumsum(np.random.normal(0, 0.5, n))
        open_p = close - np.random.normal(0, 0.2, n)
        high = np.maximum(open_p, close) + np.abs(np.random.normal(0, 0.1, n))
        low = np.minimum(open_p, close) - np.abs(np.random.normal(0, 0.1, n))
        
    volume = np.random.randint(100, 1000, n)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': open_p,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    return df

if __name__ == "__main__":
    df = generate_sample_data()
    df.to_csv("sample_data.csv", index=False)
    print("Generated sample_data.csv")
