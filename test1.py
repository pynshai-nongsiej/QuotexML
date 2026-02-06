
import pandas as pd
from data_generator import generate_sample_data
from strategy_engine import StrategyEngine
from indicator_engine import IndicatorEngine

# 1. Generate Wave Data
print("Generating ZigZag Wave Data...")
df = generate_sample_data(n=300, mode="zigzag_wave")

# 2. Run Strategy
print("Analyzing Strategy...")
strategy = StrategyEngine()
decisions = []

for i in range(50, len(df)):
    window = df.iloc[:i+1]
    result = strategy.execute(window)
    
    # Capture relevant signals
    if result['decision'] in ["UP", "DOWN"]:
        print(f"Time: {window.index[-1]} | Signal: {result['decision']} | Reason: {result['reason']}")
        decisions.append(result)

print(f"\nTotal Signals Found: {len(decisions)}")

# Diagnostic: Check if we even produced Hammers/Shooting Stars
print("\n--- Diagnostic Check ---")
full_tech = strategy.indicators.add_all_indicators(df)
for idx, row in df.iterrows():
    c_type, _ = strategy.analyze_candle_psychology(row['open'], row['high'], row['low'], row['close'])
    loc = df.index.get_loc(idx)
    
    dem = full_tech['demarker'].iloc[loc]
    zz = full_tech['zigzag_trend'].iloc[loc]
    
    # Check near extremes of the sine wave (approx every 30 candles)
    if c_type in ["Hammer", "ShootingStar"]:
        print(f"Found {c_type} at {idx} | DeM: {dem:.2f} | ZZ: {zz}")