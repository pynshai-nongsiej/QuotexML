import pandas as pd
import numpy as np
import json
from datetime import timedelta
from strategy_engine import StrategyEngine
from data_loader import DataLoader

class Backtester:
    """
    Simulates trades using the strategy engine and evaluates performance.
    """
    def __init__(self, data: pd.DataFrame, window_size: int = 100):
        self.data = data
        self.window_size = window_size
        self.strategy = StrategyEngine()

    def run(self, start_idx: int = 100, step: int = 1):
        """
        Runs backtest from start_idx to the end of data.
        """
        results = []
        
        print("Pre-calculating indicators for full dataset...")
        # CRITICAL OPTIMIZATION: Calculate indicators ONCE for the whole dataset
        # This fixes O(N^2) slowness and ensures EMA accuracy (long history)
        full_tech = self.strategy.indicators.add_all_indicators(self.data)
        
        print(f"Starting simulation on {len(self.data)} candles...")
        
        # We need to iterate carefully
        for i in range(start_idx, len(self.data) - 2, step): 
            # Get sliding window for PRICE data (Strategy needs candle patterns)
            # StrategyEngine expects the last row of the input DF to be the "current" candle
            window_data = self.data.iloc[i - self.window_size:i+1]
            
            # Get sliding window for INDICATORS
            # We must pass the exact same slice size & alignment
            window_tech = full_tech.iloc[i - self.window_size:i+1]
            
            # Get decision using optimized path
            decision_data = self.strategy.execute(window_data, pre_calc_tech=window_tech)
            
            # Label: 1-minute expiry (Price Action Scalp - Next Candle)
            # close(t+1) vs close(t)
            if i + 1 >= len(self.data): continue 
            
            current_close = self.data['close'].iloc[i]
            future_close = self.data['close'].iloc[i+1]
            
            outcome = "UP" if future_close > current_close else "DOWN"
            success = 1 if decision_data['decision'] == outcome else 0
            
            res = {
                "timestamp": str(self.data['timestamp'].iloc[i]),
                "asset": "BTCUSD", 
                "decision": decision_data['decision'],
                "outcome": outcome,
                "success": success,
                "regime": decision_data['regime'],
                "reason": decision_data['reason'],
                "confluence_score": decision_data['confluence_score']
            }
            results.append(res)
            
            if i % 5000 == 0:
                print(f"Processed {i} candles...")
            
        return pd.DataFrame(results)

    def stats(self, results_df: pd.DataFrame):
        """Calculates performance statistics."""
        # Filter for actual trades
        trades = results_df[results_df['decision'] != "WAIT"]
        
        if len(trades) == 0:
            print("\n--- Backtest Results ---")
            print("No trades executed.")
            return

        win_rate = trades['success'].mean()
        total_trades = len(trades)
        
        print("\n--- Backtest Results (Active Trades Only) ---")
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {win_rate:.2%}")
        
        # Performance by regime
        print("\nPerformance by Regime:")
        print(trades.groupby('regime')['success'].agg(['count', 'mean']))

        # Regime Distribution (All periods)
        print("\nMarket Regime Breakdown (Total Candles Analyzed):")
        regime_counts = results_df['regime'].value_counts()
        regime_pct = results_df['regime'].value_counts(normalize=True) * 100
        for regime, count in regime_counts.items():
            print(f" - {regime}: {count} candles ({regime_pct[regime]:.1f}%)")

if __name__ == "__main__":
    from data_generator import generate_sample_data
    
    # User Input for Data Size
    try:
        user_input = input("Enter number of candles to generate (default 5000): ").strip()
        n_candles = int(user_input) if user_input else 5000
    except ValueError:
        print("Invalid input. Using default 5000.")
        n_candles = 5000
    
    print(f"Generating Synthetic Market Data (ZigZag Waves - Large Scale) - {n_candles} candles...")
    # Generate enough data to trigger patterns (5k is ~3.5 days of 1m data)
    df = generate_sample_data(n=n_candles, mode="zigzag_wave")
    
    print("Initializing Backtester...")
    tester = Backtester(df)
    
    print("Running Backtest Strategy...")
    results = tester.run()
    
    if len(results) > 0:
        tester.stats(results)
        
        # Save results
        results.to_csv("backtest_results.csv", index=False)
        print("\nDetailed results saved to 'backtest_results.csv'")
    else:
        print("No trades were triggered during the backtest.")
