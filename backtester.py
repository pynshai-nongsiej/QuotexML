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
    def __init__(self, data: pd.DataFrame, window_size: int = 200):
        self.data = data
        self.window_size = window_size
        self.strategy = StrategyEngine()

    def run(self, start_idx: int = 200, step: int = 1):
        """
        Runs backtest from start_idx to the end of data.
        """
        results = []
        
        print(f"Starting simulation on {len(self.data)} candles...")
        
        # We need to iterate carefully
        for i in range(start_idx, len(self.data) - 2, step): 
            # Get sliding window for PRICE data 
            # We pass the raw data slice; the StrategyEngine will calculate indicators internally
            # (Note: This is slower than pre-calculation but ensures 100% logic parity with live trader)
            window_data = self.data.iloc[i - self.window_size:i+1]
            
            # Get decision
            decision_data = self.strategy.execute(window_data)
            
            # Label: 1-minute expiry (Price Action Scalp - Next Candle)
            if i + 1 >= len(self.data): continue 
            
            current_close = self.data['close'].iloc[i]
            future_close = self.data['close'].iloc[i+1]
            
            outcome = "UP" if future_close > current_close else "DOWN"
            success = 1 if decision_data['decision'] == outcome else 0
            
            res = {
                "timestamp": str(self.data.get('timestamp', pd.Series(range(len(self.data)))).iloc[i]),
                "decision": decision_data['decision'],
                "outcome": outcome,
                "success": success,
                "reason": decision_data['reason'],
                "confluence_score": decision_data['confluence_score']
            }
            results.append(res)
            
            if i % 1000 == 0:
                print(f"Processed {i} candles...")
            
        return pd.DataFrame(results)

    def stats(self, results_df: pd.DataFrame):
        """Calculates performance statistics."""
        # Filter for actual trades
        trades = results_df[results_df['decision'] != "WAIT"]
        
        if len(trades) == 0:
            print("\n--- Backtest Results ---")
            print("No trades executed. Strategy is highly selective.")
            return

        win_rate = trades['success'].mean()
        total_trades = len(trades)
        
        print("\n--- V-POWER SNIPE BACKTEST RESULTS ---")
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Trade Density: {(total_trades / len(results_df)) * 100:.2f}% (Trades per candle)")
        
        # Breakdown by decision type
        print("\nPerformance by Side:")
        print(trades.groupby('decision')['success'].agg(['count', 'mean']))

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
    # Generate data with enough noise & trend for rejection patterns
    df = generate_sample_data(n=n_candles, mode="zigzag_wave")
    
    print("Initializing Backtester...")
    tester = Backtester(df)
    
    print("Running Backtest Simulation...")
    results = tester.run()
    
    if len(results) > 0:
        tester.stats(results)
        
        # Save results
        results.to_csv("backtest_results.csv", index=False)
        print("\nDetailed results saved to 'backtest_results.csv'")
    else:
        print("No trades were triggered during the backtest.")
