import pandas as pd
import os
from ml_scorer import MLScorer

def optimize():
    log_file = "logs/learning_data.csv"
    
    if not os.path.exists(log_file):
        print("No learning data found. Run the live trader first.")
        return

    print("Loading learning data...")
    df = pd.read_csv(log_file)
    
    if len(df) < 20:
        print(f"Not enough data to retrain (Current: {len(df)} samples). Recommended: 50+")
        # We can still try if the user insists, but better to wait.
        if len(df) < 5: return

    # Prepare training data
    # Features are f_0, f_1, ...
    feature_cols = [c for c in df.columns if c.startswith("f_")]
    X = df[feature_cols]
    y = df["result"]

    print(f"Retraining model on {len(df)} samples...")
    scorer = MLScorer()
    scorer.train(X, y)
    
    # Analysis
    win_rate = (y.sum() / len(df)) * 100
    print(f"Overall History Win Rate: {win_rate:.2f}%")
    
    # Success by reason
    reason_stats = df.groupby("reason")["result"].agg(['count', 'mean'])
    reason_stats['win_rate'] = reason_stats['mean'] * 100
    print("\nPerformance by Setup:")
    print(reason_stats[['count', 'win_rate']].sort_values(by='win_rate', ascending=False))

if __name__ == "__main__":
    optimize()
