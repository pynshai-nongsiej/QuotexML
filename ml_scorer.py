import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import os

class MLScorer:
    """
    ML model to rank confluence strength.
    Uses binary flags and indicator values as features.
    """
    def __init__(self, model_path: str = "confluence_model.joblib"):
        self.model_path = model_path
        self.model = None
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)

    def prepare_features(self, df: pd.DataFrame, technicals: pd.DataFrame, chart_data: dict) -> np.ndarray:
        """
        Engineers features for the ML model.
        """
        # Feature vector construction
        features = [
            technicals['rsi'].iloc[-1] / 100.0,
            1.0 if chart_data['structure'] == 'bullish' else (0.0 if chart_data['structure'] == 'bearish' else 0.5),
            1.0 if chart_data['engulfing'] == 'bullish' else (0.0 if chart_data['engulfing'] == 'bearish' else 0.5),
            1.0 if chart_data['pinbar'] == 'bullish' else (0.0 if chart_data['pinbar'] == 'bearish' else 0.5),
            1.0 if chart_data['near_sr'] else 0.0,
            technicals['ema9_slope'].iloc[-1],
            technicals['ema21_slope'].iloc[-1],
            technicals['atr'].iloc[-1] / df['close'].iloc[-1] # Normalized ATR
        ]
        return np.array(features).reshape(1, -1)

    def get_score_and_features(self, df: pd.DataFrame, technicals: pd.DataFrame, chart_data: dict) -> Tuple[float, np.ndarray]:
        """Returns both the confluence score and the features used."""
        features = self.prepare_features(df, technicals, chart_data)
        score = self.get_score(features)
        return score, features

    def get_score(self, features: np.ndarray) -> float:
        """Returns confluence score [0, 1]."""
        if self.model is None:
            # Fallback to simple heuristic if no model trained
            return np.mean(features) 
        
        # Predict probability of success (class 1)
        return self.model.predict_proba(features)[0, 1]

    def train(self, X: pd.DataFrame, y: pd.Series):
        """Trains the Gradient Boosting model."""
        self.model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
        self.model.fit(X, y)
        joblib.dump(self.model, self.model_path)
        print(f"Model saved to {self.model_path}")

if __name__ == "__main__":
    print("MLScorer module loaded.")
