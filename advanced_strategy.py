import numpy as np
import pandas as pd
from data_loader import DataLoader
from chart_engine import ChartEngine
from typing import Tuple, Dict, List


# Simple Neural Network implementation (feedforward, one hidden layer)
class SimpleNN:
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 10,
        output_size: int = 1,
        lr: float = 0.01,
    ):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))
        self.lr = lr

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # clip to avoid overflow

    def _sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self._sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self._sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, output):
        # y shape (n_samples, 1)
        output_error = y - output
        output_delta = output_error * self._sigmoid_derivative(output)

        hidden_error = np.dot(output_delta, self.W2.T)
        hidden_delta = hidden_error * self._sigmoid_derivative(self.a1)

        # Update weights
        self.W2 += self.lr * np.dot(self.a1.T, output_delta)
        self.b2 += self.lr * np.sum(output_delta, axis=0, keepdims=True)
        self.W1 += self.lr * np.dot(X.T, hidden_delta)
        self.b1 += self.lr * np.sum(hidden_delta, axis=0, keepdims=True)

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            if epoch % 100 == 0:
                loss = np.mean(np.square(y - output))
                # print(f"Epoch {epoch}, loss {loss:.6f}")

    def predict(self, X):
        return self.forward(X)


def compute_technicals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute basic technical indicators: RSI, EMA slopes, ATR."""
    df = df.copy()
    # RSI
    delta = df["close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(com=13, adjust=False).mean()
    ma_down = down.ewm(com=13, adjust=False).mean()
    rs = ma_up / ma_down
    df["rsi"] = 100 - (100 / (1 + rs))

    # EMA9 and EMA21 and their slopes
    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema9_slope"] = df["ema9"].diff()
    df["ema21_slope"] = df["ema21"].diff()

    # ATR (14)
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    return df


def extract_features(df_window: pd.DataFrame, chart_data: Dict) -> np.ndarray:
    """Create feature vector from window and chart analysis."""
    tech = df_window.iloc[-1]
    features = [
        tech["rsi"] / 100.0,
        1.0
        if chart_data["structure"] == "bullish"
        else (0.0 if chart_data["structure"] == "bearish" else 0.5),
        1.0
        if chart_data["engulfing"] == "bullish"
        else (0.0 if chart_data["engulfing"] == "bearish" else 0.5),
        1.0
        if chart_data["pinbar"] == "bullish"
        else (0.0 if chart_data["pinbar"] == "bearish" else 0.5),
        1.0 if chart_data["near_sr"] else 0.0,
        tech["ema9_slope"],
        tech["ema21_slope"],
        tech["atr"] / tech["close"],  # normalized ATR
    ]
    return np.array(features).reshape(1, -1)


def generate_signal(df_window: pd.DataFrame, model: SimpleNN) -> Tuple[int, float]:
    """Return signal: 1 for CALL, 0 for PUT, and confidence."""
    # Get chart analysis
    chart_engine = ChartEngine()
    chart_data = chart_engine.analyze(df_window)
    # Features
    X = extract_features(df_window, chart_data)
    prob = model.predict(X)[0, 0]
    # Threshold for signal
    if prob > 0.6:
        signal = 1  # CALL
    elif prob < 0.4:
        signal = 0  # PUT
    else:
        signal = -1  # NO TRADE
    confidence = prob if signal == 1 else (1 - prob) if signal == 0 else 0.0
    return signal, confidence


def backtest_strategy(
    data_path: str, model: SimpleNN, initial_balance: float = 1000, stake: float = 10
) -> Dict:
    """Simple backtest over historical data."""
    loader = DataLoader(filepath=data_path)
    df = loader.df
    balance = initial_balance
    wins = 0
    total_trades = 0
    # We need to iterate windows; each step we predict next candle direction (simplify: assume 1-minute expiry)
    window_size = 50  # lookback for features
    for i in range(window_size, len(df) - 1):
        window = loader.get_window(i, window_size)
        signal, conf = generate_signal(window, model)
        if signal == -1:
            continue
        total_trades += 1
        # Determine actual outcome: next close > current close? (CALL) else PUT
        actual_move = 1 if df["close"].iloc[i + 1] > df["close"].iloc[i] else 0
        if signal == actual_move:
            wins += 1
            balance += stake * 0.8  # typical binary options payout ~80%
        else:
            balance -= stake
    win_rate = wins / total_trades if total_trades > 0 else 0
    return {
        "final_balance": balance,
        "total_trades": total_trades,
        "wins": wins,
        "win_rate": win_rate,
    }


def train_model_from_data(data_path: str, epochs: int = 500) -> SimpleNN:
    """Create training labels from historical data (next bar direction)."""
    loader = DataLoader(filepath=data_path)
    df = loader.df
    window_size = 50
    X_list = []
    y_list = []
    chart_engine = ChartEngine()
    for i in range(window_size, len(df) - 1):
        window = loader.get_window(i, window_size)
        tech = compute_technicals(window)
        chart_data = chart_engine.analyze(window)
        features = extract_features(window, chart_data).flatten()
        # label: next bar up = 1, down = 0
        y = 1 if df["close"].iloc[i + 1] > df["close"].iloc[i] else 0
        X_list.append(features)
        y_list.append(y)
    X = np.array(X_list)
    y = np.array(y_list).reshape(-1, 1)
    # Normalize features (optional)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std
    # Initialize NN
    nn = SimpleNN(input_size=X_norm.shape[1], hidden_size=15, lr=0.05)
    nn.train(X_norm, y, epochs=epochs)
    # Save normalization params for later use
    nn.X_mean = X_mean
    nn.X_std = X_std
    return nn


if __name__ == "__main__":
    # Example usage: replace with your data file path
    DATA_PATH = "historical_data.csv"  # <-- user should put their CSV here
    print("Training model...")
    model = train_model_from_data(DATA_PATH, epochs=300)
    print("Running backtest...")
    results = backtest_strategy(DATA_PATH, model)
    print(f"Backtest Results: {results}")
