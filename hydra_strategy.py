"""
HydraNet Strategy Engine v1.0
─────────────────────────────
Advanced neural-ensemble trading strategy for Quotex binary options.

Components
──────────
1.  DeepMLP          – 3-layer MLP (pure NumPy) with Adam, He init, dropout
2.  FeatureFactory   – 40+ engineered features (momentum, trend, vol, pattern…)
3.  RegimeDetector   – ADX + vol-ratio classifier
4.  HydraEnsemble    – 3 specialist networks with regime-adaptive weight voting
5.  ConfidenceGate   – Adaptive threshold based on rolling win-rate
6.  OnlineLearner    – Replay-buffer mini-batch SGD
7.  RiskManager      – Streak / drawdown filter + compounding stake sizer

All ML is pure NumPy — no external APIs, no sklearn, no PyTorch.
"""

import numpy as np
import pandas as pd
import json, os, time
from typing import Dict, List, Optional, Tuple
from collections import deque

# ════════════════════════════════════════════════════════════════════════
#  1.  DEEP MLP (Pure NumPy)
# ════════════════════════════════════════════════════════════════════════

class DeepMLP:
    """
    Multi-layer perceptron with:
    - He (Kaiming) weight initialisation
    - ReLU hidden activations, Sigmoid output
    - Adam optimiser (per-parameter adaptive LR)
    - L2 weight decay
    - Inverted dropout (train-time only)
    - Gradient clipping (max norm)
    - Early stopping support
    """

    def __init__(self, layer_sizes: List[int], lr: float = 0.001,
                 l2: float = 5e-4, dropout: float = 0.30,
                 max_grad_norm: float = 1.0):
        self.layers = layer_sizes
        self.lr = lr
        self.l2 = l2
        self.dropout = dropout
        self.max_grad_norm = max_grad_norm
        self.training = True

        # Weight init (He for ReLU layers, Xavier for output)
        self.W: List[np.ndarray] = []
        self.b: List[np.ndarray] = []
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            if i < len(layer_sizes) - 2:
                std = np.sqrt(2.0 / fan_in)           # He init
            else:
                std = np.sqrt(1.0 / fan_in)            # Xavier for output
            self.W.append(np.random.randn(fan_in, fan_out) * std)
            self.b.append(np.zeros((1, fan_out)))

        # Adam state
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0  # Adam timestep

        # Cache for backprop
        self._cache: Dict = {}

    # ── Activations ──────────────────────────────────────────────────
    @staticmethod
    def _relu(x):
        return np.maximum(0, x)

    @staticmethod
    def _relu_deriv(x):
        return (x > 0).astype(float)

    @staticmethod
    def _sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    # ── Forward ──────────────────────────────────────────────────────
    def forward(self, X: np.ndarray) -> np.ndarray:
        A = X.copy()
        self._cache['A0'] = A

        n_hidden = len(self.W) - 1
        for i in range(n_hidden):
            Z = A @ self.W[i] + self.b[i]
            A = self._relu(Z)
            # Inverted dropout during training
            if self.training and self.dropout > 0:
                mask = (np.random.rand(*A.shape) > self.dropout).astype(float)
                A = A * mask / (1.0 - self.dropout + 1e-12)
                self._cache[f'mask{i}'] = mask
            self._cache[f'Z{i}'] = Z
            self._cache[f'A{i+1}'] = A

        # Output layer — sigmoid
        Z_out = A @ self.W[-1] + self.b[-1]
        A_out = self._sigmoid(Z_out)
        self._cache[f'Z{n_hidden}'] = Z_out
        self._cache[f'A{n_hidden+1}'] = A_out
        return A_out

    # ── Backward + Adam Update ───────────────────────────────────────
    def backward(self, X: np.ndarray, y: np.ndarray):
        m = X.shape[0]
        n_layers = len(self.W)
        n_hidden = n_layers - 1
        self.t += 1

        # Output error (binary cross-entropy gradient)
        A_out = self._cache[f'A{n_hidden+1}']
        dZ = A_out - y  # shape (m, 1)

        grads_W = [None] * n_layers
        grads_b = [None] * n_layers

        # Output layer grads
        A_prev = self._cache[f'A{n_hidden}']
        grads_W[-1] = (A_prev.T @ dZ) / m + self.l2 * self.W[-1]
        grads_b[-1] = np.mean(dZ, axis=0, keepdims=True)

        # Hidden layers (reverse)
        dA = dZ @ self.W[-1].T
        for i in range(n_hidden - 1, -1, -1):
            Z = self._cache[f'Z{i}']
            dA_relu = dA * self._relu_deriv(Z)
            # Apply dropout mask
            if self.training and self.dropout > 0 and f'mask{i}' in self._cache:
                dA_relu = dA_relu * self._cache[f'mask{i}'] / (1.0 - self.dropout + 1e-12)
            A_prev = self._cache[f'A{i}']
            grads_W[i] = (A_prev.T @ dA_relu) / m + self.l2 * self.W[i]
            grads_b[i] = np.mean(dA_relu, axis=0, keepdims=True)
            if i > 0:
                dA = dA_relu @ self.W[i].T

        # Gradient clipping (max norm)
        if self.max_grad_norm > 0:
            total_norm = 0.0
            for i in range(n_layers):
                total_norm += np.sum(grads_W[i] ** 2)
                total_norm += np.sum(grads_b[i] ** 2)
            total_norm = np.sqrt(total_norm)
            if total_norm > self.max_grad_norm:
                clip_coef = self.max_grad_norm / (total_norm + 1e-10)
                for i in range(n_layers):
                    grads_W[i] = grads_W[i] * clip_coef
                    grads_b[i] = grads_b[i] * clip_coef

        # Adam update
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for i in range(n_layers):
            self.mW[i] = beta1 * self.mW[i] + (1 - beta1) * grads_W[i]
            self.vW[i] = beta2 * self.vW[i] + (1 - beta2) * (grads_W[i] ** 2)
            mW_hat = self.mW[i] / (1 - beta1 ** self.t)
            vW_hat = self.vW[i] / (1 - beta2 ** self.t)
            self.W[i] -= self.lr * mW_hat / (np.sqrt(vW_hat) + eps)

            self.mb[i] = beta1 * self.mb[i] + (1 - beta1) * grads_b[i]
            self.vb[i] = beta2 * self.vb[i] + (1 - beta2) * (grads_b[i] ** 2)
            mb_hat = self.mb[i] / (1 - beta1 ** self.t)
            vb_hat = self.vb[i] / (1 - beta2 ** self.t)
            self.b[i] -= self.lr * mb_hat / (np.sqrt(vb_hat) + eps)

    # ── Train loop ───────────────────────────────────────────────────
    def train_epochs(self, X: np.ndarray, y: np.ndarray, epochs: int = 500,
                     batch_size: int = 64, verbose: bool = False,
                     X_val: np.ndarray = None, y_val: np.ndarray = None,
                     patience: int = 25):
        self.training = True
        m = X.shape[0]

        # Early stopping state
        best_val_loss = float('inf')
        best_W = [w.copy() for w in self.W]
        best_b = [b.copy() for b in self.b]
        epochs_no_improve = 0
        use_early_stop = X_val is not None and y_val is not None

        for epoch in range(epochs):
            # Mini-batch
            indices = np.random.permutation(m)
            for start in range(0, m, batch_size):
                end = min(start + batch_size, m)
                Xb = X[indices[start:end]]
                yb = y[indices[start:end]]
                self.forward(Xb)
                self.backward(Xb, yb)

            # Validation / early stopping check
            if use_early_stop and epoch % 5 == 0:
                self.training = False
                val_pred = self.forward(X_val)
                val_loss = -np.mean(y_val * np.log(val_pred + 1e-12) +
                                    (1 - y_val) * np.log(1 - val_pred + 1e-12))
                self.training = True
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_W = [w.copy() for w in self.W]
                    best_b = [b.copy() for b in self.b]
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 5
                if epochs_no_improve >= patience:
                    if verbose:
                        self.training = False
                        val_acc = np.mean((val_pred > 0.5).astype(float) == y_val)
                        print(f"  Early stop @ epoch {epoch} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")
                        self.training = True
                    # Restore best weights
                    self.W = best_W
                    self.b = best_b
                    return

            if verbose and epoch % 100 == 0:
                self.training = False
                pred = self.forward(X)
                loss = -np.mean(y * np.log(pred + 1e-12) + (1 - y) * np.log(1 - pred + 1e-12))
                acc = np.mean((pred > 0.5).astype(float) == y)
                val_str = ""
                if use_early_stop:
                    val_pred = self.forward(X_val)
                    val_loss = -np.mean(y_val * np.log(val_pred + 1e-12) +
                                        (1 - y_val) * np.log(1 - val_pred + 1e-12))
                    val_acc = np.mean((val_pred > 0.5).astype(float) == y_val)
                    val_str = f" | Val: {val_loss:.4f} ({val_acc:.2%})"
                print(f"  Epoch {epoch:4d} | Loss: {loss:.4f} | Acc: {acc:.2%}{val_str}")
                self.training = True

        # After all epochs, restore best weights if we used early stopping
        if use_early_stop:
            self.W = best_W
            self.b = best_b

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.training = False
        out = self.forward(X)
        self.training = True
        return out

    # ── Serialization ────────────────────────────────────────────────
    def save(self, path: str):
        data = {
            'layers': self.layers,
            'lr': self.lr, 'l2': self.l2, 'dropout': self.dropout,
            't': self.t,
            'W': [w.tolist() for w in self.W],
            'b': [b.tolist() for b in self.b],
            'mW': [m.tolist() for m in self.mW],
            'vW': [v.tolist() for v in self.vW],
            'mb': [m.tolist() for m in self.mb],
            'vb': [v.tolist() for v in self.vb],
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> 'DeepMLP':
        with open(path, 'r') as f:
            data = json.load(f)
        nn = cls(data['layers'], data['lr'], data['l2'], data['dropout'])
        nn.t = data['t']
        nn.W = [np.array(w) for w in data['W']]
        nn.b = [np.array(b) for b in data['b']]
        nn.mW = [np.array(m) for m in data['mW']]
        nn.vW = [np.array(v) for v in data['vW']]
        nn.mb = [np.array(m) for m in data['mb']]
        nn.vb = [np.array(v) for v in data['vb']]
        return nn


# ════════════════════════════════════════════════════════════════════════
#  2.  FEATURE FACTORY
# ════════════════════════════════════════════════════════════════════════

class FeatureFactory:
    """Engineers 40+ trading features from raw OHLCV data (pure NumPy/Pandas)."""

    # ── Helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def _sma(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(window=period, min_periods=1).mean()

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        hl = df['high'] - df['low']
        hc = (df['high'] - df['close'].shift()).abs()
        lc = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.ewm(alpha=1.0 / period, adjust=False).mean().fillna(0)

    @staticmethod
    def _adx(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Returns (ADX, +DI, -DI)."""
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
        atr = FeatureFactory._atr(df, period)
        plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / (atr + 1e-10)
        minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / (atr + 1e-10)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        adx = dx.ewm(alpha=1/period, adjust=False).mean().fillna(0)
        return adx, plus_di.fillna(0), minus_di.fillna(0)

    @staticmethod
    def _stochastic_rsi(rsi_series: pd.Series, period: int = 14) -> pd.Series:
        min_rsi = rsi_series.rolling(period, min_periods=1).min()
        max_rsi = rsi_series.rolling(period, min_periods=1).max()
        return ((rsi_series - min_rsi) / (max_rsi - min_rsi + 1e-10)).fillna(0.5)

    @staticmethod
    def _macd(series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        ema12 = series.ewm(span=12, adjust=False).mean()
        ema26 = series.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal = macd_line.ewm(span=9, adjust=False).mean()
        hist = macd_line - signal
        return macd_line, signal, hist

    @staticmethod
    def _bollinger(series: pd.Series, period: int = 20, std: float = 2.0):
        sma = series.rolling(period, min_periods=1).mean()
        rstd = series.rolling(period, min_periods=1).std().fillna(0)
        upper = sma + std * rstd
        lower = sma - std * rstd
        return upper, sma, lower

    @staticmethod
    def _hurst_exponent(series: pd.Series, max_lag: int = 20) -> float:
        """Simplified Hurst exponent estimate using R/S analysis."""
        vals = series.dropna().values
        if len(vals) < max_lag * 2:
            return 0.5
        lags = range(2, max_lag)
        tau = []
        for lag in lags:
            chunks = [vals[i:i+lag] for i in range(0, len(vals) - lag, lag)]
            rs_vals = []
            for chunk in chunks:
                if len(chunk) < 2:
                    continue
                mean_c = np.mean(chunk)
                dev = np.cumsum(chunk - mean_c)
                R = np.max(dev) - np.min(dev)
                S = np.std(chunk)
                if S > 0:
                    rs_vals.append(R / S)
            if rs_vals:
                tau.append(np.mean(rs_vals))
            else:
                tau.append(1.0)
        if len(tau) < 2:
            return 0.5
        log_lags = np.log(list(lags)[:len(tau)])
        log_tau = np.log(np.array(tau) + 1e-10)
        # Linear regression slope = Hurst exponent
        A = np.vstack([log_lags, np.ones(len(log_lags))]).T
        try:
            result = np.linalg.lstsq(A, log_tau, rcond=None)
            H = float(result[0][0])
            return np.clip(H, 0.0, 1.0)
        except Exception:
            return 0.5

    # ── Main Feature Builder ─────────────────────────────────────────
    def build(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Returns (feature_vector_for_last_candle, feature_names).
        All features for the LAST row — uses the entire DataFrame for lookback.
        """
        df = df.copy()
        close = df['close']
        high = df['high']
        low = df['low']
        opn = df['open']
        vol = df.get('volume', pd.Series(np.ones(len(df)), index=df.index))
        ticks = df.get('ticks', pd.Series(10.0 * np.ones(len(df)), index=df.index))

        features = {}

        # ── Momentum ─────────────────────────────────────────────────
        rsi14 = self._rsi(close, 14)
        rsi7 = self._rsi(close, 7)
        stoch_rsi = self._stochastic_rsi(rsi14, 14)
        macd_line, macd_sig, macd_hist = self._macd(close)
        roc5 = close.pct_change(5).fillna(0)
        roc10 = close.pct_change(10).fillna(0)
        roc20 = close.pct_change(20).fillna(0)

        features['rsi14'] = rsi14.iloc[-1] / 100.0
        features['rsi7'] = rsi7.iloc[-1] / 100.0
        features['stoch_rsi'] = stoch_rsi.iloc[-1]
        features['macd_norm'] = macd_line.iloc[-1] / (close.iloc[-1] + 1e-10)
        features['macd_signal_norm'] = macd_sig.iloc[-1] / (close.iloc[-1] + 1e-10)
        features['macd_hist_norm'] = macd_hist.iloc[-1] / (close.iloc[-1] + 1e-10)
        features['macd_cross'] = 1.0 if macd_line.iloc[-1] > macd_sig.iloc[-1] else 0.0
        features['roc5'] = np.clip(roc5.iloc[-1] * 100, -10, 10)
        features['roc10'] = np.clip(roc10.iloc[-1] * 100, -10, 10)
        features['roc20'] = np.clip(roc20.iloc[-1] * 100, -10, 10)

        # ── Trend ────────────────────────────────────────────────────
        ema5 = self._ema(close, 5)
        ema10 = self._ema(close, 10)
        ema21 = self._ema(close, 21)
        ema50 = self._ema(close, 50)
        ema100 = self._ema(close, 100)

        px = close.iloc[-1]
        features['ema5_dist'] = (px - ema5.iloc[-1]) / (px + 1e-10)
        features['ema10_dist'] = (px - ema10.iloc[-1]) / (px + 1e-10)
        features['ema21_dist'] = (px - ema21.iloc[-1]) / (px + 1e-10)
        features['ema50_dist'] = (px - ema50.iloc[-1]) / (px + 1e-10)
        features['ema100_dist'] = (px - ema100.iloc[-1]) / (px + 1e-10)
        features['ema_slope5'] = ema5.diff().iloc[-1] / (px + 1e-10)
        features['ema_slope21'] = ema21.diff().iloc[-1] / (px + 1e-10)
        features['ema_cross_10_21'] = 1.0 if ema10.iloc[-1] > ema21.iloc[-1] else 0.0
        features['ema_cross_21_50'] = 1.0 if ema21.iloc[-1] > ema50.iloc[-1] else 0.0
        features['ema_stack'] = 1.0 if (ema5.iloc[-1] > ema10.iloc[-1] > ema21.iloc[-1] > ema50.iloc[-1]) else (
            0.0 if (ema5.iloc[-1] < ema10.iloc[-1] < ema21.iloc[-1] < ema50.iloc[-1]) else 0.5)

        adx, plus_di, minus_di = self._adx(df, 14)
        features['adx'] = adx.iloc[-1] / 100.0
        features['di_diff'] = (plus_di.iloc[-1] - minus_di.iloc[-1]) / 100.0

        # ── Volatility ───────────────────────────────────────────────
        atr14 = self._atr(df, 14)
        bb_upper, bb_mid, bb_lower = self._bollinger(close, 20, 2.0)
        bb_width = (bb_upper - bb_lower) / (bb_mid + 1e-10)
        bb_pctb = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)

        # Keltner Channel
        kc_mid = ema21
        kc_upper = kc_mid + 1.5 * atr14
        kc_lower = kc_mid - 1.5 * atr14
        kc_pos = (close - kc_lower) / (kc_upper - kc_lower + 1e-10)

        # Historical volatility ratio (short vs long)
        vol_short = close.pct_change().rolling(5, min_periods=1).std()
        vol_long = close.pct_change().rolling(20, min_periods=1).std()
        vol_ratio = vol_short / (vol_long + 1e-10)

        features['atr_norm'] = atr14.iloc[-1] / (px + 1e-10)
        features['bb_width'] = bb_width.iloc[-1]
        features['bb_pctb'] = np.clip(bb_pctb.iloc[-1], -0.5, 1.5)
        features['kc_pos'] = np.clip(kc_pos.iloc[-1], -0.5, 1.5)
        # Squeeze detection (BB inside KC)
        features['squeeze'] = 1.0 if (bb_upper.iloc[-1] < kc_upper.iloc[-1] and bb_lower.iloc[-1] > kc_lower.iloc[-1]) else 0.0
        features['vol_ratio'] = np.clip(vol_ratio.iloc[-1], 0, 5)

        # ── Volume ───────────────────────────────────────────────────
        vol_sma = self._sma(vol, 20)
        features['vol_sma_ratio'] = vol.iloc[-1] / (vol_sma.iloc[-1] + 1e-10)
        # OBV slope (simplified)
        direction = np.sign(close.diff().fillna(0))
        obv = (vol * direction).cumsum()
        obv_slope = obv.diff(5).fillna(0)
        features['obv_slope'] = obv_slope.iloc[-1] / (vol_sma.iloc[-1] + 1e-10)
        features['tick_intensity'] = np.clip(ticks.iloc[-1] / 20.0, 0, 3)

        # ── Candlestick Patterns ─────────────────────────────────────
        body = abs(close.iloc[-1] - opn.iloc[-1])
        total_range = high.iloc[-1] - low.iloc[-1] + 1e-10
        upper_wick = high.iloc[-1] - max(close.iloc[-1], opn.iloc[-1])
        lower_wick = min(close.iloc[-1], opn.iloc[-1]) - low.iloc[-1]

        # Engulfing
        if len(df) >= 2:
            prev = df.iloc[-2]
            bull_engulf = (prev['close'] < prev['open'] and close.iloc[-1] > opn.iloc[-1] and
                          close.iloc[-1] > prev['open'] and opn.iloc[-1] < prev['close'])
            bear_engulf = (prev['close'] > prev['open'] and close.iloc[-1] < opn.iloc[-1] and
                          close.iloc[-1] < prev['open'] and opn.iloc[-1] > prev['close'])
        else:
            bull_engulf = bear_engulf = False

        features['engulf_score'] = 1.0 if bull_engulf else (-1.0 if bear_engulf else 0.0)
        features['pinbar_score'] = (lower_wick - upper_wick) / total_range  # positive = bullish pin
        features['body_ratio'] = body / total_range
        features['upper_wick_ratio'] = upper_wick / total_range
        features['lower_wick_ratio'] = lower_wick / total_range

        # Inside bar
        if len(df) >= 2:
            prev = df.iloc[-2]
            inside = (high.iloc[-1] <= prev['high'] and low.iloc[-1] >= prev['low'])
        else:
            inside = False
        features['inside_bar'] = 1.0 if inside else 0.0

        # Three soldiers / crows (simplified)
        if len(df) >= 3:
            last3_bullish = all(df['close'].iloc[-i] > df['open'].iloc[-i] for i in range(1, 4))
            last3_bearish = all(df['close'].iloc[-i] < df['open'].iloc[-i] for i in range(1, 4))
        else:
            last3_bullish = last3_bearish = False
        features['three_soldiers'] = 1.0 if last3_bullish else (-1.0 if last3_bearish else 0.0)

        # ── Statistical ─────────────────────────────────────────────
        returns = close.pct_change().fillna(0)
        features['return_zscore'] = ((returns.iloc[-1] - returns.rolling(20, min_periods=1).mean().iloc[-1]) /
                                      (returns.rolling(20, min_periods=1).std().iloc[-1] + 1e-10))
        features['return_zscore'] = np.clip(features['return_zscore'], -4, 4)

        ret_window = returns.iloc[-20:] if len(returns) >= 20 else returns
        features['skewness'] = float(ret_window.skew()) if len(ret_window) > 2 else 0.0
        features['kurtosis'] = np.clip(float(ret_window.kurtosis()) if len(ret_window) > 3 else 0.0, -10, 10)

        # Hurst exponent (expensive — use cached / sampled)
        features['hurst'] = self._hurst_exponent(close.iloc[-60:]) if len(close) >= 60 else 0.5

        # ── Multi-Timeframe (simulated from 1m) ─────────────────────
        # 5-candle RSI (simulate ~5m timeframe)
        if len(df) >= 5:
            close_5m = close.iloc[::5]
            rsi_5m = self._rsi(close_5m, 14)
            features['rsi_5m'] = rsi_5m.iloc[-1] / 100.0 if len(rsi_5m) > 0 else 0.5
        else:
            features['rsi_5m'] = 0.5

        # 15-candle trend alignment
        if len(df) >= 15:
            close_15m = close.iloc[::15]
            ema_15m_fast = self._ema(close_15m, 5)
            ema_15m_slow = self._ema(close_15m, 14)
            features['htf_trend'] = 1.0 if ema_15m_fast.iloc[-1] > ema_15m_slow.iloc[-1] else 0.0
        else:
            features['htf_trend'] = 0.5

        # Compile into vector
        names = list(features.keys())
        vec = np.array([features[n] for n in names], dtype=np.float64)
        # Replace any NaN/Inf
        vec = np.nan_to_num(vec, nan=0.0, posinf=5.0, neginf=-5.0)
        return vec.reshape(1, -1), names


# ════════════════════════════════════════════════════════════════════════
#  3.  REGIME DETECTOR
# ════════════════════════════════════════════════════════════════════════

class RegimeDetector:
    """Classifies market into regimes using ADX, volatility, and trend metrics."""

    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"

    @staticmethod
    def detect(features: Dict[str, float]) -> str:
        adx = features.get('adx', 0) * 100       # un-normalize
        vol_ratio = features.get('vol_ratio', 1)
        ema_stack = features.get('ema_stack', 0.5)
        bb_width = features.get('bb_width', 0)
        di_diff = features.get('di_diff', 0) * 100

        # High volatility regime
        if vol_ratio > 2.0 and bb_width > 0.06:
            return RegimeDetector.VOLATILE

        # Strong trend
        if adx > 25:
            if di_diff > 0 and ema_stack >= 0.75:
                return RegimeDetector.TRENDING_UP
            elif di_diff < 0 and ema_stack <= 0.25:
                return RegimeDetector.TRENDING_DOWN

        # Ranging
        return RegimeDetector.RANGING


# ════════════════════════════════════════════════════════════════════════
#  4.  HYDRA ENSEMBLE
# ════════════════════════════════════════════════════════════════════════

# Feature groups for specialist differentiation
MOMENTUM_FEATURES = [
    'rsi14', 'rsi7', 'stoch_rsi', 'macd_norm', 'macd_signal_norm',
    'macd_hist_norm', 'macd_cross', 'roc5', 'roc10', 'roc20',
    'ema_slope5', 'ema_slope21', 'adx', 'di_diff',
]
REVERSAL_FEATURES = [
    'rsi14', 'stoch_rsi', 'bb_pctb', 'kc_pos', 'squeeze',
    'engulf_score', 'pinbar_score', 'body_ratio', 'upper_wick_ratio',
    'lower_wick_ratio', 'inside_bar', 'three_soldiers',
    'return_zscore', 'vol_ratio',
]


class HydraEnsemble:
    """
    Manages 3 specialist DeepMLP networks with feature-subset differentiation
    and regime-adaptive weighting.
    """

    def __init__(self, n_features: int, feature_names: List[str] = None):
        self.n_features = n_features
        self.feature_names = feature_names or []

        # Compute feature indices for each specialist
        self.momentum_idx = self._get_feature_indices(MOMENTUM_FEATURES)
        self.reversal_idx = self._get_feature_indices(REVERSAL_FEATURES)

        n_mom = len(self.momentum_idx) if self.momentum_idx is not None else n_features
        n_rev = len(self.reversal_idx) if self.reversal_idx is not None else n_features

        # 3 specialist nets — smaller architectures to prevent overfitting
        self.momentum_net = DeepMLP([n_mom, 24, 12, 1], lr=0.001, dropout=0.30, l2=5e-4)
        self.reversal_net = DeepMLP([n_rev, 20, 10, 1], lr=0.001, dropout=0.30, l2=5e-4)
        self.pattern_net = DeepMLP([n_features, 16, 1], lr=0.001, dropout=0.25, l2=5e-4)

        self.nets = {
            'momentum': self.momentum_net,
            'reversal': self.reversal_net,
            'pattern': self.pattern_net,
        }

        # Regime-specific weights
        self.regime_weights = {
            RegimeDetector.TRENDING_UP:   {'momentum': 0.50, 'reversal': 0.15, 'pattern': 0.35},
            RegimeDetector.TRENDING_DOWN: {'momentum': 0.50, 'reversal': 0.15, 'pattern': 0.35},
            RegimeDetector.RANGING:       {'momentum': 0.15, 'reversal': 0.50, 'pattern': 0.35},
            RegimeDetector.VOLATILE:      {'momentum': 0.25, 'reversal': 0.35, 'pattern': 0.40},
        }

        self.trained = False

    def _get_feature_indices(self, feature_list: List[str]) -> Optional[List[int]]:
        """Get indices of named features. Returns None if names not available."""
        if not self.feature_names:
            return None
        indices = []
        for name in feature_list:
            if name in self.feature_names:
                indices.append(self.feature_names.index(name))
        return indices if indices else None

    def _subset_features(self, X: np.ndarray, indices: Optional[List[int]]) -> np.ndarray:
        """Extract feature subset. Returns full X if indices are None."""
        if indices is None:
            return X
        return X[:, indices]

    def predict(self, X: np.ndarray, regime: str) -> Tuple[float, Dict[str, float]]:
        """
        Returns (aggregated_probability, per_specialist_probs).
        Probability > 0.5 means UP, < 0.5 means DOWN.
        """
        weights = self.regime_weights.get(regime, {'momentum': 0.33, 'reversal': 0.34, 'pattern': 0.33})
        probs = {}

        # Each specialist gets its own feature subset
        X_mom = self._subset_features(X, self.momentum_idx)
        X_rev = self._subset_features(X, self.reversal_idx)

        probs['momentum'] = float(self.momentum_net.predict(X_mom)[0, 0])
        probs['reversal'] = float(self.reversal_net.predict(X_rev)[0, 0])
        probs['pattern'] = float(self.pattern_net.predict(X)[0, 0])

        # Weighted average
        agg = sum(probs[n] * weights[n] for n in probs)
        return agg, probs

    def specialist_agreement(self, probs: Dict[str, float]) -> Tuple[bool, float]:
        """
        Check if specialists agree on direction.
        Returns (agree: bool, spread: float).
        Agreement = at least 2 of 3 on same side of 0.5.
        """
        vals = list(probs.values())
        ups = sum(1 for v in vals if v > 0.5)
        downs = sum(1 for v in vals if v <= 0.5)
        agree = ups >= 2 or downs >= 2
        spread = max(vals) - min(vals)
        return agree, spread

    def train_all(self, X: np.ndarray, y: np.ndarray, epochs: int = 300,
                  batch_size: int = 64, verbose: bool = False,
                  X_val: np.ndarray = None, y_val: np.ndarray = None):
        """Train specialists on their respective feature subsets with early stopping."""
        X_mom = self._subset_features(X, self.momentum_idx)
        X_rev = self._subset_features(X, self.reversal_idx)
        X_val_mom = self._subset_features(X_val, self.momentum_idx) if X_val is not None else None
        X_val_rev = self._subset_features(X_val, self.reversal_idx) if X_val is not None else None

        if verbose:
            print("  Training Momentum Net...")
        self.momentum_net.train_epochs(X_mom, y, epochs, batch_size, verbose,
                                       X_val=X_val_mom, y_val=y_val)
        if verbose:
            print("  Training Reversal Net...")
        self.reversal_net.train_epochs(X_rev, y, epochs, batch_size, verbose,
                                       X_val=X_val_rev, y_val=y_val)
        if verbose:
            print("  Training Pattern Net...")
        self.pattern_net.train_epochs(X, y, epochs, batch_size, verbose,
                                      X_val=X_val, y_val=y_val)
        self.trained = True

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        for name, net in self.nets.items():
            net.save(os.path.join(directory, f"{name}_net.json"))

    def load(self, directory: str) -> bool:
        """Returns True if all models loaded successfully."""
        try:
            loaded = {}
            for name in self.nets:
                path = os.path.join(directory, f"{name}_net.json")
                if not os.path.exists(path):
                    return False
                loaded[name] = DeepMLP.load(path)
            self.nets = loaded
            self.momentum_net = loaded['momentum']
            self.reversal_net = loaded['reversal']
            self.pattern_net = loaded['pattern']
            self.trained = True
            return True
        except Exception:
            return False


# ════════════════════════════════════════════════════════════════════════
#  5.  CONFIDENCE GATE
# ════════════════════════════════════════════════════════════════════════

class ConfidenceGate:
    """
    Adaptive threshold that adjusts based on rolling win rate.
    Includes probability calibration to de-saturate overfit sigmoid outputs
    and ensemble disagreement filtering.
    """

    def __init__(self, base_threshold: float = 0.68, buffer_size: int = 50,
                 max_specialist_spread: float = 0.40):
        self.base = base_threshold
        self.outcomes = deque(maxlen=buffer_size)
        self.threshold = base_threshold
        self.max_specialist_spread = max_specialist_spread

    @staticmethod
    def calibrate(raw_confidence: float) -> float:
        """
        Compress overconfident sigmoid outputs toward 0.5.
        Maps [0.5, 1.0] → [0.5, ~0.80] with a compression factor.
        """
        return 0.5 + (raw_confidence - 0.5) * 0.6

    def update(self, won: bool):
        self.outcomes.append(1.0 if won else 0.0)
        self._recalc()

    def _recalc(self):
        if len(self.outcomes) < 5:
            self.threshold = self.base
            return
        wr = np.mean(list(self.outcomes))
        if wr > 0.85:
            self.threshold = max(0.58, self.base - 0.05)  # Slightly aggressive
        elif wr > 0.75:
            self.threshold = self.base - 0.02
        elif wr < 0.55:
            self.threshold = min(0.82, self.base + 0.10)  # Very conservative
        elif wr < 0.65:
            self.threshold = self.base + 0.05
        else:
            self.threshold = self.base

    def passes(self, confidence: float, specialist_spread: float = 0.0) -> bool:
        """Check if confidence passes threshold and specialists agree."""
        if specialist_spread > self.max_specialist_spread:
            return False  # Specialists disagree too much
        return confidence >= self.threshold

    @property
    def win_rate(self) -> float:
        if not self.outcomes:
            return 0.0
        return np.mean(list(self.outcomes))


# ════════════════════════════════════════════════════════════════════════
#  6.  ONLINE LEARNER
# ════════════════════════════════════════════════════════════════════════

class OnlineLearner:
    """
    Replay buffer that stores (features, outcome) pairs.
    After accumulating enough samples, performs a mini-batch training pass.
    """

    def __init__(self, retrain_interval: int = 20, buffer_size: int = 500):
        self.retrain_interval = retrain_interval
        self.buffer_X = deque(maxlen=buffer_size)
        self.buffer_y = deque(maxlen=buffer_size)
        self.samples_since_train = 0

    def record(self, features: np.ndarray, outcome: float):
        """Record a feature vector and its binary outcome (1=UP won, 0=UP lost)."""
        self.buffer_X.append(features.flatten())
        self.buffer_y.append(outcome)
        self.samples_since_train += 1

    def should_retrain(self) -> bool:
        return (self.samples_since_train >= self.retrain_interval and
                len(self.buffer_X) >= 30)

    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        X = np.array(list(self.buffer_X))
        y = np.array(list(self.buffer_y)).reshape(-1, 1)
        self.samples_since_train = 0
        return X, y

    @property
    def size(self) -> int:
        return len(self.buffer_X)


# ════════════════════════════════════════════════════════════════════════
#  7.  RISK MANAGER + COMPOUNDING SIZER
# ════════════════════════════════════════════════════════════════════════

class RiskManager:
    """
    - Consecutive loss streak filter
    - Session drawdown protection
    - Compounding stake sizer (aggressive growth)
    """

    def __init__(self, initial_balance: float = 1000.0,
                 max_streak: int = 3, cooldown_candles: int = 2,
                 max_drawdown_pct: float = 0.15):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.max_streak = max_streak
        self.cooldown_candles = cooldown_candles
        self.max_drawdown_pct = max_drawdown_pct

        self.consecutive_losses = 0
        self.cooldown_remaining = 0
        self.session_pnl = 0.0
        self.peak_balance = initial_balance

    def can_trade(self) -> Tuple[bool, str]:
        # Cooldown after streak
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            return False, f"Cooling down ({self.cooldown_remaining} candles left)"

        # Drawdown protection
        drawdown = (self.peak_balance - self.balance) / (self.peak_balance + 1e-10)
        if drawdown > self.max_drawdown_pct:
            return False, f"Drawdown limit hit ({drawdown:.1%})"

        return True, "OK"

    def record_outcome(self, won: bool, pnl: float):
        self.session_pnl += pnl
        self.balance += pnl
        self.peak_balance = max(self.peak_balance, self.balance)

        if won:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.max_streak:
                self.cooldown_remaining = self.cooldown_candles
                self.consecutive_losses = 0

    def compute_stake(self, confidence: float, balance: float) -> float:
        """
        Compounding stake sizer:
        - Base: 3% of balance
        - Scaled by confidence (0.5–1.0 → 0.5x–1.5x multiplier)
        - After a win streak, ramp up; after losses, scale down
        """
        base_pct = 0.03  # 3% of balance base
        conf_mult = 0.5 + (confidence - 0.5) * 2.0  # Maps [0.5, 1.0] → [0.5, 1.5]
        conf_mult = np.clip(conf_mult, 0.5, 1.5)

        # Streak multiplier
        if self.consecutive_losses == 0:
            streak_mult = 1.2  # Slightly aggressive after wins
        elif self.consecutive_losses == 1:
            streak_mult = 0.8
        else:
            streak_mult = 0.5  # Very conservative

        stake = balance * base_pct * conf_mult * streak_mult
        # Floor / ceiling
        stake = max(1.0, min(stake, balance * 0.10))  # Never exceed 10% of balance
        return round(stake, 2)


# ════════════════════════════════════════════════════════════════════════
#  8.  FEATURE NORMALIZER
# ════════════════════════════════════════════════════════════════════════

class FeatureNormalizer:
    """Online z-score normalizer with running mean/std and freeze support."""

    def __init__(self, n_features: int, warmup: int = 50):
        self.n = n_features
        self.warmup = warmup
        self.count = 0
        self.mean = np.zeros(n_features)
        self.M2 = np.zeros(n_features)  # For Welford's algorithm
        self.frozen = False

    def freeze(self):
        """Freeze statistics — update() becomes a no-op."""
        self.frozen = True

    def unfreeze(self):
        """Unfreeze statistics — update() resumes."""
        self.frozen = False

    def update(self, x: np.ndarray):
        """Update running statistics with a new sample (1, n_features)."""
        if self.frozen:
            return  # Don't update during test phase
        x_flat = x.flatten()
        self.count += 1
        delta = x_flat - self.mean
        self.mean += delta / self.count
        delta2 = x_flat - self.mean
        self.M2 += delta * delta2

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Z-score normalize using running stats."""
        if self.count < self.warmup:
            return x  # Not enough data yet
        std = np.sqrt(self.M2 / (self.count - 1 + 1e-10))
        std = np.maximum(std, 1e-8)
        return (x - self.mean) / std

    def save(self, path: str):
        data = {'count': self.count, 'mean': self.mean.tolist(), 'M2': self.M2.tolist()}
        with open(path, 'w') as f:
            json.dump(data, f)

    def load(self, path: str):
        with open(path, 'r') as f:
            data = json.load(f)
        self.count = data['count']
        self.mean = np.array(data['mean'])
        self.M2 = np.array(data['M2'])


# ════════════════════════════════════════════════════════════════════════
#  9.  HYDRA STRATEGY (Main Interface)
# ════════════════════════════════════════════════════════════════════════

class HydraStrategy:
    """
    Main strategy interface — drop-in replacement for StrategyEngine.

    Usage:
        strategy = HydraStrategy()
        result = strategy.execute(df_window)
        # result = { 'decision': 'UP'/'DOWN'/'WAIT', 'confidence': ..., ... }
    """

    MODEL_DIR = "models/hydra"

    def __init__(self, initial_balance: float = 1000.0):
        self.feature_factory = FeatureFactory()
        self.regime_detector = RegimeDetector()
        # Increased baseline threshold for enhanced win rate selectivity
        self.confidence_gate = ConfidenceGate(base_threshold=0.73)
        self.online_learner = OnlineLearner(retrain_interval=20, buffer_size=500)
        self.risk_manager = RiskManager(initial_balance=initial_balance)
        self.online_learning_enabled = True  # Disable during backtesting

        # These are initialized lazily (need to know n_features)
        self.ensemble: Optional[HydraEnsemble] = None
        self.normalizer: Optional[FeatureNormalizer] = None
        self.feature_names: List[str] = []
        self._initialized = False

        # Pending trade for outcome tracking
        self._pending_features: Optional[np.ndarray] = None
        self._pending_direction: Optional[str] = None

        # Try to load saved model
        self._try_load()

    def _try_load(self):
        """Attempt to load a previously trained model."""
        norm_path = os.path.join(self.MODEL_DIR, "normalizer.json")
        if os.path.isdir(self.MODEL_DIR):
            try:
                # We need to know n_features to initialize
                meta_path = os.path.join(self.MODEL_DIR, "meta.json")
                if os.path.exists(meta_path):
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    n_features = meta['n_features']
                    self.feature_names = meta.get('feature_names', [])

                    self.ensemble = HydraEnsemble(n_features, feature_names=self.feature_names)
                    if self.ensemble.load(self.MODEL_DIR):
                        self.normalizer = FeatureNormalizer(n_features)
                        if os.path.exists(norm_path):
                            self.normalizer.load(norm_path)
                        self._initialized = True
            except Exception:
                pass

    def _initialize(self, n_features: int, feature_names: List[str]):
        """Lazy init once we know the feature count."""
        self.ensemble = HydraEnsemble(n_features, feature_names=feature_names)
        self.normalizer = FeatureNormalizer(n_features)
        self.feature_names = feature_names
        self._initialized = True

    def save_models(self):
        """Persist all models and normalizer to disk."""
        if not self._initialized or self.ensemble is None:
            return
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        self.ensemble.save(self.MODEL_DIR)
        self.normalizer.save(os.path.join(self.MODEL_DIR, "normalizer.json"))
        meta = {'n_features': self.ensemble.n_features, 'feature_names': self.feature_names}
        with open(os.path.join(self.MODEL_DIR, "meta.json"), 'w') as f:
            json.dump(meta, f)

    def train_on_historical(self, df: pd.DataFrame, epochs: int = 500, verbose: bool = True):
        """
        Train the ensemble on historical data using walk-forward labels.
        Uses 80/20 train/val split with early stopping to prevent overfitting.
        """
        if verbose:
            print("═══ HydraNet Training ═══")
            print(f"  Data: {len(df)} candles")

        # Build features for each candle (need lookback)
        min_lookback = 120
        X_list = []
        y_list = []

        for i in range(min_lookback, len(df) - 1):
            curr_close = df['close'].iloc[i]
            next_close = df['close'].iloc[i + 1]
            
            # Skip flat candles (exact ties) to prevent mathematically forced class imbalances
            if abs(next_close - curr_close) < 1e-6:
                continue
                
            window = df.iloc[max(0, i - 200):i + 1]
            features, names = self.feature_factory.build(window)
            
            # Label: 1.0 = UP, 0.0 = DOWN
            label = 1.0 if next_close > curr_close else 0.0
            
            X_list.append(features.flatten())
            y_list.append(label)

        X = np.array(X_list)
        y = np.array(y_list).reshape(-1, 1)

        if len(X) == 0:
            raise ValueError(
                f"Insufficient data for training! The dataframe only has {len(df)} candles. "
                f"HydraNet requires a minimum lookback of {min_lookback} candles just to generate features, "
                f"so len(df) must be strictly greater than {min_lookback}."
            )

        if verbose:
            print(f"  Samples: {len(X)} | Features: {X.shape[1]}")
            print(f"  Class Balance: UP={y.mean():.2%} | DOWN={1-y.mean():.2%}")

        # Always (re-)initialize ensemble with correct architecture
        # This ensures fresh training with proper feature subsets
        self._initialize(X.shape[1], names)

        # 80/20 train/validation split (temporal — no shuffle)
        val_split = int(len(X) * 0.80)
        X_train, X_val = X[:val_split], X[val_split:]
        y_train, y_val = y[:val_split], y[val_split:]

        if verbose:
            print(f"  Train split: {len(X_train)} | Val split: {len(X_val)}")

        # Update normalizer with TRAINING data only
        for row in X_train:
            self.normalizer.update(row.reshape(1, -1))

        # Normalize both splits using training statistics
        X_train_norm = np.array([self.normalizer.transform(row.reshape(1, -1)).flatten() for row in X_train])
        X_val_norm = np.array([self.normalizer.transform(row.reshape(1, -1)).flatten() for row in X_val])

        # Freeze normalizer after training to prevent test-time drift
        self.normalizer.freeze()

        # Train ensemble with early stopping
        if verbose:
            print("\n  Training Ensemble...")
        self.ensemble.train_all(X_train_norm, y_train, epochs=epochs, batch_size=64,
                                verbose=verbose, X_val=X_val_norm, y_val=y_val)

        # Save
        self.save_models()
        if verbose:
            # Training accuracy (on train split)
            preds_train = np.array([self.ensemble.predict(x.reshape(1, -1), RegimeDetector.RANGING)[0] for x in X_train_norm])
            train_acc = np.mean((preds_train > 0.5).astype(float) == y_train.flatten())
            # Validation accuracy
            preds_val = np.array([self.ensemble.predict(x.reshape(1, -1), RegimeDetector.RANGING)[0] for x in X_val_norm])
            val_acc = np.mean((preds_val > 0.5).astype(float) == y_val.flatten())
            overfit_gap = train_acc - val_acc
            gap_color = '⚠️' if overfit_gap > 0.15 else '✓'
            print(f"\n  Training Accuracy: {train_acc:.2%}")
            print(f"  Validation Accuracy: {val_acc:.2%}  {gap_color}")
            if overfit_gap > 0.15:
                print(f"  ⚠️  Overfit gap: {overfit_gap:.2%} — model may not generalize well")
            print("═══ Training Complete ═══\n")

    def execute(self, df: pd.DataFrame) -> Dict:
        """
        Main signal generator — compatible with existing StrategyEngine interface.

        Returns dict with:
            decision: 'UP' / 'DOWN' / 'WAIT'
            confidence: float
            reason: str
            confluence_score: float
            regime: str
            specialist_votes: dict
            stake: float
            metrics: dict
        """
        if len(df) < 50:
            return self._wait("Insufficient data")

        # Build features
        features_raw, names = self.feature_factory.build(df)

        # Initialize on first call
        if not self._initialized:
            self._initialize(features_raw.shape[1], names)

        # Update normalizer
        self.normalizer.update(features_raw)

        # Detect regime
        feat_dict = {names[i]: features_raw[0, i] for i in range(len(names))}
        regime = self.regime_detector.detect(feat_dict)

        # If model not trained yet, use rule-based fallback
        if not self.ensemble.trained:
            return self._rule_based_fallback(feat_dict, regime)

        # Normalize and predict
        X_norm = self.normalizer.transform(features_raw)
        prob, specialist_probs = self.ensemble.predict(X_norm, regime)

        # Check specialist agreement
        agree, spread = self.ensemble.specialist_agreement(specialist_probs)
        if not agree:
            return self._wait(
                f"Specialist disagreement (spread={spread:.2f})",
                regime=regime, confidence=0.5,
                specialist_votes=specialist_probs
            )

        # Determine direction and confidence
        if prob > 0.5:
            direction = "UP"
            raw_confidence = prob
        else:
            direction = "DOWN"
            raw_confidence = 1.0 - prob

        # Calibrate confidence (de-saturate overfit sigmoid)
        confidence = self.confidence_gate.calibrate(raw_confidence)

        # Risk check
        can_trade, risk_reason = self.risk_manager.can_trade()
        if not can_trade:
            return self._wait(risk_reason, regime=regime, confidence=confidence,
                            specialist_votes=specialist_probs)

        # Confidence gate (with specialist spread check)
        if not self.confidence_gate.passes(confidence, spread):
            return self._wait(
                f"Low confidence ({confidence:.1%} < {self.confidence_gate.threshold:.1%})",
                regime=regime, confidence=confidence,
                specialist_votes=specialist_probs
            )

        # Compute stake
        stake = self.risk_manager.compute_stake(confidence, self.risk_manager.balance)

        # Store pending for online learning
        self._pending_features = features_raw
        self._pending_direction = direction

        # Extract key metrics for dashboard
        metrics = {
            'rsi': feat_dict.get('rsi14', 0.5) * 100,
            'adx': feat_dict.get('adx', 0) * 100,
            'ema50': feat_dict.get('ema50_dist', 0),
            'bb_pctb': feat_dict.get('bb_pctb', 0.5),
            'pattern': self._describe_pattern(feat_dict),
            'px': df['close'].iloc[-1],
            'bb_up': 0, 'bb_low': 0,  # placeholder for dashboard compat
        }

        return {
            'decision': direction,
            'confidence': confidence,
            'reason': f"HydraNet {regime} | {direction} @ {confidence:.1%}",
            'confluence_score': confidence,
            'regime': regime,
            'specialist_votes': specialist_probs,
            'stake': stake,
            'features': [feat_dict.get('rsi14', 0.5) * 100, feat_dict.get('adx', 0) * 100,
                        feat_dict.get('bb_width', 0), feat_dict.get('ema50_dist', 0)],
            'metrics': metrics,
        }

    def record_outcome(self, won: bool, pnl: float = 0.0):
        """
        Call this after a trade resolves to feed the online learning system.
        Online retraining only occurs when online_learning_enabled is True.
        """
        self.confidence_gate.update(won)
        self.risk_manager.record_outcome(won, pnl)

        if not self.online_learning_enabled:
            self._pending_features = None
            self._pending_direction = None
            return

        # Store in replay buffer
        if self._pending_features is not None:
            # For UP trades: won=True means label=1.0
            # For DOWN trades: won=True means label=0.0
            if self._pending_direction == "UP":
                label = 1.0 if won else 0.0
            else:
                label = 0.0 if won else 1.0
            X_norm = self.normalizer.transform(self._pending_features)
            self.online_learner.record(X_norm, label)
            self._pending_features = None
            self._pending_direction = None

        # Online retrain check
        if self.online_learner.should_retrain() and self.ensemble is not None:
            X, y = self.online_learner.get_training_data()
            self.ensemble.train_all(X, y, epochs=50, batch_size=32, verbose=False)
            self.save_models()

    def _wait(self, reason: str, regime: str = "UNKNOWN",
              confidence: float = 0.0, specialist_votes: Dict = None) -> Dict:
        return {
            'decision': 'WAIT',
            'confidence': confidence,
            'reason': reason,
            'confluence_score': confidence,
            'regime': regime,
            'specialist_votes': specialist_votes or {},
            'stake': 0,
            'features': [50, 0, 0, 0],
            'metrics': {'rsi': 50, 'adx': 0, 'ema50': 0, 'bb_pctb': 0.5,
                       'pattern': '-', 'px': 0, 'bb_up': 0, 'bb_low': 0},
        }

    def _rule_based_fallback(self, feat: Dict, regime: str) -> Dict:
        """Simple rule-based signals when NN hasn't been trained yet."""
        rsi = feat.get('rsi14', 0.5) * 100
        bb_pctb = feat.get('bb_pctb', 0.5)
        macd_cross = feat.get('macd_cross', 0.5)
        ema_stack = feat.get('ema_stack', 0.5)

        decision = "WAIT"
        confidence = 0.5
        reason = "Rule-based (model not trained)"

        # Oversold bounce
        if rsi < 30 and bb_pctb < 0.1:
            decision = "UP"
            confidence = 0.70
            reason = "Rule: Extreme oversold"
        # Overbought reversal
        elif rsi > 70 and bb_pctb > 0.9:
            decision = "DOWN"
            confidence = 0.70
            reason = "Rule: Extreme overbought"
        # Trend continuation
        elif ema_stack == 1.0 and macd_cross == 1.0 and rsi > 50:
            decision = "UP"
            confidence = 0.65
            reason = "Rule: Trend continuation UP"
        elif ema_stack == 0.0 and macd_cross == 0.0 and rsi < 50:
            decision = "DOWN"
            confidence = 0.65
            reason = "Rule: Trend continuation DOWN"

        if decision == "WAIT":
            return self._wait(reason, regime=regime)

        return {
            'decision': decision,
            'confidence': confidence,
            'reason': reason,
            'confluence_score': confidence,
            'regime': regime,
            'specialist_votes': {},
            'stake': self.risk_manager.compute_stake(confidence, self.risk_manager.balance),
            'features': [rsi, 0, 0, 0],
            'metrics': {'rsi': rsi, 'adx': 0, 'ema50': 0, 'bb_pctb': bb_pctb,
                       'pattern': '-', 'px': 0, 'bb_up': 0, 'bb_low': 0},
        }

    @staticmethod
    def _describe_pattern(feat: Dict) -> str:
        e = feat.get('engulf_score', 0)
        p = feat.get('pinbar_score', 0)
        s = feat.get('three_soldiers', 0)
        parts = []
        if e > 0.5:
            parts.append("Bull Engulf")
        elif e < -0.5:
            parts.append("Bear Engulf")
        if p > 0.3:
            parts.append("Bull Pin")
        elif p < -0.3:
            parts.append("Bear Pin")
        if s > 0.5:
            parts.append("3 Soldiers")
        elif s < -0.5:
            parts.append("3 Crows")
        if feat.get('inside_bar', 0) > 0.5:
            parts.append("Inside Bar")
        return " + ".join(parts) if parts else "None"


# ════════════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("HydraNet Strategy Engine v1.0")
    print("Use hydra_backtester.py to test, or integrate with live_trader.py")
