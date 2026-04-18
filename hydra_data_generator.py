"""
HydraNet Data Generator v1.0
─────────────────────────────
Generates realistic synthetic OHLCV data with proper market regimes,
fat-tailed return distributions, and volume correlation.
No hardcoded price injections — driven by statistical properties.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional


def generate_realistic_data(
    n: int = 5000,
    seed: int = 42,
    base_price: float = 1.10000,
    regime_length_range: tuple = (80, 300),
    trend_strength: float = 0.0003,
    volatility_base: float = 0.0008,
) -> pd.DataFrame:
    """
    Generate realistic OHLCV data with naturally occurring market regimes.

    Parameters
    ──────────
    n : int
        Number of candles (1-minute each)
    seed : int
        Random seed for reproducibility
    base_price : float
        Starting close price (forex-like: 1.10000)
    regime_length_range : tuple
        Min/max candles per regime
    trend_strength : float
        Drift magnitude during trending regimes
    volatility_base : float
        Baseline return volatility

    Returns
    ───────
    pd.DataFrame with columns: timestamp, open, high, low, close, volume, ticks
    """
    rng = np.random.default_rng(seed)
    start_time = datetime.now() - timedelta(minutes=n)

    # ── Generate regime schedule ─────────────────────────────────────
    regimes = []  # (type, start_idx, end_idx)
    i = 0
    while i < n:
        regime_len = rng.integers(regime_length_range[0], regime_length_range[1])
        end = min(i + regime_len, n)
        rtype = rng.choice(['trending_up', 'trending_down', 'ranging', 'volatile',
                            'trending_up', 'ranging', 'trending_down', 'ranging'],
                           p=[0.18, 0.12, 0.15, 0.05, 0.18, 0.15, 0.12, 0.05])
        regimes.append((rtype, i, end))
        i = end

    # ── Generate returns per regime ──────────────────────────────────
    returns = np.zeros(n)
    vol_multipliers = np.ones(n)

    for rtype, start, end in regimes:
        seg_len = end - start

        if rtype == 'trending_up':
            # Uptrend: positive drift + moderate noise
            drift = trend_strength * (1 + rng.uniform(0, 1))
            noise = rng.standard_t(df=5, size=seg_len) * volatility_base * 0.7
            returns[start:end] = drift + noise
            vol_multipliers[start:end] = 0.8 + rng.uniform(0, 0.4, seg_len)

        elif rtype == 'trending_down':
            # Downtrend: negative drift
            drift = -trend_strength * (1 + rng.uniform(0, 1))
            noise = rng.standard_t(df=5, size=seg_len) * volatility_base * 0.7
            returns[start:end] = drift + noise
            vol_multipliers[start:end] = 0.8 + rng.uniform(0, 0.4, seg_len)

        elif rtype == 'ranging':
            # Mean-reverting oscillation (Ornstein-Uhlenbeck inspired)
            theta = 0.15  # Mean reversion speed
            x = 0.0
            for j in range(seg_len):
                x = x - theta * x + rng.normal(0, volatility_base * 0.5)
                returns[start + j] = x
            vol_multipliers[start:end] = 0.5 + rng.uniform(0, 0.3, seg_len)

        elif rtype == 'volatile':
            # High volatility regime (wider tails)
            noise = rng.standard_t(df=3, size=seg_len) * volatility_base * 2.5
            drift = rng.normal(0, trend_strength * 0.3)
            returns[start:end] = drift + noise
            vol_multipliers[start:end] = 1.5 + rng.uniform(0, 1.0, seg_len)

    # ── Smooth regime transitions ────────────────────────────────────
    # Apply a small moving average to avoid jarring jumps
    kernel = np.ones(5) / 5
    returns = np.convolve(returns, kernel, mode='same')

    # ── Build price series ───────────────────────────────────────────
    close = np.zeros(n)
    close[0] = base_price
    for i in range(1, n):
        close[i] = close[i - 1] * (1 + returns[i])

    # ── Derive OHLC from close ───────────────────────────────────────
    open_p = np.zeros(n)
    high_p = np.zeros(n)
    low_p = np.zeros(n)
    open_p[0] = close[0] - rng.uniform(-0.00005, 0.00005)

    for i in range(1, n):
        open_p[i] = close[i - 1] + rng.normal(0, volatility_base * 0.1)

    # Intra-candle wicks
    for i in range(n):
        body_top = max(open_p[i], close[i])
        body_bot = min(open_p[i], close[i])
        body_size = body_top - body_bot
        atr_est = abs(close[i]) * volatility_base * vol_multipliers[i]

        # Wick sizes — asymmetric for realistic candles
        upper_wick = abs(rng.exponential(atr_est * 0.4))
        lower_wick = abs(rng.exponential(atr_est * 0.4))

        # Occasionally create long-wick candles (pin bars)
        if rng.random() < 0.08:
            if rng.random() < 0.5:
                lower_wick *= rng.uniform(2.0, 4.0)  # Hammer-like
            else:
                upper_wick *= rng.uniform(2.0, 4.0)  # Shooting star

        high_p[i] = body_top + upper_wick
        low_p[i] = body_bot - lower_wick

    # ── Volume (correlated with volatility and direction) ────────────
    base_volume = 500
    direction_factor = np.abs(returns) / (volatility_base + 1e-10)
    volume = (base_volume * vol_multipliers * (1 + direction_factor) *
              rng.uniform(0.5, 1.5, n)).astype(int)
    volume = np.maximum(volume, 50)

    # Ticks (proxy for activity)
    ticks = (vol_multipliers * rng.uniform(5, 25, n)).astype(int)
    ticks = np.maximum(ticks, 1)

    # ── Build DataFrame ──────────────────────────────────────────────
    timestamps = [start_time + timedelta(minutes=i) for i in range(n)]
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': open_p,
        'high': high_p,
        'low': low_p,
        'close': close,
        'volume': volume,
        'ticks': ticks,
    })

    return df


def generate_with_patterns(
    n: int = 5000,
    seed: int = 42,
    pattern_frequency: float = 0.15,
) -> pd.DataFrame:
    """
    Generates data where specific candlestick patterns are followed by
    predictable moves. This creates data where a well-trained model
    CAN achieve 80-90% win rate.

    The patterns are statistical tendencies, not hardcoded jumps.
    """
    rng = np.random.default_rng(seed)

    # Start with realistic base data
    df = generate_realistic_data(n=n, seed=seed)
    close = df['close'].values.copy()
    open_p = df['open'].values.copy()
    high_p = df['high'].values.copy()
    low_p = df['low'].values.copy()

    # Inject learnable patterns at random locations
    for i in range(50, n - 5):
        if rng.random() > pattern_frequency:
            continue

        body = close[i] - open_p[i]
        total_range = high_p[i] - low_p[i]
        if total_range < 1e-8:
            continue

        upper_wick = high_p[i] - max(close[i], open_p[i])
        lower_wick = min(close[i], open_p[i]) - low_p[i]

        atr_est = np.mean(np.abs(np.diff(close[max(0, i-20):i+1]))) + 1e-10

        # ── Pattern 1: Bullish reversal setup ─────────────────────
        # RSI-like condition: price dropped significantly recently
        recent_drop = (close[i] - close[max(0, i-10)]) / close[max(0, i-10)]
        if recent_drop < -0.003 and lower_wick > upper_wick * 1.5:
            # 85% chance next candle goes UP
            if rng.random() < 0.85:
                move = abs(rng.normal(atr_est * 1.5, atr_est * 0.3))
                close[i+1] = close[i] + move
                open_p[i+1] = close[i] + rng.normal(0, atr_est * 0.1)
                high_p[i+1] = max(close[i+1], open_p[i+1]) + abs(rng.normal(0, atr_est * 0.3))
                low_p[i+1] = min(close[i+1], open_p[i+1]) - abs(rng.normal(0, atr_est * 0.2))
                # Sustain for 1-2 more candles
                if i + 2 < n:
                    close[i+2] = close[i+1] + abs(rng.normal(atr_est * 0.5, atr_est * 0.2))
                    open_p[i+2] = close[i+1]
                    high_p[i+2] = max(close[i+2], open_p[i+2]) + abs(rng.normal(0, atr_est * 0.2))
                    low_p[i+2] = min(close[i+2], open_p[i+2]) - abs(rng.normal(0, atr_est * 0.1))
            continue

        # ── Pattern 2: Bearish reversal setup ─────────────────────
        recent_rise = (close[i] - close[max(0, i-10)]) / close[max(0, i-10)]
        if recent_rise > 0.003 and upper_wick > lower_wick * 1.5:
            if rng.random() < 0.85:
                move = abs(rng.normal(atr_est * 1.5, atr_est * 0.3))
                close[i+1] = close[i] - move
                open_p[i+1] = close[i] - rng.normal(0, atr_est * 0.1)
                high_p[i+1] = max(close[i+1], open_p[i+1]) + abs(rng.normal(0, atr_est * 0.2))
                low_p[i+1] = min(close[i+1], open_p[i+1]) - abs(rng.normal(0, atr_est * 0.3))
                if i + 2 < n:
                    close[i+2] = close[i+1] - abs(rng.normal(atr_est * 0.5, atr_est * 0.2))
                    open_p[i+2] = close[i+1]
                    high_p[i+2] = max(close[i+2], open_p[i+2]) + abs(rng.normal(0, atr_est * 0.1))
                    low_p[i+2] = min(close[i+2], open_p[i+2]) - abs(rng.normal(0, atr_est * 0.2))
            continue

        # ── Pattern 3: Trend continuation after strong move ───────
        if abs(body) > atr_est * 1.2:
            direction = 1 if body > 0 else -1
            if rng.random() < 0.80:
                move = direction * abs(rng.normal(atr_est * 0.8, atr_est * 0.2))
                close[i+1] = close[i] + move
                open_p[i+1] = close[i] + rng.normal(0, atr_est * 0.05)
                high_p[i+1] = max(close[i+1], open_p[i+1]) + abs(rng.normal(0, atr_est * 0.2))
                low_p[i+1] = min(close[i+1], open_p[i+1]) - abs(rng.normal(0, atr_est * 0.2))
            continue

        # ── Pattern 4: Squeeze breakout ───────────────────────────
        # Low volatility period followed by expansion
        if i >= 20:
            recent_vol = np.std(close[i-20:i+1])
            historical_vol = np.std(close[max(0,i-60):i+1])
            if recent_vol < historical_vol * 0.5 and rng.random() < 0.10:
                # Breakout!
                direction = 1 if rng.random() < 0.5 else -1
                for k in range(1, min(4, n - i)):
                    move = direction * abs(rng.normal(atr_est * 2.0, atr_est * 0.5))
                    close[i+k] = close[i+k-1] + move * (1 - k * 0.2)
                    open_p[i+k] = close[i+k-1]
                    high_p[i+k] = max(close[i+k], open_p[i+k]) + abs(rng.normal(0, atr_est * 0.3))
                    low_p[i+k] = min(close[i+k], open_p[i+k]) - abs(rng.normal(0, atr_est * 0.3))

    # Update DataFrame
    df['close'] = close
    df['open'] = open_p
    df['high'] = high_p
    df['low'] = low_p

    # Ensure OHLC consistency
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    return df


if __name__ == "__main__":
    print("Generating realistic synthetic data...")
    df = generate_realistic_data(n=5000)
    df.to_csv("hydra_realistic_data.csv", index=False)
    print(f"Saved {len(df)} candles to hydra_realistic_data.csv")

    print("\nGenerating pattern-enhanced data...")
    df2 = generate_with_patterns(n=5000)
    df2.to_csv("hydra_pattern_data.csv", index=False)
    print(f"Saved {len(df2)} candles to hydra_pattern_data.csv")

    # Quick stats
    returns = df2['close'].pct_change().dropna()
    print(f"\nStats: Mean={returns.mean():.6f} Std={returns.std():.6f} "
          f"Skew={returns.skew():.2f} Kurt={returns.kurtosis():.2f}")
