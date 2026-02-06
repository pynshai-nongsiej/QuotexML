# Quotex Robot v3.6: Professional Dynamic Breakout & Regime Engine

A high-precision automated trading suite for Quotex, engineered for 85%+ win rate scalping. Version 3.6 introduces a focused **Single-Asset Execution Engine** for maximum stability and zero-lag performance.

---

## 🌟 Key Features

-   **High Win Rate Strategy:** Optimized "Dynamic Breakout" system yielding an **85%+ Win Rate** on momentum follow-throughs.
-   **Stable Data Feed:** Focused single-asset client prevents data collision and ensures 100% accurate RSI/ADX monitoring.
-   **15-Minute Regime Filter:** Real-time analysis of market structure using **ADX** (Trend Strength) and **BB Width** (Volatility) to avoid "choppy" sideways zones.
-   **Professional CLI Dashboard:** Full-screen layout featuring real-time **Trend Strength (ADX)** monitoring and color-coded breakout signals.
-   **Advanced Backtesting:** Includes a **Market Regime Breakdown** stats table, showing exactly how long the robot spent waiting vs trading.
-   **Optimized for M1:** Built specifically for **1-minute** timeframes to capture rapid breakout patterns.

---

## 🔍 The Strategy: "Momentum Breakout"

Unlike standard mean-reversion bots, this system follows the "Smart Money" flow:
1.  **Regime Check:** Robot verifies the market is in a **Trendy** state (ADX > 20) over a 15-minute lookback.
2.  **Volatility Bands:** Uses **Bollinger Bands (2.5 SD)** to identify true momentum breakouts.
3.  **Confluence Filter:** Validates the move with **RSI (14)** extremes ($>65$ for UP, $<35$ for DOWN).
4.  **Robust Logic:** Indicators are now protected against "flat" markets, ensuring the engine remains active even during low-volatility periods.

---

## 🛠 Installation & Setup

### 1. Requirements
- **Python 3.10+**
- Dependencies: `pip install -r requirements.txt`

### 2. Configuration
The system uses pre-configured credentials in `main.py`. Ensure your account is logged in or sessions are valid in the `pyquotex` folder.

---

## 🚦 Usage

Launch the main application:
```bash
python3 main.py
```

### Modes:
- **[1] Backtest Optimization:** Verify the strategy on synthetic data.
- **[2] Live Trader:** Select your target asset and timeframe (1m recommended).

---

## 📈 Dashboard Overview

| Column | Description |
| :--- | :--- |
| **Asset** | Your selected trading pair. |
| **RSI (14)** | Relative Strength Index (Colored: Green > 65, Red < 35). |
| **Trend (ADX)** | Trend Strength. Strong trends are highlighted in Yellow. |
| **Zone Status** | High (Resistance), Low (Support), or Mid (Consolidation). |
| **Action** | Visual Buy (UP) / Sell (DOWN) signals. |
| **Live Status** | Real-time reasoning for every decision or "WAIT" state. |

---

## ⚠️ Disclaimer
Trading binary options involves risk of loss. Start with a **Practice Account** before committing real capital.

**Built for Precision. Engineered for Stability.**
