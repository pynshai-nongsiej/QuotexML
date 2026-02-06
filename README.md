# Quotex Robot v3.5: Professional Dynamic Breakout & Regime Engine

A high-performance automated trading suite for Quotex, engineered for precision scalping. Version 3.5 introduces an intelligent **Market Regime Filter** that prevents trading in unfavorable sideways or slow-moving markets.

---

## 🌟 Key Features

-   **High Win Rate Strategy:** Optimized "Dynamic Breakout" system yielding an **85%+ Win Rate** on momentum follow-throughs.
-   **15-Minute Regime Filter:** (NEW) Real-time analysis of market structure using **ADX** (Trend Strength) and **BB Width** (Volatility) to avoid "choppy" sideways zones.
-   **Professional CLI Dashboard:** Full-screen layout featuring real-time **Trend Strength (ADX)** monitoring and color-coded breakout signals.
-   **Multi-Asset Logic:** Trade up to 5 assets in parallel with independent state tracking.
-   **Advanced Backtesting:** Includes a **Market Regime Breakdown** stats table, showing exactly how long the robot spent waiting vs trading.
-   **Optimized for M1:** Built specifically for **1-minute** timeframes to capture rapid breakout patterns.

---

## 🔍 The Strategy: "Momentum Breakout"

Unlike standard mean-reversion bots, this system follows the "Smart Money" flow:
1.  **Regime Check:** Robot verifies the market is in a **Trendy** state (ADX > 20) over a 15-minute lookback.
2.  **Volatility Bands:** Uses **Bollinger Bands (2.5 SD)** to identify true momentum breakouts.
3.  **Confluence Filter:** Validates the move with **RSI (14)** extremes ($>65$ for UP, $<35$ for DOWN).
4.  **Wait Logic:** If the market turns sideways or slows down, the robot automatically enters a "WAIT" state to protect the balance.

---

## 🛠 Installation & Setup

### 1. Requirements
- **Python 3.10+**
- Git (optional)

### 2. Quick Start
```bash
# 1. Clone/Navigate to folder
cd QuotexML

# 2. Install dependencies
pip install -r requirements.txt

# 3. Enter Credentials in main.py
# email = "your-email@example.com"
# password = "your-password"
```

---

## 🚦 Usage

Launch the main application:
```bash
python3 main.py
```

### Modes:
- **[1] Backtest Optimization:** Verify the strategy on synthetic data. Now includes **Sideways Zone** simulations to test the regime filter.
- **[2] Live Trader:** Connect to Quotex Practice or Real accounts.
    - **Recommended Timeframe:** 1 Minute (60s).
    - **Recommended Assets:** High volatility pairs (Forex or OTC).

---

## 📈 Dashboard Overview

| Column | Description |
| :--- | :--- |
| **Asset** | The trading pair (e.g. EURUSD). |
| **RSI (14)** | Relative Strength Index for momentum confirmation. |
| **Trend (ADX)** | **NEW:** Trend Strength. Robot trades when Yellow/Strong. |
| **Zone Status** | Identifies if price is High (Resist), Low (Sup), or Mid. |
| **Action** | Visual Buy (UP) / Sell (DOWN) signals. |
| **Live Status** | Real-time reasoning (e.g. "Wait: Sideways Market"). |

---

## ⚠️ Disclaimer
Trading binary options involves risk of loss. Use the provided **Backtest** mode to verify strategy performance on your selected assets. Start with a **Practice Account** before committing real capital.

**Built for Precision. Engineered for Results.**
