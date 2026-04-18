import yfinance as yf
import pandas as pd
import time
from datetime import datetime

def fetch_yfinance_data(output_file="quotex_history.csv", count=1400):
    print(f"Fetching real EUR/USD 1-minute data from Yahoo Finance...")
    
    # 1m data is available for the latest 7 days. We fetch 2 days to get enough.
    ticker = yf.Ticker("EURUSD=X")
    df = ticker.history(period="3d", interval="1m")
    
    if df.empty:
        print("Failed to fetch data from Yahoo Finance.")
        return
        
    print(f"Fetched {len(df)} candles. Formatting for HydraBacktester...")
    
    # Format to match quotex format: time, open, high, low, close, volume
    df = df.reset_index()
    
    # Rename columns
    df = df.rename(columns={
        "Datetime": "time",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    })
    
    # Convert 'time' to unix timestamp (seconds since epoch)
    df['time'] = df['time'].apply(lambda x: int(x.timestamp()))
    
    # Keep only required columns
    df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
    
    # Sort chronologically
    df = df.sort_values('time').reset_index(drop=True)
    
    if len(df) > count:
        df = df.tail(count).reset_index(drop=True)
        
    df.to_csv(output_file, index=False)
    print(f"✅ Success! Saved {len(df)} real historical 1-minute candles to {output_file}")

if __name__ == "__main__":
    fetch_yfinance_data()
