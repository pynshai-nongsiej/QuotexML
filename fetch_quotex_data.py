import asyncio
import os
import time
import pandas as pd
import sys
import json

sys.path.append(os.path.join(os.getcwd(), "pyquotex"))
from pyquotex.stable_api import Quotex

async def fetch_history_data(asset="EURUSD_otc", period=60, count=1400, output_file="quotex_history.csv"):
    email = "johnrocknongsiej123@gmail.com"
    password = "DariDaling1@"
    
    print(f"Connecting to Quotex...")
    client = Quotex(email=email, password=password, lang="en")
    client.set_account_mode("PRACTICE")
    
    check, reason = await client.connect()
    if not check:
        print(f"Connection failed: {reason}")
        return False
        
    print(f"Connected. Fetching the max historical window for {asset} (period {period}s)...")
    
    # Needs to match current asset for the websocket handler to populate candles_data
    client.api.current_asset = asset
    
    # Quotex WebSockets restrict direct historical pagination and max out at ~200 candles per handshake.
    candles = await client.get_candles(asset, time.time(), 250, period)
    
    if not candles:
        print("Failed to fetch candles or empty result.")
        await client.close()
        return False
        
    df = pd.DataFrame(candles)
    
    # Parse correct format relying on PyQuotex parsing
    if 'time' not in df.columns:
        # It's returning list of lists
        parsed = []
        for c in candles:
             if isinstance(c, list) and len(c) >= 5:
                 parsed.append({"time": c[0], "open": c[1], "close": c[2], "high": c[3], "low": c[4]})
        df = pd.DataFrame(parsed)

    df = df.drop_duplicates(subset=['time'])
    
    if df.empty:
        print("Empty dataframe after parsing.")
        await client.close()
        return False
        
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].astype(float)
        
    df['volume'] = 100
    df = df.sort_values('time').reset_index(drop=True)
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file} (Total: {len(df)} recent candles)")
    print(f"Note: Quotex API caps historical requests at ~200 candles. You can still test the backtester with this 3.5 hour window.")
    
    await client.close()
    return True

if __name__ == "__main__":
    asyncio.run(fetch_history_data("EURUSD_otc", 60, 1400, "quotex_history.csv"))
