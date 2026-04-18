import asyncio
import os
import time
import sys

sys.path.append(os.path.join(os.getcwd(), "pyquotex"))
from pyquotex.stable_api import Quotex

async def test():
    client = Quotex(email="johnrocknongsiej123@gmail.com", password="DariDaling1@", lang="en")
    client.set_account_mode("PRACTICE")
    await client.connect()
    
    # Needs codes_asset populated
    await client.get_all_assets()
    
    t1 = int(time.time())
    print("Fetching history line...")
    try:
        # get_history_line takes asset name, it does self.codes_asset[asset] internally
        c1 = await client.get_history_line("EURUSD_otc", t1, 1400)
        
        if c1 and "data" in c1:
            data = c1["data"]
            print(f"Success! Got {len(data)} lines.")
            print("Latest:", data[-1])
        else:
            print("Failed:", c1)
    except Exception as e:
        print("Error:", e)
        
    await client.close()

asyncio.run(test())
