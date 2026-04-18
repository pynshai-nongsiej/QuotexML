import asyncio
from pyquotex.stable_api import Quotex

async def get_high_yield_otc():
    client = Quotex("johnrocknongsiej123@gmail.com", "DariDaling1@")
    await client.connect()
    
    payment_str = client.get_payment()
    print("Payment str type:", type(payment_str))
    
    # Process dictionary...
    high_yield = []
    if isinstance(payment_str, dict):
        for asset, data in payment_str.items():
            if "_otc" in asset.lower() and data.get("payment", 0) >= 80:
                high_yield.append((asset, data.get("payment")))
    elif isinstance(payment_str, list):
        for asset, data in payment_str: # Unsure format
            print(asset, "-", data)
    
    print("Found:", high_yield)
    await client.close()

asyncio.run(get_high_yield_otc())
