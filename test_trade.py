"""
Test Script: Quotex Trade Placement
────────────────────────────────────
Tests connection, time sync, and places a $1 demo trade
to verify that trade execution works correctly.

Usage:
    python test_trade.py
"""

import asyncio
import sys
import os
import time
from datetime import datetime, timezone

sys.path.append(os.path.join(os.getcwd(), "pyquotex"))

from pyquotex.stable_api import Quotex
from pyquotex.expiration import get_expiration_time_quotex

os.makedirs("logs", exist_ok=True)


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


async def main():
    email = "johnrocknongsiej123@gmail.com"
    password = "DariDaling1@"

    print("=" * 60)
    print("  QUOTEX TRADE PLACEMENT TEST")
    print("=" * 60)

    # ── Step 1: Connect ──────────────────────────────────────────
    log("Initializing Quotex client...")
    client = Quotex(
        email=email,
        password=password,
        lang="en",
    )
    client.set_account_mode("PRACTICE")

    log("Connecting...")
    check, reason = await client.connect()
    if not check:
        log(f"❌ Connection FAILED: {reason}")
        return
    log(f"✅ Connected: {reason}")

    await asyncio.sleep(1)

    # ── Step 2: Check Balance ────────────────────────────────────
    balance = await client.get_balance()
    log(f"💰 Practice Balance: ${balance:.2f}")

    # ── Step 3: Server Time Sync ─────────────────────────────────
    log("Checking time sync...")
    try:
        profile = await client.get_profile()
        offset = profile.offset
        local_epoch = int(time.time())
        # Correct UTC calculation (offset is local→UTC shift in seconds)
        utc_epoch = int(datetime.now(timezone.utc).timestamp())
        log(f"  Local epoch:   {local_epoch}")
        log(f"  UTC epoch:     {utc_epoch}")
        log(f"  Profile offset: {offset}s ({offset/3600:.1f}h)")
        log(f"  Note: pyquotex get_server_timer has a known double-subtraction bug")
        log(f"  The actual server likely uses UTC — trades use time.time() directly ✅")
    except Exception as e:
        log(f"  ⚠️ Time sync check error: {e}")

    # ── Step 4: Test Expiration Calculation ───────────────────────
    log("Testing expiration time calculation (60s duration)...")
    now_ts = int(time.time())
    exp = get_expiration_time_quotex(now_ts, 60)
    secs_until = exp - now_ts
    log(f"  Now:        {datetime.fromtimestamp(now_ts).strftime('%H:%M:%S')}")
    log(f"  Expiration: {datetime.fromtimestamp(exp).strftime('%H:%M:%S')} ({secs_until}s from now)")
    if 30 <= secs_until <= 130:
        log(f"  ✅ Expiration looks reasonable")
    else:
        log(f"  ⚠️ Expiration seems off ({secs_until}s)")

    # ── Step 5: Choose Asset ─────────────────────────────────────
    # Use OTC asset directly — they're available 24/7
    asset = "EURUSD_otc"
    log(f"Using OTC asset: {asset} (available 24/7)")

    try:
        asset_name, asset_data = await client.get_available_asset(asset, force_open=True)
        if asset_data and len(asset_data) >= 3 and asset_data[2]:
            log(f"  ✅ {asset_name} is OPEN")
        else:
            log(f"  ⚠️ {asset_name} may be closed, trying {asset} directly...")
            asset_name = asset
    except Exception as e:
        log(f"  ⚠️ Asset check failed: {e}")
        asset_name = asset

    # ── Step 6: Place Test Trade ─────────────────────────────────
    amount = 1
    direction = "call"
    duration = 60

    log("")
    log("─" * 50)
    log("  PLACING $1 DEMO TRADE")
    log(f"  Asset:     {asset_name}")
    log(f"  Direction: {direction.upper()}")
    log(f"  Duration:  {duration}s")
    log(f"  Mode:      TIMER")
    log("─" * 50)

    try:
        log("Sending buy order...")
        t0 = time.time()

        # Wrap buy in a timeout to avoid infinite hangs
        try:
            status, buy_info = await asyncio.wait_for(
                client.buy(amount, asset_name, direction, duration, time_mode="TIMER"),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            elapsed = time.time() - t0
            log(f"  ⏱️  Buy timed out after {elapsed:.1f}s")
            log(f"  This usually means the server rejected the order silently.")
            log(f"  Check logs/debug_live.log for the raw payload.")

            # Check if there's a websocket error
            from pyquotex import global_value
            if global_value.check_websocket_if_error:
                log(f"  ❌ Websocket error: {global_value.websocket_error_reason}")
            
            log(f"\n  Trying fallback: buy with shorter timeout and TIME mode...")
            try:
                status, buy_info = await asyncio.wait_for(
                    client.buy(amount, asset_name, direction, duration, time_mode="TIME"),
                    timeout=15.0
                )
                log(f"  TIME mode result: status={status}, info={buy_info}")
            except asyncio.TimeoutError:
                log(f"  ❌ TIME mode also timed out. Server may be rejecting all orders.")
            
            await client.close()
            return

        elapsed = time.time() - t0
        log(f"  Buy returned in {elapsed:.1f}s")
        log(f"  Status: {status}")
        log(f"  Info:   {buy_info}")

        if status and isinstance(buy_info, dict):
            order_id = buy_info.get("id", "N/A")
            close_ts = buy_info.get("closeTimestamp", 0)
            close_time = datetime.fromtimestamp(close_ts).strftime('%H:%M:%S') if close_ts else "N/A"

            log(f"  ✅ TRADE PLACED!")
            log(f"  Order ID:  {order_id}")
            log(f"  Closes at: {close_time}")

            log(f"  Waiting for result (up to {duration + 10}s)...")
            try:
                win_amount = await asyncio.wait_for(
                    client.check_win(order_id),
                    timeout=duration + 30
                )
                profit = win_amount - amount
                emoji = "🎉 WIN" if profit > 0 else ("💔 LOSS" if profit < 0 else "🤝 DRAW")
                log(f"  {emoji}! P/L: ${profit:+.2f}")
            except asyncio.TimeoutError:
                log(f"  ⏱️  Result check timed out (trade may still be pending)")

            new_balance = await client.get_balance()
            log(f"  💰 New Balance: ${new_balance:.2f}")
        else:
            log(f"  ❌ TRADE FAILED!")
            log(f"  Reason: {buy_info}")

    except Exception as e:
        import traceback
        log(f"  ❌ EXCEPTION: {e}")
        traceback.print_exc()

    # ── Cleanup ──────────────────────────────────────────────────
    log("")
    log("Closing connection...")
    await client.close()
    log("✅ Test complete!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[Interrupted]")
