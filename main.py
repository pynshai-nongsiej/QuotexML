import pandas as pd
import os
import asyncio
from data_loader import DataLoader
from strategy_engine import StrategyEngine
from backtester import Backtester
from data_generator import generate_sample_data
from live_trader import LiveTrader
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from pyquotex.stable_api import Quotex
from rich import box

console = Console()

async def get_asset_choice(client):
    """Fetch and select a single asset from Quotex."""
    console.print("[bold yellow]Fetching available assets...[/bold yellow]")
    assets = client.get_all_asset_name()
    if not assets:
        console.print("[red]Could not fetch assets. Using default: EURUSD[/red]")
        return "EURUSD"
    
    table = Table(title="Available Quotex Assets", show_header=True, header_style="bold green", box=box.ROUNDED)
    table.add_column("#", style="dim", width=4)
    table.add_column("Asset Code")
    table.add_column("Name")
    
    for idx, (code, name) in enumerate(assets):
        table.add_row(str(idx + 1), code, name)
    
    console.print(table)
    choice = input("\nSelect Asset Number -> ").strip()
    
    try:
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(assets):
                return assets[idx][0]
    except Exception:
        pass
    
    console.print("[yellow]Invalid choice. Defaulting to EURUSD.[/yellow]")
    return "EURUSD"

async def main():
    console.print(Panel.fit(
        "[bold cyan]QUOTEX ROBOT v3.6[/bold cyan]\n[dim]High-Win Rate Dynamic Breakout Engine[/dim]", 
        border_style="blue"
    ))
    
    console.print("\n[1] [bold blue]Backtest Strategy[/bold blue] (Simulate Patterns)")
    console.print("[2] [bold green]Start Live/Practice Trader[/bold green] (Real-time)")
    mode = input("\nSelect Mode -> ")
    
    if mode == "1":
        console.print(Panel("Backtest Data Configuration", border_style="blue"))
        
        try:
            cand_input = input("Enter number of candles to simulate (default 5000): ").strip()
            n_candles = int(cand_input) if cand_input else 5000
        except ValueError:
            n_candles = 5000
            
        console.print(f"[yellow]Generating {n_candles} synthetic candles (ZigZag Mode)...[/yellow]")
        df = generate_sample_data(n=n_candles, mode="zigzag_wave")

        # Initialize Engines
        tester = Backtester(df)

        # Run Backtest
        console.print("\n[bold blue]Running Optimization Simulation...[/bold blue]")
        results = tester.run(start_idx=100, step=1)
        tester.stats(results)
        
        os.makedirs("logs", exist_ok=True)
        results.to_csv("logs/backtest_results.csv", index=False)
        console.print("\n[green]Simulation complete. Results saved to logs/backtest_results.csv[/green]")
        
    elif mode == "2":
        console.print(Panel("[bold green]Live Deployment Setup[/bold green]", border_style="green"))
        
        email = "johnrocknongsiej123@gmail.com"
        password = "DariDaling1@"
        
        # Connect to fetch real-time asset data
        temp_client = Quotex(email=email, password=password)
        check, reason = await temp_client.connect()
        if not check:
            console.print(f"[red]Authentication failed: {reason}[/red]")
            return
        
        asset = await get_asset_choice(temp_client)
        await temp_client.close()
        
        console.print(f"[bold cyan]Target Asset: {asset}[/bold cyan]")
        
        # Timeframe Selection
        console.print("\n[bold]Select Timeframe (M1/1m is RECOMMENDED)[/bold]")
        tf_input = input("Enter Minutes (default 1) -> ").strip()
        
        timeframe_map = {1: 60, 2: 120, 3: 180, 5: 300, 10: 600, 15: 900}
        timeframe = 60 # Default
        if tf_input.isdigit() and int(tf_input) in timeframe_map:
            timeframe = timeframe_map[int(tf_input)]
        
        console.print(f"[green]Applying Strategy Timeframe: {timeframe}s[/green]")

        account_type = input("\n[1] PRACTICE (Demo), [2] REAL -> ")
        mode_str = "REAL" if account_type == "2" else "PRACTICE"
        
        console.print("[bold yellow]Launching Professional Dashboard...[/bold yellow]")
        await asyncio.sleep(1) 
        
        trader = LiveTrader(email, password, assets=[asset], amount=1, timeframe=timeframe, mode=mode_str)
        try:
            await trader.start()
        except KeyboardInterrupt:
            console.print("\n[bold yellow]System standby. Connection closed.[/bold yellow]")
            await trader.stop()
        except Exception as e:
            console.print(f"[bold red]Critical Error: {e}[/bold red]")
            await trader.stop()
    else:
        console.print("[red]Invalid selection.[/red]")

if __name__ == "__main__":
    asyncio.run(main())
