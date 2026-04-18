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
    choice = input("\nSelect Asset Number (or 'MULTI' for 10x High-Yield OTC Mode) -> ").strip()
    
    if choice.upper() == "MULTI":
        console.print("[yellow]Scanning live payouts for Top 10 OTC markets...[/yellow]")
        payment_data = client.get_payment()
        otc_pairs = []
        for code, data in payment_data.items():
            if "_otc" in code.lower():
                # We prioritize 1M/short timeframe payment rate (usually turbo_payment)
                pay = data.get("payment", 0)
                if pay >= 70:
                    otc_pairs.append((code, pay))
                    
        otc_pairs.sort(key=lambda x: x[1], reverse=True)
        # Take the top yielding ones, grab 10.
        import random
        top_tier = otc_pairs[:15]
        selected = random.sample(top_tier, min(10, len(top_tier)))
        assets_list = [item[0] for item in selected]
        console.print(f"[bold green]Engaged {len(assets_list)} Concurrent Pairs: {', '.join(assets_list)}[/bold green]")
        return assets_list
        
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
        "[bold cyan]QUOTEX ROBOT v4.0[/bold cyan]\n"
        "[dim]HydraNet Neural Ensemble Engine + Legacy Strategy[/dim]", 
        border_style="blue"
    ))
    
    console.print("\n[1] [bold blue]Backtest Legacy Strategy[/bold blue] (Original V5.2)")
    console.print("[2] [bold magenta]Backtest HydraNet Strategy[/bold magenta] (Neural Ensemble)")
    console.print("[3] [bold green]Start Live/Practice Trader[/bold green] (Real-time)")
    mode = input("\nSelect Mode -> ")
    
    if mode == "1":
        console.print(Panel("Rule-Based Backtest", border_style="blue"))
        
        console.print("\n[bold]Select Strategy for Backtest:[/bold]")
        console.print("  [1] Legacy V5.2")
        console.print("  [2] ZigZag + DeMarker Scalper")
        console.print("  [3] EMA 3/7 + AO Breakout")
        strat_choice = input("Strategy -> ").strip()
        
        try:
            cand_input = input("Enter number of candles to simulate (default 5000) or 'csv' to load quotex_history.csv: ").strip()
            if cand_input.lower() == 'csv':
                n_candles = 'csv'
            else:
                n_candles = int(cand_input) if cand_input else 5000
        except ValueError:
            n_candles = 5000
            
        if n_candles == 'csv':
            console.print("[yellow]Loading live market data from quotex_history.csv...[/yellow]")
            df = pd.read_csv("quotex_history.csv")
        else:
            console.print(f"[yellow]Generating {n_candles} synthetic candles (ZigZag Mode)...[/yellow]")
            df = generate_sample_data(n=n_candles, mode="zigzag_wave")

        # Initialize Engines
        if strat_choice == "3":
            from ao_ema_strategy import AoEmaStrategy
            tester = Backtester(df, strategy=AoEmaStrategy())
        elif strat_choice == "2":
            from zigzag_strategy import ZigZagStrategy
            tester = Backtester(df, strategy=ZigZagStrategy())
        else:
            tester = Backtester(df)

        # Run Backtest
        console.print("\n[bold blue]Running Optimization Simulation...[/bold blue]")
        results = tester.run(start_idx=100, step=1)
        tester.stats(results)
        
        os.makedirs("logs", exist_ok=True)
        results.to_csv("logs/backtest_results.csv", index=False)
        console.print("\n[green]Simulation complete. Results saved to logs/backtest_results.csv[/green]")

    elif mode == "2":
        # HydraNet Backtest
        console.print(Panel("[bold magenta]HydraNet Backtester[/bold magenta]", border_style="magenta"))
        console.print("[dim]Launching walk-forward neural ensemble backtester...[/dim]\n")
        
        # Import and run the hydra backtester
        from hydra_backtester import main as hydra_main
        hydra_main()
        
    elif mode == "3":
        console.print(Panel("[bold green]Live Deployment Setup[/bold green]", border_style="green"))
        
        # Strategy selection
        console.print("\n[bold]Select Strategy:[/bold]")
        console.print("  [1] [magenta]HydraNet Ensemble[/magenta] (Neural AI — Recommended)")
        console.print("  [2] [blue]Legacy V5.2[/blue] (Rule-based CALL-only)")
        console.print("  [3] [cyan]ZigZag + DeMarker Scalper[/cyan] (1-Minute Reversals)")
        console.print("  [4] [yellow]EMA 3/7 + AO Breakout[/yellow] (1m Chart -> 5m Trade)")
        strat_choice = input("Strategy -> ").strip()
        
        if strat_choice == "4":
            strategy_type = "ao_ema"
        elif strat_choice == "3":
            strategy_type = "zigzag"
        elif strat_choice == "2":
            strategy_type = "legacy"
        else:
            strategy_type = "hydra"
        
        email = "johnrocknongsiej123@gmail.com"
        password = "DariDaling1@"
        
        # Connect to fetch real-time asset data
        temp_client = Quotex(email=email, password=password)
        check, reason = await temp_client.connect()
        if not check:
            console.print(f"[red]Authentication failed: {reason}[/red]")
            return
        
        asset_selection = await get_asset_choice(temp_client)
        await temp_client.close()
        
        # Resolve assets list
        assets_list = asset_selection if isinstance(asset_selection, list) else [asset_selection]
        console.print(f"[bold cyan]Target Assets: {', '.join(assets_list)}[/bold cyan]")
        strat_name = "HydraNet Ensemble" if strategy_type == 'hydra' else (
            "EMA + AO (1m)" if strategy_type == "ao_ema" else (
                "ZigZag + DeMarker" if strategy_type == "zigzag" else "Legacy V5.2"
            )
        )
        console.print(f"[bold magenta]Strategy: {strat_name}[/bold magenta]")
        
        # Timeframe Selection
        if strategy_type == "ao_ema":
            timeframe = 60
            console.print("[green]Auto-Locked Chart Timeframe: M1/60s (Trade Duration: 5m/300s)[/green]")
        else:
            console.print("\n[bold]Select Timeframe (M1/60s or 15s are RECOMMENDED)[/bold]")
            tf_input = input("Enter Timeframe [15s, 30s, 60s, 120s] (default 60) -> ").strip()
            
            timeframe_map = {15: 15, 30: 30, 60: 60, 1: 60, 2: 120, 5: 300}
            timeframe = 60 # Default
            val = int(tf_input) if tf_input.isdigit() else 60
            if val in timeframe_map:
                timeframe = timeframe_map[val]
            elif val == 15: # Safety catch
                timeframe = 15
            
            console.print(f"[green]Applying Strategy Timeframe: {timeframe}s[/green]")

        account_type = input("\n[1] PRACTICE (Demo), [2] REAL -> ")
        mode_str = "REAL" if account_type == "2" else "PRACTICE"
        
        console.print("[bold yellow]Launching Professional Dashboard...[/bold yellow]")
        await asyncio.sleep(1) 
        
        trader = LiveTrader(
            email, password, assets=assets_list, amount=1,
            timeframe=timeframe, mode=mode_str,
            strategy_type=strategy_type,
        )
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
