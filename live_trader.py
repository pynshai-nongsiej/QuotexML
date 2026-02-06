import asyncio
import sys
import os
import pandas as pd
from datetime import datetime
import time

# Add the pyquotex root directory to the path
sys.path.append(os.path.join(os.getcwd(), "pyquotex"))

from pyquotex.stable_api import Quotex
from strategy_engine import StrategyEngine

from rich.layout import Layout
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.console import Console
from rich import box
from rich.align import Align
from rich.text import Text

class LiveTrader:
    def __init__(self, email, password, assets=["EURUSD"], amount=1, timeframe=60, mode="PRACTICE"):
        self.email = email
        self.password = password
        self.assets = assets if isinstance(assets, list) else [assets]
        
        # Isolated Clients: One per asset to prevent data collision
        self.clients = {}
        for asset in self.assets:
            self.clients[asset] = Quotex(
                email=email,
                password=password,
                lang="en",
                asset_default=asset,
                period_default=timeframe
            )
            
        self.strategy = StrategyEngine()
        self.amount = amount
        self.timeframe = timeframe # Default is 60s (1 Minute) based on Backtest
        self.mode = mode 
        self.running = False
        self.log_file = "logs/learning_data.csv"
        self.console = Console()
        
        # Shared State for Dashboard
        self.market_state = {asset: {"price": 0.0, "rsi": 0.0, "adx": 0.0, "score": 0.0, "status": "Initializing...", "action": "-", "profit": 0.0} for asset in self.assets}
        self.global_balance = 0.0
        self.total_profit = 0.0
        
        os.makedirs("logs", exist_ok=True)

    def generate_dashboard(self):
        """Create a professional dashboard layout using Rich."""
        layout = Layout()
        layout.split(
            Layout(name="header", size=4),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        
        # --- HEADER ---
        header_text = f"QUOTEX ROBOT v3.0 | {self.mode}"
        current_time = datetime.now().strftime("%H:%M:%S")
        
        header_table = Table.grid(expand=True)
        header_table.add_column("Left", justify="left")
        header_table.add_column("Center", justify="center")
        header_table.add_column("Right", justify="right")
        
        header_table.add_row(
            f"[bold cyan]{header_text}[/bold cyan]", 
            f"[bold white]Time: {current_time}[/bold white]",
            f"[bold green]Balance: ${self.global_balance:.2f}[/bold green]"
        )
        
        layout["header"].update(Panel(header_table, style="blue"))
        
        # --- MAIN TABLE ---
        table = Table(box=box.DOUBLE_EDGE, expand=True, header_style="bold white on blue")
        table.add_column("Asset", style="cyan", no_wrap=True)
        table.add_column("Price", style="white")
        table.add_column("RSI (14)", justify="center")
        table.add_column("Trend (ADX)", justify="center")
        table.add_column("Zone Status", justify="center")
        table.add_column("Confluence", justify="right")
        table.add_column("Action", justify="center")
        table.add_column("Live Status", style="dim", width=35)
        table.add_column("Session P/L", justify="right")

        for asset in self.assets:
            data = self.market_state[asset]
            
            # Smart Styling
            rsi_val = data.get('rsi', 50)
            rsi_style = "bold green" if rsi_val > 65 else "bold red" if rsi_val < 35 else "white"
            
            zone = data.get('zone', 'Mid')
            # Inversion Logic High WR: High Zone = Breakout UP (Green), Low Zone = Breakout DOWN (Red)
            zone_style = "bold green" if zone == "High" else "bold red" if zone == "Low" else "dim"
            
            score_style = "bold yellow" if data['score'] >= 0.8 else "green" if data['score'] >= 0.6 else "dim"
            
            action_bg = ""
            if data['action'] == "UP": action_bg = "on green"
            elif data['action'] == "DOWN": action_bg = "on red"
            
            # Trend Strength (ADX)
            adx_val = data.get('adx', 0)
            adx_style = "bold yellow" if adx_val > 25 else "dim"
            
            pnl_color = "green" if data['profit'] > 0 else "red" if data['profit'] < 0 else "white"
            
            table.add_row(
                f"[bold]{asset}[/bold]",
                f"{data['price']:.5f}",
                f"[{rsi_style}]{rsi_val:.1f}[/{rsi_style}]",
                f"[{adx_style}]{adx_val:.1f}[/{adx_style}]",
                f"[{zone_style}]{zone}[/{zone_style}]",
                f"[{score_style}]{data['score']:.2f}[/{score_style}]",
                f"[bold white {action_bg}] {data['action']} [/bold white {action_bg}]",
                data['status'],
                f"[{pnl_color}]${data['profit']:.2f}[/{pnl_color}]"
            )
            
        layout["main"].update(Panel(table, title="[bold]Market Monitor[/bold]", border_style="white"))
        
        # --- FOOTER ---
        footer_text = f"Strategy: Dynamic Breakout (BB 2.5 + RSI) | Timeframe: {self.timeframe}s (M1) | Target WR: 85%"
        layout["footer"].update(Align.center(f"[dim]{footer_text}[/dim]"))
        
        return layout
        
    async def start(self):
        # Set account mode and connect all isolated clients
        connect_tasks = []
        for asset, client in self.clients.items():
            client.set_account_mode(self.mode)
            connect_tasks.append(client.connect())
            
        print("Connecting isolated asset channels...")
        results = await asyncio.gather(*connect_tasks)
        
        for (check, reason), asset in zip(results, self.assets):
             if not check:
                 print(f"Connection failed for {asset}: {reason}")
                 return
            
        # Get initial balance from the first client (Balance is account-wide)
        primary_client = self.clients[self.assets[0]]
        self.global_balance = await primary_client.get_balance()
        self.running = True
        
        # Start Dashboard Loop + Trading Loops
        with Live(self.generate_dashboard(), refresh_per_second=4, screen=True) as live:
            async def update_ui():
                while self.running:
                    # Periodic balance check using primary client
                    self.global_balance = await primary_client.get_balance() 
                    live.update(self.generate_dashboard())
                    await asyncio.sleep(1)

            tasks = [self.trading_loop(asset) for asset in self.assets]
            tasks.append(update_ui())
            
            await asyncio.gather(*tasks)

    async def stop(self):
        self.running = False
        close_tasks = [client.close() for client in self.clients.values()]
        await asyncio.gather(*close_tasks)

    def log_trade(self, features, outcome, reason, score):
        result = 1 if outcome == "WIN" else 0
        data = {
            "timestamp": time.time(),
            "reason": reason,
            "score": score,
            "result": result
        }
        if features is not None:
             for i, val in enumerate(features):
                data[f"f_{i}"] = val
                
        df = pd.DataFrame([data])
        df.to_csv(self.log_file, mode='a', index=False, header=not os.path.isfile(self.log_file))

    def calculate_trade_amount(self, balance):
        risk_percentage = 0.02
        target_amount = balance * risk_percentage
        return max(1, int(target_amount))

    async def trading_loop(self, asset):
        client = self.clients[asset]
        while self.running:
            try:
                # --- CLOCK SYNC ---
                now_ts = time.time()
                timeframe_sec = self.timeframe
                
                # Seconds elapsed in current candle
                elapsed = now_ts % timeframe_sec
                
                wait_time = timeframe_sec - elapsed - 0.5
                
                if wait_time > 2:
                    self.market_state[asset]["status"] = f"Syncing... Wait {int(wait_time)}s"
                    await asyncio.sleep(wait_time)
                
                self.market_state[asset]["status"] = "Fetching Data..."
                
                # Fetch more history for accurate indicators
                history_size = self.timeframe * 200 
                # Use isolated client for this specific asset
                candles = await client.get_candles(asset, time.time(), history_size, self.timeframe)
                
                if not candles or len(candles) < 50:
                    self.market_state[asset]["status"] = "Waiting for Data..."
                    await asyncio.sleep(1) # Short retry
                    continue

                df = pd.DataFrame(candles)
                decision = self.strategy.execute(df)
                
                # Update State
                metrics = decision.get('metrics', {}) # safely get dict
                
                # Determine Zone for UI
                price = df['close'].iloc[-1]
                zone = "Mid"
                if price >= metrics.get('dyn_res', 999999): zone = "High"
                elif price <= metrics.get('dyn_sup', 0): zone = "Low"
                
                self.market_state[asset].update({
                    "price": price,
                    "rsi": metrics.get('rsi', 50),
                    "adx": metrics.get('adx', 0),
                    "zone": zone,
                    "score": decision['confluence_score'],
                    "action": decision['decision'],
                    "status": f"{decision['reason']}"
                })

                # Execute at :00 ideally
                if decision['decision'] in ["UP", "DOWN"]: 
                     
                    target = self.calculate_trade_amount(self.global_balance)
                    direction = "call" if decision['decision'] == "UP" else "put"
                    
                    self.market_state[asset]["status"] = f"EXECUTING {direction.upper()} (${target})..."
                    status, buy_info = await client.buy(target, asset, direction, self.timeframe)
                    
                    if status:
                        self.market_state[asset]["status"] = "Trade Active..."
                        win = await client.check_win(buy_info.get('id'))
                        
                        # Accuracy Fix: Refresh Balance using current client
                        self.global_balance = await client.get_balance()
                        
                        # Robust Result Interpretation
                        profit_change = 0
                        outcome = "DRAW"
                        
                        if win > target:
                            # Likely Returns Payout (e.g., 1.85 for 1.0 bet)
                            outcome = "WIN"
                            profit_change = win - target
                        elif win == target:
                            # Returns Investment -> Draw
                            outcome = "DRAW"
                            profit_change = 0
                        elif win > 0:
                            # Likely Returns Net Profit (e.g., 0.85 for 1.0 bet)
                            outcome = "WIN"
                            profit_change = win
                        else:
                            # Returns 0 -> Loss
                            outcome = "LOSS"
                            profit_change = -target

                        self.market_state[asset]["profit"] += profit_change
                        self.market_state[asset]["status"] = outcome
                        self.log_trade(decision['features'], outcome, decision['reason'], decision['confluence_score'])
                    else:
                        self.market_state[asset]["status"] = "Execution Failed"
                
                # Wait for next minute
                await asyncio.sleep(2) 

            except Exception as e:
                self.market_state[asset]["status"] = f"Error: {str(e)[:20]}"
                await asyncio.sleep(5)

if __name__ == "__main__":
    # For standalone testing
    import json
    
    # Load credentials if possible
    email = "johnrocknongsiej123@gmail.com"
    password = "DariDaling1@"
    
    assets_test = ["EURUSD", "GBPUSD"]
    trader = LiveTrader(email, password, assets=assets_test, amount=1, mode="PRACTICE")
    
    try:
        asyncio.run(trader.start())
    except KeyboardInterrupt:
        print("Stopping...")
        asyncio.run(trader.stop())
