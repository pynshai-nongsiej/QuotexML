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
from rich.columns import Columns

class LiveTrader:
    def __init__(self, email, password, assets=["EURUSD"], amount=1, timeframe=60, mode="PRACTICE"):
        self.email = email
        self.password = password
        self.asset = assets[0] if isinstance(assets, list) else assets
        
        self.client = Quotex(
            email=email,
            password=password,
            lang="en",
            asset_default=self.asset,
            period_default=timeframe
        )
            
        self.strategy = StrategyEngine()
        self.amount = amount
        self.timeframe = timeframe 
        self.mode = mode 
        self.running = False
        self.is_connected = False
        self.reconnect_attempts = 0
        self.last_reconnect_time = "N/A"
        
        self.log_file = "logs/learning_data.csv"
        self.debug_file = "logs/debug_live.log"
        self.console = Console()
        
        # Shared State 
        self.market_state = {self.asset: {
            "price": 0.0, "rsi": 50.0, "ema50": 0.0, "bb_up": 0.0, "bb_low": 0.0,
            "adx": 0.0, "status": "Initializing...", "pattern": "-", "action": "-", "profit": 0.0
        }}
        
        # Session Stats
        self.session_stats = {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "pnl": 0.0
        }
        self.trade_history = [] 
        self.last_trade_time = 0
        self.trade_lock = asyncio.Lock()
        
        self.global_balance = 0.0
        
        os.makedirs("logs", exist_ok=True)
        with open(self.debug_file, "w") as f:
            f.write(f"--- Debug Started {datetime.now()} ---\n")

    def generate_dashboard(self):
        layout = Layout()
        layout.split(
            Layout(name="header", size=4),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        layout["body"].split_row(
            Layout(name="market", ratio=3),
            Layout(name="history", ratio=2)
        )
        
        data = self.market_state[self.asset]
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # --- HEADER ---
        header_table = Table.grid(expand=True)
        
        # Connection Status with Pulse
        conn_color = "green" if self.is_connected else "red"
        conn_text = "ONLINE" if self.is_connected else "RECONNECTING"
        conn_indicator = f"[{conn_color}]● {conn_text}[/{conn_color}]"
        
        header_table.add_row(
            f"[bold cyan]QUOTEX CALL SNIPER v5.4 | {self.mode}[/bold cyan]", 
            f"[bold white]{current_time}[/bold white]",
            conn_indicator
        )
        
        stats = self.session_stats
        pnl_color = "green" if stats['pnl'] > 0 else "red" if stats['pnl'] < 0 else "white"
        stats_line = f"[white]Balance: [bold green]${self.global_balance:.2f}[/bold green] | Wins: [green]{stats['wins']}[/green] | Losses: [red]{stats['losses']}[/red] | P/L: [{pnl_color}]${stats['pnl']:.2f}[/{pnl_color}][/white]"
        
        header_panel_content = Table.grid(expand=True)
        header_panel_content.add_row(header_table)
        header_panel_content.add_row(Align.center(stats_line))
        
        layout["header"].update(Panel(header_panel_content, style="blue"))
        
        # --- MARKET TABLE ---
        market_table = Table(box=box.DOUBLE_EDGE, expand=True, header_style="bold white on blue")
        market_table.add_column("Indicator", style="cyan")
        market_table.add_column("Value", justify="right")
        market_table.add_column("Context", justify="center")

        px = data['price']
        ema = data['ema50']
        trend_label = "UPTREND" if px > ema else "DOWNTREND"
        trend_style = "bold green" if px > ema else "bold red"
        rsi_val = float(data.get('rsi', 50))
        rsi_style = "bold green" if rsi_val > 60 else "bold red" if rsi_val < 40 else "white"
        
        bb_up, bb_low = data['bb_up'], data['bb_low']
        zone = "MID"
        zone_style = "white"
        if px >= bb_up: zone, zone_style = "OVERBOUGHT", "bold red"
        elif px <= bb_low: zone, zone_style = "OVERSOLD", "bold green"
        
        market_table.add_row("Live Price", f"{px:.5f}", "[bold white]Active[/bold white]")
        market_table.add_row("Trend (EMA50)", f"{ema:.5f}", f"[{trend_style}]{trend_label}[/{trend_style}]")
        market_table.add_row("RSI (14)", f"{rsi_val:.1f}", f"[{rsi_style}]Momentum[/{rsi_style}]")
        market_table.add_row("BB Zone", zone, f"[{zone_style}]Targeting[/{zone_style}]")
        market_table.add_row("Candle Pattern", data['pattern'], "[dim]Recognition[/dim]")
        
        action_bg = "on green" if data['action'] == "UP" else ""
        market_table.add_row(
            "[bold yellow]SIGNAL INFO[/bold yellow]", 
            f"[bold white {action_bg}] {data['action']} [/bold white {action_bg}]", 
            f"[dim]{data['status']}[/dim]"
        )
        
        layout["market"].update(Panel(market_table, title=f"[bold]Market Data: {self.asset}[/bold]", border_style="white"))
        
        # --- HISTORY TABLE ---
        history_table = Table(box=box.SIMPLE, expand=True, header_style="bold magenta")
        history_table.add_column("Time", style="dim")
        history_table.add_column("Result", justify="center")
        history_table.add_column("P/L", justify="right")

        for trade in reversed(self.trade_history[-8:]):
            res_style = "bold green" if trade['result'] == "WIN" else "bold red" if trade['result'] == "LOSS" else "white"
            pnl_style = "green" if trade['profit'] > 0 else "red" if trade['profit'] < 0 else "white"
            history_table.add_row(
                trade['time'],
                f"[{res_style}]{trade['result']}[/{res_style}]",
                f"[{pnl_style}]${trade['profit']:.2f}[/{pnl_style}]"
            )
            
        layout["history"].update(Panel(history_table, title="[bold]Last 8 Trades[/bold]", border_style="magenta"))
        
        footer_text = f"15s CALL-ONLY Sniper | Last Reconnect: {self.last_reconnect_time} | Count: {self.reconnect_attempts}"
        layout["footer"].update(Align.center(f"[dim]{footer_text}[/dim]"))
        return layout

    async def check_connection(self):
        try:
            # Check basic socket state
            connected = await self.client.check_connect()
            if not connected:
                self.is_connected = False
                self.market_state[self.asset]["status"] = "Wait... Reconnecting"
                # Deep Reconnection (re-auth + ws restart)
                check, _ = await self.client.connect()
                if check:
                    self.is_connected = True
                    self.reconnect_attempts += 1
                    self.last_reconnect_time = datetime.now().strftime("%H:%M")
                    self.market_state[self.asset]["status"] = "Back Online!"
            else:
                self.is_connected = True
        except Exception as e:
            self.is_connected = False
            with open(self.debug_file, "a") as f: f.write(f"Conn Watchdog Error: {e}\n")

    async def start(self):
        self.client.set_account_mode(self.mode)
        try:
            check, _ = await self.client.connect()
            if check:
                self.is_connected = True
                self.global_balance = await self.client.get_balance()
        except Exception as e:
            with open(self.debug_file, "a") as f: f.write(f"Initial Connect Error: {e}\n")

        self.running = True
        with Live(self.generate_dashboard(), refresh_per_second=2, screen=True) as live:
            while self.running:
                try:
                    await self.check_connection()
                    
                    if self.is_connected:
                        await self.refresh_data(self.asset)
                        self.global_balance = await self.client.get_balance() 
                    
                    live.update(self.generate_dashboard())
                    
                    # Prevent memory creep in long sessions
                    if len(self.trade_history) > 100:
                        self.trade_history = self.trade_history[-50:]
                        
                except Exception as e:
                    with open(self.debug_file, "a") as f: f.write(f"Dashboard Loop Error: {e}\n")
                
                # High-frequency pulse
                sleep_time = 2 if self.timeframe <= 15 else 5
                await asyncio.sleep(sleep_time) 

    async def refresh_data(self, asset):
        try:
            history_size = self.timeframe * 100 
            candles = await self.client.get_candles(asset, time.time(), history_size, self.timeframe)
            
            if not candles or len(candles) < 30:
                self.market_state[asset]["status"] = "Syncing Data..."
                return

            df = pd.DataFrame(candles)
            df = df.sort_values('time').reset_index(drop=True)
            for col in ['open', 'high', 'low', 'close']: df[col] = df[col].astype(float)
            
            decision = self.strategy.execute(df)
            metrics = decision.get('metrics', {})
            last_price = float(df['close'].iloc[-1])
            
            self.market_state[asset].update({
                "price": last_price,
                "rsi": metrics.get('rsi', 50.0),
                "ema50": metrics.get('ema50', last_price),
                "bb_up": metrics.get('bb_up', last_price),
                "bb_low": metrics.get('bb_low', last_price),
                "adx": metrics.get('adx', 0.0),
                "pattern": metrics.get('pattern', "None"),
                "score": decision['confluence_score'],
                "action": decision['decision'],
                "status": decision['reason']
            })

            if decision['decision'] == "UP": 
                # Avoid redundant trades on the same candle
                if time.time() - self.last_trade_time > self.timeframe: 
                    asyncio.create_task(self.execute_trade(asset, decision))
        except Exception as e:
            # Trigger reconnect logic if data fetch fails repeatedly
            self.is_connected = False
            with open(self.debug_file, "a") as f: f.write(f"Data Sync Error: {e}\n")

    async def execute_trade(self, asset, decision):
        async with self.trade_lock:
            if time.time() - self.last_trade_time < self.timeframe: return
            self.last_trade_time = time.time()
            
            target = max(1, int(self.global_balance * 0.02))
            direction = "call"
            self.market_state[asset]["status"] = "SNIPING UP..."
            
            try:
                status, buy_info = await self.client.buy(target, asset, direction, self.timeframe)
                
                # Deep Log for Debugging
                with open(self.debug_file, "a") as f:
                    f.write(f"[{datetime.now().strftime('%H:%M:%S')}] TRADE ATTEMPT - Status: {status} | Info: {buy_info}\n")
                
                if status:
                    # Non-blocking result check
                    win_amount = await self.client.check_win(buy_info.get('id'))
                    outcome = "WIN" if win_amount > target else "LOSS" if win_amount < target else "DRAW"
                    profit = win_amount - target
                    
                    self.session_stats['total_trades'] += 1
                    if outcome == "WIN": self.session_stats['wins'] += 1
                    elif outcome == "LOSS": self.session_stats['losses'] += 1
                    else: self.session_stats['draws'] += 1
                    self.session_stats['pnl'] += profit
                    
                    self.trade_history.append({
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "asset": asset,
                        "result": outcome,
                        "profit": profit
                    })
                else:
                    self.market_state[asset]["status"] = "Trade Blocked"
            except Exception as e:
                import traceback
                with open(self.debug_file, "a") as f:
                    f.write(f"[{datetime.now().strftime('%H:%M:%S')}] EXECUTION ERROR:\n{traceback.format_exc()}\n")
                self.market_state[asset]["status"] = "Execution Error"

    async def stop(self):
        self.running = False
        await self.client.close()
