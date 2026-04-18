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
from hydra_strategy import HydraStrategy
from zigzag_strategy import ZigZagStrategy
from ao_ema_strategy import AoEmaStrategy
import logging
from rich.layout import Layout
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.console import Console
from rich import box
from rich.align import Align
from rich.columns import Columns


class LiveTrader:
    def __init__(
        self,
        email,
        password,
        assets=["EURUSD"],
        amount=1,
        timeframe=60,
        mode="PRACTICE",
        strategy_type="hydra",  # "hydra" or "legacy"
    ):
        self.email = email
        self.password = password
        self.assets = assets if isinstance(assets, list) else [assets]
        # In single mode, this is just self.assets[0], needed for backwards-compat in some prints
        self.asset = self.assets[0]

        self.client = Quotex(
            email=email,
            password=password,
            lang="en",
            asset_default=self.asset,
            period_default=timeframe,
        )

        # Strategy selection
        self.strategy_type = strategy_type
        if strategy_type == "hydra":
            # Initial balance will be updated with real balance on connect
            self.strategy = HydraStrategy(initial_balance=1000.0)
            # Enable online learning for live trading adaptation
            self.strategy.online_learning_enabled = True
            # Unfreeze normalizer so model adapts to live market data
            if self.strategy.normalizer is not None:
                self.strategy.normalizer.unfreeze()
        elif strategy_type == "zigzag":
            self.strategy = ZigZagStrategy()
        elif strategy_type == "ao_ema":
            self.strategy = AoEmaStrategy()
        else:
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

        # Shared State (Dictionary of all active assets)
        self.market_state = {
            ast: {
                "price": 0.0,
                "rsi": 50.0,
                "ema50": 0.0,
                "bb_up": 0.0,
                "bb_low": 0.0,
                "adx": 0.0,
                "status": "Initializing...",
                "pattern": "-",
                "action": "-",
                "profit": 0.0,
                "regime": "-",
                "confidence": 0.0,
                "specialist_votes": {},
                "metrics": {}
            } for ast in self.assets
        }

        # Session Stats
        self.session_stats = {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "pnl": 0.0,
        }
        self.trade_history = []
        self.last_trade_time = {ast: 0 for ast in self.assets}
        self.trade_locks = {ast: asyncio.Lock() for ast in self.assets}

        self.global_balance = 0.0

        os.makedirs("logs", exist_ok=True)
        with open(self.debug_file, "w") as f:
            f.write(f"--- Debug Started {datetime.now()} ---\n")

    def generate_dashboard(self):
        if len(self.assets) > 1:
            return self.generate_multi_dashboard()
            
        layout = Layout()
        layout.split(
            Layout(name="header", size=5),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )

        layout["body"].split_row(
            Layout(name="market", ratio=3), Layout(name="history", ratio=2)
        )

        data = self.market_state[self.asset]
        current_time = datetime.now().strftime("%H:%M:%S")

        # --- HEADER ---
        header_table = Table.grid(expand=True)

        # Connection Status with Pulse
        conn_color = "green" if self.is_connected else "red"
        conn_text = "ONLINE" if self.is_connected else "RECONNECTING"
        conn_indicator = f"[{conn_color}]● {conn_text}[/{conn_color}]"

        strategy_label = "HYDRANET ENSEMBLE" if self.strategy_type == "hydra" else (
            "PURE PATTERN SEQUENCE" if self.strategy_type == "zigzag" else (
                "AO + EMA HIGHSPEED" if self.strategy_type == "ao_ema" else "LEGACY V5.4"
            )
        )
        header_table.add_row(
            f"[bold cyan]QUOTEX {strategy_label} | {self.mode}[/bold cyan]",
            f"[bold white]{current_time}[/bold white]",
            conn_indicator,
        )

        stats = self.session_stats
        pnl_color = (
            "green" if stats["pnl"] > 0 else "red" if stats["pnl"] < 0 else "white"
        )

        wr = (stats["wins"] / stats["total_trades"] * 100) if stats["total_trades"] > 0 else 0
        wr_color = "green" if wr >= 75 else "yellow" if wr >= 60 else "white"

        stats_line = (
            f"[white]Balance: [bold green]${self.global_balance:.2f}[/bold green] | "
            f"Wins: [green]{stats['wins']}[/green] | Losses: [red]{stats['losses']}[/red] | "
            f"WR: [{wr_color}]{wr:.1f}%[/{wr_color}] | "
            f"P/L: [{pnl_color}]${stats['pnl']:.2f}[/{pnl_color}][/white]"
        )

        # HydraNet-specific info line
        if self.strategy_type == "hydra":
            regime = data.get("regime", "-")
            conf = data.get("confidence", 0)
            regime_colors = {
                "TRENDING_UP": "green", "TRENDING_DOWN": "red",
                "RANGING": "yellow", "VOLATILE": "magenta"
            }
            r_color = regime_colors.get(regime, "white")
            hydra_line = (
                f"[dim]Regime: [{r_color}]{regime}[/{r_color}] | "
                f"Confidence: {conf:.1%} | "
                f"Online Buffer: {self.strategy.online_learner.size if hasattr(self.strategy, 'online_learner') else 0}[/dim]"
            )
        elif self.strategy_type == "zigzag":
            pattern = data.get("metrics", {}).get("pattern", "SCANNING")
            
            # Colorize the pattern sequence string dynamically (G=Green, R=Red, D=White)
            colored_pattern = ""
            for char in pattern:
                if char == 'G': colored_pattern += "[bold green]G[/bold green]"
                elif char == 'R': colored_pattern += "[bold red]R[/bold red]"
                elif char == 'D': colored_pattern += "[bold white]D[/bold white]"
                else: colored_pattern += char
                
            hydra_line = (
                f"[dim]Color Sequence Analysis: [/dim]{colored_pattern} | "
                f"[dim]Pure Pattern Scalper Live[/dim]"
            )
        elif self.strategy_type == "ao_ema":
            ao = data.get("metrics", {}).get("ao", 0.0)
            ema3 = data.get("metrics", {}).get("ema3", 0.0)
            ema7 = data.get("metrics", {}).get("ema7", 0.0)
            body = data.get("metrics", {}).get("body_pct", 0.0)
            
            ao_color = "green" if ao > 0 else ("red" if ao < 0 else "white")
            ema_str = f"[{'green' if ema3 > ema7 else 'red'}]{ema3:.4f}[/] vs [{ 'red' if ema3 > ema7 else 'green' }]{ema7:.4f}[/]"
            break_color = "green" if body > 120 else "yellow"
            
            hydra_line = (
                f"[dim]Oscillator: [{ao_color}]{ao:.5f}[/] | EMA: {ema_str} | Breakout Power: [{break_color}]{body:.0f}%[/][/dim]"
            )
        else:
            hydra_line = ""

        header_panel_content = Table.grid(expand=True)
        header_panel_content.add_row(header_table)
        header_panel_content.add_row(Align.center(stats_line))
        if hydra_line:
            header_panel_content.add_row(Align.center(hydra_line))

        layout["header"].update(Panel(header_panel_content, style="blue"))

        # --- MARKET TABLE ---
        market_table = Table(
            box=box.DOUBLE_EDGE, expand=True, header_style="bold white on blue"
        )
        market_table.add_column("Indicator", style="cyan")
        market_table.add_column("Value", justify="right")
        market_table.add_column("Context", justify="center")

        px = data["price"]
        ema = data["ema50"]
        trend_label = "UPTREND" if px > ema else "DOWNTREND"
        trend_style = "bold green" if px > ema else "bold red"
        rsi_val = float(data.get("rsi", 50))
        rsi_style = (
            "bold green" if rsi_val > 60 else "bold red" if rsi_val < 40 else "white"
        )

        bb_up, bb_low = data["bb_up"], data["bb_low"]
        zone = "MID"
        zone_style = "white"
        if px >= bb_up:
            zone, zone_style = "OVERBOUGHT", "bold red"
        elif px <= bb_low:
            zone, zone_style = "OVERSOLD", "bold green"

        market_table.add_row(
            "Live Price", f"{px:.5f}", "[bold white]Active[/bold white]"
        )
        market_table.add_row(
            "Trend (EMA50)",
            f"{ema:.5f}",
            f"[{trend_style}]{trend_label}[/{trend_style}]",
        )
        market_table.add_row(
            "RSI (14)", f"{rsi_val:.1f}", f"[{rsi_style}]Momentum[/{rsi_style}]"
        )
        market_table.add_row("BB Zone", zone, f"[{zone_style}]Targeting[/{zone_style}]")
        market_table.add_row(
            "Candle Pattern", data["pattern"], "[dim]Recognition[/dim]"
        )

        # HydraNet specialist votes
        if self.strategy_type == "hydra" and data.get("specialist_votes"):
            votes = data["specialist_votes"]
            if isinstance(votes, dict):
                for name, prob in votes.items():
                    direction = "UP" if prob > 0.5 else "DN"
                    conf = prob if prob > 0.5 else 1 - prob
                    v_color = "green" if direction == "UP" else "red"
                    market_table.add_row(
                        f"  {name.title()} Net",
                        f"[{v_color}]{direction}[/{v_color}]",
                        f"{conf:.1%}",
                    )

        action = data["action"]
        action_color = "on green" if action == "UP" else "on red" if action == "DOWN" else ""
        market_table.add_row(
            "[bold yellow]SIGNAL[/bold yellow]",
            f"[bold white {action_color}] {action} [/bold white {action_color}]",
            f"[dim]{data['status']}[/dim]",
        )

        layout["market"].update(
            Panel(
                market_table,
                title=f"[bold]Market Data: {self.asset}[/bold]",
                border_style="white",
            )
        )

        # --- HISTORY TABLE ---
        history_table = Table(box=box.SIMPLE, expand=True, header_style="bold magenta")
        history_table.add_column("Time", style="dim")
        history_table.add_column("Dir", justify="center")
        history_table.add_column("Result", justify="center")
        history_table.add_column("P/L", justify="right")

        for trade in reversed(self.trade_history[-10:]):
            res_style = (
                "bold green"
                if trade["result"] == "WIN"
                else "bold red"
                if trade["result"] == "LOSS"
                else "white"
            )
            pnl_style = (
                "green"
                if trade["profit"] > 0
                else "red"
                if trade["profit"] < 0
                else "white"
            )
            dir_style = "green" if trade.get("direction", "CALL") == "CALL" else "red"
            history_table.add_row(
                trade["time"],
                f"[{dir_style}]{trade.get('direction', 'CALL')}[/{dir_style}]",
                f"[{res_style}]{trade['result']}[/{res_style}]",
                f"[{pnl_style}]${trade['profit']:.2f}[/{pnl_style}]",
            )

        layout["history"].update(
            Panel(
                history_table,
                title="[bold]Last 10 Trades[/bold]",
                border_style="magenta",
            )
        )

        strategy_name = "HydraNet Ensemble" if self.strategy_type == "hydra" else "Legacy V5.4"
        footer_text = f"{strategy_name} | Asset: {self.asset} | TF: {self.timeframe}s | Reconnects: {self.reconnect_attempts}"
        layout["footer"].update(Align.center(f"[dim]{footer_text}[/dim]"))
        return layout
        
    def generate_multi_dashboard(self):
        layout = Layout()
        layout.split(
            Layout(name="header", size=5),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )

        current_time = datetime.now().strftime("%H:%M:%S")

        # --- HEADER ---
        header_table = Table.grid(expand=True)
        conn_color = "green" if self.is_connected else "red"
        conn_indicator = f"[{conn_color}]● {'ONLINE' if self.is_connected else 'RECONNECTING'}[/{conn_color}]"

        header_table.add_row(
            f"[bold cyan]QUOTEX MULTI-ASSET ENGINE ({len(self.assets)} Pairs) | {self.mode}[/bold cyan]",
            f"[bold white]{current_time}[/bold white]",
            conn_indicator,
        )

        stats = self.session_stats
        pnl_color = "green" if stats["pnl"] > 0 else "red" if stats["pnl"] < 0 else "white"
        wr = (stats["wins"] / stats["total_trades"] * 100) if stats["total_trades"] > 0 else 0
        wr_color = "green" if wr >= 75 else "yellow" if wr >= 60 else "white"

        stats_line = (
            f"[white]Balance: [bold green]${self.global_balance:.2f}[/bold green] | "
            f"Wins: [green]{stats['wins']}[/green] | Losses: [red]{stats['losses']}[/red] | "
            f"WR: [{wr_color}]{wr:.1f}%[/{wr_color}] | "
            f"P/L: [{pnl_color}]${stats['pnl']:.2f}[/{pnl_color}][/white]"
        )

        header_panel_content = Table.grid(expand=True)
        header_panel_content.add_row(header_table)
        header_panel_content.add_row(Align.center(stats_line))
        layout["header"].update(Panel(header_panel_content, style="blue"))

        # --- MULTI-MARKET TABLE ---
        market_table = Table(box=box.DOUBLE_EDGE, expand=True, header_style="bold white on blue")
        market_table.add_column("Asset", style="cyan", justify="left")
        market_table.add_column("Live Price", justify="right")
        market_table.add_column("Signal", justify="center")
        market_table.add_column("Oscillator", justify="right")
        market_table.add_column("Status / Countdown", style="dim", justify="left")

        for ast in self.assets:
            data = self.market_state[ast]
            px = data.get("price", 0.0)
            action = data.get("action", "-")
            status = data.get("status", "Waiting...")
            
            # Action styling
            action_color = "on green" if action == "UP" else "on red" if action == "DOWN" else ""
            action_fmt = f"[bold white {action_color}] {action} [/bold white {action_color}]" if action in ("UP", "DOWN") else action
            
            # Strategy specific metrics
            metrics = data.get("metrics", {})
            osc_fmt = "--"
            if self.strategy_type == "ao_ema":
                ao = metrics.get("ao", 0.0)
                ao_color = "green" if ao > 0 else ("red" if ao < 0 else "white")
                osc_fmt = f"[{ao_color}]{ao:.5f}[/]"

            market_table.add_row(
                f"[bold]{ast}[/bold]",
                f"{px:.5f}",
                action_fmt,
                osc_fmt,
                status
            )

        layout["body"].update(Panel(market_table, title="[bold]Concurrent Live Scanners[/bold]", border_style="white"))

        # --- HISTORY LOG ---
        history_str = ""
        for trade in reversed(self.trade_history[-2:]):
            res_style = "bold green" if trade["outcome"] == "WIN" else "bold red" if trade["outcome"] == "LOSS" else "bold yellow"
            history_str += f"[{trade['time']}] {trade['asset']} {trade['direction']} -> [{res_style}]{trade['outcome']} (${trade['pnl']:.2f})[/{res_style}]\n"

        layout["footer"].update(Panel(history_str.strip() if history_str else "No trades executed yet.", title="Recent Output", border_style="green"))

        return layout

    async def check_connection(self):
        try:
            connected = await self.client.check_connect()
            if not connected:
                self.is_connected = False
                self.market_state[self.asset]["status"] = "Wait... Reconnecting"
                try:
                    check, _ = await self.client.connect()
                    if check:
                        self.is_connected = True
                        self.reconnect_attempts += 1
                        self.last_reconnect_time = datetime.now().strftime("%H:%M")
                        self.market_state[self.asset]["status"] = "Back Online!"
                except Exception as reconn_err:
                    with open(self.debug_file, "a") as f:
                        f.write(f"Reconnect failed: {reconn_err}\n")
                    self.is_connected = False
            else:
                self.is_connected = True
        except Exception as e:
            self.is_connected = False
            with open(self.debug_file, "a") as f:
                f.write(f"Conn Watchdog Error: {e}\n")

    async def start(self):
        self.client.set_account_mode(self.mode)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                check, reason = await self.client.connect()
                if check:
                    self.is_connected = True
                    await asyncio.sleep(1)
                    self.global_balance = await self.client.get_balance()

                    # Sync HydraNet strategy with real account balance
                    if self.strategy_type == "hydra":
                        self.strategy.risk_manager.balance = self.global_balance
                        self.strategy.risk_manager.initial_balance = self.global_balance
                        self.strategy.risk_manager.peak_balance = self.global_balance
                        # Ensure online learning is active for live
                        self.strategy.online_learning_enabled = True
                        # Unfreeze normalizer for live market adaptation
                        if self.strategy.normalizer is not None:
                            self.strategy.normalizer.unfreeze()

                    break
                else:
                    with open(self.debug_file, "a") as f:
                        f.write(f"Connect attempt {attempt + 1} failed: {reason}\n")
            except Exception as e:
                with open(self.debug_file, "a") as f:
                    f.write(f"Connect error (attempt {attempt + 1}): {e}\n")

        self.running = True
        with Live(self.generate_dashboard(), refresh_per_second=2, screen=True) as live:
            while self.running:
                try:
                    await self.check_connection()

                    if self.is_connected:
                        fetch_coroutines = [self.refresh_data(ast) for ast in self.assets]
                        await asyncio.gather(*fetch_coroutines)
                        
                        self.global_balance = await self.client.get_balance()

                        # Keep HydraNet risk manager in sync with real balance
                        if self.strategy_type == "hydra":
                            self.strategy.risk_manager.balance = self.global_balance
                            self.strategy.risk_manager.peak_balance = max(
                                self.strategy.risk_manager.peak_balance,
                                self.global_balance
                            )

                    live.update(self.generate_dashboard())

                    # Prevent memory creep in long sessions
                    if len(self.trade_history) > 100:
                        self.trade_history = self.trade_history[-50:]

                except Exception as e:
                    with open(self.debug_file, "a") as f:
                        f.write(f"Dashboard Loop Error: {e}\n")

                # Predict exactly when the current candle will close
                current_time = time.time()
                remainder = current_time % self.timeframe
                
                # If we are within 2s of the boundary, sleep right up to the literal edge + 0.1s
                if self.timeframe - remainder <= 2.0:
                    sleep_time = (self.timeframe - remainder) + 0.1
                else:
                    # Otherwise sleep standard UI ping to keep HUD fast
                    sleep_time = min(2.0, self.timeframe - remainder - 1.0)
                    
                # Ensure sleep doesn't somehow go perfectly negative
                await asyncio.sleep(max(0.1, sleep_time))

    async def refresh_data(self, asset):
        try:
            # Stagger socket dispatches tightly to prevent backend WebSocket rate/flood limits
            idx = self.assets.index(asset)
            if idx > 0:
                await asyncio.sleep(idx * 0.05)
                
            history_size = self.timeframe * 100

            fetch_task = asyncio.create_task(
                self.client.get_candles(
                    asset, time.time(), history_size, self.timeframe
                )
            )

            try:
                candles = await asyncio.wait_for(fetch_task, timeout=15.0)
            except asyncio.TimeoutError:
                self.market_state[asset]["status"] = "Timeout - Retrying..."
                fetch_task.cancel()
                try:
                    await fetch_task
                except:
                    pass
                self.is_connected = False
                return

            if not candles or len(candles) < 30:
                self.market_state[asset]["status"] = "Syncing Data..."
                return

            df = pd.DataFrame(candles)
            df = df.sort_values("time").reset_index(drop=True)
            for col in ["open", "high", "low", "close"]:
                df[col] = df[col].astype(float)

            # EXTREMELY CRITICAL: Sometimes the Quotex API lags and doesn't append the forming candle yet.
            # We must only strip the last candle if it's ACTUALLY the currently forming open candle.
            # If its age is less than the timeframe, it hasn't closed yet. Strip it.
            last_candle_time = float(df['time'].iloc[-1])
            if (time.time() - last_candle_time) < self.timeframe:
                closed_df = df.iloc[:-1].copy()
            else:
                closed_df = df.copy()
                
            decision = self.strategy.execute(closed_df)
            
            # Enforce strict 0-3 second execution limit locking to ensure precise Quotex matching
            seconds_into = time.time() % self.timeframe
            if seconds_into > 3.0 and decision["decision"] in ("UP", "DOWN"):
                decision["decision"] = "WAIT"
                decision["reason"] += " (Awaiting Next Close)"
                
            metrics = decision.get("metrics", {})
            last_price = float(df["close"].iloc[-1]) # Keep displaying the true LIVE unclosed price on the HUD

            self.market_state[asset].update(
                {
                    "price": last_price,
                    "metrics": metrics, # Important: Expose raw metrics dict to UI
                    "rsi": metrics.get("rsi", 50.0),
                    "ema50": metrics.get("ema50", last_price),
                    "bb_up": metrics.get("bb_up", last_price),
                    "bb_low": metrics.get("bb_low", last_price),
                    "adx": metrics.get("adx", 0.0),
                    "pattern": metrics.get("pattern", "None"),
                    "score": decision["confluence_score"],
                    "action": decision["decision"],
                    "status": decision["reason"],
                    "regime": decision.get("regime", "-"),
                    "confidence": decision.get("confidence", 0),
                    "specialist_votes": decision.get("specialist_votes", {}),
                }
            )

            # Execute trade for both UP and DOWN signals
            if decision["decision"] in ("UP", "DOWN"):
                if time.time() - self.last_trade_time[asset] > self.timeframe:
                    asyncio.create_task(self.execute_trade(asset, decision))
        except Exception as e:
            self.is_connected = False
            with open(self.debug_file, "a") as f:
                f.write(f"Data Sync Error: {e}\n")

    async def execute_trade(self, asset, decision):
        async with self.trade_locks[asset]:
            if time.time() - self.last_trade_time[asset] < self.timeframe:
                return
            self.last_trade_time[asset] = time.time()

            # Stake sizing
            if self.strategy_type == "hydra":
                target = max(1, int(decision.get("stake", 1)))
            else:
                target = max(1, int(self.global_balance * 0.02))

            # Direction: UP = call, DOWN = put
            direction = "call" if decision["decision"] == "UP" else "put"
            dir_label = "CALL" if direction == "call" else "PUT"

            self.market_state[asset]["status"] = f"SNIPING {decision['decision']}..."

            # Target specific execution duration if strategy overrides it, otherwise lock onto chart timeframe
            execution_duration = decision.get("duration_override", self.timeframe)

            try:
                # Timeout to prevent hanging on unresponsive server
                try:
                    status, buy_info = await asyncio.wait_for(
                        self.client.buy(
                            target, asset, direction, duration=execution_duration,
                            time_mode="TIMER"
                        ),
                        timeout=30.0
                    )
                except asyncio.TimeoutError:
                    self.market_state[asset]["status"] = "Buy Timeout"
                    with open(self.debug_file, "a") as f:
                        f.write(f"[{datetime.now().strftime('%H:%M:%S')}] BUY TIMED OUT after 30s\n")
                    return

                # Deep Log for Debugging
                with open(self.debug_file, "a") as f:
                    f.write(
                        f"[{datetime.now().strftime('%H:%M:%S')}] TRADE ATTEMPT - "
                        f"Dir: {dir_label} | Stake: ${target} | Status: {status} | Info: {buy_info}\n"
                    )

                if status:
                    # Check result with timeout
                    try:
                        win_amount = await asyncio.wait_for(
                            self.client.check_win(buy_info.get("id")),
                            timeout=self.timeframe + 30
                        )
                    except asyncio.TimeoutError:
                        self.market_state[asset]["status"] = "Result Timeout"
                        with open(self.debug_file, "a") as f:
                            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] CHECK_WIN TIMED OUT\n")
                        return

                    # Compute definitive float profit mapping
                    # Quotex `check_win` returns the raw Profit Float.
                    try:
                        net_profit = float(win_amount)
                        if net_profit > 0.01:
                            outcome = "WIN"
                        elif net_profit < -0.01:
                            outcome = "LOSS"
                        else:
                            outcome = "DRAW"
                    except Exception:
                        outcome = "DRAW"
                        net_profit = 0

                    self.session_stats["total_trades"] += 1
                    if outcome == "WIN":
                        self.session_stats["wins"] += 1
                    elif outcome == "LOSS":
                        self.session_stats["losses"] += 1
                    else:
                        self.session_stats["draws"] += 1

                    self.session_stats["pnl"] += net_profit
                    self.market_state[asset]["profit"] += net_profit

                    trade_record = {
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "asset": asset,
                        "direction": dir_label,
                        "stake": target,
                        "outcome": outcome,
                        "pnl": net_profit,
                    }
                    self.trade_history.append(trade_record)

                    # Feed outcome to HydraNet online learner
                    if self.strategy_type == "hydra":
                        won = outcome == "WIN"
                        self.strategy.record_outcome(won, net_profit)

                else:
                    self.market_state[asset]["status"] = "Trade Blocked"
            except Exception as e:
                import traceback

                with open(self.debug_file, "a") as f:
                    f.write(
                        f"[{datetime.now().strftime('%H:%M:%S')}] EXECUTION ERROR:\n{traceback.format_exc()}\n"
                    )
                self.market_state[asset]["status"] = "Execution Error"

    async def stop(self):
        self.running = False

        # Save HydraNet models on exit
        if self.strategy_type == "hydra":
            try:
                self.strategy.save_models()
            except Exception:
                pass

        await self.client.close()
