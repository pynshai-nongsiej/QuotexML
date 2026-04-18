"""
HydraNet Backtester v1.0
────────────────────────
Walk-forward backtester with per-regime stats, equity curve,
Monte Carlo analysis, and Rich terminal output.
"""

import numpy as np
import pandas as pd
import os, time
from typing import Dict, List, Optional
from hydra_strategy import HydraStrategy, RegimeDetector
from hydra_data_generator import generate_realistic_data, generate_with_patterns

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich import box

console = Console()


class HydraBacktester:
    """
    Walk-forward backtester for HydraNet strategy.

    Split: 60% train → 40% test (no lookahead bias).
    """

    def __init__(self, df: pd.DataFrame, train_ratio: float = 0.6,
                 payout: float = 0.80, initial_balance: float = 1000.0):
        self.df = df.copy()
        self.train_ratio = train_ratio
        self.payout = payout  # Binary options payout (e.g., 80%)
        self.initial_balance = initial_balance

        split = int(len(df) * train_ratio)
        self.train_df = df.iloc[:split].copy()
        self.test_df = df.iloc[split:].copy()
        self.strategy = HydraStrategy(initial_balance=initial_balance)

    def run(self, epochs: int = 300, min_lookback: int = 120,
            verbose: bool = True) -> pd.DataFrame:
        """
        Full walk-forward backtest:
        1. Train on first 60% of data
        2. Test on remaining 40%
        """
        if verbose:
            console.print(Panel.fit(
                "[bold cyan]HYDRANET WALK-FORWARD BACKTESTER[/bold cyan]\n"
                f"[dim]Train: {len(self.train_df)} candles | Test: {len(self.test_df)} candles[/dim]",
                border_style="blue"
            ))

        # ── Phase 1: Train ───────────────────────────────────────────
        if verbose:
            console.print("\n[bold yellow]Phase 1: Training Ensemble on Historical Data...[/bold yellow]")

        self.strategy.train_on_historical(self.train_df, epochs=epochs, verbose=verbose)

        # Disable online learning during backtest to prevent model corruption
        self.strategy.online_learning_enabled = False

        # ── Phase 2: Test ────────────────────────────────────────────
        if verbose:
            console.print("\n[bold green]Phase 2: Testing on Out-of-Sample Data...[/bold green]")

        results = []
        window_size = 200
        test_start = 0

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
            disable=not verbose,
        ) as progress:
            task = progress.add_task("Backtesting", total=len(self.test_df) - 2)

            for i in range(min_lookback, len(self.test_df) - 1):
                # Build lookback window from test data
                start = max(0, i - window_size)
                window = self.test_df.iloc[start:i + 1]

                # Get signal
                decision = self.strategy.execute(window)

                # Determine actual outcome
                current_close = self.test_df['close'].iloc[i]
                future_close = self.test_df['close'].iloc[i + 1]
                actual = "UP" if future_close > current_close else "DOWN"

                # Record result
                if decision['decision'] != "WAIT":
                    won = decision['decision'] == actual
                    pnl = decision['stake'] * self.payout if won else -decision['stake']

                    # Feed back to online learner
                    self.strategy.record_outcome(won, pnl)

                    results.append({
                        'idx': i,
                        'timestamp': str(self.test_df.get('timestamp',
                                        pd.Series(range(len(self.test_df)))).iloc[i]),
                        'decision': decision['decision'],
                        'actual': actual,
                        'won': won,
                        'confidence': decision['confidence'],
                        'regime': decision.get('regime', 'UNKNOWN'),
                        'reason': decision['reason'],
                        'stake': decision['stake'],
                        'pnl': pnl,
                        'specialist_votes': str(decision.get('specialist_votes', {})),
                    })

                progress.update(task, advance=1)

        return pd.DataFrame(results)

    def stats(self, results: pd.DataFrame):
        """Print comprehensive backtest statistics using Rich."""
        if len(results) == 0:
            console.print("[red]No trades executed. Model may need more training data.[/red]")
            return

        # ── Overall Stats ────────────────────────────────────────────
        total = len(results)
        wins = results['won'].sum()
        losses = total - wins
        win_rate = wins / total * 100
        total_pnl = results['pnl'].sum()
        avg_pnl = results['pnl'].mean()
        trade_density = total / len(self.test_df) * 100

        # Equity curve
        equity = self.initial_balance + results['pnl'].cumsum()
        max_drawdown = self._max_drawdown(equity.values)

        # Consecutive stats
        max_win_streak = self._max_streak(results['won'].values, True)
        max_loss_streak = self._max_streak(results['won'].values, False)

        # Profit factor
        gross_profit = results[results['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(results[results['pnl'] < 0]['pnl'].sum())
        profit_factor = gross_profit / (gross_loss + 1e-10)

        # ── Main Stats Table ─────────────────────────────────────────
        wr_color = "green" if win_rate >= 75 else "yellow" if win_rate >= 60 else "red"
        pnl_color = "green" if total_pnl > 0 else "red"

        stats_table = Table(
            title="[bold]HYDRANET BACKTEST RESULTS[/bold]",
            box=box.DOUBLE_EDGE,
            show_header=True,
            header_style="bold white on blue",
        )
        stats_table.add_column("Metric", style="cyan", width=25)
        stats_table.add_column("Value", justify="right", width=20)

        stats_table.add_row("Total Trades", f"[bold]{total}[/bold]")
        stats_table.add_row("Wins / Losses", f"[green]{wins}[/green] / [red]{losses}[/red]")
        stats_table.add_row("Win Rate", f"[bold {wr_color}]{win_rate:.2f}%[/bold {wr_color}]")
        stats_table.add_row("Trade Density", f"{trade_density:.2f}%")
        stats_table.add_row("Total P/L", f"[{pnl_color}]${total_pnl:.2f}[/{pnl_color}]")
        stats_table.add_row("Avg P/L per Trade", f"${avg_pnl:.2f}")
        stats_table.add_row("Profit Factor", f"{profit_factor:.2f}")
        stats_table.add_row("Max Drawdown", f"[red]{max_drawdown:.2f}%[/red]")
        stats_table.add_row("Max Win Streak", f"[green]{max_win_streak}[/green]")
        stats_table.add_row("Max Loss Streak", f"[red]{max_loss_streak}[/red]")
        stats_table.add_row("Final Equity", f"[bold]${equity.iloc[-1]:.2f}[/bold]")
        stats_table.add_row("Avg Confidence", f"{results['confidence'].mean():.2%}")

        console.print(stats_table)

        # ── Per-Direction Stats ──────────────────────────────────────
        dir_table = Table(
            title="[bold]Performance by Direction[/bold]",
            box=box.ROUNDED,
            header_style="bold",
        )
        dir_table.add_column("Direction", style="cyan")
        dir_table.add_column("Trades", justify="right")
        dir_table.add_column("Win Rate", justify="right")
        dir_table.add_column("Avg Confidence", justify="right")
        dir_table.add_column("P/L", justify="right")

        for direction in ['UP', 'DOWN']:
            subset = results[results['decision'] == direction]
            if len(subset) == 0:
                continue
            wr = subset['won'].mean() * 100
            wr_c = "green" if wr >= 75 else "yellow" if wr >= 60 else "red"
            pnl = subset['pnl'].sum()
            pnl_c = "green" if pnl > 0 else "red"
            dir_table.add_row(
                f"[bold]{direction}[/bold]",
                str(len(subset)),
                f"[{wr_c}]{wr:.1f}%[/{wr_c}]",
                f"{subset['confidence'].mean():.2%}",
                f"[{pnl_c}]${pnl:.2f}[/{pnl_c}]",
            )
        console.print(dir_table)

        # ── Per-Regime Stats ─────────────────────────────────────────
        regime_table = Table(
            title="[bold]Performance by Market Regime[/bold]",
            box=box.ROUNDED,
            header_style="bold",
        )
        regime_table.add_column("Regime", style="cyan")
        regime_table.add_column("Trades", justify="right")
        regime_table.add_column("Win Rate", justify="right")
        regime_table.add_column("Avg Confidence", justify="right")
        regime_table.add_column("P/L", justify="right")

        for regime in results['regime'].unique():
            subset = results[results['regime'] == regime]
            wr = subset['won'].mean() * 100
            wr_c = "green" if wr >= 75 else "yellow" if wr >= 60 else "red"
            pnl = subset['pnl'].sum()
            pnl_c = "green" if pnl > 0 else "red"
            regime_table.add_row(
                regime,
                str(len(subset)),
                f"[{wr_c}]{wr:.1f}%[/{wr_c}]",
                f"{subset['confidence'].mean():.2%}",
                f"[{pnl_c}]${pnl:.2f}[/{pnl_c}]",
            )
        console.print(regime_table)

        # ── Confidence Breakdown ─────────────────────────────────────
        conf_table = Table(
            title="[bold]Win Rate by Confidence Band[/bold]",
            box=box.ROUNDED,
            header_style="bold",
        )
        conf_table.add_column("Confidence Range", style="cyan")
        conf_table.add_column("Trades", justify="right")
        conf_table.add_column("Win Rate", justify="right")

        bins = [(0.55, 0.65), (0.65, 0.75), (0.75, 0.85), (0.85, 1.01)]
        for lo, hi in bins:
            subset = results[(results['confidence'] >= lo) & (results['confidence'] < hi)]
            if len(subset) == 0:
                continue
            wr = subset['won'].mean() * 100
            wr_c = "green" if wr >= 75 else "yellow" if wr >= 60 else "red"
            conf_table.add_row(
                f"{lo:.0%} – {hi:.0%}",
                str(len(subset)),
                f"[{wr_c}]{wr:.1f}%[/{wr_c}]",
            )
        console.print(conf_table)

        # ── Monte Carlo Drawdown ─────────────────────────────────────
        console.print("\n[dim]Running Monte Carlo simulation (1000 paths)...[/dim]")
        mc_drawdowns = self._monte_carlo(results['pnl'].values, n_simulations=1000,
                                          initial_balance=self.initial_balance)
        p95 = np.percentile(mc_drawdowns, 95)
        p99 = np.percentile(mc_drawdowns, 99)
        console.print(f"  95th percentile max drawdown: [yellow]${p95:.2f}[/yellow]")
        console.print(f"  99th percentile max drawdown: [red]${p99:.2f}[/red]")

    @staticmethod
    def _max_drawdown(equity: np.ndarray) -> float:
        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / (peak + 1e-10) * 100
        return float(np.max(dd))

    @staticmethod
    def _max_streak(wins: np.ndarray, target: bool) -> int:
        max_s = 0
        curr = 0
        for w in wins:
            if w == target:
                curr += 1
                max_s = max(max_s, curr)
            else:
                curr = 0
        return max_s

    @staticmethod
    def _monte_carlo(pnl: np.ndarray, n_simulations: int = 1000,
                     initial_balance: float = 1000.0) -> np.ndarray:
        """Shuffle P/L values to estimate worst-case drawdown distribution."""
        drawdowns = []
        for _ in range(n_simulations):
            shuffled = np.random.permutation(pnl)
            equity = initial_balance + np.cumsum(shuffled)
            peak = np.maximum.accumulate(equity)
            dd = peak - equity
            drawdowns.append(np.max(dd))
        return np.array(drawdowns)


# ════════════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ════════════════════════════════════════════════════════════════════════

def main():
    console.print(Panel.fit(
        "[bold cyan]HYDRANET BACKTESTER[/bold cyan]\n"
        "[dim]Neural Ensemble Walk-Forward Strategy Tester[/dim]",
        border_style="blue"
    ))

    # Data source selection
    console.print("\n[bold]Select Data Source:[/bold]")
    console.print("  [1] Generate realistic synthetic data")
    console.print("  [2] Generate pattern-enhanced data (for model validation)")
    console.print("  [3] Test on Live Market Data (quotex_history.csv)")
    console.print("  [4] Load from custom CSV file")

    choice = input("\nSelect -> ").strip()

    if choice == "3":
        path = "quotex_history.csv"
        if not os.path.exists(path):
            console.print(f"[red]File not found: {path}[/red]. Please ensure quotex_history.csv exists.")
            return
        df = pd.read_csv(path)
        console.print(f"[green]Loaded {len(df)} candles from {path}[/green]")
    elif choice == "4":
        path = input("Enter CSV path: ").strip()
        if not os.path.exists(path):
            console.print(f"[red]File not found: {path}[/red]")
            return
        df = pd.read_csv(path)
        console.print(f"[green]Loaded {len(df)} candles from {path}[/green]")
    else:
        try:
            n_input = input("Number of candles (default 5000): ").strip()
            n = int(n_input) if n_input else 5000
        except ValueError:
            n = 5000

        seed_input = input("Random seed (default 42): ").strip()
        try:
            seed = int(seed_input) if seed_input else 42
        except ValueError:
            seed = 42

        console.print(f"\n[yellow]Generating {n} candles (seed={seed})...[/yellow]")
        if choice == "2":
            df = generate_with_patterns(n=n, seed=seed)
            console.print("[green]Pattern-enhanced data generated.[/green]")
        else:
            df = generate_realistic_data(n=n, seed=seed)
            console.print("[green]Realistic data generated.[/green]")

    # Training epochs
    try:
        epochs_input = input("Training epochs (default 300): ").strip()
        epochs = int(epochs_input) if epochs_input else 300
    except ValueError:
        epochs = 300

    # Starting balance
    try:
        balance_input = input("Starting balance (default 1000): ").strip()
        balance = float(balance_input) if balance_input else 1000.0
    except ValueError:
        balance = 1000.0

    # Run backtest
    tester = HydraBacktester(df, payout=0.80, initial_balance=balance)
    results = tester.run(epochs=epochs, verbose=True)

    if len(results) > 0:
        tester.stats(results)

        # Save results
        os.makedirs("logs", exist_ok=True)
        results.to_csv("logs/hydra_backtest_results.csv", index=False)
        console.print("\n[green]Results saved to logs/hydra_backtest_results.csv[/green]")
    else:
        console.print("[red]No trades were triggered during the backtest.[/red]")


if __name__ == "__main__":
    main()
