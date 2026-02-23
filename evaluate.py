"""
evaluate.py - Backtest & Performance Evaluation for BHEL DRL Agent

Usage:
    uv run evaluate.py [--model-path models/ppo_bhel_final]

Loads a trained model, runs it on the test set, and reports:
  - Cumulative Return vs Buy-and-Hold baseline
  - Sharpe Ratio
  - Max Drawdown
  - Trade count
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from environment import StockTradingEnv

# --- Configuration ---
DATA_PATH = "data/processed/BHEL_1m_model_ready.csv"
TRAIN_SPLIT = 0.80
INITIAL_BALANCE = 100_000.0
RESULTS_DIR = "results"
RISK_FREE_RATE = 0.065 / 252 / 375  # ~6.5% annual, ~375 min trading day

ACTION_LABELS = {0: "Hold", 1: "Buy", 2: "Sell"}


def compute_sharpe(returns: pd.Series, rfr: float = RISK_FREE_RATE) -> float:
    """Annualized Sharpe Ratio using 1-minute bars."""
    excess = returns - rfr
    if excess.std() == 0:
        return 0.0
    return float(np.sqrt(375 * 252) * excess.mean() / excess.std())


def compute_max_drawdown(portfolio_values: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a percentage."""
    rolling_max = portfolio_values.cummax()
    drawdown = (portfolio_values - rolling_max) / rolling_max
    return float(drawdown.min())


def run_backtest(model: PPO, df: pd.DataFrame) -> pd.DataFrame:
    """Run the trained model on the dataset and return the trade history."""
    env = StockTradingEnv(df, initial_balance=INITIAL_BALANCE)
    obs, _ = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated

    return env.get_history_df()


def buy_and_hold_value(df: pd.DataFrame) -> pd.Series:
    """Simulate a simple buy-and-hold strategy."""
    entry_price = df["Close"].iloc[0]
    shares = INITIAL_BALANCE // entry_price
    remainder = INITIAL_BALANCE - shares * entry_price
    return df["Close"] * shares + remainder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="models/best/best_model", help="Path to the saved SB3 model")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Load Data ---
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH, parse_dates=["Datetime"], index_col="Datetime")
    test_df = df.iloc[int(len(df) * TRAIN_SPLIT):].copy().reset_index(drop=True)
    print(f"Test set size: {len(test_df):,} bars")

    # --- Load Model ---
    print(f"Loading model from: {args.model_path}")
    model = PPO.load(args.model_path)

    # --- Backtest ---
    print("Running backtest...")
    history = run_backtest(model, test_df)

    # --- Metrics ---
    portfolio_values = history["portfolio_value"]
    returns = portfolio_values.pct_change().dropna()

    sharpe = compute_sharpe(returns)
    max_dd = compute_max_drawdown(portfolio_values)
    total_return = (portfolio_values.iloc[-1] / INITIAL_BALANCE - 1) * 100

    bah_values = buy_and_hold_value(test_df)
    bah_return = (bah_values.iloc[-1] / INITIAL_BALANCE - 1) * 100

    trades = history[history["action"] != 0]
    num_buys = (history["action"] == 1).sum()
    num_sells = (history["action"] == 2).sum()

    print("\n" + "=" * 50)
    print("         BACKTEST RESULTS")
    print("=" * 50)
    print(f"  Total Return (Agent)   : {total_return:+.2f}%")
    print(f"  Total Return (B&H)     : {bah_return:+.2f}%")
    print(f"  Sharpe Ratio           : {sharpe:.4f}")
    print(f"  Max Drawdown           : {max_dd * 100:.2f}%")
    print(f"  # Buy Actions          : {num_buys}")
    print(f"  # Sell Actions         : {num_sells}")
    print("=" * 50)

    # --- Plot ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("BHEL Intraday DRL Agent — Backtest Results", fontsize=14)

    # Portfolio Value
    axes[0].plot(portfolio_values.values, label=f"PPO Agent ({total_return:+.1f}%)", color="royalblue")
    axes[0].plot(bah_values.values, label=f"Buy & Hold ({bah_return:+.1f}%)", color="tomato", linestyle="--")
    axes[0].set_ylabel("Portfolio Value (₹)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Price with Buy/Sell markers
    axes[1].plot(test_df["Close"].values, color="gray", linewidth=0.7, label="BHEL Price")
    buy_steps = history[history["action"] == 1]["step"].values
    sell_steps = history[history["action"] == 2]["step"].values
    axes[1].scatter(buy_steps, test_df["Close"].iloc[buy_steps].values, marker="^", color="green", s=40, label="Buy", zorder=5)
    axes[1].scatter(sell_steps, test_df["Close"].iloc[sell_steps].values, marker="v", color="red", s=40, label="Sell", zorder=5)
    axes[1].set_ylabel("Close Price (₹)")
    axes[1].set_xlabel("Time Step (1-min bars)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "backtest_results.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to: {plot_path}")

    # Save history CSV
    history_path = os.path.join(RESULTS_DIR, "trade_history.csv")
    history.to_csv(history_path, index=False)
    print(f"Trade history saved to: {history_path}")


if __name__ == "__main__":
    main()
