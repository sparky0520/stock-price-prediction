"""
Stock Trading Gymnasium Environment for BHEL Intraday DRL.
- Action Space: Discrete(3) -> [0: Hold, 1: Buy, 2: Sell]
- Observation Space: Normalized technical indicators + account state
- Reward: Net change in portfolio value, penalized for transaction costs
"""
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


# Transaction cost (brokerage + STT approximation)
TRANSACTION_COST_PCT = 0.001  # 0.1% per trade

FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA_9", "SMA_21", "SMA_50", "EMA_9",
    "RSI_14", "MACD", "MACD_Signal",
    "BB_Upper", "BB_Lower", "ATR_14",
    "Hour", "Minute", "DayOfWeek",
]


class StockTradingEnv(gym.Env):
    """
    A Gymnasium-compatible environment for intraday stock trading.
    Optimized for memory by using pre-normalized float32 data.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame, initial_balance: float = 100_000.0, render_mode=None):
        super().__init__()

        # Expecting df to be pre-filtered and pre-normalized for performance
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.render_mode = render_mode

        # --- Spaces ---
        # Observation: all features + [balance_ratio, shares_held, cost_basis_ratio]
        n_features = len(FEATURE_COLS) + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32
        )
        # Actions: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)

        # Internal state (set by reset)
        self._current_step: int = 0
        self._balance: float = initial_balance
        self._shares_held: int = 0
        self._cost_basis: float = 0.0
        self._total_portfolio_value: float = initial_balance
        self._history: list = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._current_step = 0
        self._balance = self.initial_balance
        self._shares_held = 0
        self._cost_basis = 0.0
        self._total_portfolio_value = self.initial_balance
        self._history = []

        return self._get_observation(), {}

    def step(self, action: int):
        # Use Raw_Close for actual money/PnL calculation if it exists
        price_col = "Raw_Close" if "Raw_Close" in self.df.columns else "Close"
        current_price = float(self.df.loc[self._current_step, price_col])
        prev_portfolio_value = self._total_portfolio_value

        # --- Execute Action ---
        if action == 1:  # Buy
            if self._shares_held == 0 and self._balance >= current_price:
                cost = current_price * (1 + TRANSACTION_COST_PCT)
                self._balance -= cost
                self._shares_held = 1
                self._cost_basis = cost
        elif action == 2:  # Sell
            if self._shares_held > 0:
                proceeds = current_price * (1 - TRANSACTION_COST_PCT)
                self._balance += proceeds
                self._shares_held = 0
                self._cost_basis = 0.0

        # --- Calculate Portfolio Value and Reward ---
        self._total_portfolio_value = self._balance + self._shares_held * current_price
        reward = self._total_portfolio_value - prev_portfolio_value

        # Small penalty for forced-closure (Hour 15) for BHEL scripts compatibility
        # Crypto is 24/7, but we keep the logic to avoid breaking shared environments
        if "Hour" in self.df.columns and self.df.loc[self._current_step, "Hour"] == 15 and self._shares_held > 0:
             reward -= current_price * 0.005 

        self._history.append({
            "step": self._current_step,
            "price": current_price,
            "action": action,
            "portfolio_value": self._total_portfolio_value,
            "balance": self._balance,
            "shares_held": self._shares_held,
        })

        self._current_step += 1
        terminated = self._current_step >= len(self.df) - 1
        truncated = False

        if self.render_mode == "human":
            self.render()

        return self._get_observation(), float(reward), terminated, truncated, {}

    def render(self):
        step = self._current_step
        price = float(self.df.loc[step, "Close"])
        print(
            f"Step: {step:5d} | Price: {price:8.2f} | "
            f"Held: {self._shares_held} | Balance: {self._balance:12.2f} | "
            f"Portfolio: {self._total_portfolio_value:12.2f}"
        )

    def _get_observation(self) -> np.ndarray:
        # Features are pre-normalized in the provided DataFrame
        normalized = self.df.loc[self._current_step, FEATURE_COLS].values.astype(np.float32)

        # Account state (normalized relative to initial balance)
        balance_ratio = np.float32(self._balance / self.initial_balance)
        shares_held = np.float32(self._shares_held)
        cost_basis_ratio = np.float32(self._cost_basis / self.initial_balance)

        return np.concatenate([normalized, [balance_ratio, shares_held, cost_basis_ratio]])

    def get_history_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._history)
