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
    
    Simulates a single stock with an account that has an initial balance.
    The agent can Buy, Sell or Hold at each timestep (1-minute candle).
    Only one unit of stock can be held at a time for simplicity.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame, initial_balance: float = 100_000.0, render_mode=None):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.render_mode = render_mode

        # --- Validate feature columns ---
        missing = [c for c in FEATURE_COLS if c not in self.df.columns]
        if missing:
            raise ValueError(f"DataFrame is missing columns: {missing}")

        # --- Spaces ---
        # Observation: all features + [balance_ratio, shares_held, cost_basis_ratio]
        n_features = len(FEATURE_COLS) + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32
        )

        # Actions: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)

        # Normalization stats (computed once on init)
        self._means = self.df[FEATURE_COLS].mean()
        self._stds = self.df[FEATURE_COLS].std().replace(0, 1)

        # Internal state (set by reset)
        self._current_step: int = 0
        self._balance: float = initial_balance
        self._shares_held: int = 0
        self._cost_basis: float = 0.0
        self._total_portfolio_value: float = initial_balance
        self._history: list = []

    # ------------------------------------------------------------------
    # Core Gymnasium API
    # ------------------------------------------------------------------

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
        current_price = float(self.df.loc[self._current_step, "Close"])
        prev_portfolio_value = self._total_portfolio_value
        reward = 0.0

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

        # Small penalty for holding a position overnight (end of day indicator)
        # Encourage squaring off before market close
        if self.df.loc[self._current_step, "Hour"] == 15 and self._shares_held > 0:
            reward -= current_price * 0.005  # forced-closure penalty

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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_observation(self) -> np.ndarray:
        raw_features = self.df.loc[self._current_step, FEATURE_COLS].values.astype(np.float32)
        normalized = ((raw_features - self._means.values) / self._stds.values).astype(np.float32)

        # Account state (normalized relative to initial balance)
        balance_ratio = np.float32(self._balance / self.initial_balance)
        shares_held = np.float32(self._shares_held)
        cost_basis_ratio = np.float32(self._cost_basis / self.initial_balance)

        return np.concatenate([normalized, [balance_ratio, shares_held, cost_basis_ratio]])

    def get_history_df(self) -> pd.DataFrame:
        """Return trading history as a DataFrame for evaluation."""
        return pd.DataFrame(self._history)
