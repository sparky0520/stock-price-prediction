"""
train.py - PPO Agent Training for BHEL Intraday Stock Trading

Usage:
    uv run train.py

Trains a PPO agent on the BHEL processed dataset and saves the model.
"""
import os
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from environment import StockTradingEnv

# --- Configuration ---
DATA_PATH = "data/processed/BHEL_1m_model_ready.csv"
MODEL_SAVE_DIR = "models"
LOG_DIR = "logs/tensorboard"
TRAIN_SPLIT = 0.80          # 80% train, 20% test (chronological)
TOTAL_TIMESTEPS = 500_000
LEARNING_RATE = 3e-4
N_STEPS = 2048              # Steps per rollout buffer
BATCH_SIZE = 64
N_EPOCHS = 10
GAMMA = 0.99                # Discount factor — short-term trading, slightly lower
INITIAL_BALANCE = 100_000.0


def make_env(df: pd.DataFrame):
    """Factory for a monitored trading environment."""
    def _init():
        env = StockTradingEnv(df, initial_balance=INITIAL_BALANCE)
        env = Monitor(env)
        return env
    return _init


def main():
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # --- Load & Split Data ---
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH, parse_dates=["Datetime"], index_col="Datetime")
    df = df.reset_index(drop=True)

    split_idx = int(len(df) * TRAIN_SPLIT)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print(f"Training steps : {len(train_df):,}")
    print(f"Evaluation steps: {len(test_df):,}")

    # --- Create Environments ---
    train_env = DummyVecEnv([make_env(train_df)])
    eval_env = DummyVecEnv([make_env(test_df)])

    # --- Callbacks ---
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=MODEL_SAVE_DIR,
        name_prefix="ppo_bhel",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(MODEL_SAVE_DIR, "best"),
        log_path=LOG_DIR,
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    # --- Initialize PPO Model ---
    print("Initializing PPO agent...")
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,         # Small entropy bonus to encourage exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=LOG_DIR,
    )

    # --- Train ---
    print(f"Training for {TOTAL_TIMESTEPS:,} timesteps...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_cb, eval_cb],
        tb_log_name="ppo_bhel",
        progress_bar=True,
    )

    # --- Save Final Model ---
    final_path = os.path.join(MODEL_SAVE_DIR, "ppo_bhel_final")
    model.save(final_path)
    print(f"\n✅ Training complete. Model saved to: {final_path}")
    print(f"   TensorBoard logs: {LOG_DIR}")
    print(f"   Run: tensorboard --logdir {LOG_DIR}")


if __name__ == "__main__":
    main()
