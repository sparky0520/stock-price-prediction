"""
train.py - PPO Agent Training for Crypto (BTC/USDT) Intraday Trading

Usage:
    uv run train.py

Trains a PPO agent on the BHEL processed dataset and saves the model.
"""
import os
import pandas as pd
import numpy as np
import gc
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from environment import StockTradingEnv, FEATURE_COLS

# --- Configuration ---
DATA_PATH = "data/processed/BTC_USDT_1m_model_ready.csv"
MODEL_SAVE_DIR = "models"
LOG_DIR = "logs/tensorboard"
TRAIN_SPLIT = 0.80          # 80% train, 20% test (chronological)
TOTAL_TIMESTEPS = 4_000_000
LEARNING_RATE = 3e-4        # Step size for weight updates
N_STEPS = 2048              # Steps per rollout before updating
BATCH_SIZE = 64             # Samples per gradient update
N_EPOCHS = 10               # Times to reuse rollout data for updates
GAMMA = 0.99                # How much to value future rewards (0.99 = long term)
INITIAL_BALANCE = 100_000.0


def make_env(df: pd.DataFrame):
    """Factory for a monitored trading environment."""
    def _init():
        # df is already pre-normalized
        env = StockTradingEnv(df, initial_balance=INITIAL_BALANCE)
        env = Monitor(env)
        return env
    return _init


def main():
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # --- Load & Split Data ---
    print("Loading dataset...")
    # Read only required columns and use float32 for initial load
    raw_df = pd.read_csv(DATA_PATH, usecols=["Datetime"] + FEATURE_COLS + ["Close"], parse_dates=["Datetime"], index_col="Datetime")
    
    print("Pre-processing data (Normalization & Float32 casting)...")
    # Store raw close for PnL calculation before normalization
    raw_df["Raw_Close"] = raw_df["Close"].astype(np.float32)

    # Cast features to float32
    for col in FEATURE_COLS:
        raw_df[col] = raw_df[col].astype(np.float32)

    # Pre-normalize features once here to save CPU in environment
    means = raw_df[FEATURE_COLS].mean()
    stds = raw_df[FEATURE_COLS].std().replace(0, 1)
    raw_df[FEATURE_COLS] = (raw_df[FEATURE_COLS] - means) / stds

    # Reset index and drop Datetime
    raw_df = raw_df.reset_index(drop=True)

    split_idx = int(len(raw_df) * TRAIN_SPLIT)
    train_df = raw_df.iloc[:split_idx].copy()
    test_df = raw_df.iloc[split_idx:].copy()

    print(f"Training steps : {len(train_df):,}")
    print(f"Evaluation steps: {len(test_df):,}")

    # --- Create Environments ---
    # We pass the full slices directly; DummyVecEnv handles the rest
    train_env = DummyVecEnv([make_env(train_df)])
    eval_env = DummyVecEnv([make_env(test_df)])

    # Cleanup master dataframe to free RAM
    del raw_df
    gc.collect()

    # --- Callbacks ---
    checkpoint_cb = CheckpointCallback(
        save_freq=200_000,          # Save every ~5% of training
        save_path=MODEL_SAVE_DIR,
        name_prefix="ppo_crypto",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(MODEL_SAVE_DIR, "best"),
        log_path=LOG_DIR,
        eval_freq=100_000,          # Evaluate every ~2.5% of training
        n_eval_episodes=1,
        deterministic=True,
        render=False,
    )

    # --- Initialize PPO Model ---
    print("Initializing PPO agent...")
    
    # Custom network architecture: 2 layers of 256 neurons each
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
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
        tb_log_name="ppo_crypto",
        progress_bar=True,
    )

    # --- Save Final Model ---
    final_path = os.path.join(MODEL_SAVE_DIR, "ppo_crypto_final")
    model.save(final_path)
    print(f"\n✅ Training complete. Model saved to: {final_path}")
    print(f"   TensorBoard logs: {LOG_DIR}")
    print(f"   Run: tensorboard --logdir {LOG_DIR}")


if __name__ == "__main__":
    main()
