# Stock & Crypto Price Prediction with DRL

This project implements a Deep Reinforcement Learning (DRL) agent using PPO to trade stocks and cryptocurrencies.

## 🚀 Quick Start

### 1. Setup Environment
Install dependencies using `uv`:
```powershell
uv sync
```

### 2. Data Acquisition
Fetch historical crypto data (default: BTC/USDT, 1m timeframe, 3 years):
```powershell
uv run fetch_crypto.py
```

### 3. Data Processing
Generate technical indicators and prepare the dataset for training:
```powershell
uv run process_crypto.py
```

### 4. Training
Train the PPO agent:
```powershell
uv run train.py
```

### 5. Evaluation & Backtesting
Evaluate the trained model on the test set:
```powershell
uv run evaluate.py
```
To specify a custom model path:
```powershell
uv run evaluate.py --model-path models/ppo_bhel_final
```

### 6. Monitoring
Monitor training progress via TensorBoard:
```powershell
uv run tensorboard --logdir logs/tensorboard
```

## 📂 Project Structure
- `data/`: Raw and processed datasets.
- `models/`: Saved model checkpoints.
- `logs/`: TensorBoard logs.
- `results/`: Backtest plots and trade history.
- `environment.py`: Custom Gymnasium environment for trading.
