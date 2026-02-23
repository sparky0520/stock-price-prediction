import pandas as pd
import numpy as np
import os

# --- Reusing Indicator Logic (could be a shared module, but keeping self-contained for ease) ---

def calculate_rsi(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_bollinger_bands(series, window=20, num_std=2):
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

def calculate_atr(high, low, close, window=14):
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(window=window).mean()

def process_crypto_data():
    input_dir = "data/raw"
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    
    # Verify file exists
    # Note: Filename based on fetch_crypto.py default
    crypto_path = os.path.join(input_dir, "BTC_USDT_1m_3y.csv")
    
    if not os.path.exists(crypto_path):
        print(f"Error: {crypto_path} not found. Run fetch_crypto.py first.")
        # Try to find any csv in data/raw that looks like crypto data
        files = [f for f in os.listdir(input_dir) if f.endswith('.csv') and 'USDT' in f]
        if files:
            print(f"Found alternative file: {files[0]}")
            crypto_path = os.path.join(input_dir, files[0])
        else:
            return

    print(f"Loading crypto data from {crypto_path}...")
    df = pd.read_csv(crypto_path, parse_dates=['Datetime'], index_col='Datetime')
    print(f"Loaded {len(df)} rows.")

    # Drop potential duplicates
    df = df[~df.index.duplicated(keep='first')]
    
    # Feature Engineering
    print("Calculating technical indicators...")
    
    # Optimization: For very large datasets, some rolling ops can be slow, but pandas is generally optimized.
    
    df['SMA_9'] = df['Close'].rolling(window=9).mean()
    df['SMA_21'] = df['Close'].rolling(window=21).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    
    df['RSI_14'] = calculate_rsi(df['Close'])
    df['MACD'], df['MACD_Signal'] = calculate_macd(df['Close'])
    df['BB_Upper'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'])
    df['ATR_14'] = calculate_atr(df['High'], df['Low'], df['Close'])
    
    # Time Features
    df['Hour'] = df.index.hour
    df['Minute'] = df.index.minute
    df['DayOfWeek'] = df.index.dayofweek

    # --- Target Variable Creation for Scalping ---
    # Prediction Horizon: Next 5 minutes (5 candles)
    # Crypto minimal volatility is higher, so slightly higher threshold or same?
    # Let's keep 0.5% (0.005) as it is significant for 1m scalping
    
    future_horizon = 5
    threshold = 0.005 
    
    df['Future_Close'] = df['Close'].shift(-future_horizon)
    df['Return_5m'] = (df['Future_Close'] - df['Close']) / df['Close']
    
    conditions = [
        (df['Return_5m'] > threshold),
        (df['Return_5m'] < -threshold)
    ]
    choices = ['Buy', 'Sell']
    df['Target'] = np.select(conditions, choices, default='Hold')
    
    # Drop NaNs
    df.dropna(inplace=True)
    
    output_path = os.path.join(output_dir, "BTC_USDT_1m_model_ready.csv")
    df.to_csv(output_path)
    print(f"Processed crypto data saved to {output_path}")
    print(f"Dataset shape: {df.shape}")
    print("Target distribution:")
    print(df['Target'].value_counts(normalize=True))

if __name__ == "__main__":
    process_crypto_data()
