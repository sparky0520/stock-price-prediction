import pandas as pd
import numpy as np
import os

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

def process_data():
    input_dir = "data/raw"
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    
    intraday_path = os.path.join(input_dir, "BHEL_1m.csv")
    if not os.path.exists(intraday_path):
        print(f"Error: {intraday_path} not found. Run fetch_data.py first.")
        return

    print("Loading 1-minute intraday data...")
    df = pd.read_csv(intraday_path, parse_dates=['Datetime'], index_col='Datetime')
    
    # Feature Engineering
    print("Calculating technical indicators...")
    # Price
    df['Close'] = df['Close'].astype(float)
    
    # Moving Averages
    df['SMA_9'] = df['Close'].rolling(window=9).mean()
    df['SMA_21'] = df['Close'].rolling(window=21).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    
    # RSI
    df['RSI_14'] = calculate_rsi(df['Close'])
    
    # MACD
    df['MACD'], df['MACD_Signal'] = calculate_macd(df['Close'])
    
    # Bollinger Bands
    df['BB_Upper'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'])
    
    # ATR
    df['ATR_14'] = calculate_atr(df['High'], df['Low'], df['Close'])
    
    # Time Patterns
    df['Hour'] = df.index.hour
    df['Minute'] = df.index.minute
    df['DayOfWeek'] = df.index.dayofweek

    # --- Target Variable Creation ---
    # Prediction Horizon: Next 5 minutes (5 candles)
    # Label: Buy if return > 0.3%, Sell if return < -0.3%, else Hold (Lower threshold for 1m scalping)
    
    future_horizon = 5
    threshold = 0.003 # 0.3%
    
    df['Future_Close'] = df['Close'].shift(-future_horizon)
    df['Return_5m'] = (df['Future_Close'] - df['Close']) / df['Close']
    
    conditions = [
        (df['Return_5m'] > threshold),
        (df['Return_5m'] < -threshold)
    ]
    choices = ['Buy', 'Sell']
    df['Target'] = np.select(conditions, choices, default='Hold')
    
    # Drop NaNs created by rolling windows and shifting
    df.dropna(inplace=True)
    
    output_path = os.path.join(output_dir, "BHEL_1m_model_ready.csv")
    df.to_csv(output_path)
    print(f"Processed data saved to {output_path}")
    print(f"Dataset shape: {df.shape}")
    print("Target distribution:")
    print(df['Target'].value_counts(normalize=True))

if __name__ == "__main__":
    process_data()
