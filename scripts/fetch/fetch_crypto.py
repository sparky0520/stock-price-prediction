import ccxt
import pandas as pd
import os
import time
from datetime import datetime, timedelta

def fetch_crypto_data(symbol='BTC/USDT', timeframe='1m', years=3):
    print(f"Initializing Binance exchange for {symbol}...")
    exchange = ccxt.binance({
        'enableRateLimit': True,  # ccxt respects rate limits automatically
    })
    
    data_dir = "../../data/raw"
    os.makedirs(data_dir, exist_ok=True)
    
    # Calculate start timestamp (milliseconds)
    end_time = datetime.now()
    start_time = end_time - timedelta(days=365 * years)
    since = int(start_time.timestamp() * 1000)
    
    print(f"Fetching data from {start_time} to {end_time}...")
    
    all_ohlcv = []
    limit = 1000  # Binance limit per request
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            
            # Update 'since' to the timestamp of the last candle + 1 timeframe (in ms)
            last_timestamp = ohlcv[-1][0]
            since = last_timestamp + 60000 # 1 minute in ms
            
            # Print progress/latest date
            file_date = datetime.fromtimestamp(last_timestamp / 1000)
            print(f"Fetched up to {file_date} | Total candles: {len(all_ohlcv)}")
            
            # Stop if we reached current time
            if last_timestamp >= int(end_time.timestamp() * 1000):
                break
                
            # Extra sleep explicitly just in case, though ccxt handles it
            time.sleep(0.1) 
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            time.sleep(5) # Wait longer on error
            continue

    if not all_ohlcv:
        print("No data fetched.")
        return

    # Convert to DataFrame
    columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
    df = pd.DataFrame(all_ohlcv, columns=columns)
    
    # Process Timestamp
    df['Datetime'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Datetime', inplace=True)
    df.drop(columns=['Timestamp'], inplace=True)
    
    # Save
    safe_symbol = symbol.replace('/', '_')
    filename = f"{safe_symbol}_{timeframe}_{years}y.csv"
    output_path = os.path.join(data_dir, filename)
    
    df.to_csv(output_path)
    print(f"Successfully saved {len(df)} rows to {output_path}")

if __name__ == "__main__":
    fetch_crypto_data()
