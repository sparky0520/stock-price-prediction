import yfinance as yf
import pandas as pd
import os

def fetch_data():
    ticker_symbol = "BHEL.NS"
    data_dir = "data/raw"
    os.makedirs(data_dir, exist_ok=True)

    print(f"Fetching data for {ticker_symbol}...")
    ticker = yf.Ticker(ticker_symbol)

    # 1. Fetch 5 years of daily data
    print("Fetching 5 years of daily data...")
    daily_data = ticker.history(period="5y", interval="1d")
    daily_path = os.path.join(data_dir, "BHEL_daily_5y.csv")
    daily_data.to_csv(daily_path)
    print(f"Saved daily data to {daily_path} ({len(daily_data)} rows)")

    # 2. Fetch max available intraday data (5m interval)
    # yfinance typically allows ~60 days for 5m data
    print("Fetching max available 5-minute intraday data...")
    intraday_data = ticker.history(period="60d", interval="5m")
    
    if intraday_data.empty:
        print("Warning: No intraday data returned. The API might have temporary limits or the period is too long.")
        # Fallback to shorter period if 60d fails, though 60d is standard limit
        try:
             print("Retrying with period='1mo'...")
             intraday_data = ticker.history(period="1mo", interval="5m")
        except Exception as e:
            print(f"Retry failed: {e}")

    intraday_path = os.path.join(data_dir, "BHEL_intraday_5m.csv")
    intraday_data.to_csv(intraday_path)
    print(f"Saved intraday data to {intraday_path} ({len(intraday_data)} rows)")

if __name__ == "__main__":
    fetch_data()
