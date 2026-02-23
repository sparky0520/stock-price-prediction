import yfinance as yf
import pandas as pd
import os

def fetch_data():
    ticker_symbol = "BHEL.NS"
    data_dir = "../../data/raw"
    os.makedirs(data_dir, exist_ok=True)

    print(f"Fetching data for {ticker_symbol}...")
    ticker = yf.Ticker(ticker_symbol)

    # Fetch max available 1-minute intraday data
    # yfinance allows 7 days for 1m data
    print("Fetching max available 1-minute intraday data (last 7 days)...")
    intraday_data = ticker.history(period="7d", interval="1m")
    
    if intraday_data.empty:
        print("Warning: No 1-minute data returned.")
    else:
        # Save
        intraday_path = os.path.join(data_dir, "BHEL_1m.csv")
        intraday_data.to_csv(intraday_path)
        print(f"Saved 1-minute data to {intraday_path} ({len(intraday_data)} rows)")

if __name__ == "__main__":
    fetch_data()
