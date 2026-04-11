# =============================================================================
# data_fetch.py
# Stock Market Movement Prediction — Data Fetching
#
# Usage:
#   /opt/anaconda3/envs/seis631/bin/python data_fetch.py
#
# Output files saved to /data folder:
#   SPY_raw.csv   - raw price data
#   TSLA_raw.csv  - raw price data
# =============================================================================

import yfinance as yf
import pandas as pd
import os


# CONFIGURATION

TICKERS    = ["SPY", "TSLA"]
START_DATE = "2015-01-01"
END_DATE   = "2024-12-31"
DATA_DIR   = "data"


# DOWNLOAD

def download_data(ticker, start=START_DATE, end=END_DATE):
    """
    Download daily OHLCV data from Yahoo Finance.
    Free, no API key, full 10 years available.
    Uses Adj Close which handles splits and dividends.
    """
    print(f"  Downloading {ticker} ({start} to {end})...")

    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Rename columns to lowercase
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    # Handle adj close naming
    if "adj close" in df.columns:
        df.rename(columns={"adj close": "adj_close"}, inplace=True)

    # Keep only OHLCV + Adj Close
    cols = ["open", "high", "low", "close", "adj_close", "volume"]
    df   = df[[c for c in cols if c in df.columns]].copy()

    # Clean up index
    df.index      = pd.to_datetime(df.index)
    df.index.name = "date"
    df.sort_index(inplace=True)

    print(f"  Downloaded {len(df)} rows")
    print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Missing values: {df.isnull().sum().sum()}")

    return df

# SAVE

def save_data(df, ticker):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{ticker}_raw.csv")
    df.to_csv(path)
    print(f"  Saved to {path}")



# MAIN

if __name__ == "__main__":

    print("=" * 60)
    print("  Stock Market Movement Prediction")
    print("  Data Download")
    print("  Source: Yahoo Finance (yfinance)")
    print(f"  Tickers: {TICKERS}")
    print(f"  Period:  {START_DATE} to {END_DATE}")
    print("=" * 60)

    for ticker in TICKERS:
        print(f"\n--- {ticker} ---")
        df = download_data(ticker)
        save_data(df, ticker)

    print(f"\n{'='*60}")
    print("ALL DONE")
    print(f"{'='*60}")
    print("\nFiles saved:")
    for ticker in TICKERS:
        print(f"  data/{ticker}_raw.csv")

   