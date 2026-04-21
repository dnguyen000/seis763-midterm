# =============================================================================
# data_fetch.py
# Stock Market Movement Prediction — Data Fetching
#
#
# Output files saved to /data folder:
#   SPY_raw.csv   - raw price data
#   TSLA_raw.csv  - raw price data
#   VIX_raw.csv   - VIX volatility index data
# =============================================================================

import yfinance as yf
import pandas as pd
import os


# CONFIGURATION

TICKERS    = ["SPY", "TSLA", "^VIX"]
START_DATE = "2015-01-01"
END_DATE   = "2024-12-31"
DATA_DIR   = "data"


# DOWNLOAD

def download_data(ticker, start=START_DATE, end=END_DATE):
    """
    Download daily OHLCV data from Yahoo Finance.
    Free, no API key, full 10 years available.
    Uses Adj Close which handles splits and dividends.
    Note: VIX has no volume — filled with 0 for consistent schema.
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

    # VIX has no volume or adj_close — fill with 0 for consistent schema
    if "volume" not in df.columns:
        df["volume"] = 0
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]

    # Keep only OHLCV + Adj Close
    cols = ["open", "high", "low", "close", "adj_close", "volume"]
    df   = df[[c for c in cols if c in df.columns]].copy()

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
    clean_name = ticker.replace("^", "")          # ^VIX → VIX
    path = os.path.join(DATA_DIR, f"{clean_name}_raw.csv")
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
        clean_name = ticker.replace("^", "")
        print(f"  {DATA_DIR}/{clean_name}_raw.csv")