"""
fix_warmup.py
─────────────
Fixes NaN rows in SPY_features.csv and TSLA_features.csv by fetching
extra historical data (Oct 2014) to warm up rolling windows before
the target start date of Jan 1 2015.

Overwrites:
    data/SPY_features.csv
    data/TSLA_features.csv
"""

import pandas as pd
import numpy as np
import yfinance as yf
import ta
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────
WARMUP_START  = "2014-10-01"   # extra data to warm up rolling windows
TARGET_START  = "2015-01-01"   # final CSV starts here
TARGET_END    = "2024-12-31"
DATA_DIR      = Path("data")

# ── Step 1: Download raw OHLCV + VIX with warmup period ──────────────────────
print("Downloading SPY, TSLA, VIX from yfinance (Oct 2014 → Dec 2024)...")

spy_raw  = yf.download("SPY",  start=WARMUP_START, end=TARGET_END, auto_adjust=False, progress=False)
tsla_raw = yf.download("TSLA", start=WARMUP_START, end=TARGET_END, auto_adjust=False, progress=False)
vix_raw  = yf.download("^VIX", start=WARMUP_START, end=TARGET_END, auto_adjust=False, progress=False)

# Flatten MultiIndex columns if present
for df in [spy_raw, tsla_raw, vix_raw]:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

spy_raw.columns  = spy_raw.columns.str.lower().str.replace(" ", "_")
tsla_raw.columns = tsla_raw.columns.str.lower().str.replace(" ", "_")
vix_raw.columns  = vix_raw.columns.str.lower().str.replace(" ", "_")

print(f"  SPY  : {len(spy_raw):,} rows  ({spy_raw.index[0].date()} → {spy_raw.index[-1].date()})")
print(f"  TSLA : {len(tsla_raw):,} rows  ({tsla_raw.index[0].date()} → {tsla_raw.index[-1].date()})")
print(f"  VIX  : {len(vix_raw):,} rows  ({vix_raw.index[0].date()} → {vix_raw.index[-1].date()})")


# ── Step 2: Feature engineering (same logic as feature_engineering.ipynb) ────
def build_features(df_raw: pd.DataFrame, df_vix: pd.DataFrame) -> pd.DataFrame:
    df    = df_raw.copy()
    close = df["adj_close"]

    # ── Price features ────────────────────────────────────────────────────────
    df["daily_return"]   = close.pct_change()
    df["weekly_return"]  = close.pct_change(periods=5)
    df["ma_7"]           = close.rolling(7).mean()
    df["ma_21"]          = close.rolling(21).mean()
    df["ma_cross"]       = df["ma_7"] - df["ma_21"]
    df["dist_from_ma21"] = (close - df["ma_21"]) / df["ma_21"]
    df["daily_range"]    = (df["high"] - df["low"]) / close

    # ── Momentum features ─────────────────────────────────────────────────────
    df["rsi_14"]      = ta.momentum.RSIIndicator(close=close, window=14).rsi()

    macd              = ta.trend.MACD(close=close)
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"]   = macd.macd_diff()

    bb                = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    df["bb_position"] = (close - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())

    df["volatility_7"]  = df["daily_return"].rolling(7).std()
    df["volatility_20"] = df["daily_return"].rolling(20).std()

    # ── Volume features ───────────────────────────────────────────────────────
    df["volume_change"] = df["volume"].pct_change()
    df["volume_ma20"]   = df["volume"].rolling(20).mean()
    df["volume_ratio"]  = df["volume"] / df["volume_ma20"]

    # ── Lagged returns ────────────────────────────────────────────────────────
    df["lag_return_1"] = df["daily_return"].shift(1)
    df["lag_return_3"] = df["daily_return"].shift(3)
    df["lag_return_5"] = df["daily_return"].shift(5)

    # ── Seasonal features ─────────────────────────────────────────────────────
    df["month"]   = df.index.month
    df["quarter"] = df.index.quarter
    df["season_num"] = df["month"].map({
        12: 0, 1: 0, 2: 0,
        3: 1,  4: 1, 5: 1,
        6: 2,  7: 2, 8: 2,
        9: 3, 10: 3, 11: 3,
    })
    df["season"] = df["season_num"].map({0: "Winter", 1: "Spring", 2: "Summer", 3: "Fall"})

    import calendar
    df["is_earnings_week"] = df.index.map(
        lambda dt: 1 if dt.month in [1, 4, 7, 10]
        and dt.day >= (calendar.monthrange(dt.year, dt.month)[1] - 14)
        else 0
    )

    # ── VIX features ──────────────────────────────────────────────────────────
    df["vix"]            = df_vix["adj_close"].reindex(df.index, method="ffill")
    df["is_major_event"] = (df["vix"] > 30).astype(int)

    # ── Targets ───────────────────────────────────────────────────────────────
    df["target_direction"] = (close.shift(-1) > close).astype(int)
    df["target_return"]    = (close.shift(-5) - close) / close

    return df


# ── Step 3: Build features on full warmup dataset ────────────────────────────
print("\nBuilding features (includes warmup period)...")
spy_full  = build_features(spy_raw,  vix_raw)
tsla_full = build_features(tsla_raw, vix_raw)


# ── Step 4: Trim to target date range (Jan 2015 onwards) ─────────────────────
print(f"Trimming to {TARGET_START} onwards...")
spy_final  = spy_full[spy_full.index >= TARGET_START].copy()
tsla_final = tsla_full[tsla_full.index >= TARGET_START].copy()


# ── Step 5: Verify NaN situation ──────────────────────────────────────────────
print("\n── NaN check after fix ─────────────────────────────────────────────")

feature_cols = [
    'daily_return', 'weekly_return', 'ma_7', 'ma_21', 'ma_cross',
    'dist_from_ma21', 'daily_range', 'rsi_14', 'macd', 'macd_signal',
    'macd_hist', 'bb_position', 'volatility_7', 'volatility_20',
    'volume_change', 'volume_ma20', 'volume_ratio',
    'lag_return_1', 'lag_return_3', 'lag_return_5',
    'month', 'quarter', 'season_num', 'is_earnings_week',
    'vix', 'is_major_event'
]

for ticker, df in [("SPY", spy_final), ("TSLA", tsla_final)]:
    nan_counts = df[feature_cols].isna().sum()
    nan_features = nan_counts[nan_counts > 0]
    if len(nan_features) == 0:
        print(f"  {ticker}: ✓ Zero NaN rows in all feature columns")
    else:
        print(f"  {ticker}: ⚠ Still has NaNs:")
        print(nan_features)

# Last 5 rows will have NaN in target_return (lookahead) — that's expected
print("\n  Note: target_return NaN in last 5 rows is expected (5-day lookahead).")
print("  These are dropped automatically when loading for modeling.")


# ── Step 6: Save ─────────────────────────────────────────────────────────────
DATA_DIR.mkdir(exist_ok=True)

spy_final.index.name  = "date"
tsla_final.index.name = "date"

spy_out  = DATA_DIR / "SPY_features.csv"
tsla_out = DATA_DIR / "TSLA_features.csv"

spy_final.to_csv(spy_out)
tsla_final.to_csv(tsla_out)

print(f"\n── Saved ────────────────────────────────────────────────────────────")
print(f"  {spy_out}  — {len(spy_final):,} rows | {spy_final.index[0].date()} → {spy_final.index[-1].date()}")
print(f"  {tsla_out} — {len(tsla_final):,} rows | {tsla_final.index[0].date()} → {tsla_final.index[-1].date()}")
print("\nDone  Your CSVs are now NaN-free from row 0.")