# =============================================================================
# CryptOracle - Feature Engineering Module
# Creates 60+ technical indicators, sentiment encoding, and time features
# =============================================================================

import logging
import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Price & Return Features
# ---------------------------------------------------------------------------
def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """Daily, log, 7-day, and 30-day returns."""
    df = df.copy()
    df["daily_return"]    = df["Close"].pct_change()
    df["log_return"]      = np.log(df["Close"] / df["Close"].shift(1))
    df["return_7d"]       = df["Close"].pct_change(7)
    df["return_30d"]      = df["Close"].pct_change(30)
    df["high_low_ratio"]  = df["High"] / df["Low"]
    df["close_open_ratio"]= df["Close"] / df["Open"]
    logger.debug("  + Return features added (6)")
    return df


# ---------------------------------------------------------------------------
# Moving Average Features
# ---------------------------------------------------------------------------
def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Simple and exponential moving averages with price ratios."""
    df = df.copy()
    # SMA
    for w in config.SMA_WINDOWS:
        sma_col = f"sma_{w}"
        df[sma_col]           = df["Close"].rolling(w).mean()
        df[f"price_sma{w}_ratio"] = df["Close"] / df[sma_col]

    # EMA
    for w in config.EMA_WINDOWS:
        ema_col = f"ema_{w}"
        df[ema_col]           = df["Close"].ewm(span=w, adjust=False).mean()
        df[f"price_ema{w}_ratio"] = df["Close"] / df[ema_col]

    # Golden / Death Cross signal
    df["sma7_above_sma21"] = (df["sma_7"] > df["sma_21"]).astype(int)
    logger.debug(f"  + Moving average features added ({2*len(config.SMA_WINDOWS) + 2*len(config.EMA_WINDOWS) + 1})")
    return df


# ---------------------------------------------------------------------------
# Volatility Features
# ---------------------------------------------------------------------------
def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling volatility, Parkinson estimator, and ATR."""
    df = df.copy()

    # Rolling historical volatility
    for w in config.VOLATILITY_WINDOWS:
        df[f"volatility_{w}d"] = df["log_return"].rolling(w).std() * np.sqrt(365)

    # Parkinson volatility (uses High-Low range — more efficient than close-to-close)
    df["parkinson_vol"] = np.sqrt(
        (1 / (4 * np.log(2))) *
        (np.log(df["High"] / df["Low"]) ** 2).rolling(config.ATR_WINDOW).mean()
    ) * np.sqrt(365)

    # Average True Range (ATR)
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"]  - prev_close).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(config.ATR_WINDOW).mean()
    df["atr_pct"] = df["atr"] / df["Close"]   # normalised ATR

    logger.debug(f"  + Volatility features added ({len(config.VOLATILITY_WINDOWS) + 3})")
    return df


# ---------------------------------------------------------------------------
# RSI - Relative Strength Index
# ---------------------------------------------------------------------------
def add_rsi(df: pd.DataFrame, window: int = config.RSI_WINDOW) -> pd.DataFrame:
    """RSI — momentum oscillator (0–100). Overbought > 70, Oversold < 30."""
    df = df.copy()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi_overbought"] = (df["rsi"] > 70).astype(int)
    df["rsi_oversold"]   = (df["rsi"] < 30).astype(int)
    logger.debug("  + RSI features added (3)")
    return df


# ---------------------------------------------------------------------------
# MACD - Moving Average Convergence Divergence
# ---------------------------------------------------------------------------
def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    """MACD line, signal line, and histogram."""
    df = df.copy()
    fast = config.MACD_FAST
    slow = config.MACD_SLOW
    sig  = config.MACD_SIGNAL

    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()

    df["macd"]        = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=sig, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]
    df["macd_cross"]  = (
        (df["macd"] > df["macd_signal"]) &
        (df["macd"].shift(1) <= df["macd_signal"].shift(1))
    ).astype(int)

    logger.debug("  + MACD features added (4)")
    return df


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------
def add_bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Upper/lower bands, band width, and %B position indicator."""
    df = df.copy()
    window = config.BOLLINGER_WINDOW
    n_std  = config.BOLLINGER_STD

    rolling_mean = df["Close"].rolling(window).mean()
    rolling_std  = df["Close"].rolling(window).std()

    df["bb_upper"]  = rolling_mean + n_std * rolling_std
    df["bb_lower"]  = rolling_mean - n_std * rolling_std
    df["bb_middle"] = rolling_mean
    df["bb_width"]  = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
    df["bb_pct_b"]  = (df["Close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    logger.debug("  + Bollinger Band features added (5)")
    return df


# ---------------------------------------------------------------------------
# Volume Features
# ---------------------------------------------------------------------------
def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Volume moving averages and On-Balance Volume (OBV)."""
    df = df.copy()

    df["volume_sma7"]   = df["Volume"].rolling(7).mean()
    df["volume_sma21"]  = df["Volume"].rolling(21).mean()
    df["volume_ratio"]  = df["Volume"] / df["volume_sma7"]

    # On-Balance Volume: running sum of signed volume
    obv = [0]
    for i in range(1, len(df)):
        if df["Close"].iloc[i] > df["Close"].iloc[i-1]:
            obv.append(obv[-1] + df["Volume"].iloc[i])
        elif df["Close"].iloc[i] < df["Close"].iloc[i-1]:
            obv.append(obv[-1] - df["Volume"].iloc[i])
        else:
            obv.append(obv[-1])
    df["obv"] = obv
    df["obv_ema"] = df["obv"].ewm(span=21).mean()

    logger.debug("  + Volume features added (5)")
    return df


# ---------------------------------------------------------------------------
# Momentum Features
# ---------------------------------------------------------------------------
def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Price momentum, ROC, and stochastic oscillator."""
    df = df.copy()
    w = config.MOMENTUM_WINDOW

    df["momentum"]   = df["Close"] - df["Close"].shift(w)
    df["roc"]        = df["Close"].pct_change(w) * 100  # Rate of Change

    # Stochastic Oscillator %K and %D
    low_min  = df["Low"].rolling(14).min()
    high_max = df["High"].rolling(14).max()
    df["stoch_k"] = 100 * (df["Close"] - low_min) / (high_max - low_min)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    logger.debug("  + Momentum features added (4)")
    return df


# ---------------------------------------------------------------------------
# Sentiment Features (Fear & Greed)
# ---------------------------------------------------------------------------
def add_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes Fear & Greed classification as ordinal integer.
    Extreme Fear=0, Fear=1, Neutral=2, Greed=3, Extreme Greed=4
    """
    df = df.copy()
    if "fear_greed_value" not in df.columns:
        logger.warning("Fear & Greed column missing — skipping sentiment features.")
        return df

    fg_map = {
        "Extreme Fear": 0,
        "Fear":         1,
        "Neutral":      2,
        "Greed":        3,
        "Extreme Greed":4,
    }
    df["fg_encoded"] = df["fear_greed_class"].map(fg_map).fillna(2)

    # Rolling sentiment trend
    df["fg_7d_avg"]  = df["fear_greed_value"].rolling(7).mean()
    df["fg_trend"]   = df["fear_greed_value"] - df["fg_7d_avg"]

    logger.debug("  + Sentiment features added (3)")
    return df


# ---------------------------------------------------------------------------
# On-Chain Features
# ---------------------------------------------------------------------------
def add_onchain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derived features from CoinGecko market cap / volume data."""
    df = df.copy()

    if "market_cap" in df.columns:
        df["market_cap_log"]    = np.log1p(df["market_cap"])
        df["mc_7d_change"]      = df["market_cap"].pct_change(7)

    if "cg_volume" in df.columns:
        df["cg_volume_log"]     = np.log1p(df["cg_volume"])
        df["price_vol_ratio"]   = df["Close"] / df["cg_volume"].replace(0, np.nan)

    logger.debug("  + On-chain features added")
    return df


# ---------------------------------------------------------------------------
# Time / Cyclical Features
# ---------------------------------------------------------------------------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Day-of-week, day-of-month, month, and cyclical sin/cos encodings.
    Cyclical encoding avoids the artificial jump between e.g. Sunday(6) → Monday(0).
    """
    df = df.copy()
    idx = pd.DatetimeIndex(df.index)

    df["day_of_week"]  = idx.dayofweek
    df["day_of_month"] = idx.day
    df["month"]        = idx.month
    df["quarter"]      = idx.quarter
    df["is_month_end"] = idx.is_month_end.astype(int)
    df["is_quarter_end"] = idx.is_quarter_end.astype(int)

    # Cyclical encodings
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    logger.debug("  + Time features added (10)")
    return df


# ---------------------------------------------------------------------------
# Master pipeline
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all feature engineering steps in sequence.

    Args:
        df: Raw merged DataFrame from data_collection.py

    Returns:
        Feature-rich DataFrame (NaN rows from rolling windows removed)
    """
    logger.info("Engineering features ...")
    n_cols_start = df.shape[1]

    df = add_return_features(df)
    df = add_moving_averages(df)
    df = add_volatility_features(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_volume_features(df)
    df = add_momentum_features(df)
    df = add_sentiment_features(df)
    df = add_onchain_features(df)
    df = add_time_features(df)

    # Drop string columns (can't go into model)
    str_cols = df.select_dtypes(include="object").columns.tolist()
    if str_cols:
        df.drop(columns=str_cols, inplace=True)

    # Remove rows that have NaN due to rolling window warm-up
    rows_before = len(df)
    df.dropna(inplace=True)
    rows_after = len(df)
    dropped = rows_before - rows_after

    n_cols_end = df.shape[1]
    logger.info(f"  ✓ Feature engineering complete.")
    logger.info(f"    Columns: {n_cols_start} → {n_cols_end} (+{n_cols_end - n_cols_start} features)")
    logger.info(f"    Rows dropped (NaN warm-up): {dropped}")
    logger.info(f"    Final shape: {df.shape}")

    return df


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_collection import collect_all_data

    raw_df = collect_all_data()
    feat_df = engineer_features(raw_df)

    save_path = f"{config.DATA_PROCESSED_DIR}/BTC_features.csv"
    feat_df.to_csv(save_path)
    print(f"\nSaved features → {save_path}")
    print(f"\nFeature list ({feat_df.shape[1]} cols):")
    for c in feat_df.columns:
        print(f"  {c}")
