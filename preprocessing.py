# =============================================================================
# CryptOracle - Preprocessing Module
# Scaling, sequence creation, and train/val/test splitting
# =============================================================================

import logging
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scale features
# ---------------------------------------------------------------------------
def scale_features(df: pd.DataFrame, scaler_path: str = None):
    """
    Fit MinMaxScaler on all numeric columns.
    Each feature scaled independently to avoid leakage between them.

    Args:
        df:          Feature DataFrame
        scaler_path: If provided, save fitted scaler to this path

    Returns:
        scaled_df:   DataFrame with same shape, values in [0, 1]
        scaler:      Fitted MinMaxScaler (needed for inverse transform later)
    """
    logger.info("Scaling features ...")
    scaler = MinMaxScaler(feature_range=config.SCALE_RANGE)
    scaled_values = scaler.fit_transform(df.values)
    scaled_df = pd.DataFrame(scaled_values, index=df.index, columns=df.columns)

    if scaler_path:
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        logger.info(f"  ✓ Scaler saved → {scaler_path}")

    logger.info(f"  ✓ Scaling complete. Shape: {scaled_df.shape}")
    return scaled_df, scaler


def load_scaler(scaler_path: str):
    """Load a previously saved scaler from disk."""
    with open(scaler_path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Create sequences (sliding window)
# ---------------------------------------------------------------------------
def create_sequences(scaled_df: pd.DataFrame,
                     sequence_length: int = config.SEQUENCE_LENGTH,
                     target_col: str = config.TARGET_COLUMN):
    """
    Creates overlapping look-back sequences for LSTM input.

    Critical: target is the NEXT day's closing price (index+1), NOT the
    current day. Getting this wrong causes future data leakage — I wasted
    a day debugging this mistake in early experiments!

    Args:
        scaled_df:       Scaled DataFrame (all features)
        sequence_length: Number of past days per sample (default 60)
        target_col:      Column to predict

    Returns:
        X:  np.array of shape (n_samples, sequence_length, n_features)
        y:  np.array of shape (n_samples,)  — next-day scaled close
        dates: list of dates corresponding to the prediction day
    """
    logger.info(f"Creating sequences (look-back={sequence_length}) ...")

    if target_col not in scaled_df.columns:
        raise ValueError(f"Target column '{target_col}' not in DataFrame.")

    target_idx = scaled_df.columns.get_loc(target_col)
    values = scaled_df.values
    dates  = scaled_df.index.tolist()

    X, y, pred_dates = [], [], []

    for i in range(sequence_length, len(values) - 1):
        X.append(values[i - sequence_length: i])        # past 60 days (all features)
        y.append(values[i + 1][target_idx])             # NEXT day's close
        pred_dates.append(dates[i + 1])                 # date of what we're predicting

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    logger.info(f"  ✓ X shape: {X.shape}  |  y shape: {y.shape}")
    return X, y, pred_dates


# ---------------------------------------------------------------------------
# Train / Validation / Test Split
# ---------------------------------------------------------------------------
def split_data(X: np.ndarray, y: np.ndarray, dates: list):
    """
    Chronological split — no random shuffling (this is time-series data!).
    Random splits would leak future information into training.

    Split ratios from config: 70% train, 15% val, 15% test
    """
    n = len(X)
    train_end = int(n * config.TRAIN_RATIO)
    val_end   = int(n * (config.TRAIN_RATIO + config.VAL_RATIO))

    X_train, y_train = X[:train_end],     y[:train_end]
    X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
    X_test,  y_test  = X[val_end:],       y[val_end:]

    dates_train = dates[:train_end]
    dates_val   = dates[train_end:val_end]
    dates_test  = dates[val_end:]

    logger.info("  Data split (chronological, NO shuffle):")
    logger.info(f"    Train: {X_train.shape[0]} samples  "
                f"({dates_train[0].strftime('%Y-%m-%d')} → {dates_train[-1].strftime('%Y-%m-%d')})")
    logger.info(f"    Val:   {X_val.shape[0]} samples  "
                f"({dates_val[0].strftime('%Y-%m-%d')} → {dates_val[-1].strftime('%Y-%m-%d')})")
    logger.info(f"    Test:  {X_test.shape[0]} samples  "
                f"({dates_test[0].strftime('%Y-%m-%d')} → {dates_test[-1].strftime('%Y-%m-%d')})")

    return (X_train, y_train, dates_train,
            X_val,   y_val,   dates_val,
            X_test,  y_test,  dates_test)


# ---------------------------------------------------------------------------
# Inverse transform predictions back to real price
# ---------------------------------------------------------------------------
def inverse_transform_predictions(scaler, predictions: np.ndarray,
                                   n_features: int,
                                   target_col_idx: int) -> np.ndarray:
    """
    MinMaxScaler was fit on ALL features together, so to inverse-transform
    just the Close price we need to reconstruct a dummy full-feature array.

    Args:
        scaler:         Fitted MinMaxScaler
        predictions:    1D array of scaled predictions
        n_features:     Total number of features in original DataFrame
        target_col_idx: Column index of 'Close' in the scaled DataFrame

    Returns:
        Real-price predictions (USD)
    """
    dummy = np.zeros((len(predictions), n_features), dtype=np.float32)
    dummy[:, target_col_idx] = predictions
    real_prices = scaler.inverse_transform(dummy)[:, target_col_idx]
    return real_prices


# ---------------------------------------------------------------------------
# Master preprocessing pipeline
# ---------------------------------------------------------------------------
def preprocess(feat_df: pd.DataFrame, symbol: str = config.PRIMARY_CRYPTO):
    """
    Full preprocessing pipeline:
      1. Scale
      2. Create sequences
      3. Split chronologically

    Returns all splits plus scaler and metadata for later use.
    """
    logger.info("=" * 55)
    logger.info("PREPROCESSING PIPELINE")
    logger.info("=" * 55)

    scaler_path = f"{config.DATA_PROCESSED_DIR}/{symbol}_scaler.pkl"
    scaled_df, scaler = scale_features(feat_df, scaler_path=scaler_path)

    X, y, dates = create_sequences(scaled_df)

    splits = split_data(X, y, dates)
    X_train, y_train, dates_train = splits[0], splits[1], splits[2]
    X_val,   y_val,   dates_val   = splits[3], splits[4], splits[5]
    X_test,  y_test,  dates_test  = splits[6], splits[7], splits[8]

    # Metadata needed for inverse transform
    n_features     = scaled_df.shape[1]
    target_col_idx = scaled_df.columns.get_loc(config.TARGET_COLUMN)

    meta = {
        "n_features":     n_features,
        "target_col_idx": target_col_idx,
        "feature_names":  list(scaled_df.columns),
        "scaler_path":    scaler_path,
    }

    logger.info(f"\n  ✓ Preprocessing complete. n_features={n_features}")

    return (X_train, y_train, dates_train,
            X_val,   y_val,   dates_val,
            X_test,  y_test,  dates_test,
            scaler, meta)


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_collection import collect_all_data
    from feature_engineering import engineer_features

    raw_df  = collect_all_data()
    feat_df = engineer_features(raw_df)
    result  = preprocess(feat_df)

    X_train, y_train, dates_train = result[0], result[1], result[2]
    print(f"\nX_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"First prediction date: {dates_train[0]}")
