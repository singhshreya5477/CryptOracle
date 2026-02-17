# =============================================================================
# CryptOracle - Data Collection Module
# Fetches data from 3 sources: yfinance, Fear & Greed API, CoinGecko
# =============================================================================

import time
import logging
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

import config

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL),
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: Retry logic for flaky APIs (learned this the hard way — CoinGecko
# kept timing out without it)
# ---------------------------------------------------------------------------
def fetch_with_retry(url, params=None, max_retries=4, backoff=2.0):
    """
    Exponential backoff retry for HTTP GET requests.
    Waits 2s, 4s, 8s, 16s between attempts.
    """
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            wait = backoff ** attempt
            logger.warning(f"Request failed (attempt {attempt+1}/{max_retries}): {e}. "
                           f"Retrying in {wait:.0f}s...")
            time.sleep(wait)
    logger.error(f"All {max_retries} attempts failed for URL: {url}")
    return None


# ---------------------------------------------------------------------------
# Source 1: Historical OHLCV data via yfinance
# ---------------------------------------------------------------------------
def fetch_price_data(symbol: str, start: str, end: str = None) -> pd.DataFrame:
    """
    Downloads daily OHLCV data from Yahoo Finance.

    Args:
        symbol: Yahoo Finance ticker, e.g. 'BTC-USD'
        start:  Start date string 'YYYY-MM-DD'
        end:    End date string (None = today)

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
    """
    logger.info(f"Fetching price data for {symbol} from {start} ...")
    end = end or datetime.today().strftime("%Y-%m-%d")

    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end, interval="1d")

    if df.empty:
        raise ValueError(f"No data returned for {symbol}. Check the ticker symbol.")

    # Keep only OHLCV columns, drop timezone from index
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"

    # Sanity checks
    df.dropna(inplace=True)
    logger.info(f"  ✓ Price data: {len(df)} rows  ({df.index[0].date()} → {df.index[-1].date()})")
    return df


# ---------------------------------------------------------------------------
# Source 2: Fear & Greed Index via Alternative.me (free, no API key needed)
# ---------------------------------------------------------------------------
def fetch_fear_greed_index() -> pd.DataFrame:
    """
    Fetches the Crypto Fear & Greed Index history.
    Score 0–100: 0 = Extreme Fear, 100 = Extreme Greed.

    Returns:
        DataFrame indexed by Date with columns: fear_greed_value, fear_greed_class
    """
    logger.info("Fetching Fear & Greed Index ...")
    data = fetch_with_retry(config.FEAR_GREED_API)

    if data is None or "data" not in data:
        logger.warning("Fear & Greed API unavailable — will skip this feature.")
        return pd.DataFrame()

    records = []
    for entry in data["data"]:
        records.append({
            "Date": pd.to_datetime(int(entry["timestamp"]), unit="s"),
            "fear_greed_value": float(entry["value"]),
            "fear_greed_class": entry["value_classification"],
        })

    fg_df = pd.DataFrame(records).set_index("Date")
    fg_df.index = fg_df.index.tz_localize(None).normalize()
    fg_df = fg_df.sort_index()

    logger.info(f"  ✓ Fear & Greed: {len(fg_df)} rows  "
                f"({fg_df.index[0].date()} → {fg_df.index[-1].date()})")
    return fg_df


# ---------------------------------------------------------------------------
# Source 3: On-chain / market metrics via CoinGecko (free tier)
# ---------------------------------------------------------------------------
def fetch_coingecko_metrics(coin_id: str, days: int = 1460) -> pd.DataFrame:
    """
    Fetches market cap and total volume history from CoinGecko.
    Adds 1s delay between requests to respect rate limits.

    Args:
        coin_id: CoinGecko coin ID, e.g. 'bitcoin'
        days:    Number of days of history (max free tier ~1460)

    Returns:
        DataFrame with market_cap and total_volume columns
    """
    logger.info(f"Fetching CoinGecko metrics for '{coin_id}' ...")
    url = f"{config.COINGECKO_BASE}/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}

    # CoinGecko rate-limits aggressively on the free tier — always add delay
    time.sleep(1.5)
    data = fetch_with_retry(url, params=params)

    if data is None:
        logger.warning("CoinGecko unavailable — will skip on-chain metrics.")
        return pd.DataFrame()

    def _parse_series(raw, col_name):
        df = pd.DataFrame(raw, columns=["timestamp", col_name])
        df["Date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.normalize()
        return df.set_index("Date")[col_name]

    market_cap_s = _parse_series(data.get("market_caps", []), "market_cap")
    volume_s = _parse_series(data.get("total_volumes", []), "cg_volume")

    cg_df = pd.concat([market_cap_s, volume_s], axis=1)
    cg_df.index = cg_df.index.tz_localize(None)
    cg_df = cg_df.sort_index()

    logger.info(f"  ✓ CoinGecko: {len(cg_df)} rows  "
                f"({cg_df.index[0].date()} → {cg_df.index[-1].date()})")
    return cg_df


# ---------------------------------------------------------------------------
# Merge all sources into one aligned DataFrame
# ---------------------------------------------------------------------------
def merge_all_sources(price_df: pd.DataFrame,
                      fg_df: pd.DataFrame,
                      cg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-joins price data with auxiliary signals.
    Missing rows filled with forward-fill then backward-fill.
    (Fear & Greed doesn't update on weekends — that's expected.)
    """
    logger.info("Merging data sources ...")
    merged = price_df.copy()

    if not fg_df.empty:
        merged = merged.join(fg_df, how="left")

    if not cg_df.empty:
        merged = merged.join(cg_df, how="left")

    # Forward-fill gaps (e.g., weekends), then back-fill the start
    merged.ffill(inplace=True)
    merged.bfill(inplace=True)

    missing = merged.isnull().sum()
    if missing.any():
        logger.warning(f"Remaining NaNs after fill:\n{missing[missing > 0]}")

    logger.info(f"  ✓ Merged dataset: {merged.shape[0]} rows × {merged.shape[1]} cols")
    return merged


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def collect_all_data(symbol_key: str = config.PRIMARY_CRYPTO) -> pd.DataFrame:
    """
    Full data collection pipeline for a given crypto symbol key.

    Args:
        symbol_key: Key in config.CRYPTO_SYMBOLS, e.g. 'BTC'

    Returns:
        Merged DataFrame saved to data/raw/
    """
    ticker = config.CRYPTO_SYMBOLS[symbol_key]
    coin_id = config.COINGECKO_COIN_ID   # update config for ETH etc.

    price_df = fetch_price_data(ticker, config.DATA_START_DATE, config.DATA_END_DATE)
    fg_df    = fetch_fear_greed_index()
    cg_df    = fetch_coingecko_metrics(coin_id)

    merged = merge_all_sources(price_df, fg_df, cg_df)

    # Save raw merged data
    save_path = f"{config.DATA_RAW_DIR}/{symbol_key}_raw.csv"
    merged.to_csv(save_path)
    logger.info(f"  ✓ Saved raw data → {save_path}")

    return merged


if __name__ == "__main__":
    df = collect_all_data()
    print("\n--- Sample (first 5 rows) ---")
    print(df.head())
    print("\n--- Data types ---")
    print(df.dtypes)
    print(f"\nShape: {df.shape}")
