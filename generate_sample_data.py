# =============================================================================
# Sample Data Generator for CryptOracle
# Use this if APIs are unavailable - generates realistic synthetic data
# =============================================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_synthetic_crypto_data(
    start_date='2020-01-01',
    end_date='2024-01-01',
    initial_price=7000,
    save_path=None
):
    """
    Generates synthetic cryptocurrency price data with realistic patterns.
    
    Includes:
    - Price trends (bull/bear markets)
    - Volatility clustering
    - Fear & Greed Index
    - Simulated market cap data
    """
    
    # Generate date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n = len(dates)
    
    print(f"Generating {n} days of synthetic BTC data...")
    
    # Generate price with trend + noise
    np.random.seed(42)
    
    # Base trend (multiple regimes)
    trend = np.zeros(n)
    trend[:n//3] = np.linspace(0, 1.5, n//3)  # Bull market
    trend[n//3:2*n//3] = 1.5 + np.random.normal(0, 0.1, n//3)  # Sideways
    trend[2*n//3:] = np.linspace(1.5, 2.0, n - 2*n//3)  # Another bull
    
    # Volatility (GARCH-like clustering)
    volatility = np.zeros(n)
    volatility[0] = 0.02
    for i in range(1, n):
        volatility[i] = 0.015 + 0.85 * volatility[i-1] + 0.1 * np.random.randn()**2
    
    # Generate log returns
    log_returns = trend / 100 + volatility * np.random.randn(n)
    
    # Convert to prices
    close_prices = initial_price * np.exp(np.cumsum(log_returns))
    
    # Generate OHLCV
    high_prices = close_prices * (1 + np.abs(np.random.randn(n)) * 0.02)
    low_prices = close_prices * (1 - np.abs(np.random.randn(n)) * 0.02)
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = initial_price
    
    volume = np.random.lognormal(mean=20, sigma=0.5, size=n) * 1e6
    
    # Generate Fear & Greed Index (correlates with returns)
    fear_greed_base = 50 + 30 * np.tanh(np.cumsum(log_returns) / 2)
    fear_greed_value = np.clip(
        fear_greed_base + np.random.normal(0, 5, n),
        0, 100
    ).astype(int)
    
    # Classify Fear & Greed
    def classify_fg(val):
        if val < 25:
            return "Extreme Fear"
        elif val < 45:
            return "Fear"
        elif val < 55:
            return "Neutral"
        elif val < 75:
            return "Greed"
        else:
            return "Extreme Greed"
    
    fear_greed_class = [classify_fg(v) for v in fear_greed_value]
    
    # Generate market cap (proportional to price with some noise)
    market_cap = close_prices * 19e6 * (1 + np.random.normal(0, 0.05, n))
    
    # CoinGecko-style volume
    cg_volume = volume * close_prices * (1 + np.random.normal(0, 0.1, n))
    
    # Create DataFrame
    df = pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volume,
        'fear_greed_value': fear_greed_value,
        'fear_greed_class': fear_greed_class,
        'market_cap': market_cap,
        'cg_volume': cg_volume,
    }, index=dates)
    
    df.index.name = 'Date'
    
    # Save if path provided
    if save_path:
        df.to_csv(save_path)
        print(f"✓ Saved synthetic data to {save_path}")
    
    print(f"✓ Generated data shape: {df.shape}")
    print(f"  Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
    print(f"  Fear & Greed range: {df['fear_greed_value'].min()} - {df['fear_greed_value'].max()}")
    
    return df


if __name__ == "__main__":
    # Generate BTC data
    output_dir = os.path.join(os.path.dirname(__file__), 'data', 'raw')
    os.makedirs(output_dir, exist_ok=True)
    
    btc_path = os.path.join(output_dir, 'BTC_raw.csv')
    df_btc = generate_synthetic_crypto_data(
        start_date='2020-01-01',
        end_date='2024-12-31',
        initial_price=7000,
        save_path=btc_path
    )
    
    print("\n" + "="*60)
    print("Sample of generated data:")
    print("="*60)
    print(df_btc.head(10))
    print("\nData info:")
    print(df_btc.info())
