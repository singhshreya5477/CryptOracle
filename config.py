# =============================================================================
# CryptOracle - Configuration File
# Central place for all project settings
# =============================================================================

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models", "saved_models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Create directories if they don't exist
for d in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# Data Collection Settings
# ---------------------------------------------------------------------------
# Cryptocurrencies to analyze (Yahoo Finance tickers)
CRYPTO_SYMBOLS = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
}

# Primary crypto for modeling
PRIMARY_CRYPTO = "BTC"

# Historical data range
DATA_START_DATE = "2020-01-01"
DATA_END_DATE = None  # None = today

# Fear & Greed API
FEAR_GREED_API = "https://api.alternative.me/fng/?limit=1000&format=json"

# CoinGecko API (no key needed)
COINGECKO_BASE = "https://api.coingecko.com/api/v3"
COINGECKO_COIN_ID = "bitcoin"  # for primary crypto

# ---------------------------------------------------------------------------
# Feature Engineering Settings
# ---------------------------------------------------------------------------
# Technical indicator windows
SMA_WINDOWS = [7, 21, 50]
EMA_WINDOWS = [12, 26]
VOLATILITY_WINDOWS = [7, 30]
RSI_WINDOW = 14
BOLLINGER_WINDOW = 20
BOLLINGER_STD = 2
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
MOMENTUM_WINDOW = 10
ATR_WINDOW = 14

# ---------------------------------------------------------------------------
# Preprocessing Settings
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH = 60        # Look-back window (60 days)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15           # Remaining goes to test
TARGET_COLUMN = "Close"     # What we're predicting
SCALE_RANGE = (0, 1)        # MinMaxScaler range

# ---------------------------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------------------------
LSTM_UNITS_1 = 128          # First Bidirectional LSTM layer
LSTM_UNITS_2 = 64           # Second Bidirectional LSTM layer
ATTENTION_UNITS = 32        # Attention mechanism units
DENSE_UNITS = 32            # Dense layer units
DROPOUT_RATE = 0.2

# ---------------------------------------------------------------------------
# Training Settings
# ---------------------------------------------------------------------------
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 15
LR_REDUCE_PATIENCE = 8
LR_REDUCE_FACTOR = 0.5
MIN_LR = 1e-6

# ---------------------------------------------------------------------------
# Advanced Features
# ---------------------------------------------------------------------------
# Anomaly Detection
ANOMALY_CONTAMINATION = 0.05    # Expected % of anomalies

# Regime Detection
N_REGIMES = 3                   # Bull, Bear, Sideways
REGIME_LABELS = {
    0: "Bear Market 🐻",
    1: "Sideways Market ➡️",
    2: "Bull Market 🐂"
}

# Monte Carlo
MONTE_CARLO_RUNS = 200
MONTE_CARLO_DAYS = 30           # Days to simulate forward

# ---------------------------------------------------------------------------
# Dashboard Settings
# ---------------------------------------------------------------------------
DASHBOARD_TITLE = "CryptOracle — Sentiment-Aware Forecasting"
DASHBOARD_PORT = 8501
THEME_COLORS = {
    "primary": "#F7931A",       # Bitcoin orange
    "secondary": "#627EEA",     # Ethereum purple
    "success": "#00C896",
    "danger": "#FF4B4B",
    "neutral": "#7F8C8D",
    "background": "#0E1117",
    "surface": "#1A1F2E",
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = "INFO"
LOG_FILE = os.path.join(LOGS_DIR, "cryptoracle.log")
