# =============================================================================
# CryptOracle - Advanced Features
# Anomaly Detection, Regime Classification, Monte Carlo Simulation
# =============================================================================

import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import config

logger = logging.getLogger(__name__)


# =============================================================================
# 1. ANOMALY DETECTION — Isolation Forest
# =============================================================================
def detect_anomalies(df: pd.DataFrame,
                     contamination: float = config.ANOMALY_CONTAMINATION) -> pd.DataFrame:
    """
    Uses Isolation Forest to detect unusual price movements.
    Isolation Forest works by randomly partitioning data — anomalies
    (unusual price jumps, flash crashes) are easier to isolate and
    therefore get lower anomaly scores.

    Args:
        df:            Feature DataFrame (must contain 'Close' and 'daily_return')
        contamination: Expected proportion of anomalies (default 5%)

    Returns:
        df with new columns: 'anomaly_score', 'is_anomaly'
    """
    logger.info("Running Anomaly Detection (Isolation Forest) ...")
    df = df.copy()

    # Features for anomaly detection: price, return, volume, volatility
    anomaly_cols = [c for c in ["Close", "daily_return", "Volume",
                                 "volatility_7d", "volatility_30d",
                                 "atr_pct", "rsi"]
                    if c in df.columns]

    X_anomaly = df[anomaly_cols].fillna(0).values
    X_scaled  = StandardScaler().fit_transform(X_anomaly)

    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42
    )
    iso.fit(X_scaled)

    # score_samples: more negative = more anomalous
    df["anomaly_score"] = iso.score_samples(X_scaled)
    df["is_anomaly"]    = (iso.predict(X_scaled) == -1).astype(int)

    n_anomalies = df["is_anomaly"].sum()
    logger.info(f"  ✓ Detected {n_anomalies} anomalies "
                f"({n_anomalies/len(df)*100:.1f}% of data)")

    return df


def plot_anomalies(df: pd.DataFrame, save_path: str = None):
    """Price chart with anomaly dates marked in red."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    # Price with anomaly markers
    ax1 = axes[0]
    ax1.plot(df.index, df["Close"], color="#F7931A", linewidth=1.5, label="Close Price")
    anomalies = df[df["is_anomaly"] == 1]
    ax1.scatter(anomalies.index, anomalies["Close"],
                color="#FF4B4B", s=60, zorder=5, label=f"Anomaly ({len(anomalies)})")
    ax1.set_title("BTC Price with Anomaly Detection")
    ax1.set_ylabel("Price (USD)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Anomaly score over time
    ax2 = axes[1]
    ax2.fill_between(df.index, df["anomaly_score"],
                     alpha=0.6, color="#627EEA", label="Anomaly Score")
    ax2.axhline(df["anomaly_score"].quantile(0.05), color="#FF4B4B",
                linestyle="--", label="Anomaly threshold")
    ax2.set_title("Isolation Forest Anomaly Score (lower = more anomalous)")
    ax2.set_ylabel("Score")
    ax2.set_xlabel("Date")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"  ✓ Anomaly plot → {save_path}")
    plt.close()
    return fig


# =============================================================================
# 2. REGIME DETECTION — K-Means Clustering
# =============================================================================
def detect_regimes(df: pd.DataFrame, n_regimes: int = config.N_REGIMES) -> pd.DataFrame:
    """
    Classifies each trading day into a market regime:
        0 = Bear Market (downtrend)
        1 = Sideways Market (consolidation)
        2 = Bull Market (uptrend)

    Uses K-Means on return, volatility, and momentum features.
    The regime labels are then matched to their actual meaning by
    sorting clusters by average return.

    This allows training regime-specific strategies or giving users
    a market context indicator on the dashboard.
    """
    logger.info("Running Regime Detection (K-Means) ...")
    df = df.copy()

    regime_cols = [c for c in ["daily_return", "volatility_30d",
                                "momentum", "rsi", "return_7d",
                                "return_30d"]
                   if c in df.columns]

    X_regime = df[regime_cols].fillna(0).values
    X_scaled = StandardScaler().fit_transform(X_regime)

    # KMeans with multiple init attempts for stability
    km = KMeans(n_clusters=n_regimes, random_state=42, n_init=20, max_iter=500)
    raw_labels = km.fit_predict(X_scaled)

    # Sort cluster IDs by mean return so 0=bear, 1=sideways, 2=bull
    cluster_returns = {}
    for c in range(n_regimes):
        mask = raw_labels == c
        cluster_returns[c] = df.loc[mask, "daily_return"].mean() if "daily_return" in df.columns else 0

    sorted_clusters = sorted(cluster_returns, key=cluster_returns.get)
    label_map = {old: new for new, old in enumerate(sorted_clusters)}
    df["regime"] = np.vectorize(label_map.get)(raw_labels)
    df["regime_label"] = df["regime"].map(config.REGIME_LABELS)

    # Log regime distribution
    for r in range(n_regimes):
        pct = (df["regime"] == r).mean() * 100
        logger.info(f"    {config.REGIME_LABELS[r]}: {pct:.1f}% of days")

    logger.info(f"  ✓ Regime detection complete.")
    return df


def plot_regimes(df: pd.DataFrame, save_path: str = None):
    """Price chart with regime-coloured background bands."""
    regime_colours = {
        0: "#FF4B4B",    # Bear — red
        1: "#7F8C8D",    # Sideways — grey
        2: "#00C896",    # Bull — green
    }

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(df.index, df["Close"], color="white", linewidth=1.5, zorder=3)

    # Colour bands per regime
    if "regime" in df.columns:
        for regime_id, colour in regime_colours.items():
            mask = df["regime"] == regime_id
            ax.fill_between(df.index, df["Close"].min(), df["Close"].max(),
                            where=mask, alpha=0.15, color=colour,
                            label=config.REGIME_LABELS[regime_id])

    ax.set_title("BTC Market Regime Classification (K-Means)")
    ax.set_ylabel("Price (USD)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_facecolor("#0E1117")
    fig.patch.set_facecolor("#0E1117")
    ax.tick_params(colors="white")
    ax.yaxis.label.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.title.set_color("white")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"  ✓ Regime plot → {save_path}")
    plt.close()
    return fig


# =============================================================================
# 3. MONTE CARLO SIMULATION — Prediction Uncertainty Cones
# =============================================================================
def monte_carlo_simulation(model,
                            last_sequence: np.ndarray,
                            scaler,
                            n_features: int,
                            target_col_idx: int,
                            n_runs: int = config.MONTE_CARLO_RUNS,
                            n_days: int = config.MONTE_CARLO_DAYS) -> dict:
    """
    Generates multiple forward-looking price paths by:
      1. Taking the last known 60-day sequence
      2. Running the model forward n_days using its own predictions as input
      3. Adding small random noise to each run to capture uncertainty

    This produces a "probability cone" like weather forecast uncertainty bands,
    which is far more informative than a single point prediction.

    Args:
        model:           Trained Keras model
        last_sequence:   np.array of shape (1, seq_len, n_features) — latest window
        scaler:          Fitted MinMaxScaler
        n_features:      Total feature count
        target_col_idx:  Index of Close column
        n_runs:          Number of simulation paths (default 200)
        n_days:          Days to simulate forward (default 30)

    Returns:
        dict with 'paths', 'mean', 'lower_5', 'upper_95', 'lower_25', 'upper_75'
        all as USD arrays of length n_days
    """
    logger.info(f"Running Monte Carlo ({n_runs} paths × {n_days} days) ...")

    from preprocessing import inverse_transform_predictions

    paths = np.zeros((n_runs, n_days))
    seq   = last_sequence.copy()   # shape (1, seq_len, n_features)

    # Noise level: ~1% of the current Close value in scaled space
    noise_scale = 0.008

    for run in range(n_runs):
        run_seq = seq.copy()
        for day in range(n_days):
            pred_scaled = model.predict(run_seq, verbose=0)[0, 0]
            noise = np.random.normal(0, noise_scale)
            pred_noisy = np.clip(pred_scaled + noise, 0, 1)

            # Build next timestep: copy last step, update Close column
            next_step = run_seq[0, -1, :].copy()
            next_step[target_col_idx] = pred_noisy

            # Shift window forward
            run_seq = np.concatenate([
                run_seq[:, 1:, :],
                next_step.reshape(1, 1, -1)
            ], axis=1)

            paths[run, day] = pred_noisy

    # Convert all paths to USD
    paths_usd = np.zeros_like(paths)
    for run in range(n_runs):
        paths_usd[run] = inverse_transform_predictions(
            scaler, paths[run], n_features, target_col_idx)

    results = {
        "paths":     paths_usd,
        "mean":      np.mean(paths_usd, axis=0),
        "median":    np.median(paths_usd, axis=0),
        "lower_5":   np.percentile(paths_usd, 5,  axis=0),
        "upper_95":  np.percentile(paths_usd, 95, axis=0),
        "lower_25":  np.percentile(paths_usd, 25, axis=0),
        "upper_75":  np.percentile(paths_usd, 75, axis=0),
    }

    logger.info(f"  ✓ Monte Carlo complete.")
    logger.info(f"    30-day median forecast:  ${results['median'][-1]:,.0f}")
    logger.info(f"    90% confidence range:    "
                f"${results['lower_5'][-1]:,.0f} – ${results['upper_95'][-1]:,.0f}")

    return results


def plot_monte_carlo(mc_results: dict, last_known_prices: np.ndarray,
                     last_known_dates, symbol: str = "BTC",
                     save_path: str = None):
    """
    Beautiful probability cone chart.
    Shows 200 individual paths (faint), mean forecast, and confidence bands.
    """
    import pandas as pd
    n_days = len(mc_results["mean"])
    future_dates = pd.date_range(
        start=last_known_dates[-1], periods=n_days + 1, freq="D")[1:]

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_facecolor("#0E1117")
    fig.patch.set_facecolor("#0E1117")

    # Historical prices
    ax.plot(last_known_dates[-90:], last_known_prices[-90:],
            color="#F7931A", linewidth=2, label="Historical Price", zorder=4)

    # Individual paths (very faint)
    for path in mc_results["paths"][:50]:   # show 50 of 200 for clarity
        ax.plot(future_dates, path, color="#627EEA", alpha=0.05, linewidth=0.8)

    # Confidence bands
    ax.fill_between(future_dates,
                    mc_results["lower_5"], mc_results["upper_95"],
                    alpha=0.15, color="#627EEA", label="90% Confidence Band")
    ax.fill_between(future_dates,
                    mc_results["lower_25"], mc_results["upper_75"],
                    alpha=0.30, color="#627EEA", label="50% Confidence Band")

    # Mean forecast
    ax.plot(future_dates, mc_results["mean"],
            color="#00C896", linewidth=2.5, linestyle="--", label="Mean Forecast", zorder=5)
    ax.plot(future_dates, mc_results["median"],
            color="white", linewidth=1.5, linestyle=":", label="Median Forecast", zorder=5)

    # Divider line
    ax.axvline(last_known_dates[-1], color="#FFFFFF", linestyle="--",
               alpha=0.5, linewidth=1)
    ax.text(last_known_dates[-1], ax.get_ylim()[1] * 0.98,
            " Forecast →", color="white", fontsize=10)

    ax.set_title(f"CryptOracle — {symbol} 30-Day Monte Carlo Forecast ({n_days} paths)",
                 color="white", fontsize=14)
    ax.set_ylabel("Price (USD)", color="white")
    ax.set_xlabel("Date", color="white")
    ax.tick_params(colors="white")
    ax.legend(loc="upper left", facecolor="#1A1F2E", labelcolor="white")
    ax.grid(alpha=0.2, color="white")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"  ✓ Monte Carlo plot → {save_path}")
    plt.close()
    return fig


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(__file__))

    from data_collection import collect_all_data
    from feature_engineering import engineer_features

    raw_df  = collect_all_data()
    feat_df = engineer_features(raw_df)

    # Anomaly detection
    feat_df = detect_anomalies(feat_df)
    plot_anomalies(feat_df, save_path=f"{config.LOGS_DIR}/anomalies.png")

    # Regime detection
    feat_df = detect_regimes(feat_df)
    plot_regimes(feat_df, save_path=f"{config.LOGS_DIR}/regimes.png")

    print("✓ Advanced features complete.")
    print(f"  Anomalies found: {feat_df['is_anomaly'].sum()}")
    print(f"\nRegime distribution:\n{feat_df['regime_label'].value_counts()}")
