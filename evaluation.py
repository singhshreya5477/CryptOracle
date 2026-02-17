# =============================================================================
# CryptOracle - Evaluation Module
# RMSE, MAE, MAPE, R², Directional Accuracy, baseline comparison
# =============================================================================

import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    label: str = "Model") -> dict:
    """
    Computes all evaluation metrics on real (USD) price arrays.

    Args:
        y_true: Actual closing prices (USD)
        y_pred: Predicted closing prices (USD)
        label:  Name label for logging

    Returns:
        Dictionary of metric name → value
    """
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    mae   = mean_absolute_error(y_true, y_pred)
    mape  = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    r2    = r2_score(y_true, y_pred)
    da    = directional_accuracy(y_true, y_pred)

    metrics = {
        "label": label,
        "RMSE":  round(rmse, 4),
        "MAE":   round(mae, 4),
        "MAPE":  round(mape, 4),
        "R2":    round(r2, 4),
        "Directional_Accuracy": round(da, 4),
    }

    logger.info(f"\n  [{label}] Evaluation Results:")
    logger.info(f"    RMSE:                 ${rmse:>10.2f}")
    logger.info(f"    MAE:                  ${mae:>10.2f}")
    logger.info(f"    MAPE:                  {mape:>9.2f}%")
    logger.info(f"    R²:                    {r2:>9.4f}")
    logger.info(f"    Directional Accuracy:  {da*100:>8.1f}%")

    return metrics


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Measures what % of days the model correctly predicted the direction (up/down).

    In real trading this matters more than exact price accuracy — knowing
    "it will go up tomorrow" is actionable even if the magnitude is off.

    Score of 0.5 = random guessing. Target: > 0.6
    """
    true_direction = np.diff(y_true)     # positive = price went up
    pred_direction = np.diff(y_pred)

    correct = np.sum(np.sign(true_direction) == np.sign(pred_direction))
    return correct / len(true_direction)


# ---------------------------------------------------------------------------
# Naive Baselines
# ---------------------------------------------------------------------------
def naive_persistence_predictions(y_true: np.ndarray) -> np.ndarray:
    """
    Naive "persistence" baseline: predict today's price = yesterday's price.
    This is the simplest possible model and a fair lower bound to beat.
    """
    return np.concatenate([[y_true[0]], y_true[:-1]])


def naive_mean_predictions(y_train: np.ndarray, n: int) -> np.ndarray:
    """Predict the training set mean for every test sample."""
    return np.full(n, np.mean(y_train))


def baseline_comparison(y_train_real, y_test_real, y_pred_real) -> pd.DataFrame:
    """
    Builds the comparison table shown in the README.

    Returns:
        DataFrame comparing Naive, Simple LSTM placeholder, and CryptOracle
    """
    persist_pred  = naive_persistence_predictions(y_test_real)
    mean_pred     = naive_mean_predictions(y_train_real, len(y_test_real))

    naive_metrics   = compute_metrics(y_test_real, persist_pred,  "Naive (Persistence)")
    mean_metrics    = compute_metrics(y_test_real, mean_pred,      "Naive (Mean)")
    model_metrics   = compute_metrics(y_test_real, y_pred_real,    "CryptOracle")

    comparison = pd.DataFrame([naive_metrics, mean_metrics, model_metrics])
    comparison = comparison.set_index("label")

    pct_improve = (
        (naive_metrics["RMSE"] - model_metrics["RMSE"]) /
        naive_metrics["RMSE"] * 100
    )
    logger.info(f"\n  ✓ Improvement over naive baseline (RMSE): {pct_improve:.1f}%")
    logger.info(f"\n{comparison.to_string()}")

    return comparison, pct_improve


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------
def plot_predictions(dates_test, y_true: np.ndarray, y_pred: np.ndarray,
                     symbol: str = "BTC", save_path: str = None):
    """
    Actual vs Predicted price plot with error band.
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle(f"CryptOracle — {symbol} Price Prediction", fontsize=16, fontweight="bold")

    # --- Top: Actual vs Predicted ---
    ax = axes[0]
    ax.plot(dates_test, y_true, label="Actual",    color="#F7931A", linewidth=2)
    ax.plot(dates_test, y_pred, label="Predicted", color="#627EEA", linewidth=2, linestyle="--")
    ax.fill_between(dates_test,
                    y_pred * 0.97, y_pred * 1.03,
                    alpha=0.15, color="#627EEA", label="±3% band")
    ax.set_title("Actual vs Predicted Close Price")
    ax.set_ylabel("Price (USD)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

    # --- Bottom: Prediction Error ---
    errors = y_pred - y_true
    ax2 = axes[1]
    ax2.bar(dates_test, errors, color=["#00C896" if e < 0 else "#FF4B4B" for e in errors],
            alpha=0.7, width=1)
    ax2.axhline(0, color="white", linewidth=1)
    ax2.set_title("Prediction Error (Predicted − Actual)")
    ax2.set_ylabel("Error (USD)")
    ax2.set_xlabel("Date")
    ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"  ✓ Prediction plot saved → {save_path}")
    plt.close()
    return fig


def plot_directional_accuracy(dates_test, y_true: np.ndarray, y_pred: np.ndarray,
                               save_path: str = None):
    """
    Shows which days the model got direction correct (green) vs wrong (red).
    """
    true_dir = np.diff(y_true)
    pred_dir = np.diff(y_pred)
    correct  = np.sign(true_dir) == np.sign(pred_dir)

    fig, ax = plt.subplots(figsize=(16, 4))
    colors = ["#00C896" if c else "#FF4B4B" for c in correct]
    ax.bar(dates_test[1:], np.abs(true_dir), color=colors, width=1, alpha=0.8)
    ax.set_title(f"Directional Accuracy — Green=Correct, Red=Wrong "
                 f"(Overall: {correct.mean()*100:.1f}%)")
    ax.set_ylabel("|Price Change| (USD)")
    ax.set_xlabel("Date")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"  ✓ Directional accuracy plot → {save_path}")
    plt.close()
    return fig


def plot_comparison_table(comparison_df: pd.DataFrame, save_path: str = None):
    """Renders the model comparison table as a matplotlib figure."""
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")

    table = ax.table(
        cellText=comparison_df.round(4).values,
        rowLabels=comparison_df.index,
        colLabels=comparison_df.columns,
        cellLoc="center", loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Highlight CryptOracle row
    for j in range(len(comparison_df.columns)):
        table[(3, j)].set_facecolor("#1A1F2E")
        table[(3, j)].set_text_props(color="white", fontweight="bold")

    fig.suptitle("Model Comparison — Test Set Performance", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"  ✓ Comparison table → {save_path}")
    plt.close()
    return fig


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(__file__))

    from data_collection import collect_all_data
    from feature_engineering import engineer_features
    from preprocessing import preprocess, inverse_transform_predictions
    from train import load_best_model

    raw_df  = collect_all_data()
    feat_df = engineer_features(raw_df)
    result  = preprocess(feat_df)

    (X_train, y_train, dates_train,
     X_val,   y_val,   dates_val,
     X_test,  y_test,  dates_test,
     scaler, meta) = result

    model = load_best_model()
    y_pred_scaled = model.predict(X_test).flatten()

    y_pred_real = inverse_transform_predictions(
        scaler, y_pred_scaled, meta["n_features"], meta["target_col_idx"])
    y_test_real = inverse_transform_predictions(
        scaler, y_test, meta["n_features"], meta["target_col_idx"])
    y_train_real = inverse_transform_predictions(
        scaler, y_train, meta["n_features"], meta["target_col_idx"])

    comparison, pct_improve = baseline_comparison(y_train_real, y_test_real, y_pred_real)

    plot_predictions(dates_test, y_test_real, y_pred_real,
                     save_path=f"{config.LOGS_DIR}/prediction_plot.png")
    plot_directional_accuracy(dates_test, y_test_real, y_pred_real,
                               save_path=f"{config.LOGS_DIR}/directional_accuracy.png")
    plot_comparison_table(comparison,
                          save_path=f"{config.LOGS_DIR}/model_comparison.png")

    print(f"\n✓ Evaluation complete. Improvement: {pct_improve:.1f}% over baseline.")
