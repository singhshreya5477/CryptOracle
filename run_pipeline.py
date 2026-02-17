# =============================================================================
# CryptOracle - Master Pipeline
# Run this single script to execute the complete project end-to-end
# Usage: python run_pipeline.py
# =============================================================================

import os
import sys
import json
import logging
import numpy as np
import pandas as pd

import config

# Configure logging to both console and file
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.LOG_FILE, mode="w", encoding='utf-8'),
    ]
)
logger = logging.getLogger(__name__)


def print_banner():
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║        🔮  C R Y P T O R A C L E  🔮                        ║
║                                                              ║
║   Sentiment-Aware Multi-Signal Crypto Forecasting Engine     ║
║   Bidirectional LSTM + Custom Attention Mechanism            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def step(n: int, title: str):
    print(f"\n{'='*62}")
    print(f"  STEP {n}: {title}")
    print(f"{'='*62}")


def run_pipeline(symbol: str = config.PRIMARY_CRYPTO,
                 skip_collection: bool = False,
                 ablation: bool = False):
    """
    Runs the full CryptOracle pipeline:
        1. Data collection (3 sources)
        2. Feature engineering (60+ features)
        3. Preprocessing (scaling + sequences)
        4. Model training (BiLSTM + Attention)
        5. Evaluation + baseline comparison
        6. Advanced features (anomaly, regime, Monte Carlo)

    Args:
        symbol:          Crypto to analyse (from config.CRYPTO_SYMBOLS)
        skip_collection: If True, loads cached raw data (faster reruns)
        ablation:        If True, also trains simple LSTM for comparison
    """
    print_banner()
    logger.info(f"Starting CryptOracle pipeline for {symbol}")

    # ── Step 1: Data Collection ──────────────────────────────────────────────
    step(1, "DATA COLLECTION")
    from data_collection import collect_all_data

    raw_path = f"{config.DATA_RAW_DIR}/{symbol}_raw.csv"
    if skip_collection and os.path.exists(raw_path):
        logger.info(f"  Loading cached raw data from {raw_path}")
        raw_df = pd.read_csv(raw_path, index_col="Date", parse_dates=True)
        logger.info(f"  ✓ Loaded {len(raw_df)} rows")
    else:
        raw_df = collect_all_data(symbol)

    # ── Step 2: Feature Engineering ──────────────────────────────────────────
    step(2, "FEATURE ENGINEERING")
    from feature_engineering import engineer_features

    feat_path = f"{config.DATA_PROCESSED_DIR}/{symbol}_features.csv"
    if skip_collection and os.path.exists(feat_path):
        logger.info(f"  Loading cached features from {feat_path}")
        feat_df = pd.read_csv(feat_path, index_col="Date", parse_dates=True)
    else:
        feat_df = engineer_features(raw_df)
        feat_df.to_csv(feat_path)
        logger.info(f"  ✓ Features saved → {feat_path}")

    # ── Step 3: Preprocessing ─────────────────────────────────────────────────
    step(3, "PREPROCESSING")
    from preprocessing import preprocess, inverse_transform_predictions

    result = preprocess(feat_df, symbol)
    (X_train, y_train, dates_train,
     X_val,   y_val,   dates_val,
     X_test,  y_test,  dates_test,
     scaler, meta) = result

    input_shape = (X_train.shape[1], X_train.shape[2])
    logger.info(f"  Input shape for model: {input_shape}")

    # ── Step 4: Training ──────────────────────────────────────────────────────
    step(4, "MODEL TRAINING")
    from model import build_model, build_simple_lstm
    from train import train_model, plot_training_history, save_training_results

    # Optional ablation: train simple LSTM first
    if ablation:
        logger.info("  Running ablation study (Simple LSTM) ...")
        simple_model = build_simple_lstm(input_shape)
        simple_history = train_model(simple_model, X_train, y_train,
                                     X_val, y_val, model_name="simple_lstm")
        y_pred_simple_scaled = simple_model.predict(X_test).flatten()

    # Train main CryptOracle model
    model = build_model(input_shape)
    history = train_model(model, X_train, y_train, X_val, y_val)
    plot_training_history(history)
    save_training_results(history)

    # ── Step 5: Evaluation ────────────────────────────────────────────────────
    step(5, "EVALUATION")
    from evaluation import (compute_metrics, baseline_comparison,
                             plot_predictions, plot_directional_accuracy,
                             plot_comparison_table)

    y_pred_scaled = model.predict(X_test).flatten()

    # Inverse transform to USD
    y_pred_real  = inverse_transform_predictions(scaler, y_pred_scaled,
                                                  meta["n_features"], meta["target_col_idx"])
    y_test_real  = inverse_transform_predictions(scaler, y_test,
                                                  meta["n_features"], meta["target_col_idx"])
    y_train_real = inverse_transform_predictions(scaler, y_train,
                                                  meta["n_features"], meta["target_col_idx"])

    # Save predictions for dashboard
    pred_df = pd.DataFrame({
        "actual":    y_test_real,
        "predicted": y_pred_real
    }, index=dates_test)
    pred_df.index.name = "Date"
    pred_df.to_csv(f"{config.LOGS_DIR}/prediction_results.csv")

    # Compute metrics
    metrics = compute_metrics(y_test_real, y_pred_real, "CryptOracle")
    comparison, pct_improve = baseline_comparison(y_train_real, y_test_real, y_pred_real)

    # Ablation comparison
    if ablation:
        y_pred_simple = inverse_transform_predictions(
            scaler, y_pred_simple_scaled,
            meta["n_features"], meta["target_col_idx"])
        simple_metrics = compute_metrics(y_test_real, y_pred_simple, "Simple LSTM")

    # Generate plots
    plot_predictions(dates_test, y_test_real, y_pred_real,
                     save_path=f"{config.LOGS_DIR}/prediction_plot.png")
    plot_directional_accuracy(dates_test, y_test_real, y_pred_real,
                               save_path=f"{config.LOGS_DIR}/directional_accuracy.png")
    plot_comparison_table(comparison,
                          save_path=f"{config.LOGS_DIR}/model_comparison.png")

    # ── Step 6: Advanced Features ─────────────────────────────────────────────
    step(6, "ADVANCED FEATURES")
    from advanced_features import (detect_anomalies, detect_regimes,
                                    plot_anomalies, plot_regimes,
                                    monte_carlo_simulation, plot_monte_carlo)

    # Anomaly detection on the full feature set
    feat_df_adv = detect_anomalies(feat_df)
    plot_anomalies(feat_df_adv, save_path=f"{config.LOGS_DIR}/anomalies.png")

    feat_df_adv = detect_regimes(feat_df_adv)
    plot_regimes(feat_df_adv, save_path=f"{config.LOGS_DIR}/regimes.png")

    # Monte Carlo on the latest window
    last_seq = X_test[-1:].copy()
    mc_results = monte_carlo_simulation(model, last_seq, scaler,
                                        meta["n_features"], meta["target_col_idx"])
    plot_monte_carlo(mc_results, y_test_real, dates_test,
                     save_path=f"{config.LOGS_DIR}/monte_carlo.png")

    # ── Summary ───────────────────────────────────────────────────────────────
    step("✓", "PIPELINE COMPLETE")

    summary = {
        "symbol":             symbol,
        "n_features":         meta["n_features"],
        "test_rmse":          metrics["RMSE"],
        "test_mae":           metrics["MAE"],
        "test_mape":          metrics["MAPE"],
        "test_r2":            metrics["R2"],
        "directional_acc":    metrics["Directional_Accuracy"],
        "baseline_improvement_pct": round(pct_improve, 2),
        "mc_median_30d":      round(float(mc_results["median"][-1]), 2),
        "mc_lower_5_30d":     round(float(mc_results["lower_5"][-1]), 2),
        "mc_upper_95_30d":    round(float(mc_results["upper_95"][-1]), 2),
        "n_anomalies":        int(feat_df_adv["is_anomaly"].sum()),
        "regime_distribution": feat_df_adv["regime_label"].value_counts().to_dict(),
    }

    summary_path = f"{config.LOGS_DIR}/pipeline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*62)
    print("  📊 RESULTS SUMMARY")
    print("="*62)
    print(f"  RMSE:                 ${summary['test_rmse']:>10,.2f}")
    print(f"  MAE:                  ${summary['test_mae']:>10,.2f}")
    print(f"  MAPE:                  {summary['test_mape']:>9.2f}%")
    print(f"  R²:                    {summary['test_r2']:>9.4f}")
    print(f"  Directional Accuracy:  {summary['directional_acc']*100:>8.1f}%")
    print(f"  vs Naive Baseline:    +{summary['baseline_improvement_pct']:.1f}% improvement")
    print(f"  30-Day Forecast:      ${summary['mc_median_30d']:>10,.0f} (median)")
    print(f"  Anomalies Detected:    {summary['n_anomalies']}")
    print("="*62)
    print(f"\n  All plots saved to: {config.LOGS_DIR}")
    print(f"  Summary saved to:   {summary_path}")
    print("\n  🚀 Launch dashboard:  streamlit run dashboard.py")
    print("="*62)

    return model, summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CryptOracle Pipeline")
    parser.add_argument("--symbol", default=config.PRIMARY_CRYPTO,
                        choices=list(config.CRYPTO_SYMBOLS.keys()),
                        help="Cryptocurrency to analyse")
    parser.add_argument("--skip-collection", action="store_true",
                        help="Skip API calls and use cached data")
    parser.add_argument("--ablation", action="store_true",
                        help="Also train simple LSTM for comparison")
    args = parser.parse_args()

    run_pipeline(
        symbol=args.symbol,
        skip_collection=args.skip_collection,
        ablation=args.ablation
    )
