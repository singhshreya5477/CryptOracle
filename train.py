# =============================================================================
# CryptOracle - Training Pipeline
# Full training with callbacks, logging, and model persistence
# =============================================================================

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from tensorflow import keras

import config
from model import build_model, build_simple_lstm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
def get_callbacks(model_name: str = "cryptoracle"):
    """
    Training callbacks:
    - EarlyStopping:    stop if val_loss doesn't improve for 15 epochs
    - ReduceLROnPlateau: halve LR if val_loss stagnates for 8 epochs
    - ModelCheckpoint:  save best weights automatically
    - TensorBoard:      optional logging for visualisation
    """
    checkpoint_path = os.path.join(config.MODELS_DIR, f"{model_name}_best.keras")

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.LR_REDUCE_FACTOR,
            patience=config.LR_REDUCE_PATIENCE,
            min_lr=config.MIN_LR,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=0
        ),
    ]
    return callbacks, checkpoint_path


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------
def train_model(model,
                X_train: np.ndarray, y_train: np.ndarray,
                X_val:   np.ndarray, y_val:   np.ndarray,
                model_name: str = "cryptoracle") -> dict:
    """
    Train the model with all callbacks.

    Returns:
        history_dict: Training and validation loss/mae per epoch
    """
    logger.info("=" * 55)
    logger.info(f"TRAINING: {model_name.upper()}")
    logger.info("=" * 55)
    logger.info(f"  Train samples: {len(X_train):,}")
    logger.info(f"  Val samples:   {len(X_val):,}")
    logger.info(f"  Epochs (max):  {config.EPOCHS}")
    logger.info(f"  Batch size:    {config.BATCH_SIZE}")

    callbacks, ckpt_path = get_callbacks(model_name)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
        shuffle=False      # NEVER shuffle time-series training data
    )

    best_val_loss = min(history.history["val_loss"])
    best_epoch    = np.argmin(history.history["val_loss"]) + 1
    logger.info(f"\n  ✓ Training complete.")
    logger.info(f"    Best val_loss: {best_val_loss:.6f} at epoch {best_epoch}")
    logger.info(f"    Model saved → {ckpt_path}")

    return history.history


# ---------------------------------------------------------------------------
# Ablation study: compare Simple LSTM vs CryptOracle
# ---------------------------------------------------------------------------
def run_ablation(X_train, y_train, X_val, y_val, input_shape):
    """
    Trains both a simple LSTM and the full CryptOracle model.
    Useful for the comparison table in README — shows the improvement.
    """
    logger.info("\n--- ABLATION: Simple LSTM ---")
    simple = build_simple_lstm(input_shape)
    simple_history = train_model(simple, X_train, y_train,
                                 X_val, y_val, model_name="simple_lstm")

    logger.info("\n--- ABLATION: CryptOracle (BiLSTM + Attention) ---")
    full = build_model(input_shape)
    full_history = train_model(full, X_train, y_train,
                               X_val, y_val, model_name="cryptoracle")

    return simple, simple_history, full, full_history


# ---------------------------------------------------------------------------
# Plot training curves
# ---------------------------------------------------------------------------
def plot_training_history(history: dict, model_name: str = "cryptoracle"):
    """Saves a training/validation loss curve plot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Training History — {model_name}", fontsize=14, fontweight="bold")

    # Loss
    axes[0].plot(history["loss"],     label="Train Loss",  color="#F7931A", linewidth=2)
    axes[0].plot(history["val_loss"], label="Val Loss",    color="#627EEA", linewidth=2)
    axes[0].set_title("Loss (Huber)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # MAE
    axes[1].plot(history["mae"],     label="Train MAE",  color="#F7931A", linewidth=2)
    axes[1].plot(history["val_mae"], label="Val MAE",    color="#627EEA", linewidth=2)
    axes[1].set_title("Mean Absolute Error")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(config.LOGS_DIR, f"{model_name}_training_curve.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  ✓ Training curve saved → {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# Save training results to JSON (for README stats)
# ---------------------------------------------------------------------------
def save_training_results(history: dict, model_name: str = "cryptoracle"):
    results = {
        "model_name":     model_name,
        "trained_at":     datetime.now().isoformat(),
        "total_epochs":   len(history["loss"]),
        "best_val_loss":  float(min(history["val_loss"])),
        "best_val_mae":   float(min(history["val_mae"])),
        "final_train_loss": float(history["loss"][-1]),
    }
    save_path = os.path.join(config.LOGS_DIR, f"{model_name}_training_results.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"  ✓ Training results saved → {save_path}")
    return results


# ---------------------------------------------------------------------------
# Load saved model
# ---------------------------------------------------------------------------
def load_best_model(model_name: str = "cryptoracle"):
    """Load the best checkpoint saved during training."""
    checkpoint_path = os.path.join(config.MODELS_DIR, f"{model_name}_best.keras")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No saved model at {checkpoint_path}. Run training first.")

    from model import AttentionLayer
    model = keras.models.load_model(
        checkpoint_path,
        custom_objects={"AttentionLayer": AttentionLayer}
    )
    logger.info(f"✓ Loaded model from {checkpoint_path}")
    return model


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    from data_collection import collect_all_data
    from feature_engineering import engineer_features
    from preprocessing import preprocess

    # Full pipeline
    raw_df  = collect_all_data()
    feat_df = engineer_features(raw_df)
    result  = preprocess(feat_df)

    (X_train, y_train, dates_train,
     X_val,   y_val,   dates_val,
     X_test,  y_test,  dates_test,
     scaler, meta) = result

    input_shape = (X_train.shape[1], X_train.shape[2])

    # Train main model
    model = build_model(input_shape)
    history = train_model(model, X_train, y_train, X_val, y_val)
    plot_training_history(history)
    save_training_results(history)

    print("\n✓ Training complete! Run evaluation.py next.")
