# =============================================================================
# CryptOracle - Model Architecture
# Bidirectional LSTM with custom Attention Mechanism
# =============================================================================

import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom Attention Layer
# ---------------------------------------------------------------------------
class AttentionLayer(layers.Layer):
    """
    Additive (Bahdanau-style) attention over LSTM hidden states.

    Instead of just using the final LSTM hidden state, attention learns a
    weighted combination of ALL timestep outputs — so the model can focus on
    whichever past days are most informative for the prediction.

    This is directly inspired by "Attention Is All You Need" (Vaswani, 2017),
    adapted from NLP to time-series forecasting.

    Architecture:
        score_i = V · tanh(W · h_i + b)    for each timestep i
        alpha   = softmax(scores)           attention weights (sum to 1)
        context = Σ alpha_i · h_i           weighted context vector
    """

    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W = layers.Dense(units)          # transforms each hidden state
        self.V = layers.Dense(1)              # scores each transformed state

    def call(self, lstm_output):
        """
        Args:
            lstm_output: Tensor of shape (batch, timesteps, hidden_dim)

        Returns:
            context:     Tensor of shape (batch, hidden_dim) — attended summary
            weights:     Tensor of shape (batch, timesteps, 1) — for visualisation
        """
        # Score each timestep: (batch, timesteps, 1)
        score = self.V(tf.nn.tanh(self.W(lstm_output)))

        # Attention weights via softmax: (batch, timesteps, 1)
        weights = tf.nn.softmax(score, axis=1)

        # Weighted sum over timesteps: (batch, hidden_dim)
        context = tf.reduce_sum(weights * lstm_output, axis=1)

        return context, weights

    def get_config(self):
        config_dict = super().get_config()
        config_dict.update({"units": self.units})
        return config_dict


# ---------------------------------------------------------------------------
# Model Builder
# ---------------------------------------------------------------------------
def build_model(input_shape: tuple,
                lstm_units_1: int  = config.LSTM_UNITS_1,
                lstm_units_2: int  = config.LSTM_UNITS_2,
                attention_units: int = config.ATTENTION_UNITS,
                dense_units: int   = config.DENSE_UNITS,
                dropout_rate: float = config.DROPOUT_RATE,
                learning_rate: float = config.LEARNING_RATE) -> Model:
    """
    Builds the CryptOracle Bidirectional LSTM + Attention model.

    Input shape: (sequence_length, n_features)

    Architecture:
        Input
          └─ Bidirectional LSTM (128) + Dropout       ← reads sequence forward & backward
          └─ Bidirectional LSTM (64, return_sequences) ← feeds timestep outputs to attention
          └─ AttentionLayer (32)                       ← learns which days matter most
          └─ Dense (32, ReLU) + Dropout
          └─ Dense (1, linear)                         ← tomorrow's price (scaled)

    Why Bidirectional?
        Forward LSTM sees: day1 → day2 → ... → day60
        Backward LSTM sees: day60 → day59 → ... → day1
        Combining both captures patterns that are easier to detect from either direction.
        (e.g., a recovery pattern looks different forwards vs backwards)
    """
    logger.info("Building model architecture ...")
    logger.info(f"  Input shape: {input_shape}")

    inputs = keras.Input(shape=input_shape, name="input_sequence")

    # --- Layer 1: Bidirectional LSTM (return_sequences=True for layer 2) ---
    x = layers.Bidirectional(
        layers.LSTM(lstm_units_1, return_sequences=True,
                    kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        name="bilstm_1"
    )(inputs)
    x = layers.Dropout(dropout_rate, name="dropout_1")(x)

    # --- Layer 2: Bidirectional LSTM (return_sequences=True for attention) ---
    x = layers.Bidirectional(
        layers.LSTM(lstm_units_2, return_sequences=True,
                    kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        name="bilstm_2"
    )(x)
    x = layers.Dropout(dropout_rate, name="dropout_2")(x)

    # --- Attention ---
    context, attention_weights = AttentionLayer(attention_units, name="attention")(x)

    # --- Dense head ---
    x = layers.Dense(dense_units, activation="relu",
                      kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                      name="dense_1")(context)
    x = layers.Dropout(dropout_rate, name="dropout_3")(x)

    outputs = layers.Dense(1, activation="linear", name="output")(x)

    # --- Compile ---
    model = Model(inputs=inputs, outputs=outputs, name="CryptOracle")
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="huber",   # Huber: robust to outliers vs MSE
                  metrics=["mae"])

    logger.info(f"  ✓ Model built. Parameters: {model.count_params():,}")
    model.summary(print_fn=logger.info)

    return model


# ---------------------------------------------------------------------------
# Baseline model for comparison
# ---------------------------------------------------------------------------
def build_naive_baseline(y_train: np.ndarray) -> float:
    """
    Naive baseline: always predict yesterday's price (random walk assumption).
    This is what we need to beat — if our LSTM can't outperform this, it's useless!
    """
    # The naive prediction for any day = the previous day's value
    # For evaluation we just return a constant (mean shift) as the baseline strategy
    return float(np.mean(y_train))


def build_simple_lstm(input_shape: tuple) -> Model:
    """
    Simple single-layer LSTM for ablation comparison.
    Shows the improvement from adding Bidirectionality + Attention.
    """
    inputs = keras.Input(shape=input_shape, name="input")
    x = layers.LSTM(64, name="lstm_simple")(inputs)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="linear", name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="SimpleLSTM")
    model.compile(optimizer=keras.optimizers.Adam(config.LEARNING_RATE),
                  loss="huber", metrics=["mae"])
    return model


if __name__ == "__main__":
    # Quick architecture test
    seq_len    = config.SEQUENCE_LENGTH
    n_features = 50   # approximate

    model = build_model(input_shape=(seq_len, n_features))
    print(f"\nModel output shape for batch=4: "
          f"{model.predict(np.zeros((4, seq_len, n_features))).shape}")
