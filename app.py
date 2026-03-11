from flask import Flask, request, jsonify
import os
import json
import math
import traceback
from collections import deque

import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.getenv("MODEL_PATH", "model/lstm_autoencoder.h5")
SCALER_PATH = os.getenv("SCALER_PATH", "model/scaler.pkl")
CONFIG_PATH = os.getenv("CONFIG_PATH", "model/config.json")

EXPECTED_WINDOW = int(os.getenv("EXPECTED_WINDOW", "20"))
EXPECTED_FEATURES = int(os.getenv("EXPECTED_FEATURES", "3"))

# Keeps recent reconstruction errors in memory for simple RUL estimation
ERROR_HISTORY_SIZE = int(os.getenv("ERROR_HISTORY_SIZE", "50"))
error_history = deque(maxlen=ERROR_HISTORY_SIZE)

model = None
scaler = None
config = None


# ============================================================
# DEFAULT CONFIG
# ============================================================

DEFAULT_CONFIG = {
    "feature_names": ["current", "temperature", "vibration"],

    # Thresholds from healthy validation data
    # You should replace these with your real values from training
    "thresholds": {
        "normal": 0.010,
        "warning": 0.020,
        "critical": 0.050
    },

    # Feature weights (optional)
    "feature_weights": {
        "current": 1.0,
        "temperature": 1.0,
        "vibration": 1.0
    },

    # RUL settings
    "rul": {
        "max_hours": 100.0,
        "min_hours": 0.0
    }
}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def load_json_config(path: str):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_CONFIG


def safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def clamp(value, low, high):
    return max(low, min(high, value))


def validate_input(readings):
    """
    Validates that incoming data is shaped like:
    [
      [current, temperature, vibration],
      ...
    ]
    with EXPECTED_WINDOW rows and EXPECTED_FEATURES columns.
    """
    if not isinstance(readings, list):
        return False, "Payload field 'readings' must be a list."

    if len(readings) != EXPECTED_WINDOW:
        return False, f"'readings' must contain exactly {EXPECTED_WINDOW} rows."

    for i, row in enumerate(readings):
        if not isinstance(row, list):
            return False, f"Row {i} is not a list."
        if len(row) != EXPECTED_FEATURES:
            return False, f"Row {i} must contain exactly {EXPECTED_FEATURES} values."

        for j, value in enumerate(row):
            try:
                float(value)
            except Exception:
                return False, f"Value at row {i}, column {j} is not numeric."

    return True, "OK"


def to_numpy_2d(readings):
    arr = np.array(readings, dtype=np.float32)
    return arr


def scale_window(window_2d: np.ndarray):
    """
    window_2d shape: (20, 3)
    scaler expects 2D shape: (N, features)
    """
    scaled = scaler.transform(window_2d)
    return scaled


def reshape_for_model(window_scaled: np.ndarray):
    """
    LSTM autoencoder expects: (batch, timesteps, features)
    """
    return np.expand_dims(window_scaled, axis=0)


def compute_reconstruction(model_input: np.ndarray):
    reconstructed = model.predict(model_input, verbose=0)
    return reconstructed


def compute_errors(original_scaled: np.ndarray, reconstructed_scaled: np.ndarray):
    """
    original_scaled shape: (1, 20, 3)
    reconstructed_scaled shape: (1, 20, 3)
    """
    diff = original_scaled - reconstructed_scaled
    squared = np.square(diff)

    # Total MSE over whole window
    total_mse = float(np.mean(squared))

    # Per feature MSE over time
    feature_mse = np.mean(squared[0], axis=0)  # shape (3,)

    # Per time step MSE over features
    timestep_mse = np.mean(squared[0], axis=1)  # shape (20,)

    return total_mse, feature_mse, timestep_mse


def compute_health(error_value: float, thresholds: dict):
    """
    Convert reconstruction error to health percentage.

    Logic:
    - <= normal threshold => 100%
    - >= critical threshold => 0%
    - linear interpolation in between
    """
    t_normal = safe_float(thresholds.get("normal", 0.01), 0.01)
    t_critical = safe_float(thresholds.get("critical", 0.05), 0.05)

    if t_critical <= t_normal:
        t_critical = t_normal + 1e-6

    if error_value <= t_normal:
        return 100.0
    if error_value >= t_critical:
        return 0.0

    health = 100.0 * (1.0 - (error_value - t_normal) / (t_critical - t_normal))
    return float(clamp(health, 0.0, 100.0))


def classify_status(error_value: float, thresholds: dict):
    t_warning = safe_float(thresholds.get("warning", 0.02), 0.02)
    t_critical = safe_float(thresholds.get("critical", 0.05), 0.05)

    if error_value >= t_critical:
        return "critical"
    elif error_value >= t_warning:
        return "warning"
    else:
        return "normal"


def compute_sensor_contributions(feature_mse: np.ndarray, cfg: dict):
    """
    Convert per-feature reconstruction errors into percentages.
    """
    feature_names = cfg.get("feature_names", ["current", "temperature", "vibration"])
    feature_weights = cfg.get("feature_weights", {})

    weighted = []
    for i, fname in enumerate(feature_names):
        weight = safe_float(feature_weights.get(fname, 1.0), 1.0)
        weighted.append(float(feature_mse[i]) * weight)

    weighted = np.array(weighted, dtype=np.float32)
    total = float(np.sum(weighted))

    if total <= 1e-12:
        contributions = {fname: 0.0 for fname in feature_names}
        main_cause = "unknown"
        return contributions, main_cause

    percentages = (weighted / total) * 100.0
    contributions = {
        feature_names[i]: float(percentages[i]) for i in range(len(feature_names))
    }

    main_index = int(np.argmax(percentages))
    main_cause = feature_names[main_index]

    return contributions, main_cause


def estimate_rul_hours(current_error: float, thresholds: dict):
    """
    Very simple engineering estimate of RUL based on:
    - severity of current error
    - error trend from recent windows

    This is not true physics-based life prediction.
    It is a health-derived estimate for dashboard use.
    """
    rul_cfg = config.get("rul", {})
    max_hours = safe_float(rul_cfg.get("max_hours", 100.0), 100.0)
    min_hours = safe_float(rul_cfg.get("min_hours", 0.0), 0.0)

    t_normal = safe_float(thresholds.get("normal", 0.01), 0.01)
    t_critical = safe_float(thresholds.get("critical", 0.05), 0.05)

    if current_error <= t_normal:
        return max_hours

    if current_error >= t_critical:
        return 0.0

    # Severity term
    severity_ratio = (current_error - t_normal) / max((t_critical - t_normal), 1e-6)
    severity_ratio = clamp(severity_ratio, 0.0, 1.0)

    # Trend term using recent error slope
    trend_penalty = 0.0
    if len(error_history) >= 5:
        xs = np.arange(len(error_history), dtype=np.float32)
        ys = np.array(error_history, dtype=np.float32)

        # Simple linear slope
        slope = np.polyfit(xs, ys, 1)[0]

        # Positive slope => degrading
        if slope > 0:
            # Scale slope penalty conservatively
            trend_penalty = min(float(slope * 500.0), 0.5)

    effective_ratio = clamp(severity_ratio + trend_penalty, 0.0, 1.0)
    rul = max_hours * (1.0 - effective_ratio)

    return float(clamp(rul, min_hours, max_hours))


def format_response(
    latest_raw_values: np.ndarray,
    total_error: float,
    health: float,
    contributions: dict,
    main_cause: str,
    rul: float,
    status: str
):
    is_anomaly = status in ["warning", "critical"]

    return {
        "is_anomaly": bool(is_anomaly),
        "status": status,
        "health": round(float(health), 2),
        "rul": round(float(rul), 2),
        "reconstruction_error": round(float(total_error), 6),

        "sensor_contributions": {
            "current": round(float(contributions.get("current", 0.0)), 2),
            "temperature": round(float(contributions.get("temperature", 0.0)), 2),
            "vibration": round(float(contributions.get("vibration", 0.0)), 2),
        },

        "main_cause": main_cause,

        # Latest raw reading in the window
        "latest_values": {
            "current": round(float(latest_raw_values[0]), 4),
            "temperature": round(float(latest_raw_values[1]), 4),
            "vibration": round(float(latest_raw_values[2]), 4),
        }
    }


def startup_load():
    global model, scaler, config

    config = load_json_config(CONFIG_PATH)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")

    model = load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)


# ============================================================
# ROUTES
# ============================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Industrial AI monitoring backend is running.",
        "model_loaded": model is not None,
        "window_size": EXPECTED_WINDOW,
        "features": EXPECTED_FEATURES
    })


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "ok": True,
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "config_loaded": config is not None
    })


@app.route("/batch", methods=["POST"])
def batch_predict():
    try:
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "Invalid or missing JSON body."}), 400

        readings = payload.get("readings")
        is_valid, message = validate_input(readings)

        if not is_valid:
            return jsonify({"error": message}), 400

        # Convert raw input
        raw_window = to_numpy_2d(readings)              # (20, 3)

        # Scale
        scaled_window = scale_window(raw_window)        # (20, 3)

        # Reshape for LSTM
        model_input = reshape_for_model(scaled_window)  # (1, 20, 3)

        # Predict reconstruction
        reconstructed = compute_reconstruction(model_input)

        # Compute errors
        total_error, feature_mse, timestep_mse = compute_errors(model_input, reconstructed)

        # Store error history for crude trend / RUL estimation
        error_history.append(total_error)

        thresholds = config.get("thresholds", DEFAULT_CONFIG["thresholds"])

        # Health and status
        health = compute_health(total_error, thresholds)
        status = classify_status(total_error, thresholds)

        # Contributions
        contributions, main_cause = compute_sensor_contributions(feature_mse, config)

        # RUL estimate
        rul = estimate_rul_hours(total_error, thresholds)

        # Latest raw values
        latest = raw_window[-1]

        response = format_response(
            latest_raw_values=latest,
            total_error=total_error,
            health=health,
            contributions=contributions,
            main_cause=main_cause,
            rul=rul,
            status=status
        )

        return jsonify(response), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500


# ============================================================
# STARTUP
# ============================================================

try:
    startup_load()
    print("Model, scaler, and config loaded successfully.")
except Exception as e:
    print("Startup loading failed:")
    print(str(e))
    traceback.print_exc()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)