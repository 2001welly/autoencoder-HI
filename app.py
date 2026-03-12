from flask import Flask, request, jsonify
import os
import time
import traceback
from collections import deque

import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ============================================================
# FILE PATHS
# ============================================================
MODEL_PATH = os.getenv("MODEL_PATH", "lstm_autoencoder.keras")
SCALER_PATH = os.getenv("SCALER_PATH", "scaler.save")
THRESHOLD_PATH = os.getenv("THRESHOLD_PATH", "threshold.npy")

# ============================================================
# INPUT SETTINGS
# ============================================================
WINDOW_SIZE = 20
NUM_FEATURES = 3
FEATURE_NAMES = ["current", "temperature", "vibration"]

# ============================================================
# RUL / HEALTH SETTINGS
# ============================================================
ERROR_HISTORY_SIZE = 30          # how many recent points to keep
MIN_RUL_POINTS = 8               # minimum points before computing trend-based RUL
EMA_ALPHA = 0.2                  # smoothing factor for reconstruction error
FAILURE_MULTIPLIER = 5.0         # failure threshold = threshold * this
MAX_RUL_HOURS = 100.0            # cap displayed RUL
SAMPLE_INTERVAL_HOURS = 10.0 / 3600.0   # ESP32 sends every 10 seconds

# ============================================================
# GLOBALS
# ============================================================
model = None
scaler = None
threshold = None

raw_error_history = deque(maxlen=ERROR_HISTORY_SIZE)
smooth_error_history = deque(maxlen=ERROR_HISTORY_SIZE)


# ============================================================
# HELPERS
# ============================================================
def clamp(value, low, high):
    return max(low, min(high, value))


def validate_input(readings):
    if not isinstance(readings, list):
        return False, "Field 'readings' must be a list."

    if len(readings) != WINDOW_SIZE:
        return False, f"'readings' must contain exactly {WINDOW_SIZE} rows."

    for i, row in enumerate(readings):
        if not isinstance(row, list):
            return False, f"Row {i} must be a list."

        if len(row) != NUM_FEATURES:
            return False, f"Row {i} must contain exactly {NUM_FEATURES} values."

        for j, value in enumerate(row):
            try:
                float(value)
            except Exception:
                return False, f"Value at row {i}, column {j} is not numeric."

    return True, "OK"


def compute_total_error(x_true, x_pred):
    return float(np.mean(np.square(x_true - x_pred)))


def compute_feature_errors(x_true, x_pred):
    return np.mean(np.square(x_true[0] - x_pred[0]), axis=0)


def compute_sensor_contributions(feature_errors):
    total = float(np.sum(feature_errors))

    if total <= 1e-12:
        return {
            "current": 0.0,
            "temperature": 0.0,
            "vibration": 0.0
        }, "unknown"

    perc = (feature_errors / total) * 100.0

    contributions = {
        "current": float(perc[0]),
        "temperature": float(perc[1]),
        "vibration": float(perc[2])
    }

    main_cause = FEATURE_NAMES[int(np.argmax(perc))]
    return contributions, main_cause


def update_smoothed_error(new_error):
    """
    Exponential moving average smoothing.
    """
    raw_error_history.append(new_error)

    if len(smooth_error_history) == 0:
        smooth = new_error
    else:
        smooth = EMA_ALPHA * new_error + (1.0 - EMA_ALPHA) * smooth_error_history[-1]

    smooth_error_history.append(float(smooth))
    return float(smooth)


def compute_health(smoothed_error, anomaly_threshold):
    """
    Health based on smoothed reconstruction error.
    Uses a wider operating span so health does not immediately stick at 0.
    """
    failure_threshold = anomaly_threshold * FAILURE_MULTIPLIER

    if smoothed_error <= anomaly_threshold:
        return 100.0

    if smoothed_error >= failure_threshold:
        return 0.0

    health = 100.0 * (1.0 - (smoothed_error - anomaly_threshold) / (failure_threshold - anomaly_threshold))
    return float(clamp(health, 0.0, 100.0))


def estimate_trend():
    """
    Estimate slope of smoothed reconstruction error over recent history.
    Returns slope in error-units per hour.
    """
    if len(smooth_error_history) < MIN_RUL_POINTS:
        return None

    y = np.array(smooth_error_history, dtype=np.float64)
    x = np.arange(len(y), dtype=np.float64) * SAMPLE_INTERVAL_HOURS

    try:
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)
    except Exception:
        return None


def estimate_rul(smoothed_error, anomaly_threshold):
    """
    Trend-based RUL:
    RUL = (failure_threshold - current_smoothed_error) / degradation_rate

    If not enough history exists, or degradation rate <= 0, return MAX_RUL_HOURS.
    """
    failure_threshold = anomaly_threshold * FAILURE_MULTIPLIER

    if smoothed_error >= failure_threshold:
        return 0.0, "failed"

    slope = estimate_trend()

    if slope is None:
        return MAX_RUL_HOURS, "insufficient_history"

    # not degrading or improving
    if slope <= 0:
        return MAX_RUL_HOURS, "stable"

    distance_to_failure = failure_threshold - smoothed_error

    if distance_to_failure <= 0:
        return 0.0, "failed"

    rul_hours = distance_to_failure / slope
    rul_hours = clamp(rul_hours, 0.0, MAX_RUL_HOURS)

    return float(rul_hours), "degrading"


def load_all():
    global model, scaler, threshold

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")

    if not os.path.exists(THRESHOLD_PATH):
        raise FileNotFoundError(f"Threshold file not found: {THRESHOLD_PATH}")

    print("Loading model...")
    model = load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully.")

    print("Loading scaler...")
    scaler = joblib.load(SCALER_PATH)
    print("Scaler loaded successfully.")

    print("Loading threshold...")
    threshold_value = np.load(THRESHOLD_PATH, allow_pickle=True)
    threshold = float(threshold_value)
    print(f"Threshold loaded successfully: {threshold}")


# ============================================================
# ROUTES
# ============================================================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "LSTM autoencoder backend is running.",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "threshold_loaded": threshold is not None,
        "threshold": threshold,
        "failure_threshold": round(float(threshold * FAILURE_MULTIPLIER), 6) if threshold is not None else None
    })


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "ok": True,
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "threshold_loaded": threshold is not None
    })


@app.route("/batch", methods=["POST"])
def batch_predict():
    try:
        t0 = time.time()
        print("\n========== /batch called ==========")

        payload = request.get_json(silent=True)
        if payload is None:
            print("Invalid JSON body")
            return jsonify({"error": "Missing or invalid JSON body."}), 400

        readings = payload.get("readings")
        valid, message = validate_input(readings)
        print("Validation:", valid, message)

        if not valid:
            return jsonify({"error": message}), 400

        raw_window = np.array(readings, dtype=np.float32)
        print("raw_window shape:", raw_window.shape, "elapsed:", round(time.time() - t0, 3), "s")

        scaled_window = scaler.transform(raw_window)
        print("scaled_window shape:", scaled_window.shape, "elapsed:", round(time.time() - t0, 3), "s")

        x_input = np.expand_dims(scaled_window, axis=0)
        print("x_input shape:", x_input.shape, "elapsed:", round(time.time() - t0, 3), "s")

        # direct call for better stability on Render
        x_pred = model(x_input, training=False).numpy()
        print("x_pred shape:", x_pred.shape, "elapsed:", round(time.time() - t0, 3), "s")

        reconstruction_error = compute_total_error(x_input, x_pred)
        print("reconstruction_error:", reconstruction_error)

        # anomaly decision remains based on original threshold
        is_anomaly = reconstruction_error > threshold

        # smooth error for health / RUL
        smoothed_error = update_smoothed_error(reconstruction_error)
        print("smoothed_error:", smoothed_error)

        # health from smoothed error
        health = compute_health(smoothed_error, threshold)

        # per-feature explanation
        feature_errors = compute_feature_errors(x_input, x_pred)
        contributions, main_cause = compute_sensor_contributions(feature_errors)

        # trend-based RUL
        rul, rul_state = estimate_rul(smoothed_error, threshold)
        slope = estimate_trend()

        latest = raw_window[-1]

        response = {
            "is_anomaly": bool(is_anomaly),
            "health": round(float(health), 2),
            "rul": round(float(rul), 2),
            "rul_state": rul_state,
            "reconstruction_error": round(float(reconstruction_error), 6),
            "smoothed_error": round(float(smoothed_error), 6),
            "threshold": round(float(threshold), 6),
            "failure_threshold": round(float(threshold * FAILURE_MULTIPLIER), 6),
            "degradation_rate": round(float(slope), 6) if slope is not None else None,
            "main_cause": main_cause,
            "sensor_contributions": {
                "current": round(float(contributions["current"]), 2),
                "temperature": round(float(contributions["temperature"]), 2),
                "vibration": round(float(contributions["vibration"]), 2)
            },
            "latest_values": {
                "current": round(float(latest[0]), 4),
                "temperature": round(float(latest[1]), 4),
                "vibration": round(float(latest[2]), 4)
            }
        }

        print("response ready:", response)
        print("done /batch total time:", round(time.time() - t0, 3), "s")
        print("========== /batch finished ==========\n")

        return jsonify(response), 200

    except Exception as e:
        print("\n========== /batch ERROR ==========")
        print("Error:", str(e))
        traceback.print_exc()
        print("========== END ERROR ==========\n")
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500


# ============================================================
# STARTUP
# ============================================================
try:
    load_all()
except Exception as e:
    print("Startup failed:")
    print(str(e))
    traceback.print_exc()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)