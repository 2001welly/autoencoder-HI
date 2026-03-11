from flask import Flask, request, jsonify
import os
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
# EXPECTED INPUT SHAPE
# ============================================================
WINDOW_SIZE = 20
NUM_FEATURES = 3
FEATURE_NAMES = ["current", "temperature", "vibration"]

# ============================================================
# GLOBAL OBJECTS
# ============================================================
model = None
scaler = None
threshold = None

# store recent errors for simple health smoothing / RUL estimate
error_history = deque(maxlen=50)


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
    """
    Returns total MSE over whole batch window.
    x_true shape: (1, 20, 3)
    x_pred shape: (1, 20, 3)
    """
    return float(np.mean(np.square(x_true - x_pred)))


def compute_feature_errors(x_true, x_pred):
    """
    Returns per-feature MSE across the 20 timesteps.
    Output shape: (3,)
    """
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


def compute_health(error_value, anomaly_threshold):
    """
    Health from reconstruction error.
    We use:
    - very low error => close to 100%
    - threshold => around 50%
    - far above threshold => close to 0%

    Since you currently have one threshold only, we create:
    normal threshold = 30% of anomaly threshold
    critical threshold = 200% of anomaly threshold
    """
    t_normal = anomaly_threshold * 0.30
    t_critical = anomaly_threshold * 2.00

    if error_value <= t_normal:
        return 100.0

    if error_value >= t_critical:
        return 0.0

    health = 100.0 * (1.0 - (error_value - t_normal) / (t_critical - t_normal))
    return float(clamp(health, 0.0, 100.0))


def smooth_health(current_health):
    """
    Smooth the displayed health using recent error history.
    """
    if len(error_history) < 3:
        return current_health

    recent_avg_error = float(np.mean(error_history))
    avg_based_health = compute_health(recent_avg_error, threshold)

    # blend instant and recent behavior
    smoothed = (0.7 * current_health) + (0.3 * avg_based_health)
    return float(clamp(smoothed, 0.0, 100.0))


def estimate_rul(error_value, anomaly_threshold):
    """
    Very simple engineering estimate.
    Not true physical life prediction.
    """
    critical_limit = anomaly_threshold * 2.0

    if error_value >= critical_limit:
        return 0.0

    if error_value <= anomaly_threshold * 0.3:
        return 100.0

    ratio = (error_value - anomaly_threshold * 0.3) / (critical_limit - anomaly_threshold * 0.3)
    ratio = clamp(ratio, 0.0, 1.0)

    rul_hours = 100.0 * (1.0 - ratio)
    return float(clamp(rul_hours, 0.0, 100.0))


def load_all():
    global model, scaler, threshold

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")

    if not os.path.exists(THRESHOLD_PATH):
        raise FileNotFoundError(f"Threshold file not found: {THRESHOLD_PATH}")

    model = load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)
    threshold_value = np.load(THRESHOLD_PATH, allow_pickle=True)

    # convert numpy scalar to Python float
    threshold = float(threshold_value)

    print("Model loaded successfully.")
    print("Scaler loaded successfully.")
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
        "window_size": WINDOW_SIZE,
        "num_features": NUM_FEATURES,
        "threshold": threshold
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
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "Missing or invalid JSON body."}), 400

        readings = payload.get("readings")
        valid, message = validate_input(readings)
        if not valid:
            return jsonify({"error": message}), 400

        # raw input shape -> (20, 3)
        raw_window = np.array(readings, dtype=np.float32)

        # scale using saved scaler
        scaled_window = scaler.transform(raw_window)

        # reshape for LSTM -> (1, 20, 3)
        x_input = np.expand_dims(scaled_window, axis=0)

        # predict reconstruction
        x_pred = model.predict(x_input, verbose=0)

        # total reconstruction error
        reconstruction_error = compute_total_error(x_input, x_pred)
        error_history.append(reconstruction_error)

        # anomaly decision
        is_anomaly = reconstruction_error > threshold

        # health
        instant_health = compute_health(reconstruction_error, threshold)
        health = smooth_health(instant_health)

        # per-feature contribution
        feature_errors = compute_feature_errors(x_input, x_pred)
        contributions, main_cause = compute_sensor_contributions(feature_errors)

        # simple RUL estimate
        rul = estimate_rul(reconstruction_error, threshold)

        # latest raw values
        latest = raw_window[-1]

        response = {
            "is_anomaly": bool(is_anomaly),
            "health": round(float(health), 2),
            "rul": round(float(rul), 2),
            "reconstruction_error": round(float(reconstruction_error), 6),
            "threshold": round(float(threshold), 6),
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
    load_all()
except Exception as e:
    print("Startup failed:")
    print(str(e))
    traceback.print_exc()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)