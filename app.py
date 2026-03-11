from flask import Flask, request, jsonify
import os
import traceback
from collections import deque

import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "lstm_autoencoder.keras")
SCALER_PATH = os.getenv("SCALER_PATH", "scaler.save")
THRESHOLD_PATH = os.getenv("THRESHOLD_PATH", "threshold.npy")

WINDOW_SIZE = 20
NUM_FEATURES = 3
FEATURE_NAMES = ["current", "temperature", "vibration"]

model = None
scaler = None
threshold = None
error_history = deque(maxlen=50)


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


def compute_health(error_value, anomaly_threshold):
    t_normal = anomaly_threshold * 0.30
    t_critical = anomaly_threshold * 2.00

    if error_value <= t_normal:
        return 100.0

    if error_value >= t_critical:
        return 0.0

    health = 100.0 * (1.0 - (error_value - t_normal) / (t_critical - t_normal))
    return float(clamp(health, 0.0, 100.0))


def smooth_health(current_health):
    if len(error_history) < 3:
        return current_health

    recent_avg_error = float(np.mean(error_history))
    avg_based_health = compute_health(recent_avg_error, threshold)

    smoothed = (0.7 * current_health) + (0.3 * avg_based_health)
    return float(clamp(smoothed, 0.0, 100.0))


def estimate_rul(error_value, anomaly_threshold):
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


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "LSTM autoencoder backend is running.",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "threshold_loaded": threshold is not None,
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
        print("\n========== /batch called ==========")

        payload = request.get_json(silent=True)
        print("Incoming payload:", payload)

        if payload is None:
            return jsonify({"error": "Missing or invalid JSON body."}), 400

        readings = payload.get("readings")
        valid, message = validate_input(readings)
        print("Validation:", valid, message)

        if not valid:
            return jsonify({"error": message}), 400

        raw_window = np.array(readings, dtype=np.float32)
        print("raw_window shape:", raw_window.shape)
        print("raw_window sample:", raw_window[-1])

        scaled_window = scaler.transform(raw_window)
        print("scaled_window shape:", scaled_window.shape)
        print("scaled_window sample:", scaled_window[-1])

        x_input = np.expand_dims(scaled_window, axis=0)
        print("x_input shape:", x_input.shape)

        x_pred = model.predict(x_input, verbose=0)
        print("x_pred shape:", x_pred.shape)

        reconstruction_error = compute_total_error(x_input, x_pred)
        print("reconstruction_error:", reconstruction_error)

        error_history.append(reconstruction_error)

        is_anomaly = reconstruction_error > threshold
        instant_health = compute_health(reconstruction_error, threshold)
        health = smooth_health(instant_health)

        feature_errors = compute_feature_errors(x_input, x_pred)
        print("feature_errors:", feature_errors)

        contributions, main_cause = compute_sensor_contributions(feature_errors)
        rul = estimate_rul(reconstruction_error, threshold)

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

        print("Response:", response)
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


try:
    load_all()
except Exception as e:
    print("Startup failed:")
    print(str(e))
    traceback.print_exc()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)