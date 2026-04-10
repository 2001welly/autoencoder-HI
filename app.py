from flask import Flask, request, jsonify
import os
import time
import traceback
from collections import deque

import numpy as np
import joblib

try:
    from tensorflow.keras.models import load_model
except Exception:
    from keras.models import load_model

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
# HEALTH / RUL SETTINGS
# ============================================================
ERROR_HISTORY_SIZE = 30
MIN_RUL_POINTS = 8
EMA_ALPHA = 0.2
FAILURE_MULTIPLIER = 5.0
MAX_RUL_HOURS = 100.0
SAMPLE_INTERVAL_SECONDS = float(os.getenv("SAMPLE_INTERVAL_SECONDS", "10"))
SAMPLE_INTERVAL_HOURS = SAMPLE_INTERVAL_SECONDS / 3600.0

# Option 3 RUL weights
INSUFFICIENT_HISTORY_HEALTH_WEIGHT = 0.6
STABLE_HEALTH_WEIGHT = 0.8
DEGRADING_PROJECTED_WEIGHT = 0.7
DEGRADING_HEALTH_WEIGHT = 0.3

# ============================================================
# GLOBALS
# ============================================================
model = None
scaler = None
threshold = None
startup_error = None
is_loaded = False

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


def backend_ready():
    return model is not None and scaler is not None and threshold is not None


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
    raw_error_history.append(float(new_error))

    if len(smooth_error_history) == 0:
        smooth = float(new_error)
    else:
        smooth = EMA_ALPHA * float(new_error) + (1.0 - EMA_ALPHA) * smooth_error_history[-1]

    smooth_error_history.append(float(smooth))
    return float(smooth)


def compute_health(smoothed_error, anomaly_threshold):
    failure_threshold = anomaly_threshold * FAILURE_MULTIPLIER

    if smoothed_error <= anomaly_threshold:
        return 100.0

    if smoothed_error >= failure_threshold:
        return 0.0

    health = 100.0 * (
        1.0 - (smoothed_error - anomaly_threshold) / (failure_threshold - anomaly_threshold)
    )
    return float(clamp(health, 0.0, 100.0))


def estimate_trend():
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
    failure_threshold = anomaly_threshold * FAILURE_MULTIPLIER
    health = compute_health(smoothed_error, anomaly_threshold)
    slope = estimate_trend()

    if smoothed_error >= failure_threshold or health <= 0:
        return 0.0, "failed"

    distance_to_failure = failure_threshold - smoothed_error
    health_reserve = (health / 100.0) * MAX_RUL_HOURS

    if slope is None:
        rul = INSUFFICIENT_HISTORY_HEALTH_WEIGHT * health_reserve
        return float(clamp(rul, 0.0, MAX_RUL_HOURS)), "insufficient_history"

    if slope <= 0:
        rul = STABLE_HEALTH_WEIGHT * health_reserve
        return float(clamp(rul, 0.0, MAX_RUL_HOURS)), "stable"

    projected_hours = distance_to_failure / slope
    projected_hours = clamp(projected_hours, 0.0, MAX_RUL_HOURS)

    rul = (
        DEGRADING_PROJECTED_WEIGHT * projected_hours +
        DEGRADING_HEALTH_WEIGHT * health_reserve
    )

    return float(clamp(rul, 0.0, MAX_RUL_HOURS)), "degrading"


def compute_ood_score(raw_window, scaler_obj):
    if scaler_obj is None:
        return None, {}, None

    mins = getattr(scaler_obj, "data_min_", None)
    maxs = getattr(scaler_obj, "data_max_", None)

    if mins is None or maxs is None:
        return None, {}, None

    latest = np.array(raw_window[-1], dtype=np.float64)
    mins = np.array(mins, dtype=np.float64)
    maxs = np.array(maxs, dtype=np.float64)
    span = np.maximum(maxs - mins, 1e-6)

    below = np.maximum((mins - latest) / span, 0.0)
    above = np.maximum((latest - maxs) / span, 0.0)
    violation = below + above

    details = {
        "current": float(violation[0]),
        "temperature": float(violation[1]),
        "vibration": float(violation[2]),
    }

    main_ood_feature = FEATURE_NAMES[int(np.argmax(violation))] if np.max(violation) > 0 else None
    ood_score = float(np.mean(violation))

    return ood_score, details, main_ood_feature


def compute_trend_instability():
    if len(smooth_error_history) < MIN_RUL_POINTS:
        return 1.0

    y = np.array(smooth_error_history, dtype=np.float64)
    diffs = np.diff(y)

    if len(diffs) == 0:
        return 1.0

    diff_std = float(np.std(diffs))
    diff_mean = float(np.mean(np.abs(diffs))) + 1e-6

    instability = diff_std / diff_mean
    return float(clamp(instability, 0.0, 1.0))


def compute_error_fluctuation(anomaly_threshold):
    if len(raw_error_history) < 2:
        return 1.0

    err_std = float(np.std(np.array(raw_error_history, dtype=np.float64)))
    scale = max(0.5 * anomaly_threshold, 1e-6)
    fluctuation = err_std / scale
    return float(clamp(fluctuation, 0.0, 1.0))


def compute_confidence(ood_score, anomaly_threshold):
    error_fluctuation = compute_error_fluctuation(anomaly_threshold)
    trend_instability = compute_trend_instability()
    out_of_distribution = float(clamp((ood_score or 0.0) / 0.30, 0.0, 1.0))
    limited_history = float(1.0 - min(len(smooth_error_history) / float(MIN_RUL_POINTS), 1.0))

    total_penalty = (
        0.35 * error_fluctuation +
        0.30 * trend_instability +
        0.20 * out_of_distribution +
        0.15 * limited_history
    )

    confidence_score = 100.0 * (1.0 - total_penalty)
    confidence_score = float(clamp(confidence_score, 0.0, 100.0))

    if confidence_score >= 75:
        confidence_level = "high"
    elif confidence_score >= 45:
        confidence_level = "medium"
    else:
        confidence_level = "low"

    sources = {
        "error_fluctuation": round(error_fluctuation, 4),
        "trend_instability": round(trend_instability, 4),
        "out_of_distribution": round(out_of_distribution, 4),
        "limited_history": round(limited_history, 4)
    }

    return confidence_score, confidence_level, sources


def compute_rul_range(rul, confidence_score, health, rul_state):
    if rul <= 0:
        return 0.0, 0.0, 0.0

    uncertainty_factor = 1.0 - (confidence_score / 100.0)

    if rul_state == "insufficient_history":
        spread_ratio = 0.45 + 0.30 * uncertainty_factor
    elif rul_state == "stable":
        spread_ratio = 0.20 + 0.25 * uncertainty_factor
    else:
        spread_ratio = 0.25 + 0.35 * uncertainty_factor

    if health < 40:
        spread_ratio += 0.10

    spread_ratio = float(clamp(spread_ratio, 0.10, 0.85))

    rul_min = max(0.0, rul * (1.0 - spread_ratio))
    rul_max = min(MAX_RUL_HOURS, rul * (1.0 + spread_ratio))
    rul_std = (rul_max - rul_min) / 4.0

    return float(rul_min), float(rul_max), float(rul_std)


def build_uncertainty_reason(ood_score, ood_feature, confidence_sources):
    ranked = sorted(confidence_sources.items(), key=lambda x: x[1], reverse=True)
    top_name, top_value = ranked[0]

    if ood_score is not None and ood_score > 0.10 and ood_feature:
        return f"{ood_feature.capitalize()} is outside the training range, so prediction confidence is reduced."

    if top_name == "limited_history" and top_value > 0.5:
        return "Not enough recent history is available yet for a stable RUL estimate."

    if top_name == "trend_instability" and top_value > 0.4:
        return "Recent degradation trend is unstable, so the future failure time is uncertain."

    if top_name == "error_fluctuation" and top_value > 0.4:
        return "Reconstruction error is fluctuating, so the RUL range is wider than usual."

    return "Prediction is based on recent behaviour and is relatively stable."


def derive_status_and_led(is_anomaly, health_value):
    if is_anomaly or health_value <= 20:
        return "Anomaly", "RED"

    if health_value <= 60:
        return "Warning", "YELLOW"

    return "Normal", "GREEN"


def load_all():
    global model, scaler, threshold, is_loaded

    if is_loaded:
        return

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")

    if not os.path.exists(THRESHOLD_PATH):
        raise FileNotFoundError(f"Threshold file not found: {THRESHOLD_PATH}")

    print("Loading model...")
    model_local = load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully.")

    print("Loading scaler...")
    scaler_local = joblib.load(SCALER_PATH)
    print("Scaler loaded successfully.")

    print("Loading threshold...")
    threshold_value = np.load(THRESHOLD_PATH, allow_pickle=True)
    threshold_local = float(threshold_value)
    print(f"Threshold loaded successfully: {threshold_local}")

    model = model_local
    scaler = scaler_local
    threshold = threshold_local
    is_loaded = True


def ensure_loaded():
    global startup_error
    if backend_ready():
        return True

    try:
        load_all()
        startup_error = None
        return True
    except Exception as e:
        startup_error = str(e)
        print("Lazy load failed:")
        print(str(e))
        traceback.print_exc()
        return False


# ============================================================
# ROUTES
# ============================================================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "LSTM autoencoder backend is running.",
        "ready": backend_ready(),
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "threshold_loaded": threshold is not None,
        "threshold": threshold,
        "failure_threshold": round(float(threshold * FAILURE_MULTIPLIER), 6) if threshold is not None else None,
        "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
        "startup_error": startup_error,
        "model_path": MODEL_PATH,
        "scaler_path": SCALER_PATH,
        "threshold_path": THRESHOLD_PATH
    })


@app.route("/health", methods=["GET"])
def health_check():
    code = 200 if backend_ready() else 503
    return jsonify({
        "ok": backend_ready(),
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "threshold_loaded": threshold is not None,
        "startup_error": startup_error
    }), code


@app.route("/batch", methods=["POST"])
def batch_predict():
    if not ensure_loaded():
        return jsonify({
            "error": "Backend not ready",
            "details": startup_error or "Model, scaler, or threshold not loaded."
        }), 503

    try:
        t0 = time.time()
        print("\n========== /batch called ==========")

        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "Missing or invalid JSON body."}), 400

        readings = payload.get("readings")
        valid, message = validate_input(readings)
        if not valid:
            return jsonify({"error": message}), 400

        raw_window = np.array(readings, dtype=np.float32)
        scaled_window = scaler.transform(raw_window)
        x_input = np.expand_dims(scaled_window, axis=0)

        x_pred = model(x_input, training=False).numpy()

        reconstruction_error = compute_total_error(x_input, x_pred)
        is_anomaly = reconstruction_error > threshold
        smoothed_error = update_smoothed_error(reconstruction_error)
        health = compute_health(smoothed_error, threshold)

        feature_errors = compute_feature_errors(x_input, x_pred)
        contributions, main_cause = compute_sensor_contributions(feature_errors)

        rul, rul_state = estimate_rul(smoothed_error, threshold)
        slope = estimate_trend()

        ood_score, ood_details, ood_feature = compute_ood_score(raw_window, scaler)
        confidence_score, confidence_level, confidence_sources = compute_confidence(ood_score, threshold)
        rul_min, rul_max, rul_std = compute_rul_range(rul, confidence_score, health, rul_state)
        uncertainty_reason = build_uncertainty_reason(ood_score, ood_feature, confidence_sources)

        latest = raw_window[-1]
        status, led_status = derive_status_and_led(is_anomaly, health)

        response = {
            "is_anomaly": bool(is_anomaly),
            "status": status,
            "led_status": led_status,
            "health": round(float(health), 2),
            "rul": round(float(rul), 2),
            "rul_state": rul_state,
            "reconstruction_error": round(float(reconstruction_error), 6),
            "smoothed_error": round(float(smoothed_error), 6),
            "threshold": round(float(threshold), 6),
            "failure_threshold": round(float(threshold * FAILURE_MULTIPLIER), 6),
            "degradation_rate": round(float(slope), 6) if slope is not None else None,

            "confidence_level": confidence_level,
            "confidence_score": round(float(confidence_score), 2),
            "rul_min": round(float(rul_min), 2),
            "rul_max": round(float(rul_max), 2),
            "rul_std": round(float(rul_std), 2),
            "ood_score": round(float(ood_score), 6) if ood_score is not None else None,
            "ood_details": {
                "current": round(float(ood_details.get("current", 0.0)), 6),
                "temperature": round(float(ood_details.get("temperature", 0.0)), 6),
                "vibration": round(float(ood_details.get("vibration", 0.0)), 6)
            } if ood_details else {},
            "uncertainty_reason": uncertainty_reason,
            "uncertainty_sources": confidence_sources,

            "main_cause": main_cause,
            "sensor_contributions": {
                "current": round(float(contributions["current"]), 2),
                "temperature": round(float(contributions["temperature"]), 2),
                "vibration": round(float(contributions["vibration"]), 2)
            },
            "contrib_current": round(float(contributions["current"]), 2),
            "contrib_temperature": round(float(contributions["temperature"]), 2),
            "contrib_vibration": round(float(contributions["vibration"]), 2),

            "latest_values": {
                "current": round(float(latest[0]), 4),
                "temperature": round(float(latest[1]), 4),
                "vibration": round(float(latest[2]), 4)
            },
            "current": round(float(latest[0]), 4),
            "temperature": round(float(latest[1]), 4),
            "vibration": round(float(latest[2]), 4),

            "current_status": {
                "current": round(float(latest[0]), 4),
                "is_anomaly": bool(is_anomaly),
                "status": status
            }
        }

        print("done /batch total time:", round(time.time() - t0, 3), "s")
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


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)