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

# Option 3 RUL model weights
INSUFFICIENT_HISTORY_HEALTH_WEIGHT = 0.6
STABLE_HEALTH_WEIGHT = 0.8
DEGRADING_PROJECTED_WEIGHT = 0.7
DEGRADING_HEALTH_WEIGHT = 0.3

# ============================================================
# UNCERTAINTY SETTINGS
# ============================================================
BOOTSTRAP_SAMPLES = 200
RUL_MONTE_CARLO_SAMPLES = 300
MIN_ERROR_STD = 1e-6
OOD_WARN_THRESHOLD = 0.10

# ============================================================
# GLOBALS
# ============================================================
model = None
scaler = None
threshold = None
startup_error = None

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


def estimate_rul_core(smoothed_error, anomaly_threshold, slope):
    failure_threshold = anomaly_threshold * FAILURE_MULTIPLIER
    health = compute_health(smoothed_error, anomaly_threshold)

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


def estimate_rul(smoothed_error, anomaly_threshold):
    slope = estimate_trend()
    return estimate_rul_core(smoothed_error, anomaly_threshold, slope)


def compute_ood_score(raw_window, scaler_obj):
    """
    Measures how far the latest reading is outside the scaler's training range.
    0.0 means inside range. Larger values mean more out-of-distribution.
    """
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


def estimate_recent_error_std(anomaly_threshold, ood_score):
    if len(raw_error_history) >= 2:
        base_std = float(np.std(np.array(raw_error_history, dtype=np.float64)))
    else:
        base_std = 0.05 * anomaly_threshold

    # Inflate uncertainty if current conditions are outside training range
    if ood_score is not None:
        base_std *= (1.0 + 3.0 * float(ood_score))

    return float(max(base_std, MIN_ERROR_STD))


def bootstrap_slope_distribution():
    if len(smooth_error_history) < MIN_RUL_POINTS:
        return None

    y = np.array(smooth_error_history, dtype=np.float64)
    x = np.arange(len(y), dtype=np.float64) * SAMPLE_INTERVAL_HOURS
    n = len(y)

    slopes = []

    for _ in range(BOOTSTRAP_SAMPLES):
        idx = np.random.choice(np.arange(n), size=n, replace=True)
        idx.sort()
        xb = x[idx]
        yb = y[idx]

        if np.std(xb) < 1e-12:
            continue

        try:
            slope = np.polyfit(xb, yb, 1)[0]
            if np.isfinite(slope):
                slopes.append(float(slope))
        except Exception:
            continue

    if len(slopes) == 0:
        return None

    return np.array(slopes, dtype=np.float64)


def estimate_rul_distribution(smoothed_error, anomaly_threshold, ood_score):
    health = compute_health(smoothed_error, anomaly_threshold)
    failure_threshold = anomaly_threshold * FAILURE_MULTIPLIER

    if smoothed_error >= failure_threshold or health <= 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p10": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "slope_mean": None,
            "slope_std": None,
            "error_std": 0.0
        }

    error_std = estimate_recent_error_std(anomaly_threshold, ood_score)
    slope_now = estimate_trend()
    slope_samples = bootstrap_slope_distribution()

    if slope_samples is None:
        sampled_slopes = [slope_now] * RUL_MONTE_CARLO_SAMPLES
    else:
        sampled_slopes = np.random.choice(slope_samples, size=RUL_MONTE_CARLO_SAMPLES, replace=True)

    rng = np.random.default_rng()
    error_samples = rng.normal(
        loc=float(smoothed_error),
        scale=float(error_std),
        size=RUL_MONTE_CARLO_SAMPLES
    )

    rul_samples = []
    for err_value, slope_value in zip(error_samples, sampled_slopes):
        err_value = float(max(0.0, err_value))
        rul_i, _ = estimate_rul_core(err_value, anomaly_threshold, slope_value)
        rul_samples.append(float(clamp(rul_i, 0.0, MAX_RUL_HOURS)))

    rul_samples = np.array(rul_samples, dtype=np.float64)

    slope_mean = float(np.mean(slope_samples)) if slope_samples is not None and len(slope_samples) > 0 else slope_now
    slope_std = float(np.std(slope_samples)) if slope_samples is not None and len(slope_samples) > 0 else None

    return {
        "mean": float(np.mean(rul_samples)),
        "std": float(np.std(rul_samples)),
        "min": float(np.min(rul_samples)),
        "max": float(np.max(rul_samples)),
        "p10": float(np.percentile(rul_samples, 10)),
        "p50": float(np.percentile(rul_samples, 50)),
        "p90": float(np.percentile(rul_samples, 90)),
        "slope_mean": slope_mean,
        "slope_std": slope_std,
        "error_std": float(error_std)
    }


def compute_confidence(ood_score, rul_dist):
    slope_mean = rul_dist.get("slope_mean")
    slope_std = rul_dist.get("slope_std")
    error_std = float(rul_dist.get("error_std", 0.0))

    threshold_scale = max(float(threshold) if threshold is not None else 1.0, 1e-6)

    # Error uncertainty
    error_penalty = clamp(error_std / (0.5 * threshold_scale), 0.0, 1.0)

    # Trend uncertainty
    if slope_mean is None or slope_std is None:
        trend_penalty = 0.7
    else:
        denom = max(abs(float(slope_mean)), 1e-6)
        trend_penalty = clamp(float(slope_std) / denom, 0.0, 1.0)

    # OOD uncertainty
    ood_penalty = clamp(float(ood_score or 0.0) / 0.30, 0.0, 1.0)

    # History penalty
    history_penalty = 1.0 - min(len(smooth_error_history) / float(MIN_RUL_POINTS), 1.0)

    total_penalty = (
        0.35 * error_penalty +
        0.30 * trend_penalty +
        0.20 * ood_penalty +
        0.15 * history_penalty
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
        "error_fluctuation": float(error_penalty),
        "trend_instability": float(trend_penalty),
        "out_of_distribution": float(ood_penalty),
        "limited_history": float(history_penalty)
    }

    return confidence_score, confidence_level, sources


def build_uncertainty_reason(ood_score, ood_feature, confidence_sources):
    items = sorted(confidence_sources.items(), key=lambda kv: kv[1], reverse=True)
    top_source, top_value = items[0]

    if ood_score is not None and ood_score > OOD_WARN_THRESHOLD and ood_feature:
        return f"{ood_feature.capitalize()} is outside the training range, so prediction confidence is reduced."

    if top_source == "limited_history" and top_value > 0.5:
        return "Not enough recent history is available yet for a stable trend-based RUL estimate."

    if top_source == "trend_instability" and top_value > 0.4:
        return "Recent degradation trend is unstable, so the future failure time is uncertain."

    if top_source == "error_fluctuation" and top_value > 0.4:
        return "Reconstruction error is fluctuating, so the RUL spread is wider than usual."

    return "Prediction is based on recent behaviour and is relatively stable."


def derive_status_and_led(is_anomaly, health_value):
    if is_anomaly or health_value <= 20:
        return "Anomaly", "RED"

    if health_value <= 60:
        return "Warning", "YELLOW"

    return "Normal", "GREEN"


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
    if not backend_ready():
        return jsonify({
            "error": "Backend not ready",
            "details": startup_error or "Model, scaler, or threshold not loaded."
        }), 503

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

        x_pred = model.predict(x_input, verbose=0)
        print("x_pred shape:", x_pred.shape, "elapsed:", round(time.time() - t0, 3), "s")

        reconstruction_error = compute_total_error(x_input, x_pred)
        print("reconstruction_error:", reconstruction_error)

        is_anomaly = reconstruction_error > threshold

        smoothed_error = update_smoothed_error(reconstruction_error)
        print("smoothed_error:", smoothed_error)

        health = compute_health(smoothed_error, threshold)

        feature_errors = compute_feature_errors(x_input, x_pred)
        contributions, main_cause = compute_sensor_contributions(feature_errors)

        rul, rul_state = estimate_rul(smoothed_error, threshold)
        slope = estimate_trend()

        ood_score, ood_details, ood_feature = compute_ood_score(raw_window, scaler)
        rul_dist = estimate_rul_distribution(smoothed_error, threshold, ood_score)
        confidence_score, confidence_level, confidence_sources = compute_confidence(ood_score, rul_dist)
        uncertainty_reason = build_uncertainty_reason(ood_score, ood_feature, confidence_sources)

        latest = raw_window[-1]
        status, led_status = derive_status_and_led(is_anomaly, health)

        response = {
            # Core outputs
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

            # Uncertainty-aware outputs
            "confidence_level": confidence_level,
            "confidence_score": round(float(confidence_score), 2),
            "rul_min": round(float(rul_dist["p10"]), 2),
            "rul_max": round(float(rul_dist["p90"]), 2),
            "rul_std": round(float(rul_dist["std"]), 2),
            "ood_score": round(float(ood_score), 6) if ood_score is not None else None,
            "ood_details": {
                "current": round(float(ood_details.get("current", 0.0)), 6),
                "temperature": round(float(ood_details.get("temperature", 0.0)), 6),
                "vibration": round(float(ood_details.get("vibration", 0.0)), 6)
            } if ood_details else {},
            "uncertainty_reason": uncertainty_reason,
            "uncertainty_sources": {
                "error_fluctuation": round(float(confidence_sources["error_fluctuation"]), 4),
                "trend_instability": round(float(confidence_sources["trend_instability"]), 4),
                "out_of_distribution": round(float(confidence_sources["out_of_distribution"]), 4),
                "limited_history": round(float(confidence_sources["limited_history"]), 4)
            },

            # Explainability
            "main_cause": main_cause,
            "sensor_contributions": {
                "current": round(float(contributions["current"]), 2),
                "temperature": round(float(contributions["temperature"]), 2),
                "vibration": round(float(contributions["vibration"]), 2)
            },
            # Flat contribution fields for compatibility
            "contrib_current": round(float(contributions["current"]), 2),
            "contrib_temperature": round(float(contributions["temperature"]), 2),
            "contrib_vibration": round(float(contributions["vibration"]), 2),

            # Latest readings
            "latest_values": {
                "current": round(float(latest[0]), 4),
                "temperature": round(float(latest[1]), 4),
                "vibration": round(float(latest[2]), 4)
            },
            "current": round(float(latest[0]), 4),
            "temperature": round(float(latest[1]), 4),
            "vibration": round(float(latest[2]), 4),

            # Compatibility helper
            "current_status": {
                "current": round(float(latest[0]), 4),
                "is_anomaly": bool(is_anomaly),
                "status": status
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
    startup_error = str(e)
    print("Startup failed:")
    print(str(e))
    traceback.print_exc()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)