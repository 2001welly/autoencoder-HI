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
# KEEP THESE THE SAME AS THE OLD WORKING SETUP
# ============================================================
MODEL_PATH = os.getenv("MODEL_PATH", "lstm_autoencoder.keras")
SCALER_PATH = os.getenv("SCALER_PATH", "scaler.save")
THRESHOLD_PATH = os.getenv("THRESHOLD_PATH", "threshold.npy")

# ============================================================
# INPUT SETTINGS
# KEEP SAME WINDOW CONTRACT AS BEFORE
# ============================================================
WINDOW_SIZE = 20
NUM_FEATURES = 3
FEATURE_NAMES = ["current", "temperature", "vibration"]

# ============================================================
# HEALTH / RUL SETTINGS
# KEEP SAME CORE BEHAVIOUR AS OLD APP
# ============================================================
ERROR_HISTORY_SIZE = 30
MIN_RUL_POINTS = 8
EMA_ALPHA = 0.2
FAILURE_MULTIPLIER = 5.0
MAX_RUL_HOURS = 100.0
SAMPLE_INTERVAL_SECONDS = float(os.getenv("SAMPLE_INTERVAL_SECONDS", "10"))
SAMPLE_INTERVAL_HOURS = SAMPLE_INTERVAL_SECONDS / 3600.0

# Old Option 3 RUL weights
INSUFFICIENT_HISTORY_HEALTH_WEIGHT = 0.6
STABLE_HEALTH_WEIGHT = 0.8
DEGRADING_PROJECTED_WEIGHT = 0.7
DEGRADING_HEALTH_WEIGHT = 0.3

# ============================================================
# PRESCRIPTIVE MAINTENANCE SETTINGS
# NEW ADDITIONS ONLY
# ============================================================
PERSISTENCE_HISTORY_SIZE = 20

W_ANOMALY = 0.25
W_HEALTH = 0.20
W_RUL = 0.25
W_TREND = 0.15
W_PERSISTENCE = 0.15

MIXED_CAUSE_GAP_PERCENT = 8.0

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
anomaly_history = deque(maxlen=PERSISTENCE_HISTORY_SIZE)

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


# ============================================================
# NEW PRESCRIPTIVE HELPERS
# ADDED WITHOUT REMOVING OLD LOGIC
# ============================================================
def compute_anomaly_severity(smoothed_error, anomaly_threshold):
    failure_threshold = anomaly_threshold * FAILURE_MULTIPLIER

    if smoothed_error <= anomaly_threshold:
        return 0.0

    if smoothed_error >= failure_threshold:
        return 1.0

    severity = (smoothed_error - anomaly_threshold) / max((failure_threshold - anomaly_threshold), 1e-9)
    return float(clamp(severity, 0.0, 1.0))


def update_persistence(is_anomaly):
    anomaly_history.append(1.0 if is_anomaly else 0.0)
    if len(anomaly_history) == 0:
        return 0.0
    return float(np.mean(np.array(anomaly_history, dtype=np.float64)))


def compute_trend_factor(anomaly_threshold):
    slope = estimate_trend()
    if slope is None or slope <= 0:
        return 0.0

    failure_threshold = anomaly_threshold * FAILURE_MULTIPLIER
    error_gap = max(failure_threshold - anomaly_threshold, 1e-9)

    reference_slope = error_gap / max(MAX_RUL_HOURS, 1e-9)
    trend_factor = slope / max(reference_slope, 1e-9)

    return float(clamp(trend_factor, 0.0, 1.0))


def compute_rul_risk(rul):
    if rul is None:
        return 1.0
    return float(clamp(1.0 - (float(rul) / MAX_RUL_HOURS), 0.0, 1.0))


def compute_mps(smoothed_error, anomaly_threshold, health, rul, persistence):
    anomaly_severity = compute_anomaly_severity(smoothed_error, anomaly_threshold)
    health_degradation = float(clamp((100.0 - health) / 100.0, 0.0, 1.0))
    rul_risk = compute_rul_risk(rul)
    trend_factor = compute_trend_factor(anomaly_threshold)

    mps = (
        W_ANOMALY * anomaly_severity +
        W_HEALTH * health_degradation +
        W_RUL * rul_risk +
        W_TREND * trend_factor +
        W_PERSISTENCE * persistence
    )

    return {
        "maintenance_priority_score": float(clamp(mps, 0.0, 1.0)),
        "anomaly_severity": round(anomaly_severity, 4),
        "health_degradation": round(health_degradation, 4),
        "rul_risk": round(rul_risk, 4),
        "trend_factor": round(trend_factor, 4),
        "persistence_factor": round(float(clamp(persistence, 0.0, 1.0)), 4)
    }


def derive_urgency_level(mps):
    if mps < 0.25:
        return "NORMAL"
    elif mps < 0.50:
        return "WARNING"
    elif mps < 0.70:
        return "PLAN_MAINTENANCE"
    elif mps < 0.85:
        return "URGENT"
    return "CRITICAL"


def derive_prescription_type(contributions):
    ordered = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
    top_name, top_value = ordered[0]
    second_name, second_value = ordered[1]

    gap = float(top_value - second_value)

    if gap < MIXED_CAUSE_GAP_PERCENT:
        pair = {top_name, second_name}
        if pair == {"temperature", "current"}:
            return "ELECTRICAL_THERMAL", top_name, top_value, second_name, second_value
        if pair == {"temperature", "vibration"}:
            return "THERMAL_MECHANICAL", top_name, top_value, second_name, second_value
        if pair == {"current", "vibration"}:
            return "ELECTRICAL_MECHANICAL", top_name, top_value, second_name, second_value
        return "MIXED", top_name, top_value, second_name, second_value

    if top_name == "temperature":
        return "THERMAL", top_name, top_value, second_name, second_value
    if top_name == "vibration":
        return "MECHANICAL", top_name, top_value, second_name, second_value
    if top_name == "current":
        return "ELECTRICAL", top_name, top_value, second_name, second_value

    return "GENERAL", top_name, top_value, second_name, second_value


def build_prescription(urgency_level, prescription_type, top_cause, top_value, second_cause, second_value, health, rul, persistence, trend_factor):
    title_map = {
        "NORMAL": "Continue normal operation",
        "WARNING": "Early warning condition",
        "PLAN_MAINTENANCE": "Maintenance should be scheduled soon",
        "URGENT": "Urgent maintenance required",
        "CRITICAL": "Critical condition - immediate action required",
    }

    category_map = {
        "THERMAL": "Thermal maintenance",
        "MECHANICAL": "Mechanical maintenance",
        "ELECTRICAL": "Electrical/load-related maintenance",
        "THERMAL_MECHANICAL": "Thermal-mechanical maintenance",
        "ELECTRICAL_THERMAL": "Electrical-thermal maintenance",
        "ELECTRICAL_MECHANICAL": "Electrical-mechanical maintenance",
        "MIXED": "Mixed-condition maintenance",
        "GENERAL": "General diagnostic maintenance",
    }

    actions_map = {
        ("NORMAL", "THERMAL"): [
            "Continue monitoring",
            "Inspect cooling path during routine maintenance if trend worsens"
        ],
        ("WARNING", "THERMAL"): [
            "Increase monitoring frequency",
            "Inspect cooling fan",
            "Check ventilation openings"
        ],
        ("PLAN_MAINTENANCE", "THERMAL"): [
            "Schedule thermal inspection soon",
            "Inspect cooling fan",
            "Check ventilation",
            "Check overload condition"
        ],
        ("URGENT", "THERMAL"): [
            "Perform urgent thermal inspection",
            "Inspect cooling fan",
            "Check ventilation",
            "Reduce operating load if possible",
            "Prepare controlled shutdown if temperature keeps rising"
        ],
        ("CRITICAL", "THERMAL"): [
            "Immediate shutdown",
            "Inspect cooling system before restart",
            "Check overload condition",
            "Do not restart until fault is cleared"
        ],

        ("NORMAL", "MECHANICAL"): [
            "Continue monitoring",
            "Inspect bearings during routine maintenance if vibration trend worsens"
        ],
        ("WARNING", "MECHANICAL"): [
            "Increase monitoring frequency",
            "Inspect bearings",
            "Check looseness of mountings"
        ],
        ("PLAN_MAINTENANCE", "MECHANICAL"): [
            "Schedule mechanical inspection soon",
            "Inspect bearings",
            "Check shaft alignment",
            "Check looseness and balance"
        ],
        ("URGENT", "MECHANICAL"): [
            "Perform urgent mechanical inspection",
            "Inspect bearings and alignment immediately",
            "Reduce speed or load if possible",
            "Prepare controlled shutdown"
        ],
        ("CRITICAL", "MECHANICAL"): [
            "Immediate shutdown",
            "Inspect bearings, alignment, and looseness before restart",
            "Do not restart until fault is cleared"
        ],

        ("NORMAL", "ELECTRICAL"): [
            "Continue monitoring",
            "Inspect electrical path during routine maintenance if current trend worsens"
        ],
        ("WARNING", "ELECTRICAL"): [
            "Increase monitoring frequency",
            "Inspect wiring connections",
            "Check load condition"
        ],
        ("PLAN_MAINTENANCE", "ELECTRICAL"): [
            "Schedule electrical inspection soon",
            "Inspect wiring and terminals",
            "Check overload condition",
            "Inspect driven load for abnormal resistance"
        ],
        ("URGENT", "ELECTRICAL"): [
            "Perform urgent electrical inspection",
            "Inspect wiring, terminals, and load immediately",
            "Reduce operating load if possible",
            "Prepare controlled shutdown"
        ],
        ("CRITICAL", "ELECTRICAL"): [
            "Immediate shutdown",
            "Inspect wiring, terminals, load, and winding condition before restart",
            "Do not restart until fault is cleared"
        ],

        ("NORMAL", "THERMAL_MECHANICAL"): [
            "Continue monitoring",
            "Inspect cooling path and mechanical mounting during routine maintenance"
        ],
        ("WARNING", "THERMAL_MECHANICAL"): [
            "Increase monitoring frequency",
            "Inspect cooling fan",
            "Inspect bearings and looseness"
        ],
        ("PLAN_MAINTENANCE", "THERMAL_MECHANICAL"): [
            "Schedule combined thermal-mechanical inspection",
            "Inspect cooling system",
            "Inspect bearings and alignment"
        ],
        ("URGENT", "THERMAL_MECHANICAL"): [
            "Perform urgent combined thermal-mechanical inspection",
            "Reduce load if possible",
            "Prepare controlled shutdown"
        ],
        ("CRITICAL", "THERMAL_MECHANICAL"): [
            "Immediate shutdown",
            "Inspect cooling system, bearings, and alignment before restart"
        ],

        ("NORMAL", "ELECTRICAL_THERMAL"): [
            "Continue monitoring",
            "Inspect electrical load path and cooling path during routine maintenance"
        ],
        ("WARNING", "ELECTRICAL_THERMAL"): [
            "Increase monitoring frequency",
            "Inspect wiring connections",
            "Inspect cooling fan and ventilation"
        ],
        ("PLAN_MAINTENANCE", "ELECTRICAL_THERMAL"): [
            "Schedule combined electrical-thermal inspection",
            "Inspect wiring and terminals",
            "Inspect cooling system",
            "Check overload condition"
        ],
        ("URGENT", "ELECTRICAL_THERMAL"): [
            "Perform urgent electrical-thermal inspection",
            "Reduce load immediately if possible",
            "Prepare controlled shutdown"
        ],
        ("CRITICAL", "ELECTRICAL_THERMAL"): [
            "Immediate shutdown",
            "Inspect wiring, load path, and cooling system before restart"
        ],

        ("NORMAL", "ELECTRICAL_MECHANICAL"): [
            "Continue monitoring",
            "Inspect electrical load path and mechanical assembly during routine maintenance"
        ],
        ("WARNING", "ELECTRICAL_MECHANICAL"): [
            "Increase monitoring frequency",
            "Inspect wiring connections",
            "Inspect bearings and mountings"
        ],
        ("PLAN_MAINTENANCE", "ELECTRICAL_MECHANICAL"): [
            "Schedule combined electrical-mechanical inspection",
            "Inspect wiring and terminals",
            "Inspect bearings and alignment"
        ],
        ("URGENT", "ELECTRICAL_MECHANICAL"): [
            "Perform urgent electrical-mechanical inspection",
            "Reduce load if possible",
            "Prepare controlled shutdown"
        ],
        ("CRITICAL", "ELECTRICAL_MECHANICAL"): [
            "Immediate shutdown",
            "Inspect wiring, load path, bearings, and alignment before restart"
        ],

        ("NORMAL", "MIXED"): [
            "Continue monitoring",
            "Log condition and observe trend"
        ],
        ("WARNING", "MIXED"): [
            "Increase monitoring frequency",
            "Perform general diagnostic inspection"
        ],
        ("PLAN_MAINTENANCE", "MIXED"): [
            "Schedule general diagnostic inspection",
            "Inspect thermal, mechanical, and electrical subsystems"
        ],
        ("URGENT", "MIXED"): [
            "Perform urgent general inspection",
            "Reduce load if possible",
            "Prepare controlled shutdown"
        ],
        ("CRITICAL", "MIXED"): [
            "Immediate shutdown",
            "Perform full subsystem inspection before restart"
        ],

        ("NORMAL", "GENERAL"): [
            "Continue normal operation",
            "Continue monitoring"
        ],
        ("WARNING", "GENERAL"): [
            "Increase monitoring frequency",
            "Inspect during routine maintenance"
        ],
        ("PLAN_MAINTENANCE", "GENERAL"): [
            "Schedule maintenance soon",
            "Perform general diagnostic inspection"
        ],
        ("URGENT", "GENERAL"): [
            "Urgent maintenance required",
            "Prepare controlled shutdown"
        ],
        ("CRITICAL", "GENERAL"): [
            "Immediate shutdown",
            "Do not restart until inspected"
        ],
    }

    if urgency_level == "CRITICAL":
        auto_action = "SHUTDOWN"
    elif urgency_level == "URGENT":
        auto_action = "REDUCE_LOAD_OR_PREPARE_SHUTDOWN"
    elif urgency_level == "PLAN_MAINTENANCE":
        auto_action = "SCHEDULE_MAINTENANCE"
    else:
        auto_action = "MONITOR"

    category_label = category_map.get(prescription_type, "General diagnostic maintenance")
    prescription_title = f"{title_map.get(urgency_level, 'Maintenance advice')} - {category_label}"
    prescription_actions = actions_map.get((urgency_level, prescription_type), ["Continue monitoring"])

    prescription_reason = (
        f"{top_cause.capitalize()} is the dominant contributor at {float(top_value):.1f}%."
        f" Secondary contributor is {second_cause} at {float(second_value):.1f}%."
        f" Health is {float(health):.1f}%, estimated RUL is {float(rul):.1f} h,"
        f" persistence is {float(persistence) * 100.0:.1f}%, and trend factor is {float(trend_factor):.2f}."
    )

    return {
        "prescription_type": prescription_type,
        "prescription_category_label": category_label,
        "prescription_title": prescription_title,
        "prescription_actions": prescription_actions,
        "prescription_reason": prescription_reason,
        "auto_action": auto_action,
    }


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
# SAME OLD ROUTES, ONLY EXTENDED
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


@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "ok": backend_ready(),
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "threshold_loaded": threshold is not None,
        "threshold": round(float(threshold), 6) if threshold is not None else None,
        "failure_threshold": round(float(threshold * FAILURE_MULTIPLIER), 6) if threshold is not None else None,
        "history_length": len(smooth_error_history),
        "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS
    })


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

        # KEEP OLD INPUT CONTRACT
        readings = payload.get("readings")
        valid, message = validate_input(readings)
        if not valid:
            return jsonify({"error": message}), 400

        raw_window = np.array(readings, dtype=np.float32)
        scaled_window = scaler.transform(raw_window)
        x_input = np.expand_dims(scaled_window, axis=0)

        model_output = model(x_input, training=False)
        x_pred = model_output.numpy() if hasattr(model_output, "numpy") else np.asarray(model_output)

        reconstruction_error = compute_total_error(x_input, x_pred)

        # KEEP OLD ANOMALY DECISION
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

        # NEW PRESCRIPTIVE LAYER
        persistence = update_persistence(is_anomaly)
        mps_data = compute_mps(smoothed_error, threshold, health, rul, persistence)
        urgency_level = derive_urgency_level(mps_data["maintenance_priority_score"])

        prescription_type, top_cause, top_value, second_cause, second_value = derive_prescription_type(contributions)
        prescription_data = build_prescription(
            urgency_level=urgency_level,
            prescription_type=prescription_type,
            top_cause=top_cause,
            top_value=top_value,
            second_cause=second_cause,
            second_value=second_value,
            health=health,
            rul=rul,
            persistence=mps_data["persistence_factor"],
            trend_factor=mps_data["trend_factor"]
        )

        # OLD RESPONSE KEYS PRESERVED
        response = {
            "is_anomaly": bool(is_anomaly),
            "status": status,
            "led_status": led_status,
            "health": round(float(health), 2),
            "rul": round(float(rul), 2),
            "rul_hours": round(float(rul), 2),
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
            },

            # NEW PRESCRIPTIVE FIELDS
            "maintenance_priority_score": round(float(mps_data["maintenance_priority_score"]), 4),
            "anomaly_severity": round(float(mps_data["anomaly_severity"]), 4),
            "health_degradation": round(float(mps_data["health_degradation"]), 4),
            "rul_risk": round(float(mps_data["rul_risk"]), 4),
            "trend_factor": round(float(mps_data["trend_factor"]), 4),
            "persistence_factor": round(float(mps_data["persistence_factor"]), 4),
            "urgency_level": urgency_level,
            "prescription_type": prescription_data["prescription_type"],
            "prescription_category_label": prescription_data["prescription_category_label"],
            "prescription_title": prescription_data["prescription_title"],
            "prescription_actions": prescription_data["prescription_actions"],
            "prescription_reason": prescription_data["prescription_reason"],
            "auto_action": prescription_data["auto_action"],
            "analysis_timestamp": int(time.time()),
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