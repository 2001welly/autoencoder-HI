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
FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}

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

INSUFFICIENT_HISTORY_HEALTH_WEIGHT = 0.6
STABLE_HEALTH_WEIGHT = 0.8
DEGRADING_PROJECTED_WEIGHT = 0.7
DEGRADING_HEALTH_WEIGHT = 0.3

# ============================================================
# ADAPTIVE THRESHOLD SETTINGS
# ============================================================
ADAPTIVE_THRESHOLD_ENABLED = os.getenv("ADAPTIVE_THRESHOLD_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
ADAPTIVE_HISTORY_SIZE = int(os.getenv("ADAPTIVE_HISTORY_SIZE", "100"))
ADAPTIVE_MIN_HEALTHY_POINTS = int(os.getenv("ADAPTIVE_MIN_HEALTHY_POINTS", "20"))
ADAPTIVE_BLEND = float(os.getenv("ADAPTIVE_BLEND", "0.35"))
ADAPTIVE_STD_MULTIPLIER = float(os.getenv("ADAPTIVE_STD_MULTIPLIER", "3.0"))
ADAPTIVE_PERCENTILE = float(os.getenv("ADAPTIVE_PERCENTILE", "95.0"))
ADAPTIVE_MIN_RATIO = float(os.getenv("ADAPTIVE_MIN_RATIO", "0.80"))
ADAPTIVE_MAX_RATIO = float(os.getenv("ADAPTIVE_MAX_RATIO", "1.50"))
ADAPTIVE_UPDATE_MAX_ERROR_RATIO = float(os.getenv("ADAPTIVE_UPDATE_MAX_ERROR_RATIO", "0.90"))
ADAPTIVE_UPDATE_MIN_HEALTH = float(os.getenv("ADAPTIVE_UPDATE_MIN_HEALTH", "75.0"))
ADAPTIVE_MAX_OOD_FOR_UPDATE = float(os.getenv("ADAPTIVE_MAX_OOD_FOR_UPDATE", "0.12"))

# ============================================================
# PRESCRIPTIVE LAYER SETTINGS
# ============================================================
DOMINANT_CONTRIBUTION_THRESHOLD = float(os.getenv("DOMINANT_CONTRIBUTION_THRESHOLD", "45.0"))
PERSISTENCE_LOOKBACK = int(os.getenv("PERSISTENCE_LOOKBACK", "8"))
OPERATING_BAND_MARGIN_RATIO = float(os.getenv("OPERATING_BAND_MARGIN_RATIO", "0.05"))
WARMUP_TEMP_MARGIN_RATIO = float(os.getenv("WARMUP_TEMP_MARGIN_RATIO", "0.02"))

MPS_WEIGHTS = {
    "anomaly_severity": 0.28,
    "health_degradation": 0.22,
    "rul_risk": 0.22,
    "trend": 0.15,
    "persistence": 0.13,
}

URGENCY_ORDER = ["NORMAL", "WARNING", "PLAN_MAINTENANCE", "URGENT", "CRITICAL"]

PRESCRIPTION_RULES = {
    ("temperature", "LOW"): {
        "type": "thermal_low",
        "label": "Cold / Warm-up Management",
        "context": "Cold thermal state",
        "actions": [
            "Allow the machine to warm up to the normal operating temperature",
            "Continue monitoring temperature rise and stabilisation",
            "Check ambient conditions and confirm the machine is actually under load",
            "Verify the temperature sensor position and reading",
        ],
        "urgent_extra": "Inspect loss of heating or unexpectedly low process temperature if the machine should already be hot",
        "critical_extra": "Escalate only if temperature remains abnormally low after warm-up or if production requires a hotter operating state",
        "normal_title": "Normal thermal condition - continue monitoring",
        "warning_title": "Warning - monitor cold thermal condition",
        "plan_title": "Plan inspection for persistent low temperature state",
        "urgent_title": "Urgent check for persistent low temperature state",
        "critical_title": "Critical low temperature investigation required",
        "auto_normal": "monitor",
        "auto_warning": "monitor",
        "auto_plan": "schedule_inspection",
        "auto_urgent": "inspect_process_condition",
        "auto_critical": "inspect_process_condition",
    },
    ("temperature", "HIGH"): {
        "type": "thermal_high",
        "label": "Thermal Maintenance",
        "context": "High temperature / overheating state",
        "actions": [
            "Inspect cooling fan",
            "Check ventilation",
            "Check overload condition",
        ],
        "urgent_extra": "Reduce load if possible",
        "critical_extra": "Shut down if temperature continues rising or cooling is ineffective",
        "normal_title": "Normal thermal condition - continue monitoring",
        "warning_title": "Warning - monitor thermal condition",
        "plan_title": "Plan thermal maintenance",
        "urgent_title": "Urgent thermal maintenance required",
        "critical_title": "Critical thermal action required",
        "auto_normal": "monitor",
        "auto_warning": "monitor",
        "auto_plan": "schedule_maintenance",
        "auto_urgent": "reduce_load",
        "auto_critical": "shutdown",
    },
    ("current", "LOW"): {
        "type": "electrical_low",
        "label": "Underload / Supply Check",
        "context": "Low current / underload state",
        "actions": [
            "Check whether the machine is idling, lightly loaded, or disconnected from the intended load",
            "Inspect wiring continuity and loose or open terminals",
            "Verify load coupling and confirm the motor is actually driving the intended mechanism",
            "Check current sensor calibration and wiring",
        ],
        "urgent_extra": "Inspect supply continuity immediately if the machine should already be drawing normal current",
        "critical_extra": "Isolate and inspect the circuit only if the machine is expected to be heavily loaded but current remains abnormally low",
        "normal_title": "Normal electrical condition - continue monitoring",
        "warning_title": "Warning - monitor low current condition",
        "plan_title": "Plan inspection for persistent low current state",
        "urgent_title": "Urgent low current investigation required",
        "critical_title": "Critical low current investigation required",
        "auto_normal": "monitor",
        "auto_warning": "monitor",
        "auto_plan": "schedule_inspection",
        "auto_urgent": "inspect_supply_condition",
        "auto_critical": "inspect_supply_condition",
    },
    ("current", "HIGH"): {
        "type": "electrical_high",
        "label": "Electrical / Load Maintenance",
        "context": "High current / overload state",
        "actions": [
            "Inspect wiring",
            "Inspect terminals",
            "Inspect load condition",
            "Inspect winding stress",
        ],
        "urgent_extra": "Reduce load if possible",
        "critical_extra": "Shut down and isolate power if electrical stress persists",
        "normal_title": "Normal electrical condition - continue monitoring",
        "warning_title": "Warning - monitor electrical load condition",
        "plan_title": "Plan electrical maintenance",
        "urgent_title": "Urgent electrical maintenance required",
        "critical_title": "Critical electrical action required",
        "auto_normal": "monitor",
        "auto_warning": "monitor",
        "auto_plan": "schedule_maintenance",
        "auto_urgent": "reduce_load_and_inspect_electrical",
        "auto_critical": "shutdown",
    },
    ("vibration", "LOW"): {
        "type": "vibration_low",
        "label": "Low Vibration / Idle State",
        "context": "Low vibration / lightly-loaded state",
        "actions": [
            "Confirm whether the machine is idling or lightly loaded",
            "Continue monitoring for transition into the normal operating vibration band",
            "Verify vibration sensor mounting and signal quality",
        ],
        "urgent_extra": "Inspect the sensing chain if the machine should already be under normal mechanical load",
        "critical_extra": "Investigate sensor integrity or unexpected idle condition if low vibration persists with abnormal process behaviour",
        "normal_title": "Normal mechanical condition - continue monitoring",
        "warning_title": "Warning - monitor low vibration condition",
        "plan_title": "Plan inspection for persistent low vibration state",
        "urgent_title": "Urgent low vibration investigation required",
        "critical_title": "Critical low vibration investigation required",
        "auto_normal": "monitor",
        "auto_warning": "monitor",
        "auto_plan": "schedule_inspection",
        "auto_urgent": "inspect_sensor_and_machine_state",
        "auto_critical": "inspect_sensor_and_machine_state",
    },
    ("vibration", "HIGH"): {
        "type": "mechanical_high",
        "label": "Mechanical Maintenance",
        "context": "High vibration / mechanical fault state",
        "actions": [
            "Inspect bearings",
            "Inspect shaft alignment",
            "Check looseness",
            "Check imbalance",
        ],
        "urgent_extra": "Reduce load if possible",
        "critical_extra": "Shut down immediately to prevent bearing or shaft damage",
        "normal_title": "Normal mechanical condition - continue monitoring",
        "warning_title": "Warning - monitor mechanical condition",
        "plan_title": "Plan mechanical maintenance",
        "urgent_title": "Urgent mechanical maintenance required",
        "critical_title": "Critical mechanical action required",
        "auto_normal": "monitor",
        "auto_warning": "monitor",
        "auto_plan": "schedule_maintenance",
        "auto_urgent": "reduce_load",
        "auto_critical": "shutdown",
    },
    ("mixed", "MIXED"): {
        "type": "general_inspection",
        "label": "General Inspection",
        "context": "Mixed or unclear dominant condition",
        "actions": [
            "Perform a general inspection of current, temperature, and vibration",
            "Check operating load and ambient conditions",
            "Review recent maintenance history and recent alarms",
        ],
        "urgent_extra": "Reduce load if possible while the inspection is carried out",
        "critical_extra": "Shut down if the condition keeps deteriorating after inspection",
        "normal_title": "Normal operation - continue monitoring",
        "warning_title": "Warning - general inspection recommended",
        "plan_title": "Plan general inspection",
        "urgent_title": "Urgent general inspection required",
        "critical_title": "Critical inspection action required",
        "auto_normal": "monitor",
        "auto_warning": "monitor",
        "auto_plan": "schedule_maintenance",
        "auto_urgent": "reduce_load",
        "auto_critical": "shutdown",
    },
    ("observe", "NORMAL"): {
        "type": "observe",
        "label": "Observe / No Immediate Maintenance",
        "context": "Normal operating state",
        "actions": [
            "Continue normal operation",
            "Keep monitoring current, temperature, and vibration trends",
        ],
        "urgent_extra": "",
        "critical_extra": "",
        "normal_title": "Normal operation - continue monitoring",
        "warning_title": "Warning - continue close monitoring",
        "plan_title": "Plan follow-up monitoring",
        "urgent_title": "Urgent review of monitoring trends",
        "critical_title": "Critical review required",
        "auto_normal": "none",
        "auto_warning": "monitor",
        "auto_plan": "monitor",
        "auto_urgent": "inspect",
        "auto_critical": "inspect",
    },
}

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
adaptive_healthy_error_history = deque(maxlen=ADAPTIVE_HISTORY_SIZE)
adaptive_threshold_history = deque(maxlen=ERROR_HISTORY_SIZE)
last_adaptive_threshold = None


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
        return {name: 0.0 for name in FEATURE_NAMES}, "unknown"

    perc = (feature_errors / total) * 100.0
    contributions = {name: float(perc[i]) for i, name in enumerate(FEATURE_NAMES)}
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


def compute_adaptive_threshold(base_threshold, healthy_errors):
    if not ADAPTIVE_THRESHOLD_ENABLED:
        return float(base_threshold), False

    count = len(healthy_errors)
    if count < ADAPTIVE_MIN_HEALTHY_POINTS:
        return float(base_threshold), False

    values = np.array(healthy_errors, dtype=np.float64)
    local_mean = float(np.mean(values))
    local_std = float(np.std(values))
    local_percentile = float(np.percentile(values, ADAPTIVE_PERCENTILE))

    candidate = max(local_mean + ADAPTIVE_STD_MULTIPLIER * local_std, local_percentile)
    blended = (1.0 - ADAPTIVE_BLEND) * float(base_threshold) + ADAPTIVE_BLEND * candidate

    lower_bound = float(base_threshold) * ADAPTIVE_MIN_RATIO
    upper_bound = float(base_threshold) * ADAPTIVE_MAX_RATIO
    adaptive_threshold = clamp(blended, lower_bound, upper_bound)
    return float(adaptive_threshold), True


def get_adaptive_history_summary(healthy_errors, base_threshold):
    if len(healthy_errors) == 0:
        return {
            "count": 0,
            "ready": False,
            "mean": None,
            "std": None,
            "percentile": None,
            "base_threshold": float(base_threshold),
        }

    values = np.array(healthy_errors, dtype=np.float64)
    return {
        "count": int(len(values)),
        "ready": bool(len(values) >= ADAPTIVE_MIN_HEALTHY_POINTS),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "percentile": float(np.percentile(values, ADAPTIVE_PERCENTILE)),
        "base_threshold": float(base_threshold),
    }


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
        DEGRADING_PROJECTED_WEIGHT * projected_hours
        + DEGRADING_HEALTH_WEIGHT * health_reserve
    )
    return float(clamp(rul, 0.0, MAX_RUL_HOURS)), "degrading"


def compute_ood_score(raw_window, scaler_obj):
    if scaler_obj is None:
        return None, {}, {}, None

    mins = getattr(scaler_obj, "data_min_", None)
    maxs = getattr(scaler_obj, "data_max_", None)

    if mins is None or maxs is None:
        return None, {}, {}, None

    latest = np.array(raw_window[-1], dtype=np.float64)
    mins = np.array(mins, dtype=np.float64)
    maxs = np.array(maxs, dtype=np.float64)
    span = np.maximum(maxs - mins, 1e-6)

    below = np.maximum((mins - latest) / span, 0.0)
    above = np.maximum((latest - maxs) / span, 0.0)
    violation = below + above

    details = {name: float(violation[i]) for i, name in enumerate(FEATURE_NAMES)}
    direction_details = {}
    for i, name in enumerate(FEATURE_NAMES):
        if below[i] > 0:
            direction_details[name] = "LOW"
        elif above[i] > 0:
            direction_details[name] = "HIGH"
        else:
            direction_details[name] = "NORMAL"

    main_ood_feature = FEATURE_NAMES[int(np.argmax(violation))] if np.max(violation) > 0 else None
    ood_score = float(np.mean(violation))
    return ood_score, details, direction_details, main_ood_feature


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
        0.35 * error_fluctuation
        + 0.30 * trend_instability
        + 0.20 * out_of_distribution
        + 0.15 * limited_history
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
        "limited_history": round(limited_history, 4),
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


def derive_status_and_led(is_anomaly, health_value, urgency_level=None):
    if urgency_level in {"CRITICAL", "URGENT"}:
        return "Anomaly", "RED"
    if urgency_level in {"WARNING", "PLAN_MAINTENANCE"}:
        return "Warning", "YELLOW"
    if is_anomaly or health_value <= 20:
        return "Anomaly", "RED"
    if health_value <= 60:
        return "Warning", "YELLOW"
    return "Normal", "GREEN"


# ============================================================
# OPERATING REGION HELPERS
# ============================================================
def env_float(names, default_value):
    for name in names:
        value = os.getenv(name)
        if value is not None and str(value).strip() != "":
            try:
                return float(value)
            except Exception:
                pass
    return float(default_value)


def get_operating_bands(scaler_obj):
    mins = getattr(scaler_obj, "data_min_", None)
    maxs = getattr(scaler_obj, "data_max_", None)

    if mins is None or maxs is None:
        return {
            "current": {"low": 0.0, "high": 9999.0},
            "temperature": {"low": -50.0, "high": 150.0},
            "vibration": {"low": 0.0, "high": 9999.0},
        }

    bands = {}
    for name in FEATURE_NAMES:
        idx = FEATURE_INDEX[name]
        data_min = float(mins[idx])
        data_max = float(maxs[idx])
        span = max(data_max - data_min, 1e-6)
        default_low = data_min - OPERATING_BAND_MARGIN_RATIO * span
        default_high = data_max + OPERATING_BAND_MARGIN_RATIO * span

        aliases_low = [f"{name.upper()}_NORMAL_LOW"]
        aliases_high = [f"{name.upper()}_NORMAL_HIGH"]
        if name == "temperature":
            aliases_low.insert(0, "TEMP_NORMAL_LOW")
            aliases_high.insert(0, "TEMP_NORMAL_HIGH")
        if name == "vibration":
            aliases_low.insert(0, "VIB_NORMAL_LOW")
            aliases_high.insert(0, "VIB_NORMAL_HIGH")

        low = env_float(aliases_low, default_low)
        high = env_float(aliases_high, default_high)
        if low > high:
            low, high = high, low
        bands[name] = {"low": float(low), "high": float(high)}

    return bands


def classify_state(value, low, high):
    if value < low:
        return "LOW"
    if value > high:
        return "HIGH"
    return "NORMAL"


def compute_sensor_states(latest_values, bands):
    states = {}
    distances = {}
    for name in FEATURE_NAMES:
        value = float(latest_values[name])
        low = float(bands[name]["low"])
        high = float(bands[name]["high"])
        states[name] = classify_state(value, low, high)
        if value < low:
            distances[name] = float(low - value)
        elif value > high:
            distances[name] = float(value - high)
        else:
            distances[name] = 0.0
    return states, distances


def compute_condition_warmup_flag(raw_window, bands):
    latest_temp = float(raw_window[-1][FEATURE_INDEX["temperature"]])
    temp_low = float(bands["temperature"]["low"])
    temp_high = float(bands["temperature"]["high"])
    span = max(temp_high - temp_low, 1e-6)
    warmup_exit_temp = temp_low + WARMUP_TEMP_MARGIN_RATIO * span
    recent_temp_mean = float(np.mean(raw_window[-5:, FEATURE_INDEX["temperature"]]))
    warmup_active = latest_temp < warmup_exit_temp and recent_temp_mean < warmup_exit_temp
    return bool(warmup_active), float(warmup_exit_temp)


def should_update_adaptive_history(raw_window, reconstruction_error, active_threshold, health, ood_score, scaler_obj):
    if not ADAPTIVE_THRESHOLD_ENABLED:
        return False, "adaptive_disabled", False, None

    operating_bands = get_operating_bands(scaler_obj)
    warmup_like, warmup_exit_temp = compute_condition_warmup_flag(raw_window, operating_bands)
    if warmup_like:
        return False, "warmup_like_condition", True, float(warmup_exit_temp)

    if health < ADAPTIVE_UPDATE_MIN_HEALTH:
        return False, "health_below_update_limit", False, float(warmup_exit_temp)

    if reconstruction_error > active_threshold * ADAPTIVE_UPDATE_MAX_ERROR_RATIO:
        return False, "error_too_close_to_threshold", False, float(warmup_exit_temp)

    if ood_score is not None and ood_score > ADAPTIVE_MAX_OOD_FOR_UPDATE:
        return False, "out_of_distribution", False, float(warmup_exit_temp)

    return True, "accepted", False, float(warmup_exit_temp)


def maybe_update_adaptive_history(raw_window, reconstruction_error, active_threshold, health, ood_score, scaler_obj):
    should_update, reason, warmup_like, warmup_exit_temp = should_update_adaptive_history(
        raw_window=raw_window,
        reconstruction_error=reconstruction_error,
        active_threshold=active_threshold,
        health=health,
        ood_score=ood_score,
        scaler_obj=scaler_obj,
    )

    if should_update:
        adaptive_healthy_error_history.append(float(reconstruction_error))

    return {
        "applied": bool(should_update),
        "reason": reason,
        "warmup_like": bool(warmup_like),
        "warmup_exit_temperature": warmup_exit_temp,
        "history_count": len(adaptive_healthy_error_history),
        "history_ready": len(adaptive_healthy_error_history) >= ADAPTIVE_MIN_HEALTHY_POINTS,
    }


def compute_anomaly_severity(smoothed_error, anomaly_threshold):
    failure_threshold = anomaly_threshold * FAILURE_MULTIPLIER
    if smoothed_error <= anomaly_threshold:
        return 0.0
    if smoothed_error >= failure_threshold:
        return 1.0
    severity = (smoothed_error - anomaly_threshold) / (failure_threshold - anomaly_threshold)
    return float(clamp(severity, 0.0, 1.0))


def compute_persistence_factor(anomaly_threshold):
    history = list(smooth_error_history)[-PERSISTENCE_LOOKBACK:]
    if not history:
        return 0.0
    count_above = sum(1 for value in history if value > anomaly_threshold)
    return float(clamp(count_above / float(len(history)), 0.0, 1.0))


def compute_trend_factor(anomaly_threshold, slope):
    if slope is None or slope <= 0:
        return 0.0
    horizon_hours = max(MIN_RUL_POINTS * SAMPLE_INTERVAL_HOURS, SAMPLE_INTERVAL_HOURS)
    projected_increase = slope * horizon_hours
    scale = max(0.75 * anomaly_threshold, 1e-6)
    factor = projected_increase / scale
    return float(clamp(factor, 0.0, 1.0))


def compute_maintenance_priority(health, rul, smoothed_error, anomaly_threshold, slope):
    anomaly_severity = compute_anomaly_severity(smoothed_error, anomaly_threshold)
    health_degradation = float(clamp(1.0 - (health / 100.0), 0.0, 1.0))
    rul_risk = float(clamp(1.0 - (rul / MAX_RUL_HOURS), 0.0, 1.0))
    trend_factor = compute_trend_factor(anomaly_threshold, slope)
    persistence_factor = compute_persistence_factor(anomaly_threshold)

    weighted = (
        MPS_WEIGHTS["anomaly_severity"] * anomaly_severity
        + MPS_WEIGHTS["health_degradation"] * health_degradation
        + MPS_WEIGHTS["rul_risk"] * rul_risk
        + MPS_WEIGHTS["trend"] * trend_factor
        + MPS_WEIGHTS["persistence"] * persistence_factor
    )

    mps = 100.0 * weighted
    factors = {
        "anomaly_severity": round(anomaly_severity, 4),
        "health_degradation": round(health_degradation, 4),
        "rul_risk": round(rul_risk, 4),
        "trend_factor": round(trend_factor, 4),
        "persistence_factor": round(persistence_factor, 4),
    }
    return float(clamp(mps, 0.0, 100.0)), factors


def determine_urgency_level(mps, health, rul, anomaly_severity):
    if mps >= 85 or health <= 15 or rul <= 4 or anomaly_severity >= 0.90:
        return "CRITICAL"
    if mps >= 65 or health <= 30 or rul <= 12:
        return "URGENT"
    if mps >= 45 or health <= 55 or rul <= 30:
        return "PLAN_MAINTENANCE"
    if mps >= 20 or health <= 75 or anomaly_severity > 0:
        return "WARNING"
    return "NORMAL"


def cap_urgency(urgency_level, max_allowed):
    current_index = URGENCY_ORDER.index(urgency_level)
    max_index = URGENCY_ORDER.index(max_allowed)
    return URGENCY_ORDER[min(current_index, max_index)]


def determine_dominant_feature(contributions, is_anomaly, mps):
    if not contributions:
        return "observe", 0.0
    dominant_feature = max(contributions, key=contributions.get)
    dominant_pct = float(contributions.get(dominant_feature, 0.0))
    if not is_anomaly and mps < 20:
        return "observe", dominant_pct
    if dominant_pct < DOMINANT_CONTRIBUTION_THRESHOLD:
        return "mixed", dominant_pct
    return dominant_feature, dominant_pct


def determine_operating_region(is_anomaly, sensor_states, dominant_feature, dominant_state, warmup_like):
    if warmup_like and sensor_states.get("temperature") == "LOW":
        return "WARMUP", "Warm-up / cold-state operation"
    if not is_anomaly and all(sensor_states.get(name) == "NORMAL" for name in FEATURE_NAMES):
        return "NORMAL_OPERATION", "Normal operating region"
    if dominant_feature == "temperature" and dominant_state == "LOW":
        return "COLD_THERMAL_STATE", "Cold thermal state"
    if dominant_feature == "temperature" and dominant_state == "HIGH":
        return "OVERHEAT", "Overheating thermal state"
    if dominant_feature == "current" and dominant_state == "LOW":
        return "UNDERLOAD", "Low current / underload state"
    if dominant_feature == "current" and dominant_state == "HIGH":
        return "OVERLOAD", "High current / overload state"
    if dominant_feature == "vibration" and dominant_state == "LOW":
        return "IDLE_OR_LIGHT_LOAD", "Low vibration / lightly-loaded state"
    if dominant_feature == "vibration" and dominant_state == "HIGH":
        return "HIGH_VIBRATION", "High vibration / mechanical fault state"
    return "GENERAL_DIAGNOSTIC", "Mixed operating state"


def contextualise_priority(mps, urgency_level, dominant_feature, dominant_state, operating_region):
    adjusted_mps = float(mps)
    adjusted_urgency = urgency_level

    if operating_region == "WARMUP":
        adjusted_mps = min(adjusted_mps, 35.0)
        adjusted_urgency = cap_urgency(adjusted_urgency, "WARNING")
    elif dominant_feature in {"temperature", "current"} and dominant_state == "LOW":
        adjusted_mps = min(adjusted_mps, 55.0)
        adjusted_urgency = cap_urgency(adjusted_urgency, "PLAN_MAINTENANCE")
    elif dominant_feature == "vibration" and dominant_state == "LOW":
        adjusted_mps = min(adjusted_mps, 35.0)
        adjusted_urgency = cap_urgency(adjusted_urgency, "WARNING")

    return float(adjusted_mps), adjusted_urgency


def select_rule_key(dominant_feature, dominant_state, is_anomaly, sensor_states):
    if dominant_feature == "observe":
        return ("observe", "NORMAL")
    if dominant_feature == "mixed":
        return ("mixed", "MIXED")
    if dominant_state not in {"LOW", "HIGH"}:
        if is_anomaly:
            return ("mixed", "MIXED")
        return ("observe", "NORMAL")
    return (dominant_feature, dominant_state)


def build_actions(rule, urgency_level):
    actions = list(rule["actions"])
    if urgency_level == "URGENT" and rule.get("urgent_extra"):
        actions.append(rule["urgent_extra"])
    if urgency_level == "CRITICAL" and rule.get("urgent_extra") and rule["urgent_extra"] not in actions:
        actions.append(rule["urgent_extra"])
    if urgency_level == "CRITICAL" and rule.get("critical_extra"):
        actions.append(rule["critical_extra"])
    return actions


def build_prescription_title(rule, urgency_level):
    if urgency_level == "NORMAL":
        return rule["normal_title"]
    if urgency_level == "WARNING":
        return rule["warning_title"]
    if urgency_level == "PLAN_MAINTENANCE":
        return rule["plan_title"]
    if urgency_level == "URGENT":
        return rule["urgent_title"]
    return rule["critical_title"]


def build_auto_action(rule, urgency_level):
    if urgency_level == "NORMAL":
        return rule["auto_normal"]
    if urgency_level == "WARNING":
        return rule["auto_warning"]
    if urgency_level == "PLAN_MAINTENANCE":
        return rule["auto_plan"]
    if urgency_level == "URGENT":
        return rule["auto_urgent"]
    return rule["auto_critical"]


def build_prescription_reason(
    urgency_level,
    mps,
    dominant_feature,
    dominant_pct,
    dominant_state,
    operating_region,
    health,
    rul,
    factors,
    warmup_like,
):
    if dominant_feature == "observe":
        return (
            f"Machine condition is currently stable. Operating region is {operating_region}. "
            f"MPS is {mps:.2f}, health is {health:.1f}%, and RUL is {rul:.1f} h, so the system recommends monitoring only."
        )

    state_text = dominant_state.lower() if dominant_state else "unknown"
    dominant_text = (
        f"{dominant_feature.capitalize()} is the dominant contributor ({dominant_pct:.1f}%) and is in a {state_text} state. "
        if dominant_feature not in {"mixed", "observe"}
        else "No single sensor is strongly dominant, so a mixed inspection is recommended. "
    )

    warmup_text = "This looks like a cold/warm-up related condition. " if warmup_like else ""

    return (
        warmup_text
        + dominant_text
        + f"Operating region is {operating_region}. "
        + f"Urgency is {urgency_level} with MPS {mps:.2f}. "
        + f"Health is {health:.1f}% and estimated RUL is {rul:.1f} h. "
        + "Score drivers: "
        + f"anomaly severity {factors['anomaly_severity']:.2f}, "
        + f"health degradation {factors['health_degradation']:.2f}, "
        + f"RUL risk {factors['rul_risk']:.2f}, "
        + f"trend {factors['trend_factor']:.2f}, "
        + f"fault persistence {factors['persistence_factor']:.2f}."
    )


def compute_prescriptive_layer(raw_window, contributions, is_anomaly, health, rul, smoothed_error, anomaly_threshold, slope, scaler_obj):
    latest_values = {
        "current": float(raw_window[-1][FEATURE_INDEX["current"]]),
        "temperature": float(raw_window[-1][FEATURE_INDEX["temperature"]]),
        "vibration": float(raw_window[-1][FEATURE_INDEX["vibration"]]),
    }
    operating_bands = get_operating_bands(scaler_obj)
    sensor_states, band_distances = compute_sensor_states(latest_values, operating_bands)
    warmup_like, warmup_exit_temp = compute_condition_warmup_flag(raw_window, operating_bands)

    mps, mps_factors = compute_maintenance_priority(
        health=health,
        rul=rul,
        smoothed_error=smoothed_error,
        anomaly_threshold=anomaly_threshold,
        slope=slope,
    )

    dominant_feature, dominant_pct = determine_dominant_feature(contributions, is_anomaly, mps)
    dominant_state = sensor_states.get(dominant_feature, "NORMAL") if dominant_feature in FEATURE_NAMES else (
        "MIXED" if dominant_feature == "mixed" else "NORMAL"
    )
    operating_region, prescription_context = determine_operating_region(
        is_anomaly=is_anomaly,
        sensor_states=sensor_states,
        dominant_feature=dominant_feature,
        dominant_state=dominant_state,
        warmup_like=warmup_like,
    )

    urgency_level = determine_urgency_level(
        mps=mps,
        health=health,
        rul=rul,
        anomaly_severity=mps_factors["anomaly_severity"],
    )
    adjusted_mps, urgency_level = contextualise_priority(
        mps=mps,
        urgency_level=urgency_level,
        dominant_feature=dominant_feature,
        dominant_state=dominant_state,
        operating_region=operating_region,
    )

    rule_key = select_rule_key(dominant_feature, dominant_state, is_anomaly, sensor_states)
    rule = PRESCRIPTION_RULES[rule_key]

    prescription_title = build_prescription_title(rule, urgency_level)
    prescription_actions = build_actions(rule, urgency_level)
    prescription_reason = build_prescription_reason(
        urgency_level=urgency_level,
        mps=adjusted_mps,
        dominant_feature=dominant_feature,
        dominant_pct=dominant_pct,
        dominant_state=dominant_state,
        operating_region=operating_region,
        health=health,
        rul=rul,
        factors=mps_factors,
        warmup_like=warmup_like,
    )
    auto_action = build_auto_action(rule, urgency_level)

    return {
        "maintenance_priority_score": round(float(adjusted_mps), 2),
        "urgency_level": urgency_level,
        "prescription_type": rule["type"],
        "prescription_category_label": rule["label"],
        "prescription_title": prescription_title,
        "prescription_actions": prescription_actions,
        "prescription_reason": prescription_reason,
        "prescription_context": prescription_context,
        "auto_action": auto_action,
        "anomaly_severity": round(float(mps_factors["anomaly_severity"]), 4),
        "health_degradation": round(float(mps_factors["health_degradation"]), 4),
        "rul_risk": round(float(mps_factors["rul_risk"]), 4),
        "trend_factor": round(float(mps_factors["trend_factor"]), 4),
        "persistence_factor": round(float(mps_factors["persistence_factor"]), 4),
        "current_state": sensor_states["current"],
        "temperature_state": sensor_states["temperature"],
        "vibration_state": sensor_states["vibration"],
        "dominant_feature": dominant_feature,
        "dominant_state": dominant_state,
        "operating_region": operating_region,
        "condition_warmup_flag": bool(warmup_like),
        "warmup_exit_temperature": round(float(warmup_exit_temp), 4),
        "operating_bands": {
            name: {
                "low": round(float(operating_bands[name]["low"]), 4),
                "high": round(float(operating_bands[name]["high"]), 4),
            }
            for name in FEATURE_NAMES
        },
        "state_distance": {name: round(float(band_distances[name]), 4) for name in FEATURE_NAMES},
    }


# ============================================================
# MODEL LOADING
# ============================================================
def load_all():
    global model, scaler, threshold, is_loaded, last_adaptive_threshold

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
    last_adaptive_threshold = threshold_local
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
    return jsonify(
        {
            "message": "State-aware prescriptive maintenance backend is running.",
            "ready": backend_ready(),
            "model_loaded": model is not None,
            "scaler_loaded": scaler is not None,
            "threshold_loaded": threshold is not None,
            "threshold": last_adaptive_threshold if last_adaptive_threshold is not None else threshold,
            "base_threshold": threshold,
            "adaptive_threshold": last_adaptive_threshold,
            "failure_threshold": round(float((last_adaptive_threshold if last_adaptive_threshold is not None else threshold) * FAILURE_MULTIPLIER), 6) if threshold is not None else None,
            "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
            "dominant_contribution_threshold": DOMINANT_CONTRIBUTION_THRESHOLD,
            "operating_band_margin_ratio": OPERATING_BAND_MARGIN_RATIO,
            "adaptive_threshold_enabled": ADAPTIVE_THRESHOLD_ENABLED,
            "adaptive_history_size": ADAPTIVE_HISTORY_SIZE,
            "adaptive_min_healthy_points": ADAPTIVE_MIN_HEALTHY_POINTS,
            "adaptive_history_count": len(adaptive_healthy_error_history),
            "startup_error": startup_error,
            "model_path": MODEL_PATH,
            "scaler_path": SCALER_PATH,
            "threshold_path": THRESHOLD_PATH,
        }
    )


@app.route("/health", methods=["GET"])
def health_check():
    code = 200 if backend_ready() else 503
    return (
        jsonify(
            {
                "ok": backend_ready(),
                "model_loaded": model is not None,
                "scaler_loaded": scaler is not None,
                "threshold_loaded": threshold is not None,
                "startup_error": startup_error,
            }
        ),
        code,
    )


@app.route("/batch", methods=["POST"])
def batch_predict():
    global last_adaptive_threshold

    if not ensure_loaded():
        return (
            jsonify({
                "error": "Backend not ready",
                "details": startup_error or "Model, scaler, or threshold not loaded.",
            }),
            503,
        )

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

        base_threshold = float(threshold)
        adaptive_threshold, adaptive_ready = compute_adaptive_threshold(base_threshold, adaptive_healthy_error_history)
        active_threshold = float(adaptive_threshold)
        last_adaptive_threshold = active_threshold
        adaptive_threshold_history.append(active_threshold)

        reconstruction_error = compute_total_error(x_input, x_pred)
        is_anomaly = reconstruction_error > active_threshold
        smoothed_error = update_smoothed_error(reconstruction_error)
        health = compute_health(smoothed_error, active_threshold)

        feature_errors = compute_feature_errors(x_input, x_pred)
        contributions, main_cause = compute_sensor_contributions(feature_errors)

        rul, rul_state = estimate_rul(smoothed_error, active_threshold)
        slope = estimate_trend()

        ood_score, ood_details, ood_direction_details, ood_feature = compute_ood_score(raw_window, scaler)
        confidence_score, confidence_level, confidence_sources = compute_confidence(ood_score, active_threshold)
        rul_min, rul_max, rul_std = compute_rul_range(rul, confidence_score, health, rul_state)
        uncertainty_reason = build_uncertainty_reason(ood_score, ood_feature, confidence_sources)

        prescriptive = compute_prescriptive_layer(
            raw_window=raw_window,
            contributions=contributions,
            is_anomaly=is_anomaly,
            health=health,
            rul=rul,
            smoothed_error=smoothed_error,
            anomaly_threshold=active_threshold,
            slope=slope,
            scaler_obj=scaler,
        )

        adaptive_update = maybe_update_adaptive_history(
            raw_window=raw_window,
            reconstruction_error=reconstruction_error,
            active_threshold=active_threshold,
            health=health,
            ood_score=ood_score,
            scaler_obj=scaler,
        )
        adaptive_summary = get_adaptive_history_summary(adaptive_healthy_error_history, base_threshold)

        status, led_status = derive_status_and_led(is_anomaly, health, prescriptive["urgency_level"])
        latest = raw_window[-1]

        response = {
            "is_anomaly": bool(is_anomaly),
            "status": status,
            "led_status": led_status,
            "health": round(float(health), 2),
            "rul": round(float(rul), 2),
            "rul_state": rul_state,
            "main_cause": main_cause,
            "sensor_contributions": {
                "current": round(float(contributions["current"]), 2),
                "temperature": round(float(contributions["temperature"]), 2),
                "vibration": round(float(contributions["vibration"]), 2),
            },
            "latest_values": {
                "current": round(float(latest[0]), 4),
                "temperature": round(float(latest[1]), 4),
                "vibration": round(float(latest[2]), 4),
            },

            "reconstruction_error": round(float(reconstruction_error), 6),
            "smoothed_error": round(float(smoothed_error), 6),
            "threshold": round(float(active_threshold), 6),
            "base_threshold": round(float(base_threshold), 6),
            "adaptive_threshold": round(float(active_threshold), 6),
            "adaptive_threshold_enabled": bool(ADAPTIVE_THRESHOLD_ENABLED),
            "adaptive_threshold_ready": bool(adaptive_ready),
            "adaptive_history_count": adaptive_summary["count"],
            "adaptive_history_ready": adaptive_summary["ready"],
            "adaptive_history_mean": round(float(adaptive_summary["mean"]), 6) if adaptive_summary["mean"] is not None else None,
            "adaptive_history_std": round(float(adaptive_summary["std"]), 6) if adaptive_summary["std"] is not None else None,
            "adaptive_history_percentile": round(float(adaptive_summary["percentile"]), 6) if adaptive_summary["percentile"] is not None else None,
            "adaptive_update_applied": bool(adaptive_update["applied"]),
            "adaptive_update_reason": adaptive_update["reason"],
            "adaptive_warmup_block": bool(adaptive_update["warmup_like"]),
            "adaptive_warmup_exit_temperature": round(float(adaptive_update["warmup_exit_temperature"]), 4) if adaptive_update["warmup_exit_temperature"] is not None else None,
            "threshold_mode": "adaptive" if adaptive_ready and ADAPTIVE_THRESHOLD_ENABLED else "fixed_base_threshold",
            "failure_threshold": round(float(active_threshold * FAILURE_MULTIPLIER), 6),
            "degradation_rate": round(float(slope), 6) if slope is not None else None,
            "confidence_level": confidence_level,
            "confidence_score": round(float(confidence_score), 2),
            "rul_min": round(float(rul_min), 2),
            "rul_max": round(float(rul_max), 2),
            "rul_std": round(float(rul_std), 2),
            "ood_score": round(float(ood_score), 6) if ood_score is not None else None,
            "ood_details": {name: round(float(ood_details.get(name, 0.0)), 6) for name in FEATURE_NAMES},
            "ood_direction_details": ood_direction_details,
            "uncertainty_reason": uncertainty_reason,
            "uncertainty_sources": confidence_sources,

            "contrib_current": round(float(contributions["current"]), 2),
            "contrib_temperature": round(float(contributions["temperature"]), 2),
            "contrib_vibration": round(float(contributions["vibration"]), 2),
            "current": round(float(latest[0]), 4),
            "temperature": round(float(latest[1]), 4),
            "vibration": round(float(latest[2]), 4),
            "current_status": {
                "current": round(float(latest[0]), 4),
                "is_anomaly": bool(is_anomaly),
                "status": status,
            },

            **prescriptive,
            "analysis_timestamp": int(time.time()),
            "warmup_active": False,
            "warmup_remaining_s": 0,
            "buffering_active": False,
        }

        print("done /batch total time:", round(time.time() - t0, 3), "s")
        return jsonify(response), 200

    except Exception as e:
        print("\n========== /batch ERROR ==========")
        print("Error:", str(e))
        traceback.print_exc()
        print("========== END ERROR ==========\n")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)