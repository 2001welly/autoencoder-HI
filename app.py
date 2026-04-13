import os
import json
import math
import time
from pathlib import Path
from collections import deque
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model


# =============================================================================
# CONFIGURATION
# =============================================================================
SEQUENCE_LENGTH = int(os.getenv("SEQUENCE_LENGTH", "20"))
WINDOW_STEP_SECONDS = float(os.getenv("WINDOW_STEP_SECONDS", "4"))  # expected time between updates
SMOOTHING_ALPHA = float(os.getenv("SMOOTHING_ALPHA", "0.30"))
HISTORY_SIZE = int(os.getenv("HISTORY_SIZE", "30"))
RUL_REFERENCE_HOURS = float(os.getenv("RUL_REFERENCE_HOURS", "120"))  # used for risk normalization
MAX_RUL_HOURS = float(os.getenv("MAX_RUL_HOURS", "240"))
DEFAULT_FAILURE_MULTIPLIER = float(os.getenv("DEFAULT_FAILURE_MULTIPLIER", "3.0"))

# Artifact paths: set these on Render if your filenames differ.
MODEL_CANDIDATES = [
    os.getenv("MODEL_PATH", "").strip(),
    "model.keras",
    "lstm_autoencoder_model.keras",
    "autoencoder_model.keras",
    "lstm_autoencoder.keras",
    "model.h5",
    "lstm_autoencoder_model.h5",
]

SCALER_CANDIDATES = [
    os.getenv("SCALER_PATH", "").strip(),
    "scaler.pkl",
    "scaler.joblib",
    "minmax_scaler.pkl",
    "minmax_scaler.joblib",
]

THRESHOLD_CANDIDATES = [
    os.getenv("THRESHOLD_PATH", "").strip(),
    "threshold.pkl",
    "threshold.joblib",
    "threshold.npy",
    "threshold.json",
    "threshold.txt",
]

ENV_NORMAL_THRESHOLD = os.getenv("NORMAL_THRESHOLD")
ENV_FAILURE_THRESHOLD = os.getenv("FAILURE_THRESHOLD")


# =============================================================================
# APP
# =============================================================================
app = Flask(__name__)


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


# =============================================================================
# STATE
# =============================================================================
MODEL = None
SCALER = None
NORMAL_THRESHOLD = None
FAILURE_THRESHOLD = None

ERROR_HISTORY = deque(maxlen=HISTORY_SIZE)
SMOOTHED_ERROR_HISTORY = deque(maxlen=HISTORY_SIZE)
HEALTH_HISTORY = deque(maxlen=HISTORY_SIZE)
ANOMALY_HISTORY = deque(maxlen=HISTORY_SIZE)
LAST_SMOOTHED_ERROR = None


# =============================================================================
# UTILS
# =============================================================================
def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def first_existing_path(candidates: List[str]) -> str:
    for item in candidates:
        if not item:
            continue
        if Path(item).exists():
            return item
    return ""


def load_threshold_file(path: str) -> Tuple[float, float]:
    """
    Supports:
      - dict/json: {"threshold": x, "failure_threshold": y}
      - dict/json: {"normal_threshold": x, "failure_threshold": y}
      - list/tuple/array: [threshold, failure_threshold]
      - single float value
      - plain text number
    """
    if ENV_NORMAL_THRESHOLD is not None:
        normal_threshold = safe_float(ENV_NORMAL_THRESHOLD, 0.01)
        failure_threshold = safe_float(
            ENV_FAILURE_THRESHOLD,
            max(normal_threshold * DEFAULT_FAILURE_MULTIPLIER, normal_threshold + 1e-6),
        )
        return normal_threshold, failure_threshold

    if not path:
        normal_threshold = 0.01
        failure_threshold = max(normal_threshold * DEFAULT_FAILURE_MULTIPLIER, normal_threshold + 1e-6)
        return normal_threshold, failure_threshold

    suffix = Path(path).suffix.lower()

    if suffix in {".pkl", ".joblib"}:
        data = joblib.load(path)
    elif suffix == ".npy":
        data = np.load(path, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.shape == ():
            data = data.item()
    elif suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        try:
            data = json.loads(raw)
        except Exception:
            data = raw

    if isinstance(data, dict):
        normal_threshold = safe_float(
            data.get("threshold", data.get("normal_threshold", data.get("anomaly_threshold", 0.01))),
            0.01,
        )
        failure_threshold = safe_float(
            data.get(
                "failure_threshold",
                data.get("critical_threshold", max(normal_threshold * DEFAULT_FAILURE_MULTIPLIER, normal_threshold + 1e-6)),
            ),
            max(normal_threshold * DEFAULT_FAILURE_MULTIPLIER, normal_threshold + 1e-6),
        )
        return normal_threshold, max(failure_threshold, normal_threshold + 1e-6)

    if isinstance(data, (list, tuple, np.ndarray)):
        flat = np.array(data, dtype=float).flatten()
        if flat.size >= 2:
            normal_threshold = safe_float(flat[0], 0.01)
            failure_threshold = safe_float(
                flat[1],
                max(normal_threshold * DEFAULT_FAILURE_MULTIPLIER, normal_threshold + 1e-6),
            )
            return normal_threshold, max(failure_threshold, normal_threshold + 1e-6)
        if flat.size == 1:
            normal_threshold = safe_float(flat[0], 0.01)
            failure_threshold = max(normal_threshold * DEFAULT_FAILURE_MULTIPLIER, normal_threshold + 1e-6)
            return normal_threshold, failure_threshold

    normal_threshold = safe_float(data, 0.01)
    failure_threshold = max(normal_threshold * DEFAULT_FAILURE_MULTIPLIER, normal_threshold + 1e-6)
    return normal_threshold, failure_threshold


def load_artifacts() -> Dict[str, Any]:
    global MODEL, SCALER, NORMAL_THRESHOLD, FAILURE_THRESHOLD

    model_path = first_existing_path(MODEL_CANDIDATES)
    scaler_path = first_existing_path(SCALER_CANDIDATES)
    threshold_path = first_existing_path(THRESHOLD_CANDIDATES)

    model_loaded = False
    scaler_loaded = False
    threshold_loaded = False
    errors = []

    if model_path:
        try:
            MODEL = load_model(
                model_path,
                compile=False,
                custom_objects={"TimeDistributed": tf.keras.layers.TimeDistributed},
            )
            model_loaded = True
        except Exception as exc:
            MODEL = None
            errors.append(f"Model load failed: {exc}")
    else:
        errors.append("Model file not found.")

    if scaler_path:
        try:
            SCALER = joblib.load(scaler_path)
            scaler_loaded = True
        except Exception as exc:
            SCALER = None
            errors.append(f"Scaler load failed: {exc}")
    else:
        errors.append("Scaler file not found.")

    try:
        NORMAL_THRESHOLD, FAILURE_THRESHOLD = load_threshold_file(threshold_path)
        threshold_loaded = True
    except Exception as exc:
        NORMAL_THRESHOLD = 0.01
        FAILURE_THRESHOLD = max(NORMAL_THRESHOLD * DEFAULT_FAILURE_MULTIPLIER, NORMAL_THRESHOLD + 1e-6)
        errors.append(f"Threshold load failed: {exc}")

    return {
        "model_path": model_path or None,
        "scaler_path": scaler_path or None,
        "threshold_path": threshold_path or None,
        "model_loaded": model_loaded,
        "scaler_loaded": scaler_loaded,
        "threshold_loaded": threshold_loaded,
        "threshold": NORMAL_THRESHOLD,
        "failure_threshold": FAILURE_THRESHOLD,
        "errors": errors,
    }


ARTIFACT_STATUS = load_artifacts()


def parse_batch_payload(payload: Any) -> np.ndarray:
    """
    Accepted formats:
      1) {"batch": [{"current": 1, "temperature": 30, "vibration": 0.2}, ...]}
      2) {"data": [[1, 30, 0.2], [ ... ]]}
      3) [[1, 30, 0.2], [ ... ]]
      4) [{"current": ...,"temperature": ...,"vibration": ...}, ...]
    """
    if isinstance(payload, dict):
        for key in ("batch", "data", "window", "sequence", "samples", "payload"):
            if key in payload:
                payload = payload[key]
                break

    if not isinstance(payload, list):
        raise ValueError("Expected a list of samples or an object containing batch/data/window/sequence.")

    rows: List[List[float]] = []

    for item in payload:
        if isinstance(item, dict):
            current = safe_float(item.get("current", item.get("current ", item.get("i", 0.0))), 0.0)
            temperature = safe_float(item.get("temperature", item.get("temp", item.get("t", 0.0))), 0.0)
            vibration = safe_float(item.get("vibration", item.get("vib", item.get("v", 0.0))), 0.0)
            rows.append([current, temperature, vibration])
        elif isinstance(item, (list, tuple, np.ndarray)) and len(item) >= 3:
            rows.append([safe_float(item[0]), safe_float(item[1]), safe_float(item[2])])
        else:
            raise ValueError("Each sample must be a dict with current/temperature/vibration or a 3-value list.")

    arr = np.array(rows, dtype=np.float32)

    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("Batch must have shape (N, 3).")

    if arr.shape[0] < SEQUENCE_LENGTH:
        raise ValueError(f"Expected at least {SEQUENCE_LENGTH} samples, got {arr.shape[0]}.")

    if arr.shape[0] > SEQUENCE_LENGTH:
        arr = arr[-SEQUENCE_LENGTH:]

    return arr


def smooth_error(current_error: float) -> float:
    global LAST_SMOOTHED_ERROR
    if LAST_SMOOTHED_ERROR is None:
        LAST_SMOOTHED_ERROR = current_error
    else:
        LAST_SMOOTHED_ERROR = (SMOOTHING_ALPHA * current_error) + ((1.0 - SMOOTHING_ALPHA) * LAST_SMOOTHED_ERROR)
    return LAST_SMOOTHED_ERROR


def slope(values: List[float]) -> float:
    if len(values) < 3:
        return 0.0
    x = np.arange(len(values), dtype=float)
    y = np.array(values, dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def determine_prescription_type(contributions: Dict[str, float]) -> Tuple[str, str, float, float]:
    ordered = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
    top_name, top_value = ordered[0]
    second_name, second_value = ordered[1]
    gap = top_value - second_value

    if gap < 8.0:
        if {"temperature", "current"} == {top_name, second_name}:
            ptype = "ELECTRICAL_THERMAL"
        elif {"temperature", "vibration"} == {top_name, second_name}:
            ptype = "THERMAL_MECHANICAL"
        elif {"current", "vibration"} == {top_name, second_name}:
            ptype = "ELECTRICAL_MECHANICAL"
        else:
            ptype = "MIXED"
    else:
        if top_name == "temperature":
            ptype = "THERMAL"
        elif top_name == "vibration":
            ptype = "MECHANICAL"
        elif top_name == "current":
            ptype = "ELECTRICAL"
        else:
            ptype = "GENERAL"

    return ptype, top_name, top_value, second_value


def compute_rul_hours(smoothed_error: float, error_slope_per_step: float) -> Tuple[float, str]:
    """
    Hybrid RUL estimate:
      - If deterioration trend is positive, estimate time until failure threshold.
      - Otherwise fall back to reserve-based estimate so RUL is not forced to match health.
    """
    remaining_error = max(FAILURE_THRESHOLD - smoothed_error, 0.0)
    step_hours = max(WINDOW_STEP_SECONDS / 3600.0, 1e-6)

    if error_slope_per_step > 1e-6 and remaining_error > 0:
        rate_per_hour = error_slope_per_step / step_hours
        rul_hours = remaining_error / max(rate_per_hour, 1e-9)
    else:
        reserve_ratio = clamp(remaining_error / max(FAILURE_THRESHOLD, 1e-6), 0.0, 1.0)
        rul_hours = reserve_ratio * RUL_REFERENCE_HOURS

    rul_hours = clamp(rul_hours, 0.0, MAX_RUL_HOURS)

    if rul_hours > 72:
        rul_state = "NORMAL"
    elif rul_hours > 24:
        rul_state = "OBSERVE"
    elif rul_hours > 6:
        rul_state = "PLAN_MAINTENANCE"
    elif rul_hours > 1:
        rul_state = "URGENT"
    else:
        rul_state = "CRITICAL"

    return rul_hours, rul_state


def build_prescription(
    urgency_level: str,
    prescription_type: str,
    top_cause: str,
    top_value: float,
    health: float,
    rul_hours: float,
    persistence: float,
    trend_factor: float,
    contributions: Dict[str, float],
) -> Dict[str, Any]:
    actions_map = {
        ("NORMAL", "THERMAL"): ["Continue monitoring", "Inspect cooling path during routine maintenance if trend worsens"],
        ("WARNING", "THERMAL"): ["Increase monitoring frequency", "Inspect cooling fan", "Check ventilation openings"],
        ("PLAN_MAINTENANCE", "THERMAL"): ["Schedule thermal inspection soon", "Inspect cooling fan", "Check ventilation", "Check overload condition"],
        ("URGENT", "THERMAL"): ["Perform urgent thermal inspection", "Inspect cooling fan", "Check ventilation", "Reduce operating load if possible", "Prepare controlled shutdown if temperature keeps rising"],
        ("CRITICAL", "THERMAL"): ["Immediate shutdown", "Inspect cooling system before restart", "Check overload condition", "Do not restart until fault is cleared"],

        ("NORMAL", "MECHANICAL"): ["Continue monitoring", "Inspect bearings during routine maintenance if vibration trend worsens"],
        ("WARNING", "MECHANICAL"): ["Increase monitoring frequency", "Inspect bearings", "Check looseness of mountings"],
        ("PLAN_MAINTENANCE", "MECHANICAL"): ["Schedule mechanical inspection soon", "Inspect bearings", "Check shaft alignment", "Check looseness and balance"],
        ("URGENT", "MECHANICAL"): ["Perform urgent mechanical inspection", "Inspect bearings and alignment immediately", "Reduce speed or load if possible", "Prepare controlled shutdown"],
        ("CRITICAL", "MECHANICAL"): ["Immediate shutdown", "Inspect bearings, alignment, and looseness before restart", "Do not restart until fault is cleared"],

        ("NORMAL", "ELECTRICAL"): ["Continue monitoring", "Inspect electrical path during routine maintenance if current trend worsens"],
        ("WARNING", "ELECTRICAL"): ["Increase monitoring frequency", "Inspect wiring connections", "Check load condition"],
        ("PLAN_MAINTENANCE", "ELECTRICAL"): ["Schedule electrical inspection soon", "Inspect wiring and terminals", "Check overload condition", "Inspect driven load for abnormal resistance"],
        ("URGENT", "ELECTRICAL"): ["Perform urgent electrical inspection", "Inspect wiring, terminals, and load immediately", "Reduce operating load if possible", "Prepare controlled shutdown"],
        ("CRITICAL", "ELECTRICAL"): ["Immediate shutdown", "Inspect wiring, terminals, load, and winding condition before restart", "Do not restart until fault is cleared"],

        ("NORMAL", "THERMAL_MECHANICAL"): ["Continue monitoring", "Inspect cooling path and mechanical mounting during routine maintenance"],
        ("WARNING", "THERMAL_MECHANICAL"): ["Increase monitoring frequency", "Inspect cooling fan", "Inspect bearings and looseness"],
        ("PLAN_MAINTENANCE", "THERMAL_MECHANICAL"): ["Schedule combined thermal-mechanical inspection", "Inspect cooling system", "Inspect bearings and alignment"],
        ("URGENT", "THERMAL_MECHANICAL"): ["Perform urgent combined thermal-mechanical inspection", "Reduce load if possible", "Prepare controlled shutdown"],
        ("CRITICAL", "THERMAL_MECHANICAL"): ["Immediate shutdown", "Inspect cooling system, bearings, and alignment before restart"],

        ("NORMAL", "ELECTRICAL_THERMAL"): ["Continue monitoring", "Inspect electrical load path and cooling path during routine maintenance"],
        ("WARNING", "ELECTRICAL_THERMAL"): ["Increase monitoring frequency", "Inspect wiring connections", "Inspect cooling fan and ventilation"],
        ("PLAN_MAINTENANCE", "ELECTRICAL_THERMAL"): ["Schedule combined electrical-thermal inspection", "Inspect wiring and terminals", "Inspect cooling system", "Check overload condition"],
        ("URGENT", "ELECTRICAL_THERMAL"): ["Perform urgent electrical-thermal inspection", "Reduce load immediately if possible", "Prepare controlled shutdown"],
        ("CRITICAL", "ELECTRICAL_THERMAL"): ["Immediate shutdown", "Inspect wiring, load path, and cooling system before restart"],

        ("NORMAL", "ELECTRICAL_MECHANICAL"): ["Continue monitoring", "Inspect electrical load path and mechanical assembly during routine maintenance"],
        ("WARNING", "ELECTRICAL_MECHANICAL"): ["Increase monitoring frequency", "Inspect wiring connections", "Inspect bearings and mountings"],
        ("PLAN_MAINTENANCE", "ELECTRICAL_MECHANICAL"): ["Schedule combined electrical-mechanical inspection", "Inspect wiring and terminals", "Inspect bearings and alignment"],
        ("URGENT", "ELECTRICAL_MECHANICAL"): ["Perform urgent electrical-mechanical inspection", "Reduce load if possible", "Prepare controlled shutdown"],
        ("CRITICAL", "ELECTRICAL_MECHANICAL"): ["Immediate shutdown", "Inspect wiring, load path, bearings, and alignment before restart"],

        ("NORMAL", "MIXED"): ["Continue monitoring", "Log condition and observe trend"],
        ("WARNING", "MIXED"): ["Increase monitoring frequency", "Perform general diagnostic inspection"],
        ("PLAN_MAINTENANCE", "MIXED"): ["Schedule general diagnostic inspection", "Inspect thermal, mechanical, and electrical subsystems"],
        ("URGENT", "MIXED"): ["Perform urgent general inspection", "Reduce load if possible", "Prepare controlled shutdown"],
        ("CRITICAL", "MIXED"): ["Immediate shutdown", "Perform full subsystem inspection before restart"],

        ("NORMAL", "GENERAL"): ["Continue normal operation", "Continue monitoring"],
        ("WARNING", "GENERAL"): ["Increase monitoring frequency", "Inspect during routine maintenance"],
        ("PLAN_MAINTENANCE", "GENERAL"): ["Schedule maintenance soon", "Perform general diagnostic inspection"],
        ("URGENT", "GENERAL"): ["Urgent maintenance required", "Prepare controlled shutdown"],
        ("CRITICAL", "GENERAL"): ["Immediate shutdown", "Do not restart until inspected"],
    }

    title_map = {
        "NORMAL": "Continue normal operation",
        "WARNING": "Early warning condition",
        "PLAN_MAINTENANCE": "Maintenance should be scheduled soon",
        "URGENT": "Urgent maintenance required",
        "CRITICAL": "Critical condition - immediate action required",
    }

    category_label_map = {
        "THERMAL": "Thermal maintenance",
        "MECHANICAL": "Mechanical maintenance",
        "ELECTRICAL": "Electrical/load-related maintenance",
        "THERMAL_MECHANICAL": "Thermal-mechanical maintenance",
        "ELECTRICAL_THERMAL": "Electrical-thermal maintenance",
        "ELECTRICAL_MECHANICAL": "Electrical-mechanical maintenance",
        "MIXED": "Mixed-condition maintenance",
        "GENERAL": "General diagnostic maintenance",
    }

    actions = actions_map.get((urgency_level, prescription_type), ["Continue monitoring"])
    category_label = category_label_map.get(prescription_type, "General diagnostic maintenance")
    title = f"{title_map.get(urgency_level, 'Maintenance advice')} - {category_label}"

    if urgency_level == "CRITICAL":
        auto_action = "SHUTDOWN"
    elif urgency_level == "URGENT":
        auto_action = "REDUCE_LOAD_OR_PREPARE_SHUTDOWN"
    elif urgency_level == "PLAN_MAINTENANCE":
        auto_action = "SCHEDULE_MAINTENANCE"
    else:
        auto_action = "MONITOR"

    sorted_contrib = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
    reason = (
        f"{top_cause.capitalize()} is the dominant contributor at {top_value:.1f}%."
        f" Health is {health:.1f}%, estimated RUL is {rul_hours:.1f} hours,"
        f" persistence is {persistence * 100:.1f}%, and trend factor is {trend_factor:.2f}."
        f" Secondary contributor is {sorted_contrib[1][0]} at {sorted_contrib[1][1]:.1f}%."
    )

    return {
        "prescription_type": prescription_type,
        "prescription_category_label": category_label,
        "prescription_title": title,
        "prescription_actions": actions,
        "prescription_reason": reason,
        "auto_action": auto_action,
    }


def analyze_batch(raw_batch: np.ndarray) -> Dict[str, Any]:
    if MODEL is None or SCALER is None:
        raise RuntimeError("Model and/or scaler not loaded.")

    # Scale using training scaler
    scaled_batch = SCALER.transform(raw_batch)
    x = scaled_batch.reshape(1, SEQUENCE_LENGTH, 3)

    # Reconstruct
    reconstructed = MODEL.predict(x, verbose=0)
    squared_error = np.square(x - reconstructed)

    reconstruction_error = float(np.mean(squared_error))
    smoothed_error = float(smooth_error(reconstruction_error))

    # Per-feature contribution
    feature_mse = np.mean(squared_error, axis=(0, 1))
    total_feature_mse = float(np.sum(feature_mse)) + 1e-12
    contributions = {
        "current": float((feature_mse[0] / total_feature_mse) * 100.0),
        "temperature": float((feature_mse[1] / total_feature_mse) * 100.0),
        "vibration": float((feature_mse[2] / total_feature_mse) * 100.0),
    }

    # History update
    ERROR_HISTORY.append(reconstruction_error)
    SMOOTHED_ERROR_HISTORY.append(smoothed_error)

    error_gap = max(FAILURE_THRESHOLD - NORMAL_THRESHOLD, 1e-9)
    anomaly_severity = clamp((smoothed_error - NORMAL_THRESHOLD) / error_gap, 0.0, 1.0)

    is_anomaly = smoothed_error > NORMAL_THRESHOLD
    ANOMALY_HISTORY.append(1 if is_anomaly else 0)

    health = clamp(100.0 * (1.0 - (smoothed_error / max(FAILURE_THRESHOLD, 1e-9))), 0.0, 100.0)
    HEALTH_HISTORY.append(health)

    # Trend and persistence
    err_slope = slope(list(SMOOTHED_ERROR_HISTORY))
    trend_factor = clamp(max(err_slope, 0.0) / max(error_gap / 8.0, 1e-9), 0.0, 1.0)
    persistence = float(np.mean(ANOMALY_HISTORY)) if len(ANOMALY_HISTORY) else 0.0

    rul_hours, rul_state = compute_rul_hours(smoothed_error, err_slope)

    # MPS
    health_degradation = (100.0 - health) / 100.0
    rul_risk = clamp(1.0 - (rul_hours / max(RUL_REFERENCE_HOURS, 1e-9)), 0.0, 1.0)

    mps = (
        0.25 * anomaly_severity
        + 0.20 * health_degradation
        + 0.25 * rul_risk
        + 0.15 * trend_factor
        + 0.15 * persistence
    )
    mps = clamp(mps, 0.0, 1.0)

    if mps < 0.25:
        urgency_level = "NORMAL"
    elif mps < 0.50:
        urgency_level = "WARNING"
    elif mps < 0.70:
        urgency_level = "PLAN_MAINTENANCE"
    elif mps < 0.85:
        urgency_level = "URGENT"
    else:
        urgency_level = "CRITICAL"

    prescription_type, top_cause, top_value, second_value = determine_prescription_type(contributions)
    prescription = build_prescription(
        urgency_level=urgency_level,
        prescription_type=prescription_type,
        top_cause=top_cause,
        top_value=top_value,
        health=health,
        rul_hours=rul_hours,
        persistence=persistence,
        trend_factor=trend_factor,
        contributions=contributions,
    )

    latest_values = {
        "current": float(raw_batch[-1, 0]),
        "temperature": float(raw_batch[-1, 1]),
        "vibration": float(raw_batch[-1, 2]),
    }

    return {
        "ok": True,
        "is_anomaly": bool(is_anomaly),
        "reconstruction_error": reconstruction_error,
        "smoothed_error": smoothed_error,
        "threshold": NORMAL_THRESHOLD,
        "failure_threshold": FAILURE_THRESHOLD,
        "anomaly_severity": anomaly_severity,
        "health": health,
        "rul_hours": rul_hours,
        "rul_state": rul_state,
        "maintenance_priority_score": mps,
        "urgency_level": urgency_level,
        "main_cause": top_cause,
        "sensor_contributions": {
            "current": round(contributions["current"], 2),
            "temperature": round(contributions["temperature"], 2),
            "vibration": round(contributions["vibration"], 2),
        },
        "latest_values": latest_values,
        "trend_factor": round(trend_factor, 4),
        "persistence_factor": round(persistence, 4),
        "history_length": len(SMOOTHED_ERROR_HISTORY),
        **prescription,
        "analysis_timestamp": int(time.time()),
        "window_size": SEQUENCE_LENGTH,
    }


# =============================================================================
# ROUTES
# =============================================================================
@app.route("/", methods=["GET"])
def root():
    return jsonify(
        {
            "ok": True,
            "message": "LSTM autoencoder backend with prescriptive maintenance is running.",
            "model_loaded": MODEL is not None,
            "scaler_loaded": SCALER is not None,
            "threshold_loaded": NORMAL_THRESHOLD is not None,
            "threshold": NORMAL_THRESHOLD,
            "failure_threshold": FAILURE_THRESHOLD,
            "sequence_length": SEQUENCE_LENGTH,
            "artifact_status": ARTIFACT_STATUS,
        }
    )


@app.route("/status", methods=["GET"])
def status():
    return jsonify(
        {
            "ok": True,
            "model_loaded": MODEL is not None,
            "scaler_loaded": SCALER is not None,
            "threshold_loaded": NORMAL_THRESHOLD is not None,
            "threshold": NORMAL_THRESHOLD,
            "failure_threshold": FAILURE_THRESHOLD,
            "sequence_length": SEQUENCE_LENGTH,
            "history_length": len(SMOOTHED_ERROR_HISTORY),
            "window_step_seconds": WINDOW_STEP_SECONDS,
        }
    )


@app.route("/reload", methods=["POST"])
def reload_artifacts():
    global ARTIFACT_STATUS, LAST_SMOOTHED_ERROR
    ARTIFACT_STATUS = load_artifacts()
    ERROR_HISTORY.clear()
    SMOOTHED_ERROR_HISTORY.clear()
    HEALTH_HISTORY.clear()
    ANOMALY_HISTORY.clear()
    LAST_SMOOTHED_ERROR = None
    return jsonify({"ok": True, "artifact_status": ARTIFACT_STATUS})


@app.route("/batch", methods=["POST", "OPTIONS"])
def batch():
    if request.method == "OPTIONS":
        return jsonify({"ok": True})

    if MODEL is None or SCALER is None:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "Model and/or scaler not loaded.",
                    "model_loaded": MODEL is not None,
                    "scaler_loaded": SCALER is not None,
                    "threshold_loaded": NORMAL_THRESHOLD is not None,
                }
            ),
            500,
        )

    try:
        payload = request.get_json(force=True, silent=False)
        raw_batch = parse_batch_payload(payload)
        result = analyze_batch(raw_batch)
        return jsonify(result)

    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)