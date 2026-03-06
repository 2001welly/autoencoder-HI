"""
SENTINEL — LSTM Autoencoder Anomaly Detection API
FastAPI server for industrial motor health monitoring.

Endpoints:
  POST /batch   — accepts 20 sensor readings, returns anomaly verdict + health score
  GET  /health  — server liveness check
  GET  /status  — model info and current threshold
"""

import os
import time
import logging
import warnings
import numpy as np
import joblib
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
WINDOW_SIZE   = 20          # must match ESP32 WINDOW_SIZE
N_FEATURES    = 3           # current, temperature, vibration
MODEL_PATH    = "lstm_autoencoder.keras"
SCALER_PATH   = "scaler.save"
THRESHOLD_PATH = "threshold.npy"

FEATURE_NAMES = ["current", "temperature", "vibration"]

# Health score mapping: reconstruction error → 0–100%
# error == 0          → 100% health
# error == threshold  → 50% health
# error >= 3×threshold → 0% health
HEALTH_FLOOR  = 0.0
HEALTH_CEIL   = 100.0

# ─────────────────────────────────────────────
#  MODEL STATE  (loaded once at startup)
# ─────────────────────────────────────────────
class ModelState:
    model     = None
    scaler    = None
    threshold = None

state = ModelState()


def load_artifacts():
    """Load model, scaler and threshold from disk."""
    logger.info("Loading artifacts...")

    try:
        # Lazy-import keras so startup fails clearly if TF not installed
        import tensorflow as tf  # noqa: F401
        from tensorflow import keras
        state.model = keras.models.load_model(MODEL_PATH)
        logger.info("✅ Model loaded — input shape: %s", state.model.input_shape)
    except Exception as e:
        logger.error("❌ Failed to load model: %s", e)
        raise RuntimeError(f"Model load failed: {e}") from e

    try:
        state.scaler = joblib.load(SCALER_PATH)
        logger.info("✅ Scaler loaded — features: %d", state.scaler.n_features_in_)
    except Exception as e:
        logger.error("❌ Failed to load scaler: %s", e)
        raise RuntimeError(f"Scaler load failed: {e}") from e

    try:
        state.threshold = float(np.load(THRESHOLD_PATH))
        logger.info("✅ Threshold loaded — value: %.6f", state.threshold)
    except Exception as e:
        logger.error("❌ Failed to load threshold: %s", e)
        raise RuntimeError(f"Threshold load failed: {e}") from e


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load artifacts on startup, release on shutdown."""
    load_artifacts()
    logger.info("🚀 SENTINEL API ready")
    yield
    logger.info("🛑 SENTINEL API shutting down")


# ─────────────────────────────────────────────
#  APP
# ─────────────────────────────────────────────
app = FastAPI(
    title="SENTINEL — Industrial Anomaly Detection API",
    description="LSTM Autoencoder anomaly detection for motor health monitoring.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
#  SCHEMAS
# ─────────────────────────────────────────────
class BatchRequest(BaseModel):
    """
    20 sensor readings from the ESP32.
    Each reading is [current (A), temperature (°C), vibration (g)].
    """
    readings: list[list[float]]

    @field_validator("readings")
    @classmethod
    def validate_readings(cls, v):
        if len(v) != WINDOW_SIZE:
            raise ValueError(
                f"Expected exactly {WINDOW_SIZE} readings, got {len(v)}"
            )
        for i, row in enumerate(v):
            if len(row) != N_FEATURES:
                raise ValueError(
                    f"Reading {i}: expected {N_FEATURES} features "
                    f"(current, temperature, vibration), got {len(row)}"
                )
        return v


class BatchResponse(BaseModel):
    is_anomaly:           bool
    health:               float          # 0–100 %
    reconstruction_error: float
    threshold:            float
    main_cause:           Optional[str]  # which sensor deviated most
    sensor_contributions: dict           # % contribution per sensor
    rul:                  Optional[float] = None   # estimated hours (future use)
    inference_ms:         float          # server-side latency


# ─────────────────────────────────────────────
#  CORE LOGIC
# ─────────────────────────────────────────────
def compute_health(error: float, threshold: float) -> float:
    """
    Map reconstruction error to a 0–100% health score.

    - error = 0          → 100%
    - error = threshold  → 50%
    - error ≥ 3×threshold → 0%
    """
    max_error = threshold * 3.0
    health = 100.0 * (1.0 - min(error / max_error, 1.0))
    return round(float(np.clip(health, HEALTH_FLOOR, HEALTH_CEIL)), 2)


def compute_contributions(
    original: np.ndarray,
    reconstructed: np.ndarray
) -> tuple[dict, str]:
    """
    Per-sensor squared error, normalised to 0–100%.
    Returns (contributions_dict, main_cause_name).
    """
    # Mean squared error per feature across the window
    per_feature = np.mean((original - reconstructed) ** 2, axis=0)  # shape (3,)
    total = per_feature.sum()

    if total == 0:
        contrib = {name: 0.0 for name in FEATURE_NAMES}
        main_cause = "none"
    else:
        pcts = (per_feature / total) * 100.0
        contrib = {name: round(float(p), 2) for name, p in zip(FEATURE_NAMES, pcts)}
        main_cause = FEATURE_NAMES[int(np.argmax(per_feature))]

    return contrib, main_cause


def run_inference(readings: list[list[float]]) -> BatchResponse:
    """Scale → reshape → infer → compute metrics."""
    t0 = time.perf_counter()

    raw = np.array(readings, dtype=np.float32)          # (20, 3)

    # Scale
    scaled = state.scaler.transform(raw)                # (20, 3)

    # Reshape for LSTM: (batch=1, timesteps=20, features=3)
    x = scaled.reshape(1, WINDOW_SIZE, N_FEATURES)

    # Reconstruct
    x_hat = state.model.predict(x, verbose=0)           # (1, 20, 3)

    # Reconstruction error (MSE over full window)
    error = float(np.mean((x - x_hat) ** 2))

    is_anomaly = error > state.threshold

    health = compute_health(error, state.threshold)

    # Contributions and main cause (only meaningful if anomaly)
    contrib, main_cause = compute_contributions(
        x[0],      # (20, 3) scaled original
        x_hat[0]   # (20, 3) scaled reconstruction
    )

    if not is_anomaly:
        main_cause = "none"

    inference_ms = round((time.perf_counter() - t0) * 1000, 2)

    logger.info(
        "Inference | error=%.6f | threshold=%.6f | anomaly=%s | health=%.1f%% | cause=%s | %.1fms",
        error, state.threshold, is_anomaly, health, main_cause, inference_ms
    )

    return BatchResponse(
        is_anomaly=is_anomaly,
        health=health,
        reconstruction_error=round(error, 6),
        threshold=round(state.threshold, 6),
        main_cause=main_cause,
        sensor_contributions=contrib,
        rul=None,           # plug in your RUL model here when ready
        inference_ms=inference_ms,
    )


# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health_check():
    """Render liveness probe — always returns 200 if server is up."""
    return {"status": "ok", "service": "SENTINEL API"}


@app.get("/status", tags=["System"])
def model_status():
    """Return model readiness and configuration."""
    ready = all([state.model, state.scaler, state.threshold is not None])
    return {
        "ready":       ready,
        "threshold":   round(state.threshold, 6) if state.threshold else None,
        "window_size": WINDOW_SIZE,
        "n_features":  N_FEATURES,
        "features":    FEATURE_NAMES,
        "scaler_min":  state.scaler.data_min_.tolist() if state.scaler else None,
        "scaler_max":  state.scaler.data_max_.tolist() if state.scaler else None,
    }


@app.post("/batch", response_model=BatchResponse, tags=["Inference"])
def batch_inference(body: BatchRequest):
    """
    Main inference endpoint — called by the ESP32 every loop cycle.

    Body:
    ```json
    {
      "readings": [
        [current, temperature, vibration],
        ...  // 20 rows total
      ]
    }
    ```
    """
    if state.model is None or state.scaler is None or state.threshold is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Try again shortly.")

    try:
        return run_inference(body.readings)
    except Exception as e:
        logger.exception("Inference error: %s", e)
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
