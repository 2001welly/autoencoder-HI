from fastapi import FastAPI, Request
import numpy as np
import tensorflow as tf
import joblib
from collections import deque
import logging
import os

# ----------------------------------------------------
# CONFIGURE LOGGING
# ----------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# ----------------------------------------------------
# LOAD MODEL, SCALER, THRESHOLD
# ----------------------------------------------------
try:
    model = tf.keras.models.load_model("lstm_autoencoder.keras")
    logger.info("✅ Model loaded successfully.")
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    raise e

try:
    scaler = joblib.load("scaler.save")
    logger.info("✅ Scaler loaded successfully.")
except Exception as e:
    logger.error(f"❌ Failed to load scaler: {e}")
    raise e

try:
    THRESHOLD = float(np.load("threshold.npy"))
    logger.info(f"✅ Threshold loaded: {THRESHOLD}")
except Exception as e:
    logger.warning(f"⚠ Threshold load failed, using default 0.08: {e}")
    THRESHOLD = 0.08


# ----------------------------------------------------
# GLOBAL BUFFERS
# ----------------------------------------------------
WINDOW_SIZE = 20
data_buffer = deque(maxlen=WINDOW_SIZE)
mse_buffer = deque(maxlen=5)


# ----------------------------------------------------
# PREDICTION ENDPOINT
# ----------------------------------------------------
@app.post("/")
async def predict(request: Request):
    """
    Accepts JSON:
    {
        "current": float,
        "temperature": float,
        "vibration": float
    }

    Returns:
    {
        "is_anomaly": bool,
        "mse": float,
        "health": float,
        "threshold": float,
        "status": string
    }
    """
    try:
        content = await request.json()

        # Extract features
        current = float(content["current"])
        temperature = float(content["temperature"])
        vibration = float(content["vibration"])

        # Add to rolling window
        data_buffer.append([current, temperature, vibration])

        # ------------------------------------------------
        # WARM-UP PHASE
        # ------------------------------------------------
        if len(data_buffer) < WINDOW_SIZE:
            logger.info("Collecting data...")

            return {
                "is_anomaly": False,
                "mse": 0.0,
                "health": 100.0,
                "threshold": THRESHOLD,
                "status": f"collecting ({len(data_buffer)}/{WINDOW_SIZE})"
            }

        # ------------------------------------------------
        # PREPARE INPUT SEQUENCE
        # ------------------------------------------------
        sequence = np.array(data_buffer)                # shape (20, 3)
        scaled_seq = scaler.transform(sequence)
        input_tensor = scaled_seq.reshape(1, WINDOW_SIZE, 3).astype("float32")

        # ------------------------------------------------
        # MODEL RECONSTRUCTION
        # ------------------------------------------------
        reconstructed = model.predict(input_tensor, verbose=0)

        # Compute MSE
        mse = float(np.mean(np.square(input_tensor - reconstructed)))

        # Smooth MSE
        mse_buffer.append(mse)
        avg_mse = sum(mse_buffer) / len(mse_buffer)

        # ------------------------------------------------
        # HEALTH CALCULATION (Improved Sensitivity)
        # ------------------------------------------------
        # Stronger decay factor (3x sharper than basic exp)
        health = 100 * np.exp(-3 * avg_mse / THRESHOLD)
        health = float(np.clip(health, 0, 100))

        # Anomaly decision based on RAW MSE
        is_anomaly = bool(mse > THRESHOLD)

        logger.info(
            f"MSE: {mse:.6f} | Avg MSE: {avg_mse:.6f} | "
            f"Health: {health:.2f}% | Anomaly: {is_anomaly}"
        )

        # ------------------------------------------------
        # RETURN RESPONSE
        # ------------------------------------------------
        return {
            "is_anomaly": is_anomaly,
            "mse": round(mse, 6),
            "health": round(health, 2),
            "threshold": THRESHOLD,
            "status": "ready"
        }

    except KeyError as e:
        logger.warning(f"Missing key: {e}")
        return {"error": f"Missing field: {e}"}

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": str(e)}


# ----------------------------------------------------
# HEALTH CHECK ENDPOINT
# ----------------------------------------------------
@app.get("/health")
async def health_check():
    return {"status": "healthy"}
