from fastapi import FastAPI, Request
import numpy as np
import tensorflow as tf
import joblib
from collections import deque
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load model, scaler, and threshold
try:
    model = tf.keras.models.load_model('lstm_autoencoder.keras')
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise e

try:
    scaler = joblib.load('scaler.save')
    logger.info("Scaler loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load scaler: {e}")
    raise e

try:
    THRESHOLD = np.load('threshold.npy').item()
    logger.info(f"Threshold loaded: {THRESHOLD}")
except Exception as e:
    logger.error(f"Failed to load threshold, using default 0.08: {e}")
    THRESHOLD = 0.08

# Sliding window buffer (holds last 20 readings)
data_buffer = deque(maxlen=20)

# EMA smoothing variables
ema_mse = None          # will hold the current EMA value
alpha = 0.3             # smoothing factor (tune between 0.1 and 0.5)

@app.post("/")
async def predict(request: Request):
    global ema_mse
    try:
        content = await request.json()
        curr = float(content['current'])
        temp = float(content['temperature'])
        vibe = float(content['vibration'])

        data_buffer.append([curr, temp, vibe])

        if len(data_buffer) < 20:
            return {
                "is_anomaly": False,
                "status": "collecting",
                "message": f"Need {20 - len(data_buffer)} more readings"
            }

        # Prepare input sequence
        sequence = np.array(data_buffer)                     # (20, 3)
        scaled_seq = scaler.transform(sequence)              # (20, 3)
        input_tensor = scaled_seq.reshape(1, 20, 3).astype('float32')

        # Reconstruct
        reconstructed = model.predict(input_tensor, verbose=0)   # (1, 20, 3)

        # Compute MSE
        mse = np.mean(np.square(input_tensor - reconstructed))

        # Update EMA
        if ema_mse is None:
            ema_mse = mse
        else:
            ema_mse = alpha * mse + (1 - alpha) * ema_mse

        # Compute health using exponential decay of EMA
        health = 100 * np.exp(-ema_mse / THRESHOLD)
        health = max(0, min(100, health))   # clamp to [0, 100]

        is_anomaly = bool(mse > THRESHOLD)

        return {
            "is_anomaly": is_anomaly,
            "mse": round(float(mse), 6),
            "ema_mse": round(float(ema_mse), 6),   # optional for debugging
            "health": round(float(health), 2),
            "threshold": float(THRESHOLD),
            "status": "ready"
        }

    except KeyError as e:
        logger.warning(f"Missing key in request: {e}")
        return {"error": f"Missing field: {e}"}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": str(e)}

@app.get("/health")
async def health():
    return {"status": "healthy"}