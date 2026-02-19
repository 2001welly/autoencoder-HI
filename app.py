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

# ------------------------------------------------------------------
# 1. Load model, scaler, and threshold
# ------------------------------------------------------------------
try:
    model = tf.keras.models.load_model('lstm_autoencoder.keras')
    logger.info("✅ Model loaded successfully.")
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    raise e

try:
    scaler = joblib.load('scaler.save')
    logger.info("✅ Scaler loaded successfully.")
except Exception as e:
    logger.error(f"❌ Failed to load scaler: {e}")
    raise e

try:
    THRESHOLD = np.load('threshold.npy').item()
    logger.info(f"✅ Threshold loaded: {THRESHOLD}")
except Exception as e:
    logger.error(f"❌ Failed to load threshold, using default 0.08: {e}")
    THRESHOLD = 0.08

# ------------------------------------------------------------------
# 2. Global buffers
# ------------------------------------------------------------------
# Rolling window of the last 20 sensor readings (each reading = [current, temp, vibration])
data_buffer = deque(maxlen=20)

# Buffer for the last few MSE values (for smoothing the health score)
mse_buffer = deque(maxlen=5)          # keep last 5 MSE values

# ------------------------------------------------------------------
# 3. Prediction endpoint
# ------------------------------------------------------------------
@app.post("/")
async def predict(request: Request):
    """
    Accepts JSON: {"current": x, "temperature": y, "vibration": z}
    Returns anomaly flag, raw MSE, smoothed health (%), and status.
    """
    try:
        # Parse incoming JSON
        content = await request.json()
        curr = float(content['current'])
        temp = float(content['temperature'])
        vibe = float(content['vibration'])

        # Add to rolling window
        data_buffer.append([curr, temp, vibe])

        # Not enough data yet
        if len(data_buffer) < 20:
            return {
                "is_anomaly": False,
                "status": "collecting",
                "message": f"Need {20 - len(data_buffer)} more readings"
            }

        # Prepare input sequence: shape (1, 20, 3)
        sequence = np.array(data_buffer)                     # (20, 3)
        scaled_seq = scaler.transform(sequence)              # (20, 3)
        input_tensor = scaled_seq.reshape(1, 20, 3).astype('float32')

        # Reconstruct using the autoencoder
        reconstructed = model.predict(input_tensor, verbose=0)   # (1, 20, 3)

        # Compute MSE (mean over all time steps and features)
        mse = np.mean(np.square(input_tensor - reconstructed))

        # Update MSE buffer (for health smoothing)
        mse_buffer.append(mse)
        avg_mse = sum(mse_buffer) / len(mse_buffer)          # moving average of last N MSEs

        # Compute health using exponential decay of the averaged MSE
        health = 100 * np.exp(-avg_mse / THRESHOLD)
        health = max(0, min(100, health))                     # clamp to [0, 100]

        # Anomaly flag based on raw MSE (not averaged)
        is_anomaly = bool(mse > THRESHOLD)

        # Return all information
        return {
            "is_anomaly": is_anomaly,
            "mse": round(float(mse), 6),
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

# ------------------------------------------------------------------
# 4. Optional health check (for Render monitoring)
# ------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "healthy"}