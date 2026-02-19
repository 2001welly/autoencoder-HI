from fastapi import FastAPI, Request
import numpy as np
import tensorflow as tf
import joblib
from collections import deque
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# ------------------------------------------------------------------
# Load model, scaler, threshold
# ------------------------------------------------------------------
try:
    model = tf.keras.models.load_model('lstm_autoencoder.keras')
    logger.info("✅ Model loaded")
except Exception as e:
    logger.error(f"Model load failed: {e}")
    raise

try:
    scaler = joblib.load('scaler.save')
    logger.info("✅ Scaler loaded")
except Exception as e:
    logger.error(f"Scaler load failed: {e}")
    raise

try:
    THRESHOLD = np.load('threshold.npy').item()
    logger.info(f"✅ Threshold: {THRESHOLD}")
except Exception as e:
    logger.error(f"Threshold load failed, using default 0.08: {e}")
    THRESHOLD = 0.08

# Rolling window of last 20 readings
data_buffer = deque(maxlen=20)

@app.post("/")
async def predict(request: Request):
    try:
        # Parse JSON
        content = await request.json()
        curr = float(content['current'])
        temp = float(content['temperature'])
        vibe = float(content['vibration'])

        # Add to buffer
        data_buffer.append([curr, temp, vibe])

        # Not enough data
        if len(data_buffer) < 20:
            return {
                "is_anomaly": False,
                "status": "collecting",
                "message": f"Need {20 - len(data_buffer)} more readings"
            }

        # Prepare input
        seq = np.array(data_buffer)                     # (20,3)
        scaled = scaler.transform(seq)                  # (20,3)
        input_tensor = scaled.reshape(1, 20, 3).astype('float32')

        # Reconstruct
        reconstructed = model.predict(input_tensor, verbose=0)

        # Compute MSE
        mse = np.mean(np.square(input_tensor - reconstructed))

        # Anomaly flag
        is_anomaly = bool(mse > THRESHOLD)

        # ---- Health using linear inversion ----
        health = 100 - (mse / THRESHOLD) * 100
        health = max(0, min(100, health))   # clamp to [0,100]

        return {
            "is_anomaly": is_anomaly,
            "mse": round(float(mse), 6),
            "health": round(float(health), 2),
            "threshold": float(THRESHOLD),
            "status": "ready"
        }

    except KeyError as e:
        logger.warning(f"Missing key: {e}")
        return {"error": f"Missing field: {e}"}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": str(e)}

@app.get("/health")
async def health():
    return {"status": "healthy"}