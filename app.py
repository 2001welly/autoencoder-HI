from fastapi import FastAPI, Request
import numpy as np
import tensorflow as tf
import joblib
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

# ------------------------------------------------------------------
# Sensor names for reporting
# ------------------------------------------------------------------
SENSOR_NAMES = ['current', 'temperature', 'vibration']

# ------------------------------------------------------------------
# Batch prediction endpoint
# ------------------------------------------------------------------
@app.post("/batch")
async def predict_batch(request: Request):
    try:
        data = await request.json()
        readings = data['readings']   # list of 20 lists, each [curr, temp, vibe]

        if len(readings) != 20:
            return {"error": "Exactly 20 readings required"}

        # Convert to numpy array and scale
        seq = np.array(readings, dtype=np.float32)           # (20, 3)
        scaled = scaler.transform(seq)                        # (20, 3)
        input_tensor = scaled.reshape(1, 20, 3)

        # Reconstruct
        reconstructed = model.predict(input_tensor, verbose=0)

        # Compute overall MSE
        mse = np.mean(np.square(input_tensor - reconstructed))

        # Compute per-sensor MSE (average over time steps)
        mse_per_sensor = np.mean(np.square(input_tensor - reconstructed), axis=(0, 1))  # shape (3,)
        contributions = (mse_per_sensor / np.sum(mse_per_sensor)) * 100
        main_cause = SENSOR_NAMES[np.argmax(mse_per_sensor)]

        is_anomaly = bool(mse > THRESHOLD)

        # Linear inversion health
        health = 100 - (mse / THRESHOLD) * 100
        health = max(0, min(100, health))

        # Build response
        response = {
            "is_anomaly": is_anomaly,
            "mse": round(float(mse), 6),
            "health": round(float(health), 2),
            "threshold": float(THRESHOLD),
            "sensor_contributions": {
                "current": round(float(contributions[0]), 1),
                "temperature": round(float(contributions[1]), 1),
                "vibration": round(float(contributions[2]), 1)
            },
            "main_cause": main_cause
        }

        return response

    except KeyError as e:
        logger.warning(f"Missing key in request: {e}")
        return {"error": f"Missing field: {e}"}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": str(e)}

# ------------------------------------------------------------------
# Health check endpoint
# ------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "healthy"}