from fastapi import FastAPI, Request
import numpy as np
import tensorflow as tf
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load model, scaler, threshold
model = tf.keras.models.load_model('lstm_autoencoder.keras')
scaler = joblib.load('scaler.save')
THRESHOLD = np.load('threshold.npy').item()

@app.post("/batch")
async def predict_batch(request: Request):
    try:
        data = await request.json()
        readings = data['readings']   # expected list of 20 lists, each [curr, temp, vibe]

        if len(readings) != 20:
            return {"error": "Exactly 20 readings required"}

        # Convert to numpy array and scale
        seq = np.array(readings, dtype=np.float32)           # (20, 3)
        scaled = scaler.transform(seq)                        # (20, 3)
        input_tensor = scaled.reshape(1, 20, 3)

        # Reconstruct
        reconstructed = model.predict(input_tensor, verbose=0)
        mse = np.mean(np.square(input_tensor - reconstructed))

        is_anomaly = bool(mse > THRESHOLD)

        # Linear inversion health
        health = 100 - (mse / THRESHOLD) * 100
        health = max(0, min(100, health))

        return {
            "is_anomaly": is_anomaly,
            "mse": round(float(mse), 6),
            "health": round(float(health), 2),
            "threshold": float(THRESHOLD)
        }

    except Exception as e:
        logger.error(f"Error: {e}")
        return {"error": str(e)}