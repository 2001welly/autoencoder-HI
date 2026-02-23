from fastapi import FastAPI, Request
import numpy as np
import tensorflow as tf
import joblib
import logging
import time
from firebase import firebase   # Correct import for the 'firebase' package

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# ------------------------------------------------------------------
# Firebase initialization
# ------------------------------------------------------------------
FIREBASE_URL = "https://welly001-8ffca-default-rtdb.firebaseio.com/"
fb = firebase.FirebaseApplication(FIREBASE_URL, None)

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
# RUL configuration – temperature threshold
# ------------------------------------------------------------------
TEMP_FAILURE = 60.0          # °C – adjust based on your setup
MAX_HISTORY = 100
MIN_READINGS_FOR_RUL = 10

# In‑memory temperature history: list of (timestamp, temperature)
temp_history = []

SENSOR_NAMES = ['current', 'temperature', 'vibration']

# Helper to convert NumPy types to Python native types for JSON serialization
def convert_to_serializable(obj):
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj

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

        latest_temp = readings[-1][1]   # temperature is second element

        # ------------------------------------------------------------------
        # Autoencoder anomaly detection
        # ------------------------------------------------------------------
        seq = np.array(readings, dtype=np.float32)
        scaled = scaler.transform(seq)
        input_tensor = scaled.reshape(1, 20, 3)

        reconstructed = model.predict(input_tensor, verbose=0)

        mse = np.mean(np.square(input_tensor - reconstructed))
        mse_per_sensor = np.mean(np.square(input_tensor - reconstructed), axis=(0, 1))
        contributions = (mse_per_sensor / np.sum(mse_per_sensor)) * 100
        main_cause = SENSOR_NAMES[np.argmax(mse_per_sensor)]

        is_anomaly = bool(mse > THRESHOLD)
        health = 100 - (mse / THRESHOLD) * 100
        health = max(0, min(100, health))

        # ------------------------------------------------------------------
        # RUL calculation based on temperature trend
        # ------------------------------------------------------------------
        current_time = time.time()
        temp_history.append((current_time, latest_temp))

        if len(temp_history) > MAX_HISTORY:
            temp_history.pop(0)

        rul = -1.0   # -1 = not enough data or no degradation

        if len(temp_history) >= MIN_READINGS_FOR_RUL:
            times = np.array([t for t, _ in temp_history])
            temps = np.array([temp for _, temp in temp_history])

            t0 = times[0]
            times_hours = (times - t0) / 3600.0

            try:
                coeffs = np.polyfit(times_hours, temps, 1)
                slope = coeffs[0]       # °C per hour
                intercept = coeffs[1]

                if slope > 0:
                    t_fail_hours = (TEMP_FAILURE - intercept) / slope
                    current_hours = (current_time - t0) / 3600.0
                    rul_hours = t_fail_hours - current_hours

                    if latest_temp >= TEMP_FAILURE:
                        rul = 0.0
                    elif rul_hours < 0:
                        rul = 0.0
                    else:
                        rul = round(rul_hours, 2)
                else:
                    rul = -1.0
            except Exception as e:
                logger.warning(f"RUL fitting failed: {e}")
                rul = -1.0

        # ------------------------------------------------------------------
        # Prepare data for Firebase (convert numpy types to Python natives)
        # ------------------------------------------------------------------
        firebase_data = {
            "current": readings[-1][0],
            "temperature": latest_temp,
            "vibration": readings[-1][2],
            "is_anomaly": is_anomaly,
            "health": round(health, 2),
            "contrib_current": round(contributions[0], 1),
            "contrib_temperature": round(contributions[1], 1),
            "contrib_vibration": round(contributions[2], 1),
            "main_cause": main_cause,
            "rul": rul
        }

        # Convert any remaining numpy types to Python natives
        firebase_data = {k: convert_to_serializable(v) for k, v in firebase_data.items()}

        # ------------------------------------------------------------------
        # Update Firebase
        # ------------------------------------------------------------------
        try:
            fb.patch('/sensor', firebase_data)
            logger.info("Firebase updated with RUL")
        except Exception as e:
            logger.error(f"Firebase update failed: {e}")

        # ------------------------------------------------------------------
        # Build response
        # ------------------------------------------------------------------
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
            "main_cause": main_cause,
            "rul": rul
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