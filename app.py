from fastapi import FastAPI, Request
import numpy as np
import tensorflow as tf
import joblib
import logging
import time
from firebase import firebase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# -------------------------------------------------------
# Firebase initialization
# -------------------------------------------------------
FIREBASE_URL = "https://welly001-8ffca-default-rtdb.firebaseio.com/"
fb = firebase.FirebaseApplication(FIREBASE_URL, None)

# -------------------------------------------------------
# Load ML components
# -------------------------------------------------------
try:
    model = tf.keras.models.load_model('lstm_autoencoder.keras')
    logger.info("✅ Autoencoder model loaded")
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
    logger.info(f"✅ Threshold loaded: {THRESHOLD}")
except Exception as e:
    logger.warning(f"Threshold load failed, using default 0.08")
    THRESHOLD = 0.08

# -------------------------------------------------------
# Health degradation history
# -------------------------------------------------------
health_history = []

MAX_HISTORY = 200
MIN_POINTS_FOR_RUL = 15

SENSOR_NAMES = ['current', 'temperature', 'vibration']

# -------------------------------------------------------
# Utility function
# -------------------------------------------------------
def convert_to_serializable(obj):
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj


# -------------------------------------------------------
# Batch Prediction Endpoint
# -------------------------------------------------------
@app.post("/batch")
async def predict_batch(request: Request):

    try:
        data = await request.json()
        readings = data['readings']

        if len(readings) != 20:
            return {"error": "Exactly 20 readings required"}

        # -------------------------------------------------------
        # Prepare data for Autoencoder
        # -------------------------------------------------------
        seq = np.array(readings, dtype=np.float32)

        scaled = scaler.transform(seq)

        input_tensor = scaled.reshape(1, 20, 3)

        reconstructed = model.predict(input_tensor, verbose=0)

        # -------------------------------------------------------
        # Reconstruction error
        # -------------------------------------------------------
        mse = np.mean(np.square(input_tensor - reconstructed))

        mse_per_sensor = np.mean(np.square(input_tensor - reconstructed), axis=(0,1))

        contributions = (mse_per_sensor / np.sum(mse_per_sensor)) * 100

        main_cause = SENSOR_NAMES[np.argmax(mse_per_sensor)]

        # -------------------------------------------------------
        # Anomaly detection
        # -------------------------------------------------------
        is_anomaly = bool(mse > THRESHOLD)

        # -------------------------------------------------------
        # Health Index
        # -------------------------------------------------------
        health = 100 - (mse / THRESHOLD) * 100
        health = max(0, min(100, health))

        # -------------------------------------------------------
        # Store health history
        # -------------------------------------------------------
        current_time = time.time()

        health_history.append((current_time, health))

        if len(health_history) > MAX_HISTORY:
            health_history.pop(0)

        # -------------------------------------------------------
        # Prognostics RUL estimation
        # -------------------------------------------------------
        rul = -1.0

        if len(health_history) >= MIN_POINTS_FOR_RUL:

            times = np.array([t for t,_ in health_history])
            health_values = np.array([h for _,h in health_history])

            t0 = times[0]

            times_hours = (times - t0) / 3600.0

            try:

                coeffs = np.polyfit(times_hours, health_values, 1)

                slope = coeffs[0]
                intercept = coeffs[1]

                if slope < 0:

                    failure_time = -intercept / slope

                    current_time_hours = (current_time - t0) / 3600.0

                    rul_hours = failure_time - current_time_hours

                    if rul_hours < 0:
                        rul = 0.0
                    else:
                        rul = round(rul_hours,2)

                else:
                    rul = -1.0

            except Exception as e:
                logger.warning(f"RUL calculation failed: {e}")
                rul = -1.0

        # -------------------------------------------------------
        # Prepare Firebase update
        # -------------------------------------------------------
        firebase_data = {

            "current": readings[-1][0],
            "temperature": readings[-1][1],
            "vibration": readings[-1][2],

            "is_anomaly": is_anomaly,

            "health": round(health,2),

            "contrib_current": round(contributions[0],1),
            "contrib_temperature": round(contributions[1],1),
            "contrib_vibration": round(contributions[2],1),

            "main_cause": main_cause,

            "rul": rul
        }

        firebase_data = {k: convert_to_serializable(v) for k,v in firebase_data.items()}

        # -------------------------------------------------------
        # Update Firebase
        # -------------------------------------------------------
        try:
            fb.patch('/sensor', firebase_data)
            logger.info("Firebase updated successfully")

        except Exception as e:
            logger.error(f"Firebase update failed: {e}")

        # -------------------------------------------------------
        # API response
        # -------------------------------------------------------
        response = {

            "is_anomaly": is_anomaly,

            "mse": round(float(mse),6),

            "health": round(float(health),2),

            "threshold": float(THRESHOLD),

            "sensor_contributions": {

                "current": round(float(contributions[0]),1),
                "temperature": round(float(contributions[1]),1),
                "vibration": round(float(contributions[2]),1)
            },

            "main_cause": main_cause,

            "rul": rul
        }

        return response

    except KeyError as e:
        logger.warning(f"Missing key: {e}")
        return {"error": f"Missing field {e}"}

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": str(e)}


# -------------------------------------------------------
# Health check
# -------------------------------------------------------
@app.get("/health")
async def health_check():
    return {"status": "healthy"}