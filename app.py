from fastapi import FastAPI, Request
import numpy as np
import tensorflow as tf
import joblib
import logging
import time
import requests
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# -------------------------------------------------------
# Firebase & Model Configuration
# -------------------------------------------------------
# Adding .json to the end of the URL is required for the Firebase REST API
FIREBASE_URL = "https://welly001-8ffca-default-rtdb.firebaseio.com/sensor.json"

# Load AI Components
try:
    model = tf.keras.models.load_model('lstm_autoencoder.keras')
    scaler = joblib.load('scaler.save')
    try:
        THRESHOLD = np.load('threshold.npy').item()
    except:
        THRESHOLD = 0.08  # Default safety threshold
    logger.info("✅ AI Models and Scaler loaded successfully")
except Exception as e:
    logger.error(f"❌ Initialization failed: {e}")

# -------------------------------------------------------
# Prognostics & Smoothing Settings
# -------------------------------------------------------
ALPHA = 0.15                # Smoothing: Low value = stable health/RUL
HEALTH_FAILURE_LIMIT = 20.0 # RUL counts down to this health percentage
SAMPLE_INTERVAL = 10        # Record 1 point every 10s to calculate trend
MIN_POINTS_FOR_RUL = 12     # Need ~2 mins of data before predicting
MAX_HISTORY = 500           

# Global State
health_history = []         # List of (timestamp, smoothed_health)
last_smoothed_health = 100.0
SENSOR_NAMES = ['current', 'temperature', 'vibration']

def convert_to_serializable(obj):
    if isinstance(obj, (np.float32, np.float64)): return float(obj)
    if isinstance(obj, (np.int32, np.int64)): return int(obj)
    return obj

@app.post("/batch")
async def predict_batch(request: Request):
    global last_smoothed_health, health_history
    
    try:
        data = await request.json()
        readings = data['readings']
        current_time = time.time()

        # 1. AI Inference (Multivariate Analysis)
        seq = np.array(readings, dtype=np.float32)
        scaled = scaler.transform(seq)
        input_tensor = scaled.reshape(1, 20, 3)
        reconstructed = model.predict(input_tensor, verbose=0)

        # 2. Calculate Combined Error (MSE) from all 3 sensors
        mse = np.mean(np.square(input_tensor - reconstructed))
        mse_per_sensor = np.mean(np.square(input_tensor - reconstructed), axis=(0,1))
        contributions = (mse_per_sensor / np.sum(mse_per_sensor)) * 100
        main_cause = SENSOR_NAMES[np.argmax(mse_per_sensor)]

        # 3. Health Calculation with Exponential Smoothing
        raw_health = 100 - (mse / THRESHOLD) * 100
        raw_health = max(0, min(100, raw_health))
        
        # This prevents RUL from jumping due to momentary noise
        smoothed_health = (ALPHA * raw_health) + (1 - ALPHA) * last_smoothed_health
        last_smoothed_health = smoothed_health

        # 4. History Tracking for RUL
        if not health_history or (current_time - health_history[-1][0]) >= SAMPLE_INTERVAL:
            health_history.append((current_time, smoothed_health))
            if len(health_history) > MAX_HISTORY:
                health_history.pop(0)

        # 5. RUL Calculation (Predicting when health hits 20%)
        rul = -1.0
        if len(health_history) >= MIN_POINTS_FOR_RUL:
            times = np.array([t for t, _ in health_history])
            vals = np.array([h for _, h in health_history])

            t0 = times[0]
            times_hours = (times - t0) / 3600.0 # Normalize to hours

            coeffs = np.polyfit(times_hours, vals, 1)
            slope = coeffs[0]  # Health decay per hour
            intercept = coeffs[1]

            if slope < -0.1: # If motor is actually degrading
                fail_time_hrs = (HEALTH_FAILURE_LIMIT - intercept) / slope
                current_time_hrs = (current_time - t0) / 3600.0
                rul = max(0.0, round(float(fail_time_hrs - current_time_hrs), 2))
            else:
                rul = -1.0 # System stable

        # 6. Prepare and Send Data to Firebase via REST API
        firebase_data = {
            "current": readings[-1][0],
            "temperature": readings[-1][1],
            "vibration": readings[-1][2],
            "is_anomaly": bool(mse > THRESHOLD),
            "health": round(smoothed_health, 2),
            "rul": rul,
            "main_cause": main_cause,
            "contrib_current": round(contributions[0], 1),
            "contrib_temperature": round(contributions[1], 1),
            "contrib_vibration": round(contributions[2], 1)
        }

        firebase_data = {k: convert_to_serializable(v) for k, v in firebase_data.items()}
        
        # Using requests.patch for the Firebase REST API
        requests.patch(FIREBASE_URL, data=json.dumps(firebase_data))

        return firebase_data

    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}