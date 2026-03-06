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

# --- Firebase initialization ---
FIREBASE_URL = "https://welly001-8ffca-default-rtdb.firebaseio.com/"
fb = firebase.FirebaseApplication(FIREBASE_URL, None)

# --- Load ML components ---
model = tf.keras.models.load_model('lstm_autoencoder.keras')
scaler = joblib.load('scaler.save')
try:
    THRESHOLD = np.load('threshold.npy').item()
except:
    THRESHOLD = 0.08

# --- Constants for Stability ---
ALPHA = 0.15                # Smoothing factor (Lower = smoother health)
HEALTH_SAMPLE_INTERVAL = 10 # Only record health every 10 seconds for RUL trend
MIN_POINTS_FOR_RUL = 10     # Need 100 seconds of data before predicting RUL
FAILURE_THRESHOLD = 20.0    # We consider the machine "failed" at 20% health
MAX_HISTORY = 500

# --- State Variables ---
health_history = []         # List of (timestamp, smoothed_health)
last_raw_health = 100.0     # Global for smoothing

SENSOR_NAMES = ['current', 'temperature', 'vibration']

def convert_to_serializable(obj):
    if isinstance(obj, (np.float32, np.float64)): return float(obj)
    if isinstance(obj, (np.int32, np.int64)): return int(obj)
    return obj

@app.post("/batch")
async def predict_batch(request: Request):
    global last_raw_health
    try:
        data = await request.json()
        readings = data['readings']
        current_time = time.time()

        # 1. Prediction & MSE
        seq = np.array(readings, dtype=np.float32)
        scaled = scaler.transform(seq)
        input_tensor = scaled.reshape(1, 20, 3)
        reconstructed = model.predict(input_tensor, verbose=0)
        
        mse = np.mean(np.square(input_tensor - reconstructed))
        mse_per_sensor = np.mean(np.square(input_tensor - reconstructed), axis=(0,1))
        contributions = (mse_per_sensor / np.sum(mse_per_sensor)) * 100
        main_cause = SENSOR_NAMES[np.argmax(mse_per_sensor)]
        is_anomaly = bool(mse > THRESHOLD)

        # 2. Exponential Smoothing for Health
        # This prevents the RUL from jumping because of a 1-second vibration spike
        raw_health = max(0, min(100, 100 - (mse / THRESHOLD) * 100))
        smoothed_health = (ALPHA * raw_health) + (1 - ALPHA) * last_raw_health
        last_raw_health = smoothed_health

        # 3. Timed History Sampling (The Fix for the 0.1h RUL error)
        # We only add to history if enough time has passed to show a real trend
        if not health_history or (current_time - health_history[-1][0]) >= HEALTH_SAMPLE_INTERVAL:
            health_history.append((current_time, smoothed_health))
            if len(health_history) > MAX_HISTORY:
                health_history.pop(0)

        # 4. Stable RUL Estimation
        rul = -1.0
        if len(health_history) >= MIN_POINTS_FOR_RUL:
            times = np.array([t for t, _ in health_history])
            vals = np.array([h for _, h in health_history])
            
            # Use relative time in hours for the regression
            t0 = times[0]
            times_hours = (times - t0) / 3600.0
            
            coeffs = np.polyfit(times_hours, vals, 1)
            slope, intercept = coeffs[0], coeffs[1]

            # Only predict RUL if the health is actually trending DOWN
            if slope < -0.05: # Slope must be negative (degrading)
                # Solve: FAILURE_THRESHOLD = slope * fail_time + intercept
                fail_time_hours = (FAILURE_THRESHOLD - intercept) / slope
                current_time_hours = (current_time - t0) / 3600.0
                rul_val = fail_time_hours - current_time_hours
                rul = max(0.0, round(float(rul_val), 2))
            else:
                rul = -1.0 # Stable system

        # 5. Firebase & Response
        firebase_data = {
            "current": readings[-1][0],
            "temperature": readings[-1][1],
            "vibration": readings[-1][2],
            "is_anomaly": is_anomaly,
            "health": round(smoothed_health, 2),
            "contrib_current": round(contributions[0], 1),
            "contrib_temperature": round(contributions[1], 1),
            "contrib_vibration": round(contributions[2], 1),
            "main_cause": main_cause,
            "rul": rul
        }
        
        firebase_data = {k: convert_to_serializable(v) for k,v in firebase_data.items()}
        fb.patch('/sensor', firebase_data)

        return firebase_data

    except Exception as e:
        logger.error(f"Error: {e}")
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}