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
# Firebase & Model Config
# -------------------------------------------------------
FIREBASE_URL = "https://welly001-8ffca-default-rtdb.firebaseio.com/"
fb = firebase.FirebaseApplication(FIREBASE_URL, None)

# Load AI Components
model = tf.keras.models.load_model('lstm_autoencoder.keras')
scaler = joblib.load('scaler.save')
try:
    THRESHOLD = np.load('threshold.npy').item()
except:
    THRESHOLD = 0.08  # Fallback

# -------------------------------------------------------
# Industrial RUL & Smoothing Settings
# -------------------------------------------------------
ALPHA = 0.15                # Smoothing: 0.15 means health changes slowly/stably
HEALTH_FAILURE_LIMIT = 20.0 # Motor is "Failed" when health hits 20%
SAMPLE_INTERVAL = 10        # Record 1 health point every 10 seconds for trend
MIN_POINTS_FOR_RUL = 12     # Need 2 mins of data (12 * 10s) to show a trend
MAX_HISTORY = 500           # Keep up to ~80 minutes of history

# State Variables
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

        # 1. AI Inference (Multivariate - involves all sensors)
        seq = np.array(readings, dtype=np.float32)
        scaled = scaler.transform(seq)
        input_tensor = scaled.reshape(1, 20, 3)
        reconstructed = model.predict(input_tensor, verbose=0)

        # 2. Calculate Combined Error (MSE)
        mse = np.mean(np.square(input_tensor - reconstructed))
        mse_per_sensor = np.mean(np.square(input_tensor - reconstructed), axis=(0,1))
        contributions = (mse_per_sensor / np.sum(mse_per_sensor)) * 100
        main_cause = SENSOR_NAMES[np.argmax(mse_per_sensor)]

        # 3. Health Calculation with Exponential Smoothing
        # This uses the MSE from ALL parameters to determine health
        raw_health = 100 - (mse / THRESHOLD) * 100
        raw_health = max(0, min(100, raw_health))
        
        # Smooth the health to prevent "jittery" RUL
        smoothed_health = (ALPHA * raw_health) + (1 - ALPHA) * last_smoothed_health
        last_smoothed_health = smoothed_health

        # 4. Timed History Sampling
        # We only record a point for the RUL "Line" every 10 seconds
        if not health_history or (current_time - health_history[-1][0]) >= SAMPLE_INTERVAL:
            health_history.append((current_time, smoothed_health))
            if len(health_history) > MAX_HISTORY:
                health_history.pop(0)

        # 5. Multivariate RUL Calculation (Linear Regression)
        rul = -1.0
        if len(health_history) >= MIN_POINTS_FOR_RUL:
            times = np.array([t for t, _ in health_history])
            vals = np.array([h for _, h in health_history])

            # Convert time to relative hours for accurate slope
            t0 = times[0]
            times_hours = (times - t0) / 3600.0

            coeffs = np.polyfit(times_hours, vals, 1)
            slope = coeffs[0]      # Health loss per hour
            intercept = coeffs[1]

            # Only calculate RUL if the motor is actually degrading (negative slope)
            if slope < -0.1: 
                # Solve for time when Health = HEALTH_FAILURE_LIMIT (20%)
                # 20 = slope * hrs + intercept  =>  hrs = (20 - intercept) / slope
                fail_time_hrs = (HEALTH_FAILURE_LIMIT - intercept) / slope
                current_time_hrs = (current_time - t0) / 3600.0
                
                calculated_rul = fail_time_hrs - current_time_hrs
                rul = max(0.0, round(float(calculated_rul), 2))
            else:
                rul = -1.0 # Stable system, RUL is not applicable

        # 6. Final Results & Firebase Update
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
        fb.patch('/sensor', firebase_data)

        return firebase_data

    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}