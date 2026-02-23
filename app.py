from fastapi import FastAPI, Request
import numpy as np
import tensorflow as tf
import joblib
import logging
import time
from datetime import datetime

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
# RUL configuration – temperature threshold
# ------------------------------------------------------------------
# Absolute failure temperature (°C) – adjust based on your setup
TEMP_FAILURE = 60.0   # Example: motor housing failure at 60°C
# Maximum number of temperature readings to keep for trend analysis
MAX_HISTORY = 100
# Minimum number of readings required to compute RUL
MIN_READINGS_FOR_RUL = 10

# In‑memory temperature history: list of (timestamp, temperature)
temp_history = []

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

        # Extract the latest temperature (last reading in batch)
        latest_temp = readings[-1][1]   # temperature is second element

        # ------------------------------------------------------------------
        # Autoencoder anomaly detection (existing code)
        # ------------------------------------------------------------------
        seq = np.array(readings, dtype=np.float32)           # (20, 3)
        scaled = scaler.transform(seq)                        # (20, 3)
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
        # Add current reading to history (with server timestamp)
        current_time = time.time()
        temp_history.append((current_time, latest_temp))

        # Keep only the last MAX_HISTORY readings
        if len(temp_history) > MAX_HISTORY:
            temp_history.pop(0)

        # Default RUL value (unknown / not degrading)
        rul = -1.0   # -1 indicates not enough data or no degradation

        if len(temp_history) >= MIN_READINGS_FOR_RUL:
            # Extract times and temperatures
            times = np.array([t for t, _ in temp_history])
            temps = np.array([temp for _, temp in temp_history])

            # Convert times to hours since first reading (for numerical stability)
            t0 = times[0]
            times_hours = (times - t0) / 3600.0

            # Fit linear trend: temperature = slope * time + intercept
            # Use polyfit with degree 1
            try:
                coeffs = np.polyfit(times_hours, temps, 1)
                slope = coeffs[0]       # °C per hour
                intercept = coeffs[1]

                # Only consider positive slope (degradation)
                if slope > 0:
                    # Time to reach failure threshold (hours)
                    # T_failure = slope * t_fail + intercept  => t_fail = (T_failure - intercept)/slope
                    t_fail_hours = (TEMP_FAILURE - intercept) / slope
                    # Current time in hours since t0
                    current_hours = (current_time - t0) / 3600.0
                    rul_hours = t_fail_hours - current_hours

                    # If already above threshold, set RUL to 0
                    if latest_temp >= TEMP_FAILURE:
                        rul = 0.0
                    elif rul_hours < 0:
                        rul = 0.0
                    else:
                        rul = round(rul_hours, 2)   # hours, with 2 decimals
                else:
                    # Slope <= 0 means no degradation or cooling
                    rul = -1.0   # signal "no degradation trend"
            except Exception as e:
                logger.warning(f"RUL fitting failed: {e}")
                rul = -1.0

        # ------------------------------------------------------------------
        # Update Firebase (existing code + RUL)
        # ------------------------------------------------------------------
        if Firebase.ready() and signupOK:   # signupOK and fbdo need to be accessible
            FirebaseJson json
            json.set("current", readings[-1][0])   # last current
            json.set("temperature", latest_temp)
            json.set("vibration", readings[-1][2])
            json.set("is_anomaly", is_anomaly)
            json.set("health", round(health, 2))
            json.set("contrib_current", round(contributions[0], 1))
            json.set("contrib_temperature", round(contributions[1], 1))
            json.set("contrib_vibration", round(contributions[2], 1))
            json.set("main_cause", main_cause)
            json.set("rul", rul)          # add RUL field

            if Firebase.RTDB.updateNode(&fbdo, "/sensor", &json):
                logger.info("Firebase updated with RUL")
            else:
                logger.error(f"Firebase update failed: {fbdo.errorReason()}")

        # ------------------------------------------------------------------
        # Build response (same as before, optionally include RUL)
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
            "rul": rul                     # include in API response if needed
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