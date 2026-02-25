from fastapi import FastAPI, Request
import numpy as np
import tensorflow as tf
import joblib
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
# NOTE: Firebase writes are handled entirely by the ESP32 after it receives
# the JSON response from this server. No Firebase SDK needed here.

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
# RUL configuration
# ------------------------------------------------------------------
TEMP_FAILURE = 60.0          # °C – failure temperature
MAX_HISTORY = 200            # keep more history for stable regression
MIN_READINGS_FOR_RUL = 20   # need enough points before estimating

# In-memory stores (persist for the lifetime of this server process)
temp_history = []            # list of (timestamp, temperature)

# Exponential moving average for health (smooths fluctuations)
ema_health = None
EMA_ALPHA = 0.2              # lower = smoother, higher = more reactive

# Last valid RUL (for stability: only update when we have enough data)
last_rul = None

SENSOR_NAMES = ['current', 'temperature', 'vibration']


# ------------------------------------------------------------------
# Batch prediction endpoint
# ------------------------------------------------------------------
@app.post("/batch")
async def predict_batch(request: Request):
    global ema_health, last_rul

    try:
        data = await request.json()
        readings = data['readings']   # list of 20 lists [curr, temp, vibe]

        if len(readings) != 20:
            return {"error": "Exactly 20 readings required"}

        latest_temp = readings[-1][1]

        # ------------------------------------------------------------------
        # Autoencoder anomaly detection
        # ------------------------------------------------------------------
        seq = np.array(readings, dtype=np.float32)
        scaled = scaler.transform(seq)
        input_tensor = scaled.reshape(1, 20, 3)

        reconstructed = model.predict(input_tensor, verbose=0)

        mse = float(np.mean(np.square(input_tensor - reconstructed)))
        mse_per_sensor = np.mean(np.square(input_tensor - reconstructed), axis=(0, 1))
        contributions = (mse_per_sensor / np.sum(mse_per_sensor)) * 100
        main_cause = SENSOR_NAMES[int(np.argmax(mse_per_sensor))]

        is_anomaly = bool(mse > THRESHOLD)

        # Raw health from MSE
        raw_health = 100.0 - (mse / THRESHOLD) * 100.0
        raw_health = max(0.0, min(100.0, raw_health))

        # Smooth health with exponential moving average to prevent jumping
        if ema_health is None:
            ema_health = raw_health
        else:
            ema_health = EMA_ALPHA * raw_health + (1.0 - EMA_ALPHA) * ema_health

        health = round(ema_health, 2)

        # ------------------------------------------------------------------
        # RUL calculation — temperature linear trend
        # ------------------------------------------------------------------
        current_time = time.time()
        temp_history.append((current_time, latest_temp))

        # Cap history size
        if len(temp_history) > MAX_HISTORY:
            temp_history.pop(0)

        rul = last_rul if last_rul is not None else -1.0

        if len(temp_history) >= MIN_READINGS_FOR_RUL:
            times = np.array([t for t, _ in temp_history])
            temps = np.array([tmp for _, tmp in temp_history])

            t0 = times[0]
            times_hours = (times - t0) / 3600.0

            try:
                coeffs = np.polyfit(times_hours, temps, 1)
                slope = coeffs[0]        # °C per hour
                intercept = coeffs[1]    # °C at t0

                # Goodness-of-fit check: only trust regression if slope is meaningful
                # (at least 0.01 °C/hour increase — avoids noise-driven nonsense)
                if slope >= 0.01:
                    current_hours = (current_time - t0) / 3600.0
                    current_temp_pred = slope * current_hours + intercept

                    if latest_temp >= TEMP_FAILURE:
                        rul = 0.0
                    else:
                        # Hours until predicted temperature hits failure point
                        t_fail_hours = (TEMP_FAILURE - intercept) / slope
                        rul_hours = t_fail_hours - current_hours

                        if rul_hours < 0:
                            rul = 0.0
                        else:
                            rul = round(float(rul_hours), 2)

                    # Sanity cap: RUL shouldn't be astronomically large
                    if rul > 9999:
                        rul = -1.0

                    last_rul = rul
                else:
                    # Temperature not rising — keep last known RUL or report N/A
                    rul = last_rul if last_rul is not None else -1.0

            except Exception as e:
                logger.warning(f"RUL fitting failed: {e}")
                rul = last_rul if last_rul is not None else -1.0

        # ------------------------------------------------------------------
        # Build and return response — ESP32 writes this to Firebase directly
        # ------------------------------------------------------------------
        contrib_c = round(float(contributions[0]), 1)
        contrib_t = round(float(contributions[1]), 1)
        contrib_v = round(float(contributions[2]), 1)

        logger.info(f"Served | anomaly={is_anomaly} health={health:.1f}% RUL={rul} cause={main_cause}")

        return {
            "is_anomaly":   is_anomaly,
            "mse":          round(mse, 6),
            "health":       health,
            "threshold":    float(THRESHOLD),
            "sensor_contributions": {
                "current":     contrib_c,
                "temperature": contrib_t,
                "vibration":   contrib_v
            },
            "main_cause":   main_cause,
            "rul":          rul
        }

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
async def health_check():
    return {"status": "healthy"}
