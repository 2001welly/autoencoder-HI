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
HEALTH_FAILURE = 20.0        # % — health at which machine is considered failed
MAX_HISTORY = 200            # rolling window of (timestamp, health) pairs
MIN_READINGS_FOR_RUL = 20   # minimum number of readings before attempting fit
MIN_WINDOW_HOURS = 0.5       # minimum 30 minutes of history before trusting the trend

# In-memory stores (persist for the lifetime of this server process)
health_history = []          # list of (timestamp, smoothed_health)

# Exponential moving average for health (smooths fluctuations)
ema_health = None
EMA_ALPHA = 0.2              # lower = smoother, higher = more reactive

# Last valid RUL (held when slope is flat — avoids N/A flickering)
last_rul = None

SENSOR_NAMES = ['current', 'temperature', 'vibration']


# ------------------------------------------------------------------
# Batch prediction endpoint
# ------------------------------------------------------------------
@app.post("/batch")
async def predict_batch(request: Request):
    global ema_health, last_rul, health_history

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
        # RUL calculation — health degradation trend
        # Uses smoothed health history instead of raw temperature so that
        # RUL reflects overall machine condition, not just one sensor.
        # ------------------------------------------------------------------
        current_time = time.time()
        health_history.append((current_time, health))

        if len(health_history) > MAX_HISTORY:
            health_history.pop(0)

        rul = last_rul if last_rul is not None else -1.0

        if len(health_history) >= MIN_READINGS_FOR_RUL:
            times   = np.array([t for t, _ in health_history])
            healths = np.array([h for _, h in health_history])

            t0 = times[0]
            elapsed_hours = (current_time - t0) / 3600.0
            times_hours   = (times - t0) / 3600.0

            # Require at least MIN_WINDOW_HOURS of history before trusting the trend.
            # 20 readings at 500 ms = ~10 seconds — far too short for a meaningful slope.
            if elapsed_hours >= MIN_WINDOW_HOURS:
                try:
                    coeffs    = np.polyfit(times_hours, healths, 1)
                    slope     = coeffs[0]      # % health per hour (negative = degrading)
                    intercept = coeffs[1]      # health at t0

                    logger.info(f"RUL fit | slope={slope:.4f} %/h  elapsed={elapsed_hours:.3f}h  health={health:.1f}%")

                    # Only compute RUL when health is genuinely declining.
                    # -0.5 %/hour = 12% drop per day — a real degradation signal.
                    if slope < -0.5:
                        current_hours = elapsed_hours

                        if health <= HEALTH_FAILURE:
                            rul = 0.0
                        else:
                            # Hours until projected health hits HEALTH_FAILURE
                            t_fail_hours = (HEALTH_FAILURE - intercept) / slope
                            rul_hours    = t_fail_hours - current_hours
                            rul = 0.0 if rul_hours < 0 else round(float(rul_hours), 2)

                        # Sanity cap
                        if rul > 9999:
                            rul = -1.0

                        last_rul = rul
                    else:
                        # Health stable or improving — clear stale value, show N/A
                        last_rul = None
                        rul = -1.0

                except Exception as e:
                    logger.warning(f"RUL fitting failed: {e}")
                    rul = last_rul if last_rul is not None else -1.0
            else:
                logger.info(f"RUL pending | only {elapsed_hours*60:.1f} min of history (need {MIN_WINDOW_HOURS*60:.0f} min)")
                rul = -1.0

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
