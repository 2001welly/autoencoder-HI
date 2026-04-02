from flask import Flask, request, jsonify
import os
import time
import traceback
from collections import deque

# --- FORCE COMPATIBILITY WITH COLAB H5 MODELS ---
os.environ["TF_USE_LEGACY_KERAS"] = "1" 

import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ============================================================
# FILE PATHS
# ============================================================
MODEL_PATH = "lstm_autoencoder.h5"
SCALER_PATH = "scaler.save"
THRESHOLD_PATH = "threshold.npy"

# ============================================================
# SETTINGS
# ============================================================
WINDOW_SIZE = 20
FEATURE_NAMES = ["current", "temperature", "vibration"]

# ============================================================
# GLOBALS
# ============================================================
model = None
scaler = None
threshold = 0.0

def load_all():
    global model, scaler, threshold
    print("--- STARTING FILE LOADING ---")
    
    # 1. Load Threshold
    if os.path.exists(THRESHOLD_PATH):
        try:
            val = np.load(THRESHOLD_PATH)
            threshold = float(val)
            print(f"✅ Threshold loaded: {threshold}")
        except Exception as e:
            print(f"❌ Error loading threshold: {e}")
    
    # 2. Load Scaler
    if os.path.exists(SCALER_PATH):
        try:
            scaler = joblib.load(SCALER_PATH)
            print("✅ Scaler loaded successfully")
        except Exception as e:
            print(f"❌ Error loading scaler: {e}")

    # 3. Load Model
    if os.path.exists(MODEL_PATH):
        try:
            # compile=False avoids errors with Keras 3 metadata in Keras 2
            model = load_model(MODEL_PATH, compile=False)
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            traceback.print_exc()
    
    print("--- LOADING PROCESS COMPLETE ---")

# Trigger loading
load_all()

@app.route('/', methods=['GET'])
def index():
    if model and scaler and threshold > 0:
        return f"Motor API Active. Threshold: {threshold}", 200
    return "API Running but model files failed to load.", 500

@app.route('/batch', methods=['POST'])
def batch_process():
    t0 = time.time()
    try:
        data = request.get_json()
        if not data or 'readings' not in data:
            return jsonify({"error": "No readings provided"}), 400

        readings = np.array(data['readings']) # Expected shape: (20, 3)
        
        # 1. Scale data
        scaled_data = scaler.transform(readings)
        
        # 2. Prepare for LSTM (1, 20, 3)
        input_data = np.expand_dims(scaled_data, axis=0)
        
        # 3. Predict / Reconstruct
        reconstructed = model.predict(input_data, verbose=0)
        
        # 4. Calculate MAE
        mae_loss = np.mean(np.abs(input_data - reconstructed))
        
        # 5. Health Index
        health_index = max(0, min(100, (1 - (mae_loss / threshold)) * 100))
        is_anomaly = mae_loss > threshold

        return jsonify({
            "health_index": round(health_index, 2),
            "mae_loss": round(float(mae_loss), 6),
            "threshold": round(threshold, 6),
            "is_anomaly": bool(is_anomaly),
            "status": "success"
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Use Render's port
    port = int(os.getenv("PORT", 10000))
    app.run(host='0.0.0.0', port=port)