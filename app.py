from flask import Flask, request, jsonify
import os
import time
import traceback
from collections import deque

import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ============================================================
# FILE PATHS - Updated to .h5 for better compatibility
# ============================================================
MODEL_PATH = os.getenv("MODEL_PATH", "lstm_autoencoder.h5")
SCALER_PATH = os.getenv("SCALER_PATH", "scaler.save")
THRESHOLD_PATH = os.getenv("THRESHOLD_PATH", "threshold.npy")

# ============================================================
# INPUT SETTINGS
# ============================================================
WINDOW_SIZE = 20
NUM_FEATURES = 3
FEATURE_NAMES = ["current", "temperature", "vibration"]

# ============================================================
# GLOBALS
# ============================================================
model = None
scaler = None
threshold = None

def load_all():
    global model, scaler, threshold
    
    print("Loading model from:", MODEL_PATH)
    if os.path.exists(MODEL_PATH):
        # compile=False avoids errors with custom optimizers during loading
        model = load_model(MODEL_PATH, compile=False)
        print("Model loaded successfully.")
    else:
        print(f"ERROR: Model file not found at {MODEL_PATH}")

    print("Loading scaler from:", SCALER_PATH)
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        print("Scaler loaded successfully.")
    
    print("Loading threshold from:", THRESHOLD_PATH)
    if os.path.exists(THRESHOLD_PATH):
        threshold = np.load(THRESHOLD_PATH)
        print(f"Threshold loaded: {threshold}")

# Initial load
try:
    load_all()
except Exception as e:
    print("Startup failed error:", str(e))
    traceback.print_exc()

@app.route('/', methods=['GET'])
def index():
    return "Motor Health API is Running", 200

@app.route('/batch', methods=['POST'])
def batch_process():
    # ... (Your existing batch processing logic remains the same)
    return jsonify({"status": "processing"}), 200

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port)