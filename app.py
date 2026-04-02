from flask import Flask, request, jsonify
import os
import time
import traceback
from collections import deque

# --- FORCES COMPATIBILITY WITH COLAB MODELS ---
os.environ["TF_USE_LEGACY_KERAS"] = "1" 

import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ============================================================
# FILE PATHS - Fixed to match your GitHub exactly
# ============================================================
MODEL_PATH = "lstm_autoencoder.h5"
SCALER_PATH = "scaler.save"
THRESHOLD_PATH = "threshold.npy"

# ============================================================
# GLOBALS
# ============================================================
model = None
scaler = None
threshold = 0.0  # Default to 0 until loaded

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
    else:
        print(f"❌ CRITICAL: {THRESHOLD_PATH} not found!")

    # 2. Load Scaler
    if os.path.exists(SCALER_PATH):
        try:
            scaler = joblib.load(SCALER_PATH)
            print("✅ Scaler loaded successfully")
        except Exception as e:
            print(f"❌ Error loading scaler: {e}")
    else:
        print(f"❌ CRITICAL: {SCALER_PATH} not found!")

    # 3. Load Model
    if os.path.exists(MODEL_PATH):
        try:
            # compile=False is safer for production inference
            model = load_model(MODEL_PATH, compile=False)
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    else:
        print(f"❌ CRITICAL: {MODEL_PATH} not found!")
    
    print("--- LOADING PROCESS COMPLETE ---")

# Trigger loading immediately on startup
load_all()

@app.route('/', methods=['GET'])
def index():
    if model and scaler and threshold > 0:
        return f"API Active. Threshold: {threshold}", 200
    return "API Running but files are missing!", 500

@app.route('/batch', methods=['POST'])
def batch_process():
    # ... (Keep your existing batch logic here)
    # Just ensure it uses the global 'threshold' variable
    pass

if __name__ == "__main__":
    # Render uses the PORT environment variable
    port = int(os.getenv("PORT", 10000))
    app.run(host='0.0.0.0', port=port)