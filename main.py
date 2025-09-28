# Project ChronoTrust - AI Backend V2
# Supports the "Kinetic Calibration Game" with a Mixture-of-Experts model.
# Features smooth Sigmoid scoring for more accurate results.

import json
import math
import numpy as np
from sklearn.ensemble import IsolationForest
from flask import Flask
from flask_socketio import SocketIO, emit

# --- CONFIGURATION ---
# The number of data points to collect for EACH phase of the calibration game.
# The Flutter app sends data 10 times per second. 10 seconds = 100 samples.
SAMPLES_PER_PHASE = 100 

# --- APPLICATION SETUP ---
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# --- IN-MEMORY DATA STORAGE (for a single user demo) ---
# This will now store data for each phase separately.
# e.g., {'still': [[...], [...]], 'walking': [[...], ...]}
enrollment_data = {}

# This will store a separate AI model for each phase.
# e.g., {'still': IsolationForest(), 'walking': IsolationForest()}
models = {}

# --- HELPER FUNCTIONS ---

def parse_sensor_data(json_string):
    """Safely parses the JSON string from the client."""
    try:
        payload = json.loads(json_string)
        # V2: The payload now contains phase and data
        phase = payload.get('phase')
        data = payload.get('data')
        
        # Extract the 6 features
        features = [
            data.get('accel_x', 0), data.get('accel_y', 0), data.get('accel_z', 0),
            data.get('gyro_x', 0), data.get('gyro_y', 0), data.get('gyro_z', 0)
        ]
        return phase, features
    except (json.JSONDecodeError, AttributeError):
        return None, None

def sigmoid(x):
    """A smooth function that maps any number to a value between 0 and 1."""
    return 1 / (1 + math.exp(-x))

def normalize_score(raw_score):
    """
    Converts the raw anomaly score from the AI model into an intuitive 0-100% trust score.
    - Anomaly score of 0.1 (very normal) -> ~95% trust
    - Anomaly score of -0.1 (very abnormal) -> ~5% trust
    """
    # We scale the score to make the sigmoid curve more responsive
    scaled_score = raw_score * 15 
    # The sigmoid function gives a value from 0 to 1
    normalized = sigmoid(scaled_score)
    # Scale to 0-100 and format
    trust_score = normalized * 100
    return round(trust_score, 1)

# --- SOCKET.IO EVENT HANDLERS ---

@socketio.on('connect')
def handle_connect():
    """Handles a new client connection."""
    print('✅ Client connected')
    emit('status', {'message': 'Connected to ChronoTrust AI Backend V2'})

@socketio.on('enroll')
def handle_enroll(json_string):
    """
    Handles the enrollment process, now collecting data for different phases.
    """
    phase, features = parse_sensor_data(json_string)
    if features is None or phase is None or phase == 'intro' or phase == 'completing':
        return

    # Initialize the list for a new phase if it doesn't exist
    if phase not in enrollment_data:
        enrollment_data[phase] = []

    # Add data, but only up to the sample limit for that phase
    if len(enrollment_data[phase]) < SAMPLES_PER_PHASE:
        enrollment_data[phase].append(features)
        print(f" collected sample {len(enrollment_data[phase])}/{SAMPLES_PER_PHASE} for phase '{phase}'")

    # Check if we have collected enough data for ALL phases to complete enrollment
    # For the demo, we'll consider enrollment complete when the 'walking' phase is done.
    if phase == 'walking' and len(enrollment_data.get('walking', [])) >= SAMPLES_PER_PHASE:
        train_models()


def train_models():
    """
    Trains a separate AI model for each phase of collected data.
    This is the "Mixture of Experts" approach.
    """
    global models
    models = {} # Clear any old models
    print("\n--- Starting AI Model Training ---")

    for phase, data in enrollment_data.items():
        if len(data) > 0:
            print(f"Training model for phase: '{phase}' with {len(data)} samples...")
            # Create and train an Isolation Forest model for this specific phase
            model = IsolationForest(contamination='auto', random_state=42)
            model.fit(data)
            models[phase] = model # Store the trained model
            print(f"✅ Model for '{phase}' trained successfully.")
    
    print("--- AI Model Training Complete ---\n")
    emit('enrollment_complete', {'message': 'SECURE - Calibration Complete'})


@socketio.on('predict')
def handle_predict(json_string):
    """
    Handles real-time prediction using the mixture of expert models.
    """
    if not models:
        emit('error', {'message': 'Models not trained yet. Please enroll first.'})
        return

    _, features = parse_sensor_data(json_string)
    if features is None:
        return

    # Reshape data for a single prediction
    live_data = np.array(features).reshape(1, -1)

    # Get a score from EACH specialist model
    scores = []
    for phase, model in models.items():
        # The decision_function gives a raw anomaly score. Higher is more normal.
        raw_score = model.decision_function(live_data)[0]
        scores.append(raw_score)
        # print(f"Score from '{phase}' model: {raw_score:.3f}") # Debug log

    # For the final score, we take the BEST score from any of the models.
    # This means if the user's motion matches ANY of the trained contexts (sitting, walking, etc.),
    # their score will be high. This makes the system robust.
    best_raw_score = max(scores)
    
    # Use the smooth sigmoid function to get the final trust score
    trust_score = normalize_score(best_raw_score)

    status = "SECURE" if trust_score > 75 else "ANOMALY DETECTED"
    
    # print(f"Best Raw Score: {best_raw_score:.3f} -> Trust Score: {trust_score}%") # Debug log
    emit('prediction_result', {'trustScore': trust_score, 'status': status})


@socketio.on('disconnect')
def handle_disconnect():
    """Handles a client disconnection."""
    print('❌ Client disconnected')

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    # Use eventlet for production-ready WebSocket performance
    socketio.run(app, host='0.0.0.0', port=5001, debug=False)

