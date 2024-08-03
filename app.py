from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for all routes

# Load the trained model and scaler
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([
        data['attendance'], 
        data['financial_situation'], 
        data['learning_environment'], 
        data['previous_grades']
    ]).reshape(1, -1)
    
    # Apply scaling
    features_scaled = scaler.transform(features)
    
    prediction = model.predict(features_scaled)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
