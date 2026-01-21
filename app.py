"""
Wine Cultivar Origin Prediction System - Web Application
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model components
MODEL_PATH = 'model/wine_cultivar_model.pkl'
SCALER_PATH = 'model/scaler.pkl'
FEATURES_PATH = 'model/feature_names.pkl'

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    print("✓ Model and components loaded successfully!")
    print(f"✓ Features: {feature_names}")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None
    scaler = None
    feature_names = None

@app.route('/')
def home():
    """Render home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        if model is None or scaler is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get features from form
        features = []
        for feature in feature_names:
            value = float(request.form.get(feature, 0))
            features.append(value)
        
        # Reshape and scale
        input_data = np.array(features).reshape(1, -1)
        input_scaled = scaler.transform(input_data)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        
        # Response
        result = {
            'prediction': f'Cultivar {prediction}',
            'prediction_code': int(prediction),
            'probability_cultivar_0': float(probabilities[0]),
            'probability_cultivar_1': float(probabilities[1]),
            'probability_cultivar_2': float(probabilities[2]),
            'confidence': float(max(probabilities)) * 100
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)