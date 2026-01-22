"""
Flask Web Application for House Price Prediction
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import os
from lightweight_model import load_lightweight_model

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'model/house_price_model.json'

def load_model():
    """Load the trained model from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please train the model first.")
    
    model_data = load_lightweight_model(MODEL_PATH)
    return model_data['model'], model_data['scaler'], model_data['feature_columns']

# Load model at startup
try:
    model, scaler, feature_columns = load_model()
    print("Model loaded successfully!")
    print(f"Features: {feature_columns}")
except Exception as e:
    print(f"Error loading model: {e}")
    model, scaler, feature_columns = None, None, None

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html', features=feature_columns)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
        
        # Get input data from form
        input_data = []
        input_values = {}
        
        for feature in feature_columns:
            value = float(request.form.get(feature, 0))
            input_data.append(value)
            input_values[feature] = value
        
        # Reshape and scale input
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        return render_template('index.html', 
                             features=feature_columns,
                             prediction=prediction,
                             input_values=input_values)
    
    except Exception as e:
        return render_template('index.html', 
                             features=feature_columns,
                             error=f"Error making prediction: {str(e)}")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions."""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        
        # Prepare input
        input_data = [data.get(feature, 0) for feature in feature_columns]
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        
        return jsonify({
            'prediction': float(prediction),
            'input': data
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Debug mode should only be enabled in development, not production
    # Set DEBUG=True environment variable for development
    debug_mode = os.environ.get('DEBUG', 'False').lower() == 'true'
    app.run(debug=debug_mode, host='0.0.0.0', port=5000)
