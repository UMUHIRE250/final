import os
import joblib
from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(_name_)

# Correct path to the model and scaler
model_path = os.path.join('path_to_saved_model', 'best_model.joblib')
scaler_path = os.path.join('path_to_saved_model', 'scaler.save')

# Load the model and the scaler
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except Exception as e:
    model = None
    scaler = None
    print(f"Error loading model: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    if not model or not scaler:
        return jsonify({'error': 'Model or scaler not loaded'}), 500
    
    data = request.json
    features = ['date', 'b3001', 'b3002', 'b3003', 'b3004', 'b3005', 'b3006', 'b3007', 'b3008', 'b3009', 'b3010', 'b3011', 'b3012', 'b3013']
    
    try:
        # Extract feature values from the received JSON data
        input_data = [data[feature] for feature in features]
        
        # Preprocess input data
        input_df = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)
        location = prediction[0]
        
        return jsonify({'location': location})
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({'error': 'Invalid input data'}), 400

if _name_ == '_main_':
    app.run(debug=True)