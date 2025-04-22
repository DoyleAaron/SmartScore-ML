from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    if not data or 'input' not in data or 'model' not in data:
        return jsonify({'error': 'Missing input or model filename'}), 400

    input_data = data['input']
    model_filename = data['model']

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app', 'ML', model_filename)

    try:
        model = joblib.load(model_path)
    except Exception as e:
        return jsonify({'error': f'Error loading model: {e}'}), 500

    try:
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)
        return jsonify({'prediction': round(float(prediction[0]), 1)})
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {e}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
