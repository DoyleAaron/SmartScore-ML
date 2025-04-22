from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

# I had to get some help from ChatGPT to get this code working as I now need to structure it differently depending on the model

app = Flask(__name__)

# --- Points Prediction ---
@app.route('/predict/points', methods=['POST'])
def predict_points():
    data = request.json

    if not data or 'input' not in data or 'model' not in data:
        return jsonify({'error': 'Missing input or model filename'}), 400

    input_data = data['input']
    model_filename = data['model']
    model_path = os.path.join('app', 'ML', model_filename)

    try:
        model = joblib.load(model_path)
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)
        return jsonify({'prediction': round(float(prediction[0]), 1)})
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {e}'}), 500


# --- Transfer Recommendation ---
@app.route('/predict/transfer', methods=['POST'])
def predict_transfer():
    data = request.json

    if not data or 'input' not in data or 'model' not in data:
        return jsonify({'error': 'Missing input or model filename'}), 400

    input_data = data['input']
    model_filename = data['model']
    model_path = os.path.join('app', 'ML', model_filename)

    try:
        # Load model
        model = joblib.load(model_path)
        input_df = pd.DataFrame([input_data])

        # Get second closest match
        distances, indices = model.kneighbors(input_df)
        recommended_index = int(indices[0][1])

        # Load the correct CSV
        if 'goalkeeper' in model_filename:
            df = pd.read_csv('app/ML/24_25_prem_keeper_stats.csv')
        elif 'defender' in model_filename:
            df = pd.read_csv('app/ML/24_25_defending_clean.csv')
        else:
            df = pd.read_csv('app/ML/24_25_prem_player_stats_clean.csv')

        df = df.fillna(df.mean(numeric_only=True))

        player = df.iloc[recommended_index]

        print(f"Recommended player: {player['Player']}")  # <- DEBUG

        return jsonify({
            'player': player['Player'],
            'position': player['Pos'],
            'team': player['Squad'],
            'distance': round(float(distances[0][1]), 3)
        })

    except Exception as e:
        import traceback
        print("Error occurred:", e)
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
