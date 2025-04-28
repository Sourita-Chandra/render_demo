from flask import Flask, request, jsonify # type: ignore
import joblib # type: ignore
import numpy as np # type: ignore

app = Flask(__name__)

# Home Route
@app.route('/', methods=['GET'])
def home():
    return "Welcome to Crop Prediction API! Please POST data to /predict"

# Load the saved model and encoder
try:
    rf_model = joblib.load("final_crop.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
except Exception as e:
    raise RuntimeError(f"Failed to load model files: {str(e)}")

REQUIRED_FIELDS = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'moisture']

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No input data provided"}), 400

    if "data" in data and isinstance(data["data"], list):
        entries = data["data"]
        
        if not all(all(field in entry for field in REQUIRED_FIELDS) for entry in entries):
            return jsonify({"error": f"Each entry must contain all required fields: {REQUIRED_FIELDS}"}), 400
        
        avg_values = [np.mean([entry[field] for entry in entries]) for field in REQUIRED_FIELDS]
    
    elif all(field in data for field in REQUIRED_FIELDS):
        avg_values = [data[field] for field in REQUIRED_FIELDS]
    else:
        return jsonify({"error": f"Missing required fields: {REQUIRED_FIELDS}"}), 400

    try:
        input_features = np.array(avg_values).reshape(1, -1)

        prediction = rf_model.predict(input_features)[0]

        probabilities = rf_model.predict_proba(input_features)[0]
        confidence = np.max(probabilities) * 100

        predicted_crop = label_encoder.inverse_transform([prediction])[0]

        result = {
            "prediction": [predicted_crop],
            "confidence": [f"{confidence:.2f}%"]
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5051, debug=True)
