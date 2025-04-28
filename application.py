from flask import Flask, request, jsonify # type: ignore
import joblib # type: ignore
import numpy as np # type: ignore

app = Flask(__name__)

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

    # Check if input is a batch (list inside "data") or a single object
    if "data" in data and isinstance(data["data"], list):
        entries = data["data"]
        
        if not all(all(field in entry for field in REQUIRED_FIELDS) for entry in entries):
            return jsonify({"error": f"Each entry must contain all required fields: {REQUIRED_FIELDS}"}), 400
        
        # Compute average of each attribute
        avg_values = [np.mean([entry[field] for entry in entries]) for field in REQUIRED_FIELDS]
    
    elif all(field in data for field in REQUIRED_FIELDS):
        # Single input case
        avg_values = [data[field] for field in REQUIRED_FIELDS]
    else:
        return jsonify({"error": f"Missing required fields: {REQUIRED_FIELDS}"}), 400

    try:
        # Convert input into a NumPy array
        input_features = np.array(avg_values).reshape(1, -1)

        # Predict the crop label
        prediction = rf_model.predict(input_features)[0]

        # Get prediction probabilities
        probabilities = rf_model.predict_proba(input_features)[0]
        confidence = np.max(probabilities) * 100  # Convert to percentage

        # Decode the predicted crop name
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
