from flask import Flask, request, jsonify 
from flasgger import Swagger 
import joblib 
import numpy as np  

app = Flask(__name__)
swagger = Swagger(app)

# Load model and encoder
try:
    rf_model = joblib.load("final_crop.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
except Exception as e:
    raise RuntimeError(f"Failed to load model files: {str(e)}")

REQUIRED_FIELDS = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'moisture']

@app.route('/predict', methods=['POST'])
def predict():
   
    data = request.get_json()

    if not data or "data" not in data:
        return jsonify({"error": "Input must be a JSON with 'data' field containing a list of entries."}), 400

    entries = data["data"]

    if not isinstance(entries, list) or len(entries) == 0:
        return jsonify({"error": "'data' must be a non-empty list."}), 400

    # Validate required fields
    for idx, entry in enumerate(entries):
        missing = [field for field in REQUIRED_FIELDS if field not in entry or entry[field] is None]
        if missing:
            return jsonify({
                "error": f"Entry {idx+1} is missing fields: {missing}"
            }), 400

    try:
        avg_values = [np.mean([entry[field] for entry in entries]) for field in REQUIRED_FIELDS]
        input_features = np.array(avg_values).reshape(1, -1)

        prediction = rf_model.predict(input_features)[0]
        confidence = np.max(rf_model.predict_proba(input_features)[0]) * 100
        predicted_crop = label_encoder.inverse_transform([prediction])[0]

        return jsonify({
            "prediction": [predicted_crop],
            "confidence": [round(confidence, 2)]
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5051, debug=True)
