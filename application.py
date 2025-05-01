from flask import Flask, request, jsonify
from flasgger import Swagger
import joblib
import numpy as np

app = Flask(__name__)
swagger = Swagger(app)

# Load model and label encoder
rf_model = joblib.load("final_crop.pkl")
label_encoder = joblib.load("label_encoder.pkl")

REQUIRED_FIELDS = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'moisture']

@app.route('/predict', methods=['POST'])
def predict():
   
    data = request.get_json()

    if not data or "data" not in data:
        return jsonify({"error": "Missing 'data' field"}), 400

    entries = data["data"]

    for entry in entries:
        for field in REQUIRED_FIELDS:
            if field not in entry or entry[field] is None:
                return jsonify({"error": f"Missing field: {field}"}), 400

    try:
        avg_values = [np.mean([e[field] for e in entries]) for field in REQUIRED_FIELDS]
        input_features = np.array(avg_values).reshape(1, -1)

        prediction = rf_model.predict(input_features)[0]
        confidence = np.max(rf_model.predict_proba(input_features)[0]) * 100
        crop = label_encoder.inverse_transform([prediction])[0]

        return jsonify({
            "prediction": [crop],
            "confidence": [round(confidence, 2)]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5051, debug=True)
