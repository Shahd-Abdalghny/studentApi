from flask import Flask, request, jsonify
from flask_cors import CORS 
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app) 
# Load pipeline
with open("student_pipeline.pkl", "rb") as f:
    pipeline = joblib.load(f)

@app.route("/")
def home():
    return "Student Performance Prediction API"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        input_df = pd.DataFrame([data])

        prediction = pipeline.predict(input_df)

        return jsonify({
            "predicted_score": round(float(prediction[0]),2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
