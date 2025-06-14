import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
from urllib.parse import urlparse
from libs.FeaturesExtract import FeatureExtraction
from libs.convert import convertion

# --- Environment variables and defaults ---
MODEL_PATH = os.getenv("PICKLE_MODEL_PATH", "./models/trainedmodel.pkl")

# Initialize Flask
app = Flask(__name__)
CORS(app, origins=[f"chrome-extension://{os.getenv('CHROME_EXT_ID')}"])

# Load model
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# Grab the feature names the model expects
FEATURE_NAMES = list(model.feature_names_in_)

def get_root_url(url):
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}/"

@app.route("/check_url", methods=["POST"])
def check_url():
    data = request.get_json(force=True)
    url  = data.get("url", "")
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    features = FeatureExtraction(url).getFeaturesList()
    df = pd.DataFrame([features], columns=FEATURE_NAMES)
    y_pred = model.predict(df)[0]
    try:
        y_prob = model.predict_proba(df)[0][1]
    except AttributeError:
        y_prob = None

    # Only use model prediction for decision
    decision = "Safe" if y_pred == 1 else "Phishing"

    # Use convert.py logic
    conversion_result = convertion(url, y_pred)

    return jsonify({
        "result": decision,
        "prediction": int(y_pred),
        "probability": round(float(y_prob), 4) if y_prob is not None else None,
        "conversion": conversion_result
    })

@app.route("/config/threshold", methods=["GET", "POST"])
def config_threshold():
    # Threshold config endpoint kept for compatibility, but not used in prediction
    if request.method == "POST":
        threshold = float(request.get_json().get("threshold"))
        return jsonify(status="ok", threshold=threshold)
    return jsonify(threshold=None)

@app.route("/predict_email", methods=["POST"])
def predict_email():
    data = request.get_json(force=True)
    email_text = data.get("email", "")
    if not email_text:
        return jsonify({"error": "No email content provided"}), 400

    # Load model and vectorizer
    with open("model/model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("model/vectorizer.pkl", "rb") as vec_file:
        vectorizer = pickle.load(vec_file)

    X = vectorizer.transform([email_text])
    prediction = model.predict(X)[0]
    confidence = model.predict_proba(X)[0][1]  # Probability of phishing

    verdict = "Phishing Email" if prediction == 1 else "Safe Email"
    return jsonify({
        "prediction": verdict,
        "confidence": float(confidence)
    })


if __name__ == "__main__":
    # This will start the server with auto-reload on code changes
    app.run(debug=True, port=5030)
