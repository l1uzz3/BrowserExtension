# ─── app.py ───────────────────────────────────────────────────────────────────────
import os
import json
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from urllib.parse import urlparse

from libs.FeaturesExtract import featureExtraction
from pycaret.classification import predict_model

# Load environment variables, model path, threshold, etc.
MODEL_PATH = os.getenv("PICKLE_MODEL_PATH", "models/phishing_model.pkl")
THRESHOLD  = float(os.getenv("PH_THRESHOLD", 0.8))
ALERTS_FILE = os.getenv("ALERTS_PATH", "alerts.json")

# Initialize Flask
app = Flask(__name__)
CORS(app, origins=[f"chrome-extension://{os.getenv('CHROME_EXT_ID')}"])

# Load the saved PyCaret pipeline + model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at '{MODEL_PATH}'")

MODEL = joblib.load(MODEL_PATH)

def get_root_url(url):
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}/"

@app.route("/check_url", methods=["POST"])
def check_url():
    data = request.get_json(force=True)
    url  = data.get("url", "")
    root_url = get_root_url(url)
    # 1) Extract features
    features = featureExtraction(root_url)   

    # 2) (optional) verify features columns here…

    # 3) Run PyCaret’s predict_model
    result = predict_model(MODEL, data=features)

    # Debug: log whatever columns were returned
    print("DEBUG: predict_model returned columns:", list(result.columns))

    # 4) Try to pick out the positive‐class probability
    if "Score" in result.columns:
        prob = float(result.loc[0, "Score"])
    else:
        # Remove all known feature columns, plus "Label" and "prediction_label"
        known = list(features.columns) + ["Label", "prediction_label"]
        prob_cols = [c for c in result.columns if c not in known]
        if len(prob_cols) == 1:
            prob = float(result.loc[0, prob_cols[0]])
        else:
            # Last fallback: call predict_proba on the raw array
            arr = features.to_numpy()
            prob = float(MODEL.predict_proba(arr)[0, 1])

    decision = "PHISHING" if prob >= THRESHOLD else "LEGITIMATE"
    return jsonify(decision=decision, score=prob)


@app.route("/report-risky-url", methods=["POST"])
def report_risky_url():
    entry = request.get_json(force=True)
    score     = entry.get("predictionScore", entry.get("score", 0.0))
    verdict   = entry.get("verdict", "Phishing" if score >= THRESHOLD else "Safe")
    timestamp = entry.get("timestamp", datetime.utcnow().isoformat() + "Z")
    record = {
        "url":     entry.get("url", ""),
        "score":   score,
        "verdict": verdict,
        "timestamp": timestamp
    }
    try:
        with open(ALERTS_FILE, "r", encoding="utf-8") as f:
            alerts = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        alerts = []
    alerts.append(record)
    with open(ALERTS_FILE, "w", encoding="utf-8") as f:
        json.dump(alerts, f, indent=2)
    return jsonify(status="ok"), 200


@app.route("/config/threshold", methods=["GET", "POST"])
def config_threshold():
    global THRESHOLD
    if request.method == "POST":
        THRESHOLD = float(request.get_json().get("threshold"))
        return jsonify(status="ok", threshold=THRESHOLD)
    return jsonify(threshold=THRESHOLD)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5030))
    app.run(host="0.0.0.0", port=port, debug=True)
