import os
import pandas as pd
from pycaret.classification import load_model, predict_model
from FeaturesExtract import featureExtraction

# 1) Load your saved classifier once at startup
MODEL_PATH = r'C:\Users\lavin\Desktop\BrwoserExtension\backend\models\phishing_model.pkl'
model = load_model(MODEL_PATH)

def classify_url(url: str) -> dict:
    """
    Given a URL, extract features, run the saved model, 
    and return a dict with 'prediction' and 'score'.
    """
    # 2) Extract features into a 1×10 DataFrame
    feat_df = featureExtraction(url)  # must return a DataFrame with exactly the 10 columns

    # 3) Run PyCaret's predict_model
    result = predict_model(model, data=feat_df)
    # predict_model returns the original feat_df plus:
    #  - “Label” column (ground truth, if you provided it; here it’s NaN)
    #  - “prediction_label” column (the predicted class)
    #  - “Score” column (probability/confidence for the positive class)

    pred = result.at[0, 'prediction_label']
    score = result.at[0, 'Score']
    return {'prediction': pred, 'score': float(score)}

# 4) Example usage:
if __name__ == "__main__":
    test_url = "http://example-phishingsite.com/login"
    out = classify_url(test_url)
    print(f"URL: {test_url}\n→ Prediction: {out['prediction']}, Confidence: {out['score']:.3f}")
