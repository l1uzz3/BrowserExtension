import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Paths to your CSV files
LEGIT_PATH = "../datasets/legit.csv"
PHISH_PATH = "../datasets/phis.csv"
MODEL_OUT = "../models/trainedmodel.pkl"

# Load legitimate URLs
legit_df = pd.read_csv(LEGIT_PATH)
legit_df['label'] = 'legitimate'

# Load phishing URLs
phish_df = pd.read_csv(PHISH_PATH)
phish_df['label'] = 'phishing'

# Combine datasets
df = pd.concat([legit_df, phish_df], ignore_index=True)

# Drop 'Index' column if present
if 'Index' in df.columns:
    df = df.drop(columns=['Index'])

# Drop non-feature columns (like 'url' if present)
feature_cols = [col for col in df.columns if col not in ['url', 'label']]
X = df[feature_cols]
y = df['label'].map({'legitimate': 1, 'phishing': 0})

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Ensure the output directory exists
os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

# Save the trained model
with open(MODEL_OUT, "wb") as f:
    pickle.dump(model, f)

print(f"Model trained and saved to {MODEL_OUT}")