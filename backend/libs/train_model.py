import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
import os
from tqdm import tqdm

# Enable tqdm for pandas
tqdm.pandas(desc="Processing rows")

# Path to the combined phishing dataset (features already extracted)
DATA_PATH = "../datasets/phishing.csv"
MODEL_OUT = "../models/trainedmodel_test.pkl"

# Load dataset
df = pd.read_csv(DATA_PATH)

# Drop 'Index' column if present
if 'Index' in df.columns:
    df = df.drop(columns=['Index'])

# Split into features and label
X = df.drop(columns=['class'])
y = df['class']

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model using parallelized HistGradientBoostingClassifier
print("🚀 Training the model...")
model = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)
print("✅ Model training complete.")

# Predict on test set
y_pred = model.predict(X_test)

# Classification report
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Phishing', 'Legitimate']))

# Ensure output directory exists
os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

# Save the trained model
with open(MODEL_OUT, "wb") as f:
    pickle.dump(model, f)

print(f"✅ Model saved to {MODEL_OUT}")
