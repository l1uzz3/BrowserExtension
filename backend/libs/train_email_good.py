import os
import time
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

# ─── CONFIGURATION ─────────────────────────────
EMAIL_DATA_PATH = "../datasets/emails.csv"
RESULTS_CSV_PATH = "models/email_classification_results.csv"
PLOT_PATH = "models/email_classification_metrics_plot.png"
CM_FOLDER = "models/email_confusion_matrices"
TOP_FEATURES_PATH = "models/email_top_features.png"

os.makedirs("models", exist_ok=True)
os.makedirs(CM_FOLDER, exist_ok=True)

# ─── LOAD DATA ────────────────────────────────
print("📂 Loading email dataset...")
df = pd.read_csv(EMAIL_DATA_PATH)
df['Email Text'] = df['Email Text'].fillna('')
df['Email Type'] = df['Email Type'].fillna('')
df = df[df['Email Type'].isin(['Safe Email', 'Phishing Email'])]

df['label'] = df['Email Type'].map({'Safe Email': 0, 'Phishing Email': 1})
print(f"✅ Loaded {len(df)} valid emails.")

# ─── VECTORIZATION ────────────────────────────
print("🔤 Vectorizing emails (word-level TF-IDF)...")
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X = vectorizer.fit_transform(df['Email Text'])
y = df['label']

# ─── FEATURE IMPORTANCE ───────────────────────
print("🔎 Performing Chi² feature analysis...")
chi2_scores, _ = chi2(X, y)
top_indices = np.argsort(chi2_scores)[::-1][:20]
top_features = vectorizer.get_feature_names_out()[top_indices]
top_scores = chi2_scores[top_indices]

plt.figure(figsize=(10, 6))
plt.barh(range(len(top_features)), top_scores[::-1], color='green')
plt.yticks(range(len(top_features)), top_features[::-1])
plt.xlabel("Chi² Score")
plt.title("Top Discriminative Email Tokens")
plt.tight_layout()
plt.savefig(TOP_FEATURES_PATH)
plt.close()
print(f"📊 Top feature importance plot saved to {TOP_FEATURES_PATH}")

# ─── TRAIN/TEST SPLIT ─────────────────────────
print("🔀 Splitting train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# ─── MODELS ───────────────────────────────────
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
    'Voting Classifier': VotingClassifier(estimators=[
        ('lr', LogisticRegression(max_iter=1000, n_jobs=-1)),
        ('rf', RandomForestClassifier(n_estimators=100, n_jobs=-1)),
        ('gb', GradientBoostingClassifier())
    ], voting='soft')
}

results = []

# ─── TRAIN AND EVALUATE ───────────────────────
print("🧠 Training and evaluating models...")
for name, model in tqdm(models.items(), desc="Models"):
    print(f"\n🚀 Training {name}...")
    start = time.time()
    model.fit(X_train, y_train)
    duration = time.time() - start
    print(f"⏱️  {name} training time: {duration:.2f} seconds")

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"📊 {name} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1-score: {f1:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    cm_path = os.path.join(CM_FOLDER, f"{name.replace(' ', '_')}_cm.png")
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", xticklabels=["Safe", "Phishing"], yticklabels=["Safe", "Phishing"])
    plt.title(f"Confusion Matrix: {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    print(f"🖼️  Saved confusion matrix to {cm_path}")

    model_path = os.path.join("models", f"{name.replace(' ', '_')}_email.pkl")
    joblib.dump(model, model_path)
    print(f"💾 Saved model to {model_path}")

    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-score': f1,
        'Train Time (s)': duration
    })

# ─── SAVE RESULTS ─────────────────────────────
results_df = pd.DataFrame(results)
results_df.to_csv(RESULTS_CSV_PATH, index=False)
print(f"\n💾 Results saved to {RESULTS_CSV_PATH}")
print("\n📊 Summary of results:")
print(results_df)

# ─── PLOT METRICS ─────────────────────────────
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-score']
results_df.set_index('Model')[metrics_to_plot].plot(kind='bar', figsize=(10, 6))
plt.title("Email Classification Model Performance")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.grid(axis='y')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(PLOT_PATH)
plt.show()
print(f"📊 Plot saved to {PLOT_PATH}")
