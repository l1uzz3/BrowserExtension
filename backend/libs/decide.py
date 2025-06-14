import time
import psutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import re
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from tqdm import tqdm
from scipy.sparse import hstack

# ─── CONFIGURATION ─────────────────────────────
LEGIT_PATH = "../datasets/legit.csv"
PHISH_PATH = "../datasets/phis.csv"
CPU_LOG_PATH = "models/cpu_usage_decide.log"
RESULTS_CSV_PATH = "models/classification_results.csv"
PLOT_PATH = "models/classification_metrics_plot.png"
CM_FOLDER = "models/confusion_matrices"
TOP_FEATURES_PATH = "models/top_discriminative_features.png"

# ─── FOLDER SETUP ──────────────────────────────
os.makedirs("models", exist_ok=True)
os.makedirs(CM_FOLDER, exist_ok=True)

# ─── PREPROCESSING FUNCTION ───────────────────
def preprocess_url(url):
    url = url.lower()
    url = re.sub(r"https?://", "", url)
    url = re.sub(r"[^a-z0-9]", " ", url)
    return url

# ─── LOAD DATA ────────────────────────────────
print("\U0001F4C2 Loading legitimate domains...")
legit_df = pd.read_csv(LEGIT_PATH, header=None, names=['url'])
legit_df['label'] = 'legitimate'
print(f"✅ Loaded {len(legit_df)} legitimate samples.")

print("\U0001F4C2 Loading phishing URLs...")
phish_df = pd.read_csv(PHISH_PATH)
phish_df['label'] = 'phishing'
print(f"✅ Loaded {len(phish_df)} phishing samples.")

df = pd.concat([legit_df, phish_df], ignore_index=True)
df.dropna(subset=['url', 'label'], inplace=True)
df.drop_duplicates(subset='url', inplace=True)
df['url'] = df['url'].apply(preprocess_url)
print(f"🧮 Total samples after cleaning: {len(df)}")

# ─── FEATURE ENGINEERING ───────────────────────
df['url_length'] = df['url'].apply(len)
df['dot_count'] = df['url'].apply(lambda x: x.count('.'))
df['has_https'] = df['url'].apply(lambda x: int('https' in x))

# ─── TRAIN/TEST SPLIT ─────────────────────────
print("🔀 Splitting train/test sets...")
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# ─── TF-IDF VECTORIZATION ─────────────────────
print("🔤 Vectorizing URLs (char-level TF-IDF n-grams)...")
vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=10000)
X_train_text = vectorizer.fit_transform(df_train['url'])
X_test_text = vectorizer.transform(df_test['url'])

X_train_extra = df_train[['url_length', 'dot_count', 'has_https']].values
X_test_extra = df_test[['url_length', 'dot_count', 'has_https']].values

X_train = hstack([X_train_text, X_train_extra])
X_test = hstack([X_test_text, X_test_extra])

y_train = df_train['label'].map({'legitimate': 1, 'phishing': 0})
y_test = df_test['label'].map({'legitimate': 1, 'phishing': 0})

# ─── FEATURE IMPORTANCE ───────────────────────
print("🔎 Performing Chi² feature analysis...")
X_full = vectorizer.fit_transform(df['url'])
y_full = df['label'].map({'legitimate': 1, 'phishing': 0})
chi2_scores, _ = chi2(X_full, y_full)
top_indices = np.argsort(chi2_scores)[::-1][:20]
top_features = vectorizer.get_feature_names_out()[top_indices]
top_scores = chi2_scores[top_indices]

plt.figure(figsize=(10, 6))
plt.barh(range(len(top_features)), top_scores[::-1], color='darkblue')
plt.yticks(range(len(top_features)), top_features[::-1])
plt.xlabel("Chi² Score")
plt.title("Top Discriminative Character N-Grams")
plt.tight_layout()
plt.savefig(TOP_FEATURES_PATH)
plt.close()
print(f"📊 Top feature importance plot saved to {TOP_FEATURES_PATH}")

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

# ─── CPU SNAPSHOT: BEFORE TRAINING ─────────────
with open(CPU_LOG_PATH, "w") as f:
    f.write("🧠 CPU Snapshot BEFORE training:\n")
    f.write(f"CPU usage: {psutil.cpu_percent(interval=1)}%\n")
    f.write(f"CPU cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} logical\n")
    f.write(f"Memory usage: {psutil.virtual_memory().percent}%\n")

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
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Phishing", "Legit"], yticklabels=["Phishing", "Legit"])
    plt.title(f"Confusion Matrix: {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    print(f"🖼️  Saved confusion matrix to {cm_path}")

    model_path = os.path.join("models", f"{name.replace(' ', '_')}.pkl")
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

# ─── CPU SNAPSHOT: AFTER TRAINING ─────────────
time.sleep(2)
with open(CPU_LOG_PATH, "a") as f:
    f.write("\n🧠 CPU Snapshot AFTER training:\n")
    f.write(f"CPU usage: {psutil.cpu_percent(interval=1)}%\n")
    f.write(f"Memory usage: {psutil.virtual_memory().percent}%\n")

# ─── SAVE RESULTS ─────────────────────────────
results_df = pd.DataFrame(results)
results_df.to_csv(RESULTS_CSV_PATH, index=False)
print(f"\n💾 Results saved to {RESULTS_CSV_PATH}")
print("\n📊 Summary of results:")
print(results_df)

# ─── PLOT METRICS ─────────────────────────────
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-score']
results_df.set_index('Model')[metrics_to_plot].plot(kind='bar', figsize=(10, 6))
plt.title("Model Performance Metrics")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.grid(axis='y')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(PLOT_PATH)
plt.show()
print(f"📊 Plot saved to {PLOT_PATH}")
