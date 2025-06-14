import time
import psutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from sklearn.utils import shuffle
from tqdm import tqdm
import os
import numpy as np

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

# ─── LOAD DATA ────────────────────────────────
print("📂 Loading legitimate domains...")
legit_df = pd.read_csv(LEGIT_PATH, header=None, names=['url'])
legit_df['label'] = 'legitimate'
print(f"✅ Loaded {len(legit_df)} legitimate samples.")

print("📂 Loading phishing URLs...")
phish_df = pd.read_csv(PHISH_PATH)
phish_df['label'] = 'phishing'
print(f"✅ Loaded {len(phish_df)} phishing samples.")

df = pd.concat([legit_df, phish_df], ignore_index=True)
df.dropna(subset=['url', 'label'], inplace=True)
df.drop_duplicates(subset='url', inplace=True)
print(f"🧮 Total samples after cleaning: {len(df)}")

# ─── TRAIN/TEST SPLIT ─────────────────────────
print("🔀 Splitting train/test sets...")
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# ─── VECTORIZATION ────────────────────────────
print("🔤 Vectorizing URLs (word-level n-grams)...")
# Step 1: Full fit for Chi² analysis
temp_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2), token_pattern=r'[A-Za-z0-9]{3,}')
X_full = temp_vectorizer.fit_transform(df['url'])
y_full = df['label'].map({'legitimate': 1, 'phishing': 0})
chi2_scores, _ = chi2(X_full, y_full)
top_indices = np.argsort(chi2_scores)[::-1][:20]
top_features = temp_vectorizer.get_feature_names_out()[top_indices]
top_words = set(top_features)

# Step 2: Filter vocabulary
filtered_words = [word for word in temp_vectorizer.get_feature_names_out() if word not in top_words]
clean_vocab = {word: i for i, word in enumerate(filtered_words)}

# Step 3: Rebuild vectorizer without those top words
vectorizer = CountVectorizer(
    analyzer='word', 
    ngram_range=(1, 2), 
    token_pattern=r'[A-Za-z0-9]{3,}', 
    vocabulary=clean_vocab)
X_train = vectorizer.transform(df_train['url'])
X_test = vectorizer.transform(df_test['url'])

y_train = df_train['label'].map({'legitimate': 1, 'phishing': 0})
y_test = df_test['label'].map({'legitimate': 1, 'phishing': 0})

# ─── FEATURE IMPORTANCE ANALYSIS ──────────────
print("🔎 Performing Chi² feature analysis...")
# ✅ Repeat full fit with a fresh vectorizer for Chi² (no filtered vocab)
chi_vec = CountVectorizer(analyzer='word', ngram_range=(1, 2), token_pattern=r'[A-Za-z0-9]{3,}')
X_full = chi_vec.fit_transform(df['url'])
y_full = df['label'].map({'legitimate': 1, 'phishing': 0})
chi2_scores, _ = chi2(X_full, y_full)
top_indices = np.argsort(chi2_scores)[::-1][:20]
top_features = chi_vec.get_feature_names_out()[top_indices]
top_scores = chi2_scores[top_indices]


plt.figure(figsize=(10, 6))
plt.barh(range(len(top_features)), top_scores[::-1], color='darkblue')
plt.yticks(range(len(top_features)), top_features[::-1])
plt.xlabel("Chi² Score")
plt.title("Top Discriminative Word N-Grams")
plt.tight_layout()
plt.savefig(TOP_FEATURES_PATH)
plt.close()
print(f"📊 Top feature importance plot saved to {TOP_FEATURES_PATH}")

# ─── DEFINE MODELS ────────────────────────────
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
}

# ─── CPU SNAPSHOT: BEFORE TRAINING ─────────────
with open(CPU_LOG_PATH, "w") as f:
    f.write("🧠 CPU Snapshot BEFORE training:\n")
    f.write(f"CPU usage: {psutil.cpu_percent(interval=1)}%\n")
    f.write(f"CPU cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} logical\n")
    f.write(f"Memory usage: {psutil.virtual_memory().percent}%\n")

results = []

# ─── TRAIN AND EVALUATE ───────────────────────
print("🧠 Training and evaluating models sequentially (with internal parallelism)...")
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

    # ─── CONFUSION MATRIX ─────────────────────
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

    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-score': f1,
        'Train Time (s)': duration
    })

# ─── SHORT DELAY TO STABILIZE CPU ──────────────
time.sleep(2)

# ─── CPU SNAPSHOT: AFTER TRAINING ──────────────
with open(CPU_LOG_PATH, "a") as f:
    f.write("\n🧠 CPU Snapshot AFTER training:\n")
    f.write(f"CPU usage: {psutil.cpu_percent(interval=1)}%\n")
    f.write(f"Memory usage: {psutil.virtual_memory().percent}%\n")

# ─── SAVE & PRINT RESULTS ──────────────────────
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
