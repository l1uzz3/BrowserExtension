import os
import time
import psutil
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.ensemble import (
    HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# ─── CONFIGURATION ─────────────────────────────
DATA_PATH = "../datasets/phishing.csv"
CPU_LOG_PATH = "models/new_cpu_usage_phishing.log"
RESULTS_CSV_PATH = "models/new_phishing_model_results.csv"
PLOT_PATH = "models/new_phishing_model_metrics_plot.png"
CM_FOLDER = "models/new_confusion_matrices"
ROC_FOLDER = "models/new_roc_curves"
MODEL_FOLDER = "models"

# ─── FOLDER SETUP ──────────────────────────────
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(CM_FOLDER, exist_ok=True)
os.makedirs(ROC_FOLDER, exist_ok=True)

# ─── LOAD AND PREPARE DATA ─────────────────────
df = pd.read_csv(DATA_PATH)

if 'Index' in df.columns:
    df = df.drop(columns=['Index'])

X = df.drop(columns=['class'])
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─── MODELS TO TRAIN ────────────────────────────
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
    "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, max_depth=3, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42),
    "Voting Classifier": VotingClassifier(estimators=[
        ('lr', LogisticRegression(max_iter=1000, n_jobs=-1)),
        ('rf', RandomForestClassifier(n_estimators=100, n_jobs=-1))
    ], voting='soft')
}

# ─── CPU SNAPSHOT: BEFORE TRAINING ─────────────
with open(CPU_LOG_PATH, "w") as f:
    f.write("🧠 CPU Snapshot BEFORE training:\n")
    f.write(f"CPU usage: {psutil.cpu_percent(interval=1)}%\n")
    f.write(f"CPU cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} logical\n")
    f.write(f"Memory usage: {psutil.virtual_memory().percent}%\n")

results = []

# ─── TRAIN AND EVALUATE ────────────────────────
print("🚀 Training models on phishing.csv features...")
for name, model in tqdm(models.items(), desc="Models"):
    print(f"\n🔧 Training: {name}")
    start = time.time()
    model.fit(X_train, y_train)
    duration = time.time() - start
    print(f"⏱️  {name} training time: {duration:.2f} sec")

    y_pred = model.predict(X_test)

    # Check if predict_proba or decision_function is available for ROC
    try:
        y_scores = model.predict_proba(X_test)[:, 1]
    except AttributeError:
        try:
            y_scores = model.decision_function(X_test)
        except AttributeError:
            y_scores = y_pred  # fallback (not ideal for ROC, but avoids crash)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=-1)
    rec = recall_score(y_test, y_pred, pos_label=-1)
    f1 = f1_score(y_test, y_pred, pos_label=-1)
    roc_auc = roc_auc_score(y_test, y_scores)

    print(f"✅ {name}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, ROC AUC={roc_auc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_path = os.path.join(CM_FOLDER, f"{name.replace(' ', '_')}_cm.png")
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Phishing (-1)", "Legit (1)"], yticklabels=["Phishing (-1)", "Legit (1)"])
    plt.title(f"Confusion Matrix: {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()

    # ROC Curve
    try:
        fpr, tpr, _ = roc_curve(y_test, y_scores, pos_label=1)
        plt.figure()
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve: {name}")
        plt.legend()
        plt.tight_layout()
        roc_path = os.path.join(ROC_FOLDER, f"{name.replace(' ', '_')}_roc.png")
        plt.savefig(roc_path)
        plt.close()
        print(f"📈 ROC curve saved to {roc_path}")
    except Exception as e:
        print(f"⚠️ Could not plot ROC for {name}: {e}")

    # Save model
    model_path = os.path.join(MODEL_FOLDER, f"{name.replace(' ', '_')}.pkl")
    joblib.dump(model, model_path)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-score": f1,
        "ROC AUC": roc_auc,
        "Train Time (s)": duration
    })

# ─── CPU SNAPSHOT: AFTER TRAINING ──────────────
time.sleep(1)
with open(CPU_LOG_PATH, "a") as f:
    f.write("\n🧠 CPU Snapshot AFTER training:\n")
    f.write(f"CPU usage: {psutil.cpu_percent(interval=1)}%\n")
    f.write(f"Memory usage: {psutil.virtual_memory().percent}%\n")

# ─── SAVE RESULTS & PLOT ───────────────────────
results_df = pd.DataFrame(results)
results_df.to_csv(RESULTS_CSV_PATH, index=False)

print("\n📊 Summary of results:")
print(results_df)

# Overall metrics plot
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC AUC']
results_df.set_index('Model')[metrics_to_plot].plot(kind='bar', figsize=(12, 6))
plt.title("Performance Comparison on Phishing.csv")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.grid(axis='y')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(PLOT_PATH)
plt.show()

print(f"\n✅ All plots and results saved to 'models/' folder.")
