import os
import time
import psutil
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pycaret.classification import load_model, predict_model
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score


# ─── CONFIG ─────────────────────────────────────────
FEATURES_CSV = "models/extracted_features_full.csv"
OUTPUT_DIR = "models/predictions"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SAMPLE_SIZE = 20000
SEED = 123


MODEL_VARIANTS = {
    "default": "models/xgb_gpu_final",
    "baseline": "models/xgb_gpu_final_baseline",
    "spw_3.31": "models/xgb_gpu_final_spw_3.31",
    "spw_3.31_dropped": "models/xgb_gpu_final_spw_3.31_dropped"
}

BATCH_SIZE = 5000
ROC_PLOT_PATH = os.path.join(OUTPUT_DIR, "roc_comparison_updated.png")

# # ─── LOAD FULL FEATURE DATA ────────────────────────
# print(f"📂 Loading full dataset from: {FEATURES_CSV}")
# df_full = pd.read_csv(FEATURES_CSV)
# print(f"✅ Loaded shape: {df_full.shape}")

# ─── LOAD AND SAMPLE ─────────────────────────────────────
print(f"📂 Loading full extracted dataset from: {FEATURES_CSV}")
df_full = pd.read_csv(FEATURES_CSV)
df_sample = df_full.sample(SAMPLE_SIZE, random_state=SEED)
print(f"✅ Sampled {SAMPLE_SIZE} rows for benchmarking.\n")
# ---------------------------------------------------------------

# ─── PREDICT FUNCTION (BATCHED) ─────────────────────
def batch_predict(model, data, batch_size=5000):
    results = []
    for i in tqdm(range(0, len(data), batch_size), desc="🔮 Predicting batches"):
        batch = data.iloc[i:i+batch_size]
        pred = predict_model(model, data=batch)
        results.append(pred)
    return pd.concat(results, ignore_index=True)

# ─── PERFORMANCE MONITORING ────────────────────────
def log_stats(start, label):
    end = time.time()
    mem = psutil.virtual_memory().percent
    cpu = psutil.cpu_percent(interval=1)
    print(f"🕒 {label} - Time: {end - start:.2f}s | CPU: {cpu}% | Mem: {mem}%")
    return end - start, cpu, mem

# ─── MAIN INFERENCE LOOP ───────────────────────────
metrics = {}
total_start = time.time()

for name, path in MODEL_VARIANTS.items():
    print(f"\n🚀 Running inference for model: {name}")
    model = load_model(path)
    X = df_sample.drop(columns=['type']).copy()
    y_true = df_sample['type'].map({'Phishing': 1, 'Legitimate': 0})
    
    # Apply dropped columns only for specific variants
    if name == "spw_3.31_dropped":
        drop_cols = ["Web_Forwards", "Mouse_Over"]
        X = X.drop(columns=drop_cols, errors='ignore')
        print(f"🧹 Dropped columns for {name}: {drop_cols}")

    try:
        model_features = model.feature_names_in_
        X = X[model_features]
    except Exception as e:
        if 'type' in str(e):
            pass  # ignore intentional type column drop
        else:
            print(f"⚠️ Feature mismatch: {e}")

    start = time.time()
    preds = predict_model(model, data=X)
    elapsed, cpu, mem = log_stats(start, f"Inference {name}")

    preds['type'] = df_sample['type']
    output_path = os.path.join(OUTPUT_DIR, f"predictions_{name}.csv")
    preds.to_csv(output_path, index=False)
    print(f"✅ Saved predictions to {output_path}")

    # ─── ROC AUC ─────────────────────
    if 'Score' in preds.columns:
        y_score = preds['Score']
    else:
        try:
            y_score = model.predict_proba(X)[:, 1]
        except Exception:
            print(f"⚠️ Cannot compute probabilities for {name}. Skipping ROC.")
            continue

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    metrics[name] = {"fpr": fpr, "tpr": tpr, "auc": roc_auc, "time": elapsed, "cpu": cpu, "mem": mem}



# ─── PLOT COMPARISON ROC ───────────────────────────
plt.figure(figsize=(10, 8))
for name, m in metrics.items():
    plt.plot(m['fpr'], m['tpr'], label=f"{name} (AUC = {m['auc']:.4f})")

plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig(ROC_PLOT_PATH)
print(f"\n📈 ROC comparison plot saved to: {ROC_PLOT_PATH}")

# ─── SAVE METRICS TO CSV (DETAILED) ───────────────────────────
detailed_metrics = []
for name, m in metrics.items():
    preds_path = os.path.join(OUTPUT_DIR, f"predictions_{name}.csv")
    df_preds = pd.read_csv(preds_path)
    if "type" in df_preds.columns and "prediction_label" in df_preds.columns:
        y_true = df_preds["type"].map({'Phishing': 1, 'Legitimate': 0})
        y_pred = df_preds['prediction_label'].map({'Phishing': 1, 'Legitimate': 0})

        recall = recall_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        detailed_metrics.append({
            "Model": name,
            "AUC": round(m["auc"], 6),
            "Recall": round(recall, 4),
            "Precision": round(precision, 4),
            "F1": round(f1, 4),
            "Time_sec": round(m["time"], 2),
            "CPU_%": m["cpu"],
            "Memory_%": m["mem"]
        })

# Ensure output folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save final table
metrics_df = pd.DataFrame(detailed_metrics)
metrics_csv_path = os.path.join(OUTPUT_DIR, "inference_metrics_summary.csv")
metrics_df.to_csv(metrics_csv_path, index=False)
print(f"\n📄 Updated metrics summary saved to: {metrics_csv_path}")



# ─── TOTAL TIME ─────────────────────────────────────
total_elapsed = time.time() - total_start
print(f"\n🧮 Total script time: {total_elapsed:.2f} seconds.")


