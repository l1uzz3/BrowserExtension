import os
import time
import subprocess
import pandas as pd
from libs.FeaturesExtract import featureExtraction
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pycaret.classification import setup, create_model, tune_model, pull

# ─── CONFIGURATION ───────────────────────────────────────────
DATASET_PATH = "datasets/BenchmarkSubset.csv"  # 2k sample dataset
DEFAULT_METRICS_CSV = "models/xgb_gpu_default.csv"
TUNED_METRICS_CSV = "models/xgb_gpu_tuned.csv"
GPU_LOG = "models/gpu_usage.log"
NUM_THREADS = 64
SEED = 123

# ─── LOAD DATASET ────────────────────────────────────────────
print(f"📂 Loading dataset from: {DATASET_PATH}")
df = pd.read_csv(DATASET_PATH)
df['type'] = df['type'].astype(str).str.capitalize()
print(f"✅ Loaded {len(df)} rows. Type distribution:\n{df['type'].value_counts()}\n")

# ─── FEATURE EXTRACTION ──────────────────────────────────────
print(f"⚙️ Extracting features using {NUM_THREADS} threads...")

def safe_extract(idx_url_type):
    idx, url, type = idx_url_type
    try:
        feats = featureExtraction(url)
        feats['type'] = type
        return feats
    except Exception as e:
        print(f"⚠️  Skipped index {idx} ({url}) due to error: {e}")
        return None

tasks = [(i, row['url'], row['type']) for i, row in df.iterrows()]
results = []

with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
    futures = [executor.submit(safe_extract, t) for t in tasks]
    for fut in tqdm(as_completed(futures), total=len(futures), desc="Feature extraction"):
        r = fut.result()
        if r is not None:
            results.append(r)

print(f"✅ Extracted features for {len(results)} URLs.")
df_train = pd.concat(results, ignore_index=True)

# ─── MODEL SETUP ─────────────────────────────────────────────
print("⚙️ Setting up PyCaret with GPU enabled...")
clf = setup(
    data=df_train,
    target='type',
    session_id=SEED,
    use_gpu=True,
    verbose=True
)

# ─── DEFAULT XGBOOST ─────────────────────────────────────────
print("⚙️ Creating default XGBoost model...")
xgb_default = create_model("xgboost", verbose=False)
default_results = pull()
default_results.to_csv(DEFAULT_METRICS_CSV, index=False)
print(f"✅ Saved default metrics to {DEFAULT_METRICS_CSV}\n")

# ─── GPU SNAPSHOT: BEFORE TUNING ─────────────────────────────
with open(GPU_LOG, "w") as f:
    f.write("📸 GPU Snapshot BEFORE tuning:\n")
    subprocess.run(["nvidia-smi"], stdout=f, stderr=subprocess.STDOUT)

# ─── TUNED XGBOOST WITH OPTUNA ───────────────────────────────
print("⚙️ Creating and tuning XGBoost with Optuna (GPU)...")
start_time = time.time()
xgb = create_model("xgboost", verbose=False)
tuned = tune_model(
    xgb,
    optimize='AUC',
    n_iter=20,
    search_library='optuna',
    early_stopping=True,
)
elapsed = time.time() - start_time
print(f"✅ Tuning completed in {elapsed:.2f} seconds.")

# ─── GPU SNAPSHOT: AFTER TUNING ──────────────────────────────
with open(GPU_LOG, "a") as f:
    f.write("\n📸 GPU Snapshot AFTER tuning:\n")
    subprocess.run(["nvidia-smi"], stdout=f, stderr=subprocess.STDOUT)

# ─── RESULTS REPORT ──────────────────────────────────────────
tunes = pull()
tunes.to_csv(TUNED_METRICS_CSV, index=False)
print(f"\n📈 Saved tuned metrics to {TUNED_METRICS_CSV}")

# ─── COMPARE METRICS ─────────────────────────────────────────
print("\n🔍 Comparing Default vs. Tuned XGBoost (based on mean scores):")
default_mean = pd.read_csv(DEFAULT_METRICS_CSV).mean(numeric_only=True)
tuned_mean = pd.read_csv(TUNED_METRICS_CSV).mean(numeric_only=True)

metrics_to_compare = ["Accuracy", "AUC", "F1"]
for metric in metrics_to_compare:
    d = default_mean.get(metric, None)
    t = tuned_mean.get(metric, None)
    if d is not None and t is not None:
        better = "TUNED" if t > d else "DEFAULT"
        print(f"🔹 {metric}: Default = {d:.4f}, Tuned = {t:.4f} → Best: {better}")

print(f"\n✅ GPU usage log saved to {GPU_LOG}")
print("🚀 Benchmarking complete! Review metrics and GPU logs for benchmark.md.")
# ─── END OF SCRIPT ───────────────────────────────────────────