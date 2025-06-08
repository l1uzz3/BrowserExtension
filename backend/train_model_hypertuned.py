import os
import time
import subprocess
import pandas as pd
from pycaret.classification import setup, create_model, tune_model, pull, save_model

# ─── CONFIGURATION ───
FEATURES_CSV = "models/extracted_features_full.csv"
TUNED_METRICS_CSV = "models/xgb_gpu_tuned_full_hypertuned.csv"
GPU_LOG = "models/gpu_usage_full_hypertuned.log"
MODEL_NAME = "models/xgb_gpu_final"
SEED = 123

# ─── LOAD EXTRACTED FEATURES ───
print(f"📂 Loading extracted features from: {FEATURES_CSV}")
df_train = pd.read_csv(FEATURES_CSV)
print(f"✅ Loaded {len(df_train)} feature rows.")

# ─── MODEL SETUP ───
print("⚙️ Setting up PyCaret with GPU enabled...")
clf = setup(
    data=df_train,
    target='type',
    session_id=SEED,
    use_gpu=True,
    verbose=True
)

# ─── GPU SNAPSHOT: BEFORE TUNING ───
with open(GPU_LOG, "w") as f:
    f.write("📸 GPU Snapshot BEFORE tuning:\n")
    subprocess.run(["nvidia-smi"], stdout=f, stderr=subprocess.STDOUT)

# ─── TUNED XGBOOST WITH OPTUNA ───
print("⚙️ Hypertuning XGBoost with Optuna (GPU)...")
start_time = time.time()
xgb = create_model("xgboost", verbose=False)
tuned = tune_model(
    xgb,
    optimize='AUC',
    n_iter=40,
    search_library='optuna',
    early_stopping=True,
)
elapsed = time.time() - start_time
print(f"✅ Tuning completed in {elapsed:.2f} seconds.")

# ─── GPU SNAPSHOT: AFTER TUNING ───
with open(GPU_LOG, "a") as f:
    f.write("\n📸 GPU Snapshot AFTER tuning:\n")
    subprocess.run(["nvidia-smi"], stdout=f, stderr=subprocess.STDOUT)

# ─── SAVE TUNED METRICS ───
tunes = pull()
tunes.to_csv(TUNED_METRICS_CSV, index=False)
print(f"\n📊 Saved hypertuned metrics to {TUNED_METRICS_CSV}")

# ─── SAVE MODEL ───
save_model(tuned, MODEL_NAME)
print(f"✅ Saved hypertuned model to {MODEL_NAME}.pkl")

print(f"✅ GPU usage log saved to {GPU_LOG}")
print("🚀 Hypertuned training complete! Ready for evaluation or deployment.")
