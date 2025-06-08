import os
import time
import subprocess
import pandas as pd
from pycaret.classification import setup, create_model, tune_model, pull, save_model
from tqdm import tqdm

# ─── CONFIG ────────────────────────────────────────────────
SEED = 123
N_ITER = 80
OPTIMIZE_METRIC = 'AUC'
FEATURES_CSV = "models/extracted_features_full.csv"
VARIANT_CONFIGS = [
    {
        "name": "baseline",
        "scale_pos_weight": None,
        "drop_columns": [],
    },
    {
        "name": "spw_3.31",
        "scale_pos_weight": 3.31,
        "drop_columns": [],
    },
    {
        "name": "spw_3.31_dropped",
        "scale_pos_weight": 3.31,
        "drop_columns": ["Web_Forwards", "Mouse_Over"],
    }
]

# ─── LOAD FEATURES ─────────────────────────────────────────
print(f"📂 Loading extracted features from: {FEATURES_CSV}")
df_base = pd.read_csv(FEATURES_CSV)
print(f"✅ Loaded shape: {df_base.shape}")

# ─── VARIANT TRAINING LOOP ────────────────────────────────
results_summary = []

for config in tqdm(VARIANT_CONFIGS, desc="🔁 Running model variants", unit="variant"):
    name = config['name']
    spw = config['scale_pos_weight']
    drop_cols = config['drop_columns']

    print(f"\n🚀 Running variant: {name}")
    df = df_base.copy()
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True, errors='ignore')
        print(f"🧹 Dropped columns: {drop_cols}")

    tuned_metrics_csv = f"models/xgb_gpu_tuned_full_{name}.csv"
    gpu_log = f"models/gpu_usage_full_{name}.log"
    model_name = f"models/xgb_gpu_final_{name}"

    # GPU log: before
    with open(gpu_log, "w") as f:
        f.write("📸 GPU Snapshot BEFORE tuning:\n")
        subprocess.run(["nvidia-smi"], stdout=f, stderr=subprocess.STDOUT)

    clf = setup(
        data=df,
        target='type',
        session_id=SEED,
        use_gpu=True,
        verbose=False,
        fold=5,
    )

    start_time = time.time()
    xgb = create_model("xgboost", verbose=False, scale_pos_weight=spw if spw is not None else 1.0)

    tuned = tune_model(
        xgb,
        optimize=OPTIMIZE_METRIC,
        n_iter=N_ITER,
        search_library='optuna',
        early_stopping=True
    )
    elapsed = time.time() - start_time
    print(f"⏱️ Tuning completed in {elapsed:.2f} seconds.")

    # GPU log: after
    with open(gpu_log, "a") as f:
        f.write("\n📸 GPU Snapshot AFTER tuning:\n")
        subprocess.run(["nvidia-smi"], stdout=f, stderr=subprocess.STDOUT)

    tunes = pull()
    tunes.to_csv(tuned_metrics_csv, index=False)
    save_model(tuned, model_name)
    print(f"✅ Saved model and metrics for {name} variant")

    # Extract key scores
    means = tunes.mean(numeric_only=True)
    results_summary.append({
        "Variant": name,
        "Accuracy": means.get("Accuracy", 0),
        "AUC": means.get("AUC", 0),
        "Recall": means.get("Recall", 0),
        "Precision": means.get("Prec.", 0),
        "F1": means.get("F1", 0),
    })

# ─── COMPARISON ───────────────────────────────────────────
print("\n📊 Summary Comparison:")
df_summary = pd.DataFrame(results_summary)
print(df_summary.sort_values(by='AUC', ascending=False).to_string(index=False))
df_summary.to_csv("models/summary_comparison_variants.csv", index=False)
print("✅ Summary saved to models/summary_comparison_variants.csv")
