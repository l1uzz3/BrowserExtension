import os
import time
import pandas as pd
from pycaret.classification import setup, create_model, tune_model, pull
from datetime import datetime

# ─── CONFIGURATION ───
FEATURES_CSV = "backend/models/extracted_features_full.csv"
SAMPLE_SIZE = 20000
SEED = 123
SCALE_POS_WEIGHTS = [1.0, 2.0, 3.31, 5.0, 7.33, 10.0, 15.0]
N_ITER = 80
BENCHMARK_CSV = "backend/models/scale_pos_weight_benchmark_results.csv"

# ─── LOAD AND SAMPLE ───
print(f"📂 Loading full extracted dataset from: {FEATURES_CSV}")
df_full = pd.read_csv(FEATURES_CSV)
df_sample = df_full.sample(SAMPLE_SIZE, random_state=SEED)
print(f"✅ Sampled {SAMPLE_SIZE} rows for benchmarking.\n")

# ─── BENCHMARKING LOOP ───
records = []
for spw in SCALE_POS_WEIGHTS:
    print(f"⚙️ Benchmarking XGBoost with scale_pos_weight={spw}...")
    setup(data=df_sample, target='type', session_id=SEED, use_gpu=True, verbose=False)
    model = create_model('xgboost', verbose=False)
    model.set_params(scale_pos_weight=spw)

    start_time = time.time()
    tuned_model = tune_model(model, n_iter=N_ITER, search_library='optuna', optimize='AUC', early_stopping=True)
    duration = time.time() - start_time

    metrics = pull().mean(numeric_only=True)
    record = {
        'scale_pos_weight': spw,
        'AUC': metrics.get('AUC', None),
        'Accuracy': metrics.get('Accuracy', None),
        'F1': metrics.get('F1', None),
        'Recall': metrics.get('Recall', None),
        'Prec.': metrics.get('Prec.', None),
        'Time_sec': round(duration, 2),
        'Timestamp': datetime.now().isoformat()
    }
    print(f"✅ Completed in {duration:.2f} seconds: AUC={record['AUC']:.4f}, F1={record['F1']:.4f}\n")
    records.append(record)

# ─── SAVE RESULTS ───
df_benchmark = pd.DataFrame(records)
df_benchmark.to_csv(BENCHMARK_CSV, index=False)
print(f"Benchmarking completed successfully!")
print("📊 Summary of results:")
print(df_benchmark)
print(f"📊 Benchmark results saved to {BENCHMARK_CSV}")

