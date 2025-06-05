#!/usr/bin/env python3
"""
train_model.py  (place this inside backend/libs/)

This version skips PCA fitting entirely (no new pca_model.pkl is created).
It assumes pca_model.pkl already exists at …/backend/model/pca_model.pkl.
It processes only the first 10,000 URLs from the CSV (for a larger batch run),
then runs PyCaret with GPU disabled (to suppress cuML warnings).

To run:
    (.venv) C:\…\BrowserExtension\backend\libs> python train_model.py
"""

import os
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─── IMPORT low‐level and high‐level extractors ──────────────────────────────────────
import ExtractFunc as lowlev               # only needed if featureExtraction calls lowlev internally
from FeaturesExtract import featureExtraction

# ─── IMPORT PyCaret classification tools ─────────────────────────────────────────────
from pycaret.classification import setup, compare_models, save_model

# ─── DETERMINE RELEVANT PATHS ─────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))                   # …/backend/libs
BACKEND_DIR       = os.path.normpath(os.path.join(HERE, os.pardir)) # …/backend

# 1) CSV of URLs+labels lives in …/backend/datasets/dataset.csv
COMBINED_CSV = os.path.join(BACKEND_DIR, 'datasets', 'dataset.csv')

# 2) Where the existing PCA lives (we will only check for it, never overwrite)
PCA_DIR      = os.path.join(BACKEND_DIR, 'model')
PCA_FILENAME = 'pca_model.pkl'
PCA_PATH     = os.path.join(PCA_DIR, PCA_FILENAME)

# 3) Where to save final classifier (inside …/backend/models/)
CLASSIFIER_DIR  = os.path.join(BACKEND_DIR, 'models')
CLASSIFIER_NAME = 'phishing_model'
CLASSIFIER_PATH = os.path.join(CLASSIFIER_DIR, CLASSIFIER_NAME)

# Ensure output folder for classifier exists
os.makedirs(CLASSIFIER_DIR, exist_ok=True)


# ─── 1) LOAD CSV (URLs + labels) ───────────────────────────────────────────────────────
print("⏳ Loading combined CSV of URLs + labels…")
if not os.path.exists(COMBINED_CSV):
    raise FileNotFoundError(f"Could not find '{COMBINED_CSV}' (looked in backend/datasets).")
df = pd.read_csv(COMBINED_CSV)

assert 'url' in df.columns and 'label' in df.columns, (
    f"CSV must have 'url' and 'label' columns; found: {df.columns.tolist()}"
)

# Normalize label capitalization (e.g. "phishing" → "Phishing")
df['label'] = df['label'].astype(str).str.capitalize()

print(f"✅ Loaded {len(df)} total URLs. Label distribution:\n{df['label'].value_counts()}\n")

# ─── SLICE TO FIRST 10,000 ROWS ─────────────────────────────────────────────────────────
df = df.head(10000)
print(f"🔍 Slicing down to the first {len(df)} URLs for this run.\n")
print(df['label'].value_counts(), "\n")

# ─── 2) VERIFY EXISTING PCA (NO FITTING HERE) ──────────────────────────────────────────
if not os.path.exists(PCA_PATH):
    raise FileNotFoundError(
        f"PCA file not found at '{PCA_PATH}'. "
        "Please ensure you have already generated pca_model.pkl before running this script."
    )
print(f"🔍 Found existing PCA at '{PCA_PATH}'. Skipping PCA fitting.\n")


# ─── 3) SECOND PASS: EXTRACT FULL 10 FEATURES + LABEL FOR EACH URL ──────────────────
print("🔍 Second pass: building full 10‐feature vectors (via featureExtraction)…")

def extract_features_for_row(idx_url_label):
    idx, url, lbl = idx_url_label
    try:
        feats = featureExtraction(url)
        feats['label'] = lbl
        return feats
    except Exception as e:
        print(f"⚠️  Skipped URL at index {idx} ({url}) due to error: {e}")
        return None

# Build a list of (index, url, label) for the first 10,000 rows
tasks = [(idx, row['url'], row['label']) for idx, row in df.iterrows()]
feature_rows = []
skipped_count = 0

# Run up to 20 threads in parallel to extract features
with ThreadPoolExecutor(max_workers=20) as executor:
    futures = {executor.submit(extract_features_for_row, tpl): tpl[0] for tpl in tasks}
    for fut in tqdm(as_completed(futures), total=len(futures), desc="Feature pass"):
        result = fut.result()
        if result is not None:
            feature_rows.append(result)
        else:
            skipped_count += 1

print(f"\n✅ Built feature‐rows for {len(feature_rows)} URLs. Skipped {skipped_count}.\n")

# Combine all returned dicts into a single DataFrame
df_features = pd.concat(feature_rows, ignore_index=True)
print(f"✅ Combined features shape: {df_features.shape}")
print(df_features['label'].value_counts(), "\n")


# ─── 4) TRAIN A CLASSIFIER WITH PYCARET (GPU DISABLED) ───────────────────────────────
print("⚙️ Starting PyCaret setup on full 10 features (GPU disabled)…")
clf = setup(
    data=df_features,
    target='label',
    session_id=123,
    verbose=False,   # suppress most console output
    use_gpu=False    # ensure PyCaret does not try to import cuML
)

print("🏋️ Training & comparing models (this may take several minutes)…")
best_model = compare_models()

save_model(best_model, CLASSIFIER_PATH)
print(f"✅ Saved classifier as '{CLASSIFIER_PATH}.pkl'\n")

print("🎉 Training complete! Classifier is ready (no new PCA was created).")
