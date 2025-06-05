#!/usr/bin/env python3
"""
train_model.py  (place this inside backend/libs/)

Runs in two main phases—both parallelized with 20 threads:

1) FIRST PASS: read combined CSV of URLs+labels (one level up),
   fetch each URL concurrently, compute raw DOM flags via ExtractFunc.py,
   fit PCA(n_components=1) on them, and save it.
2) SECOND PASS: for each URL, call featureExtraction(url) concurrently (which will load that PCA),
   build the full 10‐feature vector + label, and accumulate.
3) Use PyCaret to train a classifier on those 10 features + label, and save it.

To run:
    (.venv) C:\…\BrwoserExtension\backend\libs> python train_model.py
"""

import os
import pandas as pd
import httpx
import pickle as pk
from tqdm import tqdm
from sklearn.decomposition import PCA
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─── IMPORT low‐level and high‐level extractors from the same folder ──────────
import ExtractFunc as lowlev
from FeaturesExtract import featureExtraction

# ─── IMPORT PyCaret classification tools ───────────────────────────────────────
from pycaret.classification import setup, compare_models, save_model

# ─── DETERMINE RELEVANT PATHS ───────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))                 # …/backend/libs
BACKEND_DIR       = os.path.normpath(os.path.join(HERE, os.pardir))   # …/backend

# 1) CSV of URLs+labels lives in …/backend/datasets/dataset.csv (adjust if different)
COMBINED_CSV = os.path.join(BACKEND_DIR, 'datasets', 'dataset.csv')

# 2) Where to save PCA and final classifier (both inside …/backend/)
PCA_DIR        = os.path.join(BACKEND_DIR, 'model')
PCA_FILENAME   = 'pca_model.pkl'
CLASSIFIER_DIR = os.path.join(BACKEND_DIR, 'models')
CLASSIFIER_NAME= 'phishing_model'

PCA_PATH        = os.path.join(PCA_DIR, PCA_FILENAME)
CLASSIFIER_PATH = os.path.join(CLASSIFIER_DIR, CLASSIFIER_NAME)

# Ensure output folders exist
os.makedirs(PCA_DIR, exist_ok=True)
os.makedirs(CLASSIFIER_DIR, exist_ok=True)


# ─── 1) LOAD CSV (URLs + labels) ─────────────────────────────────────────────────

print("⏳ Loading combined CSV of URLs + labels…")
if not os.path.exists(COMBINED_CSV):
    raise FileNotFoundError(f"Could not find '{COMBINED_CSV}' (looked in backend/datasets).")
df = pd.read_csv(COMBINED_CSV)

assert 'url' in df.columns and 'label' in df.columns, (
    f"CSV must have 'url' and 'label' columns; found: {df.columns.tolist()}"
)

# Normalize label capitalization (e.g. "phishing" → "Phishing")
df['label'] = df['label'].astype(str).str.capitalize()

print(f"✅ Loaded {len(df)} URLs. Label distribution:\n{df['label'].value_counts()}\n")


# ─── 2) FIRST PASS: COLLECT RAW DOM FLAGS FOR PCA ───────────────────────────────
print("🔍 First pass: fetching each URL & collecting raw DOM flags for PCA…")

def fetch_dom_flags(url: str) -> dict:
    """
    Returns a dict with keys 'iFrame', 'Web_Forwards', 'Mouse_Over' for the given URL.
    """
    try:
        response = httpx.get(url, timeout=5.0)
    except Exception:
        response = ""

    try:
        iframe_flag     = lowlev.iframe(response)
        mouseOver_flag  = lowlev.mouseOver(response)
        forwarding_flag = lowlev.forwarding(response)
    except Exception:
        iframe_flag, mouseOver_flag, forwarding_flag = 1, 1, 1

    return {
        'iFrame':       iframe_flag,
        'Web_Forwards': forwarding_flag,
        'Mouse_Over':   mouseOver_flag
    }

dom_rows = []
with ThreadPoolExecutor(max_workers=20) as executor:
    futures = {executor.submit(fetch_dom_flags, url): idx for idx, url in enumerate(df['url'])}
    for fut in tqdm(as_completed(futures), total=len(futures), desc="DOM pass"):
        dom_rows.append(fut.result())

dom_df = pd.DataFrame(dom_rows)
print(f"✅ Collected DOM flags: {dom_df.shape[0]} rows, 3 columns.\n")


# ─── 3) FIT & SAVE PCA(n_components=1) ON THOSE 3 COLUMNS ───────────────────────
print("⚙️ Fitting PCA(n_components=1) on [iFrame, Web_Forwards, Mouse_Over]…")
pca = PCA(n_components=1, random_state=42)
pca.fit(dom_df[['iFrame', 'Web_Forwards', 'Mouse_Over']])

with open(PCA_PATH, 'wb') as f:
    pk.dump(pca, f)
print(f"✅ Saved PCA model to '{PCA_PATH}'\n")


# ─── 4) SECOND PASS: EXTRACT FULL 10 FEATURES + LABEL FOR EACH URL ───────────────
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

# Prepare iterable of (index, url, label)
tasks = [(idx, row['url'], row['label']) for idx, row in df.iterrows()]
feature_rows = []
skipped_count = 0

with ThreadPoolExecutor(max_workers=20) as executor:
    futures = {executor.submit(extract_features_for_row, tpl): tpl[0] for tpl in tasks}
    for fut in tqdm(as_completed(futures), total=len(futures), desc="Feature pass"):
        result = fut.result()
        if result is not None:
            feature_rows.append(result)
        else:
            skipped_count += 1

print(f"\n✅ Built feature‐rows for {len(feature_rows)} URLs. Skipped {skipped_count}.\n")

df_features = pd.concat(feature_rows, ignore_index=True)
print(f"✅ Combined features shape: {df_features.shape}")
print(df_features['label'].value_counts(), "\n")


# ─── 5) TRAIN A CLASSIFIER WITH PYCARET ──────────────────────────────────────────
print("⚙️ Starting PyCaret setup on full 10 features…")
clf = setup(
    data=df_features,
    target='label',
    session_id=123,
    silent=True,
    verbose=False
)

print("🏋️ Training & comparing models (this may take several minutes)…")
best_model = compare_models()

save_model(best_model, CLASSIFIER_PATH)
print(f"✅ Saved classifier as '{CLASSIFIER_PATH}.pkl'\n")

print("🎉 Training complete! PCA + classifier are ready.")
