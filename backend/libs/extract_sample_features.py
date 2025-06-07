import sys
import os
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


# Add current directory to path so we can import from sibling modules
sys.path.append(os.path.dirname(__file__))

from backend.libs.FeaturesExtract import featureExtraction

# Paths
dataset_path = os.path.join(os.path.dirname(__file__), "..", "datasets", "sampled_URL_dataset.csv")
output_path = os.path.join(os.path.dirname(__file__), "..", "datasets", "sampled_features.csv")

# Load sampled URLs
sampled_df = pd.read_csv(dataset_path)

# Progress bar
tqdm.pandas()

# Run feature extraction
def safe_extract(url, label):
    try:
        features = featureExtraction(url)
        if isinstance(features, dict):
            features["label"] = label
            return features
    except Exception as e:
        print(f"[ERROR] {url} -> {e}")
    return None

print("\n⚙️ Extracting features in parallel...")
results = []
with ThreadPoolExecutor(max_workers=20) as executor:
    futures = {
        executor.submit(safe_extract, row["url"], row["type"]): idx
        for idx, row in sampled_df.iterrows()
    }

    for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting features"):
        result = future.result()
        if result is not None:
            results.append(result)
# Save results
features_df = pd.DataFrame(results)
features_df.to_csv(output_path, index=False)
print(f"\n✅ Features saved to: {output_path}")
