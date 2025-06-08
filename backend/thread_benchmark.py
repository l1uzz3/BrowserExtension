import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from libs.FeaturesExtract import featureExtraction

df = pd.read_csv("datasets/BenchmarkSubset.csv")
df['type'] = df['type'].astype(str).str.capitalize()

def safe_extract(idx_url_type):
    idx, url, type = idx_url_type
    try:
        feats = featureExtraction(url)
        feats['type'] = type
        return feats
    except Exception:
        return None

thread_counts = [16, 32, 48, 64]
tasks = [(i, row['url'], row['type']) for i, row in df.iterrows()]
timings = {}

for max_workers in thread_counts:
    results = []
    start = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(safe_extract, t) for t in tasks]
        for fut in as_completed(futures):
            r = fut.result()
            if r is not None:
                results.append(r)
    end = time.time()
    total_time = round(end - start, 2)
    timings[max_workers] = total_time
    print(f"[{max_workers} threads] Time taken: {total_time} seconds")

print("\n📊 Benchmark Summary:")
for workers, t in sorted(timings.items()):
    print(f"  Threads: {workers:<2} → Time: {t} sec")
print("\n✅ Feature extraction completed successfully.")