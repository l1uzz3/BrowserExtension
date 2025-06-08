# save_sample.py
import pandas as pd

df = pd.read_csv("datasets/Sampled_URL_Dataset.csv")
df.sample(n=2000, random_state=42).to_csv("datasets/BenchmarkSubset.csv", index=False)
