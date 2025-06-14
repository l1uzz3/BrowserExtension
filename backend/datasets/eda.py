import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.parse import urlparse

# Load legit.csv (domains only)
legit = pd.read_csv("legit.csv", header=None, names=["url"])
legit["label"] = "legitimate"
# Convert domains to URLs for uniformity
legit["url"] = "http://" + legit["url"]

# Load phis.csv (has header 'url')
phis = pd.read_csv("phis.csv")
phis["label"] = "phishing"

# Combine datasets
df = pd.concat([legit, phis], ignore_index=True)

# Basic info
print("First 5 rows:\n", df.head())
print("\nInfo:")
print(df.info())
print("\nMissing values:\n", df.isnull().sum())
print("\nClass balance:\n", df['label'].value_counts())

# Extract domain for domain-level analysis
df['domain'] = df['url'].apply(lambda x: urlparse(x).netloc)

# Top domains
print("\nTop 10 domains:\n", df['domain'].value_counts().head(10))

# URL length
df['url_length'] = df['url'].apply(len)
plt.figure(figsize=(8,4))
sns.histplot(df['url_length'], bins=50, kde=True)
plt.title('URL Length Distribution')
plt.xlabel('URL Length')
plt.ylabel('Count')
plt.show()

# URL length by label
plt.figure(figsize=(8,4))
sns.histplot(data=df, x='url_length', hue='label', bins=50, kde=True, element='step')
plt.title('URL Length by Label')
plt.xlabel('URL Length')
plt.ylabel('Count')
plt.show()

# Domain frequency by label
top_domains = df['domain'].value_counts().head(10).index
plt.figure(figsize=(10,5))
sns.countplot(data=df[df['domain'].isin(top_domains)], y='domain', hue='label')
plt.title('Top 10 Domains by Label')
plt.show()

# Save summary stats
df.describe(include='all').to_csv("eda_summary.csv")
print("\nSummary statistics saved to eda_summary.csv")