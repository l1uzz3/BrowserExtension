import pandas as pd

# 1) Load each CSV, renaming its “type” column to a common name, e.g. “label”

# Phishing_URLs.csv has header:    url,Type
# Values in that “Type” column are “Phishing”
phish_df = pd.read_csv('Phishing_URLs.csv')  
# Rename “Type” → “label”
phish_df = phish_df.rename(columns={'Type': 'label'})

# URL_dataset.csv has header:    url,type
# Values in that “type” column are “legitimate”
good_df = pd.read_csv('URL_dataset.csv')
# Rename “type” → “label”
good_df = good_df.rename(columns={'type': 'label'})


# 2) Check that each DataFrame indeed has columns ["url", "label"]
print("Phishing DF columns:", phish_df.columns.tolist())
print("Legitimate DF columns:", good_df.columns.tolist())


# 3) Take the first 100 000 rows from each
phish_sample = phish_df.iloc[:100000].copy()
good_sample  = good_df.iloc[:100000].copy()

# 4) (Optional) If you want to normalize the label values, you can do it here.
#    For example, if you’d rather have “Phishing” vs. “Legitimate” (capitalized),
#    you could overwrite the existing labels:
phish_sample['label'] = 'Phishing'
good_sample['label']  = 'Legitimate'


# 5) Concatenate them
combined = pd.concat([good_sample, phish_sample], ignore_index=True)

print("Combined shape before shuffle:", combined.shape)
print(combined['label'].value_counts())


# 6) (Optional) Shuffle them so that “Phishing” / “Legitimate” aren’t block‐grouped
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

# 7) Save to disk
combined.to_csv('dataset.csv', index=False)
print("Wrote out: dataset.csv  (total rows = {})".format(len(combined)))
