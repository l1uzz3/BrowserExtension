import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from FeaturesExtract import FeatureExtraction

# --- Input files ---
LEGIT_PATH = '../datasets/legit.csv'
PHISH_PATH = "../datasets/phis.csv"
OUT_PATH = "../datasets/phishing_features.csv"

# --- Feature names (order must match FeatureExtraction.getFeaturesList()) ---
feature_names = [
    "UsingIp", "LongUrl", "ShortUrl", "Symbol@", "Redirecting//",
    "PrefixSuffix", "SubDomains", "HTTPS", "DomainRegLen", "Favicon",
    "NonStdPort", "HTTPSDomainURL", "RequestURL", "AnchorURL", "LinksInScriptTags",
    "ServerFormHandler", "InfoEmail", "AbnormalURL", "WebsiteForwarding",
    "StatusBarCust", "DisableRightClick", "UsingPopupWindow", "IframeRedirection",
    "AgeofDomain", "DNSRecording", "WebsiteTraffic", "PageRank", "GoogleIndex",
    "LinksPointingToPage", "StatsReport"
]

def extract_features(idx_url_label):
    idx, url, label = idx_url_label
    feats = FeatureExtraction(url).getFeaturesList()
    return [idx, *feats, label]

rows = []

# --- Collect legit.csv (domains only, label as legitimate) ---
legit_urls = []
with open(LEGIT_PATH, newline='', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        url = line.strip()
        if not url or url == "url":
            continue
        legit_urls.append((idx, url, "legitimate"))

# --- Collect phis.csv (header 'url', label as phishing) ---
phish_urls = []
with open(PHISH_PATH, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for idx, row in enumerate(reader, start=len(legit_urls)):
        url = row.get("url") or row.get("URL") or row.get("Url")
        if not url:
            continue
        phish_urls.append((idx, url, "phishing"))

all_urls = legit_urls + phish_urls

print(f"Starting feature extraction for {len(all_urls)} URLs using 30 threads...")

# --- Parallel feature extraction ---
with ThreadPoolExecutor(max_workers=30) as executor:
    futures = [executor.submit(extract_features, item) for item in all_urls]
    for i, future in enumerate(as_completed(futures), 1):
        rows.append(future.result())
        if i % 100 == 0 or i == len(all_urls):
            print(f"Processed {i}/{len(all_urls)} URLs")

# --- Sort rows by index to keep order ---
rows.sort(key=lambda x: x[0])

# --- Write output ---
with open(OUT_PATH, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Index'] + feature_names + ['label'])
    writer.writerows(rows)

print("Feature extraction and CSV writing completed.")