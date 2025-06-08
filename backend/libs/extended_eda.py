import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ─── CONFIGURATION ───────────────────────────────────────────────
CSV_PATH = "backend/models/extracted_features_full.csv"
OUTPUT_DIR = "eda_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── LOAD DATA ───────────────────────────────────────────────────
print(f"📂 Loading extracted features from: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
print(f"✅ Loaded shape: {df.shape}")
print(df['type'].value_counts())

# ─── CLASS DISTRIBUTION ──────────────────────────────────────────
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='type')
plt.title("Class Distribution")
plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"))
plt.close()
print("\nClass Distribution Counts:")
print(df['type'].value_counts())

# ─── FEATURE DISTRIBUTIONS ───────────────────────────────────────
features = [col for col in df.columns if col != "type"]
for feature in features:
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x=feature, hue='type', kde=True, element="step", common_norm=False)
    plt.title(f"Distribution of {feature} by Class")
    safe_feature = feature.replace("/", "_")  # Replace / with _
    plt.savefig(os.path.join(OUTPUT_DIR, f"{safe_feature}_distribution.png"))
    plt.close()
    print(f"\nSummary statistics for {feature}:")
    print(df.groupby('type')[feature].describe())

# ─── CORRELATION MATRIX ──────────────────────────────────────────
corr = df[features].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_matrix.png"))
plt.close()
print("\nCorrelation Matrix:")
print(corr)
# ─── FEATURE IMPORTANCE (XGBoost) ────────────────────────────────
print("⚙️ Training temporary XGBoost to extract feature importances…")
le = LabelEncoder()
y = le.fit_transform(df["type"])
X = df[features]

X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

model = XGBClassifier(tree_method="gpu_hist", enable_categorical=False, use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)

importances = model.feature_importances_
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x="Importance", y="Feature", palette="viridis")
plt.title("XGBoost Feature Importances")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "xgb_feature_importance.png"))
plt.close()

# ─── DONE ────────────────────────────────────────────────────────
print("📊 EDA complete! All visualizations saved in the 'eda_outputs/' folder.")
