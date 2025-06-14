import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

# Paths to your CSV files
LEGIT_PATH = "../datasets/legit.csv"
PHISH_PATH = "../datasets/phis.csv"

print("Loading legitimate domains...")
legit_df = pd.read_csv(LEGIT_PATH, header=None, names=['url'])
legit_df['label'] = 'legitimate'
print(f"Loaded {len(legit_df)} legitimate samples.")

print("Loading phishing URLs...")
phish_df = pd.read_csv(PHISH_PATH)
phish_df['label'] = 'phishing'
print(f"Loaded {len(phish_df)} phishing samples.")

df = pd.concat([legit_df, phish_df], ignore_index=True)
print(f"Total samples: {len(df)}")

print("Vectorizing URLs...")
vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(3,5))
X = vectorizer.fit_transform(df['url'])
y = df['label'].map({'legitimate': 1, 'phishing': 0})

print("Splitting train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}

results = []

print("Training and evaluating models:")
for name, model in tqdm(models.items(), desc="Models"):
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    print(f"Predicting with {name}...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"{name} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1-score: {f1:.4f}")
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-score': f1
    })

results_df = pd.DataFrame(results)
print("\nSummary of results:")
print(results_df)
