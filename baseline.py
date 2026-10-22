import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
from data import load

df = load()
train, test = train_test_split(df, test_size=0.1, stratify=df["category"], random_state=42)

clf = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), min_df=2),
    LogisticRegression(max_iter=1000, class_weight="balanced"))
clf.fit(train.text, train.category)

print("Macro-F1 (English held-out):", f1_score(test.category, clf.predict(test.text), average="macro"))
joblib.dump(clf, "baseline.joblib")
test.to_csv("bitext_test.csv", index=False)