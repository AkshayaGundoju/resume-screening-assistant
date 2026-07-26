from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def build_pipeline(classifier):
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                sublinear_tf=True,
                stop_words="english",
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.9,
            ),
        ),
        ("clf", classifier),
    ])



from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np

candidates = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        C=1.0,
        random_state=42,
    ),

    "Multinomial Naive Bayes": MultinomialNB(
        alpha=0.1
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    ),
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

results = {}

for name, clf in candidates.items():

    pipe = build_pipeline(clf)

    scores = cross_val_score(
        pipe,
        X_train,
        y_train,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
    )

    results[name] = scores

    print(
        f"{name:26s} macro-F1 = {scores.mean():.4f} (+/- {scores.std():.4f})"
    )

    