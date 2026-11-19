"""The compared models. Each entry: (estimator, uses physics features)."""
from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_models(seed: int = 42) -> dict:
    def logreg():
        # class_weight="balanced": a missed failure counts ~28x more than a false alarm
        return make_pipeline(StandardScaler(),
                             LogisticRegression(class_weight="balanced", max_iter=5000))

    def forest():
        return RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                      class_weight="balanced_subsample",
                                      n_jobs=-1, random_state=seed)

    return {
        # Baseline: always "no failure". High accuracy, useless in practice.
        "Always 'no failure'": (DummyClassifier(strategy="most_frequent"), False),
        "Logistic regression (raw sensors)": (logreg(), False),
        "Logistic regression (+ physics features)": (logreg(), True),
        "Random forest (raw sensors)": (forest(), False),
        "Random forest (+ physics features)": (forest(), True),
    }
