"""Model quality checks. These fail if a change makes the models clearly worse."""
import numpy as np
import pytest

from fp import data
from fp.evaluate import best_f1_threshold, metrics
from fp.models import build_models


@pytest.fixture(scope="module")
def pr_auc():
    """Test-set PR-AUC of every model (fitted once for all tests)."""
    df = data.load()
    train, test = data.split(df, seed=42)
    out = {}
    for name, (model, physics) in build_models(42).items():
        X_tr, y_tr = data.make_xy(train, physics)
        X_te, y_te = data.make_xy(test, physics)
        model.fit(X_tr, y_tr)
        out[name] = metrics(y_te, model.predict_proba(X_te)[:, 1])["pr_auc"]
    return out


def test_accuracy_trap():
    """Always predicting 'no failure' is ~96.6 % accurate but catches nothing."""
    y = np.array([0] * 966 + [1] * 34)
    m = metrics(y, np.zeros(len(y)))
    assert m["accuracy"] == pytest.approx(0.966)
    assert m["recall"] == 0.0 and m["f1"] == 0.0


def test_best_model_quality(pr_auc):
    assert pr_auc["Random forest (+ physics features)"] >= 0.80


def test_every_model_beats_the_baseline(pr_auc):
    base = pr_auc["Always 'no failure'"]
    for name, v in pr_auc.items():
        if name != "Always 'no failure'":
            assert v > 5 * base, f"{name}: PR-AUC {v:.3f}"


def test_physics_features_help(pr_auc):
    assert pr_auc["Random forest (+ physics features)"] > pr_auc["Random forest (raw sensors)"] + 0.05
    assert pr_auc["Logistic regression (+ physics features)"] > pr_auc["Logistic regression (raw sensors)"]


def test_threshold_selection_maximises_f1():
    y = np.array([0, 0, 0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.65, 0.7, 0.9])
    t = best_f1_threshold(y, proba)
    assert 0.65 < t <= 0.70                     # separates perfectly: F1 = 1
    assert metrics(y, proba, t)["f1"] == 1.0


def test_confusion_counts_add_up():
    y = np.array([0, 1, 1, 0, 1])
    m = metrics(y, np.array([0.9, 0.8, 0.1, 0.2, 0.7]), 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 1)
