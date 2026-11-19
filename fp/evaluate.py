"""Metrics for an imbalanced problem and decision-threshold selection.

Accuracy alone is misleading here: with 3.4 % failures, always answering
"no failure" is 96.6 % accurate and catches nothing. The main metrics are
recall (share of real failures caught), precision (share of alarms that are
real), F1 and the area under the precision-recall curve (PR-AUC).
"""
from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def metrics(y_true, proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    single_class = len(np.unique(proba)) == 1
    return {
        "threshold": round(float(threshold), 3),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "roc_auc": 0.5 if single_class else float(roc_auc_score(y_true, proba)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def best_f1_threshold(y_true, proba) -> float:
    """Threshold in [0.05, 0.95] that gives the highest F1 on the given data."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.951, 0.01):
        f1 = f1_score(y_true, (proba >= t).astype(int), zero_division=0)
        if f1 > best_f1 + 1e-12:
            best_t, best_f1 = float(t), f1
    return best_t


def tuned_threshold(model, X_train, y_train, seed: int = 42) -> float:
    """Choose the threshold with 5-fold cross-validation on the TRAINING set only,
    so the test set stays untouched until the final evaluation."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = cross_val_predict(clone(model), X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    return best_f1_threshold(y_train, oof)


def recall_per_failure_mode(test_df, pred, modes) -> dict:
    """Of the real failures of each mode in the test set, how many were caught?"""
    out = {}
    failed = test_df["Machine failure"] == 1
    for m in modes:
        mask = failed & (test_df[m] == 1)
        n = int(mask.sum())
        out[m] = {"n": n, "caught": int(np.asarray(pred)[mask.to_numpy()].sum())}
    return out
