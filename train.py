"""Train and evaluate all models, write results/ (metrics + plots).

    python train.py            # full run, about 1 minute on a laptop
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay

from fp import data
from fp.evaluate import metrics, recall_per_failure_mode, tuned_threshold
from fp.models import build_models

SEED = 42
RESULTS = Path(__file__).resolve().parent / "results"


def run(seed: int = SEED, results_dir: Path = RESULTS, plots: bool = True) -> dict:
    df = data.load()
    train_df, test_df = data.split(df, seed=seed)
    results = {"dataset": {
        "rows": len(df), "failures": int(df[data.TARGET].sum()),
        "train_rows": len(train_df), "test_rows": len(test_df),
        "test_failures": int(test_df[data.TARGET].sum()),
    }, "models": {}}
    fitted = {}

    for name, (model, physics) in build_models(seed).items():
        X_tr, y_tr = data.make_xy(train_df, physics)
        X_te, y_te = data.make_xy(test_df, physics)
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]
        thr = tuned_threshold(model, X_tr, y_tr, seed) if name != "Always 'no failure'" else 0.5
        results["models"][name] = {
            "physics_features": physics,
            "default_threshold": metrics(y_te, proba, 0.5),
            "tuned_threshold": metrics(y_te, proba, thr),
        }
        fitted[name] = (model, X_te, y_te, proba, thr)
        m = results["models"][name]["tuned_threshold"]
        print(f"{name:42s} thr={m['threshold']:.2f}  acc={m['accuracy']:.3f}  prec={m['precision']:.3f}  "
              f"rec={m['recall']:.3f}  f1={m['f1']:.3f}  pr_auc={m['pr_auc']:.3f}")

    best = max((n for n in results["models"] if n != "Always 'no failure'"),
               key=lambda n: results["models"][n]["tuned_threshold"]["pr_auc"])
    model, X_te, y_te, proba, thr = fitted[best]
    pred = (proba >= thr).astype(int)
    results["best_model"] = best
    results["best_model_recall_per_failure_mode"] = recall_per_failure_mode(test_df, pred, data.FAILURE_MODES)

    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    (results_dir / "metrics.md").write_text(markdown_table(results))

    if plots:
        plot_confusion(fitted, results_dir / "confusion_matrices.png")
        plot_pr(fitted, results_dir / "pr_curves.png")
        imp = permutation_importance(model, X_te, y_te, scoring="average_precision",
                                     n_repeats=10, random_state=seed, n_jobs=-1)
        plot_importance(X_te.columns, imp, best, results_dir / "feature_importance.png")
    return results


def markdown_table(r: dict) -> str:
    rows = ["| Model | Threshold | Accuracy | Precision | Recall | F1 | PR-AUC | Missed failures (FN) | False alarms (FP) |",
            "|---|---|---|---|---|---|---|---|---|"]
    for name, m in r["models"].items():
        t = m["tuned_threshold"]
        rows.append(f"| {name} | {t['threshold']:.2f} | {t['accuracy']:.3f} | {t['precision']:.3f} | "
                    f"{t['recall']:.3f} | {t['f1']:.3f} | {t['pr_auc']:.3f} | {t['fn']} | {t['fp']} |")
    lines = ["\n".join(rows), "", f"Best model (PR-AUC): **{r['best_model']}**", "",
             "| Failure mode | Failures in test set | Caught |", "|---|---|---|"]
    for mode, v in r["best_model_recall_per_failure_mode"].items():
        lines.append(f"| {mode} | {v['n']} | {v['caught']} |")
    return "\n".join(lines) + "\n"


def plot_confusion(fitted, path):
    names = [n for n in fitted if n != "Always 'no failure'"]
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4))
    for ax, n in zip(axes, names):
        _, _, y, proba, thr = fitted[n]
        ConfusionMatrixDisplay.from_predictions(y, (proba >= thr).astype(int), ax=ax, colorbar=False,
                                                display_labels=["OK", "Failure"], cmap="Blues")
        ax.set_title(n.replace(" (", "\n("), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_pr(fitted, path):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for n, (_, _, y, proba, _) in fitted.items():
        if n == "Always 'no failure'":
            continue
        PrecisionRecallDisplay.from_predictions(y, proba, ax=ax, name=n)
    base = float(np.mean(fitted["Always 'no failure'"][2]))
    ax.axhline(base, ls="--", c="grey", lw=1, label=f"chance level ({base:.3f})")
    ax.set_title("Precision-recall curves (test set)")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_importance(columns, imp, name, path):
    order = np.argsort(imp.importances_mean)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.barh(np.array(columns)[order], imp.importances_mean[order], xerr=imp.importances_std[order], color="#0f766e")
    ax.set_xlabel("drop in PR-AUC when the feature is shuffled")
    ax.set_title(f"Permutation importance: {name}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    run()
