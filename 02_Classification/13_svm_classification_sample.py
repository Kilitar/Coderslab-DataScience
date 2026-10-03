"""
02_Classification/13_svm_classification_sample.py

Support vector machine - sample implementation:
Replicates and extends the official course slides:
1. Synthetic dataset creation: make_classification(n_samples=600, n_features=5, n_classes=2, random_state=42).
2. Split into train and test sets in 70/30 ratio (random_state=42).
3. Train default SVC() (RBF kernel, C=1.0, gamma='scale').
4. Evaluate accuracy and precision (92.2% and 92.4%).
5. Train and compare different kernels: 'rbf', 'linear', 'poly', 'sigmoid'.
6. Plot decision regions on key features.
7. Hyperparameter tuning (GridSearchCV over C and gamma).
8. Precompute results to JSON cache for fast Streamlit loading.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

def run_svm_sample():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    plots_dir = base_dir / "plots"
    data_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    print("=== 1. Vytvoření syntetického datasetu (make_classification) ===")
    X, y = make_classification(
        n_samples=600,
        n_features=5,
        n_classes=2,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    print(f"Rozdělení: X_train = {X_train.shape}, X_test = {X_test.shape}")

    # 2. Výchozí model ze slajdů (SVC bez hyperparametrů)
    print("\n=== 2. Výchozí model SVC() (RBF jádro, default) ===")
    clf_svc = SVC(random_state=42)
    clf_svc.fit(X_train, y_train)

    y_pred_base = clf_svc.predict(X_test)
    acc_base = float(accuracy_score(y_test, y_pred_base))
    prec_base = float(precision_score(y_test, y_pred_base))
    rec_base = float(recall_score(y_test, y_pred_base))
    f1_base = float(f1_score(y_test, y_pred_base))
    cm_base = confusion_matrix(y_test, y_pred_base).tolist()

    print(f"Accuracy score (default RBF):  {acc_base:.4f} ({acc_base*100:.1f} %)")
    print(f"Precision score (default RBF): {prec_base:.4f} ({prec_base*100:.1f} %)")
    print(f"Počet podpůrných vektorů:     {len(clf_svc.support_)}")

    # 3. Srovnání jader ze slajdů kurzu (rbf, linear, poly, sigmoid)
    print("\n=== 3. Srovnání jader ze slajdů kurzu ===")
    kernels = ["rbf", "linear", "poly", "sigmoid"]
    kernel_results = {}
    models = {}

    for k in kernels:
        m = SVC(kernel=k, random_state=42)
        m.fit(X_train, y_train)
        models[k] = m
        pred = m.predict(X_test)

        acc = float(accuracy_score(y_test, pred))
        prec = float(precision_score(y_test, pred))
        rec = float(recall_score(y_test, pred))
        f1 = float(f1_score(y_test, pred))
        cm = confusion_matrix(y_test, pred).tolist()

        kernel_results[k] = {
            "kernel": k,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "n_support_vectors": int(len(m.support_)),
            "support_vectors_per_class": [int(c) for c in m.n_support_],
            "confusion_matrix": cm
        }
        print(f"Kernel: {k:7s} | Accuracy: {acc*100:.1f} % | Precision: {prec*100:.1f} % | SVs: {len(m.support_)}")

    # 4. Hyperparameter tuning (GridSearchCV over C and gamma for RBF and poly)
    print("\n=== 4. Optimalizace hyperparametrů (GridSearchCV) ===")
    param_grid = [
        {
            "kernel": ["rbf"],
            "C": [0.1, 1.0, 5.0, 10.0, 50.0],
            "gamma": ["scale", "auto", 0.01, 0.1, 1.0]
        },
        {
            "kernel": ["poly"],
            "C": [0.1, 1.0, 5.0, 10.0],
            "degree": [2, 3, 4],
            "coef0": [0.0, 1.0]
        },
        {
            "kernel": ["linear"],
            "C": [0.01, 0.1, 1.0, 10.0]
        }
    ]

    grid = GridSearchCV(SVC(random_state=42), param_grid, cv=5, scoring="precision")
    grid.fit(X_train, y_train)
    best_clf = grid.best_estimator_
    best_pred = best_clf.predict(X_test)

    best_acc = float(accuracy_score(y_test, best_pred))
    best_prec = float(precision_score(y_test, best_pred))
    best_rec = float(recall_score(y_test, best_pred))
    best_f1 = float(f1_score(y_test, best_pred))
    best_cm = confusion_matrix(y_test, best_pred).tolist()

    print(f"Nejlepší parametry: {grid.best_params_}")
    print(f"Optimalizovaná Precision: {best_prec*100:.2f} % | Accuracy: {best_acc*100:.2f} %")

    # 5. Vizualizace dělících oblastí na 2 nejdůležitějších příznacích
    # (Trénujeme 2D model pro zobrazení rozhodovací hranice plot_decision_regions)
    X_train_2d = X_train[:, :2]
    X_test_2d = X_test[:, :2]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))

    viz_kernels = ["linear", "rbf", "poly"]
    for ax, k in zip(axes, viz_kernels):
        clf_2d = SVC(kernel=k, random_state=42).fit(X_train_2d, y_train)
        Z = clf_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
        scatter = ax.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test, cmap="coolwarm", edgecolors="k", s=35)
        sv = clf_2d.support_vectors_
        ax.scatter(sv[:, 0], sv[:, 1], s=90, facecolors="none", edgecolors="black", linewidths=1.2, label=f"Podpůrné vektory ({len(sv)})")

        ax.set_title(f"Jádro: {k.upper()} (2D projekce)\nSV: {len(sv)}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Příznak 0 (Feature 0)")
        ax.set_ylabel("Příznak 1 (Feature 1)")
        ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    plot_regions_path = plots_dir / "svm_sample_decision_regions.png"
    plt.savefig(plot_regions_path, dpi=200)
    plt.close()
    print(f"Uložen graf rozhodovacích oblastí: {plot_regions_path}")

    # Plot 2: Metriky jader
    fig, ax = plt.subplots(figsize=(8, 4.5))
    df_metrics = pd.DataFrame(kernel_results).T
    bar_width = 0.35
    index = np.arange(len(kernels))

    ax.bar(index - bar_width/2, df_metrics["accuracy"] * 100, bar_width, label="Accuracy (%)", color="#1f77b4")
    ax.bar(index + bar_width/2, df_metrics["precision"] * 100, bar_width, label="Precision (%)", color="#2ca02c")
    ax.set_xticks(index)
    ax.set_xticklabels([k.upper() for k in kernels], fontsize=11, fontweight="bold")
    ax.set_ylabel("Metrika v %", fontsize=11)
    ax.set_ylim(80, 100)
    ax.set_title("Srovnání jader SVM na školním datasetu (make_classification)", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for i in index:
        ax.text(i - bar_width/2, df_metrics["accuracy"].iloc[i]*100 + 0.4, f"{df_metrics['accuracy'].iloc[i]*100:.1f}%", ha="center", fontsize=9)
        ax.text(i + bar_width/2, df_metrics["precision"].iloc[i]*100 + 0.4, f"{df_metrics['precision'].iloc[i]*100:.1f}%", ha="center", fontsize=9)

    plt.tight_layout()
    plot_bar_path = plots_dir / "svm_sample_kernel_comparison.png"
    plt.savefig(plot_bar_path, dpi=200)
    plt.close()
    print(f"Uložen graf srovnání metrik jader: {plot_bar_path}")

    # 6. JSON Export
    payload = {
        "metadata": {
            "exercise": "Support vector machine - sample implementation",
            "dataset": "make_classification(600, 5, 2)",
            "samples_total": 600,
            "samples_train": len(X_train),
            "samples_test": len(X_test),
            "features_count": 5
        },
        "default_model_rbf": {
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
            "accuracy": round(acc_base, 4),
            "precision": round(prec_base, 4),
            "recall": round(rec_base, 4),
            "f1_score": round(f1_base, 4),
            "n_support_vectors": int(len(clf_svc.support_)),
            "confusion_matrix": cm_base
        },
        "kernel_comparison": kernel_results,
        "grid_search_best": {
            "best_params": grid.best_params_,
            "accuracy": round(best_acc, 4),
            "precision": round(best_prec, 4),
            "recall": round(best_rec, 4),
            "f1_score": round(best_f1, 4),
            "confusion_matrix": best_cm,
            "n_support_vectors": int(len(best_clf.support_))
        }
    }

    json_path = data_dir / "svm_classification_sample_precomputed.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Výsledky úspěšně uloženy do: {json_path}")

if __name__ == "__main__":
    run_svm_sample()
