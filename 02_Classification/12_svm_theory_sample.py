"""
02_Classification/12_svm_theory_sample.py

Precomputes diagnostic datasets, kernel comparisons, support vector analyses,
and the 3D blanket analogy for the Support Vector Machine (SVM) theory module.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_circles, make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def run_svm_theory_sample():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    plots_dir = base_dir / "plots"
    data_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    print("Generuji syntetická nelineární data (soustředné kruhy pro analogii s dekou)...")
    # 1. Non-linear dataset (make_circles)
    X_circ, y_circ = make_circles(n_samples=300, factor=0.4, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_circ, y_circ, test_size=0.3, random_state=42)

    # 2. Kernel comparison
    kernels = ["linear", "poly", "rbf", "sigmoid"]
    kernel_results = {}
    models = {}

    for k in kernels:
        kwargs = {"kernel": k, "random_state": 42}
        if k == "poly":
            kwargs["degree"] = 2
        clf = SVC(**kwargs)
        clf.fit(X_train, y_train)
        models[k] = clf

        y_pred = clf.predict(X_test)
        n_sv = int(len(clf.support_))
        sv_per_class = [int(c) for c in clf.n_support_]

        kernel_results[k] = {
            "kernel": k,
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred)), 4),
            "recall": round(float(recall_score(y_test, y_pred)), 4),
            "f1_score": round(float(f1_score(y_test, y_pred)), 4),
            "total_support_vectors": n_sv,
            "support_vectors_per_class": sv_per_class,
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
        }
        print(f"Kernel {k:8s}: Accuracy = {kernel_results[k]['accuracy']*100:.2f} %, SVs = {n_sv}")

    # Plot 1: Kernel comparison 2D decision boundaries
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    x_min, x_max = X_circ[:, 0].min() - 0.3, X_circ[:, 0].max() + 0.3
    y_min, y_max = X_circ[:, 1].min() - 0.3, X_circ[:, 1].max() + 0.3
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

    for ax, k in zip(axes, kernels):
        clf = models[k]
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
        # Plot data points
        scatter = ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="coolwarm", edgecolors="k", s=30)
        # Highlight support vectors
        sv = clf.support_vectors_
        ax.scatter(sv[:, 0], sv[:, 1], s=80, facecolors="none", edgecolors="black", linewidths=1.2, label=f"SV ({len(sv)})")

        ax.set_title(f"Jádro: {k.upper()}\nAcc: {kernel_results[k]['accuracy']*100:.1f} % | SV: {len(sv)}", fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    plot_kernel_path = plots_dir / "svm_kernel_comparison.png"
    plt.savefig(plot_kernel_path, dpi=200)
    plt.close()
    print(f"Uložen graf srovnání jader: {plot_kernel_path}")

    # 3. Parameter Sensitivity: C and Gamma Grid
    c_values = [0.1, 1.0, 10.0, 100.0]
    gamma_values = [0.01, 0.1, 1.0, 10.0]
    grid_results = []

    for c in c_values:
        for g in gamma_values:
            clf = SVC(kernel="rbf", C=c, gamma=g, random_state=42)
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            grid_results.append({
                "C": c,
                "gamma": g,
                "accuracy": round(float(accuracy_score(y_test, pred)), 4),
                "precision": round(float(precision_score(y_test, pred)), 4),
                "n_support_vectors": int(len(clf.support_))
            })

    # Plot 2: C vs Gamma Decision Boundary 4x4
    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    for i, c in enumerate(c_values):
        for j, g in enumerate(gamma_values):
            ax = axes[i, j]
            clf = SVC(kernel="rbf", C=c, gamma=g, random_state=42).fit(X_train, y_train)
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="coolwarm", s=15, alpha=0.6)
            if i == 0:
                ax.set_title(f"gamma = {g}", fontsize=11, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"C = {c}", fontsize=11, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle("Vliv hyperparametrů C a gamma na tvar dělící hranice RBF jádra", fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    plot_grid_path = plots_dir / "svm_c_gamma_grid.png"
    plt.savefig(plot_grid_path, dpi=200)
    plt.close()
    print(f"Uložen graf mřížky C a gamma: {plot_grid_path}")

    # 4. Blanket Analogy: 3D Mapping Data (x1, x2 -> x1^2 + x2^2)
    # Showing how lifting points to 3D separates circles linearly
    sample_points_3d = []
    for idx in range(min(100, len(X_circ))):
        x1 = float(X_circ[idx, 0])
        x2 = float(X_circ[idx, 1])
        z = float(x1**2 + x2**2)
        sample_points_3d.append({
            "x1": round(x1, 4),
            "x2": round(x2, 4),
            "z": round(z, 4),
            "class": int(y_circ[idx])
        })

    # Save JSON cache
    payload = {
        "metadata": {
            "module": "Support vector machine - theoretical introduction",
            "samples_total": len(X_circ),
            "samples_train": len(X_train),
            "samples_test": len(X_test)
        },
        "kernel_comparison": kernel_results,
        "grid_sensitivity": grid_results,
        "blanket_analogy_3d": sample_points_3d
    }

    json_path = data_dir / "svm_theory_precomputed.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Uložen JSON soubor: {json_path}")

if __name__ == "__main__":
    run_svm_theory_sample()
