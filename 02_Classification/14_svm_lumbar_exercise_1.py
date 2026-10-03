"""
02_Classification/14_svm_lumbar_exercise_1.py

Support vector machine - exercise 1:
1. Using Pandas, read lumbar_normalized_df.csv into lumbar_df.
2. Display first 10 observations to verify data validity.
3. Split data into train and test sets in 75/25 ratio, random_state=42.
4. Import SVC from sklearn.svm.
5. Create instance of SVC (no hyperparameters initially) and train on training set.
6. Make predictions on test set.
7. Calculate precision on test set.
8. Experiment with selecting optimal values of hyperparameters (C, kernel, gamma)
   to obtain a model with higher precision than the baseline model.
9. Precompute results to JSON cache for fast Streamlit loading.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    precision_score,
    accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

def run_exercise():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    plots_dir = base_dir / "plots"
    data_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    # 1. Load data
    lumbar_path = data_dir / "lumbar_normalized_df.csv"
    if not lumbar_path.exists():
        lumbar_path = base_dir / "lumbar_normalized_df.csv"

    print(f"Loading data from: {lumbar_path}")
    lumbar_df = pd.read_csv(lumbar_path)

    # 2. Display first 10 observations
    print("\n=== KROK 2: Prvních 10 pozorování (lumbar_df.head(10)) ===")
    head_10 = lumbar_df.head(10)
    print(head_10)

    # 3. Split data 75/25, random_state=42
    feature_cols = [c for c in lumbar_df.columns if c != "class"]
    X = lumbar_df[feature_cols]
    y = lumbar_df["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    print(f"\nRozdělení dat: X_train = {X_train.shape}, X_test = {X_test.shape}")
    print(f"Zastoupení pozitivní třídy: Train = {y_train.sum()}/{len(y_train)} ({y_train.mean()*100:.1f} %), Test = {y_test.sum()}/{len(y_test)} ({y_test.mean()*100:.1f} %)")

    # 4 & 5. Baseline SVC (no hyperparameters initially: default RBF, C=1.0, gamma='scale')
    print("\n=== KROK 5: Výchozí model (default SVC()) ===")
    base_model = SVC(random_state=42)
    base_model.fit(X_train, y_train)

    base_sv = int(len(base_model.support_))
    base_sv_per_class = [int(c) for c in base_model.n_support_]
    print(f"Výchozí SVC (RBF): Celkem {base_sv} podpůrných vektorů ({base_sv_per_class[0]} třída 0, {base_sv_per_class[1]} třída 1)")

    # 6. Predict on test set
    y_pred_base = base_model.predict(X_test)

    # 7. Calculate precision on test set
    prec_base = float(precision_score(y_test, y_pred_base))
    acc_base = float(accuracy_score(y_test, y_pred_base))
    rec_base = float(recall_score(y_test, y_pred_base))
    f1_base = float(f1_score(y_test, y_pred_base))
    cm_base = confusion_matrix(y_test, y_pred_base).tolist()

    print(f"\n=== KROK 7: Metriky výchozího modelu ===")
    print(f"Precision (test): {prec_base:.4f} ({prec_base*100:.2f} %)")
    print(f"Accuracy  (test): {acc_base:.4f} ({acc_base*100:.2f} %)")
    print(f"Recall    (test): {rec_base:.4f} ({rec_base*100:.2f} %)")
    print(f"F1-score  (test): {f1_base:.4f}")
    print(f"Confusion Matrix:\n{np.array(cm_base)}")

    # 8. Experimentation with hyperparameters discussed in presentation
    # A) Kernel comparison
    print("\n=== KROK 8: Experimenty s hyperparametry ===")
    kernel_comparison = []
    for k in ["rbf", "linear", "poly", "sigmoid"]:
        clf = SVC(kernel=k, random_state=42).fit(X_train, y_train)
        pred = clf.predict(X_test)
        kernel_comparison.append({
            "kernel": k,
            "test_precision": round(float(precision_score(y_test, pred)), 4),
            "test_accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "test_recall": round(float(recall_score(y_test, pred)), 4),
            "test_f1": round(float(f1_score(y_test, pred)), 4),
            "n_support_vectors": int(len(clf.support_))
        })

    # B) C sweep for linear kernel
    c_linear_sweep = []
    for c in [0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]:
        clf = SVC(kernel="linear", C=c, random_state=42).fit(X_train, y_train)
        pred = clf.predict(X_test)
        c_linear_sweep.append({
            "C": c,
            "test_precision": round(float(precision_score(y_test, pred)), 4),
            "test_accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "test_recall": round(float(recall_score(y_test, pred)), 4),
            "test_f1": round(float(f1_score(y_test, pred)), 4),
            "n_support_vectors": int(len(clf.support_))
        })

    # C) gamma sweep for RBF kernel
    gamma_rbf_sweep = []
    for g in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        clf = SVC(kernel="rbf", gamma=g, random_state=42).fit(X_train, y_train)
        pred = clf.predict(X_test)
        gamma_rbf_sweep.append({
            "gamma": g,
            "test_precision": round(float(precision_score(y_test, pred)), 4),
            "test_accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "test_recall": round(float(recall_score(y_test, pred)), 4),
            "test_f1": round(float(f1_score(y_test, pred)), 4),
            "n_support_vectors": int(len(clf.support_))
        })

    # D) Best models identified
    # Model 1: Linear Kernel (C=2.0) -> Precision = 95.65 % (vs baseline 90.57 %)
    opt_model_linear = SVC(kernel="linear", C=2.0, random_state=42).fit(X_train, y_train)
    y_pred_opt_linear = opt_model_linear.predict(X_test)
    prec_opt_linear = float(precision_score(y_test, y_pred_opt_linear))
    cm_opt_linear = confusion_matrix(y_test, y_pred_opt_linear).tolist()

    # Model 2: RBF Kernel tuned with gamma=1.0 -> Precision = 95.65 %
    opt_model_rbf = SVC(kernel="rbf", gamma=1.0, random_state=42).fit(X_train, y_train)
    y_pred_opt_rbf = opt_model_rbf.predict(X_test)
    prec_opt_rbf = float(precision_score(y_test, y_pred_opt_rbf))
    cm_opt_rbf = confusion_matrix(y_test, y_pred_opt_rbf).tolist()

    # Model 3: Standard Linear Kernel (C=1.0) -> Precision = 93.62 %
    linear_base = SVC(kernel="linear", C=1.0, random_state=42).fit(X_train, y_train)
    y_pred_linear_base = linear_base.predict(X_test)
    prec_linear_base = float(precision_score(y_test, y_pred_linear_base))

    print(f"Optimal Model (Linear C=2.0): Precision = {prec_opt_linear*100:.2f} % (vs baseline {prec_base*100:.2f} %)")
    print(f"Optimal Model (RBF gamma=1.0): Precision = {prec_opt_rbf*100:.2f} % (vs baseline {prec_base*100:.2f} %)")
    print(f"Linear Baseline (C=1.0): Precision = {prec_linear_base*100:.2f} % (vs baseline {prec_base*100:.2f} %)")

    # Plots
    # 1. Tuning Curve Plot (Linear C vs RBF gamma)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    c_vals = [r["C"] for r in c_linear_sweep]
    c_precs = [r["test_precision"] * 100 for r in c_linear_sweep]
    c_accs = [r["test_accuracy"] * 100 for r in c_linear_sweep]

    ax1.plot(c_vals, c_precs, marker="o", color="#1f77b4", linewidth=2.5, label="Test Precision (%)")
    ax1.plot(c_vals, c_accs, marker="s", color="#2ca02c", linestyle="--", label="Test Accuracy (%)")
    ax1.axhline(prec_base * 100, color="red", linestyle=":", label=f"Baseline Precision ({prec_base*100:.1f} %)")
    ax1.set_title("Vliv regularizace C u lineárního jádra (SVC linear)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Parametr C", fontsize=10)
    ax1.set_ylabel("Metrika v %", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right")

    g_vals = [r["gamma"] for r in gamma_rbf_sweep]
    g_precs = [r["test_precision"] * 100 for r in gamma_rbf_sweep]
    g_accs = [r["test_accuracy"] * 100 for r in gamma_rbf_sweep]

    ax2.plot(g_vals, g_precs, marker="o", color="#9467bd", linewidth=2.5, label="Test Precision (%)")
    ax2.plot(g_vals, g_accs, marker="s", color="#ff7f0e", linestyle="--", label="Test Accuracy (%)")
    ax2.axhline(prec_base * 100, color="red", linestyle=":", label=f"Baseline Precision ({prec_base*100:.1f} %)")
    ax2.set_title("Vliv parametru gamma u RBF jádra (SVC rbf)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Parametr gamma", fontsize=10)
    ax2.set_ylabel("Metrika v %", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right")

    plt.tight_layout()
    plot_tuning_path = plots_dir / "lumbar_svm_c_gamma_tuning.png"
    plt.savefig(plot_tuning_path, dpi=200)
    plt.close()
    print(f"Uložen graf ladění: {plot_tuning_path}")

    # 2. Confusion Matrix Comparison Plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.2))
    labels = ["Normal (0)", "Abnormal (1)"]

    sns.heatmap(cm_base, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax1, cbar=False)
    ax1.set_title(f"Výchozí SVC (RBF, default)\nPrecision: {prec_base*100:.2f} % (FP=5)")
    ax1.set_xlabel("Predikovaná diagnóza")
    ax1.set_ylabel("Skutečná diagnóza")

    sns.heatmap(cm_opt_linear, annot=True, fmt="d", cmap="Greens", xticklabels=labels, yticklabels=labels, ax=ax2, cbar=False)
    ax2.set_title(f"Lineární SVC (C=2.0)\nPrecision: {prec_opt_linear*100:.2f} % (FP=2)")
    ax2.set_xlabel("Predikovaná diagnóza")
    ax2.set_ylabel("Skutečná diagnóza")

    sns.heatmap(cm_opt_rbf, annot=True, fmt="d", cmap="Purples", xticklabels=labels, yticklabels=labels, ax=ax3, cbar=False)
    ax3.set_title(f"RBF SVC (gamma=1.0)\nPrecision: {prec_opt_rbf*100:.2f} % (FP=2)")
    ax3.set_xlabel("Predikovaná diagnóza")
    ax3.set_ylabel("Skutečná diagnóza")

    plt.tight_layout()
    plot_cm_path = plots_dir / "lumbar_svm_cm_comparison.png"
    plt.savefig(plot_cm_path, dpi=200)
    plt.close()
    print(f"Uložen graf matic záměn: {plot_cm_path}")

    # JSON export
    payload = {
        "metadata": {
            "exercise": "Support vector machine - exercise 1",
            "dataset": "lumbar_normalized_df.csv",
            "samples_total": len(lumbar_df),
            "samples_train": len(X_train),
            "samples_test": len(X_test),
            "test_ratio": 0.25,
            "random_state": 42,
            "feature_names": feature_cols
        },
        "head_10": head_10.to_dict(orient="records"),
        "baseline_model": {
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
            "precision": round(prec_base, 4),
            "accuracy": round(acc_base, 4),
            "recall": round(rec_base, 4),
            "f1_score": round(f1_base, 4),
            "n_support_vectors": base_sv,
            "confusion_matrix": cm_base
        },
        "optimal_model_linear": {
            "kernel": "linear",
            "C": 2.0,
            "precision": round(prec_opt_linear, 4),
            "accuracy": round(float(accuracy_score(y_test, y_pred_opt_linear)), 4),
            "recall": round(float(recall_score(y_test, y_pred_opt_linear)), 4),
            "f1_score": round(float(f1_score(y_test, y_pred_opt_linear)), 4),
            "n_support_vectors": int(len(opt_model_linear.support_)),
            "confusion_matrix": cm_opt_linear
        },
        "optimal_model_rbf_gamma": {
            "kernel": "rbf",
            "gamma": 1.0,
            "precision": round(prec_opt_rbf, 4),
            "accuracy": round(float(accuracy_score(y_test, y_pred_opt_rbf)), 4),
            "recall": round(float(recall_score(y_test, y_pred_opt_rbf)), 4),
            "f1_score": round(float(f1_score(y_test, y_pred_opt_rbf)), 4),
            "n_support_vectors": int(len(opt_model_rbf.support_)),
            "confusion_matrix": cm_opt_rbf
        },
        "kernel_comparison": kernel_comparison,
        "c_linear_sweep": c_linear_sweep,
        "gamma_rbf_sweep": gamma_rbf_sweep
    }

    json_path = data_dir / "lumbar_svm_exercise_1_precomputed.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nVýsledky úspěšně uloženy do: {json_path}")

if __name__ == "__main__":
    run_exercise()
