"""
02_Classification/08_logistic_regression_penguins_exercise_2.py

Logistic regression - exercise 2:
1. Using Pandas, load penguins_df_normalized.csv into penguins_df.
2. Display first 10 observations to verify validity.
3. Split data into train/test 70/30, random_state=42.
4. Import LogisticRegression from sklearn.linear_model.
5. Create instance with multi_class and class_weight='balanced'. Fit on X_train, y_train.
6. Make predictions on test set.
7. Calculate precision on test set.
8. Experiment to choose optimal hyperparameters (C, solver, scaling) for better precision.
9. Precompute results to JSON cache for fast Streamlit loading.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score,
    accuracy_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

def run_exercise():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    plots_dir = base_dir / "plots"
    data_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    # 1. Load data
    penguins_path = data_dir / "penguins_df_normalized.csv"
    if not penguins_path.exists():
        penguins_path = base_dir / "penguins_df_normalized.csv"
        
    print(f"Loading data from: {penguins_path}")
    penguins_df = pd.read_csv(penguins_path)

    # 2. Display first 10 observations
    print("\n=== KROK 2: Prvních 10 pozorování (penguins_df.head(10)) ===")
    head_10 = penguins_df.head(10)
    print(head_10)

    # 3. Split data 70/30, random_state=42
    feature_cols = [c for c in penguins_df.columns if c != "species"]
    X = penguins_df[feature_cols]
    y = penguins_df["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    print(f"\nRozdělení dat: X_train = {X_train.shape}, X_test = {X_test.shape}")
    print(f"Počet tříd: {y.nunique()} ({list(y.unique())})")

    # 4 & 5. Baseline model with class_weight='balanced'
    print("\n=== KROK 5: Výchozí model (class_weight='balanced', C=1.0) ===")
    # V Scikit-learn 1.8+ je multinomiální link výchozím nastavením
    base_model = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
    base_model.fit(X_train, y_train)

    # 6. Predict on test set
    y_pred_base = base_model.predict(X_test)

    # 7. Calculate precision on test set
    classes = sorted(list(y.unique()))
    prec_weighted_base = float(precision_score(y_test, y_pred_base, average="weighted", zero_division=0))
    prec_macro_base = float(precision_score(y_test, y_pred_base, average="macro", zero_division=0))
    acc_base = float(accuracy_score(y_test, y_pred_base))
    rec_weighted_base = float(recall_score(y_test, y_pred_base, average="weighted", zero_division=0))
    f1_weighted_base = float(f1_score(y_test, y_pred_base, average="weighted", zero_division=0))
    cm_base = confusion_matrix(y_test, y_pred_base, labels=classes).tolist()

    # Precision po jednotlivých druzích
    prec_per_class_base = precision_score(y_test, y_pred_base, average=None, labels=classes, zero_division=0)
    prec_class_dict_base = {cls: round(float(p), 4) for cls, p in zip(classes, prec_per_class_base)}

    print(f"Výchozí Weighted Precision : {prec_weighted_base*100:.2f} %")
    print(f"Výchozí Macro Precision    : {prec_macro_base*100:.2f} %")
    print(f"Výchozí Accuracy           : {acc_base*100:.2f} %")
    print(f"Precision per class: {prec_class_dict_base}")
    print(f"Matice záměn 3x3:\n{np.array(cm_base)}")

    # 8. Experiment with hyperparameters (C sweep & scaling)
    print("\n=== KROK 8: Experimentování s hyperparametry ===")
    c_candidates = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0]
    sweep_results = []

    best_c = 1.0
    best_c_prec = prec_weighted_base

    for c in c_candidates:
        m = LogisticRegression(C=c, class_weight="balanced", random_state=42, max_iter=5000)
        m.fit(X_train, y_train)
        yp = m.predict(X_test)

        pw = float(precision_score(y_test, yp, average="weighted", zero_division=0))
        pm = float(precision_score(y_test, yp, average="macro", zero_division=0))
        acc = float(accuracy_score(y_test, yp))
        rec = float(recall_score(y_test, yp, average="weighted", zero_division=0))
        f1 = float(f1_score(y_test, yp, average="weighted", zero_division=0))
        cm = confusion_matrix(y_test, yp, labels=classes).tolist()

        sweep_results.append({
            "C": c,
            "precision_weighted": round(pw, 4),
            "precision_macro": round(pm, 4),
            "accuracy": round(acc, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": cm
        })
        print(f"C = {c:8.1f} | Weighted Prec: {pw*100:.2f} % | Macro Prec: {pm*100:.2f} % | Accuracy: {acc*100:.2f} %")

        if pw > best_c_prec:
            best_c_prec = pw
            best_c = c

    print(f"\n>> Nejlepší hodnota parametru C: C = {best_c} (Precision: {best_c_prec*100:.2f} % vs {prec_weighted_base*100:.2f} % u baseline)")

    # Model s nejlepším C
    opt_model_c = LogisticRegression(C=best_c, class_weight="balanced", random_state=42, max_iter=5000)
    opt_model_c.fit(X_train, y_train)
    y_pred_opt_c = opt_model_c.predict(X_test)
    cm_opt_c = confusion_matrix(y_test, y_pred_opt_c, labels=classes).tolist()

    # Expertní varianta: Pipeline se StandardScalerem
    print("\n--- Expertní varianta: Pipeline se StandardScalerem ---")
    pipe_scaler = Pipeline([
        ("scaler", StandardScaler()),
        ("log_reg", LogisticRegression(C=1.0, class_weight="balanced", random_state=42, max_iter=1000))
    ])
    pipe_scaler.fit(X_train, y_train)
    y_pred_scaler = pipe_scaler.predict(X_test)
    scaler_pw = float(precision_score(y_test, y_pred_scaler, average="weighted", zero_division=0))
    scaler_pm = float(precision_score(y_test, y_pred_scaler, average="macro", zero_division=0))
    scaler_acc = float(accuracy_score(y_test, y_pred_scaler))
    scaler_rec = float(recall_score(y_test, y_pred_scaler, average="weighted", zero_division=0))
    scaler_f1 = float(f1_score(y_test, y_pred_scaler, average="weighted", zero_division=0))
    cm_scaler = confusion_matrix(y_test, y_pred_scaler, labels=classes).tolist()
    print(f"StandardScaler Pipeline -> Weighted Prec: {scaler_pw*100:.2f} %, Accuracy: {scaler_acc*100:.2f} %")

    # One-vs-Rest varianta
    ovr_model = OneVsRestClassifier(LogisticRegression(C=best_c, class_weight="balanced", random_state=42, max_iter=5000))
    ovr_model.fit(X_train, y_train)
    y_pred_ovr = ovr_model.predict(X_test)
    ovr_pw = float(precision_score(y_test, y_pred_ovr, average="weighted", zero_division=0))
    ovr_acc = float(accuracy_score(y_test, y_pred_ovr))
    cm_ovr = confusion_matrix(y_test, y_pred_ovr, labels=classes).tolist()

    # Uložení do JSON cache
    cache_payload = {
        "metadata": {
            "exercise": "Logistic regression - exercise 2",
            "dataset": "penguins_df_normalized.csv",
            "classes": classes,
            "samples_total": len(penguins_df),
            "samples_train": len(X_train),
            "samples_test": len(X_test),
            "test_size": 0.3,
            "random_state": 42
        },
        "head_10": head_10.to_dict(orient="records"),
        "features": feature_cols,
        "baseline_model": {
            "C": 1.0,
            "class_weight": "balanced",
            "precision_weighted": round(prec_weighted_base, 4),
            "precision_macro": round(prec_macro_base, 4),
            "accuracy": round(acc_base, 4),
            "recall": round(rec_weighted_base, 4),
            "f1_score": round(f1_weighted_base, 4),
            "precision_per_class": prec_class_dict_base,
            "confusion_matrix": cm_base
        },
        "optimal_c_model": {
            "C": best_c,
            "class_weight": "balanced",
            "precision_weighted": round(best_c_prec, 4),
            "precision_macro": round(float(precision_score(y_test, y_pred_opt_c, average="macro", zero_division=0)), 4),
            "accuracy": round(float(accuracy_score(y_test, y_pred_opt_c)), 4),
            "recall": round(float(recall_score(y_test, y_pred_opt_c, average="weighted", zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, y_pred_opt_c, average="weighted", zero_division=0)), 4),
            "confusion_matrix": cm_opt_c
        },
        "standard_scaler_model": {
            "C": 1.0,
            "class_weight": "balanced",
            "precision_weighted": round(scaler_pw, 4),
            "precision_macro": round(scaler_pm, 4),
            "accuracy": round(scaler_acc, 4),
            "recall": round(scaler_rec, 4),
            "f1_score": round(scaler_f1, 4),
            "confusion_matrix": cm_scaler
        },
        "ovr_model": {
            "C": best_c,
            "precision_weighted": round(ovr_pw, 4),
            "accuracy": round(ovr_acc, 4),
            "confusion_matrix": cm_ovr
        },
        "c_sweep": sweep_results
    }

    out_json = data_dir / "penguins_logistic_exercise_2_precomputed.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(cache_payload, f, indent=2, ensure_ascii=False)
    print(f"\nUloženo do: {out_json}")

    # Diagnostický graf: Sweep C a Matice záměn
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    c_vals = [r["C"] for r in sweep_results]
    pw_vals = [r["precision_weighted"] for r in sweep_results]
    acc_vals = [r["accuracy"] for r in sweep_results]

    axes[0].plot(c_vals, pw_vals, marker="o", linewidth=2.5, color="#1f77b4", label="Weighted Precision")
    axes[0].plot(c_vals, acc_vals, marker="s", linewidth=2, color="#2ca02c", linestyle="--", label="Accuracy")
    axes[0].axvline(1.0, color="gray", linestyle="--", alpha=0.7, label="Výchozí C = 1.0 (59.6 %)")
    axes[0].axvline(best_c, color="red", linestyle=":", linewidth=2, label=f"Optimální C = {best_c} ({best_c_prec*100:.1f} %)")
    axes[0].set_xscale("log")
    axes[0].set_title("Vliv parametru C na Precision (Tučňáci)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Parametr C (log scale)")
    axes[0].set_ylabel("Hodnota metriky")
    axes[0].set_ylim(0.50, 0.98)
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    # 3x3 Matice záměn pro nejlepší model
    sns.heatmap(cm_opt_c, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[1],
                xticklabels=classes, yticklabels=classes)
    axes[1].set_title(f"Matice záměn optimálního modelu (C={best_c})\\nWeighted Precision: {best_c_prec*100:.1f} %", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Predikovaný druh")
    axes[1].set_ylabel("Skutečný druh")

    plt.tight_layout()
    plot_file = plots_dir / "penguins_logistic_c_tuning.png"
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"Graf uložen do: {plot_file}")

if __name__ == "__main__":
    run_exercise()
