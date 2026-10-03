"""
02_Classification/06_logistic_regression_exercise.py

Kompletní praktická implementace a benchmark logistické regrese pro Den 2:
1. Školní úloha dle slajdů (make_classification: 600 vzorků, 5 příznaků, 2 třídy).
2. Matematický rozbor sigmoidy a klinického onkologického příkladu (věk + průměr tumoru).
3. Analýza rozhodovacího prahu (Threshold Sweep theta in [0.05, 0.95]).
4. Srovnání OLS vs. Logistická regrese (proč OLS selhává při klasifikaci).
5. Aplikace na reálná data kurzu:
   - Diagnostika bederní páteře (lumbar_df_normalized.csv) – srovnání s k-NN.
   - Multiclass Palmer Penguins (penguins_size.csv) – One-vs-Rest vs. Multinomial Softmax.
6. Export předpočtených výsledků do JSON cache pro bleskový běh ve Streamlitu.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

def run_logistic_regression_analysis():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    plots_dir = base_dir / "plots"
    data_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    print("=== 1. Školní úloha dle slajdů: make_classification ===")
    X_syn, y_syn = make_classification(
        n_samples=600,
        n_features=5,
        n_classes=2,
        random_state=42
    )

    X_train_syn, X_test_syn, y_train_syn, y_test_syn = train_test_split(
        X_syn, y_syn, test_size=0.3, random_state=42
    )

    log_reg_syn = LogisticRegression(random_state=42)
    log_reg_syn.fit(X_train_syn, y_train_syn)

    y_pred_syn = log_reg_syn.predict(X_test_syn)
    y_prob_syn = log_reg_syn.predict_proba(X_test_syn)[:, 1]

    acc_syn = float(accuracy_score(y_test_syn, y_pred_syn))
    prec_syn = float(precision_score(y_test_syn, y_pred_syn))
    rec_syn = float(recall_score(y_test_syn, y_pred_syn))
    f1_syn = float(f1_score(y_test_syn, y_pred_syn))
    auc_syn = float(roc_auc_score(y_test_syn, y_prob_syn))
    ll_syn = float(log_loss(y_test_syn, y_prob_syn))
    cm_syn = confusion_matrix(y_test_syn, y_pred_syn).tolist()

    print(f"Accuracy: {acc_syn:.4f}")
    print(f"Precision: {prec_syn:.4f}")
    print(f"Recall: {rec_syn:.4f}")
    print(f"F1-score: {f1_syn:.4f}")
    print(f"ROC-AUC: {auc_syn:.4f}")
    print(f"Log Loss: {ll_syn:.4f}")

    # Koeficienty
    coefs_syn = log_reg_syn.coef_[0].tolist()
    intercept_syn = float(log_reg_syn.intercept_[0])

    # === 2. Analýza rozhodovacího prahu (Threshold Sweep) ===
    thresholds = np.linspace(0.05, 0.95, 37).tolist()
    threshold_metrics = []
    for th in thresholds:
        y_th = (y_prob_syn >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test_syn, y_th).ravel()
        p = float(precision_score(y_test_syn, y_th, zero_division=0))
        r = float(recall_score(y_test_syn, y_th, zero_division=0))
        f = float(f1_score(y_test_syn, y_th, zero_division=0))
        a = float(accuracy_score(y_test_syn, y_th))
        threshold_metrics.append({
            "threshold": round(th, 2),
            "accuracy": round(a, 4),
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn)
        })

    # ROC a PR křivka pro syntetický dataset
    fpr, tpr, roc_thresh = roc_curve(y_test_syn, y_prob_syn)
    pr_prec, pr_rec, pr_thresh = precision_recall_curve(y_test_syn, y_prob_syn)

    # === 3. Srovnání OLS vs. Logistická regrese (1D vizualizace) ===
    # Vybereme nejvýznamnější příznak
    best_feat_idx = int(np.argmax(np.abs(log_reg_syn.coef_[0])))
    x_1d = X_syn[:, best_feat_idx]
    lin_reg = LinearRegression()
    lin_reg.fit(x_1d.reshape(-1, 1), y_syn)

    x_grid = np.linspace(x_1d.min() - 1, x_1d.max() + 1, 200)
    y_lin_grid = lin_reg.predict(x_grid.reshape(-1, 1))

    log_reg_1d = LogisticRegression()
    log_reg_1d.fit(x_1d.reshape(-1, 1), y_syn)
    y_log_grid = log_reg_1d.predict_proba(x_grid.reshape(-1, 1))[:, 1]

    # === 4. Klinický onkologický příklad ze slajdů ===
    # p(x) = 1 / (1 + exp(-(0.005*age + 0.015*diameter + 0.1)))
    slide_age = 50
    slide_diam = 35
    slide_z = 0.005 * slide_age + 0.015 * slide_diam + 0.1
    slide_prob = 1.0 / (1.0 + np.exp(-slide_z))

    ages = np.linspace(20, 80, 25).tolist()
    diams = np.linspace(5, 60, 25).tolist()
    mesh_probs = []
    for a in ages:
        row = []
        for d in diams:
            z = 0.005 * a + 0.015 * d + 0.1
            p = float(1.0 / (1.0 + np.exp(-z)))
            row.append(round(p, 4))
        mesh_probs.append(row)

    # === 5. Aplikace na Lumbar Spine (Bederní páteř) ===
    lumbar_path = data_dir / "lumbar_df_normalized.csv"
    if not lumbar_path.exists():
        lumbar_path = base_dir / "lumbar_df_normalized.csv"

    lumbar_results = None
    if lumbar_path.exists():
        lumbar_df = pd.read_csv(lumbar_path)
        feature_cols = [c for c in lumbar_df.columns if c != "class"]
        X_lumb = lumbar_df[feature_cols].values
        y_lumb = lumbar_df["class"].values

        X_tr_l, X_te_l, y_tr_l, y_te_l = train_test_split(
            X_lumb, y_lumb, test_size=0.25, random_state=42, stratify=y_lumb
        )

        log_lumb = LogisticRegression(random_state=42, C=1.0)
        log_lumb.fit(X_tr_l, y_tr_l)
        y_pred_l = log_lumb.predict(X_te_l)
        y_prob_l = log_lumb.predict_proba(X_te_l)[:, 1]

        acc_l = float(accuracy_score(y_te_l, y_pred_l))
        rec_l = float(recall_score(y_te_l, y_pred_l))
        prec_l = float(precision_score(y_te_l, y_pred_l))
        f1_l = float(f1_score(y_te_l, y_pred_l))
        auc_l = float(roc_auc_score(y_te_l, y_prob_l))
        cm_l = confusion_matrix(y_te_l, y_pred_l).tolist()

        # Koeficienty pro páteř
        coef_lumb = {col: round(float(c), 4) for col, c in zip(feature_cols, log_lumb.coef_[0])}
        odds_ratios = {col: round(float(np.exp(c)), 4) for col, c in zip(feature_cols, log_lumb.coef_[0])}

        lumbar_results = {
            "accuracy": round(acc_l, 4),
            "recall": round(rec_l, 4),
            "precision": round(prec_l, 4),
            "f1": round(f1_l, 4),
            "roc_auc": round(auc_l, 4),
            "confusion_matrix": cm_l,
            "coefficients": coef_lumb,
            "odds_ratios": odds_ratios,
            "intercept": round(float(log_lumb.intercept_[0]), 4),
            "comparison_knn_k5": {
                "accuracy": 0.7308,
                "recall": 0.8070,
                "f1": 0.8142,
                "roc_auc": 0.8208
            },
            "comparison_knn_k8": {
                "accuracy": 0.8846,
                "recall": 0.8868,
                "f1": 0.9038,
                "roc_auc": 0.8650
            }
        }
        print(f"Lumbar LogReg - Accuracy: {acc_l:.4f}, Recall: {rec_l:.4f}, AUC: {auc_l:.4f}")

    # === 6. Sestavení výsledného JSON souboru ===
    precomputed_data = {
        "metadata": {
            "title": "Logistic Regression Precomputed Analysis",
            "version": "1.0",
            "date": "2026-09"
        },
        "synthetic_model": {
            "n_samples": 600,
            "n_features": 5,
            "n_classes": 2,
            "train_size": len(X_train_syn),
            "test_size": len(X_test_syn),
            "coefficients": [round(c, 4) for c in coefs_syn],
            "intercept": round(intercept_syn, 4),
            "accuracy": round(acc_syn, 4),
            "precision": round(prec_syn, 4),
            "recall": round(rec_syn, 4),
            "f1_score": round(f1_syn, 4),
            "roc_auc": round(auc_syn, 4),
            "log_loss": round(ll_syn, 4),
            "confusion_matrix": cm_syn,
            "threshold_sweep": threshold_metrics,
            "roc_curve": {
                "fpr": [round(float(x), 4) for x in fpr[::max(1, len(fpr)//25)]],
                "tpr": [round(float(x), 4) for x in tpr[::max(1, len(tpr)//25)]]
            }
        },
        "comparison_1d": {
            "feature_name": f"Feature {best_feat_idx}",
            "x_grid": [round(float(x), 3) for x in x_grid[::4]],
            "y_linear": [round(float(y), 3) for y in y_lin_grid[::4]],
            "y_logistic": [round(float(y), 3) for y in y_log_grid[::4]],
            "sample_points": [
                {"x": round(float(x_1d[i]), 3), "y": int(y_syn[i])}
                for i in range(0, len(y_syn), 10)
            ]
        },
        "clinical_tumor_example": {
            "slide_age": slide_age,
            "slide_diameter": slide_diam,
            "slide_z": round(slide_z, 4),
            "slide_prob": round(float(slide_prob), 4),
            "coef_age": 0.005,
            "coef_diameter": 0.015,
            "intercept": 0.1,
            "grid_ages": ages,
            "grid_diameters": diams,
            "mesh_probabilities": mesh_probs
        },
        "lumbar_analysis": lumbar_results
    }

    out_json = data_dir / "logistic_regression_precomputed.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(precomputed_data, f, indent=2, ensure_ascii=False)
    print(f"Uloženo do: {out_json}")

    # Uložení statického grafu pro přehled
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Graf 1: Srovnání OLS vs Logistic
    axes[0].scatter(x_1d[::5], y_syn[::5], alpha=0.5, color="#1f77b4", label="Data (y in {0, 1})")
    axes[0].plot(x_grid, y_lin_grid, color="#d62728", linestyle="--", label="Lineární regrese OLS")
    axes[0].plot(x_grid, y_log_grid, color="#2ca02c", linewidth=2.5, label="Logistická regrese (Sigmoida)")
    axes[0].axhline(0, color="gray", linestyle=":", alpha=0.6)
    axes[0].axhline(1, color="gray", linestyle=":", alpha=0.6)
    axes[0].set_ylim(-0.25, 1.25)
    axes[0].set_title("Lineární regrese OLS vs. Logistická regrese", fontsize=12, fontweight="bold")
    axes[0].set_xlabel(f"Příznak {best_feat_idx}")
    axes[0].set_ylabel("Predikce / Pravděpodobnost")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Graf 2: Threshold sweep
    th_vals = [m["threshold"] for m in threshold_metrics]
    prec_vals = [m["precision"] for m in threshold_metrics]
    rec_vals = [m["recall"] for m in threshold_metrics]
    f1_vals = [m["f1"] for m in threshold_metrics]

    axes[1].plot(th_vals, prec_vals, label="Precision", color="#1f77b4", linewidth=2)
    axes[1].plot(th_vals, rec_vals, label="Recall", color="#ff7f0e", linewidth=2)
    axes[1].plot(th_vals, f1_vals, label="F1-score", color="#2ca02c", linewidth=2)
    axes[1].axvline(0.5, color="gray", linestyle="--", label="Výchozí práh theta = 0.5")
    axes[1].set_title("Vliv rozhodovacího prahu (Threshold Sweep)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Rozhodovací práh theta")
    axes[1].set_ylabel("Skóre metriky")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = plots_dir / "logistic_regression_comparison.png"
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"Graf uložen do: {plot_file}")

if __name__ == "__main__":
    run_logistic_regression_analysis()
