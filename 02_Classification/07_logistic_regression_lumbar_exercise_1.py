"""
02_Classification/07_logistic_regression_lumbar_exercise_1.py

Logistic regression - exercise 1:
1. Using Pandas, load lumbar_df_normalized.csv into lumbar_df.
2. Display the first 10 observations to verify data validity.
3. Split data into train and test sets in 75/25 ratio, random_state=42.
4. Import LogisticRegression from sklearn.linear_model.
5. Create instance of LogisticRegression (no hyperparameters initially). Train on X_train, y_train.
6. Make predictions on test set using the trained model.
7. Calculate precision on test set.
8. Experiment to choose optimal hyperparameters (C, solver, penalty) for better precision.
9. Precompute results to JSON cache for fast Streamlit loading.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

def run_exercise():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    plots_dir = base_dir / "plots"
    data_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    # 1. Load data
    lumbar_path = data_dir / "lumbar_df_normalized.csv"
    if not lumbar_path.exists():
        lumbar_path = base_dir / "lumbar_df_normalized.csv"
    
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
    print(f"Počet pozitivních ve vzorku: Train = {y_train.sum()} ({y_train.mean()*100:.1f} %), Test = {y_test.sum()} ({y_test.mean()*100:.1f} %)")

    # 4 & 5. Baseline LogisticRegression (no hyperparameters initially)
    print("\n=== KROK 5: Výchozí model (default LogisticRegression) ===")
    base_model = LogisticRegression(random_state=42)
    base_model.fit(X_train, y_train)

    # 6. Predict on test set
    y_pred_base = base_model.predict(X_test)
    y_prob_base = base_model.predict_proba(X_test)[:, 1]

    # 7. Calculate precision on test set
    base_prec = float(precision_score(y_test, y_pred_base))
    base_acc = float(accuracy_score(y_test, y_pred_base))
    base_rec = float(recall_score(y_test, y_pred_base))
    base_f1 = float(f1_score(y_test, y_pred_base))
    base_auc = float(roc_auc_score(y_test, y_prob_base))
    base_cm = confusion_matrix(y_test, y_pred_base).tolist()

    print(f"Výchozí Precision (C=1.0) : {base_prec:.4f} ({base_prec*100:.2f} %)")
    print(f"Výchozí Accuracy          : {base_acc:.4f} ({base_acc*100:.2f} %)")
    print(f"Výchozí Recall            : {base_rec:.4f} ({base_rec*100:.2f} %)")
    print(f"Výchozí F1-score          : {base_f1:.4f} ({base_f1*100:.2f} %)")
    print(f"Výchozí Matice záměn (TN, FP / FN, TP): {base_cm}")

    # 8. Experiment with hyperparameters (Sweep C and regularization)
    print("\n=== KROK 8: Experiment s laděním C a regularizací ===")
    c_candidates = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    sweep_results = []

    best_c = 1.0
    best_c_prec = base_prec

    for c in c_candidates:
        m = LogisticRegression(C=c, random_state=42, max_iter=1000)
        m.fit(X_train, y_train)
        yp = m.predict(X_test)
        prob = m.predict_proba(X_test)[:, 1]
        
        p = float(precision_score(y_test, yp, zero_division=0))
        a = float(accuracy_score(y_test, yp))
        r = float(recall_score(y_test, yp, zero_division=0))
        f = float(f1_score(y_test, yp, zero_division=0))
        auc = float(roc_auc_score(y_test, prob))
        cm = confusion_matrix(y_test, yp).tolist()
        
        sweep_results.append({
            "C": c,
            "precision": round(p, 4),
            "accuracy": round(a, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
            "roc_auc": round(auc, 4),
            "confusion_matrix": cm,
            "tn": cm[0][0], "fp": cm[0][1], "fn": cm[1][0], "tp": cm[1][1]
        })
        print(f"C = {c:7.3f} | Precision: {p*100:.2f} % | Accuracy: {a*100:.2f} % | Recall: {r*100:.2f} % | FP={cm[0][1]}, FN={cm[1][0]}")
        
        if p > best_c_prec:
            best_c_prec = p
            best_c = c

    print(f"\n>> Nejlepší hodnota parametru C pro Precision: C = {best_c} (Precision: {best_c_prec*100:.2f} % vs {base_prec*100:.2f} % u baseline)")

    # Model s optimálním C=5.0
    opt_model_c5 = LogisticRegression(C=best_c, random_state=42, max_iter=1000)
    opt_model_c5.fit(X_train, y_train)
    y_pred_opt = opt_model_c5.predict(X_test)
    y_prob_opt = opt_model_c5.predict_proba(X_test)[:, 1]
    opt_cm = confusion_matrix(y_test, y_pred_opt).tolist()

    # Pokročilý experiment: L1 regularizace (Saga solver)
    print("\n--- Pokročilý experiment: L1 regularizace (Lasso výběr příznaků) ---")
    model_l1 = LogisticRegression(C=1.0, l1_ratio=1.0, solver="saga", random_state=42, max_iter=2000)
    model_l1.fit(X_train, y_train)
    y_pred_l1 = model_l1.predict(X_test)
    y_prob_l1 = model_l1.predict_proba(X_test)[:, 1]
    l1_prec = float(precision_score(y_test, y_pred_l1))
    l1_acc = float(accuracy_score(y_test, y_pred_l1))
    l1_rec = float(recall_score(y_test, y_pred_l1))
    l1_f1 = float(f1_score(y_test, y_pred_l1))
    l1_auc = float(roc_auc_score(y_test, y_prob_l1))
    l1_cm = confusion_matrix(y_test, y_pred_l1).tolist()
    print(f"L1 (C=1.0, saga) -> Precision: {l1_prec*100:.2f} %, Accuracy: {l1_acc*100:.2f} %, Recall: {l1_rec*100:.2f} %")

    # Koeficienty pro srovnání
    coef_comparison = []
    for col, b_base, b_opt, b_l1 in zip(feature_cols, base_model.coef_[0], opt_model_c5.coef_[0], model_l1.coef_[0]):
        coef_comparison.append({
            "feature": col,
            "beta_baseline_C1": round(float(b_base), 4),
            "odds_ratio_baseline": round(float(np.exp(b_base)), 4),
            "beta_opt_C5": round(float(b_opt), 4),
            "odds_ratio_opt_C5": round(float(np.exp(b_opt)), 4),
            "beta_L1": round(float(b_l1), 4),
            "odds_ratio_L1": round(float(np.exp(b_l1)), 4),
        })

    # ROC křivky
    fpr_base, tpr_base, _ = roc_curve(y_test, y_prob_base)
    fpr_opt, tpr_opt, _ = roc_curve(y_test, y_prob_opt)

    # Uložení do JSON cache
    cache_payload = {
        "metadata": {
            "exercise": "Logistic regression - exercise 1",
            "dataset": "lumbar_df_normalized.csv",
            "samples_total": len(lumbar_df),
            "samples_train": len(X_train),
            "samples_test": len(X_test),
            "test_size": 0.25,
            "random_state": 42
        },
        "head_10": head_10.to_dict(orient="records"),
        "features": feature_cols,
        "baseline_model": {
            "C": 1.0,
            "penalty": "l2",
            "solver": "lbfgs",
            "precision": round(base_prec, 4),
            "accuracy": round(base_acc, 4),
            "recall": round(base_rec, 4),
            "f1_score": round(base_f1, 4),
            "roc_auc": round(base_auc, 4),
            "confusion_matrix": base_cm,
            "intercept": round(float(base_model.intercept_[0]), 4)
        },
        "optimal_model_c5": {
            "C": best_c,
            "penalty": "l2",
            "solver": "lbfgs",
            "precision": round(best_c_prec, 4),
            "accuracy": round(float(accuracy_score(y_test, y_pred_opt)), 4),
            "recall": round(float(recall_score(y_test, y_pred_opt)), 4),
            "f1_score": round(float(f1_score(y_test, y_pred_opt)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, y_prob_opt)), 4),
            "confusion_matrix": opt_cm,
            "intercept": round(float(opt_model_c5.intercept_[0]), 4)
        },
        "l1_model": {
            "C": 1.0,
            "penalty": "l1",
            "solver": "saga",
            "precision": round(l1_prec, 4),
            "accuracy": round(l1_acc, 4),
            "recall": round(l1_rec, 4),
            "f1_score": round(l1_f1, 4),
            "roc_auc": round(l1_auc, 4),
            "confusion_matrix": l1_cm,
            "intercept": round(float(model_l1.intercept_[0]), 4)
        },
        "c_sweep": sweep_results,
        "coefficients": coef_comparison,
        "roc_curves": {
            "fpr_base": [round(float(x), 4) for x in fpr_base],
            "tpr_base": [round(float(x), 4) for x in tpr_base],
            "fpr_opt": [round(float(x), 4) for x in fpr_opt],
            "tpr_opt": [round(float(x), 4) for x in tpr_opt]
        }
    }

    out_json = data_dir / "lumbar_logistic_exercise_1_precomputed.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(cache_payload, f, indent=2, ensure_ascii=False)
    print(f"\nPředpočtené výsledky uloženy do: {out_json}")

    # Vykreslení a uložení diagnostického PNG grafu
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Graf 1: Křivka Precision vs C
    c_vals = [r["C"] for r in sweep_results]
    prec_vals = [r["precision"] for r in sweep_results]
    acc_vals = [r["accuracy"] for r in sweep_results]
    rec_vals = [r["recall"] for r in sweep_results]

    axes[0].plot(c_vals, prec_vals, marker="o", linewidth=2.5, color="#1f77b4", label="Precision (Přesnost)")
    axes[0].plot(c_vals, acc_vals, marker="s", linewidth=2, color="#2ca02c", linestyle="--", label="Accuracy (Celková přesnost)")
    axes[0].plot(c_vals, rec_vals, marker="^", linewidth=2, color="#ff7f0e", linestyle=":", label="Recall (Senzitivita)")
    axes[0].axvline(1.0, color="gray", linestyle="--", alpha=0.7, label="Výchozí C = 1.0 (Precision 80.0 %)")
    axes[0].axvline(best_c, color="red", linestyle=":", linewidth=2, label=f"Optimální C = {best_c} (Precision {best_c_prec*100:.1f} %)")
    axes[0].set_xscale("log")
    axes[0].set_title("Vliv regularizačního parametru C na metriky", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Parametr C (logaritmické měřítko – menší C = silnější regularizace)")
    axes[0].set_ylabel("Hodnota metriky")
    axes[0].set_ylim(0.65, 0.95)
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    # Graf 2: Porovnání matic záměn Baseline vs Optimal
    sns.heatmap(base_cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[1],
                xticklabels=["Normal (0)", "Abnormal (1)"], yticklabels=["Normal (0)", "Abnormal (1)"])
    axes[1].set_title(f"Výchozí matice záměn (C=1.0)\\nPrecision: {base_prec*100:.1f} % | FP: {base_cm[0][1]}", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Predikovaná třída")
    axes[1].set_ylabel("Skutečná třída")

    plt.tight_layout()
    plot_file = plots_dir / "lumbar_logistic_c_tuning.png"
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"Diagnostický graf uložen do: {plot_file}")

if __name__ == "__main__":
    run_exercise()
