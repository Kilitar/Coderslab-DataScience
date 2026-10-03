"""
02_Classification/11_decision_tree_penguins_exercise_2.py

Decision tree in classification - exercise 2:
1. Using Pandas, into penguins_df read penguins_df_normalized.csv.
2. Display first 10 observations to verify data validity.
3. Split data into train and test sets in 70/30 ratio, random_state=42.
4. Import DecisionTreeClassifier from sklearn.tree.
5. Create instance without hyperparameters initially. Train on training set.
6. Make predictions on test set using the trained model.
7. Calculate precision on test set (macro, weighted, and per-class).
8. Experiment to choose optimal values of hyperparameters (criterion, max_depth, min_samples_split, min_samples_leaf)
   that produce a model with better precision than the original model.
9. Precompute results to JSON cache for fast Streamlit loading.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
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
    target_col = "species"
    X = penguins_df[feature_cols]
    y = penguins_df[target_col]

    classes = sorted(y.unique().tolist())
    print(f"Třídy tučňáků: {classes}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    print(f"Rozdělení: X_train = {X_train.shape}, X_test = {X_test.shape}")
    print("Zastoupení tříd v testovací sadě:")
    print(y_test.value_counts())

    # 4 & 5. Baseline DecisionTreeClassifier (no hyperparameters initially)
    print("\n=== KROK 5: Výchozí model (default DecisionTreeClassifier) ===")
    base_model = DecisionTreeClassifier(random_state=42)
    base_model.fit(X_train, y_train)

    base_depth = int(base_model.get_depth())
    base_leaves = int(base_model.get_n_leaves())
    print(f"Výchozí strom: max_depth = {base_depth}, počet listů = {base_leaves}")

    # 6. Predict on test set
    y_pred_base = base_model.predict(X_test)

    # 7. Calculate precision on test set
    prec_macro_base = float(precision_score(y_test, y_pred_base, average="macro"))
    prec_weighted_base = float(precision_score(y_test, y_pred_base, average="weighted"))
    prec_per_class_base = {
        cls: round(float(p), 4)
        for cls, p in zip(classes, precision_score(y_test, y_pred_base, average=None))
    }
    acc_base = float(accuracy_score(y_test, y_pred_base))
    rec_macro_base = float(recall_score(y_test, y_pred_base, average="macro"))
    f1_macro_base = float(f1_score(y_test, y_pred_base, average="macro"))
    cm_base = confusion_matrix(y_test, y_pred_base, labels=classes).tolist()

    print(f"\n=== KROK 7: Metriky výchozího modelu ===")
    print(f"Precision (macro):    {prec_macro_base:.4f} ({prec_macro_base*100:.2f} %)")
    print(f"Precision (weighted): {prec_weighted_base:.4f} ({prec_weighted_base*100:.2f} %)")
    print(f"Per-class Precision:  {prec_per_class_base}")
    print(f"Accuracy  (test):     {acc_base:.4f} ({acc_base*100:.2f} %)")
    print(f"Recall    (macro):    {rec_macro_base:.4f} ({rec_macro_base*100:.2f} %)")
    print(f"Confusion Matrix (labels={classes}):\n{np.array(cm_base)}")

    # 8. Hyperparameter Experimentation
    # A) max_depth sweep (1 to 10)
    print("\n=== KROK 8: Experimenty s hyperparametry ===")
    depth_sweep = []
    for d in range(1, 11):
        clf = DecisionTreeClassifier(max_depth=d, random_state=42)
        clf.fit(X_train, y_train)
        pred_train = clf.predict(X_train)
        pred_test = clf.predict(X_test)

        depth_sweep.append({
            "depth": d,
            "train_prec_macro": round(float(precision_score(y_train, pred_train, average="macro", zero_division=0)), 4),
            "test_prec_macro": round(float(precision_score(y_test, pred_test, average="macro", zero_division=0)), 4),
            "test_prec_weighted": round(float(precision_score(y_test, pred_test, average="weighted", zero_division=0)), 4),
            "train_accuracy": round(float(accuracy_score(y_train, pred_train)), 4),
            "test_accuracy": round(float(accuracy_score(y_test, pred_test)), 4),
            "test_recall_macro": round(float(recall_score(y_test, pred_test, average="macro", zero_division=0)), 4),
            "test_f1_macro": round(float(f1_score(y_test, pred_test, average="macro", zero_division=0)), 4),
            "n_leaves": int(clf.get_n_leaves())
        })

    # B) Criterion sweep ('gini' vs 'entropy' vs 'log_loss')
    crit_sweep = []
    for crit in ["gini", "entropy", "log_loss"]:
        clf = DecisionTreeClassifier(criterion=crit, random_state=42).fit(X_train, y_train)
        pred_test = clf.predict(X_test)
        crit_sweep.append({
            "criterion": crit,
            "test_prec_macro": round(float(precision_score(y_test, pred_test, average="macro")), 4),
            "test_prec_weighted": round(float(precision_score(y_test, pred_test, average="weighted")), 4),
            "test_accuracy": round(float(accuracy_score(y_test, pred_test)), 4),
            "depth": int(clf.get_depth()),
            "n_leaves": int(clf.get_n_leaves())
        })

    # C) min_samples_split sweep
    split_sweep = []
    for s in [2, 3, 4, 5, 6, 7, 8, 10, 15, 20]:
        clf = DecisionTreeClassifier(min_samples_split=s, random_state=42).fit(X_train, y_train)
        pred_test = clf.predict(X_test)
        split_sweep.append({
            "min_samples_split": s,
            "test_prec_macro": round(float(precision_score(y_test, pred_test, average="macro")), 4),
            "test_accuracy": round(float(accuracy_score(y_test, pred_test)), 4),
            "n_leaves": int(clf.get_n_leaves())
        })

    # D) min_samples_leaf sweep
    leaf_sweep = []
    for l in [1, 2, 3, 4, 5, 6, 8, 10]:
        clf = DecisionTreeClassifier(min_samples_leaf=l, random_state=42).fit(X_train, y_train)
        pred_test = clf.predict(X_test)
        leaf_sweep.append({
            "min_samples_leaf": l,
            "test_prec_macro": round(float(precision_score(y_test, pred_test, average="macro")), 4),
            "test_accuracy": round(float(accuracy_score(y_test, pred_test)), 4),
            "n_leaves": int(clf.get_n_leaves())
        })

    # Optimal Model identified: max_depth=4
    # (Removes the noisy 5th level split, reducing FP on Chinstrap from 4 to 3!)
    opt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
    opt_model.fit(X_train, y_train)
    y_pred_opt = opt_model.predict(X_test)

    prec_macro_opt = float(precision_score(y_test, y_pred_opt, average="macro"))
    prec_weighted_opt = float(precision_score(y_test, y_pred_opt, average="weighted"))
    prec_per_class_opt = {
        cls: round(float(p), 4)
        for cls, p in zip(classes, precision_score(y_test, y_pred_opt, average=None))
    }
    acc_opt = float(accuracy_score(y_test, y_pred_opt))
    rec_macro_opt = float(recall_score(y_test, y_pred_opt, average="macro"))
    f1_macro_opt = float(f1_score(y_test, y_pred_opt, average="macro"))
    cm_opt = confusion_matrix(y_test, y_pred_opt, labels=classes).tolist()

    print(f"\nOptimální model (max_depth=4):")
    print(f"  Precision (macro):    {prec_macro_opt:.4f} ({prec_macro_opt*100:.2f} %) [Baseline: {prec_macro_base*100:.2f} %]")
    print(f"  Precision (weighted): {prec_weighted_opt:.4f} ({prec_weighted_opt*100:.2f} %) [Baseline: {prec_weighted_base*100:.2f} %]")
    print(f"  Per-class Precision:  {prec_per_class_opt} (Chinstrap stoupl z {prec_per_class_base['Chinstrap']*100:.2f}% na {prec_per_class_opt['Chinstrap']*100:.2f}%)")
    print(f"  Accuracy:             {acc_opt:.4f} ({acc_opt*100:.2f} %) [Baseline: {acc_base*100:.2f} %]")
    print(f"  Počet listů:          {opt_model.get_n_leaves()} (vs baseline {base_leaves})")

    # Feature importances
    feat_imp_base = {col: round(float(imp), 4) for col, imp in zip(feature_cols, base_model.feature_importances_)}
    feat_imp_opt = {col: round(float(imp), 4) for col, imp in zip(feature_cols, opt_model.feature_importances_)}

    # Plots
    # 1. Depth sweep plot
    fig, ax = plt.subplots(figsize=(9, 5))
    depths = [d["depth"] for d in depth_sweep]
    test_p_macro = [d["test_prec_macro"] * 100 for d in depth_sweep]
    test_p_w = [d["test_prec_weighted"] * 100 for d in depth_sweep]
    test_accs = [d["test_accuracy"] * 100 for d in depth_sweep]

    ax.plot(depths, test_p_macro, marker="o", color="#1f77b4", linewidth=2.5, label="Test Precision Macro (%)")
    ax.plot(depths, test_p_w, marker="s", color="#2ca02c", linewidth=2, linestyle="--", label="Test Precision Weighted (%)")
    ax.plot(depths, test_accs, marker="^", color="#ff7f0e", linewidth=1.5, alpha=0.8, label="Test Accuracy (%)")
    ax.axhline(prec_macro_base * 100, color="red", linestyle=":", label=f"Baseline Macro Precision ({prec_macro_base*100:.1f} %)")
    ax.axvline(4, color="green", linestyle="-.", alpha=0.7, label="Optimum: max_depth=4")
    ax.set_title("Vliv hyperparametru max_depth na Precision tučňáků (Multiclass)", fontsize=13, fontweight="bold")
    ax.set_xlabel("max_depth", fontsize=11)
    ax.set_ylabel("Metrika v %", fontsize=11)
    ax.set_xticks(depths)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plot_curve_path = plots_dir / "penguins_tree_depth_curve.png"
    plt.savefig(plot_curve_path, dpi=200)
    plt.close()
    print(f"Uložen graf křivky hloubky: {plot_curve_path}")

    # 2. Optimal tree structure plot (max_depth=4)
    fig, ax = plt.subplots(figsize=(16, 8))
    plot_tree(
        opt_model,
        feature_names=feature_cols,
        class_names=classes,
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax
    )
    ax.set_title("Struktura optimálního rozhodovacího stromu pro druhy tučňáků (max_depth=4)", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plot_tree_path = plots_dir / "penguins_tree_structure_opt.png"
    plt.savefig(plot_tree_path, dpi=200)
    plt.close()
    print(f"Uložen graf optimálního stromu: {plot_tree_path}")

    # 3. Confusion matrix comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cm_base, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax1, cbar=False)
    ax1.set_title(f"Výchozí strom (depth=5)\nMacro Precision: {prec_macro_base*100:.2f} % | Acc: {acc_base*100:.2f} %")
    ax1.set_xlabel("Predikovaný druh")
    ax1.set_ylabel("Skutečný druh")

    sns.heatmap(cm_opt, annot=True, fmt="d", cmap="Greens", xticklabels=classes, yticklabels=classes, ax=ax2, cbar=False)
    ax2.set_title(f"Optimální strom (max_depth=4)\nMacro Precision: {prec_macro_opt*100:.2f} % | Acc: {acc_opt*100:.2f} %")
    ax2.set_xlabel("Predikovaný druh")
    ax2.set_ylabel("Skutečný druh")

    plt.tight_layout()
    plot_cm_path = plots_dir / "penguins_tree_cm_comparison.png"
    plt.savefig(plot_cm_path, dpi=200)
    plt.close()
    print(f"Uložen graf matic záměn: {plot_cm_path}")

    # JSON export
    payload = {
        "metadata": {
            "exercise": "Decision tree in classification - exercise 2",
            "dataset": "penguins_df_normalized.csv",
            "samples_total": len(penguins_df),
            "samples_train": len(X_train),
            "samples_test": len(X_test),
            "test_ratio": 0.30,
            "random_state": 42,
            "classes": classes,
            "feature_names": feature_cols
        },
        "head_10": head_10.to_dict(orient="records"),
        "baseline_model": {
            "model_type": "DecisionTreeClassifier(random_state=42)",
            "max_depth": base_depth,
            "n_leaves": base_leaves,
            "precision_macro": round(prec_macro_base, 4),
            "precision_weighted": round(prec_weighted_base, 4),
            "precision_per_class": prec_per_class_base,
            "accuracy": round(acc_base, 4),
            "recall_macro": round(rec_macro_base, 4),
            "f1_macro": round(f1_macro_base, 4),
            "confusion_matrix": cm_base,
            "feature_importances": feat_imp_base
        },
        "optimal_model_depth4": {
            "max_depth": 4,
            "n_leaves": int(opt_model.get_n_leaves()),
            "precision_macro": round(prec_macro_opt, 4),
            "precision_weighted": round(prec_weighted_opt, 4),
            "precision_per_class": prec_per_class_opt,
            "accuracy": round(acc_opt, 4),
            "recall_macro": round(rec_macro_opt, 4),
            "f1_macro": round(f1_macro_opt, 4),
            "confusion_matrix": cm_opt,
            "feature_importances": feat_imp_opt
        },
        "depth_sweep": depth_sweep,
        "criterion_sweep": crit_sweep,
        "split_sweep": split_sweep,
        "leaf_sweep": leaf_sweep
    }

    json_path = data_dir / "penguins_decision_tree_exercise_2_precomputed.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nVýsledky úspěšně uloženy do: {json_path}")

if __name__ == "__main__":
    run_exercise()
