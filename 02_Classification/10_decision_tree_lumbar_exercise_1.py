"""
02_Classification/10_decision_tree_lumbar_exercise_1.py

Decision tree in classification - exercise 1:
1. Using Pandas, into lumbar_df read lumbar_normalized_df.csv.
2. Display first 10 observations to verify data validity.
3. Split data into train and test sets in 75/25 ratio, random_state=42.
4. Import DecisionTreeClassifier from sklearn.tree.
5. Create instance without hyperparameters initially. Train on training set.
6. Make predictions on test set using the trained model.
7. Calculate precision on test set.
8. Experiment to choose optimal values of a hyperparameter (max_depth, min_samples_split, etc.)
   that will allow creating a model with better precision than the baseline model.
9. Precompute results to JSON cache for fast Streamlit loading.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    precision_score,
    accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
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
        lumbar_path = data_dir / "lumbar_df_normalized.csv"
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

    # 4 & 5. Baseline DecisionTreeClassifier (no hyperparameters initially)
    print("\n=== KROK 5: Výchozí model (default DecisionTreeClassifier) ===")
    base_model = DecisionTreeClassifier(random_state=42)
    base_model.fit(X_train, y_train)

    base_depth = int(base_model.get_depth())
    base_leaves = int(base_model.get_n_leaves())
    print(f"Výchozí strom: max_depth = {base_depth}, počet listů = {base_leaves}")

    # 6. Predict on test set
    y_pred_base = base_model.predict(X_test)
    y_prob_base = base_model.predict_proba(X_test)[:, 1]

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
    # A) max_depth sweep (1 to 15)
    print("\n=== KROK 8: Experimenty s hyperparametry ===")
    depth_sweep = []
    for d in range(1, 16):
        clf = DecisionTreeClassifier(max_depth=d, random_state=42)
        clf.fit(X_train, y_train)
        train_pred = clf.predict(X_train)
        test_pred = clf.predict(X_test)

        depth_sweep.append({
            "depth": d,
            "train_precision": round(float(precision_score(y_train, train_pred)), 4),
            "test_precision": round(float(precision_score(y_test, test_pred)), 4),
            "train_accuracy": round(float(accuracy_score(y_train, train_pred)), 4),
            "test_accuracy": round(float(accuracy_score(y_test, test_pred)), 4),
            "test_recall": round(float(recall_score(y_test, test_pred)), 4),
            "test_f1": round(float(f1_score(y_test, test_pred)), 4),
            "n_leaves": int(clf.get_n_leaves())
        })

    # B) min_samples_split sweep
    split_sweep = []
    for s in [2, 5, 10, 15, 20, 25, 30, 40, 50]:
        clf = DecisionTreeClassifier(min_samples_split=s, random_state=42)
        clf.fit(X_train, y_train)
        test_pred = clf.predict(X_test)
        split_sweep.append({
            "min_samples_split": s,
            "test_precision": round(float(precision_score(y_test, test_pred)), 4),
            "test_accuracy": round(float(accuracy_score(y_test, test_pred)), 4),
            "test_recall": round(float(recall_score(y_test, test_pred)), 4),
            "test_f1": round(float(f1_score(y_test, test_pred)), 4),
            "depth": int(clf.get_depth()),
            "n_leaves": int(clf.get_n_leaves())
        })

    # C) min_samples_leaf sweep
    leaf_sweep = []
    for l in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]:
        clf = DecisionTreeClassifier(min_samples_leaf=l, random_state=42)
        clf.fit(X_train, y_train)
        test_pred = clf.predict(X_test)
        leaf_sweep.append({
            "min_samples_leaf": l,
            "test_precision": round(float(precision_score(y_test, test_pred)), 4),
            "test_accuracy": round(float(accuracy_score(y_test, test_pred)), 4),
            "test_recall": round(float(recall_score(y_test, test_pred)), 4),
            "test_f1": round(float(f1_score(y_test, test_pred)), 4),
            "depth": int(clf.get_depth()),
            "n_leaves": int(clf.get_n_leaves())
        })

    # D) Criterion comparison (Gini vs Entropy)
    crit_gini = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42).fit(X_train, y_train)
    crit_entropy = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42).fit(X_train, y_train)
    crit_comparison = {
        "gini_depth3": {
            "precision": round(float(precision_score(y_test, crit_gini.predict(X_test))), 4),
            "accuracy": round(float(accuracy_score(y_test, crit_gini.predict(X_test))), 4),
            "recall": round(float(recall_score(y_test, crit_gini.predict(X_test))), 4)
        },
        "entropy_depth3": {
            "precision": round(float(precision_score(y_test, crit_entropy.predict(X_test))), 4),
            "accuracy": round(float(accuracy_score(y_test, crit_entropy.predict(X_test))), 4),
            "recall": round(float(recall_score(y_test, crit_entropy.predict(X_test))), 4)
        }
    }

    # Best models identified
    # Model 1: Decision Stump (max_depth=1) -> Highest raw Precision (0.9556)
    stump_model = DecisionTreeClassifier(max_depth=1, random_state=42).fit(X_train, y_train)
    y_pred_stump = stump_model.predict(X_test)
    prec_stump = float(precision_score(y_test, y_pred_stump))

    # Model 2: Balanced Tuned Tree (max_depth=3) -> High Precision (0.9167) + Better Recall & Structure
    tuned_model_d3 = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train, y_train)
    y_pred_d3 = tuned_model_d3.predict(X_test)
    prec_d3 = float(precision_score(y_test, y_pred_d3))

    # Model 3: Leaf-regularized tree (min_samples_leaf=4) -> Precision 0.8824
    tuned_model_leaf = DecisionTreeClassifier(min_samples_leaf=4, random_state=42).fit(X_train, y_train)
    y_pred_leaf = tuned_model_leaf.predict(X_test)
    prec_leaf = float(precision_score(y_test, y_pred_leaf))

    print(f"Optimal Model (max_depth=1): Precision = {prec_stump*100:.2f} % (vs baseline {prec_base*100:.2f} %)")
    print(f"Balanced Model (max_depth=3): Precision = {prec_d3*100:.2f} % (vs baseline {prec_base*100:.2f} %)")
    print(f"Leaf-tuned Model (min_samples_leaf=4): Precision = {prec_leaf*100:.2f} % (vs baseline {prec_base*100:.2f} %)")

    # Feature importances of baseline and tuned model
    feat_imp_base = {col: round(float(imp), 4) for col, imp in zip(feature_cols, base_model.feature_importances_)}
    feat_imp_d3 = {col: round(float(imp), 4) for col, imp in zip(feature_cols, tuned_model_d3.feature_importances_)}

    # Plots
    # 1. Depth sweep plot
    fig, ax = plt.subplots(figsize=(10, 5))
    depths = [d["depth"] for d in depth_sweep]
    test_precisions = [d["test_precision"] * 100 for d in depth_sweep]
    train_precisions = [d["train_precision"] * 100 for d in depth_sweep]
    test_accuracies = [d["test_accuracy"] * 100 for d in depth_sweep]

    ax.plot(depths, test_precisions, marker="o", color="#1f77b4", linewidth=2.5, label="Test Precision (%)")
    ax.plot(depths, train_precisions, marker="s", color="#2ca02c", linestyle="--", alpha=0.7, label="Train Precision (%)")
    ax.plot(depths, test_accuracies, marker="^", color="#ff7f0e", linewidth=1.5, alpha=0.8, label="Test Accuracy (%)")
    ax.axhline(prec_base * 100, color="red", linestyle=":", label=f"Baseline Precision ({prec_base*100:.1f} %)")
    ax.set_title("Vliv hyperparametru max_depth na Precision a Accuracy (Bederní páteř)", fontsize=13, fontweight="bold")
    ax.set_xlabel("max_depth", fontsize=11)
    ax.set_ylabel("Metrika v %", fontsize=11)
    ax.set_xticks(depths)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plot_curve_path = plots_dir / "lumbar_tree_depth_precision_curve.png"
    plt.savefig(plot_curve_path, dpi=200)
    plt.close()
    print(f"Uložen graf: {plot_curve_path}")

    # 2. Tree structure plot for max_depth=3
    fig, ax = plt.subplots(figsize=(14, 7))
    plot_tree(
        tuned_model_d3,
        feature_names=feature_cols,
        class_names=["Normal (0)", "Abnormal (1)"],
        filled=True,
        rounded=True,
        fontsize=9,
        ax=ax
    )
    ax.set_title("Struktura optimálního rozhodovacího stromu (max_depth=3)", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plot_tree_path = plots_dir / "lumbar_tree_structure_d3.png"
    plt.savefig(plot_tree_path, dpi=200)
    plt.close()
    print(f"Uložen graf stromu: {plot_tree_path}")

    # 3. Tree structure plot for Decision Stump (max_depth=1)
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_tree(
        stump_model,
        feature_names=feature_cols,
        class_names=["Normal (0)", "Abnormal (1)"],
        filled=True,
        rounded=True,
        fontsize=10,
        ax=ax
    )
    ax.set_title("Rozhodovací pařez (Decision Stump, max_depth=1) - Max Precision", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plot_stump_path = plots_dir / "lumbar_tree_structure_stump.png"
    plt.savefig(plot_stump_path, dpi=200)
    plt.close()
    print(f"Uložen graf pařezu: {plot_stump_path}")

    # 4. Confusion matrix comparison plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    cm_stump = confusion_matrix(y_test, y_pred_stump)
    cm_d3 = confusion_matrix(y_test, y_pred_d3)

    sns.heatmap(cm_base, annot=True, fmt="d", cmap="Blues", ax=ax1, cbar=False)
    ax1.set_title(f"Výchozí strom (depth={base_depth})\nPrecision: {prec_base*100:.2f} %")
    ax1.set_xlabel("Predikovaná třída")
    ax1.set_ylabel("Skutečná třída")

    sns.heatmap(cm_stump, annot=True, fmt="d", cmap="Greens", ax=ax2, cbar=False)
    ax2.set_title(f"Rozhodovací pařez (depth=1)\nPrecision: {prec_stump*100:.2f} %")
    ax2.set_xlabel("Predikovaná třída")
    ax2.set_ylabel("Skutečná třída")

    sns.heatmap(cm_d3, annot=True, fmt="d", cmap="Purples", ax=ax3, cbar=False)
    ax3.set_title(f"Vyvážený strom (depth=3)\nPrecision: {prec_d3*100:.2f} %")
    ax3.set_xlabel("Predikovaná třída")
    ax3.set_ylabel("Skutečná třída")

    plt.tight_layout()
    plot_cm_path = plots_dir / "lumbar_tree_cm_comparison.png"
    plt.savefig(plot_cm_path, dpi=200)
    plt.close()
    print(f"Uložen graf matic záměn: {plot_cm_path}")

    # JSON export
    payload = {
        "metadata": {
            "exercise": "Decision tree in classification - exercise 1",
            "dataset": "lumbar_normalized_df.csv",
            "samples_total": len(lumbar_df),
            "samples_train": len(X_train),
            "samples_test": len(X_test),
            "test_ratio": 0.25,
            "random_state": 42,
            "feature_names": feature_cols,
            "target_distribution": {
                "train_abnormal": int(y_train.sum()),
                "train_normal": int(len(y_train) - y_train.sum()),
                "test_abnormal": int(y_test.sum()),
                "test_normal": int(len(y_test) - y_test.sum())
            }
        },
        "head_10": head_10.to_dict(orient="records"),
        "baseline_model": {
            "model_type": "DecisionTreeClassifier(random_state=42)",
            "max_depth": base_depth,
            "n_leaves": base_leaves,
            "precision": round(prec_base, 4),
            "accuracy": round(acc_base, 4),
            "recall": round(rec_base, 4),
            "f1_score": round(f1_base, 4),
            "confusion_matrix": cm_base,
            "feature_importances": feat_imp_base
        },
        "stump_model": {
            "max_depth": 1,
            "precision": round(prec_stump, 4),
            "accuracy": round(float(accuracy_score(y_test, y_pred_stump)), 4),
            "recall": round(float(recall_score(y_test, y_pred_stump)), 4),
            "f1_score": round(float(f1_score(y_test, y_pred_stump)), 4),
            "confusion_matrix": cm_stump.tolist(),
            "split_feature": "degree_spondylolisthesis",
            "split_threshold": round(float(stump_model.tree_.threshold[0]), 4)
        },
        "tuned_model_depth3": {
            "max_depth": 3,
            "precision": round(prec_d3, 4),
            "accuracy": round(float(accuracy_score(y_test, y_pred_d3)), 4),
            "recall": round(float(recall_score(y_test, y_pred_d3)), 4),
            "f1_score": round(float(f1_score(y_test, y_pred_d3)), 4),
            "confusion_matrix": cm_d3.tolist(),
            "n_leaves": int(tuned_model_d3.get_n_leaves()),
            "feature_importances": feat_imp_d3
        },
        "depth_sweep": depth_sweep,
        "split_sweep": split_sweep,
        "leaf_sweep": leaf_sweep,
        "criterion_comparison": crit_comparison
    }

    json_path = data_dir / "lumbar_decision_tree_exercise_1_precomputed.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nVýsledky úspěšně uloženy do: {json_path}")

if __name__ == "__main__":
    run_exercise()
