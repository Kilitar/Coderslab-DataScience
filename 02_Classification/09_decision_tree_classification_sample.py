"""
02_Classification/09_decision_tree_classification_sample.py

Praktická implementace a demonstrace rozhodovacího stromu v klasifikaci pro Den 2:
1. Přesná replikace školního příkladu ze slajdů 17-21 (make_classification: 600, 5, 2, max_depth=5).
2. Vizualizace struktury stromu (plot_tree) a uložení diagramu.
3. Srovnání kritérií větvení: Gini Impurity vs. Shannonova Entropie.
4. Analýza přetrénování: Sweep hloubky max_depth in [1, 15] (Train vs. Test křivky).
5. Interaktivní podklady pro výpočet entropie a informačního zisku.
6. Export předpočtených výsledků do JSON cache pro bleskové načtení ve Streamlitu.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

def run_analysis():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    plots_dir = base_dir / "plots"
    data_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    # 1. Školní data (Slajd 17)
    print("=== 1. Generování syntetických dat (make_classification) ===")
    X, y = make_classification(
        n_samples=600,
        n_features=5,
        n_classes=2,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 2. Výchozí školní model: max_depth=5, criterion='gini' (Slajd 18)
    clf_tree = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42)
    clf_tree.fit(X_train, y_train)

    y_pred = clf_tree.predict(X_test)
    y_pred_train = clf_tree.predict(X_train)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred))
    rec = float(recall_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"Test Accuracy : {acc:.4f} ({acc*100:.2f} %)")
    print(f"Test Precision: {prec:.4f} ({prec*100:.2f} %)")
    print(f"Test Recall   : {rec:.4f} ({rec*100:.2f} %)")
    print(f"Matice záměn  : {cm}")

    # Feature importances
    feature_names = [f"x[{i}]" for i in range(5)]
    feat_imp = {fn: round(float(imp), 4) for fn, imp in zip(feature_names, clf_tree.feature_importances_)}
    print(f"Významnosti příznaků: {feat_imp}")

    # 3. Vykreslení a uložení struktury stromu (Slajd 21)
    plt.figure(figsize=(16, 9))
    plot_tree(clf_tree, feature_names=feature_names, class_names=["0", "1"], filled=True, rounded=True, fontsize=8)
    tree_plot_path = plots_dir / "decision_tree_sample_plot.png"
    plt.savefig(tree_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Graf stromu uložen do: {tree_plot_path}")

    # 4. Srovnání kritérií: Gini vs. Entropie
    clf_entropy = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42)
    clf_entropy.fit(X_train, y_train)
    y_pred_ent = clf_entropy.predict(X_test)

    ent_acc = float(accuracy_score(y_test, y_pred_ent))
    ent_prec = float(precision_score(y_test, y_pred_ent))

    # 5. Analýza přetrénování: Sweep max_depth od 1 do 15
    depth_sweep = []
    for depth in range(1, 16):
        m = DecisionTreeClassifier(criterion="gini", max_depth=depth, random_state=42)
        m.fit(X_train, y_train)
        
        tr_a = float(accuracy_score(y_train, m.predict(X_train)))
        te_a = float(accuracy_score(y_test, m.predict(X_test)))
        tr_p = float(precision_score(y_train, m.predict(X_train)))
        te_p = float(precision_score(y_test, m.predict(X_test)))
        
        depth_sweep.append({
            "depth": depth,
            "train_accuracy": round(tr_a, 4),
            "test_accuracy": round(te_a, 4),
            "train_precision": round(tr_p, 4),
            "test_precision": round(te_p, 4),
            "n_leaves": int(m.get_n_leaves())
        })

    # Diagnostický graf Train vs. Test
    fig, ax = plt.subplots(figsize=(10, 5))
    depths = [d["depth"] for d in depth_sweep]
    tr_accs = [d["train_accuracy"] for d in depth_sweep]
    te_accs = [d["test_accuracy"] for d in depth_sweep]

    ax.plot(depths, tr_accs, marker="o", color="#1f77b4", linewidth=2, label="Trénovací přesnost (Train Accuracy)")
    ax.plot(depths, te_accs, marker="s", color="#d62728", linewidth=2.5, label="Testovací přesnost (Test Accuracy)")
    ax.axvline(5, color="green", linestyle="--", label="Školní hloubka max_depth = 5")
    ax.set_title("Diagnostika přetrénování rozhodovacího stromu (Train vs. Test)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Maximální hloubka stromu (max_depth)")
    ax.set_ylabel("Přesnost (Accuracy)")
    ax.set_ylim(0.70, 1.02)
    ax.legend()
    ax.grid(True, alpha=0.3)

    overfit_plot_path = plots_dir / "decision_tree_overfitting_curve.png"
    plt.savefig(overfit_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Graf přetrénování uložen do: {overfit_plot_path}")

    # 6. Sestavení a uložení do JSON cache
    payload = {
        "metadata": {
            "exercise": "Decision tree in classification - sample",
            "samples_total": len(X),
            "samples_train": len(X_train),
            "samples_test": len(X_test),
            "features_count": 5
        },
        "school_model": {
            "criterion": "gini",
            "max_depth": 5,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": cm,
            "feature_importances": feat_imp,
            "n_leaves": int(clf_tree.get_n_leaves()),
            "actual_depth": int(clf_tree.get_depth())
        },
        "criterion_comparison": {
            "gini": {
                "accuracy": round(acc, 4),
                "precision": round(prec, 4)
            },
            "entropy": {
                "accuracy": round(ent_acc, 4),
                "precision": round(ent_prec, 4)
            }
        },
        "depth_sweep": depth_sweep,
        "toy_weather_example": {
            "total_samples": 6,
            "sunny": 3,
            "rainy": 3,
            "root_entropy": 1.0,
            "root_gini": 0.5,
            "split_temperature": {
                "low": {"rainy": 2, "sunny": 1, "entropy": 0.918, "gini": 0.375},
                "high": {"rainy": 0, "sunny": 3, "entropy": 0.0, "gini": 0.0},
                "weighted_entropy": 0.459,
                "information_gain": 0.541,
                "weighted_gini": 0.25
            }
        }
    }

    out_json = data_dir / "decision_tree_classification_precomputed.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Výsledky úspěšně uloženy do: {out_json}")

if __name__ == "__main__":
    run_analysis()
