"""
01_hyperparameter_optimization_sample.py
========================================
Praktická implementace a benchmark 3 optimalizačních technik:
1. GridSearchCV (DecisionTreeClassifier)
2. RandomizedSearchCV (Support Vector Machine - SVC)
3. Bayesian Optimization s knihovnou Hyperopt (DecisionTreeClassifier)

Dataset: lumbar_normalized_df.csv (detekce patologie páteře)
Trénovací a testovací sada: 75/25, random_state=42
Výstupy: JSON cache + diagnostické grafy
"""

import os
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import uniform

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from hyperopt import hp, fmin, tpe, Trials, space_eval


def run_hyperparameter_optimization_sample():
    print("=" * 70)
    print("🚀 SPOUŠTÍM BENCHMARK OPTIMALIZACE HYPERPARAMETRŮ (SESSION 2 / DEN 3)")
    print("=" * 70)

    # Cesty k datům a výstupům
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    plots_dir = base_dir / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    csv_path = base_dir.parent / "02_Classification" / "lumbar_normalized_df.csv"
    if not csv_path.exists():
        csv_path = base_dir.parent / "02_Classification" / "data" / "lumbar_normalized_df.csv"

    print(f"📂 Načítám dataset: {csv_path}")
    df = pd.read_csv(csv_path)

    # Binární mapování: Pokud je už 0/1, použijeme přímo, jinak mapujeme Abnormal -> 1, Normal -> 0
    if "class" in df.columns:
        X = df.drop(columns=["class"])
        if df["class"].dtype == object:
            y = (df["class"] == "Abnormal").astype(int)
        else:
            y = df["class"].astype(int)
    else:
        raise ValueError("Sloupec 'class' nebyl v datasetu nalezen!")

    # Rozdělení 75/25
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"📊 Trénovací vzorky: {X_train.shape[0]}, Testovací vzorky: {X_test.shape[0]}")

    results = {
        "dataset_summary": {
            "n_samples": len(df),
            "n_features": X.shape[1],
            "feature_names": list(X.columns),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "class_distribution_train": {
                "Abnormal": int((y_train == 1).sum()),
                "Normal": int((y_train == 0).sum())
            },
            "class_distribution_test": {
                "Abnormal": int((y_test == 1).sum()),
                "Normal": int((y_test == 0).sum())
            }
        }
    }

    # =========================================================================
    # 1. GridSearchCV (DecisionTreeClassifier)
    # =========================================================================
    print("\n--- [1/3] Spouštím GridSearchCV (DecisionTreeClassifier) ---")
    dt_param_grid = {
        "max_depth": np.arange(3, 21, 2).tolist(),  # [3, 5, 7, 9, 11, 13, 15, 17, 19]
        "criterion": ["entropy", "gini"],
        "max_features": [None, "log2", "sqrt"],
    }
    total_grid_combos = len(dt_param_grid["max_depth"]) * len(dt_param_grid["criterion"]) * len(dt_param_grid["max_features"])
    print(f"Kombinací v mřížce: {total_grid_combos}, celkem běhů při 5-fold CV: {total_grid_combos * 5}")

    t0 = time.time()
    dt_model = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(
        dt_model,
        dt_param_grid,
        cv=5,
        scoring="recall",
        n_jobs=-1,
        return_train_score=True
    )
    grid_search.fit(X_train, y_train)
    grid_time = time.time() - t0

    best_dt_params = grid_search.best_params_
    best_dt_estimator = grid_search.best_estimator_
    dt_test_preds = best_dt_estimator.predict(X_test)

    grid_results = {
        "time_seconds": round(grid_time, 4),
        "total_combinations": total_grid_combos,
        "best_cv_recall": round(float(grid_search.best_score_), 4),
        "best_params": {
            "max_depth": int(best_dt_params["max_depth"]),
            "criterion": str(best_dt_params["criterion"]),
            "max_features": str(best_dt_params["max_features"])
        },
        "test_metrics": {
            "accuracy": round(float(accuracy_score(y_test, dt_test_preds)), 4),
            "precision": round(float(precision_score(y_test, dt_test_preds)), 4),
            "recall": round(float(recall_score(y_test, dt_test_preds)), 4),
            "f1": round(float(f1_score(y_test, dt_test_preds)), 4)
        },
        "confusion_matrix": confusion_matrix(y_test, dt_test_preds).tolist(),
        "top_5_models": []
    }

    # Vyfiltrovat top 5 kombinací
    cv_res_df = pd.DataFrame(grid_search.cv_results_)
    top5_df = cv_res_df.sort_values(by="mean_test_score", ascending=False).head(5)
    for _, row in top5_df.iterrows():
        grid_results["top_5_models"].append({
            "params": {k: (str(v) if v is None else v) for k, v in row["params"].items()},
            "mean_test_recall": round(float(row["mean_test_score"]), 4),
            "std_test_recall": round(float(row["std_test_score"]), 4),
            "mean_train_recall": round(float(row["mean_train_score"]), 4)
        })

    results["grid_search"] = grid_results
    print(f"GridSearchCV dokončen za {grid_time:.2f} s. Nejlepší parametry: {best_dt_params}, Recall na testu: {grid_results['test_metrics']['recall']}")

    # =========================================================================
    # 2. RandomizedSearchCV (Support Vector Machine - SVC)
    # =========================================================================
    print("\n--- [2/3] Spouštím RandomizedSearchCV (Support Vector Machine) ---")
    svc_param_dist = {
        "C": uniform(loc=0.01, scale=4.0),
        "gamma": uniform(loc=0.001, scale=0.1),
        "kernel": ["rbf", "linear"]
    }
    n_iter_rand = 20
    t0 = time.time()
    svc_model = SVC(random_state=42)
    random_search = RandomizedSearchCV(
        svc_model,
        param_distributions=svc_param_dist,
        n_iter=n_iter_rand,
        cv=5,
        scoring="precision",
        random_state=42,
        n_jobs=-1,
        return_train_score=True
    )
    random_search.fit(X_train, y_train)
    rand_time = time.time() - t0

    best_svc_params = random_search.best_params_
    best_svc_estimator = random_search.best_estimator_
    svc_test_preds = best_svc_estimator.predict(X_test)

    random_results = {
        "time_seconds": round(rand_time, 4),
        "n_iter": n_iter_rand,
        "best_cv_precision": round(float(random_search.best_score_), 4),
        "best_params": {
            "C": round(float(best_svc_params["C"]), 4),
            "gamma": round(float(best_svc_params["gamma"]), 4),
            "kernel": str(best_svc_params["kernel"])
        },
        "test_metrics": {
            "accuracy": round(float(accuracy_score(y_test, svc_test_preds)), 4),
            "precision": round(float(precision_score(y_test, svc_test_preds)), 4),
            "recall": round(float(recall_score(y_test, svc_test_preds)), 4),
            "f1": round(float(f1_score(y_test, svc_test_preds)), 4)
        },
        "confusion_matrix": confusion_matrix(y_test, svc_test_preds).tolist(),
        "trials_sampled": []
    }

    rand_cv_df = pd.DataFrame(random_search.cv_results_)
    for idx, row in rand_cv_df.iterrows():
        random_results["trials_sampled"].append({
            "iter": idx + 1,
            "C": round(float(row["params"]["C"]), 4),
            "gamma": round(float(row["params"]["gamma"]), 4),
            "kernel": str(row["params"]["kernel"]),
            "mean_test_precision": round(float(row["mean_test_score"]), 4)
        })

    results["random_search"] = random_results
    print(f"RandomizedSearchCV dokončen za {rand_time:.2f} s. Nejlepší parametry: {random_results['best_params']}, Precision na testu: {random_results['test_metrics']['precision']}")

    # =========================================================================
    # 3. Bayesian Search s knihovnou Hyperopt (DecisionTreeClassifier)
    # =========================================================================
    print("\n--- [3/3] Spouštím Bayesian Search (Hyperopt) ---")
    space = {
        "min_samples_split": hp.uniform("min_samples_split", 5, 25),
        "min_samples_leaf": hp.choice("min_samples_leaf", [5, 10, 15, 20, 25, 30]),
        "max_depth": hp.choice("max_depth", [None, 3, 5, 7, 10]),
        "criterion": hp.choice("criterion", ["gini", "entropy"])
    }

    t0 = time.time()
    def objective(params):
        clf = DecisionTreeClassifier(
            min_samples_split=int(params["min_samples_split"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_depth=params["max_depth"],
            criterion=params["criterion"],
            random_state=42
        )
        scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="recall", n_jobs=-1)
        # Hyperopt minimalizuje, proto vracíme záporné skóre
        return -scores.mean()

    trials = Trials()
    max_evals = 35
    best_raw = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        rstate=np.random.default_rng(42)
    )
    bayes_time = time.time() - t0
    best_bayes_params = space_eval(space, best_raw)

    best_bayes_dt = DecisionTreeClassifier(
        min_samples_split=int(best_bayes_params["min_samples_split"]),
        min_samples_leaf=int(best_bayes_params["min_samples_leaf"]),
        max_depth=best_bayes_params["max_depth"],
        criterion=best_bayes_params["criterion"],
        random_state=42
    )
    best_bayes_dt.fit(X_train, y_train)
    bayes_test_preds = best_bayes_dt.predict(X_test)

    # Sledování konvergence v čase
    history_losses = [-t["result"]["loss"] for t in trials.trials]
    cum_max = np.maximum.accumulate(history_losses)

    bayes_results = {
        "time_seconds": round(bayes_time, 4),
        "max_evals": max_evals,
        "best_cv_recall": round(float(max(history_losses)), 4),
        "best_params": {
            "min_samples_split": int(best_bayes_params["min_samples_split"]),
            "min_samples_leaf": int(best_bayes_params["min_samples_leaf"]),
            "max_depth": str(best_bayes_params["max_depth"]),
            "criterion": str(best_bayes_params["criterion"])
        },
        "test_metrics": {
            "accuracy": round(float(accuracy_score(y_test, bayes_test_preds)), 4),
            "precision": round(float(precision_score(y_test, bayes_test_preds)), 4),
            "recall": round(float(recall_score(y_test, bayes_test_preds)), 4),
            "f1": round(float(f1_score(y_test, bayes_test_preds)), 4)
        },
        "convergence_history": [round(float(x), 4) for x in history_losses],
        "cumulative_best": [round(float(x), 4) for x in cum_max]
    }
    results["bayesian_search"] = bayes_results
    print(f"Hyperopt dokončen za {bayes_time:.2f} s. Nejlepší parametry: {bayes_results['best_params']}, Recall na testu: {bayes_results['test_metrics']['recall']}")

    # =========================================================================
    # Uložení JSON cache
    # =========================================================================
    out_json = data_dir / "hyperparameter_optimization_sample_precomputed.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Uložena cache: {out_json}")

    # =========================================================================
    # Vykreslení a uložení diagnostických grafů
    # =========================================================================
    plt.style.use("seaborn-v0_8-whitegrid")

    # 1. GridSearchCV Heatmap (max_depth vs criterion)
    plt.figure(figsize=(10, 5))
    pivot_df = cv_res_df[cv_res_df["param_max_features"].isna()].pivot(
        index="param_max_depth", columns="param_criterion", values="mean_test_score"
    )
    sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="Blues", cbar_kws={"label": "5-Fold Mean Recall"})
    plt.title("GridSearchCV: Validační Recall pro DecisionTree (max_features=None)", fontsize=13, fontweight="bold")
    plt.xlabel("Criterion", fontsize=11)
    plt.ylabel("Max Depth", fontsize=11)
    plt.tight_layout()
    p1 = plots_dir / "hyperopt_grid_search_heatmap.png"
    plt.savefig(p1, dpi=200)
    plt.close()
    print(f"📈 Uložen graf: {p1}")

    # 2. RandomizedSearchCV Scatter (C vs gamma)
    plt.figure(figsize=(10, 5))
    scatter = plt.scatter(
        rand_cv_df["params"].apply(lambda p: p["C"]),
        rand_cv_df["params"].apply(lambda p: p["gamma"]),
        c=rand_cv_df["mean_test_score"],
        cmap="viridis",
        s=120,
        edgecolors="black"
    )
    plt.colorbar(scatter, label="5-Fold Mean Precision")
    plt.title("RandomizedSearchCV: Náhodné vzorky parametrů C a gamma pro SVC", fontsize=13, fontweight="bold")
    plt.xlabel("Parametr C (uniform [0.01, 4.01])", fontsize=11)
    plt.ylabel("Parametr gamma (uniform [0.001, 0.101])", fontsize=11)
    plt.tight_layout()
    p2 = plots_dir / "hyperopt_random_search_scatter.png"
    plt.savefig(p2, dpi=200)
    plt.close()
    print(f"📈 Uložen graf: {p2}")

    # 3. Hyperopt Convergence Curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, max_evals + 1), history_losses, marker="o", linestyle="--", color="gray", alpha=0.6, label="Iterační skóre (Recall)")
    plt.plot(range(1, max_evals + 1), cum_max, marker="s", linewidth=2.5, color="#d95f02", label="Dosud nejlepší hodnota (Cumulative Best)")
    plt.title("Hyperopt (Bayesian Optimization): Konvergence k optimu (TPE)", fontsize=13, fontweight="bold")
    plt.xlabel("Iterace pokusu (Trial index)", fontsize=11)
    plt.ylabel("Validační Recall (5-Fold CV)", fontsize=11)
    plt.legend(frameon=True)
    plt.tight_layout()
    p3 = plots_dir / "hyperopt_bayesian_convergence.png"
    plt.savefig(p3, dpi=200)
    plt.close()
    print(f"📈 Uložen graf: {p3}")

    print("\n✅ Všechny výpočty, cache i grafy byly úspěšně vygenerovány.")


if __name__ == "__main__":
    run_hyperparameter_optimization_sample()
