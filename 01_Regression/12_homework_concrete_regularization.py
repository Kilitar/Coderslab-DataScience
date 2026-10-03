"""
Homework: Lineární regrese s regularizací – Pevnost betonu (Concrete Compressive Strength)
========================================================================================
Tento skript řeší zadání 'Linear regression with regularization - exercise':
Zadání obsahuje zřejmý metodický rozpor:
- Nadpis: 'Linear regression with regularization - exercise'
- Dataset: 'concrete_data_preprocessed.csv' (spojitý cíl csMPa)
- Metrika: 'Choose the same metric as in exercise 1' (ve cvičení 1 to byly R2 a RMSE)
- Text v těle však uvádí: 'LogisticRegression', 'C and penalty', 'classifier'.

Skript proto implementuje obě komplementární roviny:
1. PRIMÁRNÍ REGRESNÍ ŘEŠENÍ (Intended Task):
   - Regularizovaná lineární regrese (ElasticNet pokrývající L1 Lasso i L2 Ridge + porovnání s Ridge a Lasso).
   - Ladění hyperparametrů (alpha a l1_ratio) pomocí RandomizedSearchCV.
   - Vyhodnocení stejnými regresními metrikami jako ve cvičení 1 (R2, RMSE, MAE).
   - Analýza smrštění koeficientů oproti neomezenému OLS z cvičení 1.

2. DOSLOVNÉ KLASIFIKAČNÍ ŘEŠENÍ (Literal LogisticRegression):
   - Binarizace pevnosti betonu (csMPa >= 35 MPa, norma pro vysoce pevný konstrukční beton).
   - LogisticRegression s parametry C a penalty ('l1', 'l2') laděnými přes RandomizedSearchCV.
   - Vyhodnocení klasifikačními metrikami (Accuracy, Precision, Recall, F1, Matice záměn).
"""

import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import loguniform, uniform
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.metrics import (
    r2_score,
    root_mean_squared_error,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

warnings.filterwarnings("ignore")

# Cesty
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "plots"
DATA_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "concrete_data_preprocessed.csv"
PRECOMPUTED_JSON_PATH = DATA_DIR / "concrete_regularization_precomputed.json"

FEATURE_NAMES = [
    "cement", "slag", "flyash", "water",
    "superplasticizer", "coarseaggregate", "fineaggregate", "age"
]
TARGET_NAME = "csMPa"

FEATURE_DESCRIPTIONS = {
    "cement": "Obsah cementu (kg/m³)",
    "slag": "Obsah vysokopecní strusky (kg/m³)",
    "flyash": "Obsah popílku (kg/m³)",
    "water": "Obsah záměsové vody (kg/m³)",
    "superplasticizer": "Obsah superplastifikátoru (kg/m³)",
    "coarseaggregate": "Obsah hrubého kameniva / štěrku (kg/m³)",
    "fineaggregate": "Obsah jemného kameniva / písku (kg/m³)",
    "age": "Doba zrání betonu (dny)"
}


def run_regularized_regression(X_train, X_test, y_train, y_test):
    print("=" * 70)
    print("1. Primární regresní řešení: ElasticNet (L1 + L2) via RandomizedSearchCV")
    print("=" * 70)
    
    # 1. Baseline neomezený OLS z cvičení 1 pro porovnání
    ols = LinearRegression()
    ols.fit(X_train, y_train)
    y_pred_ols = ols.predict(X_test)
    ols_r2 = r2_score(y_test, y_pred_ols)
    ols_rmse = root_mean_squared_error(y_test, y_pred_ols)
    
    # 2. Hyperparametrický prostor pro ElasticNet
    param_dist_en = {
        "alpha": loguniform(1e-4, 1e2),
        "l1_ratio": uniform(0.0, 1.0)
    }
    
    rs_en = RandomizedSearchCV(
        estimator=ElasticNet(random_state=42, max_iter=10000),
        param_distributions=param_dist_en,
        n_iter=60,
        cv=5,
        scoring="r2",
        random_state=42,
        n_jobs=-1
    )
    rs_en.fit(X_train, y_train)
    
    best_params_en = rs_en.best_params_
    best_model_en = rs_en.best_estimator_
    y_pred_en_train = best_model_en.predict(X_train)
    y_pred_en_test = best_model_en.predict(X_test)
    
    # 3. Ridge a Lasso pro srovnání
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train, y_train)
    
    lasso = Lasso(alpha=0.1, random_state=42)
    lasso.fit(X_train, y_train)
    
    reg_metrics = {
        "ols": {
            "r2": float(round(ols_r2, 4)),
            "rmse": float(round(ols_rmse, 4)),
            "mae": float(round(mean_absolute_error(y_test, y_pred_ols), 4))
        },
        "elastic_net": {
            "best_params": {
                "alpha": float(round(best_params_en["alpha"], 6)),
                "l1_ratio": float(round(best_params_en["l1_ratio"], 4))
            },
            "train": {
                "r2": float(round(r2_score(y_train, y_pred_en_train), 4)),
                "rmse": float(round(root_mean_squared_error(y_train, y_pred_en_train), 4)),
                "mae": float(round(mean_absolute_error(y_train, y_pred_en_train), 4))
            },
            "test": {
                "r2": float(round(r2_score(y_test, y_pred_en_test), 4)),
                "rmse": float(round(root_mean_squared_error(y_test, y_pred_en_test), 4)),
                "mae": float(round(mean_absolute_error(y_test, y_pred_en_test), 4))
            }
        },
        "ridge": {
            "test_r2": float(round(r2_score(y_test, ridge.predict(X_test)), 4)),
            "test_rmse": float(round(root_mean_squared_error(y_test, ridge.predict(X_test)), 4))
        },
        "lasso": {
            "test_r2": float(round(r2_score(y_test, lasso.predict(X_test)), 4)),
            "test_rmse": float(round(root_mean_squared_error(y_test, lasso.predict(X_test)), 4))
        }
    }
    
    # Srovnání koeficientů
    coef_comparison = {
        feat: {
            "ols": float(round(ols.coef_[i], 4)),
            "elastic_net": float(round(best_model_en.coef_[i], 4)),
            "ridge": float(round(ridge.coef_[i], 4)),
            "lasso": float(round(lasso.coef_[i], 4))
        }
        for i, feat in enumerate(FEATURE_NAMES)
    }
    
    print(f"Nejlepší hyperparametry ElasticNet: {best_params_en}")
    print(f"OLS Test R2:        {ols_r2:.4f}, RMSE: {ols_rmse:.4f} MPa")
    print(f"ElasticNet Test R2: {reg_metrics['elastic_net']['test']['r2']:.4f}, RMSE: {reg_metrics['elastic_net']['test']['rmse']:.4f} MPa")
    
    return rs_en, best_model_en, reg_metrics, coef_comparison


def run_logistic_classification(X_train, X_test, y_train_cont, y_test_cont):
    print("\n" + "=" * 70)
    print("2. Doslovné klasifikační řešení: LogisticRegression(C, penalty) na betonu")
    print("=" * 70)
    
    # Binarizace cíle: >= 35 MPa (Vysokopevnostní beton)
    threshold = 35.0
    y_train_bin = (y_train_cont >= threshold).astype(int)
    y_test_bin = (y_test_cont >= threshold).astype(int)
    
    print(f"Podíl třídy 1 (>= {threshold} MPa) v Train: {y_train_bin.mean()*100:.1f} %")
    print(f"Podíl třídy 1 (>= {threshold} MPa) v Test:  {y_test_bin.mean()*100:.1f} %")
    
    # Hyperparametry C a penalty dle zadání
    param_dist_lr = {
        "C": loguniform(1e-3, 1e2),
        "penalty": ["l1", "l2"]
    }
    
    # Použijeme solver saga, který podporuje jak L1 tak L2 penalizaci
    lr = LogisticRegression(solver="saga", random_state=42, max_iter=10000)
    
    rs_lr = RandomizedSearchCV(
        estimator=lr,
        param_distributions=param_dist_lr,
        n_iter=60,
        cv=5,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1
    )
    rs_lr.fit(X_train, y_train_bin)
    
    best_params_lr = rs_lr.best_params_
    best_model_lr = rs_lr.best_estimator_
    y_pred_lr = best_model_lr.predict(X_test)
    
    cm = confusion_matrix(y_test_bin, y_pred_lr)
    
    clf_metrics = {
        "threshold_mpa": threshold,
        "best_params": {
            "C": float(round(best_params_lr["C"], 6)),
            "penalty": str(best_params_lr["penalty"])
        },
        "test": {
            "accuracy": float(round(accuracy_score(y_test_bin, y_pred_lr), 4)),
            "precision": float(round(precision_score(y_test_bin, y_pred_lr), 4)),
            "recall": float(round(recall_score(y_test_bin, y_pred_lr), 4)),
            "f1": float(round(f1_score(y_test_bin, y_pred_lr), 4)),
            "confusion_matrix": cm.tolist()
        },
        "coefficients": {
            feat: float(round(best_model_lr.coef_[0][i], 4))
            for i, feat in enumerate(FEATURE_NAMES)
        },
        "intercept": float(round(best_model_lr.intercept_[0], 4))
    }
    
    print(f"Nejlepší parametry LogisticRegression: {best_params_lr}")
    print(f"Test Accuracy: {clf_metrics['test']['accuracy']:.4f} ({clf_metrics['test']['accuracy']*100:.2f} %)")
    print(f"Test F1:       {clf_metrics['test']['f1']:.4f}")
    print(f"Matice záměn:\n{cm}")
    
    return rs_lr, best_model_lr, clf_metrics


def generate_plots(rs_en, coef_comparison, clf_metrics):
    print("\n" + "=" * 70)
    print("3. Generování diagnostických PNG grafů")
    print("=" * 70)
    
    # 1. Graf rozdělení náhodného vzorkování ElasticNet (alpha vs l1_ratio)
    cv_res = pd.DataFrame(rs_en.cv_results_)
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        cv_res["param_alpha"],
        cv_res["param_l1_ratio"],
        c=cv_res["mean_test_score"],
        cmap="viridis",
        s=50,
        alpha=0.85,
        edgecolors="k"
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Validační R² skóre (5-fold CV)", fontsize=10)
    ax.set_xscale("log")
    ax.set_title("RandomizedSearchCV: Prohledávání hyperparametrů ElasticNet\n(alpha vs. l1_ratio)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Regularizační parametr alpha (log scale)", fontsize=10)
    ax.set_ylabel("Poměr penalizace l1_ratio (0 = Ridge L2, 1 = Lasso L1)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    p1 = PLOTS_DIR / "concrete_reg_alpha_tuning.png"
    fig.savefig(p1, dpi=200)
    plt.close(fig)
    print(f"Uložen graf: {p1}")
    
    # 2. Srovnání koeficientů: OLS vs ElasticNet vs Ridge vs Lasso
    coef_df = pd.DataFrame(coef_comparison).T
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(FEATURE_NAMES))
    width = 0.2
    
    ax.bar(x - 1.5 * width, coef_df["ols"], width, label="Neomezený OLS", color="#1f77b4", alpha=0.8)
    ax.bar(x - 0.5 * width, coef_df["ridge"], width, label="Ridge (L2, a=1)", color="#ff7f0e", alpha=0.8)
    ax.bar(x + 0.5 * width, coef_df["lasso"], width, label="Lasso (L1, a=0.1)", color="#2ca02c", alpha=0.8)
    ax.bar(x + 1.5 * width, coef_df["elastic_net"], width, label="ElasticNet (Opt)", color="#d62728", alpha=0.85)
    
    ax.axhline(0, color="k", linestyle="-", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(FEATURE_NAMES, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Regresní koeficient β (MPa / 1σ)", fontsize=10)
    ax.set_title("Vliv regularizace na regresní koeficienty: OLS vs. Ridge vs. Lasso vs. ElasticNet", fontsize=11, fontweight="bold")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    p2 = PLOTS_DIR / "concrete_reg_coefficients_comparison.png"
    fig.savefig(p2, dpi=200)
    plt.close(fig)
    print(f"Uložen graf: {p2}")
    
    # 3. Matice záměn pro LogisticRegression
    cm = np.array(clf_metrics["test"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Standardní (<35 MPa)", "Vysokopevnostní (≥35 MPa)"],
        yticklabels=["Standardní (<35 MPa)", "Vysokopevnostní (≥35 MPa)"],
        ax=ax
    )
    ax.set_title(f"Matice záměn: LogisticRegression na betonu\nAccuracy = {clf_metrics['test']['accuracy']*100:.1f} %, F1 = {clf_metrics['test']['f1']:.3f}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predikovaná třída")
    ax.set_ylabel("Skutečná třída")
    plt.tight_layout()
    p3 = PLOTS_DIR / "concrete_logistic_cm.png"
    fig.savefig(p3, dpi=200)
    plt.close(fig)
    print(f"Uložen graf: {p3}")


def main():
    df = pd.read_csv(CSV_PATH)
    X = df[FEATURE_NAMES]
    y = df[TARGET_NAME]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 1. Regresní řešení
    rs_en, best_model_en, reg_metrics, coef_comparison = run_regularized_regression(
        X_train, X_test, y_train, y_test
    )
    
    # 2. Klasifikační řešení
    rs_lr, best_model_lr, clf_metrics = run_logistic_classification(
        X_train, X_test, y_train, y_test
    )
    
    # 3. Grafy
    generate_plots(rs_en, coef_comparison, clf_metrics)
    
    # Uložení do JSON
    precomputed = {
        "metadata": {
            "task": "Homework: Linear regression with regularization (Concrete)",
            "conflict_explanation": (
                "Zadání obsahuje copy-paste rozpor mezi titulkem (Linear regression with regularization) "
                "a tělem textu (LogisticRegression, C, penalty, classifier). "
                "Poskytujeme primární regresní řešení (ElasticNet pokrývající L1/L2) "
                "i doslovné klasifikační řešení (LogisticRegression na binarizované pevnosti csMPa >= 35 MPa)."
            ),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "features": FEATURE_NAMES,
            "feature_descriptions": FEATURE_DESCRIPTIONS
        },
        "regression_solution": {
            "metrics": reg_metrics,
            "coefficients": coef_comparison
        },
        "classification_solution": clf_metrics
    }
    
    with open(PRECOMPUTED_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(precomputed, f, indent=2, ensure_ascii=False)
    print(f"\nUložena JSON cache: {PRECOMPUTED_JSON_PATH}")
    print("HOTOVO!")


if __name__ == "__main__":
    main()
