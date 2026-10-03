# ==============================================================================
# Homework: Logistická regrese na škálovaných datech diabetu (diabetes_scaled.csv)
# ==============================================================================
# Zadání:
# 1. Import potřebných knihoven pro logistickou regresi ze Scikit-learn.
# 2. Načtení škálovaného datasetu diabetes_scaled.csv.
# 3. Rozdělení na trénovací a testovací sadu v poměru 70/30 (stratifikovaně).
# 4. Vytvoření slovníku s hyperparametry (zejména inverzní síla regularizace C).
# 5. Vytvoření instance modelu LogisticRegression (lr).
# 6. Použití RandomizedSearchCV pro nalezení nejlepší sady hyperparametrů
#    (se stejnou optimalizační metrikou jako v cvičení 1: F1-score / Recall).
# 7. Trénování finálního modelu na parametrech vrácených z RandomizedSearchCV.
# 8. Otestování efektivity na testovací sadě (F1, Recall, Precision, Accuracy, ROC-AUC, matice záměn).
# ==============================================================================

import os
import json
import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve, precision_recall_curve
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "diabetes_scaled.csv")
JSON_OUTPUT_PATH = os.path.join(DATA_DIR, "diabetes_logistic_precomputed.json")

def run_diabetes_logistic_regression():
    print("=" * 80)
    print("KROK 1 & 2: Načtení škálovaných dat diabetes_scaled.csv")
    print("=" * 80)
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Soubor {CSV_PATH} neexistuje! Nejprve spusťte předzpracování.")

    df = pd.read_csv(CSV_PATH)
    print(f"Načteno: {df.shape[0]} řádků, {df.shape[1]} sloupců")
    
    feature_cols = [c for c in df.columns if c != 'outcome']
    X = df[feature_cols].values
    y = df['outcome'].values

    print("\n" + "=" * 80)
    print("KROK 3: Rozdělení dat na trénovací a testovací sadu v poměru 70/30")
    print("=" * 80)
    # Použijeme stejný random_state=42 jako v k-NN pro férové porovnání
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    print(f"Trénovací sada: {X_train.shape[0]} vzorků (0: {(y_train == 0).sum()}, 1: {(y_train == 1).sum()})")
    print(f"Testovací sada: {X_test.shape[0]} vzorků (0: {(y_test == 0).sum()}, 1: {(y_test == 1).sum()})")

    print("\n" + "=" * 80)
    print("KROK 4 & 5: Definice hyperparametrů (C) a instance LogisticRegression")
    print("=" * 80)
    # Hyperparametr C: inverzní síla regularizace v log-škále od 1e-4 do 1e3
    # Zahrneme i solver a penalty kompatibilní s regularizací
    param_distributions = {
        'C': loguniform(1e-4, 1e3),
        'penalty': ['l2', 'l1'],
        'solver': ['liblinear', 'saga'],
        'max_iter': [1000]
    }

    lr = LogisticRegression(random_state=42)

    print("\n" + "=" * 80)
    print("KROK 6: RandomizedSearchCV pro nalezení nejlepších hyperparametrů")
    print("=" * 80)
    # Zvolená metrika: stejná jako v cvičení 1 (F1-score / Recall pro medicínský kontext)
    # Porovnáme i běhy pro Recall, Accuracy a ROC-AUC
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scorings = ['f1', 'recall', 'accuracy', 'roc_auc']
    random_search_results = {}
    best_models = {}

    for sc in scorings:
        rs = RandomizedSearchCV(
            estimator=lr,
            param_distributions=param_distributions,
            n_iter=100,
            scoring=sc,
            cv=cv,
            random_state=42,
            n_jobs=-1,
            return_train_score=True
        )
        rs.fit(X_train, y_train)
        best_models[sc] = rs.best_estimator_
        
        # Extrakce bodů pro vizualizaci 100 náhodných pokusů
        trials = []
        for i in range(len(rs.cv_results_['params'])):
            trials.append({
                "trial": i + 1,
                "C": float(rs.cv_results_['params'][i]['C']),
                "penalty": str(rs.cv_results_['params'][i]['penalty']),
                "solver": str(rs.cv_results_['params'][i]['solver']),
                "mean_test_score": float(rs.cv_results_['mean_test_score'][i]),
                "std_test_score": float(rs.cv_results_['std_test_score'][i]),
                "mean_train_score": float(rs.cv_results_['mean_train_score'][i]),
            })

        random_search_results[sc] = {
            "best_params": {
                "C": float(rs.best_params_['C']),
                "penalty": str(rs.best_params_['penalty']),
                "solver": str(rs.best_params_['solver']),
                "max_iter": int(rs.best_params_['max_iter'])
            },
            "best_score": float(rs.best_score_),
            "trials": trials
        }
        print(f"Metrika [{sc.upper():8s}]: Nejlepší CV skóre = {rs.best_score_:.4f} | C = {rs.best_params_['C']:.5f} ({rs.best_params_['penalty']})")

    # Primární model: F1 (stejně jako u k-NN)
    primary_metric = 'f1'
    primary_lr = best_models[primary_metric]

    print("\n" + "=" * 80)
    print("KROK 7 & 8: Trénování a evaluace na testovací sadě")
    print("=" * 80)
    test_evaluations = {}

    for sc, model in best_models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = float(accuracy_score(y_test, y_pred))
        rec = float(recall_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred))
        auc = float(roc_auc_score(y_test, y_prob))
        cm = confusion_matrix(y_test, y_pred).tolist()

        test_evaluations[sc] = {
            "accuracy": acc,
            "recall": rec,
            "precision": prec,
            "f1": f1,
            "roc_auc": auc,
            "confusion_matrix": cm,
            "tn": cm[0][0],
            "fp": cm[0][1],
            "fn": cm[1][0],
            "tp": cm[1][1]
        }

        print(f"\nModel optimalizovaný na [{sc.upper()}]:")
        print(f"  Accuracy:  {acc * 100:.2f} %")
        print(f"  Recall:    {rec * 100:.2f} % ({cm[1][1]} z 81 zachyceno)")
        print(f"  Precision: {prec * 100:.2f} %")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {auc:.4f}")
        print(f"  Matice záměn: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")

    # Analýza vah koeficientů modelu (Feature Importance & Odds Ratios)
    coefs = primary_lr.coef_[0]
    intercept = float(primary_lr.intercept_[0])
    feature_importance = []
    for feat, coef in zip(feature_cols, coefs):
        odds_ratio = float(np.exp(coef))
        feature_importance.append({
            "feature": feat,
            "coefficient": float(coef),
            "odds_ratio": odds_ratio,
            "abs_coef": float(abs(coef)),
            "direction": "Zvyšuje riziko (+)" if coef > 0 else "Snižuje riziko (-)"
        })
    feature_importance = sorted(feature_importance, key=lambda x: x["abs_coef"], reverse=True)

    print("\nKoeficienty a Odds Ratios (Poměr šancí) pro primární model:")
    for fi in feature_importance:
        print(f"  - {fi['feature']:25s}: w = {fi['coefficient']:+.4f} | Odds Ratio = {fi['odds_ratio']:.3f}x ({fi['direction']})")

    # Křivka vlivu C na váhy (Regularization Path)
    c_sweep_values = np.logspace(-4, 3, 50)
    reg_path = []
    for c_val in c_sweep_values:
        clf_c = LogisticRegression(C=c_val, penalty='l2', solver='liblinear', random_state=42)
        clf_c.fit(X_train, y_train)
        y_te_pred = clf_c.predict(X_test)
        y_te_prob = clf_c.predict_proba(X_test)[:, 1]
        
        row_c = {
            "C": float(c_val),
            "log10_C": float(np.log10(c_val)),
            "test_accuracy": float(accuracy_score(y_test, y_te_pred)),
            "test_f1": float(f1_score(y_test, y_te_pred)),
            "test_recall": float(recall_score(y_test, y_te_pred)),
            "test_roc_auc": float(roc_auc_score(y_test, y_te_prob))
        }
        for f_name, w in zip(feature_cols, clf_c.coef_[0]):
            row_c[f"w_{f_name}"] = float(w)
        reg_path.append(row_c)

    # ROC křivka a Precision-Recall křivka
    primary_probs = primary_lr.predict_proba(X_test)[:, 1]
    fpr, tpr, roc_thresh = roc_curve(y_test, primary_probs)
    prec_pts, rec_pts, pr_thresh = precision_recall_curve(y_test, primary_probs)

    # Přímé porovnání s k-NN modelem z předchozího cvičení
    knn_cache_path = os.path.join(DATA_DIR, "diabetes_knn_precomputed.json")
    knn_comparison = None
    if os.path.exists(knn_cache_path):
        with open(knn_cache_path, "r", encoding="utf-8") as f:
            knn_data = json.load(f)
            knn_eval = knn_data["test_evaluations"]["f1"]
            knn_comparison = {
                "knn": {
                    "model_name": f"k-NN (k={knn_data['primary_best_params']['n_neighbors']})",
                    "accuracy": knn_eval["accuracy"],
                    "recall": knn_eval["recall"],
                    "precision": knn_eval["precision"],
                    "f1": knn_eval["f1"],
                    "roc_auc": knn_eval["roc_auc"],
                    "tn": knn_eval["tn"],
                    "fp": knn_eval["fp"],
                    "fn": knn_eval["fn"],
                    "tp": knn_eval["tp"]
                },
                "logreg": {
                    "model_name": f"Logistic Regression (C={random_search_results['f1']['best_params']['C']:.3f})",
                    "accuracy": test_evaluations["f1"]["accuracy"],
                    "recall": test_evaluations["f1"]["recall"],
                    "precision": test_evaluations["f1"]["precision"],
                    "f1": test_evaluations["f1"]["f1"],
                    "roc_auc": test_evaluations["f1"]["roc_auc"],
                    "tn": test_evaluations["f1"]["tn"],
                    "fp": test_evaluations["f1"]["fp"],
                    "fn": test_evaluations["f1"]["fn"],
                    "tp": test_evaluations["f1"]["tp"]
                }
            }

    # Uložení kompletních předpočtených výsledků do JSON
    precomputed = {
        "dataset_info": {
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "feature_names": feature_cols,
            "class_counts_test": {
                "healthy_0": int((y_test == 0).sum()),
                "diabetes_1": int((y_test == 1).sum())
            }
        },
        "random_search_results": random_search_results,
        "test_evaluations": test_evaluations,
        "primary_metric": primary_metric,
        "feature_importance": feature_importance,
        "intercept": intercept,
        "regularization_path": reg_path,
        "roc_curve": {
            "fpr": [float(x) for x in fpr],
            "tpr": [float(x) for x in tpr],
            "thresholds": [float(x) for x in roc_thresh]
        },
        "pr_curve": {
            "precision": [float(x) for x in prec_pts],
            "recall": [float(x) for x in rec_pts],
            "thresholds": [float(x) for x in pr_thresh]
        },
        "model_comparison": knn_comparison
    }

    with open(JSON_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(precomputed, f, ensure_ascii=False, indent=2)

    print(f"\nVýsledky úspěšně uloženy do: {JSON_OUTPUT_PATH}")
    return precomputed

if __name__ == "__main__":
    run_diabetes_logistic_regression()
