# ==============================================================================
# Homework: K-Nearest Neighbors na škálovaných datech diabetu (diabetes_scaled.csv)
# ==============================================================================
# Úkol:
# 1. Import knihoven pro k-NN a evaluaci ze Scikit-learn.
# 2. Načtení škálovaných dat diabetes_scaled.csv.
# 3. Rozdělení na trénovací a testovací sadu v poměru 70/30 (se stratifikací).
# 4. Definice mřížky hyperparametrů (n_neighbors, metric).
# 5. Vytvoření instance KNeighborsClassifier (knn).
# 6. Hledání optimálních hyperparametrů pomocí GridSearchCV (diskuse metriky: Recall vs F1 vs Accuracy).
# 7. Trénování finálního modelu na nejlepších parametrech.
# 8. Otestování efektivity na testovací sadě (Accuracy, Recall, Precision, F1, ROC-AUC, matice záměn).
# ==============================================================================

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve, precision_recall_curve
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "diabetes_scaled.csv")
JSON_OUTPUT_PATH = os.path.join(DATA_DIR, "diabetes_knn_precomputed.json")

def run_diabetes_knn():
    print("=" * 80)
    print("KROK 1 & 2: Načtení škálovaného datasetu diabetes_scaled.csv")
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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    print(f"Trénovací sada: {X_train.shape[0]} vzorků (Outcome 0: {(y_train == 0).sum()}, Outcome 1: {(y_train == 1).sum()})")
    print(f"Testovací sada: {X_test.shape[0]} vzorků (Outcome 0: {(y_test == 0).sum()}, Outcome 1: {(y_test == 1).sum()})")

    print("\n" + "=" * 80)
    print("KROK 4, 5 & 6: Definice hyperparametrů a GridSearchCV")
    print("=" * 80)
    # Hyperparametry: n_neighbors od 1 do 35, metriky euclidean, manhattan, minkowski (p=3)
    param_grid = {
        'n_neighbors': list(range(1, 36)),
        'metric': ['euclidean', 'manhattan', 'minkowski'],
        'weights': ['uniform', 'distance']
    }

    # Provedeme GridSearchCV s různými optimalizačními metrikami pro srovnání
    # Z medicínského hlediska:
    # 1. Recall: minimalizuje falešně negativní pacienty (diabetik není poslán domů bez péče)
    # 2. F1-score: harmonický průměr Recall a Precision
    # 3. ROC-AUC: schopnost separace tříd napříč prahy
    # 4. Accuracy: celková přesnost (učebnicový default)

    scorings = ['recall', 'f1', 'roc_auc', 'accuracy']
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_results = {}
    best_models = {}

    for sc in scorings:
        grid = GridSearchCV(
            estimator=KNeighborsClassifier(),
            param_grid=param_grid,
            scoring=sc,
            cv=cv,
            n_jobs=-1,
            return_train_score=True
        )
        grid.fit(X_train, y_train)
        best_models[sc] = grid.best_estimator_
        
        grid_results[sc] = {
            "best_params": grid.best_params_,
            "best_score": float(grid.best_score_),
            "cv_results": {
                "params": grid.cv_results_['params'],
                "mean_test_score": [float(v) for v in grid.cv_results_['mean_test_score']],
                "std_test_score": [float(v) for v in grid.cv_results_['std_test_score']],
                "mean_train_score": [float(v) for v in grid.cv_results_['mean_train_score']],
            }
        }
        print(f"Metrika [{sc.upper():8s}]: Nejlepší skóre CV = {grid.best_score_:.4f} | Parametry: {grid.best_params_}")

    # Primární model zvolený dle medicínské závažnosti (F1-score / Recall)
    # Standardní zadání kurzu volí F1 nebo Recall pro vyvážení
    primary_metric = 'f1'
    primary_knn = best_models[primary_metric]
    print(f"\nPrimárně zvolený optimalizační model pro evaluaci: {primary_metric.upper()} ({grid_results[primary_metric]['best_params']})")

    print("\n" + "=" * 80)
    print("KROK 7 & 8: Evaluace na testovací sadě (Test Set)")
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
        print(f"  Recall:    {rec * 100:.2f} % (Zachyceno {cm[1][1]} z {cm[1][0] + cm[1][1]} diabetiků)")
        print(f"  Precision: {prec * 100:.2f} %")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {auc:.4f}")
        print(f"  Matice záměn: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")

    # Podrobný průběh n_neighbors od 1 do 35 pro různé metriky vzdálenosti (weights='uniform')
    sweep_data = []
    for metric_name in ['euclidean', 'manhattan', 'minkowski']:
        for k in range(1, 36):
            clf = KNeighborsClassifier(n_neighbors=k, metric=metric_name, weights='uniform')
            clf.fit(X_train, y_train)
            
            y_tr_pred = clf.predict(X_train)
            y_te_pred = clf.predict(X_test)
            y_te_prob = clf.predict_proba(X_test)[:, 1]

            sweep_data.append({
                "metric": metric_name,
                "k": k,
                "train_acc": float(accuracy_score(y_train, y_tr_pred)),
                "test_acc": float(accuracy_score(y_test, y_te_pred)),
                "test_recall": float(recall_score(y_test, y_te_pred)),
                "test_precision": float(precision_score(y_test, y_te_pred, zero_division=0)),
                "test_f1": float(f1_score(y_test, y_te_pred)),
                "test_roc_auc": float(roc_auc_score(y_test, y_te_prob))
            })

    # ROC křivka a Precision-Recall křivka pro primární model
    primary_probs = primary_knn.predict_proba(X_test)[:, 1]
    fpr, tpr, roc_thresholds = roc_curve(y_test, primary_probs)
    prec_pts, rec_pts, pr_thresholds = precision_recall_curve(y_test, primary_probs)

    roc_curve_data = {
        "fpr": [float(x) for x in fpr],
        "tpr": [float(x) for x in tpr],
        "thresholds": [float(x) for x in roc_thresholds]
    }
    pr_curve_data = {
        "precision": [float(x) for x in prec_pts],
        "recall": [float(x) for x in rec_pts],
        "thresholds": [float(x) for x in pr_thresholds]
    }

    # Uložení kompletní analýzy do lehkého JSONu pro bleskový běh UI
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
        "grid_results": grid_results,
        "test_evaluations": test_evaluations,
        "k_sweep": sweep_data,
        "roc_curve": roc_curve_data,
        "pr_curve": pr_curve_data,
        "primary_metric": primary_metric,
        "primary_best_params": grid_results[primary_metric]["best_params"]
    }

    with open(JSON_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(precomputed, f, ensure_ascii=False, indent=2)

    print(f"\nVýsledky úspěšně uloženy do: {JSON_OUTPUT_PATH}")
    return precomputed

if __name__ == "__main__":
    run_diabetes_knn()
