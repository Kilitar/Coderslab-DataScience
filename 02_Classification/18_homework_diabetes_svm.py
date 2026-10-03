# ==============================================================================
# Homework: Support Vector Machine (SVM) s Bayesovskou optimalizací (Hyperopt)
# ==============================================================================
# Zadání:
# 1. Import knihoven pro SVM (SVC) ze Scikit-learn a Hyperopt (hp, fmin, tpe, Trials).
# 2. Načtení škálovaného datasetu diabetes_scaled.csv.
# 3. Rozdělení na trénovací a testovací sadu v poměru 70/30 (stratifikovaně).
# 4. Vytvoření slovníku hyperparametrů pro Hyperopt (kernel, gamma, C).
# 5. Definice účelové funkce (objective function) pro křížovou validaci a metriku F1/Recall.
# 6. Spuštění Bayesovské optimalizace (TPE) přes fmin.
# 7. Trénování finálního modelu s nejlepšími parametry.
# 8. Otestování na testovací sadě.
# 9. Zodpovězení otázky: Který ze 3 modelů (k-NN, LogReg, SVM) dosáhl nejlepšího výkonu?
# ==============================================================================

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "diabetes_scaled.csv")
JSON_OUTPUT_PATH = os.path.join(DATA_DIR, "diabetes_svm_precomputed.json")

def run_diabetes_svm():
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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    print(f"Trénovací sada: {X_train.shape[0]} vzorků (0: {(y_train == 0).sum()}, 1: {(y_train == 1).sum()})")
    print(f"Testovací sada: {X_test.shape[0]} vzorků (0: {(y_test == 0).sum()}, 1: {(y_test == 1).sum()})")

    print("\n" + "=" * 80)
    print("KROK 4 & 5: Definice hyperparametrického prostoru a objective funkce (Hyperopt)")
    print("=" * 80)
    kernel_options = ['linear', 'rbf', 'poly', 'sigmoid']
    space = {
        'kernel': hp.choice('kernel', kernel_options),
        'C': hp.loguniform('C', np.log(1e-3), np.log(1e3)),
        'gamma': hp.loguniform('gamma', np.log(1e-4), np.log(1e1)),
        'degree': hp.choice('degree', [2, 3, 4])
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    history_trials = []

    def objective(params):
        clf = SVC(
            kernel=params['kernel'],
            C=params['C'],
            gamma=params['gamma'],
            degree=params['degree'],
            probability=True,
            random_state=42
        )
        # Stejná metrika jako v cvičení 1 a 2: F1-score
        scores = cross_val_score(clf, X_train, y_train, scoring='f1', cv=cv, n_jobs=-1)
        mean_f1 = float(np.mean(scores))
        std_f1 = float(np.std(scores))
        
        # Hyperopt minimalizuje loss, proto minimalizujeme -mean_f1
        loss = -mean_f1

        history_trials.append({
            "kernel": params['kernel'],
            "C": float(params['C']),
            "gamma": float(params['gamma']),
            "degree": int(params['degree']),
            "f1": mean_f1,
            "std": std_f1,
            "loss": loss
        })

        return {
            'loss': loss,
            'status': STATUS_OK,
            'f1': mean_f1,
            'std': std_f1
        }

    print("\n" + "=" * 80)
    print("KROK 6: Bayesovská optimalizace (Tree of Parzen Estimators - TPE)")
    print("=" * 80)
    trials = Trials()
    best_raw = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials,
        rstate=np.random.default_rng(42)
    )

    # Převod indexů hp.choice na konkrétní hodnoty
    best_params = {
        'kernel': kernel_options[best_raw['kernel']],
        'C': float(best_raw['C']),
        'gamma': float(best_raw['gamma']),
        'degree': [2, 3, 4][best_raw['degree']]
    }
    best_cv_f1 = -trials.best_trial['result']['loss']
    print(f"Bayesovská optimalizace dokončena (100 iterací).")
    print(f"Nejlepší F1 skóre z 5-Fold CV: {best_cv_f1:.4f}")
    print(f"Optimální parametry: {best_params}")

    print("\n" + "=" * 80)
    print("KROK 7 & 8: Trénování modelu na nejlepších parametrech a testování")
    print("=" * 80)
    best_svm = SVC(
        kernel=best_params['kernel'],
        C=best_params['C'],
        gamma=best_params['gamma'],
        degree=best_params['degree'],
        probability=True,
        random_state=42
    )
    best_svm.fit(X_train, y_train)

    y_pred = best_svm.predict(X_test)
    y_prob = best_svm.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    rec = float(recall_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    auc = float(roc_auc_score(y_test, y_prob))
    cm = confusion_matrix(y_test, y_pred).tolist()

    n_support_vectors = int(np.sum(best_svm.n_support_))
    support_per_class = [int(x) for x in best_svm.n_support_]

    print("=" * 60)
    print("VÝSLEDKY OPTIMALIZOVANÉHO SVM NA TESTOVACÍ SADĚ:")
    print("=" * 60)
    print(f"  Accuracy:  {acc * 100:.2f} %")
    print(f"  Recall:    {rec * 100:.2f} % (Zachyceno {cm[1][1]} z 81 diabetiček)")
    print(f"  Precision: {prec * 100:.2f} %")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {auc:.4f}")
    print(f"  Počet podpůrných vektorů: {n_support_vectors} ({support_per_class[0]} zdravých, {support_per_class[1]} diabetiček)")
    print(f"  Matice záměn: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")

    # ROC křivka a Precision-Recall křivka pro SVM
    fpr, tpr, roc_thresh = roc_curve(y_test, y_prob)
    prec_pts, rec_pts, pr_thresh = precision_recall_curve(y_test, y_prob)

    # Analýza konvergence TPE (kumulativní minimum loss)
    losses = [t['loss'] for t in history_trials]
    cum_best = []
    current_best = float('inf')
    for l in losses:
        if l < current_best:
            current_best = l
        cum_best.append(-current_best)

    convergence_data = [
        {"iteration": i + 1, "trial_f1": history_trials[i]["f1"], "best_f1": cum_best[i]}
        for i in range(len(history_trials))
    ]

    print("\n" + "=" * 80)
    print("KROK 9: VELKÉ SROVNÁNÍ VŠECH TŘÍ MODELŮ (k-NN vs. LogReg vs. SVM)")
    print("=" * 80)
    # Načteme předchozí výsledky
    knn_cache = os.path.join(DATA_DIR, "diabetes_knn_precomputed.json")
    lr_cache = os.path.join(DATA_DIR, "diabetes_logistic_precomputed.json")

    all_models_comparison = []

    # 1. k-NN
    if os.path.exists(knn_cache):
        with open(knn_cache, "r", encoding="utf-8") as f:
            k_d = json.load(f)
            k_eval = k_d["test_evaluations"]["f1"]
            all_models_comparison.append({
                "model_name": f"1. k-NN (k={k_d['primary_best_params']['n_neighbors']}, {k_d['primary_best_params']['metric']})",
                "family": "k-Nearest Neighbors",
                "accuracy": k_eval["accuracy"],
                "recall": k_eval["recall"],
                "precision": k_eval["precision"],
                "f1": k_eval["f1"],
                "roc_auc": k_eval["roc_auc"],
                "tn": k_eval["tn"],
                "fp": k_eval["fp"],
                "fn": k_eval["fn"],
                "tp": k_eval["tp"],
                "pros": "Jednoduchý, neparametrický, nevyžaduje linearitu",
                "cons": "Pomalý při velké inferenci, citlivý na šum, nekalibrované pravděpodobnosti"
            })

    # 2. Logistic Regression
    if os.path.exists(lr_cache):
        with open(lr_cache, "r", encoding="utf-8") as f:
            l_d = json.load(f)
            l_eval = l_d["test_evaluations"]["f1"]
            all_models_comparison.append({
                "model_name": f"2. Logistická regrese (C={l_d['random_search_results']['f1']['best_params']['C']:.3f}, {l_d['random_search_results']['f1']['best_params']['penalty'].upper()})",
                "family": "Logistic Regression",
                "accuracy": l_eval["accuracy"],
                "recall": l_eval["recall"],
                "precision": l_eval["precision"],
                "f1": l_eval["f1"],
                "roc_auc": l_eval["roc_auc"],
                "tn": l_eval["tn"],
                "fp": l_eval["fp"],
                "fn": l_eval["fn"],
                "tp": l_eval["tp"],
                "pros": "Vynikající interpretovatelnost (Odds Ratios), přímé pravděpodobnosti, rychlý běh",
                "cons": "Předpokládá lineární hranici na logit škále, citlivá na kolinearitu"
            })

    # 3. SVM
    all_models_comparison.append({
        "model_name": f"3. Support Vector Machine ({best_params['kernel'].upper()}, C={best_params['C']:.2f}, γ={best_params['gamma']:.4f})",
        "family": "Support Vector Machines",
        "accuracy": acc,
        "recall": rec,
        "precision": prec,
        "f1": f1,
        "roc_auc": auc,
        "tn": cm[0][0],
        "fp": cm[0][1],
        "fn": cm[1][0],
        "tp": cm[1][1],
        "pros": "Robustní vícerozměrná nadrovina, maximalizace okraje (margin), vysoká přesnost",
        "cons": "Černá skříňka při nelineárním jádře, citlivá na ladění C a gamma"
    })

    print("\nSouhrnná tabulka výkonu na testovací sadě (231 pacientek):")
    for m in all_models_comparison:
        print(f"\n{m['model_name']}:")
        print(f"  Accuracy:  {m['accuracy'] * 100:.2f} %")
        print(f"  Recall:    {m['recall'] * 100:.2f} % ({m['tp']} z 81 zachyceno, {m['fn']} uniklo)")
        print(f"  Precision: {m['precision'] * 100:.2f} %")
        print(f"  F1-Score:  {m['f1']:.4f}")
        print(f"  ROC-AUC:   {m['roc_auc']:.4f}")

    # Uložení předpočtených výsledků do JSON pro bleskové a interaktivní zobrazení v aplikaci
    precomputed = {
        "dataset_info": {
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "feature_names": feature_cols
        },
        "best_params": best_params,
        "best_cv_f1": best_cv_f1,
        "test_evaluation": {
            "accuracy": acc,
            "recall": rec,
            "precision": prec,
            "f1": f1,
            "roc_auc": auc,
            "confusion_matrix": cm,
            "tn": cm[0][0],
            "fp": cm[0][1],
            "fn": cm[1][0],
            "tp": cm[1][1],
            "n_support_vectors": n_support_vectors,
            "support_per_class": support_per_class
        },
        "trials_history": history_trials,
        "convergence": convergence_data,
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
        "all_models_comparison": all_models_comparison
    }

    with open(JSON_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(precomputed, f, ensure_ascii=False, indent=2)

    print(f"\nVýsledky úspěšně uloženy do: {JSON_OUTPUT_PATH}")
    return precomputed

if __name__ == "__main__":
    run_diabetes_svm()
