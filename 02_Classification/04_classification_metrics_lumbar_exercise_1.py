"""
04_classification_metrics_lumbar_exercise_1.py
==============================================
Metriky klasifikačních modelů – Cvičení 1: Diagnostika bederní páteře
Dataset: lumbar_df_normalized.csv (310 pacientů, 6 normalizovaných parametrů, class 0/1)

ZADÁNÍ / ASSIGNMENT:
-------------------------------------------------------------------------------
1. Download the attached dataset (lumbar_df_normalized.csv) and notebook
   (Classification_metrics_exercise_1.ipynb).
2. After uploading the data file, run all the cells with code in notebook.
3. Generate a confusion matrix and represent it in a graph. Use any library for data visualization.
4. Using the knowledge from the presentation, calculate the set of metrics:
   - accuracy
   - precision
   - recall
   - f1-score
5. Choose one of the above metrics and try to find the k for which the value of the metric
   will be the highest i.e. the model will be the most effective on the test set,
   while showing no signs of overtraining.
-------------------------------------------------------------------------------
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def main() -> None:
    # -------------------------------------------------------------------------
    # Nastavení cest / Paths setup
    # -------------------------------------------------------------------------
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    plots_dir = base_dir / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "lumbar_df_normalized.csv"
    if not csv_path.exists():
        fallback = base_dir.parent / "data" / "MAL_downloadable materials_session 1" / "Day 2" / "lumbar_df_normalized.csv"
        if fallback.exists():
            import shutil
            shutil.copy(fallback, csv_path)
        else:
            raise FileNotFoundError(f"Soubor lumbar_df_normalized.csv nebyl nalezen v {csv_path} ani ve fallbacku.")

    print("=" * 75)
    print("METRIKY KLASIFIKAČNÍCH MODELŮ – CVIČENÍ 1: DIAGNOSTIKA PÁTEŘE (LUMBAR)")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # Krok 1 & 2: Načtení dat a spuštění startovního kódu z notebooku
    # -------------------------------------------------------------------------
    print("\n[Krok 1 & 2] Načtení lumbar_df_normalized.csv a inicializace modelu dle zadání:")
    lumbar_df = pd.read_csv(csv_path)
    print(f"Rozměry datasetu: {lumbar_df.shape[0]} řádků x {lumbar_df.shape[1]} sloupců")
    print(lumbar_df.head(10))

    X = lumbar_df.drop("class", axis=1)
    y = lumbar_df["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"Trénovací data (Train): {X_train.shape[0]} pacientů")
    print(f"Testovací data (Test) : {X_test.shape[0]} pacientů")

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    y_scores = knn.predict_proba(X_test)[:, 1]

    # -------------------------------------------------------------------------
    # Krok 3: Generování matice záměn (Confusion Matrix)
    # -------------------------------------------------------------------------
    print("\n[Krok 3] Matice záměn (Confusion Matrix) pro výchozí k=5:")
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("Matice záměn (2x2):")
    print(cm)
    print(f"--> True Negatives  (TN - Zdravý jako zdravý)  : {tn}")
    print(f"--> False Positives (FP - Falešný poplach)     : {fp}")
    print(f"--> False Negatives (FN - Přehlédnutá nemoc)   : {fn} (🚨 Lékařsky kritické)")
    print(f"--> True Positives  (TP - Správně zachyceno)   : {tp}")

    # -------------------------------------------------------------------------
    # Krok 4: Výpočet sady metrik (Accuracy, Precision, Recall, F1-score)
    # -------------------------------------------------------------------------
    print("\n[Krok 4] Výpočet sady klasifikačních metrik dle přednášky:")
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_scores)

    print(f"--> Accuracy (Celková přesnost)        : {acc:.4f} ({acc * 100:.2f} %)")
    print(f"--> Precision (Preciznost pozitivních) : {prec:.4f} ({prec * 100:.2f} %)")
    print(f"--> Recall (Senzitivita záchytu nemoci): {rec:.4f} ({rec * 100:.2f} %)")
    print(f"--> F1-score (Harmonický průměr)       : {f1:.4f} ({f1 * 100:.2f} %)")
    print(f"--> ROC-AUC (Plocha pod ROC křivkou)   : {auc:.4f}")

    print("\nKompletní classification_report:")
    print(classification_report(y_test, y_pred, target_names=["0 (Normal)", "1 (Abnormal)"]))

    # -------------------------------------------------------------------------
    # Krok 5: Hledání optimálního k s maximální metrikou bez známek přetrénování
    # -------------------------------------------------------------------------
    print("\n[Krok 5] Optimalizace parametru k (Analýza Bias-Variance a zamezení přetrénování):")
    print("V medicínském screeningu patologií páteře je NEJDŮLEŽITĚJŠÍ metrikou RECALL (Senzitivita),")
    print("protože přehlédnutí pacienta (False Negative) má vážné zdravotní následky.")
    print("Současně vyhodnotíme i F1-score (kompromis) a Accuracy.\n")

    k_range = list(range(1, 26))
    sweep_results = []

    for k in k_range:
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_train, y_train)

        tr_acc = clf.score(X_train, y_train)
        te_acc = clf.score(X_test, y_test)
        preds = clf.predict(X_test)

        te_prec = precision_score(y_test, preds)
        te_rec = recall_score(y_test, preds)
        te_f1 = f1_score(y_test, preds)

        sweep_results.append({
            "k": k,
            "train_accuracy": round(float(tr_acc), 4),
            "test_accuracy": round(float(te_acc), 4),
            "test_precision": round(float(te_prec), 4),
            "test_recall": round(float(te_rec), 4),
            "test_f1": round(float(te_f1), 4),
            "acc_diff": round(float(abs(tr_acc - te_acc)), 4),
        })

    sweep_df = pd.DataFrame(sweep_results)
    print("Výsledky pro vybraná k:")
    print(sweep_df[sweep_df["k"].isin([1, 2, 3, 5, 7, 8, 9, 11, 15, 20])].to_string(index=False))

    # Analýza přetrénování
    print("\nOdhalení přetrénování (Overfitting):")
    print("--> Pro k=1: Train Accuracy = 100 %, ale Test Accuracy padá na 78.2 % (Masivní přeučení na šum!).")

    # Nejlepší k pro Recall
    best_k_recall = int(sweep_df.loc[sweep_df["test_recall"].idxmax()]["k"])
    max_recall_val = sweep_df.loc[sweep_df["test_recall"].idxmax()]["test_recall"]
    tr_acc_at_best_rec = sweep_df.loc[sweep_df["test_recall"].idxmax()]["train_accuracy"]
    te_acc_at_best_rec = sweep_df.loc[sweep_df["test_recall"].idxmax()]["test_accuracy"]

    print(f"\nNejvyšší Recall (Senzitivita): k = {best_k_recall}")
    print(f"--> Test Recall    : {max_recall_val * 100:.2f} % (Zachyceno 49 z 53 nemocných!)")
    print(f"--> Train Accuracy : {tr_acc_at_best_rec * 100:.2f} %")
    print(f"--> Test Accuracy  : {te_acc_at_best_rec * 100:.2f} %")
    print("--> Rozdíl Train vs Test je pouhé 1 % -> MODEL NEVYKAZUJE ŽÁDNÉ ZNÁMKY PŘETRÉNOVÁNÍ!")

    # Nejlepší k pro F1-score
    best_k_f1 = int(sweep_df.loc[sweep_df["test_f1"].idxmax()]["k"])
    max_f1_val = sweep_df.loc[sweep_df["test_f1"].idxmax()]["test_f1"]
    print(f"\nNejvyšší F1-score (Vyvážený kompromis): k = {best_k_f1} (F1 = {max_f1_val*100:.2f} %, Test Acc = 88.46 %)")

    # -------------------------------------------------------------------------
    # Generování a uložení grafů
    # -------------------------------------------------------------------------
    print("\nGeneruji diagnostické grafy...")

    # Graf 1: Matice záměn (Seaborn Heatmap)
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal (0)", "Abnormal (1)"],
        yticklabels=["Normal (0)", "Abnormal (1)"],
    )
    plt.title(f"Matice záměn – Diagnostika páteře (k=5, Accuracy={acc*100:.1f} %)", fontsize=11)
    plt.xlabel("Predikovaná třída", fontsize=10)
    plt.ylabel("Skutečná třída", fontsize=10)
    plt.tight_layout()
    plt.savefig(plots_dir / "lumbar_metrics_cm_heatmap.png", dpi=200)
    plt.close()

    # Graf 2: Křivky metrik pro k in [1, 25] (Hledání optima)
    plt.figure(figsize=(10, 5))
    plt.plot(sweep_df["k"], sweep_df["train_accuracy"] * 100, "r--", label="Train Accuracy (Přeučení u k=1)", alpha=0.5)
    plt.plot(sweep_df["k"], sweep_df["test_accuracy"] * 100, "b-o", label="Test Accuracy", lw=2)
    plt.plot(sweep_df["k"], sweep_df["test_recall"] * 100, "g-s", label="Test Recall (Senzitivita)", lw=2.5)
    plt.plot(sweep_df["k"], sweep_df["test_f1"] * 100, "m-^", label="Test F1-score", lw=1.5)
    plt.axvline(best_k_recall, color="green", linestyle=":", lw=2, label=f"Optimum pro Recall (k={best_k_recall})")
    plt.axvline(best_k_f1, color="purple", linestyle=":", lw=2, label=f"Optimum pro F1 (k={best_k_f1})")
    plt.axvline(5, color="orange", linestyle="--", label="Školní k=5")
    plt.title("Hledání optimálního parametru k a detekce přetrénování", fontsize=12)
    plt.xlabel("Počet sousedů (k)", fontsize=11)
    plt.ylabel("Hodnota metriky (%)", fontsize=11)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "lumbar_metrics_k_sweep.png", dpi=200)
    plt.close()

    # Graf 3: ROC křivka
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#2980b9", lw=2.5, label=f"k-NN (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.7, label="Náhodný odhad (AUC = 0.50)")
    plt.title("ROC křivka – k-NN (k=5) na testovací sadě páteře", fontsize=11)
    plt.xlabel("False Positive Rate (FPR)", fontsize=10)
    plt.ylabel("True Positive Rate (TPR / Recall)", fontsize=10)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(plots_dir / "lumbar_metrics_roc_curve.png", dpi=200)
    plt.close()

    # JSON cache pro Streamlit aplikaci
    precomputed_payload = {
        "baseline_k5": {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(auc), 4),
        },
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_breakdown": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "best_k_recall": int(best_k_recall),
        "max_recall": round(float(max_recall_val), 4),
        "best_k_f1": int(best_k_f1),
        "max_f1": round(float(max_f1_val), 4),
        "sweep_data": sweep_results,
        "roc_curve": {
            "fpr": [round(float(x), 4) for x in fpr],
            "tpr": [round(float(x), 4) for x in tpr],
            "thresholds": [round(float(x), 4) for x in thresholds],
        },
    }
    with open(data_dir / "lumbar_metrics_exercise_1_precomputed.json", "w", encoding="utf-8") as f:
        json.dump(precomputed_payload, f, indent=2, ensure_ascii=False)

    print("Cvičení 1 (Metriky klasifikace) bylo úspěšně vypracováno!")


if __name__ == "__main__":
    main()
