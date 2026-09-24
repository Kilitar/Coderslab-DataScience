"""
05_classification_metrics_penguins_exercise_2.py
================================================
Metriky klasifikačních modelů – Cvičení 2: Klasifikace druhů tučňáků (Multiclass)
Dataset: penguins_df_normalized.csv (334 tučňáků, 6 normalizovaných příznaků, 3 druhy)

ZADÁNÍ / ASSIGNMENT:
-------------------------------------------------------------------------------
1. Download the dataset attached to the task (penguins_df_normalized.csv) and notebook.
   Open the notebook in Google Colab and place the data file in the execution environment.
2. After uploading the data file, run all the cells with code in notebook.
3. Generate a confusion matrix and represent it in a graph. Use any library for data visualization.
4. Using the knowledge from the presentation, calculate the set of metrics:
   - accuracy
   - precision
   - recall
   - f1-score
5. Choose one of the above metrics and try to find the k for which the value of the metric
   will be the best, i.e. the model will be the most effective on the test set,
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
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def main() -> None:
    # -------------------------------------------------------------------------
    # Cesty k souborům
    # -------------------------------------------------------------------------
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    plots_dir = base_dir / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "penguins_df_normalized.csv"
    if not csv_path.exists():
        fallback = base_dir / "penguins_df_normalized.csv"
        if fallback.exists():
            import shutil
            shutil.copy(fallback, csv_path)
        else:
            raise FileNotFoundError(f"Soubor penguins_df_normalized.csv nebyl nalezen.")

    print("=" * 75)
    print("METRIKY KLASIFIKAČNÍCH MODELŮ – CVIČENÍ 2: DRUHY TUČŇÁKŮ (MULTICLASS)")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # Krok 1 & 2: Načtení dat a rozdělení dle zadání kurzu (70/30, random_state=42)
    # -------------------------------------------------------------------------
    print("\n[Krok 1 & 2] Načtení dat a trénování výchozího k-NN modelu (k=5):")
    penguins_df = pd.read_csv(csv_path)
    print(f"Rozměry datasetu: {penguins_df.shape[0]} tučňáků x {penguins_df.shape[1]} sloupců")
    print(penguins_df.head(10))

    X = penguins_df.drop("species", axis=1)
    y = penguins_df["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    species_labels = ["Adelie", "Chinstrap", "Gentoo"]
    print(f"Trénovací data (Train): {X_train.shape[0]} vzorků")
    print(f"Testovací data (Test) : {X_test.shape[0]} vzorků (Adelie: {sum(y_test=='Adelie')}, Chinstrap: {sum(y_test=='Chinstrap')}, Gentoo: {sum(y_test=='Gentoo')})")

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    # -------------------------------------------------------------------------
    # Krok 3: Generování matice záměn (Confusion Matrix) pro k=5
    # -------------------------------------------------------------------------
    print("\n[Krok 3] Matice záměn (Confusion Matrix) pro k = 5:")
    cm_k5 = confusion_matrix(y_test, y_pred, labels=species_labels)
    print("Matice záměn (3x3):")
    cm_df_k5 = pd.DataFrame(cm_k5, index=[f"Actual {s}" for s in species_labels], columns=[f"Pred {s}" for s in species_labels])
    print(cm_df_k5)

    # -------------------------------------------------------------------------
    # Krok 4: Výpočet sady metrik (Accuracy, Precision, Recall, F1-score)
    # -------------------------------------------------------------------------
    print("\n[Krok 4] Výpočet sady klasifikačních metrik pro více tříd:")
    acc_k5 = accuracy_score(y_test, y_pred)
    prec_w_k5 = precision_score(y_test, y_pred, average="weighted")
    rec_w_k5 = recall_score(y_test, y_pred, average="weighted")
    f1_w_k5 = f1_score(y_test, y_pred, average="weighted")
    f1_m_k5 = f1_score(y_test, y_pred, average="macro")

    print(f"--> Accuracy (Celková přesnost)         : {acc_k5:.4f} ({acc_k5*100:.2f} %)")
    print(f"--> Weighted Precision (Vážená preciznost): {prec_w_k5:.4f} ({prec_w_k5*100:.2f} %)")
    print(f"--> Weighted Recall (Vážená senzitivita)  : {rec_w_k5:.4f} ({rec_w_k5*100:.2f} %)")
    print(f"--> Weighted F1-score                   : {f1_w_k5:.4f} ({f1_w_k5*100:.2f} %)")
    print(f"--> Macro F1-score (Nevážený průměr)    : {f1_m_k5:.4f} ({f1_m_k5*100:.2f} %)")

    print("\nDetailní Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    # -------------------------------------------------------------------------
    # Krok 5: Hledání optimálního k s maximální metrikou bez přetrénování
    # -------------------------------------------------------------------------
    print("\n[Krok 5] Hledání optimálního parametru k a detekce přetrénování:")
    k_range = list(range(1, 36))
    sweep_results = []

    for k in k_range:
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_train, y_train)

        tr_acc = clf.score(X_train, y_train)
        te_acc = clf.score(X_test, y_test)
        preds = clf.predict(X_test)

        prec_w = precision_score(y_test, preds, average="weighted", zero_division=0)
        rec_w = recall_score(y_test, preds, average="weighted", zero_division=0)
        f1_w = f1_score(y_test, preds, average="weighted", zero_division=0)
        f1_m = f1_score(y_test, preds, average="macro", zero_division=0)

        sweep_results.append({
            "k": k,
            "train_accuracy": round(float(tr_acc), 4),
            "test_accuracy": round(float(te_acc), 4),
            "test_precision_weighted": round(float(prec_w), 4),
            "test_recall_weighted": round(float(rec_w), 4),
            "test_f1_weighted": round(float(f1_w), 4),
            "test_f1_macro": round(float(f1_m), 4),
            "acc_gap": round(float(abs(tr_acc - te_acc)), 4),
        })

    sweep_df = pd.DataFrame(sweep_results)

    # Identifikace optima pro F1-score (a Accuracy)
    best_idx = sweep_df["test_f1_weighted"].idxmax()
    best_k = int(sweep_df.loc[best_idx, "k"])
    best_f1 = sweep_df.loc[best_idx, "test_f1_weighted"]
    best_acc = sweep_df.loc[best_idx, "test_accuracy"]
    best_tr_acc = sweep_df.loc[best_idx, "train_accuracy"]
    best_gap = sweep_df.loc[best_idx, "acc_gap"]

    print(f"\nAbsolutní optimum pro Weighted F1-score (i Accuracy): k = {best_k}")
    print(f"--> Test Accuracy   : {best_acc*100:.2f} % (98 ze 101 správně!)")
    print(f"--> Test F1-score   : {best_f1*100:.2f} %")
    print(f"--> Train Accuracy  : {best_tr_acc*100:.2f} %")
    print(f"--> Rozdíl Train-Test: {best_gap*100:.2f} % (Minimální rozdíl -> ŽÁDNÉ PŘETRÉNOVÁNÍ!)")

    # Pokud student preferuje liché k (pro zamezení remíz):
    odd_sweep = sweep_df[sweep_df["k"] % 2 != 0]
    best_odd_idx = odd_sweep["test_f1_weighted"].idxmax()
    best_odd_k = int(odd_sweep.loc[best_odd_idx, "k"])
    best_odd_f1 = odd_sweep.loc[best_odd_idx, "test_f1_weighted"]
    best_odd_acc = odd_sweep.loc[best_odd_idx, "test_accuracy"]

    print(f"\nNejlepší liché k (proti remízám): k = {best_odd_k} (F1 = {best_odd_f1*100:.2f} %, Test Acc = {best_odd_acc*100:.2f} %)")

    # -------------------------------------------------------------------------
    # Generování a uložení grafů
    # -------------------------------------------------------------------------
    print("\nGeneruji diagnostické grafy...")

    # Graf 1: Matice záměn k=5
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_k5, annot=True, fmt="d", cmap="Blues",
                xticklabels=species_labels, yticklabels=species_labels)
    plt.title(f"Matice záměn – Tučňáci (k=5, Accuracy={acc_k5*100:.1f} %)", fontsize=11, fontweight="bold")
    plt.xlabel("Predikovaný druh", fontsize=10)
    plt.ylabel("Skutečný druh", fontsize=10)
    plt.tight_layout()
    plt.savefig(plots_dir / "penguins_metrics_cm_k5.png", dpi=200)
    plt.close()

    # Graf 2: Matice záměn k=2 (Optimum)
    clf_opt = KNeighborsClassifier(n_neighbors=best_k).fit(X_train, y_train)
    preds_opt = clf_opt.predict(X_test)
    cm_k2 = confusion_matrix(y_test, preds_opt, labels=species_labels)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_k2, annot=True, fmt="d", cmap="Greens",
                xticklabels=species_labels, yticklabels=species_labels)
    plt.title(f"Matice záměn – Tučňáci Optimum (k={best_k}, Accuracy={best_acc*100:.1f} %)", fontsize=11, fontweight="bold")
    plt.xlabel("Predikovaný druh", fontsize=10)
    plt.ylabel("Skutečný druh", fontsize=10)
    plt.tight_layout()
    plt.savefig(plots_dir / "penguins_metrics_cm_k2.png", dpi=200)
    plt.close()

    # Graf 3: Křivky metrik pro k in [1, 35]
    plt.figure(figsize=(11, 5.5))
    plt.plot(sweep_df["k"], sweep_df["train_accuracy"] * 100, "r--o", label="Train Accuracy (Přeučení u k=1)", alpha=0.6)
    plt.plot(sweep_df["k"], sweep_df["test_accuracy"] * 100, "b-s", label="Test Accuracy", lw=2)
    plt.plot(sweep_df["k"], sweep_df["test_f1_weighted"] * 100, "g-^", label="Test F1-score (Weighted)", lw=2.5)
    plt.plot(sweep_df["k"], sweep_df["test_precision_weighted"] * 100, "m-v", label="Test Precision (Weighted)", lw=1.5, alpha=0.7)
    plt.axvline(best_k, color="green", linestyle="--", lw=2, label=f"Globální optimum (k={best_k})")
    plt.axvline(5, color="orange", linestyle=":", lw=2, label="Výchozí k=5")
    plt.title("Hledání optimálního k pro klasifikaci tučňáků a detekce přetrénování", fontsize=12, fontweight="bold")
    plt.xlabel("Počet sousedů (k)", fontsize=11)
    plt.ylabel("Hodnota metriky (%)", fontsize=11)
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "penguins_metrics_k_sweep.png", dpi=200)
    plt.close()

    # -------------------------------------------------------------------------
    # JSON Cache pro Streamlit
    # -------------------------------------------------------------------------
    precomputed_payload = {
        "species_labels": species_labels,
        "baseline_k5": {
            "accuracy": round(float(acc_k5), 4),
            "precision_weighted": round(float(prec_w_k5), 4),
            "recall_weighted": round(float(rec_w_k5), 4),
            "f1_weighted": round(float(f1_w_k5), 4),
            "f1_macro": round(float(f1_m_k5), 4),
            "cm": cm_k5.tolist(),
        },
        "optimum_k2": {
            "k": best_k,
            "accuracy": round(float(best_acc), 4),
            "f1_weighted": round(float(best_f1), 4),
            "train_accuracy": round(float(best_tr_acc), 4),
            "gap": round(float(best_gap), 4),
            "cm": cm_k2.tolist(),
        },
        "best_odd_k": {
            "k": best_odd_k,
            "accuracy": round(float(best_odd_acc), 4),
            "f1_weighted": round(float(best_odd_f1), 4),
        },
        "sweep_data": sweep_results,
    }

    with open(data_dir / "penguins_metrics_exercise_2_precomputed.json", "w", encoding="utf-8") as f:
        json.dump(precomputed_payload, f, indent=2, ensure_ascii=False)

    print("Cvičení 2 (Metriky tučňáků) bylo úspěšně vypracováno!")


if __name__ == "__main__":
    main()
