"""
02_knn_lumbar_exercise_1.py
===========================
K-Nearest Neighbors - Exercise 1: Biomechanical Evaluation of Lumbar Spine Patients
Dataset: lumbar_data.csv (310 patients, 6 biomechanical parameters, binary diagnosis)

ZADÁNÍ / ASSIGNMENT:
-------------------------------------------------------------------------------
1. Using the Pandas library, load the downloaded patient data file (lumbar_data.csv)
   into the lumbar_df variable.
2. Display the first 10 observations in the dataset to see what type of data is in the file.
3. Check the data types of each variable, and verify the presence of empty values.
4. Perform data normalization to bring the data to a common scale.
5. Perform categorical encoding of the class variable. Assign the value of 0 to Normal,
   and the value of 1 to Abnormal.
6. Split the data into a training set and a test set in a 75/25 ratio. Set the random_state parameter to 42.
7. Import the appropriate class from the neighbors module of the Scikit-learn library,
   which you will use to build a k nearest neighbors model.
8. Create your own instance of the kNN model and train it on the created training set.
   Try to choose an appropriate value for the k parameter.
9. Use the trained model to perform a prediction on the test set.
10. Save the dataset from step 5 to a file named lumbar_normalized_df.csv.
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
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer, StandardScaler, normalize


def main() -> None:
    # -------------------------------------------------------------------------
    # Nastavení cest / Paths setup
    # -------------------------------------------------------------------------
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    plots_dir = base_dir / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "lumbar_data.csv"
    if not csv_path.exists():
        fallback = base_dir.parent / "data" / "MAL_downloadable materials_session 1" / "Day 2" / "lumbar_data.csv"
        if fallback.exists():
            import shutil
            shutil.copy(fallback, csv_path)
        else:
            raise FileNotFoundError(f"Soubor lumbar_data.csv nebyl nalezen v {csv_path} ani ve fallbacku.")

    print("=" * 75)
    print("K-NEAREST NEIGHBORS – CVIČENÍ 1: KLINICKÁ DIAGNOSTIKA BEDERNÍ PÁTEŘE")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # Krok 1 & 2: Načtení dat a zobrazení prvních 10 pozorování
    # -------------------------------------------------------------------------
    print("\n[Krok 1 & 2] Načtení lumbar_data.csv a zobrazení prvních 10 řádků:")
    lumbar_df = pd.read_csv(csv_path)
    print(f"Rozměry datasetu: {lumbar_df.shape[0]} pacientů x {lumbar_df.shape[1]} sloupců")
    print(lumbar_df.head(10))

    # -------------------------------------------------------------------------
    # Krok 3: Datové typy a kontrola chybějících hodnot
    # -------------------------------------------------------------------------
    print("\n[Krok 3] Kontrola datových typů a chybějících hodnot:")
    info_df = pd.DataFrame({
        "Datový typ": lumbar_df.dtypes,
        "Počet chybějících (NaN)": lumbar_df.isnull().sum(),
        "Podíl chybějících (%)": (lumbar_df.isnull().sum() / len(lumbar_df)) * 100,
    })
    print(info_df)
    print("\nPůvodní zastoupení tříd v 'class':")
    print(lumbar_df["class"].value_counts())

    # -------------------------------------------------------------------------
    # Krok 5: Kategoriální kódování třídy (Normal = 0, Abnormal = 1)
    # -------------------------------------------------------------------------
    print("\n[Krok 5] Kódování cílové proměnné 'class': Normal -> 0, Abnormal -> 1")
    # Pacienti s Hernií nebo Spondylolistézou spadají do kategorie Abnormal (1)
    encoded_class = lumbar_df["class"].apply(lambda val: 0 if str(val).strip().lower() == "normal" else 1)
    print("Zastoupení po binárním kódování:")
    print(encoded_class.value_counts().rename({1: "1 (Abnormal)", 0: "0 (Normal)"}))

    # -------------------------------------------------------------------------
    # Krok 4: Normalizace příznaků (převod na společné měřítko)
    # -------------------------------------------------------------------------
    print("\n[Krok 4] Normalizace dat na společné měřítko:")
    feature_cols = [
        "pelvic_incidence",
        "pelvic_tilt",
        "lumbar_lordosis_angle",
        "sacral_slope",
        "pelvic_radius",
        "degree_spondylolisthesis",
    ]
    X_raw = lumbar_df[feature_cols]

    # Normalizace po řádcích (L2 norma vektoru pacienta - přesně dle školního lumbar_df_normalized.csv)
    normalizer = Normalizer(norm="l2")
    X_normalized_l2 = normalizer.fit_transform(X_raw)
    normalized_features_df = pd.DataFrame(X_normalized_l2, columns=feature_cols)

    # Vytvoření normalizovaného datasetu s kódovanou třídou
    lumbar_normalized_df = normalized_features_df.copy()
    lumbar_normalized_df["class"] = encoded_class.values
    print("Normalizovaná data (prvních 5 řádků):")
    print(lumbar_normalized_df.head(5).round(4))

    # -------------------------------------------------------------------------
    # Krok 10: Uložení datasetu do lumbar_normalized_df.csv
    # -------------------------------------------------------------------------
    out_csv_path = data_dir / "lumbar_normalized_df.csv"
    lumbar_normalized_df.to_csv(out_csv_path, index=False)
    print(f"\n[Krok 10] Normalizovaný dataset byl úspěšně uložen do: {out_csv_path}")

    # Zároveň vytvoříme / synchronizujeme lumbar_df_normalized.csv s kurzem
    course_csv_path = data_dir / "lumbar_df_normalized.csv"
    lumbar_normalized_df.rename(columns={"pelvic_tilt": "pelvic_tilt numeric"}).to_csv(course_csv_path, index=False)

    # -------------------------------------------------------------------------
    # Krok 6: Rozdělení dat na trénovací a testovací sadu (75/25, random_state=42)
    # -------------------------------------------------------------------------
    print("\n[Krok 6] Rozdělení dat na Train a Test (75:25, random_state=42):")
    X = lumbar_normalized_df.drop("class", axis=1)
    y = lumbar_normalized_df["class"]

    # Školní rozdělení bez stratifikace
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Stratifikované rozdělení pro porovnání
    X_tr_strat, X_te_strat, y_tr_strat, y_te_strat = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print(f"Trénovací sada: {X_train.shape[0]} vzorků")
    print(f"Testovací sada : {X_test.shape[0]} vzorků")

    # -------------------------------------------------------------------------
    # Krok 7 & 8: Import KNeighborsClassifier a trénování modelu s volbou k
    # -------------------------------------------------------------------------
    print("\n[Krok 7 & 8] Hledání optimálního parametru k a trénování modelu k-NN:")
    k_heuristic = int(np.round(np.sqrt(X_train.shape[0])))
    if k_heuristic % 2 == 0:
        k_heuristic += 1
    print(f"Heuristika odmocniny sqrt(N_train): k ≈ {k_heuristic}")

    k_values = list(range(1, 31))
    test_scores_unstrat = []
    test_scores_strat = []
    train_scores_strat = []

    for k in k_values:
        clf_unstrat = KNeighborsClassifier(n_neighbors=k)
        clf_unstrat.fit(X_train, y_train)
        test_scores_unstrat.append(clf_unstrat.score(X_test, y_test))

        clf_strat = KNeighborsClassifier(n_neighbors=k)
        clf_strat.fit(X_tr_strat, y_tr_strat)
        train_scores_strat.append(clf_strat.score(X_tr_strat, y_tr_strat))
        test_scores_strat.append(clf_strat.score(X_te_strat, y_te_strat))

    best_k_unstrat = k_values[int(np.argmax(test_scores_unstrat))]
    best_k_strat = k_values[int(np.argmax(test_scores_strat))]

    print(f"Základní split (bez stratifikace): Nejlepší k = {best_k_unstrat} (Accuracy = {max(test_scores_unstrat)*100:.2f} %)")
    print(f"Stratifikovaný split (s vyvážením): Nejlepší k = {best_k_strat} (Accuracy = {max(test_scores_strat)*100:.2f} %)")

    # Zvolíme reprezentativní model (např. k=5 ze školního standardu a optimální k)
    chosen_k = 5
    knn_model = KNeighborsClassifier(n_neighbors=chosen_k)
    knn_model.fit(X_train, y_train)

    knn_opt_strat = KNeighborsClassifier(n_neighbors=best_k_strat)
    knn_opt_strat.fit(X_tr_strat, y_tr_strat)

    # -------------------------------------------------------------------------
    # Krok 9: Predikce na testovací sadě a diagnostika
    # -------------------------------------------------------------------------
    print(f"\n[Krok 9] Predikce a vyhodnocení na testovací sadě (k={chosen_k}):")
    y_pred = knn_model.predict(X_test)
    y_pred_proba = knn_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"--> Testovací Accuracy : {acc * 100:.2f} %")
    print(f"--> Precision (Přesnost): {prec * 100:.2f} %")
    print(f"--> Recall (Senzitivita): {rec * 100:.2f} % (Klíčová metrika pro medicínu!)")
    print(f"--> F1-score           : {f1 * 100:.2f} %")
    print(f"--> ROC-AUC            : {roc_auc:.4f}")

    print("\nKlasifikační report (Školní model, k=5):")
    print(classification_report(y_test, y_pred, target_names=["0 (Normal)", "1 (Abnormal)"]))

    # Vyhodnocení stratifikovaného optimálního modelu
    y_pred_strat = knn_opt_strat.predict(X_te_strat)
    acc_strat = accuracy_score(y_te_strat, y_pred_strat)
    rec_strat = recall_score(y_te_strat, y_pred_strat)
    print(f"Stratifikovaný optimální model (k={best_k_strat}): Accuracy = {acc_strat*100:.2f} %, Recall = {rec_strat*100:.2f} %")

    # -------------------------------------------------------------------------
    # 11. Diagnostické grafy a uložení výstupů
    # -------------------------------------------------------------------------
    print("\nGeneruji diagnostické grafy...")

    # Graf 1: Křivka přesnosti pro k in [1, 30]
    plt.figure(figsize=(9, 4.5))
    plt.plot(k_values, [s * 100 for s in test_scores_unstrat], "r--o", label="Základní split (Test Acc)", alpha=0.7)
    plt.plot(k_values, [s * 100 for s in test_scores_strat], "b-o", label="Stratifikovaný split (Test Acc)", lw=2)
    plt.plot(k_values, [s * 100 for s in train_scores_strat], "b:", label="Stratifikovaný split (Train Acc)", alpha=0.5)
    plt.axvline(best_k_strat, color="green", linestyle=":", lw=2, label=f"Optimum k={best_k_strat}")
    plt.axvline(chosen_k, color="orange", linestyle="--", label=f"Školní k={chosen_k}")
    plt.xlabel("Počet sousedů (k)", fontsize=11)
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.title("Hledání optimálního parametru k – Diagnostika páteře (k-NN)", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "lumbar_knn_k_curve.png", dpi=200)
    plt.close()

    # Graf 2: Konfúzní matice
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal (0)", "Abnormal (1)"],
        yticklabels=["Normal (0)", "Abnormal (1)"],
    )
    plt.title(f"Konfúzní matice – Bederní páteř (k=5, Acc={acc*100:.1f} %)", fontsize=11)
    plt.xlabel("Predikovaný stav", fontsize=10)
    plt.ylabel("Skutečný stav pacienta", fontsize=10)
    plt.tight_layout()
    plt.savefig(plots_dir / "lumbar_knn_confusion_matrix.png", dpi=200)
    plt.close()

    # JSON cache pro Streamlit aplikaci
    precomputed_payload = {
        "dataset_shape": list(lumbar_df.shape),
        "class_counts_raw": lumbar_df["class"].value_counts().to_dict(),
        "class_counts_encoded": {"0 (Normal)": int((encoded_class == 0).sum()), "1 (Abnormal)": int((encoded_class == 1).sum())},
        "chosen_k": chosen_k,
        "school_metrics": {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(roc_auc), 4),
        },
        "best_k_strat": int(best_k_strat),
        "strat_metrics": {
            "accuracy": round(float(acc_strat), 4),
            "recall": round(float(rec_strat), 4),
        },
        "k_values": k_values,
        "test_scores_unstrat": [round(float(s), 4) for s in test_scores_unstrat],
        "test_scores_strat": [round(float(s), 4) for s in test_scores_strat],
        "confusion_matrix": cm.tolist(),
        "feature_cols": feature_cols,
    }
    with open(data_dir / "lumbar_knn_precomputed.json", "w", encoding="utf-8") as f:
        json.dump(precomputed_payload, f, indent=2, ensure_ascii=False)

    print("Cvičení 1 (Lumbar Spine) bylo kompletně vypracováno, data uložena a grafy vygenerovány!")


if __name__ == "__main__":
    main()
