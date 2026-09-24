"""
03_knn_penguins_exercise_2.py
=============================
K-Nearest Neighbors - Exercise 2: Multiclass Species Classification on Palmer Penguins
Dataset: penguins_size.csv (344 observations, 6 features, 3 penguin species)

ZADÁNÍ / ASSIGNMENT:
-------------------------------------------------------------------------------
1. Using the Pandas library, load the downloaded penguin data file (penguins_size.csv)
   into the penguins_df variable.
2. Display the first 10 observations in the dataset to see what type of data is in the file.
3. Check the data types of each variable, and verify the presence of empty values.
4. Perform data normalization to bring the data to a common scale.
5. Split the data into a training set and a test set in a 70/30 ratio. Set the random_state parameter to 42.
6. Import the appropriate class from the neighbors module of the Scikit-learn library,
   which you will use to build a k nearest neighbors model.
7. Create your own instance of the kNN model and train it on the created training set.
   Try to choose an appropriate value for the k parameter.
8. Use the trained model to perform a prediction on the test set.
9. Save the dataset from step 4 to a file named penguins_df_normalized.csv.
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer, StandardScaler


def main() -> None:
    # -------------------------------------------------------------------------
    # Cesty k datům a výstupům
    # -------------------------------------------------------------------------
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    plots_dir = base_dir / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "penguins_size.csv"
    if not csv_path.exists():
        fallback = base_dir.parent / "data" / "MAL_downloadable materials_session 1" / "Day 2" / "penguins_size.csv"
        if fallback.exists():
            import shutil
            shutil.copy(fallback, csv_path)
        else:
            raise FileNotFoundError(f"Soubor penguins_size.csv nebyl nalezen v {csv_path} ani ve fallbacku.")

    print("=" * 75)
    print("K-NEAREST NEIGHBORS – CVIČENÍ 2: KLASIFIKACE TUČŇÁKŮ (PALMER PENGUINS)")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # Krok 1 & 2: Načtení dat a zobrazení prvních 10 pozorování
    # -------------------------------------------------------------------------
    print("\n[Krok 1 & 2] Načtení penguins_size.csv a zobrazení prvních 10 řádků:")
    penguins_df = pd.read_csv(csv_path)
    print(f"Rozměry datasetu: {penguins_df.shape[0]} tučňáků x {penguins_df.shape[1]} sloupců")
    print(penguins_df.head(10))

    # -------------------------------------------------------------------------
    # Krok 3: Kontrola datových typů a chybějících hodnot
    # -------------------------------------------------------------------------
    print("\n[Krok 3] Kontrola datových typů a ověření přítomnosti prázdných hodnot:")
    info_df = pd.DataFrame({
        "Datový typ": penguins_df.dtypes,
        "Počet chybějících (NaN)": penguins_df.isnull().sum(),
        "Podíl chybějících (%)": (penguins_df.isnull().sum() / len(penguins_df)) * 100,
    })
    print(info_df)
    print("\nZastoupení jednotlivých druhů tučňáků:")
    print(penguins_df["species"].value_counts())

    # Odstranění chybějících hodnot (10 řádků s NaN)
    clean_df = penguins_df.dropna().copy()
    print(f"Rozměry po odstranění NaN: {clean_df.shape[0]} řádků")

    # -------------------------------------------------------------------------
    # Krok 4: Normalizace dat na společné měřítko a kódování kategorií
    # -------------------------------------------------------------------------
    print("\n[Krok 4] Normalizace číselných dat a zakódování kategoriálních proměnných:")
    # Kódování ostrova a pohlaví dle školního standardu
    island_map = {"Biscoe": 0, "Dream": 1, "Torgersen": 2}
    sex_map = {".": 0, "FEMALE": 1, "MALE": 2}
    clean_df["island"] = clean_df["island"].map(island_map)
    clean_df["sex"] = clean_df["sex"].map(sex_map)

    num_cols = ["culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g"]
    normalizer = Normalizer(norm="l2")
    clean_df[num_cols] = normalizer.fit_transform(clean_df[num_cols])

    # Uspořádání sloupců dle školní šablony
    cols_order = [
        "culmen_length_mm",
        "culmen_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
        "island",
        "sex",
        "species",
    ]
    penguins_df_normalized = clean_df[cols_order].reset_index(drop=True)
    print("Normalizovaný dataset (prvních 5 řádků):")
    print(penguins_df_normalized.head(5).round(4))

    # -------------------------------------------------------------------------
    # Krok 9: Uložení normalizovaného datasetu do penguins_df_normalized.csv
    # -------------------------------------------------------------------------
    out_csv_path = data_dir / "penguins_df_normalized.csv"
    penguins_df_normalized.to_csv(out_csv_path, index=False)
    print(f"\n[Krok 9] Normalizovaný dataset byl úspěšně uložen do: {out_csv_path}")

    # -------------------------------------------------------------------------
    # Krok 5: Rozdělení dat na trénovací a testovací sadu (70/30, random_state=42)
    # -------------------------------------------------------------------------
    print("\n[Krok 5] Rozdělení dat na Train a Test (70:30, random_state=42):")
    X = penguins_df_normalized.drop("species", axis=1)
    y = penguins_df_normalized["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    print(f"Trénovací sada: {X_train.shape[0]} vzorků")
    print(f"Testovací sada : {X_test.shape[0]} vzorků")

    # -------------------------------------------------------------------------
    # Krok 6 & 7: Import KNeighborsClassifier a trénování modelu s volbou k
    # -------------------------------------------------------------------------
    print("\n[Krok 6 & 7] Trénování k-NN a hledání optimálního parametru k:")
    k_heuristic = int(np.round(np.sqrt(X_train.shape[0])))
    if k_heuristic % 2 == 0:
        k_heuristic += 1
    print(f"Heuristika odmocniny sqrt(N_train): k ≈ {k_heuristic}")

    k_values = list(range(1, 36))
    train_scores = []
    test_scores = []

    for k in k_values:
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_train, y_train)
        train_scores.append(clf.score(X_train, y_train))
        test_scores.append(clf.score(X_test, y_test))

    best_k = k_values[int(np.argmax(test_scores))]
    print(f"Optimální hodnota hyperparametru k: k = {best_k} (Test Accuracy = {max(test_scores)*100:.2f} %)")

    chosen_k = 5
    knn_model = KNeighborsClassifier(n_neighbors=chosen_k)
    knn_model.fit(X_train, y_train)

    # -------------------------------------------------------------------------
    # Krok 8: Predikce na testovací sadě a vyhodnocení
    # -------------------------------------------------------------------------
    print(f"\n[Krok 8] Predikce a diagnostika na testovací sadě (k={chosen_k}):")
    y_pred = knn_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro")
    rec_macro = recall_score(y_test, y_pred, average="macro")
    f1_macro = f1_score(y_test, y_pred, average="macro")

    print(f"--> Testovací Accuracy : {acc * 100:.2f} %")
    print(f"--> Macro Precision    : {prec_macro * 100:.2f} %")
    print(f"--> Macro Recall       : {rec_macro * 100:.2f} %")
    print(f"--> Macro F1-score     : {f1_macro * 100:.2f} %")

    print("\nDetailní klasifikační report pro jednotlivé druhy tučňáků:")
    print(classification_report(y_test, y_pred))

    # -------------------------------------------------------------------------
    # Diagnostické grafy a uložení
    # -------------------------------------------------------------------------
    print("Generuji diagnostické grafy...")

    # Graf 1: Křivka k in [1, 35]
    plt.figure(figsize=(9, 4.5))
    plt.plot(k_values, [s * 100 for s in test_scores], "b-o", label="Test Accuracy", lw=2)
    plt.plot(k_values, [s * 100 for s in train_scores], "r--", label="Train Accuracy", alpha=0.5)
    plt.axvline(best_k, color="green", linestyle=":", lw=2, label=f"Optimum k={best_k}")
    plt.axvline(chosen_k, color="orange", linestyle="--", label=f"Školní k={chosen_k}")
    plt.title("Závislost přesnosti na počtu sousedů k (Palmer Penguins Exercise 2)", fontsize=12)
    plt.xlabel("Počet sousedů (k)", fontsize=11)
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "penguins_ex2_k_curve.png", dpi=200)
    plt.close()

    # Graf 2: Konfúzní matice
    cm = confusion_matrix(y_test, y_pred, labels=knn_model.classes_)
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=knn_model.classes_,
        yticklabels=knn_model.classes_,
    )
    plt.title(f"Konfúzní matice tučňáků (k={chosen_k}, Acc={acc*100:.1f} %)", fontsize=11)
    plt.xlabel("Predikovaný druh", fontsize=10)
    plt.ylabel("Skutečný druh", fontsize=10)
    plt.tight_layout()
    plt.savefig(plots_dir / "penguins_ex2_confusion_matrix.png", dpi=200)
    plt.close()

    # Uložení JSON cache
    precomputed_payload = {
        "dataset_shape": list(penguins_df.shape),
        "clean_shape": list(clean_df.shape),
        "species_counts": penguins_df["species"].value_counts().to_dict(),
        "chosen_k": chosen_k,
        "best_k": int(best_k),
        "metrics_k5": {
            "accuracy": round(float(acc), 4),
            "precision_macro": round(float(prec_macro), 4),
            "recall_macro": round(float(rec_macro), 4),
            "f1_macro": round(float(f1_macro), 4),
        },
        "k_values": k_values,
        "test_scores": [round(float(s), 4) for s in test_scores],
        "train_scores": [round(float(s), 4) for s in train_scores],
        "confusion_matrix": cm.tolist(),
        "classes": list(knn_model.classes_),
    }
    with open(data_dir / "penguins_exercise_2_precomputed.json", "w", encoding="utf-8") as f:
        json.dump(precomputed_payload, f, indent=2, ensure_ascii=False)

    print("Cvičení 2 (Palmer Penguins) bylo kompletně vypracováno, dataset uložen a grafy vygenerovány!")


if __name__ == "__main__":
    main()
