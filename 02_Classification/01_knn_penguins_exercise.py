"""
01_knn_penguins_exercise.py
===========================
Klasifikace k nejbližších sousedů (k-NN) na datech Palmer Penguins (penguins_size.csv).

Tento skript pokrývá:
1. Školní baseline (přesná replikace z K_nearest_neighbors_-_sample_implementation.pdf):
   - Odstranění 'island' a 'sex', odstranění NaN
   - train_test_split (70/30, random_state=42)
   - KNeighborsClassifier(n_neighbors=5) bez škálování (Accuracy ~80.2 %)
2. Inspekci sousedů pomocí metody .kneighbors()
3. Expertní rozbor vlivu škálování (StandardScaler, MinMaxScaler, RobustScaler)
4. Analýzu vlivu hyperparametru k (Bias-Variance tradeoff, k in [1, 45])
5. Vliv vah ('uniform' vs. 'distance') a metrik (Euclidean vs. Manhattan)
6. Profesionální Scikit-learn Pipeline s OneHotEncoderem pro ostrov a pohlaví
7. Vyhodnocení vícetřídních metrik (Confusion Matrix, Precision, Recall, F1)
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler


def main() -> None:
    # -------------------------------------------------------------------------
    # 1. Cesty k datům a výstupům
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
            raise FileNotFoundError(f"Dataset penguins_size.csv nebyl nalezen v {csv_path} ani ve fallbacku.")

    # -------------------------------------------------------------------------
    # 2. Načtení dat a průzkum
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("1. NAČTENÍ A PRŮZKUM DATASETU PALMER PENGUINS")
    print("=" * 70)
    raw_df = pd.read_csv(csv_path)
    print(f"Původní tvar dat: {raw_df.shape}")
    print("\nZastoupení cílových tříd (species):")
    print(raw_df["species"].value_counts())
    print("\nPočty chybějících hodnot (NaN):")
    print(raw_df.isnull().sum())

    # -------------------------------------------------------------------------
    # 3. Školní baseline (Replikace z PDF kurzu)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("2. ŠKOLNÍ BASELINE (Dle K_nearest_neighbors_-_sample_implementation.pdf)")
    print("=" * 70)
    school_df = raw_df.drop(["island", "sex"], axis=1)
    school_df.dropna(inplace=True)
    print(f"Tvar po odstranění 'island', 'sex' a NaN: {school_df.shape}")

    X_school = school_df.drop("species", axis=1)
    y_school = school_df["species"]

    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X_school, y_school, test_size=0.3, random_state=42
    )

    knn_school = KNeighborsClassifier(n_neighbors=5)
    knn_school.fit(X_train_s, y_train_s)
    y_pred_s = knn_school.predict(X_test_s)
    school_acc = knn_school.score(X_test_s, y_test_s)
    print(f"Školní model (bez škálování, k=5, 4 numerické příznaky):")
    print(f"--> Testovací Accuracy: {school_acc * 100:.2f} %")

    # -------------------------------------------------------------------------
    # 4. Inspekční metoda .kneighbors()
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("3. INSPEKCE METODOU .kneighbors() PRO PRVNÍ TESTOVACÍ VZOREK")
    print("=" * 70)
    sample_x = X_test_s.iloc[0:1]
    sample_y = y_test_s.iloc[0]
    distances, indices = knn_school.kneighbors(sample_x, n_neighbors=5)
    print(f"Skutečný druh vzorku: {sample_y}")
    print(f"Predikovaný druh vzorku: {knn_school.predict(sample_x)[0]}")
    print("Indexy 5 nejbližších trénovacích vzorků:", indices[0])
    print("Vzdálenosti k 5 nejbližším sousedům:", np.round(distances[0], 2))
    nearest_neighbors_classes = y_train_s.iloc[indices[0]].values
    print("Druhy 5 nejbližších sousedů:", nearest_neighbors_classes)

    # -------------------------------------------------------------------------
    # 5. Expertní analýza: Proč neškálovaný model selhává?
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("4. EXPERTNÍ ANALÝZA: VLIV ŠKÁLOVÁNÍ NA k-NN")
    print("=" * 70)
    stats_df = pd.DataFrame({
        "Min": X_school.min(),
        "Max": X_school.max(),
        "Průměr": X_school.mean(),
        "Sm. odchylka (std)": X_school.std(),
        "Rozptyl (var)": X_school.var()
    })
    print("Popisná statistika příznaků:")
    print(stats_df.round(2))
    var_body_mass = X_school["body_mass_g"].var()
    var_culmen_depth = X_school["culmen_depth_mm"].var()
    print(f"\nPoměr rozptylu body_mass_g / culmen_depth_mm: {var_body_mass / var_culmen_depth:.1f}x")
    print("Rozdíl v gramech zcela dominuje výpočtu Eukleidovské vzdálenosti!")

    # Srovnání škálovačů
    scalers = {
        "Neškálováno (Baseline)": None,
        "StandardScaler (Z-score)": StandardScaler(),
        "MinMaxScaler [0, 1]": MinMaxScaler(),
        "RobustScaler (IQR)": RobustScaler(),
    }

    scaling_results = {}
    for name, scaler in scalers.items():
        if scaler is None:
            model = KNeighborsClassifier(n_neighbors=5)
            model.fit(X_train_s, y_train_s)
            acc = model.score(X_test_s, y_test_s)
        else:
            pipe = Pipeline([("scaler", scaler), ("knn", KNeighborsClassifier(n_neighbors=5))])
            pipe.fit(X_train_s, y_train_s)
            acc = pipe.score(X_test_s, y_test_s)
        scaling_results[name] = acc
        print(f"--> {name:25s}: Accuracy = {acc * 100:.2f} %")

    # -------------------------------------------------------------------------
    # 6. Analýza hyperparametru k (Bias-Variance Tradeoff)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("5. ANALÝZA VLIVU HYPERPARAMETRU k (k in [1, 45])")
    print("=" * 70)
    k_range = list(range(1, 46))
    train_scores_unscaled = []
    test_scores_unscaled = []
    train_scores_scaled = []
    test_scores_scaled = []

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_s)
    X_test_scaled = scaler.transform(X_test_s)

    for k in k_range:
        # Neškálováno
        clf_unscaled = KNeighborsClassifier(n_neighbors=k)
        clf_unscaled.fit(X_train_s, y_train_s)
        train_scores_unscaled.append(clf_unscaled.score(X_train_s, y_train_s))
        test_scores_unscaled.append(clf_unscaled.score(X_test_s, y_test_s))

        # Škálováno
        clf_scaled = KNeighborsClassifier(n_neighbors=k)
        clf_scaled.fit(X_train_scaled, y_train_s)
        train_scores_scaled.append(clf_scaled.score(X_train_scaled, y_train_s))
        test_scores_scaled.append(clf_scaled.score(X_test_scaled, y_test_s))

    best_k_scaled = k_range[int(np.argmax(test_scores_scaled))]
    print(f"Optimální k pro škálovaný model: k = {best_k_scaled} (Test Accuracy = {max(test_scores_scaled) * 100:.2f} %)")

    # -------------------------------------------------------------------------
    # 7. Profesionální Pipeline s kategoriálními proměnnými
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("6. PROFESIONÁLNÍ PIPELINE SE VŠEMI PŘÍZNAKY (VČETNĚ OSTROVA A POHLAVÍ)")
    print("=" * 70)
    # Vyčištění dat se zachováním island a sex
    clean_df = raw_df.copy()
    # Oprava chybného záznamu pohlaví '.' pokud existuje
    clean_df = clean_df[clean_df["sex"] != "."]
    clean_df.dropna(inplace=True)

    X_full = clean_df.drop("species", axis=1)
    y_full = clean_df["species"]

    X_tr_f, X_te_f, y_tr_f, y_te_f = train_test_split(
        X_full, y_full, test_size=0.3, random_state=42, stratify=y_full
    )

    num_cols = ["culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g"]
    cat_cols = ["island", "sex"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
        ]
    )

    full_pipeline = Pipeline([
        ("prep", preprocessor),
        ("knn", KNeighborsClassifier()),
    ])

    param_grid = {
        "knn__n_neighbors": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
        "knn__weights": ["uniform", "distance"],
        "knn__p": [1, 2],  # Manhattan vs Euclidean
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(full_pipeline, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
    grid_search.fit(X_tr_f, y_tr_f)

    print(f"Nejlepší parametry z GridSearchCV:")
    print(grid_search.best_params_)
    print(f"Nejlepší validační přesnost v CV: {grid_search.best_score_ * 100:.2f} %")

    best_model = grid_search.best_estimator_
    y_pred_full = best_model.predict(X_te_f)
    full_test_acc = accuracy_score(y_te_f, y_pred_full)
    print(f"Výsledná testovací přesnost (plný model): {full_test_acc * 100:.2f} %")

    print("\nKlasifikační report (Precision, Recall, F1):")
    print(classification_report(y_te_f, y_pred_full))

    # -------------------------------------------------------------------------
    # 8. Vygenerování a uložení diagnostických grafů
    # -------------------------------------------------------------------------
    print("Generuji a ukládám grafy do 02_Classification/plots/...")

    # Graf 1: Křivka k vs Accuracy (Škálováno vs Neškálováno)
    plt.figure(figsize=(10, 5))
    plt.plot(k_range, test_scores_unscaled, "r--o", label="Neškálováno (Test Accuracy)", alpha=0.7)
    plt.plot(k_range, test_scores_scaled, "b-o", label="StandardScaler (Test Accuracy)", lw=2)
    plt.plot(k_range, train_scores_scaled, "b--", label="StandardScaler (Train Accuracy)", alpha=0.5)
    plt.axvline(best_k_scaled, color="green", linestyle=":", label=f"Optimum k={best_k_scaled}")
    plt.title("Vliv hyperparametru k a škálování na přesnost modelu k-NN (Palmer Penguins)", fontsize=12)
    plt.xlabel("Počet sousedů (k)", fontsize=11)
    plt.ylabel("Accuracy", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "knn_k_accuracy_curve.png", dpi=200)
    plt.close()

    # Graf 2: Srovnání škálovacích metod
    plt.figure(figsize=(8, 4))
    methods = list(scaling_results.keys())
    scores = [s * 100 for s in scaling_results.values()]
    colors = ["#d9534f", "#0275d8", "#5cb85c", "#f0ad4e"]
    bars = plt.bar(methods, scores, color=colors, width=0.55)
    plt.ylim(70, 103)
    plt.title("Srovnání přesnosti k-NN (k=5) podle metody předzpracování", fontsize=12)
    plt.ylabel("Testovací Accuracy (%)", fontsize=11)
    plt.grid(axis="y", alpha=0.3)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.1f} %", ha="center", fontweight="bold")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(plots_dir / "knn_scaling_comparison.png", dpi=200)
    plt.close()

    # Graf 3: Konfúzní matice plného modelu
    cm = confusion_matrix(y_te_f, y_pred_full, labels=best_model.classes_)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=best_model.classes_, yticklabels=best_model.classes_)
    plt.title("Konfúzní matice – Plný k-NN model (Pipeline + GridSearchCV)", fontsize=11)
    plt.xlabel("Predikovaná třída", fontsize=10)
    plt.ylabel("Skutečná třída", fontsize=10)
    plt.tight_layout()
    plt.savefig(plots_dir / "knn_confusion_matrix.png", dpi=200)
    plt.close()

    # Uložení předpočítaných výsledků pro Streamlit aplikaci
    precomputed_payload = {
        "school_acc": round(float(school_acc), 4),
        "scaling_results": {k: round(float(v), 4) for k, v in scaling_results.items()},
        "best_k_scaled": int(best_k_scaled),
        "k_range": k_range,
        "test_scores_unscaled": [round(float(s), 4) for s in test_scores_unscaled],
        "test_scores_scaled": [round(float(s), 4) for s in test_scores_scaled],
        "train_scores_scaled": [round(float(s), 4) for s in train_scores_scaled],
        "full_pipeline_best_params": grid_search.best_params_,
        "full_cv_acc": round(float(grid_search.best_score_), 4),
        "full_test_acc": round(float(full_test_acc), 4),
        "classes": list(best_model.classes_),
        "confusion_matrix": cm.tolist(),
    }
    with open(data_dir / "penguins_knn_precomputed.json", "w", encoding="utf-8") as f:
        json.dump(precomputed_payload, f, indent=2, ensure_ascii=False)

    print("Všechny výpočty i grafy byly úspěšně vygenerovány a uloženy!")


if __name__ == "__main__":
    main()
