"""
Scikit-learn Cvičení - Syntetický klasifikační dataset a k-NN model
Machine Learning Course - CodersLab

Úkoly:
1. Import potřebných modulů z knihovny scikit-learn:
   - z modulu `datasets`: metoda pro vytvoření syntetického datasetu pro klasifikaci (`make_classification`)
   - z modulu `model_selection`: metoda pro rozdělení datasetu na trénovací a testovací sadu (`train_test_split`)
   - z modulu `neighbors`: třída pro model k-nejbližších sousedů (`KNeighborsClassifier`)
   - z modulu `metrics` (v zadání zmíněn model_selection): metoda pro výpočet přesnosti modelu (`accuracy_score`)
2. Vytvoření syntetického datasetu s parametry:
   - Počet vzorků (samples): 1000
   - Počet proměnných (features): 10
   - Počet významných proměnných (informative features): 6
3. Rozdělení datasetu v poměru 70:30 (trénovací : testovací).
4. Vytvoření instance modelu k-NN s počtem sousedů k = 4.
5. Natrénování modelu na trénovací sadě a predikce na testovací sadě.
6. Výpočet hodnoty metriky accuracy na základě predikcí a skutečných popisků z testovací sady.
"""

# ==============================================================================
# 1. Importy
# ==============================================================================
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def main():
    print("=" * 70)
    print("SCIKIT-LEARN CVIČENÍ: SYNTETICKÁ KLASIFIKACE & k-NN (k = 4)")
    print("=" * 70)

    # ==============================================================================
    # 2. Vytvoření syntetického datasetu
    # ==============================================================================
    # n_samples = 1000: počet vzorků
    # n_features = 10: celkový počet proměnných
    # n_informative = 6: počet významných proměnných nesoucích informaci pro klasifikaci
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=6,
        random_state=42
    )
    print(f"\n1. Syntetický dataset vytvořen:")
    print(f"   - Matice příznaků X: {X.shape} (1000 řádků, 10 sloupců)")
    print(f"   - Cílový vektor y:   {y.shape} (1000 hodnot)")

    # ==============================================================================
    # 3. Rozdělení datasetu (poměr 70:30)
    # ==============================================================================
    # test_size = 0.30 odpovídá 30 % dat pro testování, 70 % pro trénování
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42
    )
    print(f"\n2. Rozdělení na trénovací a testovací sadu (70:30):")
    print(f"   - Trénovací sada (X_train, y_train): {X_train.shape[0]} vzorků (70 %)")
    print(f"   - Testovací sada  (X_test, y_test):   {X_test.shape[0]} vzorků (30 %)")

    # ==============================================================================
    # 4. Vytvoření instance modelu k-nejbližších sousedů (k = 4)
    # ==============================================================================
    knn = KNeighborsClassifier(n_neighbors=4)
    print(f"\n3. Model inicializován:")
    print(f"   - {knn}")

    # ==============================================================================
    # 5. Trénování modelu a predikce na testovací sadě
    # ==============================================================================
    knn.fit(X_train, y_train)
    print(f"\n4. Model úspěšně natrénován na trénovací sadě.")

    y_pred = knn.predict(X_test)
    print(f"   - Predikce na testovací sadě vygenerovány (počet predikcí: {len(y_pred)})")

    # ==============================================================================
    # 6. Výpočet a vyhodnocení metriky přesnosti (Accuracy)
    # ==============================================================================
    acc = accuracy_score(y_test, y_pred)
    model_score = knn.score(X_test, y_test)

    print("\n" + "=" * 70)
    print("VÝSLEDKY VYHODNOCENÍ MODELU")
    print("=" * 70)
    print(f"Skutečné popisky (prvních 15): {y_test[:15]}")
    print(f"Predikované popisky (prvních 15): {y_pred[:15]}")
    print("-" * 70)
    print(f"Přesnost modelu (accuracy_score): {acc:.4f} ({acc * 100:.2f} %)")
    print(f"Kontrolní ověření přes model.score(): {model_score:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
