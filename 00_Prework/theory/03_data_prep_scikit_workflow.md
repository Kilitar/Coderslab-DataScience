# Příprava dat, Scikit-learn a životní cyklus modelu (Prework 03)

> **Zdrojový materiál kurzu:**  
> `8-Scikit_learn.txt`, `9-ML_Related_Problems.txt`, `10-Data_preparation.txt`  
> (Čištění dat, imputace, kódování, škálování, prevence Data Leakage, Scikit-learn API, MLOps úskalí 2026)

---

## 1. Zlaté pravidlo: Garbage In, Garbage Out (GIGO)

V praxi se často říká, že **až 80 % času datového vědce zabere čištění a příprava dat**. Žádný, ani ten nejpokročilejší algoritmus (včetně hlubokých neuronových sítí), nedokáže vykouzlit spolehlivé predikce z dat plných duplicit, chybějících hodnot, nekonzistentních formátů a metodických chyb.

Nekvalitní příprava dat vede k:
- Zkresleným parametrům modelu a falešně optimistickým metrikám.
- Finančním ztrátám při nasazení do produkce.
- Selhání modelu v reálném světě (tichý kolaps přesnosti).

---

## 2. Klíčové kroky přípravy dat (Preprocessing Pipeline)

```text
    ┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │  Surová data    │ ──> │ Odstranění chyb  │ ──> │   Kódování      │ ──> │   Škálování     │
    │  (Raw Dataset)  │     │ a imputace NaN   │     │ kategorií (OHE) │     │  (Standardize)  │
    └─────────────────┘     └──────────────────┘     └─────────────────┘     └─────────────────┘
                                                                                      │
                                                                                      v
                                                                             ┌─────────────────┐
                                                                             │  Trénování ML   │
                                                                             │     modelu      │
                                                                             └─────────────────┘
```

### A. Duplicity a chybějící hodnoty (Missing Data)
- **Duplicity:** Opakující se identické řádky zkreslují rozdělení a převažují váhu některých vzorků. V Pandas odstraňujeme pomocí `df.drop_duplicates()`.
- **Chybějící hodnoty (`NaN` / `None`):**
  1. *Odstranění (`df.dropna()`):* Vhodné pouze tehdy, když chybí jen zanedbatelné procento řádků (< 1–2 %) a chybění je zcela náhodné.
  2. *Imputace (Doplnění):*
     - **Numerická data:** Mediánem (u šikmých rozdělení) nebo průměrem (u normálních rozdělení) – `SimpleImputer(strategy='median')`.
     - **Kategoriální data:** Nejčastější hodnotou (modus) nebo vytvořením nové explicitní kategorie `"Unknown"` – `SimpleImputer(strategy='most_frequent')`.
     - **Pokročilé metody:** `KNNImputer` (doplnění podle $k$ nejbližších sousedních řádků) nebo `IterativeImputer` (vícenásobná regrese).

### B. Kódování kategoriálních proměnných (Categorical Encoding)
Většina matematických modelů rozumí pouze číslům. Textové kategorie musíme transformovat:

1. **One-Hot Encoding (OHE):**
   - Pro nominální kategorie **bez přirozeného pořadí** (např. barva: červená, modrá, zelená; značka vozu).
   - Vytvoří pro každou unikátní hodnotu samostatný binární sloupec (0 nebo 1).
   - *Pozor na multicollinearitu:* V lineárních modelech vždy nastavujeme `drop='first'`, abychom eliminovali redundantní sloupce.
2. **Ordinal Encoding:**
   - Pro ordinální kategorie **s jasnou hierarchií a pořadím** (např. vzdělání: Základní < Střední < Vysokoškolské; nebo kvalita brusu: Fair < Good < Ideal).
   - Přiřazuje rostoucí celá čísla ($0, 1, 2, \dots$).
3. **Target / Frequency Encoding:**
   - Pro sloupce s obrovským počtem unikátních hodnot (vysoká kardinalita – např. PSČ nebo ID prodejny), kde by One-Hot Encoding vytvořil tisíce sloupců a vedl ke kletbě dimenzionality.

### C. Škálování příznaků (Feature Scaling)
Různé sloupce mají často dramaticky odlišné rozsahy (např. věk: 18–70 let vs. roční příjem: 300 000 – 5 000 000 Kč). 

1. **Standardizace (Z-score Normalization):**
   $$z = \frac{x - \mu}{\sigma}$$
   - Transformuje data tak, aby měla **střed $\mu = 0$ a směrodatnou odchylku $\sigma = 1$**.
   - `StandardScaler` v Scikit-learn. Neomezuje rozsah na pevný interval, zachovává outliery.
2. **Min-Max Normalizace (MinMaxScaler):**
   $$x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}} \in [0, 1]$$
   - Stlačí všechna data přesně do intervalu $[0, 1]$. Je extrémně náchylná na odlehlé hodnoty.
3. **RobustScaler:**
   - Využívá medián a mezichvartilové rozpětí (IQR). Ideální volba pro data s extrémními outliery.

> [!IMPORTANT]
> **Kdy MUSÍTE škálovat?**  
> Vždy u modelů závislých na vzdálenostech nebo gradientním sestupu: **Lineární regrese, Ridge, Lasso, Logistic Regression, k-NN, SVM, Neuronové sítě**.  
> **Kdy škálovat NEMUSÍTE?**  
> U stromových algoritmů (**Decision Tree, Random Forest, XGBoost, LightGBM**), které se rozhodují pouze na základě uspořádání hodnot.

---

## 3. Kardinální hřích ML: Data Leakage (Únik informací)

Jednou z nejničivějších metodických chyb v praxi je **Data Leakage**. Jde o situaci, kdy se informace z testovací nebo validační sady neúmyslně dostanou do procesu trénování.

### ❌ Špatný postup (Častá chyba začátečníků):
```python
# CHYBA: Škálování celého datasetu před rozdělením!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) # Vypočítal průměr i z budoucích testovacích dat!
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)
```

### ✅ Správný postup:
```python
# SPRÁVNĚ: Nejprve striktní rozdělení, fitování POUZE na train!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # Spočte mu a sigma pouze z trénovacích dat
X_test_scaled = scaler.transform(X_test)       # Aplikuje stejné parametry na test
```

---

## 4. Jednotná architektura Scikit-learn (Unified API)

Síla Scikit-learn spočívá v přísně dodržovaném a konzistentním objektovém rozhraní:

1. **Estimator (Odhadce):**
   - Jakýkoliv objekt schopný se učit z dat.
   - Metoda: `.fit(X, y)`
2. **Transformer (Transformátor):**
   - Objekt pro předúpravu dat (škálování, imputace, enkódování).
   - Metody: `.transform(X)` a zrychlené `.fit_transform(X)`
3. **Predictor (Prediktor):**
   - Natrénovaný model schopný dělat odhady na nových datech.
   - Metody: `.predict(X)`, `.predict_proba(X)` (pro klasifikaci), `.score(X, y)`
4. **Pipeline & ColumnTransformer:**
   - Zlatý standard pro produkční kód: spojuje preprocessing a model do jednoho atomického celku, který **zcela vylučuje Data Leakage**.

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), ["age", "income"]),
    ("cat", OneHotEncoder(drop="first"), ["gender", "education"])
])

model_pipeline = Pipeline(steps=[
    ("prep", preprocessor),
    ("regressor", Ridge(alpha=10.0))
])

# Jeden fit a jeden predict – bezpečně, čistě, bez úniku dat
model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)
```

---

## 5. Úskalí a limity v praxi (MLOps & Model Lifecycle 2026)

Trénováním modelu práce nekončí – naopak teprve začíná:

### 1. Kompromis mezi zkreslením a rozptylem (Bias-Variance Tradeoff)
- **Vysoké zkreslení (Underfitting):** Model je příliš jednoduchý (např. přímka na zakřivená data). Má špatný výkon jak na trénovací, tak na testovací sadě.
- **Vysoký rozptyl (Overfitting):** Model je příliš složitý (např. neomezený strom nebo polynom 15. stupně). Naučil se šum a specifika trénovacích dat nazpaměť. Na trénovací sadě má $R^2 = 0.99$, ale na testovací sadě zkolabuje.
- **Optimální bod:** Minimalizuje celkovou chybu $\text{Chyba} = \text{Bias}^2 + \text{Variance} + \text{Šum}$.

### 2. Stárnutí modelů v produkci (Drift)
- **Data Drift (Covariate Shift):** Mění se rozdělení vstupních veličin $P(X)$ (např. na web začnou chodit uživatelé z jiné demografické skupiny nebo z jiné země).
- **Concept Drift:** Mění se samotný vztah mezi vstupy a výstupem $P(y \mid X)$ (např. změna spotřebitelského chování během ekonomické krize nebo pandemie – historická data přestávají platit).
- *Řešení:* Kontinuální monitoring metrik v produkci a automatizované přetrénovávání (Retraining triggers).

### 3. Vysvětlitelnost a etika (Explainable AI – XAI)
S platností evropského nařízení **EU AI Act (2024–2026)** roste důraz na to, aby rozhodnutí modelů s vysokým rizikem (např. credit scoring, nábor zaměstnanců, medicína) byla zpětně auditovatelná. Využívají se metody jako **SHAP (Shapley Additive Explanations)** a **LIME**, které dokáží rozklíčovat podíl jednotlivých faktorů na každé jednotlivé predikci.

---

## 6. 📚 Doporučená literatura a oficiální zdroje (2026)

Pro hlubší studium a každodenní praxi doporučujeme tyto ověřené zdroje:

1. **Oficiální dokumentace:**
   - [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html) – jeden z nejlépe zpracovaných manuálů v historii open-source softwaru.
   - [Pandas Documentation](https://pandas.pydata.org/docs/) – reference pro manipulaci s datovými rámci.
   - [Plotly Python Open Source Graphing Library](https://plotly.com/python/) – interaktivní vizualizace.
2. **Knihy považované za bibli oboru:**
   - *Aurélien Géron:* **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow** (3rd Edition, O'Reilly) – praktická kniha s důrazem na kód.
   - *Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani:* **An Introduction to Statistical Learning (ISLR)** – volně dostupná kniha (včetně Python verze) s dokonalým matematicko-statistickým základem.
   - *Chip Huyen:* **Designing Machine Learning Systems** (O'Reilly) – standard pro MLOps a nasazování modelů do produkce.
