# 🔬 Data Science & Machine Learning Portfolio (Streamlit App)

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.24+-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![Coders Lab](https://img.shields.io/badge/Vychází_z-Coders_Lab_CZ-00C48C?style=for-the-badge)](https://coderslab.cz/cz/)
[![AI Orchestration](https://img.shields.io/badge/AI_Orchestration-Google_Antigravity-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://deepmind.google/)

> Komplexní interaktivní výuková platforma a analytické portfolio pro regresní modelování, strojové učení a diagnostiku modelů v Pythonu.

---

## 🎓 Uznání autorství, AI orchestrace & Poděkování (Attribution)

### 🏫 Vzdělávací základ & Kurikulum
Základní pedagogická struktura, datové sady a cvičení v tomto projektu vycházejí z materiálů a sylabu akreditovaného kurzu **Data Science / Machine Learning** vzdělávací IT akademie **[Coders Lab Česká republika](https://coderslab.cz/cz/)**.

### 🤖 Realizace & Autonomní AI Orchestrace (Google Antigravity)
Tento repozitář byl koncipován, architektonicky navržen a softwarově implementován formou **pokročilého párového programování (Pair Programming) a agentní AI orchestrace v prostředí [Google Antigravity (Antigravity IDE)](https://deepmind.google/)**.

- **Synergie člověk + AI:** Uživatel definuje strategické zadání, věcné mantinely a ověřuje pedagogickou integritu; AI agent Antigravity obstarává end-to-end softwarové inženýrství, matematické formulace v KaTeX, konstrukci Scikit-learn pipeline, diagnostiku chyb, generování interaktivních komponent v Plotly/Streamlit a automatické spouštění Jupyter sešitů.
- **Přidaná hodnota nad rámec původních osnov:**
  - Moderní **Streamlit webová aplikace** s interaktivními vizualizacemi v Plotly a hierarchickou navigací.
  - **What-If inferenční simulátory** napojené na reálně natrénované Scikit-learn modely v reálném čase.
  - **Expertní diagnostika:** Gauss-Markovovy teorémy, analýza multikolinearity (VIF), prostorová mapa odchylek v cenách nemovitostí v Seattlu, Cost-Complexity Pruning rozhodovacích stromů a geometrický rozbor metrik k-NN.
  - **Rigorózní metodické rozbory:** eliminace Data Leakage, správná práce s křížovou validací (`GridSearchCV`, `StratifiedKFold`) a ochrana před přetrénováním.

---

## 🌟 O projektu & Architektura aplikace

Aplikace slouží jako referenční příručka a praktická laboratoř pro **Den 1: Regresní modely (Supervised Learning)**. Každé téma je rozděleno do tří provázaných pilířů:

1. **📖 Teoretický základ:** Matematické formulace ($y = \mathbf{X}\boldsymbol{\beta} + \varepsilon$, ztrátové funkce, regularizační pokuty), geometrické interpretace a metodické poznámky.
2. **🎯 Školní výsledky zadání:** Přesné splnění úloh z kurzu včetně natrénování modelů a vyhodnocení základních metrik.
3. **🔬 Expertní analýza & Kritika:** Hlubší vhled za hranice základních zadání (analýza časového posunu u nemovitostí, gemologické paradoxy u diamantů, Rungeův jev u polynomů, prořezávání stromů).

```mermaid
flowchart TD
    A["00. Den 0: Prework (Úvod do DS & ML)"] --> B["01. Den 1: Lineární regrese OLS"]
    B --> C["02. Den 1: Metriky regresních modelů"]
    C --> D["03. Den 1: Regularizace (Ridge, Lasso, Elastic Net)"]
    D --> E["04. Den 1: Polynomiální regrese"]
    E --> F["05. Den 1: Rozhodovací stromy & Ansámbly"]
    F --> G["06. Den 1: Závěr Dne 1 (Velká syntéza regrese)"]
    G --> H["07. Den 1: Extras (Expertní laboratoř)"]
    H --> I["08. Den 2: K-Nearest Neighbors (k-NN)"]
    I --> J["09. Den 2: Metriky klasifikačních modelů"]
```

---

## 📊 Reálné datasety & Benchmarky

Experimenty jsou prováděny na reálných datových sadách z praxe:

### 1. Nemovitosti v King County, Seattle (`kc_house_data.csv`)
- **21 613 prodejů domů** z let 2014–2015.
- Cíl: Predikce prodejní ceny na základě plochy, lokality, stavu a počtu pokojů.
- **Baseline OLS:** $R^2 \approx 0.695$, $\text{RMSE} \approx 203\,150\text{ USD}$
- **Optimální rozhodovací strom (GridSearchCV):** $R^2 = 0.793$, $\text{RMSE} = 176\,810\text{ USD}$  
  *(Díky schopnosti ohraničit prestižní čtvrti pomocí souřadnic `lat` a `long` strom snížil chybu RMSE o téměř 35 000 USD).*

### 2. Diamanty (`diamonds.csv`)
- **53 908 certifikovaných diamantů** hodnocených dle gemologických standardů 4C (Carat, Cut, Color, Clarity).
- Cíl: Predikce ceny a zachycení nelineárního růstu hodnoty s hmotností.
- **Lineární OLS:** $R^2 = 0.9095$, $\text{RMSE} = 1\,172.5\text{ USD}$
- **Polynom 2. stupně (Kvadratický OLS):** $R^2 = 0.9655$, $\text{RMSE} = 723.7\text{ USD}$
- **Polynom 3. stupně (Surový OLS):** $R^2 = 0.8591$, $\text{RMSE} = 1\,463.0\text{ USD}$ *(Kolaps variance kvůli multikolinearitě 219 členů!)*
- **Polynom 3. stupně + Ridge ($\alpha=100$):** $R^2 = 0.9744$, $\text{RMSE} = 623.2\text{ USD}$ *(Záchrana modelu regularizací)*
- **Optimální rozhodovací strom (GridSearchCV):** $R^2 = 0.9785$, $\text{RMSE} = 587.0\text{ USD}$, $\text{MAE} = 307.3\text{ USD}$  
  *(Dokonale zachycuje psychologické cenové skoky u kulatých karátů `carat >= 1.00`).*

### 3. Antarktičtí tučňáci Palmer Penguins (`penguins_size.csv`)
- **344 tučňáků** tří druhů (*Adelie*, *Chinstrap*, *Gentoo*) ze souostroví Palmer.
- Cíl: Multitřídní klasifikace druhu na základě morfologie (zobák, ploutev, hmotnost, ostrov, pohlaví).
- **Školní baseline (bez škálování, $k=5$):** $\text{Accuracy} \approx 80.6\,\%$ *(Hmotnost v gramech tvořila $99.98\,\%$ celkové vzdálenosti!)*
- **Plný Pipeline s OneHot kódováním ostrova a pohlaví:** $\text{Accuracy} = 99.0\,\%$, $\text{CV} = 100.0\,\%$

### 4. Biomechanika bederní páteře (`lumbar_data.csv`)
- **310 pacientů** (100 zdravých `Normal`, 210 s diagnózou `Abnormal`: výhřez ploténky *Hernia* a posun obratle *Spondylolisthesis*).
- Cíl: Klinická klasifikace stavu páteře na základě 6 anatomických úhlů pánve.
- **Školní model ($k=5$, $L_2$ normalizace):** $\text{Accuracy} = 73.08\,\%$, $\text{Recall} = 80.70\,\%$, $\text{ROC-AUC} = 0.8208$.
- **Optimální model se stratifikací ($k=8$):** $\text{Accuracy} = 88.46\,\%$, $\text{Recall} = 88.68\,\%$.

---

## 🎛️ Hlavní interaktivní funkce (Day 1 Extras)

- **🎛️ What-If simulátor inference:** Kalkulátor cen domů i diamantů volající **skutečně natrénované Scikit-learn pipeliny** v reálném čase.
- **🗺️ Geografická mapa nemovitostí v Seattlu:** Interaktivní mapa 1 500 domů s barevným vyznačením odchylek v odhadu cen ($y - \hat{y}$) odhalující, kde přesně OLS selhává a jak strom lokalizuje bohaté pobřežní oblasti.
- **🔬 Diagnostika Gauss-Markov:** Ověření předpokladů pro vlastnost BLUE (Best Linear Unbiased Estimator) a interaktivní analýza multikolinearity přes VIF faktor.
- **🌲 Cost-Complexity Pruning (`ccp_alpha`):** Vizuální sledování prořezávání rozhodovacího stromu od masivního přeučení ($>36\,000$ listů) až k optimálnímu generalizovanému modelu.
- **🌳 Průvodce výběrem modelu:** Interaktivní rozhodovací diagram v SVG pro volbu správného algoritmu v byznysové praxi.
- **📥 Scikit-learn Cheatsheet:** Kompletní tahák v PDF ke stažení vygenerovaný přes ReportLab.

---

## 🚀 Jak aplikaci spustit lokálně (Quickstart)

### 1. Klonování repozitáře
```bash
git clone https://github.com/Kilitar/Coderslab-DataScience.git
cd Coderslab-DataScience
```

### 2. Vytvoření virtuálního prostředí
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalace závislostí
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Spuštění Streamlit aplikace
```bash
streamlit run app.py
```
Aplikace se automaticky otevře ve vašem výchozím webovém prohlížeči na adrese `http://localhost:8501`.

---

## 📂 Struktura repozitáře

```text
Coderslab-DataScience/
├── app.py                                # Hlavní vstupní bod Streamlit aplikace (navigace)
├── requirements.txt                      # Seznam závislostí pro běh projektu
├── 01_Regression/                        # Podklady, skripty a teorie pro Den 1
│   ├── 01_linear_regression_exercise_1.py
│   ├── 02_linear_regression_exercise_2.py
│   ├── 03_regression_metrics_exercise_1.py
│   ├── 04_regression_metrics_exercise_2.py
│   ├── 05_regularization_exercise_1.py
│   ├── 06_regularization_exercise_2.py
│   ├── 07_polynomial_regression_exercise.py
│   ├── 08_decision_tree_exercise_1.py
│   ├── 09_decision_tree_exercise_2.py
│   ├── data/                            # Datové sady (CSV) a optimalizované JSON cache
│   │   ├── kc_house_data.csv
│   │   ├── diamonds.csv
│   │   ├── diamonds_poly_precomputed.json
│   │   └── kc_time_analysis_precomputed.json
│   └── theory/                          # Detailní teoretické průvodce a rekapitulace
│       ├── 01_linear_regression_theory.md
│       ├── 02_regression_metrics_theory.md
│       ├── 03_regularization_theory.md
│       ├── 05_polynomial_regression_theory.md
│       ├── 06_decision_tree_regression_theory.md
│       └── 07_day_1_summary.md
├── 02_Classification/                    # Podklady, skripty a teorie pro Den 2 (Klasifikace)
│   ├── 01_knn_penguins_exercise.py       # Úvodní analýza: Palmer Penguins (Analýza selhání neškálování)
│   ├── 01_knn_penguins_exercise.ipynb    # Vypracovaný sešit úvodní demonstrace
│   ├── 02_knn_lumbar_exercise_1.py       # Cvičení 1: Diagnostika bederní páteře (10 kroků zadání, L2 norma)
│   ├── 02_knn_lumbar_exercise_1.ipynb    # Vypracovaný a spuštěný Jupyter Notebook páteře
│   ├── 03_knn_penguins_exercise_2.py     # Cvičení 2: Druhy tučňáků (9 kroků zadání, normalizace, k in [1, 35])
│   ├── 03_knn_penguins_exercise_2.ipynb  # Vypracovaný a spuštěný Jupyter Notebook tučňáků
│   ├── 04_classification_metrics_lumbar_exercise_1.py # Cvičení 1: Metriky klasifikace páteře (CM, Precision, Recall, F1, k-sweep)
│   ├── 04_classification_metrics_lumbar_exercise_1.ipynb # Vypracovaný a spuštěný Jupyter Notebook metrik
│   ├── 05_classification_metrics_penguins_exercise_2.py # Cvičení 2: Multiclass metriky tučňáků (CM 3x3, Weighted F1, k-sweep)
│   ├── 05_classification_metrics_penguins_exercise_2.ipynb # Vypracovaný a spuštěný Jupyter Notebook multiclass metrik
│   ├── data/                            # Datové sady (Palmer Penguins, Lumbar) a JSON cache
│   │   ├── lumbar_data.csv
│   │   ├── lumbar_normalized_df.csv     # Výsledný normalizovaný dataset páteře
│   │   ├── lumbar_df_normalized.csv
│   │   ├── lumbar_knn_precomputed.json
│   │   ├── lumbar_metrics_exercise_1_precomputed.json # Předpočtené metriky páteře, sweep k in [1, 25] a ROC
│   │   ├── penguins_size.csv
│   │   ├── penguins_df_normalized.csv   # Výsledný normalizovaný dataset tučňáků dle kroku 9
│   │   ├── penguins_knn_precomputed.json
│   │   ├── penguins_exercise_2_precomputed.json
│   │   └── penguins_metrics_exercise_2_precomputed.json # Předpočtené multiclass metriky a sweep k in [1, 35]
│   ├── plots/                           # Diagnostické PNG vizualizace
│   │   ├── lumbar_knn_k_curve.png
│   │   ├── lumbar_knn_confusion_matrix.png
│   │   ├── lumbar_metrics_cm_heatmap.png # Matice záměn páteře (k=5)
│   │   ├── lumbar_metrics_k_sweep.png    # Křivky metrik Train vs Test, Recall, F1
│   │   ├── lumbar_metrics_roc_curve.png  # ROC křivka páteře (AUC = 0.821)
│   │   ├── penguins_ex2_k_curve.png
│   │   ├── penguins_ex2_confusion_matrix.png
│   │   ├── penguins_metrics_cm_k5.png    # Matice záměn tučňáků pro k=5
│   │   ├── penguins_metrics_cm_k2.png    # Matice záměn tučňáků pro optimum k=2
│   │   ├── penguins_metrics_k_sweep.png  # Křivky metrik pro k in [1, 35]
│   │   ├── knn_k_accuracy_curve.png
│   │   ├── knn_scaling_comparison.png
│   │   └── knn_confusion_matrix.png
│   └── theory/                          # Teoretické markdown příručky
│       ├── 01_knn_theory.md             # Eukleidovská, Manhattanská, Minkowského metrika, Voronoi, kletba dimenzionality
│       ├── 02_knn_implementation_guide.md # Scikit-learn KNeighborsClassifier, kd-tree/ball-tree, .kneighbors()
│       └── 03_classification_metrics_theory.md # Matice záměn, Precision, Recall, Specificity, F1, ROC-AUC, Log Loss
└── views/                               # Jednotlivé podstránky Streamlit aplikace
    ├── 00_home.py                       # Úvodní rozcestník a sylabus
    ├── 00_prework_*.py                  # Moduly přípravného bloku (Prework)
    ├── 01_*.py                          # Stránky úloh, notebooků a analýz Dne 1
    ├── 01_extras_*.py                   # Day 1 Extras (What-If, Mapy, Pruning, Taháky)
    ├── 02_knn_*.py                      # Den 2 moduly k-NN (Teorie, Implementace, Výsledky, Expertní analýzy, Notebooky)
    └── 02_metrics_*.py                  # Den 2 moduly metrik (Teorie matice záměn, Multiclass evaluace, Expertní analýzy & Cost matrix, Notebooky)
```

---

## 🛠️ Použité technologie & Knihovny

- **AI Orchestrace & Vývojové prostředí:** [Google Antigravity IDE](https://deepmind.google/) (Agentic Pair Programming & Workflow Orchestration)
- **Frontend & Dashboard:** [Streamlit](https://streamlit.io/)
- **Machine Learning & Preprocessing:** [Scikit-learn](https://scikit-learn.org/) (LinearRegression, Ridge, Lasso, ElasticNet, DecisionTreeRegressor, HistGradientBoostingRegressor, KNeighborsClassifier, StandardScaler, Normalizer, PolynomialFeatures)
- **Data Wrangling:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Interaktivní vizualizace:** [Plotly Express & Graph Objects](https://plotly.com/python/)
- **Generování PDF:** [ReportLab](https://www.reportlab.com/)

---

## 📜 Licence & Podmínky užití

Tento projekt je určen pro vzdělávací a studijní účely v rámci studia Data Science. Původní koncepty úloh náleží vzdělávací akademii **[Coders Lab](https://coderslab.cz/cz/)**. Kód implementace aplikace, interaktivní moduly a doplňující analýzy jsou volně k dispozici pod licencí MIT.
