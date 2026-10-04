# Den 3: Náhodný les (Random Forest) – Teorie, Implementace v Scikit-learn & SOTA 10/2026

> **Cíl modulu:** Detailně prostudovat algoritmus **Náhodný les (*Random Forest*)** jako nejdůležitější a nejpoužívanější aplikaci techniky **Bagging** (*Bootstrap Aggregating*). Rozebrat princip **dvojité náhodnosti** (*Feature Bagging* / *Random Subspaces*), matematické důvody redukce rozptylu, volbu počtu stromů přes **Out-Of-Bag (OOB)** evaluaci, kompletní API v knihovně **Scikit-learn** (`RandomForestClassifier` a `RandomForestRegressor`), benchmark na realitních datech z Melbourne a doplnit expertní kritiku a moderní rozšíření k **říjnu 2026**.

---

## 1. Originální teorie kurzu (Přehled konceptů z prezentací)

### 1.1 Proč zrovna rozhodovací stromy v ansámblu?
V předchozí části o Baggingu jsme se seznámili s kompromisem mezi vychýlením a rozptylem (*Bias-Variance Trade-off*):
- **Mělké stromy (např. Decision Tree Stumps):** Mají nízký rozptyl, ale vysoké vychýlení (*high bias*). Jsou vhodné pro sekvenční metody (**Boosting**).
- **Hluboké stromy (plně vyrostlé stromy bez ořezání):** Dokáží se perfektně přizpůsobit i nejsložitějším nelineárním vazbám v trénovacích datech. Mají **velmi nízké vychýlení (*low bias*)**, ale **extrémně vysoký rozptyl (*high variance*)** – jsou přeučené (*overfitted*) a přecitlivělé na drobný šum.

Protože technika **Bagging** funguje jako reduktor rozptylu, jsou plně vyrostlé hluboké rozhodovací stromy **ideálním bázovým modelem** pro paralelní ansámbl.

---

### 1.2 Klíčový trik Random Forestu: Dvojitá náhodnost (*Feature Bagging*)
Samotný Bagging aplikovaný na stromy náhodně sampluje pouze trénovací pozorování (řádky) s vracením. **Leo Breiman** (2001) však zjistil, že pokud v datech existuje několik málo dominantních prediktorů, téměř každý strom v klasickém Baggingu si vybere tento prediktor jako kořenový uzel. Výsledkem jsou stromy, které sice trénovaly na různých řádcích, ale jejich predikce jsou silně korelované ($\rho > 0$).

**Random Forest zavádí druhý stupeň náhodnosti:**
1. **Náhodný výběr vzorků (Bootstrapping):** Každý strom se trénuje na náhodném výběru $N$ pozorování s vracením (*with replacement*).
2. **Náhodný výběr příznaků (*Random Subspaces / Feature Bagging*):** Při každém dělení uzlu v každém stromu se nehledá nejlepší dělicí kritérium přes všechny příznaky, nýbrž **pouze z náhodně vylosované podmnožiny $m$ příznaků** (kde $m < p$, typicky $m \approx \sqrt{p}$ pro klasifikaci a $m \approx \frac{p}{3}$ pro regresi).

```
Původní dataset: 10 příznaků (f1 až f10)
  ├── Strom 1: trénován na náhodných řádcích a podmnožině příznaků [f1, f4, f8]
  ├── Strom 2: trénován na náhodných řádcích a podmnožině příznaků [f3, f5, f6]
  └── Strom 3: trénován na náhodných řádcích a podmnožině příznaků [f2, f7, f9, f10]
```

**Důsledek:** Každý strom se dívá na data z úplně jiné perspektivy. Dojde k **drastickému snížení korelace $\rho$ mezi stromy**, což podle vzorce:
$$\text{Var}(\text{Ansámbl}) = \rho \sigma^2 + \frac{1 - \rho}{M} \sigma^2$$
srazí celkový rozptyl modelu na minimum!

---

### 1.3 Příklady fungování ze slajdů kurzu

#### A. Klasifikace: Výběr destinace dovolené
- **Dataset preferencí zákazníků:** rozpočet, počet dní, typ dopravy, počet cestujících, přítomnost dětí, preferovaná teplota, zájem o výlety.
- **Kritérium dělení uzlů:** *Gini index* nebo *Entropie* (Informační zisk).
- **Ansámbl o 5 stromech:**
  - Strom 1: Španělsko
  - Strom 2: Portugal sko
  - Strom 3: Španělsko
  - Strom 4: Španělsko
  - Strom 5: Portugalsko
- **Agregace (Většinové hlasování / Majority Voting):** Poměr 3:2 $\to$ Finální doporučení: **Španělsko**.

#### B. Regrese: Odhad ceny zájezdu
- **Dataset charakteristik cesty:** počet dní, typ dopravy, počet cestujících, počet dětí, stravování.
- **Kritérium dělení uzlů:** *MSE* (*Mean Squared Error*) nebo *MAE* (*Mean Absolute Error*).
- **Ansámbl o 3 stromech predikuje cenu zájezdu:**
  - Strom 1: 4 000 AUD
  - Strom 2: 4 600 AUD
  - Strom 3: 4 950 AUD
- **Agregace (Aritmetický průměr):**
  $$\hat{y} = \frac{4000 + 4600 + 4950}{3} = \frac{13550}{3} = 4516.67\text{ AUD}$$

---

### 1.4 Jak volit počet stromů (`n_estimators`) a role Out-of-Bag (OOB)
- Při malém počtu stromů se může stát, že některá pozorování nebo důležité příznaky nebudou vůbec vybrány do tréninku.
- Díky losování s vracením zůstává u každého stromu přibližně **36.8 % dat nevyužito** (tzv. **Out-of-Bag – OOB data**).
- **OOB evaluace:** OOB vzorky slouží jako bezplatná validační sada. Každé trénovací pozorování je vyhodnoceno pouze těmi stromy, v jejichž trénovacím bootstrap vzorku se **nenacházelo**.
- **Určení optimálního počtu stromů:** Postupně zvyšujeme počet stromů a sledujeme OOB skóre (nebo OOB chybu). Jakmile se křivka OOB chyby zploští a přestane klesat, další přidávání stromů již nezvyšuje přesnost, pouze prodlužuje dobu výpočtu.

---

### 1.5 Silné a slabé stránky Random Forestu

| Výhody (Silné stránky) | Nevýhody (Slabé stránky) |
| :--- | :--- |
| **Vysoká predikční přesnost:** Zvládá komplexní, vysoce nelineární interakce mezi proměnnými. | **Slabší interpretovatelnost:** Oproti jednomu stromu jde o „černou skříňku“ tvořenou stovkami stromů. |
| **Odolnost vůči přeučení:** Díky dvojité náhodnosti model skvěle generalizuje. | **Výpočetní a paměťová náročnost:** Trénování a uchování stovek stromů vyžaduje více RAM a CPU. |
| **Imunita vůči chybějícím datům:** Teoretická odolnost při výběru náhradních štěpení (*surrogate splits*). | **Neschopnost extrapolace v regresi:** Predikce je ohraničena minimem a maximem trénovacích hodnot. |
| **Měření důležitosti příznaků:** Vestavěný výpočet *Feature Importance* (MDI). | **Pomalá inference:** Predikce vyžaduje průchod všemi stromy (např. 500 stromů při nízké latenci). |

---

## 2. Implementace v knihovně Scikit-learn

V knihovně `scikit-learn` máme pro Random Forest dvě paralelní třídy v modulu `sklearn.ensemble`:
1. `RandomForestClassifier` – pro úlohy klasifikace.
2. `RandomForestRegressor` – pro úlohy regrese.

Obě třídy sdílejí téměř totožné rozhraní a parametry:

### 2.1 Klíčové hyperparametry konstruktoru
- `n_estimators` *(int, default=100)*: Počet rozhodovacích stromů v lese.
- `criterion` *(str)*:
  - Pro klasifikaci: `"gini"` (výchozí) nebo `"entropy"` / `"log_loss"`.
  - Pro regresi: `"squared_error"` (MSE, výchozí), `"absolute_error"` (MAE), `"friedman_mse"`, `"poisson"`.
- `max_depth` *(int, default=None)*: Maximální hloubka stromů. `None` znamená neomezenou hloubku (stromy rostou až do čistých listů nebo limitu vzorků).
- `min_samples_split` *(int nebo float, default=2)*: Minimální počet vzorků nutný k rozdělení vnitřního uzlu.
- `min_samples_leaf` *(int nebo float, default=1)*: Minimální počet vzorků požadovaný v koncovém listu.
- `max_features` *(int, float, str, default="sqrt" pro klasifikaci / 1.0 pro regresi)*: Počet náhodně vybíraných příznaků při každém štěpení uzlu.
- `max_leaf_nodes` *(int, default=None)*: Maximální počet listů v každém stromu.
- `min_impurity_decrease` *(float, default=0.0)*: K rozdělení uzlu dojde pouze v případě, že snížení nečistoty je větší nebo rovno této hodnotě.
- `bootstrap` *(bool, default=True)*: Zda se mají vytvářet bootstrap vzorky s vracením. Pokud `False`, použije se pro všechny stromy celý dataset (vhodné pouze s `max_features < 1.0`).

### 2.2 Runtime a řídicí parametry
- `random_state` *(int, default=None)*: Seed pseudonáhodného generátoru pro reprodukovatelnost.
- `n_jobs` *(int, default=None)*: Počet paralelních procesů pro trénink a predikci. Hodnota `-1` využije všechna dostupná CPU jádra.
- `class_weight` *(dict nebo "balanced", default=None)*: Vyvážení vah tříd při nevyváženém datasetu (pouze u klasifikace).
- `warm_start` *(bool, default=False)*: Při nastavení na `True` lze volat `fit()` opakovaně a přidávat nové stromy k existujícímu lesu.
- `oob_score` *(bool, default=False)*: Zda počítat Out-of-Bag skóre. Pro klasifikaci vrací Accuracy, pro regresi koeficient determinace $R^2$.

### 2.3 Atributy po natrénování (`fit`)
- `estimators_`: Seznam natrénovaných instancí `DecisionTreeClassifier` / `DecisionTreeRegressor`.
- `feature_names_in_`: Názvy nezávislých proměnných použitých při tréninku.
- `feature_importances_`: Pole relativních důležitostí jednotlivých proměnných (součet je roven 1.0).
- `oob_score_`: Vypočtené skóre na OOB datech (pokud `oob_score=True`).

### 2.4 Základní metody
- `.fit(X, y)`: Natrénování lesa na datech.
- `.predict(X)`: Predikce cílové veličiny pro nová data.
- `.score(X, y)`: Výpočet výchozí metriky (Accuracy pro klasifikaci, $R^2$ pro regresi).
- `.predict_proba(X)`: Predikce pravděpodobností příslušnosti ke třídám (pouze pro klasifikaci).

---

### 2.5 Benchmark z přednášky: Reality Melbourne (`melb_house_data.csv`)

V prezentaci je demonstrováno srovnání samotného rozhodovacího stromu a náhodného lesa při predikci cen nemovitostí v Melbourne:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# 1. Příprava dat
melbourne_data = pd.read_csv("melb_house_data.csv")
X = melbourne_data.drop("price", axis=1)
y = melbourne_data["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)

# 2. Samostatný rozhodovací strom (Benchmark)
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
mae_dt = mean_absolute_error(y_test, y_pred_dt)
# Výsledek: MAE = 376 988 AUD

# 3. Náhodný les s výchozími parametry
rf = RandomForestRegressor(random_state=42, n_estimators=100, oob_score=True)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
# Výsledek: MAE = 272 109 AUD
```

#### Analýza výsledku:
- **Pokles absolutní chyby:** z 376 988 AUD na 272 109 AUD, což představuje **zlepšení o 104 879 AUD** (pokles chyby o **27.8 %**).
- Model se bez jakéhokoliv ladění hyperparametrů mýlí v průměru o více než 100 000 AUD méně na každé nemovitosti!

---

## 3. Expertní kritické zhodnocení (Naše analýza & Úskalí)

Přestože výklad v prezentaci skvěle ilustruje základní principy, z pohledu profesionálního Data Science je nutné poukázat na 4 zásadní praktická úskalí a zjednodušení:

### 3.1 Mýtus: „Random Forest si sám poradí s chybějícími hodnotami v Pythonu“
- **Tvrzení ze slajdu 16:** *„Due to the mechanism of drawing subsets with replacement and using decision trees, the random forest is relatively immune to missing values. This means that it is not necessary to remove observations with empty values or fill them in.“*
- **Kritická realita v Scikit-learn:**  
  V původním algoritmu Lea Breimana (implementovaném v jazyce R nebo Fortranu) se pro chybějící hodnoty používala technika *Surrogate Splits* (náhradní dělení). **Ve standardním `RandomForestRegressor` a `RandomForestClassifier` v Scikit-learn však toto neplatí!**
  Pokud do `rf.fit(X_train, y_train)` předáte data obsahující `NaN`, Python okamžitě zhavaruje s chybou:
  `ValueError: Input contains NaN, infinity or a value too large for dtype('float32').`
- **Správné řešení:** Vždy je nutné zapojit explicitní imputaci v `Pipeline`:
  ```python
  from sklearn.pipeline import Pipeline
  from sklearn.impute import SimpleImputer
  
  rf_pipeline = Pipeline([
      ('imputer', SimpleImputer(strategy='median')),
      ('rf', RandomForestRegressor(random_state=42, n_jobs=-1))
  ])
  ```
  *(Poznámka: Teprve experimentální podpora v novějších verzích přes parametrizaci histogramů u `HistGradientBoostingRegressor` zvládá NaNs nativně).*

### 3.2 Zkreslení MDI (*Mean Decrease in Impurity*) při výpočtu důležitosti proměnných
- Atribut `rf.feature_importances_` měří celkový pokles Gini impurity nebo MSE přisuzovaný danému příznaku napříč všemi stromy.
- **Kardinální vada MDI:** MDI silně **nadhodnocuje numerické proměnné s vysokou kardinalitou** (mnoho unikátních hodnot, ID, časová razítka) a náhodný šum.
- **Expertní standard:** Pro seriózní interpretaci je nutné použít **Permutační důležitost** (`sklearn.inspection.permutation_importance`) na testovací sadě, která měří skutečný propad skóre po náhodném zamíchání sloupců.

### 3.3 Extrapolační bariéra v regresi
- Rozhodovací stromy a náhodné lesy **nedokáží extrapolovat mimo rozsah trénovacích dat**.
- Predikce v každém listu je konstantní průměr. Pokud nejdražší dům v trénovacích datech stál 5 000 000 AUD, náhodný les pro ultra-luxusní vilu za 25 000 000 AUD nikdy nepředpoví více než 5 000 000 AUD!
- Pro data se silným trendem (např. inflace, růst trhu, extrapolace v čase) je nutné kombinovat Random Forest s lineárním modelem (např. residuální modelování).

### 3.4 Riziko prostorového a časového úniku při OOB evaluaci
- OOB evaluace spoléhá na předpoklad, že pozorování jsou nezávislá a identicky rozdělená (*i.i.d.*).
- U realitních dat (Melbourne, King County) jsou sousední domy v jedné ulici prostorově autokorelované. Pokud se jeden dům dostane do tréninku a vedlejší dům do OOB, model těží z prostorového úniku informací (*Spatial Data Leakage*).
- OOB skóre je pak přehnaně optimistické. Řešením je prostorová nebo bloková křížová validace (*Spatial / Group K-Fold*).

---

## 4. Moderní ML & AI kontext (Stav k 10/2026)

Jak se s náhodnými lesy pracuje v moderním produkčním machine learningu k říjnu 2026?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Moderní ekosystém lesů k 10/2026                      │
└─────────────────────────────────────────────────────────────────────────────┘
          │                                                  │
   Hardwarová akcelerace                              Interpretovatelnost (XAI)
   - cuML Random Forest (NVIDIA GPUs, 50x zrychlení)  - TreeSHAP (shap.TreeExplainer)
   - Polars & DuckDB rychlé předzpracování            - Partial Dependence Plots (PDP)
          │                                                  │
   Moderní varianty algoritmů                         Automatizovaný MLOps
   - ExtraTrees (Extremely Randomized Trees)          - Optuna (Bayesovská optimalizace)
   - HistGradientBoosting (Nativní NaNs & Speed)      - MLflow experiment tracking
```

### 4.1 Moderní varianty: ExtraTrees a Histogramové stromy
1. **ExtraTrees (*Extremely Randomized Trees* – `ExtraTreesClassifier/Regressor`):**
   - Jde o další krok za Random Forest: prahy pro dělení uzlů se netestují optimálně, ale losují se **zcela náhodně** a vybere se nejlepší z nich.
   - Výsledek: Ještě nižší rozptyl, vyšší výpočetní rychlost a často překonává klasický Random Forest na zašuměných datech.
2. **HistGradientBoosting vs. Random Forest:**
   - Pro velké datasety (> 100k řádků) moderní praxe preferuje binování spojitých proměnných do 256 košů (*integer bins*), což eliminuje nutnost neustálého řazení hodnot.

### 4.2 GPU Akcelerace: cuML (NVIDIA RAPIDS)
- Trénování lesa s 500 stromy na milionech řádků může na vícejádrovém CPU trvat desítky minut.
- Knihovna `cuml.ensemble.RandomForestRegressor` umožňuje trénovat les přímo na tensor jádrech GPU s 10x až 50x zrychlením při zachování 100% kompatibility s API Scikit-learn.

### 4.3 Vysvětlitelné AI (XAI): TreeSHAP
- Místo zastaralého MDI se dnes standardně nasazuje algoritmus **TreeSHAP** (`shap.TreeExplainer`):
- Umožňuje přesně vyčíslit přínos každé jednotlivé proměnné k predikci konkrétního domu (lokální vysvětlení) i globální vliv na celý trh, a to v polynomiálním čase $O(T L D^2)$, kde $T$ je počet stromů, $L$ počet listů a $D$ maximální hloubka.

---

## 5. Shrnutí klíčových poznatků

1. **Random Forest = Bagging + Feature Subspacing:** Slučuje náhodné vzorkování řádků s vracením a náhodný výběr sloupců při každém dělení.
2. **Dekorelace je klíč k úspěchu:** Tím, že se stromy dívají na různé příznaky, nejsou jejich predikce korelované a průměrováním se efektivně eliminuje rozptyl.
3. **Out-of-Bag (OOB):** Přibližně 36.8 % vzorků zůstává mimo trénink každého stromu a poskytuje spolehlivý odhad generalizační schopnosti bez nutnosti dělení na validační sadu.
4. **Scikit-learn realita:** Parametry `n_estimators`, `max_features` a `min_samples_leaf` jsou hlavní páky pro ladění. Pro chybějící data je vždy nutné použít `Pipeline` s `SimpleImputer`.
