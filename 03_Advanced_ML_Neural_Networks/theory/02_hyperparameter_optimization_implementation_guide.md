# Praktická implementace optimalizace hyperparametrů v Pythonu

Tento průvodce detailně rozebírá implementaci tří stěžejních technik ladění hyperparametrů v Pythonu:
1. **Grid Search (`GridSearchCV` z knihovny Scikit-learn)**
2. **Randomized Search (`RandomizedSearchCV` z knihovny Scikit-learn)**
3. **Bayesovská optimalizace (`fmin`, `Trials` z knihovny Hyperopt)**

---

## 1. Grid Search (`GridSearchCV`)

Třída `GridSearchCV` provádí vyčerpávající prohledávání všech zadaných kombinací hyperparametrů s využitím $K$-násobné křížové validace.

### Klíčové argumenty konstruktoru:
- `estimator`: Instance modelu ze Scikit-learn (např. `DecisionTreeClassifier()`).
- `param_grid`: Slovník, kde klíčem je název hyperparametru a hodnotou je seznam/pole testovaných hodnot.
- `scoring`: Cílová evaluační metrika (např. `'recall'`, `'precision'`, `'f1'`, `'accuracy'`, `'roc_auc'`).
- `cv`: Schéma křížové validace (např. celé číslo `cv=5` pro 5-fold, nebo objekt `StratifiedKFold`).
- `n_jobs`: Počet paralelních procesů (`-1` využije všechna dostupná jádra procesoru).
- `return_train_score`: Pokud `True`, zaznamenává i skóre na trénovacích foldech (vhodné pro detekci přeučení).

### Klíčové metody a atributy:
- `.fit(X, y)`: Spustí prohledávání a vyhodnocení všech kombinací na trénovacích datech.
- `.best_params_`: Slovník s nejlepší nalezenou kombinací hyperparametrů.
- `.best_score_`: Nejvyšší průměrné validační skóre dosažené v křížové validaci.
- `.best_estimator_`: Plně natrénovaná instance modelu s nejlepšími parametry.
- `.cv_results_`: Detailní slovník/DataFrame obsahující výsledky všech foldů pro každou kombinaci.
- `.predict(X)`: Predikuje výstup přímo pomocí nejlepšího modelu (`best_estimator_`).

### Ukázka kódu:
```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# Definice mřížky parametrů
params_grid = {
    'max_depth': np.arange(3, 21, 2), # celá čísla 3, 5, ..., 19
    'criterion': ["entropy", "gini"],
    'max_features': [None, "log2", "sqrt"],
}

# Vytvoření modelu a GridSearchCV
model = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(model, params_grid, cv=5, scoring="recall", n_jobs=-1)

# Spuštění optimalizace na trénovacích datech
grid_search.fit(X_train, y_train)

# Získání nejlepších parametrů a modelu
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Evaluace na nezávislé testovací sadě
test_recall = best_model.score(X_test, y_test)
```

---

## 2. Randomized Search (`RandomizedSearchCV`)

Třída `RandomizedSearchCV` náhodně vzorkuje kombinace z definovaných rozsahů či pravděpodobnostních rozdělení. Díky tomu je výpočetně řádově úspornější než `GridSearchCV`.

### Klíčové argumenty:
- `estimator`: Instance modelu.
- `param_distributions`: Slovník, kde hodnotami mohou být jak diskrétní seznamy, tak **spojité distribuce** ze `scipy.stats`.
- `n_iter`: Pevný počet náhodných pokusů (např. `n_iter=20`).
- `scoring`, `cv`, `n_jobs`: Shodné jako u `GridSearchCV`.

### Spojité distribuce v `scipy.stats`:
- `uniform(loc=0, scale=4)`: Rovnoměrné rozdělení v intervalu $[0, 4]$.
- `truncnorm(a=0, b=1, loc=0.25, scale=0.1)`: Oříznuté normální rozdělení v mezích $[a, b]$.
- `loguniform(1e-4, 1e2)`: Log-uniformní rozdělení pro parametry typu $C$ nebo $\gamma$.

### Ukázka kódu:
```python
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

# Definice distribucí pro spojité hyperparametry
param_dist = {
    'C': uniform(loc=0.1, scale=9.9), # vzorkování z intervalu [0.1, 10.0]
    'gamma': uniform(loc=0.01, scale=0.99), # vzorkování z intervalu [0.01, 1.0]
    'kernel': ['rbf', 'linear']
}

model = SVC(random_state=42)
random_search = RandomizedSearchCV(
    model, 
    param_distributions=param_dist, 
    n_iter=15, 
    cv=5, 
    scoring="precision", 
    random_state=42, 
    n_jobs=-1
)
random_search.fit(X_train, y_train)

best_params = random_search.best_params_
best_model = random_search.best_estimator_
```

---

## 3. Bayesovská optimalizace (`hyperopt`)

Knihovna `hyperopt` staví pravděpodobnostní model závislosti ztrátové funkce na hyperparametrech (algoritmus TPE – *Tree-structured Parzen Estimator*).

### Hlavní stavební kameny:
1. `space`: Definice prostoru parametrů:
   - `hp.choice(label, options)`: Diskrétní výběr ze seznamu možností.
   - `hp.uniform(label, low, high)`: Spojité rovnoměrné rozdělení.
   - `hp.loguniform(label, low, high)`: Logaritmické rozdělení (pro parametry škály).
   - `hp.randint(label, upper)`: Celá čísla v rozsahu $[0, \text{upper}-1]$.
2. `objective(params)`: Cílová funkce, která:
   - Převezme parametry.
   - Natrénuje model.
   - Spočítá metriku na validačních datech (nebo pomocí křížové validace).
   - **Vrátí hodnotu s opačným znaménkem (`-score`)**, protože `fmin` ze své podstaty funkci **minimalizuje**!
3. `Trials()`: Objekt zaznamenávající historii každého pokusu (hodnoty parametrů, čas, ztrátu).
4. `fmin()`: Hlavní optimalizační smyčka:
   - `fn`: Cílová funkce.
   - `space`: Prostor parametrů.
   - `algo=tpe.suggest`: Bayesovský algoritmus TPE.
   - `max_evals`: Počet iterací.
   - `trials`: Objekt `Trials`.
5. `space_eval(space, best)`: Dekóduje nejlepší nalezené parametry zpět do lidsky čitelné podoby (převádí indexy z `hp.choice` na skutečné hodnoty).

### Ukázka kódu:
```python
from hyperopt import hp, fmin, tpe, Trials, space_eval
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# Definice vyhledávacího prostoru
space = {
    'max_depth': hp.choice('max_depth', [3, 5, 7, 10, None]),
    'criterion': hp.choice('criterion', ['gini', 'entropy']),
    'min_samples_split': hp.uniform('min_samples_split', 0.01, 0.5),
    'min_samples_leaf': hp.choice('min_samples_leaf', [1, 2, 4, 8, 16]),
}

def objective(params):
    # Přetypování případných celočíselných hodnot
    model = DecisionTreeClassifier(
        max_depth=params['max_depth'],
        criterion=params['criterion'],
        min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'],
        random_state=42
    )
    # 5-fold křížová validace
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="recall")
    # Minimalizujeme zápornou hodnotu metriky
    return -scores.mean()

trials = Trials()
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=30,
    trials=trials,
    rstate=np.random.default_rng(42)
)

best_params = space_eval(space, best)
```

---

## 4. Srovnání metod z hlediska praxe

| Vlastnost | `GridSearchCV` | `RandomizedSearchCV` | `Hyperopt` (Bayesian) |
| :--- | :--- | :--- | :--- |
| **Knihovna** | Scikit-learn | Scikit-learn | `hyperopt` |
| **Integrace do Pipeline** | Nativní a triviální | Nativní a triviální | Vyžaduje obalovací funkci |
| **Počet laděných parametrů** | 1 až 2 | 3 až 10 | 5 až 50+ |
| **Doporučený rozpočet iterací** | Všechny body ($V^P$) | 20 až 60 iterací | 30 až 150 iterací |
| **Učení z předchozích kroků** | Ne | Ne | **Ano (TPE)** |
| **Vhodnost pro začátek** | Vynikající pro hrubý přehled | Nejlepší poměr rychlost/výkon | Ideální pro finální ladění soutěžních modelů |
