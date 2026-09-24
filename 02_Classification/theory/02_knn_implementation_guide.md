# k-NN v Scikit-learn: Praktická implementace, indexační stromy a vliv předzpracování (09/2026)

> **Zdrojový materiál kurzu:**  
> `K_nearest_neighbors_-_sample_implementation.pdf`  
> (Třída `KNeighborsClassifier`, hyperparametry, indexační algoritmy `kd_tree` a `ball_tree`, metoda `.kneighbors()`, demonstrace na datech Palmer Penguins, hodnocení metriky Accuracy a kritika školního řešení).

---

## 1. Třída `KNeighborsClassifier` v Scikit-learn

V knihovně Scikit-learn je algoritmus k nejbližších sousedů pro klasifikaci implementován ve třídě:
```python
from sklearn.neighbors import KNeighborsClassifier
```

Při vytváření instance modelu lze konfigurovat klíčové hyperparametry:

```python
knn = KNeighborsClassifier(
    n_neighbors=5,          # Počet uvažovaných sousedů (výchozí: 5)
    weights="uniform",      # 'uniform' (stejné váhy) nebo 'distance' (vážení 1/d)
    algorithm="auto",       # 'auto', 'ball_tree', 'kd_tree', 'brute'
    leaf_size=30,           # Velikost listu pro kd_tree nebo ball_tree
    p=2,                    # Mocnina pro Minkowského metriku (1 = Manhattan, 2 = Euclidean)
    metric="minkowski",     # Metrika vzdálenosti ('minkowski', 'euclidean', 'manhattan')
    n_jobs=-1               # Paralelizace výpočtu vzdáleností přes všechna jádra CPU
)
```

---

## 2. Podkapota: Indexační algoritmy (`algorithm`)

Při predikci musí k-NN najít $k$ nejbližších bodů. V Scikit-learn máme 4 možnosti, jak k tomu přistoupit:

```text
                                  ALGORITMY VYHLEDÁVÁNÍ
                                             |
           +---------------------------------+---------------------------------+
           |                                                                   |
      BRUTE FORCE                                                      STROMOVÉ INDEXY
  - Počítá vzdálenost ke všem vzorkům                             - Hierarchické dělení prostoru
  - Čas: O(N · d)                                                 - Čas dotazu: O(d · log N)
  - Vhodné pro malé N nebo vysoké d                               - Vhodné pro velké N a nízké d
                                                                               |
                                                     +-------------------------+-------------------------+
                                                     |                                                   |
                                                  KD-TREE                                            BALL-TREE
                                       - Dělí prostor rovinami kolmými k osám             - Dělí prostor vnořenými hypersférami
                                       - Rychlý pro d < 20                                - Zvládá i d > 20 a obecné metriky
```

### 1. `brute` (Hrubá síla)
- Pro každý testovací bod spočte vzdálenost ke **všem $N$ bodům v trénovací sadě**.
- Časová složitost: $\mathcal{O}(N \cdot d)$.
- Výhoda: Žádný čas na budování indexu, minimální paměť.
- Kdy použít: Pro malé datasety nebo když je počet dimenzí $d$ extrémně vysoký ($d > 50$), kde stromové indexy degenerují na lineární průchod.

### 2. `kd_tree` ($k$-rozměrný strom)
- Rekurzivně rozděluje prostor hyperrovinami kolmými na jednotlivé osy souřadnic (podobně jako rozhodovací strom, ale pro geometrické uspořádání bodů).
- Časová složitost dotazu: $\mathcal{O}(d \cdot \log N)$.
- Limit: Pro $d > 20$ trpí kletbou dimenzionality a jeho rychlost klesá k hrubé síle.

### 3. `ball_tree` (Strom hypersfér)
- Místo pravoúhlých rovin rozděluje data do vnořených $d$-rozměrných koulí (hypersfér).
- Efektivnější než `kd_tree` v prostorech s vyšší dimenzí a umožňuje použití obecných metrik vzdáleností (včetně Haversinovy vzdálenosti pro souřadnice na zemském povrchu).

### 4. `auto` (Výchozí volba)
- Scikit-learn se sám rozhodne:
  - Pokud $d$ je malé a $N$ dostatečné, použije `kd_tree`.
  - Pokud je metrika složitější nebo $d$ střední, použije `ball_tree`.
  - Pokud jsou data řídká (`scipy.sparse`), použije `brute`.

---

## 3. Inspekční metoda `.kneighbors()`

Velkou výhodou modelu `KNeighborsClassifier` je možnost nahlédnout do vnitřního rozhodovacího procesu. Pomocí metody `.kneighbors()` můžeme pro jakýkoliv vstup získat:
1. Matice vzdáleností k nejbližším $k$ sousedům.
2. Indexy těchto sousedů v trénovací sadě.

```python
# Získání vzdáleností a indexů 5 nejbližších sousedů pro testovací vzorek
distances, indices = knn.kneighbors(X_test.iloc[0:1], n_neighbors=5)

print("Indexy nejbližších bodů:", indices)
print("Vzdálenosti k sousedům:", distances)
```

Tato metoda je neocenitelná pro **vysvětlitelné AI (XAI)**, protože zákazníkovi či lékaři můžeme přesně ukázat: *„Váš případ jsme klasifikovali jako rizikový, protože se nejvíce podobá pacientům č. 42, 115 a 201 z naší databáze.“*

---

## 4. Školní demonstrace: Palmer Penguins Dataset

V materiálech kurzu je k-NN demonstrován na populárním datovém souboru **Palmer Penguins** (`penguins_size.csv`), který obsahuje biometrické údaje tří druhů antarktických tučňáků:
- **Druhy (Species):** *Adelie* (152), *Gentoo* (124), *Chinstrap* (68).
- **Příznaky:**
  - `culmen_length_mm`: Délka horního hřebene zobáku (v mm).
  - `culmen_depth_mm`: Výška zobáku (v mm).
  - `flipper_length_mm`: Délka ploutve (v mm).
  - `body_mass_g`: Hmotnost těla (v gramech).
  - `island`: Ostrov odchytu (*Torgersen*, *Biscoe*, *Dream*).
  - `sex`: Pohlaví (*MALE*, *FEMALE*).

### Školní kód z prezentace:
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 1. Načtení dat
penguins_df = pd.read_csv("penguins_size.csv")

# 2. Rychlé čištění: vyhození kategoriálních sloupců a chybějících hodnot
penguins_df = penguins_df.drop(["island", "sex"], axis=1)
penguins_df.dropna(inplace=True)

# 3. Train/Test split (70 % train, 30 % test)
X = penguins_df.drop("species", axis=1)
y = penguins_df["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Trénování k-NN s k=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# 5. Vyhodnocení přesnosti
accuracy = knn.score(X_test, y_test)
print(f"k-NN model accuracy: {accuracy:.4f}")  # Vychází cca 0.8020 (80.2 %)
```

---

## 5. Hluboká kritická analýza: Proč školní model dosáhl pouhých 80 %?

Výsledek 80.2 % vypadá na první pohled uspokojivě, ale v kontextu moderního Data Science jde o **zásadní metodické selhání**. Datový soubor Palmer Penguins je totiž téměř dokonale separabilní a správně postavený model dosahuje přesnosti **98–100 %**.

Proč byl školní model tak slabý? Prezentace kurzu správně zmiňuje tři hlavní hříchy:

### Hřích č. 1: Zanedbání škálování proměnných (Dominance hmotnosti)
Podívejme se na rozptyly a měřítka jednotlivých proměnných:

| Proměnná | Typická hodnota | Směrodatná odchylka ($\sigma$) | Rozptyl ($\sigma^2$) |
| :--- | :--- | :--- | :--- |
| `culmen_depth_mm` | 13 – 21 mm | ~ 1.97 mm | **~ 3.9** |
| `culmen_length_mm`| 32 – 60 mm | ~ 5.46 mm | **~ 29.8** |
| `flipper_length_mm`| 170 – 230 mm | ~ 14.0 mm | **~ 197.0** |
| `body_mass_g` | 2 700 – 6 300 g | ~ 802.0 g | **~ 643 200.0** |

Při výpočtu Eukleidovské vzdálenosti dvou tučňáků:
$$d = \sqrt{(L_1 - L_2)^2 + (D_1 - D_2)^2 + (F_1 - F_2)^2 + (M_1 - M_2)^2}$$

Pokud se dva tučňáci liší v délce zobáku o 5 mm, přispějí do sumy pod odmocninou číslem $5^2 = 25$.  
Pokud se ale liší v hmotnosti o 300 gramů (zcela běžný rozdíl), přispějí číslem $300^2 = 90\,000$!

> **Důsledek:** Hmotnost `body_mass_g` tvoří přes **99.9 % celkové vzdálenosti**. k-NN v neškálovaném modelu prakticky **zcela ignoroval délku a tvar zobáku** a klasifikoval tučňáky téměř výhradně podle toho, jak jsou tlustí! Jelikož samice druhu Gentoo mají podobnou hmotnost jako velcí samci druhu Adelie, docházelo k masivním záměnám.

### Hřích č. 2: Vyhození kategoriálních proměnných (`island`)
V biologické realitě je ostrov klíčovým faktorem:
- Druh *Gentoo* hnízdí téměř výhradně na ostrově **Biscoe**.
- Druh *Chinstrap* hnízdí na ostrově **Dream**.
- Zahodit proměnnou `island` znamená připravit model o přirozený geografický filtr.

### Hřích č. 3: Náhodné číslo $k = 5$ bez ladění
Hodnota $k=5$ je pouze konvence. Pro $N \approx 233$ trénovacích vzorků je $\sqrt{N} \approx 15$. Skutečně optimální hodnota $k$ může být jiná v závislosti na hustotě jednotlivých tříd.

---

## 6. Profesionální Scikit-learn Pipeline (Moderní standard 2026)

Jak se má úloha řešit správně a robustně podle moderních standardů?

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# 1. Definice předzpracování odděleně pro číselné a textové sloupce
num_features = ["culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g"]
cat_features = ["island", "sex"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_features)
    ]
)

# 2. Kompletní Pipeline (Preprocesor + Model)
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("knn", KNeighborsClassifier())
])

# 3. Mřížkové hledání optimálních hyperparametrů s křížovou validací
param_grid = {
    "knn__n_neighbors": list(range(1, 25, 2)),  # Liché hodnoty 1..23
    "knn__weights": ["uniform", "distance"],
    "knn__p": [1, 2]  # 1 = Manhattan, 2 = Euclidean
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)

print(f"Nejlepší hyperparametry: {grid.best_params_}")
print(f"Validační přesnost: {grid.best_score_:.4f}")
print(f"Testovací přesnost: {grid.score(X_test, y_test):.4f}")
```

### Výsledek:
Testovací přesnost vyskočí z **80.2 % na 98–100 %**.  
Tím je názorně dokázáno, že **kvalita modelu strojového učení nezávisí jen na volbě algoritmu, ale především na precizní přípravě a škálování dat**.
