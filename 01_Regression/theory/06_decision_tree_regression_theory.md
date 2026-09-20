# Rozhodovací strom v regresi (Decision Tree in Regression): Teoretický rozbor (09/2026)

> **Zdrojový materiál kurzu:**  
> `Decision_tree_in_regression_-_theoretical_introduction.pdf`  
> (Struktura stromu, rekurzivní binární dělení, kritéria MSE/MAE, hyperparametry, prořezávání, výhody a limity)

---

## 1. Úvod do rozhodovacích stromů v regresi

Rozhodovací strom (**Decision Tree**) představuje zásadní koncepční posun oproti lineární a polynomiální regresi. Zatímco lineární modely jsou **parametrické** (předpokládají globální tvar funkce $y = \mathbf{X}\boldsymbol{\beta}$), rozhodovací strom je **neparametrický model**:
- Nepředpokládá žádný předem daný matematický tvar závislosti (přímku, parabolu, hyperbolu).
- Rozděluje příznakový prostor na soustavu vzájemně se nepřekrývajících vícerozměrných obdélníkových oblastí (hyper-boxes).
- V každé oblasti aproximuje cílovou veličinu **lokální konstantou** (obvykle průměrem hodnot v dané oblasti).

Výsledná predikční funkce je **po částech konstantní (piecewise constant)** schodovitá funkce.

---

## 2. Anatomie rozhodovacího stromu

Model je hierarchická stromová struktura orientovaného grafu orientovaná shora dolů:

```text
                 [ Kořenový uzel (Root Node) ]
                   Je plocha <= 100 m²?
                       /          \
                   ANO /            \ NE
                     v                v
     [ List (Leaf Node) ]     [ Vnitřní uzel (Decision Node) ]
      Predikce: 400 000 USD      Má více než 3 ložnice?
                                   /          \
                               ANO /            \ NE
                                 v                v
                       [ List (Leaf Node) ]   [ List (Leaf Node) ]
                        Predikce: 500 000 USD  Predikce: 450 000 USD
```

### Klíčové komponenty stromu:
1. **Kořenový uzel (Root Node):** Počáteční uzel stromu, který obsahuje 100 % trénovacích dat. Zde probíhá první a statisticky nejvýznamnější rozdělení.
2. **Vnitřní rozhodovací uzly (Internal / Decision Nodes):** Uzly, v nichž se testuje podmínka na vybrané proměnné (např. $x_j \le s$). Podle výsledku testu putuje pozorování do levé nebo pravé větve.
3. **Větve (Branches / Edges):** Spojnice reprezentující výsledek rozhodnutí (pravda / nepravda).
4. **Koncové listy (Leaf / Terminal Nodes):** Uzly bez potomků. Neobsahují žádné další otázky, ale nesou **konečnou predikci** modelu pro všechna pozorování, která do daného listu propadnou.

---

## 3. Jak se rozhodovací strom trénuje? (Algoritmus CART)

V Scikit-learn je implementován algoritmus **CART (Classification and Regression Trees)**. Trénování probíhá pomocí hladového přístupu zvaného **rekurzivní binární dělení (Recursive Binary Splitting)**.

### Princip rekurzivního binárního dělení:
Algoritmus je „hladový“ (greedy), protože v každém kroku volí to nejlepší okamžité rozdělení bez ohledu na to, jaká rozdělení budou následovat dále.

1. Prochází postupně **všechny prediktory** $j \in \{1, 2, \dots, p\}$.
2. Pro každý prediktor $j$ testuje všechny možné **prahové hodnoty** (split points) $s$, které rozdělí trénovací množinu $D$ na dvě podmnožiny:
   $$R_1(j, s) = \{X \mid X_j \le s\} \quad \text{a} \quad R_2(j, s) = \{X \mid X_j > s\}$$
3. Hledá takovou dvojici $(j^*, s^*)$, která minimalizuje celkovou chybu v obou vzniklých potomcích.

### Kritérium dělení pro regresi:

#### A. Kvadratická chyba (Mean Squared Error – výchozí kritérium):
Pro každou oblast $R_m$ je predikcí průměr cílové proměnné trénovacích pozorování v této oblasti:
$$\hat{y}_{R_m} = \frac{1}{N_m} \sum_{i \in R_m} y_i$$

Cílem je minimalizovat součet čtverců reziduí (RSS):
$$\min_{j, s} \left[ \sum_{i \in R_1(j, s)} (y_i - \hat{y}_{R_1})^2 + \sum_{i \in R_2(j, s)} (y_i - \hat{y}_{R_2})^2 \right]$$

To je ekvivalentní minimalizaci váženého rozptylu cílové proměnné v obou dceřiných uzlech.

#### B. Absolutní chyba (Mean Absolute Error):
$$\hat{y}_{R_m} = \text{median}(\{y_i \mid i \in R_m\})$$
Minimalizuje součet absolutních odchylek. Je robustnější vůči odlehlým hodnotám (outlierům) v cílové proměnné $y$.

---

## 4. Práce s různými typy proměnných

1. **Spojité numerické proměnné (např. plocha, karát, nájezd):**
   - Strom seřadí unikátní hodnoty proměnné a jako kandidáty na prahové hodnoty $s$ testuje středy mezi sousedními hodnotami.
2. **Kategoriální proměnné:**
   - Dělí kategorie na dvě disjunktní podmnožiny (např. $\text{Color} \in \{D, E, F\}$ vs. $\text{Color} \in \{G, H, I, J\}$).
3. **Invariance vůči monotónním transformacím (Obrovská výhoda):**
   - Rozhodovací strom zajímá pouze **pořadí hodnot**, nikoliv jejich absolutní velikost.
   - Proto **není potřeba žádné škálování** (`StandardScaler`, `MinMaxScaler`) ani logaritmická transformace! Výsledné rozdělení i predikce jsou při jakékoliv striktně rostoucí transformaci vstupních veličin naprosto identické.

---

## 5. Příklad z kurzu: Predikce ceny nemovitosti

Mějme dům s parametry: plocha v $\text{m}^2$ a počet ložnic.

1. **První split (Kořen):**  
   Model vyhodnotil plochu jako nejdůležitější prediktor pro minimalizaci rozptylu cen. Prahová hodnota byla stanovena na $100\,\text{m}^2$.
   - Pokud $\text{plocha} \le 100\,\text{m}^2 \implies$ list s predikcí **400 000 USD**.
2. **Druhý split (pro domy $> 100\,\text{m}^2$):**  
   Model testuje počet ložnic s prahem $3$.
   - Pokud $\text{ložnice} \le 3 \implies$ list s predikcí **450 000 USD**.
   - Pokud $\text{ložnice} > 3 \implies$ list s průměrem 5 domů z trénovacích dat (500k, 600k, 400k, 620k, 380k USD):
     $$\hat{y} = \frac{500\,000 + 600\,000 + 400\,000 + 620\,000 + 380\,000}{5} = \mathbf{500\,000\text{ USD}}$$

---

## 6. Klíčové hyperparametry v Scikit-learn (`DecisionTreeRegressor`)

Pokud necháme strom růst bez omezení, bude se dělit tak dlouho, dokud v každém listu nezůstane jediný vzorek. V takovém případě bude trénovací $R^2 = 1.0$ a $\text{MSE}_{\text{train}} = 0$, ale model bude fatálně přeučený (**overfitted**) a na testovacích datech selže.

Proto je kritické regulovat složitost stromu pomocí hyperparametrů:

| Hyperparametr | Výchozí hodnota | Popis a vliv na model |
| :--- | :---: | :--- |
| `max_depth` | `None` | Maximální hloubka stromu. Nejdůležitější parametr pro kontrolu přeučení. Malá hloubka = underfitting, velká = overfitting. |
| `min_samples_split` | `2` | Minimální počet pozorování v uzlu nutný k tomu, aby mohl být dále rozdělen. Vyšší hodnota tlumí větvení v malých šumových vzorcích. |
| `min_samples_leaf` | `1` | Minimální počet pozorování, která musí skončit v každém koncovém listu. Zabraňuje vzniku listů reprezentujících izolované outliery. |
| `max_features` | `None` | Počet náhodně vybíraných příznaků testovaných při každém dělení. |
| `criterion` | `"squared_error"` | Funkce pro měření kvality rozdělení (`"squared_error"`, `"absolute_error"`, `"poisson"`). |
| `ccp_alpha` | `0.0` | Parametr prořezávání podle složitosti (Cost-Complexity Pruning). |

---

## 7. Výhody a nevýhody rozhodovacího stromu

### 🟢 Výhody:
1. **Intuitivní interpretovatelnost:** Rozhodovací logiku lze snadno vizualizovat jako vývojový diagram, kterému rozumí i lidé bez technického či matematického vzdělání.
2. **Není nutná předúprava dat:** Nevyžaduje normalizaci, standardizaci ani centrování prediktorů.
3. **Přirozené modelování nelinearit a interakcí:** Zachycuje složité vztahy a hierarchické podmínky bez nutnosti generovat polynomiální či interakční členy.
4. **Odolnost vůči multikolinearitě:** Pokud dvě proměnné silně korelují, strom si jednoduše vybere jednu z nich a druhou ignoruje.
5. **Robustnost vůči outlierům na vstupech:** Extrémní hodnota na vstupu pouze ovlivní pořadí, ale neposune prahovou hodnotu o kilometry (na rozdíl od OLS koeficientů).

### 🔴 Nevýhody a úskalí:
1. **Schodovitá / po částech konstantní predikce:** Strom neumí predikovat hladké spojité křivky. Výstupem je vždy diskrétní sada konstantních hladin.
2. **Neschopnost extrapolace trendů:** Rozhodovací strom **nikdy nemůže predikovat hodnotu vyšší než maximum (ani nižší než minimum) z trénovací sady**. Pokud např. ceny nemovitostí v čase rostou, strom mimo historický rozsah selže a predikuje konstantní strop.
3. **Vysoká variance a nestabilita:** Malá změna v trénovacích datech (např. přidání několika řádků) může způsobit zcela jiné rozdělení v kořenovém uzlu a převrátit celou strukturu stromu.
4. **Vysoké riziko přeučení (Overfitting):** Bez pečlivého ladění `max_depth` a `min_samples_leaf` si strom zapamatuje šum v trénovacích datech.

---

## 8. Moderní pohled: Přemostění k ansámblovým metodám

Samostatný rozhodovací strom (Single Decision Tree) má v praxi kvůli vysoké varianci a schodovitosti omezenou predikční přesnost. Je však **základním stavebním kamenem (weak learnerem)** pro nejvýkonnější algoritmy současného strojového učení:

1. **Random Forest (Bagging):**  
   Trénuje stovky hlubokých, záměrně přeučených stromů na náhodných podvýběrech dat i příznaků a jejich predikce průměruje. Tím dramaticky snižuje varianci bez nárůstu zkreslení (biasu).
2. **Gradient Boosted Decision Trees – GBDT (Boosting: XGBoost, LightGBM, CatBoost):**  
   Trénuje mělké stromy sekvenčně za sebou, kde každý další strom opravuje reziduální chyby předchozích stromů. V současnosti jde o absolutní špičku pro tabulková data v průmyslu.
