# Rozhodovací stromy v klasifikaci: Teoretický rozbor, kritéria větvení a implementace (09/2026)

> **Zdrojový výukový materiál kurzu:**  
> `Decision_tree_in_classification_-_introduction_and_implementation.pdf`  
> Prezentace a metodické pokyny vzdělávacího programu Data Science & Machine Learning (Coders Lab).

---

## 1. Od logistické regrese k rozhodovacím stromům

V předchozích kapitolách jsme se zabývali logistickou regresí. Přestože je logistická regrese elegantním a dobře interpretovatelným modelem, má svá vrozená omezení:
1. **Předpoklad linearity v logitu:** Rozhodovací hranice mezi třídami je lineární nadrovina ($\mathbf{x}^T \boldsymbol{\beta} = 0$). Pokud mají třídy složitý, nelineární nebo prstencovitý tvar, logistická regrese bez manuálního feature engineeringu selhává.
2. **Přirozený multiclass:** Logistická regrese je původně binární model; pro více než dvě třídy vyžaduje buď dekompozici One-vs-Rest, nebo multinomiální generalizaci (Softmax).
3. **Citlivost na škálování:** Při zapnuté regularizaci vyžaduje striktní standardizaci příznaků (jak jsme viděli u tučňáků).

**Klasifikační rozhodovací strom (Decision Tree Classifier)** naproti tomu nabízí zcela odlišné paradigma:
- Rozděluje příznakový prostor pomocí pravoúhlých (osově orientovaných) řezů na disjunktní oblasti.
- **Nativně podporuje libovolný počet tříd (multiclass)** bez nutnosti vytvářet dílčí binární modely.
- **Je 100% invariantní vůči monotónním transformacím a škálování příznaků** (je mu lhostejné, zda je veličina v milimetrech, gramech či logaritmech).
- Výsledný model je přirozeně interpretovatelný jako hierarchická posloupnost pravidel typu *IF-THEN*.

---

## 2. Anatomie klasifikačního rozhodovacího stromu

Struktura stromu je hierarchický orientovaný graf sestávající ze tří typů prvků:
1. **Kořenový uzel (Root Node):**  
   Výchozí uzel obsahující celou trénovací sadu dat. Zde probíhá první optimální rozdělení na základě nejlepšího příznaku a prahu.
2. **Vnitřní / Rozhodovací uzly (Internal / Decision Nodes):**  
   Uzly, které testují konkrétní podmínku (např. $x_j \le 42{,}5$) a směrují pozorování do levého nebo pravého podstromu.
3. **Koncové / Listové uzly (Leaf Nodes):**  
   Uzly bez potomků. Reprezentují finální predikci třídy (obvykle modus / většinová třída vzorků spadajících do daného listu) a aposteriorní pravděpodobnostní rozdělení tříd.

```text
                  [ Kořenový uzel (Root) ]
                      x[0] <= 1.45 ?
                       /          \
                     ANO          NE
                     /              \
           [ Vnitřní uzel ]     ( List: Gentoo )
             x[2] <= 0.8 ?          N = 34
              /        \
            ANO        NE
            /            \
     ( List: Adelie )  ( List: Chinstrap )
         N = 40            N = 27
```

---

## 3. Jak dělíme data v klasifikaci? Kritéria větvení

V regresních úlohách bylo cílem dělení minimalizovat rozptyl a chybu MSE či MAE. V klasifikaci je cílová proměnná diskrétní ($y \in \{1, \dots, C\}$).

Cílem algoritmu je provést takové rozdělení, aby vzniklé dceřiné uzly byly **co nejčistší (nejvíce homogenní)**. Uzel je **dokonale čistý (Pure Node)**, pokud všechna pozorování v něm patří do jediné třídy.

Pro kvantifikaci nečistoty (nebo míry chaosu) se používají dvě hlavní kritéria:
1. **Shannonova Entropie & Informační zisk (Information Gain – IG)**
2. **Giniho index nečistoty (Gini Impurity)**

---

## 4. Shannonova Entropie a Informační zisk (Information Gain)

Pojem entropie pochází z termodynamiky a teorie informace (Claude Shannon, 1948). V kontextu rozhodovacích stromů vyjadřuje **míru nejistoty, neuspořádanosti či chaosu** v rozdělení tříd v daném uzlu.

### Matematická formulace entropie:
$$H(S) = -\sum_{i=1}^C p_i \log_2(p_i)$$

kde:
- $C$: počet unikátních tříd v datasetu,
- $p_i$: empirická pravděpodobnost (relativní četnost) výskytu třídy $i$ v daném uzlu ($p_i = \frac{n_i}{N}$).
- Konvence: Pokud $p_i = 0$, pak $0 \cdot \log_2(0) \equiv 0$.

### Vlastnosti entropie:
- **Minimální entropie ($H = 0$):** Uzel je dokonale čistý (všechny vzorky patří do jedné třídy). Další dělení nemá smysl.
- **Maximální entropie pro binární případ ($H = 1{,}0$):** Nastává při dokonale vyrovnaném stavu ($p_1 = 0{,}5, p_2 = 0{,}5$). Míra nejistoty je maximální.
- **Jednotky:** Bity (shannony) díky základu logaritmu 2.

---

### Numerický příklad ze slajdů kurzu: Shannonova entropie (Počasí)
Mějme vzorek 6 pozorování popisujících dny: 3 slunečné (*Sunny*) a 3 deštivé (*Rainy*).

1. **Entropie v kořenovém uzlu (Parent Node):**
   $$p(\text{Sunny}) = \frac{3}{6} = 0{,}5, \quad p(\text{Rainy}) = \frac{3}{6} = 0{,}5$$
   $$H(\text{parent}) = -\left( 0{,}5 \log_2(0{,}5) + 0{,}5 \log_2(0{,}5) \right) = -(0{,}5 \cdot (-1) + 0{,}5 \cdot (-1)) = -(-1) = 1{,}0$$

2. **Dělení podle proměnné Teplota (*Temperature*):**
   Proměnná má dvě hodnoty: *Low* (Nízká) a *High* (Vysoká).
   - Pro *Temperature = Low* máme 3 pozorování: 1 *Sunny* a 2 *Rainy*.
     $$H(\text{Low}) = -\left( \frac{1}{3} \log_2\left(\frac{1}{3}\right) + \frac{2}{3} \log_2\left(\frac{2}{3}\right) \right) = -\left( \frac{1}{3} \cdot (-1{,}585) + \frac{2}{3} \cdot (-0{,}585) \right) = -(-0{,}528 - 0{,}390) = 0{,}918$$
   - Pro *Temperature = High* máme 3 pozorování: 3 *Sunny* a 0 *Rainy*.
     $$H(\text{High}) = -\left( \frac{3}{3} \log_2(1) + 0 \right) = 0{,}0 \quad \text{(Dokonale čistý uzel!)}$$

3. **Vážená entropie dětí (Weighted Child Entropy):**
   $$H(\text{children}) = \frac{3}{6} \cdot H(\text{Low}) + \frac{3}{6} \cdot H(\text{High}) = 0{,}5 \cdot 0{,}918 + 0{,}5 \cdot 0{,}0 = 0{,}459$$

4. **Informační zisk (Information Gain – IG):**
   $$\text{IG} = H(\text{parent}) - H(\text{children}) = 1{,}0 - 0{,}459 = 0{,}541$$

Algoritmus spočítá $\text{IG}$ pro všechny dostupné proměnné (např. vlhkost, srážky) a pro první split vybere proměnnou s **nejvyšším informačním ziskem**.

---

## 5. Giniho index nečistoty (Gini Impurity)

Giniho index (Corrado Gini, 1912) je alternativní kritérium, které měří pravděpodobnost, že náhodně vybraný prvek z daného uzlu by byl nesprávně klasifikován, kdyby byl náhodně označen podle rozdělení tříd v tomto uzlu.

### Matematická formulace Giniho indexu:
$$\text{Gini}(S) = 1 - \sum_{i=1}^C p_i^2$$

### Vlastnosti Giniho indexu:
- **Minimální Gini ($\text{Gini} = 0$):** Uzel je dokonale čistý ($p_1 = 1 \implies 1 - 1^2 = 0$).
- **Maximální Gini pro binární případ ($\text{Gini} = 0{,}5$):** Nastává při rovnoměrném rozdělení ($p_1 = 0{,}5, p_2 = 0{,}5 \implies 1 - (0{,}25 + 0{,}25) = 0{,}5$).
- **Obecné maximum pro $C$ tříd:** $\text{Gini}_{\max} = 1 - \frac{1}{C}$.

---

### Numerický příklad ze slajdů kurzu: Gini index
1. **Kořenový uzel (3 Sunny, 3 Rainy ze 6 dnů):**
   $$\text{Gini}(\text{parent}) = 1 - \left( \left(\frac{3}{6}\right)^2 + \left(\frac{3}{6}\right)^2 \right) = 1 - (0{,}25 + 0{,}25) = 0{,}5$$

2. **Dělení podle teploty (varianta ze slajdu 14: 4 dny Low [3 Rainy, 1 Sunny] a 2 dny High [2 Sunny, 0 Rainy]):**
   - Pro *Temperature = Low*:
     $$\text{Gini}(\text{Low}) = 1 - \left( \left(\frac{3}{4}\right)^2 + \left(\frac{1}{4}\right)^2 \right) = 1 - (0{,}5625 + 0{,}0625) = 1 - 0{,}625 = 0{,}375$$
   - Pro *Temperature = High*:
     $$\text{Gini}(\text{High}) = 1 - \left( \left(\frac{2}{2}\right)^2 + 0 \right) = 1 - 1 = 0{,}0$$

3. **Vážený Gini index po rozdělení:**
   $$\text{Gini}_{\text{split}} = \frac{4}{6} \cdot 0{,}375 + \frac{2}{6} \cdot 0{,}0 = \frac{1{,}5}{6} = 0{,}25$$
   Index nečistoty klesl z $0{,}50$ na $0{,}25$ (pokles o polovinu).

---

## 6. Srovnání kritérií: Gini index versus Entropie

```text
       Hodnota nečistoty
   1.0 +         /\   (Entropie H(p), maximum = 1.0)
       |        /  \
       |       /    \
   0.5 +      /  /\  \ (Gini Impurity, maximum = 0.5)
       |     /  /  \  \
       |    /  /    \  \
   0.0 +---+--+------+--+---> p (Pravděpodobnost třídy 1)
          0.0       0.5      1.0
```

| Vlastnost | Gini Impurity (`criterion="gini"`) | Shannonova Entropie (`criterion="entropy"`) |
| :--- | :--- | :--- |
| **Výpočetní náročnost** | **Velmi nízká** (pouze násobení a sčítání $p_i^2$) | **Vyšší** (vyžaduje výpočet $\log_2(p_i)$) |
| **Výchozí volba** | Ano (výchozí v Scikit-learn) | Ne |
| **Chování při větvení** | Preferuje izolaci nejpočetnější třídy | Vede k mírně vyváženějším stromům |
| **Výsledná přesnost** | V 98 % úloh statisticky totožná | V 98 % úloh statisticky totožná |

---

## 7. Implementace v Pythonu a Scikit-learn

V knihovně Scikit-learn je algoritmus implementován ve třídě `DecisionTreeClassifier` v modulu `sklearn.tree`.

### Školní ukázka ze slajdů kurzu:

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score
import matplotlib.pyplot as plt

# 1. Syntetický dataset (600 vzorků, 5 příznaků, 2 třídy)
X, y = make_classification(n_samples=600, n_features=5, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Inicializace stromu s omezenou hloubkou (max_depth=5)
clf_tree = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42)
clf_tree.fit(X_train, y_train)

# 3. Predikce a metriky
y_pred = clf_tree.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")

# 4. Vizuální vykreslení struktury stromu
plt.figure(figsize=(14, 10))
plot_tree(clf_tree, filled=True, fontsize=8)
plt.show()
```

---

## 8. Klíčová úskalí a expertní diagnostika

### 1. Náchylnost k brutálnímu přetrénování (Overfitting)
Pokud necháme rozhodovací strom růst bez omezení (`max_depth=None`, `min_samples_split=2`), strom bude větvit data tak dlouho, dokud každý list nebude obsahovat vzorky pouze jediné třídy ($\text{Gini} = 0$).
- Trénovací přesnost dosáhne **100 %**.
- Testovací přesnost se však zhroutí, protože strom se naučil i náhodný šum trénovací sady.

### 2. Omezení růstu (Regularizace stromu / Pre-pruning):
- `max_depth`: Maximální povolená hloubka stromu (nejdůležitější parametr!).
- `min_samples_split`: Minimální počet vzorků v uzlu potřebný k dalšímu dělení (např. 10 nebo 20).
- `min_samples_leaf`: Minimální počet vzorků, který musí zůstat v koncovém listu (zabraňuje listům s 1 vzorkem).
- `max_features`: Počet náhodně zvažovaných příznaků při každém dělení.

### 3. Ortogonální rozhodovací hranice (Axis-Aligned Splits)
Stromy dělí prostor vždy rovnoběžně s osami souřadnic ($x_j \le \theta$). Pokud je skutečná separační hranice diagonální (např. $x_1 + x_2 \ge 10$), strom ji musí aproximovat schodovitou strukturou s desítkami zbytečných větvení.

### 4. Nestabilita (Vysoká variance)
Nepatrná změna v trénovacích datech může zcela změnit volbu prvního splitu v kořenovém uzlu, což převrátí celou strukturu podstromů. Tato vysoká variance byla hlavní motivací pro vznik **náhodných lesů (Random Forest)**.

---

## 9. Moderní ML & AI kontext (Stav k 09/2026)

### A. Cost-Complexity Pruning (`ccp_alpha` – Post-pruning)
Namísto tupého omezování hloubky se v moderním Scikit-learn strom nechá vyrůst a následně se prořezává minimalizací nákladové funkce:
$$R_\alpha(T) = R(T) + \alpha |T|$$
kde $R(T)$ je chybovost stromu a $|T|$ je počet listů. Cestu nejlepších hodnot $\alpha$ lze získat metodou `cost_complexity_pruning_path`.

### B. Ansámblové metody (State-of-the-Art pro tabulková data)
Samotný rozhodovací strom se dnes v produkci nasazuje zřídka (slouží spíše jako transparentní auditní model). Tvoří však základ nejvýkonnějších algoritmů současnosti:
- **Random Forest:** Bagging mnoha hlubokých nekorelovaných stromů.
- **Gradient Boosting (LightGBM, XGBoost, CatBoost, HistGradientBoostingClassifier):** Postupné trénování mělkých stromů na reziduích předchozích kroků.

### C. Pokročilá interpretovatelnost: TreeSHAP
Pro stromy existuje specializovaný algoritmus **TreeSHAP** (Lundberg et al., 2020), který dokáže spočítat přesné Shapleyho hodnoty v polynomiálním čase $O(TLD^2)$ namísto exponenciálního času, což umožňuje detailní vysvětlení každé predikce i pro obří modely.
