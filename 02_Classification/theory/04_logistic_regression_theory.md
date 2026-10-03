# Logistická regrese: Teoretický rozbor, matematické základy a implementace (09/2026)

> **Zdrojový materiál kurzu:**  
> `Logistic_regression_-_introduction_and_implementation.pdf`  
> Prezentace a metodické pokyny vzdělávacího programu Data Science & Machine Learning (Coders Lab).

---

## 1. Proč logistická regrese, když jsme v bloku klasifikace?

V úvodu druhého dne kurzu vyvstává přirozená otázka: **Proč probíráme model s názvem „regrese“, když řešíme klasifikační úlohy?**

Algoritmy strojového učení lze principiálně rozdělit podle typu výstupu:
1. **Regrese:** Predikce spojité numerické veličiny ($y \in \mathbb{R}$).
2. **Klasifikace:** Predikce příslušnosti do diskrétní třídy ($y \in \{0, 1\}$ pro binární úlohu, resp. $y \in \{1, \dots, K\}$ pro multitřídní úlohu).

Některé algoritmy lze teoreticky aplikovat na oba typy problémů, ale v praxi mají zásadní omezení. Typickým příkladem je **běžná lineární regrese (OLS)**: pokud ji zkusíme použít pro binární klasifikaci kódovanou jako $y \in \{0, 1\}$, narazíme na nepřekonatelné překážky.

Logistická regrese je ve skutečnosti **klasifikační model**, který využívá lineární kombinaci vstupních proměnných, avšak její výstup transformuje nelineární funkcí (tzv. *link function*) na **aposteriorní pravděpodobnost příslušnosti k pozitivní třídě**:

$$P(Y = 1 \mid \mathbf{X} = \mathbf{x}) \in (0, 1)$$

Proto logistická regrese patří mezi základní pilíře **zobecněných lineárních modelů (Generalized Linear Models – GLM)**.

---

## 2. Lineární regrese versus logistická regrese

Oba algoritmy sdílejí lineární jádro: tvoří váženou sumu vstupních příznaků s posunem (interceptem):

$$z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n = \mathbf{x}^T \boldsymbol{\beta}$$

### Proč selhává lineární regrese při klasifikaci?

```text
       Predikce lineární regrese:           Predikce logistické regrese (Sigmoida):
  y ^                                  P(Y=1) ^
    |                / (y > 1!)               |                   .---' (Asymptota 1.0)
1.0 +---------------+                   1.0 +------------------.-+
    |              /                          |                .'  
    |             /                           |              .' (Oblast inflexe)
    |            /                            |            .' 
    |           /                             |          .' 
0.0 +----------/----+                   0.0 +-+---------'--------+ (Asymptota 0.0)
    |         / (y < 0!)                      |
    +--------+----------> x                   +--------+----------> x
```

1. **Neomezený obor hodnot ($y \in \mathbb{R}$):**  
   Lineární funkce $y = \mathbf{x}^T \boldsymbol{\beta}$ může pro extrémní hodnoty $\mathbf{x}$ nabývat hodnot menších než 0 nebo větších než 1. Taková čísla nelze interpretovat jako pravděpodobnost.
2. **Extrémní citlivost na odlehlá pozorování (Outliers):**  
   Přidání bodu s extrémně vysokou hodnotou $x$ (např. pacient s obřím tumorem) posune celou přímku OLS dolů, což může paradoxně zhoršit klasifikaci bodů poblíž rozhodovací hranice.
3. **Nestejný rozptyl chyb (Heteroskedasticita):**  
   Pro binární proměnnou je rozptyl chyby roven $\text{Var}(y \mid x) = p(x)(1 - p(x))$, který závisí na hodnotě $x$. Tím je porušen klíčový předpoklad Gauss-Markovova teorému o homoskedasticitě.
4. **Nenormální rozdělení reziduí:**  
   Chyby $y - \hat{y}$ mohou nabývat pouze dvou hodnot: $1 - \hat{y}$ nebo $-\hat{y}$, tudíž nemohou mít normální rozdělení.

---

## 3. Logistická funkce (Sigmoida)

Klíčovým prvkem logistické regrese je **logistická (sigmoidální) funkce** $\sigma(z)$, která mapuje libovolné reálné číslo $z \in (-\infty, +\infty)$ do intervalu $(0, 1)$:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

kde:
- $z$: lineární kombinace vstupních proměnných ($z = \beta_0 + \sum_{j=1}^n \beta_j x_j$),
- $e$: Eulerovo číslo ($e \approx 2{,}71828$),
- $\sigma(z) = p(\mathbf{x})$: výsledná aposteriorní pravděpodobnost $P(Y = 1 \mid \mathbf{x})$.

### Vlastnosti sigmoidy:
- **Symetrie vůči počátku:** $\sigma(0) = \frac{1}{1 + 1} = 0{,}5$.
- **Asymptotické chování:**  
  Pro $z \to +\infty$ platí $\sigma(z) \to 1$.  
  Pro $z \to -\infty$ platí $\sigma(z) \to 0$.
- **Monotonie:** Funkce je všude ostře rostoucí a hladká (diferencovatelná ve všech bodech).
- **Elegantní derivace:**  
  $$\frac{d\sigma(z)}{dz} = \sigma(z) \cdot (1 - \sigma(z))$$
  Tato vlastnost výrazně zjednodušuje optimalizaci gradientními metodami.

---

## 4. Matematická formulace: Od pravděpodobnosti k šancím a logitu

Abychom plně porozuměli parametrům $\beta_j$, musíme se seznámit se třemi navzájem provázanými koncepty:

### A. Pravděpodobnost ($p$)
Poměr příznivých výsledků ku všem možným:
$$p \in (0, 1)$$

### B. Šance (Odds)
Poměr pravděpodobnosti, že jev nastane, k pravděpodobnosti, že jev nenastane:
$$\text{Odds} = \frac{p}{1 - p} \in (0, +\infty)$$
*Příklad:* Je-li pravděpodobnost onemocnění $p = 0{,}80$, pak šance je $\frac{0{,}80}{0{,}20} = 4$ (šance 4 ku 1).

### C. Logaritmus šancí (Logit / Log-Odds)
Aplikací přirozeného logaritmu na šanci získáme tzv. **logitovou funkci**:
$$\text{logit}(p) = \ln\left(\frac{p}{1 - p}\right) \in (-\infty, +\infty)$$

V logistické regresi předpokládáme, že právě **logit je lineární funkcí vstupních proměnných**:

$$\ln\left(\frac{p(\mathbf{x})}{1 - p(\mathbf{x})}\right) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n$$

Inverzí tohoto vztahu získáme zpět pravděpodobnost vyjádřenou sigmoidou:

$$p(\mathbf{x}) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \dots + \beta_n x_n)}} = \frac{1}{1 + e^{-\mathbf{x}^T \boldsymbol{\beta}}}$$

---

## 5. Klinický příklad z kurzu: Diagnostika tumoru

V materiálech kurzu je uveden realistický medicínský příklad: predikce malignity (zhoubnosti) nádoru na základě věku pacienta a průměru nádoru v milimetrech:

### Zadání parametrů a vstupů:
- Nezávislé proměnné:
  - $x_1$: Věk pacienta ($\text{věk} = 50\text{ let}$)
  - $x_2$: Průměr tumoru ($\text{průměr} = 35\text{ mm}$)
- Odhadnuté koeficienty modelu:
  - $a_1 = 0{,}005$ (koeficient pro věk)
  - $a_2 = 0{,}015$ (koeficient pro průměr tumoru v mm)
  - $b = 0{,}1$ (intercept / absolutní člen)

### Výpočet lineární kombinace ($z$):
$$z = a_1 x_1 + a_2 x_2 + b = 0{,}005 \cdot 50 + 0{,}015 \cdot 35 + 0{,}1$$
$$z = 0{,}25 + 0{,}525 + 0{,}1 = 0{,}875$$

### Transformace sigmoidou na pravděpodobnost:
$$p(\mathbf{x}) = \frac{1}{1 + e^{-0{,}875}} = \frac{1}{1 + 0{,}41686} = \frac{1}{1{,}41686} \approx 0{,}7058 \quad (70{,}6\,\%)$$

**Závěr modelu:** Pravděpodobnost, že nádor daného pacienta je zhoubný, činí $70{,}6\,\%$.

---

## 6. Rozhodovací práh (Cutoff Threshold)

Logistická regrese vrací **pravděpodobnost** $p(\mathbf{x}) \in (0, 1)$. Pro přiřazení do konkrétní diskrétní třídy $\hat{y} \in \{0, 1\}$ musíme zvolit **rozhodovací práh $\theta$**:

$$\hat{y} = \begin{cases} 1 & \text{pokud } p(\mathbf{x}) \ge \theta \\ 0 & \text{pokud } p(\mathbf{x}) < \theta \end{cases}$$

### Výchozí práh ($\theta = 0{,}5$):
- V Scikit-learn je výchozí hranice nastavena na $\theta = 0{,}5$ (50 %).
- Pokud $p(\mathbf{x}) = 0{,}706 \ge 0{,}5 \implies$ model predikuje maligní nádor ($\hat{y} = 1$).
- Pokud by $p(\mathbf{x}) = 0{,}24 < 0{,}5 \implies$ model predikuje benigní nádor ($\hat{y} = 0$).

### Proč je volba prahu $\theta$ kritickým byznysovým a lékařským rozhodnutím?

V medicínské diagnostice (i detekci podvodů) jsou náklady na chyby asymetrické:
- **Falešně negativní výsledek (FN – přehlédnutí zhoubného nádoru):** Ohrožení života pacienta, fatální selhání.
- **Falešně pozitivní výsledek (FP – falešný poplach):** Provedení kontrolní biopsie, stres pacienta, ale život není bezprostředně ohrožen.

V takovém případě je nutné **snížit rozhodovací práh** (např. na $\theta = 0{,}20$ nebo $0{,}30$):
- Tím maximalizujeme **Recall (senzitivitu)** – zachytíme téměř všechny nemocné pacienty za cenu mírného nárůstu falešných poplachů (pokles Precision).

```text
Nízký práh (theta = 0.20)      Výchozí práh (theta = 0.50)      Vysoký práh (theta = 0.80)
-----------------------------------------------------------------------------------------
Vysoký Recall (málo FN)        Vyvážený poměr                   Vysoká Precision (málo FP)
Vhodné pro: Medicína, Fraud    Obecná klasifikace               Vhodné pro: Automatické bany, SPAM
```

---

## 7. Ztrátová funkce: Log Loss (Binary Cross-Entropy)

V lineární regresi jsme minimalizovali reziduální součet čtverců chyb (RSS / MSE) pomocí metody nejmenších čtverců (OLS). 

### Proč nelze v logistické regresi použít MSE?
Pokud bychom do vzorce MSE dosadili sigmoidální funkci $\hat{y}_i = \sigma(\mathbf{x}_i^T \boldsymbol{\beta})$, výsledná ztrátová funkce by byla **nekonvexní** (plná lokálních minim a sedlových bodů). Gradientní algoritmy by v nich uvízly a nenašly globální optimum.

Proto logistická regrese používá **logaritmickou ztrátu (Log Loss / Binary Cross-Entropy)**:

$$L_{\text{log}}(\boldsymbol{\beta}) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln(p_i) + (1 - y_i) \ln(1 - p_i) \right]$$

kde:
- $y_i \in \{0, 1\}$ je skutečná hodnota (Ground Truth),
- $p_i = P(y_i = 1 \mid \mathbf{x}_i)$ je predikovaná pravděpodobnost modelu.

### Mechanismus penalizace:
1. **Pokud je skutečnost $y_i = 1$:**  
   Druhý člen $(1 - y_i) \ln(1 - p_i)$ odpadá a zbývá $-\ln(p_i)$.
   - Pokud model predikuje $p_i = 1{,}0 \implies \text{ztráta} = -\ln(1) = 0$.
   - Pokud model predikuje $p_i \to 0{,}0 \implies \text{ztráta} \to +\infty$ (brutální pokuta za sebevědomou chybu).
2. **Pokud je skutečnost $y_i = 0$:**  
   První člen $y_i \ln(p_i)$ odpadá a zbývá $-\ln(1 - p_i)$.
   - Pokud model predikuje $p_i = 0{,}0 \implies \text{ztráta} = -\ln(1) = 0$.
   - Pokud model predikuje $p_i \to 1{,}0 \implies \text{ztráta} \to +\infty$.

### Detailní rozbor školního příkladu (SPAM filtr ze slajdů):
Mějme 3 testovací e-maily:
- E-mail 0: skutečnost $y_0 = 1$ (SPAM), model odhadl $p_0 = 0{,}9$
- E-mail 1: skutečnost $y_1 = 0$ (HAM), model odhadl $p_1 = 0{,}4$
- E-mail 2: skutečnost $y_2 = 1$ (SPAM), model odhadl $p_2 = 0{,}6$

> ⚠️ **Metodická poznámka ke slajdům kurzu:**  
> Ve slajdu 13 je výpočet uveden s dekadickým logaritmem ($\log_{10}$), kde vyšlo:  
> $L = -\frac{1}{3} \cdot (-0{,}04575 - 0{,}22185 - 0{,}22185) = -\frac{1}{3} \cdot (-0{,}48945) \approx 0{,}16165$.  
> **V moderním strojovém učení i knihovně Scikit-learn se však striktně používá přirozený logaritmus ($\ln$)!**  
> Se standardním přirozeným logaritmem vychází ztráta:  
> $$L = -\frac{1}{3} \cdot \left[ \ln(0{,}9) + \ln(1 - 0{,}4) + \ln(0{,}6) \right] = -\frac{1}{3} \cdot \left[ -0{,}10536 - 0{,}51083 - 0{,}51083 \right] = -\frac{1}{3} \cdot (-1{,}12702) \approx 0{,}37567\text{ nats}$$  
> Ztrátové funkce v různých logaritmických bázích se liší pouze konstantním násobkem ($\ln(x) = \ln(10) \cdot \log_{10}(x) \approx 2{,}3026 \cdot \log_{10}(x)$), takže optimální parametry $\boldsymbol{\beta}$ jsou totožné.

---

## 8. Optimalizace: Metoda maximální věrohodnosti (MLE)

Pro logistickou regresi neexistuje analytické řešení v uzavřeném tvaru (jako normální rovnice $(X^T X)^{-1} X^T y$ v OLS). Parametry $\boldsymbol{\beta}$ se odhadují **Metodou maximální věrohodnosti (Maximum Likelihood Estimation – MLE)**.

### Princip věrohodnosti (Likelihood):
Předpokládejme, že pozorování jsou nezávislá a pocházejí z Bernoulliho rozdělení. Věrohodnostní funkce $L(\boldsymbol{\beta})$ vyjadřuje sdruženou pravděpodobnost, že při zvolených parametrech $\boldsymbol{\beta}$ naměříme přesně ta data, která máme v trénovací sadě:

$$L(\boldsymbol{\beta}) = \prod_{i=1}^N p_i^{y_i} \cdot (1 - p_i)^{1 - y_i}$$

Zlogaritmováním získáme **Log-Likelihood ($\ell(\boldsymbol{\beta})$)**:

$$\ell(\boldsymbol{\beta}) = \sum_{i=1}^N \left[ y_i \ln(p_i) + (1 - y_i) \ln(1 - p_i) \right]$$

Úkolem MLE je **maximalizovat věrohodnost** $\ell(\boldsymbol{\beta})$, což je exaktně ekvivalentní **minimalizaci negativní logaritmické ztráty** (NLL / Log Loss):

$$\boldsymbol{\beta}^* = \arg\max_{\boldsymbol{\beta}} \ell(\boldsymbol{\beta}) = \arg\min_{\boldsymbol{\beta}} L_{\text{log}}(\boldsymbol{\beta})$$

### Numerické řešení v Scikit-learn (Solvery):
Protože gradient nelze přímo položit nule a vyjádřit $\boldsymbol{\beta}$, Scikit-learn používá pokročilé numerické optimalizační algoritmy:
- **`lbfgs` (výchozí):** Quasi-Newtonova metoda s omezenou pamětí (Limited-memory Broyden–Fletcher–Goldfarb–Shanno). Rychlá konvergence na středních datech.
- **`liblinear`:** Koordinátový sestup, vynikající pro menší datasety a $L_1$ regularizaci.
- **`saga`:** Stochastický gradientní sestup (Stochastic Average Gradient), ideální pro rozsáhlé datasety s mnoha tisíci řádky; podporuje $L_1$, $L_2$ i Elastic Net.
- **`newton-cg`:** Newtonova metoda sdružených gradientů pro hladké funkce a $L_2$ regularizaci.

---

## 9. Implementace v Pythonu a Scikit-learn

Knihovna Scikit-learn nabízí třídu `LogisticRegression` v modulu `sklearn.linear_model`.

### Školní ukázka ze slajdů kurzu:

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score

# 1. Vytvoření syntetického klasifikačního datasetu (600 vzorků, 5 příznaků, 2 třídy)
X, y = make_classification(
    n_samples=600, 
    n_features=5, 
    n_classes=2, 
    random_state=42
)

# 2. Rozdělení na trénovací a testovací sadu (70 % train, 30 % test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. Inicializace a trénování modelu
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

# 4. Predikce a výpočet evaluačních metrik
y_pred = log_reg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
```

---

## 10. Kritické zhodnocení a skrytá úskalí (Expertní pohled)

Učebnicový výklad často pomíjí zásadní implementační aspekty, které v praxi vedou k fatálním chybám:

### 1. Skrytá defaultní regularizace v Scikit-learn
- Ve třídě `LogisticRegression` je **ve výchozím stavu zapnutá $L_2$ regularizace** (`penalty='l2'`, parametr $C=1{,}0$).
- Parametr $C$ je inverzní regularizační síla ($C = \frac{1}{\lambda}$). Malé $C$ znamená silnou regularizaci, velké $C$ znamená slabou regularizaci.
- **Důsledek:** Pokud do modelu pustíte neškálované proměnné (např. věk v jednotkách a příjem v desítkách tisíc), $L_2$ penalizace potlačí váhu příjmu mnohonásobně více než váhu věku!
- **Pravidlo z praxe:** **Před logistickou regresí VŽDY použijte `StandardScaler` v rámci `Pipeline`!**

### 2. Dokonalá separabilita a divergence koeficientů (Hauck-Donnerův jev)
- Pokud existuje lineární hranice, která dokonale oddělí třídu 0 od třídy 1 (tzv. *perfect separation*), MLE se pokusí posunout pravděpodobnosti na přesně 0 a 1.
- Koeficienty $\boldsymbol{\beta}$ a intercept $b$ budou divergovat do $\pm\infty$ a směrodatné odchylky explodují.
- **Obrana:** Právě regularizace ($L_2$ Ridge s rozumným $C$) drží koeficienty v mezích a zabraňuje divergenci.

### 3. Multikolinearita
- Stejně jako u lineární regrese OLS, vysoká korelace mezi nezávislými proměnnými způsobuje vysokou varianci odhadů $\boldsymbol{\beta}$.
- Řešením je kontrola VIF (Variance Inflation Factor), redukce dimenze (PCA) nebo $L_1$ Lasso regularizace (`penalty='l1'`, `solver='saga'`), která redundantní příznaky vynuluje.

### 4. Interpretace koeficientů: Poměr šancí (Odds Ratio)
- Na rozdíl od OLS, kde $\beta_j$ představuje přímou změnu v $y$, v logistické regresi $\beta_j$ představuje **změnu logitové funkce**.
- Pro byznysovou interpretaci používáme exponenciálu koeficientu $\exp(\beta_j)$, což je **poměr šancí (Odds Ratio – OR)**:
  $$\text{OR}_j = e^{\beta_j}$$
  - $\text{OR} = 1{,}0$: Příznak nemá na šanci žádný vliv.
  - $\text{OR} > 1{,}0$: Zvýšení $x_j$ o jednotku násobí šanci pozitivního výsledku faktorem $\text{OR}$ (např. $\text{OR} = 1{,}25 \implies$ šance roste o $25\,\%$).
  - $\text{OR} < 1{,}0$: Zvýšení $x_j$ snižuje šanci pozitivního výsledku.

---

## 11. Moderní ML & AI kontext (Stav k 09/2026)

### A. Kalibrace pravděpodobností (Probability Calibration)
Predikce `predict_proba()` nemusí odpovídat skutečné empirické frekvenci jevu (model může být přehnaně sebevědomý nebo naopak podseknutý regularizací).
- V moderním ML hodnotíme kvalitu kalibrace pomocí **Brier Score** a kalibračních křivek (**Reliability Curves**).
- Ke kalibraci používáme `CalibratedClassifierCV` (Plattovo škálování pomocí logistické kalibrace nebo izotonickou regresi).

### B. Explainable Boosting Machines (EBM / InterpretML)
Jako moderní nástupce logistické regrese pro tabulková data se prosadily **Explainable Boosting Machines (EBM)** z balíčku `interpret`.
- EBM zachovávají dokonalou aditivní interpretovatelnost logistické regrese ($g(p) = \beta_0 + \sum f_j(x_j) + \sum f_{jk}(x_j, x_k)$), avšak namísto lineárních vah $\beta_j x_j$ se tvar křivky $f_j(x_j)$ učí pomocí mělkých gradientně boostovaných stromů.
- Poskytují přesnost srovnatelnou s XGBoostem při 100% průhlednosti každého příznaku.

### C. Pokročilá interpretovatelnost: SHAP (SHapley Additive exPlanations)
Pro logistické modely lze využít `shap.LinearExplainer`, který přesně dekomponuje logit predikce na příspěvky jednotlivých příznaků na základě kooperativní teorie her.

### D. MLOps: Threshold Tuning a monitorování pravděpodobnostního driftu
V produkčních systémech se rozhodovací práh $\theta$ nikdy nevolí naslepo jako 0,5:
- Používá se **Cost Matrix Optimization** (minimalizace očekávané celkové finanční ztráty: $\mathbb{E}[\text{Cost}] = C_{\text{FP}} \cdot \text{FP} + C_{\text{FN}} \cdot \text{FN}$).
- Sleduje se drift rozdělení predikovaných pravděpodobností (Population Stability Index – PSI) pro odhalení stárnutí modelu v čase.
