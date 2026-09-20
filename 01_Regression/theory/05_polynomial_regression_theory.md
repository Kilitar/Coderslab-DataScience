# Polynomiální Regrese: Teoretický rozbor a moderní pohled (09/2026)

> **Zdrojový materiál kurzu:**  
> `Polynomial_regression_-_introduction_and_implementation.pdf`  
> (Úvod do polynomiální regrese, matematická definice, implementace pomocí `PolynomialFeatures`, porovnání stupňů polynomu, riziko přetrénování)

---

## 1. Co je polynomiální regrese?

Polynomiální regrese je rozšířením lineární regrese, které umožňuje modelovat **nelineární vztahy** mezi nezávislými proměnnými (prediktory $X$) a závislou proměnnou (odezvou $y$) pomocí polynomů vyšších stupňů.

### Základní rovnice pro jeden prediktor:

V jednoduché lineární regresi (polynom 1. stupně) modelujeme vztah přímkou:
$$y = a_1 x + b + \epsilon$$

Pokud vztah v datech vykazuje zakřivení (např. parabolický tvar), přecházíme na polynom 2. stupně (kvadratickou funkci):
$$y = b + a_1 x + a_2 x^2 + \epsilon$$

Obecný tvar pro polynom $n$-tého stupně pro jednu nezávislou proměnnou:
$$y = b + a_1 x + a_2 x^2 + a_3 x^3 + \dots + a_n x^n + \epsilon = b + \sum_{j=1}^{n} a_j x^j + \epsilon$$

Kde:
- $y$ = závislá (vysvětlovaná) proměnná
- $x$ = nezávislá (vysvětlující) proměnná
- $b$ = absolutní člen (intercept)
- $a_1, a_2, \dots, a_n$ = regresní koeficienty jednotlivých mocnin
- $n$ = stupeň polynomu (degree)
- $\epsilon$ = náhodná chyba (reziduum / šum)

---

## 2. Klíčový teoretický princip: Proč je polynomiální regrese stále „lineárním“ modelem?

Častým omylem začátečníků je představa, že polynomiální regrese je „nelineární model“. Ve statistice a strojovém učení se však linearita posuzuje **vzhledem k parametrům (koeficientům)**, nikoli vzhledem k proměnným:

$$\text{Lineární v parametrech } \beta: \quad y = \beta_0 + \beta_1 z_1 + \beta_2 z_2 + \dots + \beta_n z_n + \epsilon$$

Pokud provedeme substituci příznaků:
$$z_1 = x, \quad z_2 = x^2, \quad z_3 = x^3, \quad \dots, \quad z_n = x^n$$

rovnice přejde na standardní tvar **vícerozměrné lineární regrese**:
$$y = \mathbf{z}^T \boldsymbol{\beta} + \epsilon$$

To znamená:
1. Model se stále učí a optimalizuje pomocí analytické metody nejmenších čtverců (OLS): $\boldsymbol{\beta} = (\mathbf{Z}^T \mathbf{Z})^{-1} \mathbf{Z}^T \mathbf{y}$ nebo gradientního sestupu.
2. Ztrátová funkce zůstává striktně konvexní (má jedno globální minimum).
3. Veškerá „nelinearita“ křivky v původním prostoru $x$ vzniká pouze nelineární transformací příznakového prostoru (Feature Mapping $\phi(x)$).

---

## 3. Polynomy pro vícerozměrná data a interakční členy

Pokud máme více než jeden prediktor (např. $x_1$ a $x_2$), polynomiální rozšíření stupně $d=2$ generuje nejen mocniny, ale také **interakční členy (cross-product terms)**:

$$\phi(x_1, x_2) = [1, \, x_1, \, x_2, \, x_1^2, \, x_1 x_2, \, x_2^2]$$

Obecný model pro dva prediktory 2. stupně:
$$y = b + a_1 x_1 + a_2 x_2 + a_3 x_1^2 + a_4 x_2^2 + a_5 (x_1 x_2) + \epsilon$$

- Člen $x_1 x_2$ zachycuje **synergii / interakci**: vliv proměnné $x_1$ na $y$ závisí na aktuální hodnotě proměnné $x_2$ (např. vliv počtu pokojů na cenu závisí na lukrativitě lokality).

---

## 4. Průběh v Scikit-learn a ukázka z kurzu

Kurz demonstruje polynomiální regresi na syntetických nelineárních datech vygenerovaných kubickou funkcí se šumem:
$$y = X - 2X^2 + 0.2X^3 + \mathcal{N}(-3, 3)$$

### Krok 1: Selhání přímky (Stupeň 1)
Lineární model nedokáže zakřivení postihnout:
- $R^2 \approx 0.09$
- Model trpí vysokým vychýlením (**Underfitting**).

### Krok 2: Kvadratická transformace (Stupeň 2)
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly_2 = PolynomialFeatures(degree=2)
X_poly_2 = poly_2.fit_transform(X)

model_2 = LinearRegression()
model_2.fit(X_poly_2, y)
# Výsledek: R2 vzroste na cca 0.51, parabola lépe vystihuje trend, ale nestačí na inflexní bod
```

### Krok 3: Kubická transformace (Stupeň 3)
```python
poly_3 = PolynomialFeatures(degree=3)
X_poly_3 = poly_3.fit_transform(X)

model_3 = LinearRegression()
model_3.fit(X_poly_3, y)
# Výsledek: R2 vystoupá na cca 0.97, RMSE klesne na 2.59 – odpovídá generujícímu procesu
```

### Krok 4: Extrémní stupeň (Stupeň 10) – Přetrénování (Overfitting)
Při stupni 10 křivka divoce osciluje, snaží se procházet každým jednotlivým bodem včetně náhodného šumu. Výsledkem je dramatická ztráta schopnosti generalizace na nová testovací data.

### Důležitý implementační detail: Vizualizace nelineární křivky
Při vykreslování predikcí nelineárního modelu pomocí `plt.plot(X, y_pred)` je nezbytné body **seřadit podle vzestupné hodnoty $X$**, jinak `matplotlib` propojí body cik-cak čarami:

```python
import operator
import matplotlib.pyplot as plt

# Seřazení podle X před vykreslením
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(X, y_pred_poly), key=sort_axis)
X_sorted, y_pred_sorted = zip(*sorted_zip)

plt.scatter(X, y, s=10, label='Data')
plt.plot(X_sorted, y_pred_sorted, color='green', label='Polynomiální model')
plt.legend()
plt.show()
```

---

## 5. Kritické zhodnocení a skrytá úskalí v praxi

Materiál kurzu vysvětluje základní intuici srozumitelně, avšak v reálné praxi má nekritické nasazení `PolynomialFeatures` několik zásadních rizik, na která kurz neupozorňuje:

> [!CAUTION]
> ### 1. Extrémní multikolinearita (Induced Multicollinearity)
> Mezi mocninami $x, x^2, x^3, \dots$ existuje ze své podstaty obrovská korelace (např. pro kladná čísla je korelace mezi $x$ a $x^2$ často > 0.95).  
> **Důsledek v OLS regresi:** Matice $\mathbf{X}^T \mathbf{X}$ je téměř singulární (špatně podmíněná). Odhadnuté koeficienty mají gigantické rozptyly, ztrácejí jakoukoli statistickou interpretovatelnost a sebemenší změna v trénovacích datech otočí znaménka koeficientů o 180°.

> [!WARNING]
> ### 2. Rungeho fenomén a katastrofální nestabilita na okrajích (Boundary Oscillation)
> Polynomy vysokých stupňů trpí tzv. **Rungeho fenoménem**: u okrajů intervalu trénovacích dat křivka dramaticky diverguje do kladného či záporného nekonečna.  
> **Důsledek:** Jakákoli extrapolace (predikce mimo rozsah trénovacích dat) u polynomiální regrese selhává naprosto fatálně.

> [!WARNING]
> ### 3. Kombinatorická exploze dimenzí (Kletba dimenzionality)
> Počet generovaných příznaků roste kombinatoricky podle vzorce:
> $$\binom{p + d}{d} = \frac{(p + d)!}{p! \, d!}$$
> - Pokud máme **1 prediktor** a stupeň 10: vznikne 11 příznaků (zvládnutelné).
> - Pokud máme v datasetu nemovitostí **20 prediktorů** a zvolíme stupeň 3: vznikne $\binom{23}{3} = 1\,771$ příznaků!
> - Pro stupeň 4 je to již $10\,626$ příznaků.  
> To vede k extrémním nárokům na paměť, přeurčenosti modelu a nutnosti drastické regularizace.

> [!IMPORTANT]
> ### 4. Absolutní nutnost škálování (Feature Scaling) před mocněním
> Pokud vstupní veličina $x$ leží v intervalu $[10, 1000]$ (např. obytná plocha):
> - $x^1 \approx 10^3$
> - $x^2 \approx 10^6$
> - $x^3 \approx 10^9$  
> Numerické knihovny a optimalizátory narážejí na floating-point zaokrouhlovací chyby a regularizační algoritmy (Lasso/Ridge) jsou kompletně paralyzovány, protože váha penalizuje členy s jiným měřítkem zcela nesmyslně.

---

## 6. AI & Modern ML Rozšíření (Stav k září 2026)

Jak se dnes nelineární vztahy v tabulkových datech řeší na špičkové úrovni v moderní datové vědě?

### 1. Splines a B-splines (`SplineTransformer` v Scikit-learn)
Místo globálních polynomů, které oscilují a ovlivňují celou křivku při změně jednoho bodu, se v moderní praxi používají **po částech polynomiální funkce (Splines)**, obvykle kubické B-splines:
- Interval je rozdělen uzly (knots).
- Mezi uzly se fitují kubické křivky, které jsou v uzlech hladce svázány (spojité první i druhé derivace).
- V Scikit-learn je k dispozici nativní `SplineTransformer`, který nahrazuje `PolynomialFeatures` bez rizika Rungeho oscilací.

### 2. Generalized Additive Models (GAMs) a Explainable Boosting Machines (EBM)
Místo ručního volení stupně polynomu se využívají GAM modely:
$$y = \beta_0 + f_1(x_1) + f_2(x_2) + f_{12}(x_1, x_2) + \dots$$
Kde $f_i$ jsou hladké neparametrické funkce (např. z knihovny `InterpretML` / EBM). EBM se učí nelinearity pomocí gradientního boostingu jednodimenzionálních mělkých stromů, což poskytuje přesnost srovnatelnou s XGBoostem při zachování 100% interpretovatelnosti.

### 3. Symbolická regrese řízená AI (`PySR`)
Pokud v moderní vědecké praxi potřebujeme nalézt skutečný fyzikální či ekonomický matematický vzorec, nepoužívá se hrubá síla `PolynomialFeatures`, ale **symbolická regrese** (např. framework `PySR` založený na genetickém programování a neuronových sítích). AI sama prohledá prostor elementárních operací ($+, -, \times, /, \sin, \log, \sqrt{}$) a nabídne Pareto-optimální rovnice vyvažující přesnost a stručnost.

### 4. Robustní Scikit-learn Pipeline s regularizací
Když už se polynomiální regrese použije, v moderním kódu se **vždy kombinuje s normalizací a regularizací (Ridge / Lasso / ElasticNet)** zapouzdřenou v pipeline:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import RidgeCV

# Best practice pipeline pro polynomiální model:
# 1. Škálování vstupních dat
# 2. Generování polynomiálních členů (včetně interakcí)
# 3. Opětovné přeškálování generovaných mocnin
# 4. Ridge regrese s automatickou křížovou validací regularizačního parametru alpha
poly_ridge_pipeline = Pipeline([
    ('scaler_in', StandardScaler()),
    ('poly', PolynomialFeatures(degree=3, include_bias=False)),
    ('scaler_poly', StandardScaler()),
    ('ridge', RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]))
])

poly_ridge_pipeline.fit(X_train, y_train)
```

### 5. XAI a Partial Dependence Plots (PDP)
K interpretaci složitých nelineárních a interakčních závislostí se v roce 2026 rutinně používají:
- **Partial Dependence Plots (PDP)** a **Individual Conditional Expectation (ICE)** grafy (`sklearn.inspection.PartialDependenceDisplay`).
- **SHAP (Shapley Additive exPlanations)** pro dekompozici vlivu jednotlivých transformovaných příznaků na výslednou predikci.

---

## 7. Praktický rozhodovací checklist

| Kritérium | Kdy použít polynomy? | Kdy zvolit modernější alternativu? |
| :--- | :--- | :--- |
| **Dimenze ($p$)** | Malý počet prediktorů ($p < 5$) | Velký počet prediktorů ($p > 10 \implies$ kombinatorická exploze) |
| **Znalost domény** | Teorie předpovídá konkrétní zákonitost (např. kinetická energie $\propto v^2$) | Vztah je neznámý černý box $\implies$ použijte GBDT / EBM / Splines |
| **Extrapolace** | Nikdy! Polynomy u okrajů extrémně divergují | Pro bezpečnou extrapolaci raději regularizované lineární modely |
| **Stabilita vah** | Vyžaduje regularizaci (`RidgeCV`) | Bez regularizace je matice singulární vlivem multikolinearity |
| **Pipeline** | Vždy zabalit do `sklearn.pipeline.Pipeline` | Zabránit úniku informací (data leakage) při transformaci |
