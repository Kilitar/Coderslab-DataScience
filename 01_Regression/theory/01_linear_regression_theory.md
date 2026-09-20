# Lineární Regrese: Teoretický rozbor a moderní pohled (09/2026)

> **Zdrojové materiály kurzu:**  
> - `Linear_regression.pdf` (Teoretické principy, metoda nejmenších čtverců, gradientní sestup)  
> - `Linear_regression_-_sample_implementation.pdf` (Praktická implementace v Scikit-learn na datech King County)

---

## 1. Co je lineární regrese?

Lineární regrese je základní a klíčový algoritmus řízeného učení (Supervised Learning), jehož úkolem je predikovat **spojitou cílovou proměnnou** ($y$) na základě jedné nebo více **nezávislých proměnných / příznaků** ($X$).

Algoritmus předpokládá, že vztah mezi vstupními příznaky a výstupem lze modelovat jako lineární kombinaci:

### Jednoduchá lineární regrese (1 příznak):
$$y = ax + b + \epsilon$$

Kde:
- $y$ = závislá (vysvětlovaná) proměnná (např. spotřeba paliva, cena nemovitosti)
- $x$ = nezávislá (vysvětlující) proměnná (např. průměrná rychlost, obytná plocha)
- $a$ = směrnice / regresní koeficient (slope) – určuje, o kolik se změní $y$, pokud se $x$ zvýší o 1 jednotku
- $b$ = absolutní člen (intercept) – hodnota $y$, pokud je $x = 0$
- $\epsilon$ = náhodná chyba (reziduum / šum), kterou model nedokáže vysvětlit

### Vícerozměrná lineární regrese (Multiple Linear Regression):
$$y = b + a_1 x_1 + a_2 x_2 + \dots + a_p x_p + \epsilon = \mathbf{X}\mathbf{w} + b + \epsilon$$

---

## 2. Hledání optimálních parametrů

Cílem je najít takové parametry ($a, b$, resp. vektor vah $\mathbf{w}$), které minimalizují chybu mezi skutečnými hodnotami $y_i$ a predikovanými hodnotami $\hat{y}_i$.

### Účelová (ztrátová) funkce: Sum of Squared Errors (SSE / RSS)
$$J(a, b) = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \sum_{i=1}^{n} (y_i - (a x_i + b))^2$$

Kurz prezentuje dvě metody minimalizace této funkce:

### Metoda 1: Metoda nejmenších čtverců (OLS – Ordinary Least Squares)
Analytické (uzavřené) řešení spočívající v položení parciálních derivací ztrátové funkce rovno nule:

$$a = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n} (x_i - \bar{x})^2} = \frac{\text{Cov}(x, y)}{\text{Var}(x)}$$

$$b = \bar{y} - a \bar{x}$$

V maticovém tvaru (pro více proměnných):
$$\mathbf{w} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

*Poznámka:* `sklearn.linear_model.LinearRegression` standardně využívá analytické řešení (přes LAPACK rutinu / SVD rozklad `scipy.linalg.lstsq`), což je rychlé a exaktní pro běžně velké matice.

### Metoda 2: Gradientní sestup (Gradient Descent)
Iterativní numerická optimalizace využívaná v situacích, kdy je matice $\mathbf{X}$ příliš velká pro inverzi $(\mathbf{X}^T \mathbf{X})^{-1}$, nebo při obecných ztrátových funkcích:

$$\theta_{\text{nové}} = \theta_{\text{staré}} - \alpha \cdot \frac{\partial J}{\partial \theta}$$

Kde:
- $\alpha$ (v kódu kurzu označeno jako $L$) = rychlost učení (learning rate)
- Parciální derivace pro $a$: $\frac{\partial J}{\partial a} = -\frac{2}{n} \sum_{i=1}^{n} x_i (y_i - \hat{y}_i)$
- Parciální derivace pro $b$: $\frac{\partial J}{\partial b} = -\frac{2}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)$

---

## 3. Vyhodnocení kvality modelu

### Koeficient determinace ($R^2$ Score)
Udává podíl rozptylu cílové proměnné vysvětlený modelem:
$$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2} = 1 - \frac{SS_{res}}{SS_{tot}}$$

- $R^2 = 1$: Dokonalá predikce.
- $R^2 = 0$: Model predikuje pouze průměr $\bar{y}$.
- $R^2 < 0$: Model je na testovacích datech horší než predikce pouhým průměrem.

### Analýza reziduí
Rezidua $e_i = y_i - \hat{y}_i$ by měla vykazovat vlastnosti bílého šumu:
- Normální rozdělení kolem nuly
- Konstantní rozptyl (homoskedasticita)
- Absenci systematických vzorů (nelinearity)

---

## 4. Implementace v Scikit-learn (Shrnutí z prezentace)

Kurz ukazuje základní postup na datech nemovitostí v King County (`kc_house_data.csv`):

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 1. Načtení dat a odstranění neprediktivních identifikátorů
housing_df = pd.read_csv("kc_house_data.csv")
housing_df = housing_df.drop(["id", "date"], axis=1)

# 2. Rozdělení na train a test
X = housing_df.drop("price", axis=1)
y = housing_df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Inicializace a trénování modelu
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# 4. Inspekce parametrů
print("Váhy příznaků (coef_):", lin_reg.coef_)
print("Absolutní člen (intercept_):", lin_reg.intercept_)

# 5. Predikce a evaluace
y_pred = lin_reg.predict(X_test)
print("R2 skóre na testovacích datech:", r2_score(y_test, y_pred))
```

---

## 5. Kritické připomínky k prezentaci kurzu

Následující body představují zásadní metodická zjednodušení v prezentaci, na která je v praxi nutné dát pozor:

> [!WARNING]
> ### 1. Závažná chyba při interpretaci koeficientů bez standardizace!
> V prezentaci se tvrdí: *„Čím vyšší je hodnota koeficientu, tím silnější je vliv proměnné na výsledek.“* A autoři srovnávají koeficient `-34308` (pro počet ložnic) s koeficientem `112` (pro obytnou plochu ve sqft).  
> **Toto tvrzení je bez standardizace příznaků metodicky chybné.**
> - Počet ložnic se pohybuje v jednotkách (1 až 5), zatímco plocha `sqft_living` se pohybuje v tisících (500 až 10 000 sqft).
> - Zvýšení plochy o 1 000 sqft zvýší cenu o $112 \times 1000 = 112\,000\ \text{USD}$, což má v reálu mnohem větší celkový vliv než změna počtu ložnic o 1!
> - **Pravidlo:** Absolutní velikosti koeficientů lze pro určení důležitosti (Feature Importance) porovnávat **POUZE tehdy, pokud jsou všechny proměnné převedeny na stejné měřítko (např. pomocí `StandardScaler`)**!

> [!WARNING]
> ### 2. Zjednodušený přístup k multikolinearitě
> V prezentaci se uvádí: *„Při vysoké korelaci dvou proměnných jednu vyhoďte.“*
> - Pár-korelační matice (Pearson) nestačí. Proměnná může být lineární kombinací 3 nebo 4 jiných proměnných (tzv. skrytá multikolinearita).
> - Pro správnou detekci je nutné použít **VIF (Variance Inflation Factor)**.
> - Místo manuálního a svévolného mazání sloupců (což může vést ke ztrátě doménových informací) se v moderní praxi používá **L2 regularizace (Ridge regrese)** nebo **ElasticNet**, které multikolinearitu stabilizují automaticky.

> [!NOTE]
> ### 3. Citlivost na odlehlé hodnoty vs. jejich unáhlené mazání
> Prezentace doporučuje se odlehlým hodnotám vyhnout (vymazat je).
> - U nemovitostí ale extrémní hodnoty (luxusní vily) často nejsou chyby měření, nýbrž validní data.
> - Jednoduché smazání zkreslí schopnost modelu predikovat dražší segment trhu.
> - Moderní přístup: Použití **logaritmické transformace cíle** (`log(price)`) nebo **robustní regrese** (Huber Regressor, RANSAC).

---

## 6. AI & Modern ML Rozšíření (Stav k září 2026)

Jak se dnes lineární modely reálně používají v moderní datové vědě obohacené o AI a pokročilé techniky?

### 1. LLM-Assisted Feature Engineering (CAAFE)
Lineární regrese je omezena tím, že nedokáže sama zachytit nelinearity ani interakce mezi příznaky (např. poměr `sqft_living / bedrooms` nebo `price_per_sqft` v daném zip kódu).
- **V roce 2026:** K datům přistupují specializovaní AI agenti / LLM (metodiky typu *Context-Aware Automated Feature Engineering*), kteří na základě doménového popisu tabulky automaticky navrhují smysluplné nelineární transformace a interakce předtím, než data vstoupí do lineárního modelu.

### 2. Explainable Boosting Machines (EBM / InterpretML)
Pokud je cílem perfektní interpretovatelnost (proč banka zamítla úvěr nebo jak model oceňuje dům), lineární modely mají konkurenci v podobě **EBM (Generalized Additive Models s interakcemi)**:
$$g(y) = \beta_0 + \sum f_i(x_i) + \sum f_{ij}(x_i, x_j)$$
EBM zachovává exaktní tabulkovou interpretovatelnost každého příznaku podobně jako regresní koeficient, ale křivka $f_i$ může být libovolně nelineární (vytrénovaná boostingem jednorozměrných stromů).

### 3. Symbolic Regression (PySR)
Namísto manuálního hádání polynomiálních členů ($x^2, \sqrt{x}, \log(x)$) se dnes používá **symbolická regrese řízená strojovým učením** (např. knihovna `PySR`). Algoritmus automaticky prozkoumá prostor matematických vzorců a najde nejjednodušší uzavřenou analytickou formuli, která minimalizuje chybu a penalizuje složitost.

### 4. Moderní Scikit-learn Pipeline s transformací cílové proměnné
V moderním kódu nikdy netrénujeme `LinearRegression` na neškálovaných surových datech bez ošetření distribuce:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
import numpy as np

# Robustní moderní pipeline:
# 1. Standardizace vstupů pro srovnatelnost koeficientů
# 2. Log-transformace výstupu (ceny) pro normalizaci rozdělení reziduí
modern_model = TransformedTargetRegressor(
    regressor=Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ]),
    func=np.log1p,
    inverse_func=np.expm1
)

modern_model.fit(X_train, y_train)
```

### 5. XAI a globální interpretace přes SHAP
Místo spoléhání se na surové koeficienty se pro interpretaci lineárních i nelineárních modelů univerzálně používají **SHAP (Shapley Additive exPlanations)** hodnoty:
- Umožňují sjednotit interpretaci lineárních modelů, Random Forestů i XGBoostu na stejném matematickém základě (teorie kooperativních her).

---

## 7. Shrnutí a doporučený checklist pro praxi

1. **Prověřit distribuci $y$**: Je sešikmená? Zvážit $\log(y)$.
2. **Standardizovat $X$**: Chceme-li interpretovat význam koeficientů, vždy použít `StandardScaler`.
3. **Zkontrolovat multikolinearitu**: Spočítat VIF, při vysokých hodnotách (> 5–10) přejít na Ridge regresi.
4. **Vizuální diagnostika reziduí**: Zkontrolovat scatter plot $y_{pred}$ vs. $e_i$ a Q-Q plot normality.
5. **Použít Pipeline**: Zamezit data leakage mezi trénovací a testovací sadou.
