# Regularizace v lineární regresi (L1 Lasso, L2 Ridge, Elastic Net): Teoretický rozbor a moderní pohled (09/2026)

> **Zdrojový materiál kurzu:**  
> `Regularization_in_linear_regression_model_-_introduction_and_implementation.pdf`  
> (Úvod do regularizace, kompromis vychýlení a rozptylu, Lasso L1, Ridge L2, ladění hyperparametru $\alpha$)

---

## 1. Proč regularizace? Kompromis mezi vychýlením a rozptylem (Bias-Variance Tradeoff)

Při trénování modelů strojového učení čelíme fundamentálnímu kompromisu mezi dvěma zdroji chyb:
1. **Chyba vychýlení (Bias):** Zjednodušující předpoklady modelu. Vysoké vychýlení vede k **podtrénování (underfitting)** – model je příliš primitivní na to, aby zachytil skutečné zákonitosti v datech (např. prokládání přímky parabolou).
2. **Chyba rozptylu (Variance):** Citlivost modelu na náhodný šum a fluktuace v trénovací sadě. Vysoký rozptyl vede k **přetrénování (overfitting)** – model se „naučil nazpaměť“ trénovací data včetně jejich šumu a na nových datech selhává.

$$\text{Celková očekávaná chyba} = \text{Bias}^2 + \text{Variance} + \sigma^2_{\text{šum}}$$

Kde $\sigma^2_{\text{šum}}$ je neredukovatelný šum samotného procesu měření.

### Co je regularizace?
**Regularizace** je matematická technika, která vědomě zavádí mírné vychýlení (bias) do optimalizačního kritéria výměnou za **dramatické snížení rozptylu (variance)**.  
Místo pouhé minimalizace ztrátové funkce na trénovacích datech ($\text{Loss}$) minimalizujeme penalizovanou účelovou funkci:

$$\mathcal{J}(\beta) = \text{Loss}(y, \hat{y}) + \lambda \cdot \mathcal{R}(\beta)$$

Kde:
- $\text{Loss}(y, \hat{y})$ je chybová funkce (v lineární regresi obvykle $\text{MSE} = \frac{1}{n} \sum (y_i - \hat{y}_i)^2$).
- $\mathcal{R}(\beta)$ je **regularizační penalizační člen**, který penalizuje příliš vysoké hodnoty regresních koeficientů.
- $\lambda \ge 0$ (v Scikit-learn značený jako parametr `alpha`) je **hyperparametr síly regularizace**:
  - Pokud $\lambda = 0$, dostáváme klasickou neomezenou OLS regresi.
  - Pokud $\lambda \to \infty$, všechny koeficienty jsou stlačeny k nule (predikce se redukuje na pouhý průměr $\bar{y}$).

---

## 2. Dva pilíře kurzu: Lasso (L1) vs. Ridge (L2)

| Vlastnost | Lasso ($L_1$) | Ridge ($L_2$) | Elastic Net ($L_1 + L_2$) |
| :--- | :--- | :--- | :--- |
| **Penalizační člen $\mathcal{R}(\beta)$** | $\sum_{j=1}^{k} \|\beta_j\|$ (Manhattan / $L_1$ norma) | $\sum_{j=1}^{k} \beta_j^2$ (Euklidovská / $L_2^2$ norma) | $\rho \sum \|\beta_j\| + \frac{1-\rho}{2} \sum \beta_j^2$ |
| **Účelová funkce $\mathcal{J}(\beta)$** | $\text{MSE} + \alpha \sum_{j=1}^{k} \|\beta_j\|$ | $\text{MSE} + \alpha \sum_{j=1}^{k} \beta_j^2$ | $\text{MSE} + \alpha \left[ \rho \|\beta\|_1 + \frac{1-\rho}{2} \|\beta\|_2^2 \right]$ |
| **Nulování vah (Sparsity)** | **Ano** (nastavuje méně důležité $\beta_j = 0$) | **Ne** (smršťuje váhy k nule, ale nikdy ne přesně na nulu) | **Ano** (řízeno poměrem $\rho = \text{l1\_ratio}$) |
| **Výběr příznaků (Feature Selection)** | **Automatický vestavěný výběr** | Žádný (všechny proměnné zůstávají v modelu) | **Flexibilní výběr** |
| **Chování při multikolinearitě** | Vybere náhodně jeden korelující prediktor a ostatní vynuluje (nestabilní) | **Ideální**: rozdělí váhu a zmenší rozptyl korelujících prediktorů | **Optimální**: sdružuje korelující prediktory (grouping effect) |
| **Analytické řešení** | Neexistuje v uzavřeném tvaru (vyžaduje Coordinate Descent) | **Existuje**: $\hat{\beta} = (X^T X + \alpha I)^{-1} X^T y$ | Neexistuje v uzavřeném tvaru |

---

### Geometrická interpretace: Proč Lasso nuluje váhy a Ridge ne?

Představme si optimalizaci ve 2D prostoru pro dva parametry $(\beta_1, \beta_2)$:
- Vrstevnice ztrátové funkce MSE tvoří elipsy se středem v neomezeném OLS odhadu $\hat{\beta}_{\text{OLS}}$.
- Regularizační penalizace definuje ohraničenou přípustnou oblast:
  - **Pro Ridge ($L_2$):** Přípustná oblast je **hladký kruh** $\beta_1^2 + \beta_2^2 \le t$. Vrstevnice elipsy se kruhu dotkne téměř jistě v obecném bodě na oblouku, kde jsou obě souřadnice nenulové ($\beta_1 \ne 0, \beta_2 \ne 0$).
  - **Pro Lasso ($L_1$):** Přípustná oblast je **kosočtverec s ostrými rohy** $|\beta_1| + |\beta_2| \le t$, které leží přesně na souřadnicových osách. Elipsa se kosočtverce s vysokou pravděpodobností dotkne právě v některém z těchto ostrých rohů, kde jedna ze souřadnic leží přesně na ose ($= 0$).

---

## 3. Kritické metodické chyby a opomenutí v prezentaci kurzu

> [!CAUTION]
> ### 1. ABSOLUTNÍ METODICKÁ CHYBA: Absence škálování příznaků (Feature Scaling)!
> Na snímcích 9 a 16 prezentace kurzu je uveden následující kód:
> ```python
> # KÓD Z PREZENTACE KURZU:
> lasso_reg = Lasso(alpha=0.1)
> lasso_reg.fit(X_train, y_train)  # <-- ZCELA ŠPATNĚ! X_train NEBYLO ŠKÁLOVÁNO!
> 
> ridge_reg = Ridge(alpha=10)
> ridge_reg.fit(X_train, y_train)  # <-- ZCELA ŠPATNĚ!
> ```
> **Proč je to fatální chyba?**  
> V klasické OLS regresi škálování nezmění předpovědi ani $R^2$ – koeficienty se pouze automaticky přizpůsobí měřítku ($\beta_j \propto \frac{1}{\sigma_j}$).  
> **U regularizace je však situace diametrálně odlišná!**  
> Penalizační člen $\alpha \sum |\beta_j|$ nebo $\alpha \sum \beta_j^2$ trestá všechny koeficienty **stejným metrem**, bez ohledu na to, v jakých jednotkách je daná proměnná měřena!
> - Pokud je $X_1 = \text{sqft\_living}$ v rozsahu $[500, 5000]$ (koeficient je malý, např. $150$), penalizace $\beta_1^2 = 22\,500$.
> - Pokud je $X_2 = \text{bedrooms}$ v rozsahu $[1, 5]$ (koeficient je obrovský, např. $40\,000$), penalizace $\beta_2^2 = 1\,600\,000\,000$.
> 
> *Výsledek:* Model s neškálovanými daty zdecimuje proměnné s malými číselnými hodnotami (jako počet pokojů či pater) a bude tolerovat proměnné s velkými čísly.  
> **Zlaté pravidlo aplikovaného ML:** Před použitím L1/L2 regularizace je **škálování příznaků (např. pomocí StandardScaler či RobustScaler) v praxi zásadní podmínkou férové penalizace**, aby jednotky a měřítka proměnných neurčovaly jejich nespravedlivé potlačení.

> [!WARNING]
> ### 2. Data Leakage při ladění hyperparametru $\alpha$ na testovací sadě!
> Na snímku 17 prezentace autor testuje různé hodnoty $\alpha$ v cyklu:
> ```python
> alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
> for alpha in alphas:
>     ridge_red = Ridge(alpha=alpha)
>     ridge_red.fit(X_train, y_train)
>     y_pred_ridge = ridge_red.predict(X_test)  # <-- DATA LEAKAGE!
>     mean_squared_error(y_test, y_pred_ridge)
> ```
> **Proč je to metodická chyba (Data Snooping)?**  
> Pokud vybíráme nejlepší hodnotu $\alpha$ na základě chyby na `X_test`, testovací sada přestává být nezávislým hodnotícím vzorkem! Informace z testovací sady prosákla do výběru hyperparametrů.  
> **Správný postup:** Hyperparametr $\alpha$ se musí ladit **výhradně na trénovací sadě pomocí křížové validace** (`RidgeCV`, `LassoCV` nebo `GridSearchCV(cv=5)`). Testovací sada musí zůstat zapečetěna až do finálního vyhodnocení.

> [!NOTE]
> ### 3. Překladatelský nonsens: „Trade-off between variance and load“
> Na snímku 3 prezentace se píše: *„One of these is the trade-off between variance and load.“*  
> Zde se jedná o strojový/otrocký překlad z polštiny: polské slovo *obciążenie* znamená v běžné řeči „zátěž/náklad (load)“, ale v matematické statistice a teorii pravděpodobnosti je to odborný termín pro **vychýlení (bias)**.  
> Správný anglický termín je **Bias-Variance Tradeoff** (česky: *kompromis mezi vychýlením a rozptylem*).

> [!IMPORTANT]
> ### 4. Úplná absence Elastic Net v praktické části
> Ačkoliv je Elastic Net letmo zmíněn v úvodu, v kódu zcela chybí.  
> Přitom v reálných datasetech (např. King County Housing i Diamonds), kde existují silně korelované proměnné (`sqft_living` a `sqft_above`, nebo `carat`, `x`, `y`, `z`), má čisté Lasso tendenci chovat se nestabilně. **Elastic Net** kombinuje obě penalizace a je moderním standardem.

---

## 4. Moderní ML & AI Context (Stav k 09/2026)

Jak se regularizované lineární modely používají v moderní datové vědě a produkčním ML:

### 1. `Pipeline` a automatizovaná Cross-Validation (`RidgeCV`, `LassoCV`, `ElasticNetCV`)
V produkčním kódu se nikdy nepíší ruční `for` cykly ani se nevolá `StandardScaler` mimo pipeline:
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV

# Zajišťuje absolutní prevenci data leakage a automatické ladění alpha + l1_ratio
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
        alphas=np.logspace(-4, 3, 50),
        cv=5,
        random_state=42
    ))
])
pipeline.fit(X_train, y_train)
```

### 2. GPU-akcelerované řešiče (RAPIDS cuML)
Při trénování na stamilionech řádků (např. kliky v online reklamě nebo finanční transakce) trvá CPU Coordinate Descent desítky minut.  
S knihovnou `cuml.linear_model.Ridge` nebo `cuml.linear_model.Lasso` (NVIDIA RAPIDS) probíhá výpočet přímo na CUDA jádrech a maticových tenzorech s 50× až 100× zrychlením oproti CPU Scikit-learn.

### 3. Sparse Modeling a komprese v éře LLM
Koncept $L_1$ regularizace z Lassa zažívá obrovskou renesanci v hlubokém učení a GenAI:
- **Model Pruning:** Přidáním $L_1$ penalizace na váhy neuronové sítě nebo LoRA adaptérů se většina vah vynuluje, což umožňuje komprimovat obří modely do mobilních telefonů a Edge zařízení s minimální ztrátou kvality.
- **Sparse Autoencoders (SAE):** V mechanistické interpretovatelnosti (Anthropic, DeepMind) se $L_1$ regularizace používá k extrakci řídkých, srozumitelných konceptů z latentních vrstev transformérů.

### 4. Explainable Boosting Machines (EBM / InterpretML)
Pokud chceme interpretovatelnost lineárního modelu, ale výkon nelineárních modelů, nasazují se v roce 2026 **Generalized Additive Models (GAMs)** s regularizací:
$$g(\mathbb{E}[y]) = \beta_0 + \sum f_j(x_j) + \sum f_{ij}(x_i, x_j)$$
EBM automaticky modeluje hladké nelineární křivky pro jednotlivé příznaky a aplikuje regularizaci na složitost funkcí $f_j$, čímž eliminuje nutnost manuálního vytváření polynomů.

---

## 5. Rychlý tahák implementace v Pythonu (Best Practices)

```python
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Definice gridu pro regularizaci (logaritmická škála)
alphas = np.logspace(-3, 3, 50)

# 2. Dvě úrovně ochrany před únikem dat (Data Leakage):

# Varianta A: RidgeCV uvnitř Pipeline (vhodné pro rychlé prototypování)
# -> Scaler se naučí jednou na celém X_train; vnější testovací sada X_test je striktně izolována.
# -> Upozornění: Uvnitř interních foldů RidgeCV sdílejí validační podmnožiny statistiky celého X_train.
ridge_cv_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RidgeCV(alphas=alphas, cv=5, scoring='neg_mean_squared_error'))
])

# Varianta B: GridSearchCV obalující celou Pipeline (100% rigorózní izolace bez fold-leakage)
# -> StandardScaler se bezpečně učí znovu uvnitř každého jednotlivého foldu z jeho 4/5 trénovacích dat.
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, Lasso

strict_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge())
])
grid_ridge = GridSearchCV(
    estimator=strict_pipe,
    param_grid={'model__alpha': alphas},
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

# 3. Trénování a nalezení optimálního alpha
grid_ridge.fit(X_train, y_train)
best_alpha_ridge = grid_ridge.best_params_['model__alpha']
print(f"Optimální Ridge alpha (GridSearchCV): {best_alpha_ridge:.4f}")

# 4. Jednorázové finální vyhodnocení na zapečetěném testovacím vzorku
y_pred_ridge = grid_ridge.predict(X_test)
print(f"Ridge Test MAE: ${mean_absolute_error(y_test, y_pred_ridge):,.2f} | R²: {r2_score(y_test, y_pred_ridge):.4f}")
```
