# Elastic Net Regrese: Teoretický rozbor a moderní pohled (09/2026)

> **Kontext a návaznost v kurzu:**  
> Tento modul přímo navazuje na téma **Regularizace v lineární regresi (L1 Lasso a L2 Ridge)**.  
> V materiálech LMS je Elastic Net zmíněn v sylabu, ale chyběl k němu ucelený teoretický rozbor a praktický aparát. Elastic Net představuje přirozenou syntézu obou přístupů a řeší fundamentální nedostatky čistého Lassa.

---

## 1. Proč vznikl Elastic Net? Selhání čistého Lassa

V předchozím modulu jsme viděli dva základní přístupy k regularizaci:
- **Ridge ($L_2$):** Skvěle zvládá multikolinearitu a smršťuje váhy, ale **nevybírá příznaky** (žádný koeficient nenuluje).
- **Lasso ($L_1$):** Provádí automatický výběr příznaků nulováním koeficientů (vytváří řídká řešení / sparsity), ale má **tři zásadní praktická omezení**:

### Tři fatální limity čistého Lassa (Zou & Hastie, 2005):

1. **Problém $p > n$ (Vysoká dimenzionalita):**  
   Pokud je počet příznaků $p$ větší než počet vzorků $n$ (např. v genomice, textové analýze či marketingových profilech), Lasso dokáže vybrat **maximálně $n$ příznaků**. Zbývající příznaky automaticky vynuluje bez ohledu na jejich predikční sílu.
   
2. **Nestabilita a selhání při multikolinearitě (Chybějící Grouping Effect):**  
   Pokud máme skupinu silně korelujících proměnných (např. obytná plocha `sqft_living`, plocha nadzemních podlaží `sqft_above` a rozměry domu), Lasso vybere **náhodně jednu z nich** a ostatní nemilosrdně pošle k nule. Výběr je navíc extrémně nestabilní – sebemenší změna v datech způsobí, že Lasso vybere úplně jinou proměnnou ze skupiny.

3. **Horší predikční výkon než Ridge při silných korelacích:**  
   Empirické i teoretické studie prokázaly, že pokud mezi prediktory existují silné lineární vazby, Ridge regrese téměř vždy překonává Lasso v celkové predikční přesnosti ($R^2$, RMSE).

---

## 2. Matematická formulace Elastic Net

V roce 2005 matematici a statistici **Hui Zou a Trevor Hastie** navrhli novou regularizační techniku nazvanou **Elastic Net**, která kombinuje $L_1$ i $L_2$ penalizaci do jedné účelové funkce.

### Účelová funkce (Loss Function) v Scikit-learn:

$$\mathcal{J}(\boldsymbol{\beta}) = \frac{1}{2n} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \alpha \cdot \rho \|\boldsymbol{\beta}\|_1 + \frac{\alpha (1 - \rho)}{2} \|\boldsymbol{\beta}\|_2^2$$

Kde:
- $\frac{1}{2n} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2$ je standardní střední čtvercová chyba (MSE).
- $\|\boldsymbol{\beta}\|_1 = \sum_{j=1}^{p} |\beta_j|$ je **$L_1$ penalizace (Lasso člen)**.
- $\|\boldsymbol{\beta}\|_2^2 = \sum_{j=1}^{p} \beta_j^2$ je **$L_2$ penalizace (Ridge člen)**.
- $\alpha \ge 0$ (parametr `alpha`) je **celková síla regularizace**:
  - $\alpha = 0 \implies$ běžná neomezená OLS regrese.
  - $\alpha \to \infty \implies$ všechny váhy jsou staženy k nule.
- $\rho \in [0, 1]$ (parametr `l1_ratio`) je **poměr zastoupení $L_1$ ku $L_2$**:
  - $\rho = 1 \implies$ **Čisté Lasso** (100 % $L_1$).
  - $\rho = 0 \implies$ **Čistá Ridge regrese** (100 % $L_2$).
  - $0 < \rho < 1 \implies$ **Skutečný Elastic Net**, kde $L_1$ část zajišťuje nulování nepotřebných vah a $L_2$ část stabilizuje multikolinearitu a zajišťuje skupinový efekt.

---

## 3. Geometrická interpretace a Grouping Effect

### Geometrie přípustné oblasti (Constraint Region)

Proč Elastic Net funguje tak elegantně, pochopíme při pohledu na tvar omezení ve 2D prostoru:
- **Ridge ($L_2$):** Hladká kružnice $\beta_1^2 + \beta_2^2 \le t$ (nemá žádné rohy $\implies$ žádné nulování).
- **Lasso ($L_1$):** Kosočtverec $|\beta_1| + |\beta_2| \le t$ s ostrými rohy na osách (způsobuje nulování vah, ale ploché hrany způsobují skokový výběr proměnných).
- **Elastic Net:** **Zaoblený kosočtverec (hypocykloidní tvar)**:
  - Má **ostré hroty na souřadnicových osách**, což znamená, že vrstevnice elipsy MSE se mohou dotknout osy a nastavit $\beta_j = 0$ (zachovává sparsity a feature selection).
  - Mezi hroty jsou však hrany **přísně konvexně zaoblené** (díky kvadratickému členu $L_2$), což odstraňuje singularitu a vynucuje skupinový efekt.

![Geometrie regularizace](https://raw.githubusercontent.com/Kilitar/Coderslab-DataScience/main/01_Regression/plots/16_regularization_geometric_contours.png)


### Seskupovací efekt (The Grouping Effect)

Klíčovou vlastností Elastic Netu, kterou Zou a Hastie matematicky dokázali, je tzv. **Grouping Effect**:  
Pokud jsou dva prediktory silně korelované (např. $\text{corr}(x_i, x_j) \to 1$), jejich výsledné regresní koeficienty v modelu Elastic Net konvergují k sobě:

$$|\hat{\beta}_i - \hat{\beta}_j| \le \frac{1}{\alpha(1 - \rho)} \sqrt{2(1 - r)}$$

Kde $r = \text{corr}(x_i, x_j)$.  
Pokud je korelace $r \approx 1$, rozdíl mezi koeficienty se blíží nule!  
**Praktický význam:** Pokud máme 5 různých měření velikosti pozemku nebo 10 korelujících finančních ukazatelů, Elastic Net je **buď vybere všechny společně a rozdělí mezi ně váhu, nebo je všechny společně pošle k nule**. Nestane se, že by náhodně ponechal jeden a zbytek zahodil jako Lasso.

---

## 4. Důležitý detail: Naivní Elastic Net vs. skutečný Elastic Net

Při přímé minimalizaci součtu MSE + L1 + L2 dochází k tzv. **dvojímu smrštění (double shrinkage)**:
1. $L_1$ norma zmenší velikost koeficientů (odečte konstantu $\alpha \rho$).
2. $L_2$ norma znovu zmenší koeficienty (vydělí je faktorem $1 + \alpha(1-\rho)$).

To vede k tomu, že predikované hodnoty $\hat{y}$ mají zbytečně velkou systematickou chybu vychýlení (bias). Zou a Hastie proto zavedli korekční faktor:

$$\hat{\boldsymbol{\beta}}_{\text{ElasticNet}} = (1 + \lambda_2) \cdot \hat{\boldsymbol{\beta}}_{\text{Naive}}$$

*Poznámka:* Implementace v `sklearn.linear_model.ElasticNet` provádí tuto optimalizaci pomocí algoritmu **Coordinate Descent**, který konverguje rychle a spolehlivě i pro rozsáhlé matice.

---

## 5. Porovnání všech tří regularizačních technik

| Vlastnost | Ridge ($L_2$) | Lasso ($L_1$) | Elastic Net ($L_1 + L_2$) |
| :--- | :--- | :--- | :--- |
| **Penalizace** | $\|\beta\|_2^2 = \sum \beta_j^2$ | $\|\beta\|_1 = \sum \|\beta_j\|$ | $\rho \|\beta\|_1 + \frac{1-\rho}{2} \|\beta\|_2^2$ |
| **Počet hyperparametrů** | 1 (`alpha`) | 1 (`alpha`) | 2 (`alpha`, `l1_ratio`) |
| **Nulování koeficientů (Sparsity)** | Ne (váhy $\to 0$, ale $\ne 0$) | **Ano** (vytváří řídký model) | **Ano** (řízeno `l1_ratio`) |
| **Chování při $p > n$** | Funguje bez omezení počtu vah | Vybere maximálně $n$ proměnných | **Funguje bez omezení**, může vybrat všech $p$ proměnných |
| **Chování při korelaci ($r \to 1$)** | Rozdělí váhy mezi proměnné | Vybere náhodně jednu, ostatní vynuluje | **Grouping effect** (vybere celou skupinu a rozdělí váhu) |
| **Výpočetní náročnost ladění** | Velmi nízká (1D grid) | Nízká až střední (1D grid) | Vyšší (2D grid pro $\alpha$ a $\rho$) |
| **Doporučené použití** | Všechny příznaky mají malý vliv | Očekáváme malý počet klíčových příznaků | **Reálná data s neznámou strukturou a kolinearitou** |

---

## 6. Správná implementace v Scikit-learn (Best Practices)

Podobně jako u Ridge a Lassa platí nekompromisní pravidla:
1. **Před trénováním VŽDY standardizovat příznaky (`StandardScaler`)**.
2. **Ladění $\alpha$ i `l1_ratio` provádět výhradně křížovou validací v rámci `Pipeline`**.

### Ukázkový kód s `ElasticNetCV`:

```python
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score

# 1. Definice mřížky hyperparametrů
# Tip z praxe: Prozkoumejte hodnoty l1_ratio blízko 1 (např. 0.7, 0.9, 0.99), 
# protože i malé zastoupení L2 výrazně stabilizuje model.
l1_ratios = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99]
alphas = np.logspace(-4, 2, 50)

# 2. Sestavení produkční Pipeline s vestavěnou 5-Fold Cross-Validation
elastic_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', ElasticNetCV(
        l1_ratio=l1_ratios,
        alphas=alphas,
        cv=5,
        max_iter=10000,
        random_state=42,
        n_jobs=-1
    ))
])

# 3. Trénování modelu (transformace probíhá bezpečně uvnitř jednotlivých foldů)
elastic_pipe.fit(X_train, y_train)

# 4. Zjištění nalezených optimálních hyperparametrů
best_model = elastic_pipe.named_steps['model']
print(f"Optimální alpha: {best_model.alpha_:.5f}")
print(f"Optimální l1_ratio: {best_model.l1_ratio_:.2f}")

# 5. Vyhodnocení na nedotčené testovací sadě
y_pred = elastic_pipe.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Test RMSE: {rmse:,.2f}")
print(f"Test R²:   {r2:.4f}")

# 6. Analýza vybraných příznaků
zero_coefs = np.sum(best_model.coef_ == 0)
active_coefs = np.sum(best_model.coef_ != 0)
print(f"Aktivních příznaků: {active_coefs} / {X_train.shape[1]} (Vynulováno: {zero_coefs})")
```

---

## 7. Moderní ML & AI Kontext (Stav k 09/2026)

### 1. Elastic Net jako univerzální filtr před GBDT modely
V moderních soutěžích (Kaggle) a produkčních bankovních systémech se Elastic Net rutinně používá v **dvoufázovém modelování**:
- **Fáze 1 (Screening):** Z tabulky s tisíci generovanými atributy se pomocí `ElasticNetCV` rychle odfiltrují irelevantní signály při zachování skupin korelujících proměnných.
- **Fáze 2 (Ensemble):** Přeživší příznaky vstupují do gradientního boostingu (LightGBM, CatBoost, XGBoost) nebo neuronových sítí.

### 2. GPU Akcelerace (NVIDIA RAPIDS cuML)
Scikit-learn počítá Elastic Net na CPU přes Coordinate Descent sekvenčně. Pro tabulková data s desítkami milionů řádků se dnes využívá `cuml.linear_model.ElasticNet`, který paralelně optimalizuje váhy na GPU tisícinásobnou rychlostí.

### 3. Sparse Deep Learning a regularizace vah
Principy Elastic Netu se uplatňují při regularizaci velkých neuronových sítí:
- Samotná $L_2$ regularizace v hlubokých sítích odpovídá technice **Weight Decay** (AdamW).
- Kombinace $L_1$ a $L_2$ na vahách umožňuje redukovat komplexitu modelů a vytvářet řídké architektury pro Edge AI a on-device inference.

---

## 8. Praktický rozhodovací checklist

> [!TIP]
> **Kdy zvolit Elastic Net namísto Ridge či Lassa?**
> 1. **Máte podezření na multikolinearitu, ale zároveň chcete řídký model (výběr proměnných)?** $\implies$ Zvolte **Elastic Net**.
> 2. **Je počet sloupců větší než počet řádků ($p > n$)?** $\implies$ Lasso selže na stropu $n$, zvolte **Elastic Net**.
> 3. **Nevíte předem, která regularizace bude fungovat lépe?** $\implies$ Zadejte mřížku pro `l1_ratio` od 0.1 do 1.0 v `ElasticNetCV`. Model si sám křížovou validací najde, zda preferuje více Lasso nebo Ridge chování.
