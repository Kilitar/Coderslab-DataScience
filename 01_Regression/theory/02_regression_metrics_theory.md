# Metriky regresních modelů: Teoretický rozbor a moderní pohled (09/2026)

> **Zdrojový materiál kurzu:**  
> `Metrics_of_regression_models_-_description_and_sample_of_Python_calculations.pdf`  
> (Definice reziduí, MAE, MSE, RMSE, R² a Adjusted R²)

---

## 1. Co je chyba regrese (Reziduum)?

Základním stavebním kamenem veškerého vyhodnocování regresních modelů je **predikční chyba (reziduum)** pro $i$-té pozorování:

$$e_i = y_i - \hat{y}_i$$

Kde:
- $y_i$ = skutečná hodnota závislé proměnné (ground truth)
- $\hat{y}_i$ = hodnota predikovaná modelem

### Vlastnosti ideálních reziduí (Gauss-Markovovy předpoklady):
1. **Nulová střední hodnota:** $\mathbb{E}[e] = 0$ (model není systematicky vychýlený nahoru ani dolů).
2. **Homoskedasticita:** Rozptyl reziduí je konstantní napříč všemi hodnotami predikcí ($\text{Var}(e_i) = \sigma^2$).
3. **Absence autokorelace:** Rezidua jsou vzájemně nezávislá ($\text{Cov}(e_i, e_j) = 0$ pro $i \ne j$).
4. **Normalita:** Rezidua mají přibližně normální rozdělení $e \sim \mathcal{N}(0, \sigma^2)$.

Protože při tisících pozorováních nelze zkoumat chyby jednotlivě, zavádějí se **agregované souhrnné metriky**.

---

## 2. Pět základních metrik kurzu

### 1. MAE (Mean Absolute Error – Střední absolutní chyba)
Průměr absolutních hodnot odchylek:

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

- **Jednotky:** Shodné s jednotkami cílové proměnné (např. USD, Kč, litry).
- **Interpretace:** V průměru se model plete o $\text{MAE}$ jednotek.
- **Odolnost:** **Vysoká odolnost vůči odlehlým hodnotám**. Lineární penalizace chyb nezpůsobuje dramatické vychýlení celkové metriky jedním extrémním případem.

### 2. MSE (Mean Squared Error – Střední kvadratická chyba)
Průměr druhých mocnin odchylek:

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

- **Jednotky:** Kvadratické ($\text{USD}^2, \text{Kč}^2$) – obtížná přímá byznysová interpretace.
- **Matematický význam:** Přímo odpovídá ztrátové funkci OLS ($L_2$ loss). Je hladce diferencovatelná (má spojité derivace), což je ideální pro optimalizační algoritmy (Gradient Descent).
- **Odolnost:** **Extrémně citlivá na odlehlé hodnoty** (chyba o velikosti 10 jednotek přispěje do sumy hodnotou $100$, zatímco chyba 100 jednotek hodnotou $10\,000$).

### 3. RMSE (Root Mean Squared Error – Odmocnina střední kvadratické chyby)
Odmocněním MSE se metrika vrací do původních jednotek cílové proměnné:

$$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

- **Jednotky:** Původní jednotky ($y$).
- **Vztah k MAE:** Vždy platí $\text{RMSE} \ge \text{MAE}$.  
  - Pokud $\text{RMSE} \approx \text{MAE}$, všechny predikční chyby mají podobnou velikost.
  - Pokud $\text{RMSE} \gg \text{MAE}$, model dělá u menšího počtu vzorků obrovské excesivní chyby.

### 4. $R^2$ (Koeficient determinace – $R^2$ Score)
Udává podíl rozptylu cílové proměnné vysvětlený regresním modelem vzhledem k triviálnímu základnímu modelu (predikce průměrem $\bar{y}$):

$$R^2 = 1 - \frac{\text{SSE}}{\text{SST}} = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

- $R^2 = 1$: Dokonalý model (nulová rezidua).
- $R^2 = 0$: Model má stejnou přesnost jako pouhý průměr $\bar{y}$.
- $R^2 < 0$: Model je na testovacích datech **horší než predikce pouhým průměrem** (častý jev při přetrénování!).

### 5. Adjusted $R^2$ (Upravený koeficient determinace)
Řeší zásadní slabinu klasického $R^2$: s přidáním každé další nezávislé proměnné (i kdyby to byla náhodná čísla) $R^2$ na trénovacích datech **nikdy neklesne**. Adjusted $R^2$ penalizuje model za počet příznaků $k$:

$$R^2_{\text{adj}} = 1 - \left[ \frac{(1 - R^2)(n - 1)}{n - k - 1} \right]$$

Kde:
- $n$ = počet pozorování
- $k$ = počet prediktorů (nezávislých proměnných)

Pokud přidáme proměnnou, která nezvyšuje vysvětlený rozptyl více, než odpovídá penalizaci za ztrátu stupňů volnosti, $R^2_{\text{adj}}$ **klesne**.

---

## 3. Kritické připomínky a metodické chyby v prezentaci kurzu

> [!WARNING]
> ### 1. Chyba v signatuře kódu Scikit-learn v celé prezentaci!
> Na snímcích 10, 13, 16 a 19 prezentace uvádí syntaxi:
> ```python
> # CHYBNĚ v prezentaci kurzu:
> mae = mean_absolute_error(y_pred, y_true)
> r2 = r2_score(y_pred, y_true)
> ```
> **V knihovně Scikit-learn je oficiální a striktní pořadí argumentů:**
> ```python
> # SPRÁVNĚ dle Scikit-learn specifikace:
> mae = mean_absolute_error(y_true, y_pred)
> r2 = r2_score(y_true, y_pred)
> ```
> *Důsledek:* U MAE a MSE je vzorec symetrický ($|a - b| = |b - a|$), takže číselný výsledek náhodou vyjde stejně.  
> **Ale u $R^2$ je prohození fatální chybou!** Pokud prohodíte argumenty v `r2_score(y_pred, y_true)`, ve jmenovateli se spočítá rozptyl predikovaných hodnot namísto rozptylu skutečných hodnot, což vrátí **zcela nesmyslné číslo**!

> [!WARNING]
> ### 2. Mýtus: „$R^2$ nabývá hodnot v rozmezí 0 až 1“
> V prezentaci se doslova uvádí: *„$R^2$ takes values in the range 0-1.“*  
> **To je pravda pouze pro OLS model s interceptem na trénovacích datech.**  
> Na testovací sadě, u nelineárních modelů nebo u regularizovaných modelů může $R^2$ vyjít **hluboko záporné** (např. $-0.75$ nebo $-15.2$). Znamená to, že model natolik přestřelil, že součet čtverců jeho chyb je násobně větší než rozptyl dat kolem průměru.

> [!NOTE]
> ### 3. Absence relativních byznysových metrik (MAPE, WAPE)
> Pro management a klenotníka je absolutní chyba 500 USD nečitelná, pokud neznají hodnotu kamene (u diamantu za 1 000 USD je to obrovská 50% chyba, u diamantu za 50 000 USD zanedbatelné 1 %).  
> V moderní praxi nesmí chybět **MAPE** a **WAPE**.

---

## 4. AI & Modern ML Rozšíření metrik (Stav k září 2026)

Jak se dnes hodnotí a optimalizují regresní modely v produkčním strojovém učení:

### 1. MAPE a WAPE (Relativní procentuální chyby)
- **MAPE (Mean Absolute Percentage Error):**
  $$\text{MAPE} = \frac{100\%}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$
  *Nevýhoda:* Pokud $y_i = 0$ (např. nulové tržby nebo prodeje), nastává dělení nulou.
- **WAPE (Weighted Absolute Percentage Error) – průmyslový standard v retailu:**
  $$\text{WAPE} = \frac{\sum |y_i - \hat{y}_i|}{\sum y_i}$$
  Dělí celkovou sumu absolutních chyb celkovou sumou skutečných hodnot. Nedochází k dělení nulou a dává robustní procentuální číslo přesnosti.

### 2. Huber Loss (Smooth L1 Loss)
Kombinuje to nejlepší z MSE a MAE:
$$L_\delta(e) = \begin{cases} \frac{1}{2} e^2 & \text{pro } |e| \le \delta \\ \delta \cdot (|e| - \frac{1}{2}\delta) & \text{jinak} \end{cases}$$
- Pro malé chyby funguje kvadraticky jako MSE (rychlá konvergence gradientu).
- Pro velké chyby nad práh $\delta$ přechází v lineární MAE (imunita vůči odlehlým hodnotám).
- Využívá se univerzálně v moderních neuronových sítích, PyTorch a `HuberRegressor` v Scikit-learn.

### 3. Kvantilová regrese a Pinball Loss (Predikční intervaly)
V roce 2026 se v byznysu zřídka predikuje pouze bodový odhad (střední hodnota). Klenotník nebo banka potřebují **odhad rizika** (např. 10% a 90% kvantil):
$$L_q(y, \hat{y}) = \max(q(y - \hat{y}), (q - 1)(y - \hat{y}))$$
Natrénováním modelů pro $q=0.1$ a $q=0.9$ získáme **80% predikční interval**, který přímo vymezuje cenové pásmo s garantovanou marží.

### 4. Automatizovaný MLOps Monitoring metrik (Data & Concept Drift)
V produkci se modely hodnotí v reálném čase:
- **Evidently AI / MLflow:** Automaticky sledují klouzavý průměr RMSE a MAE.
- Pokud se na trhu změní chování zákazníků (např. inflace nebo změna poptávky po diamantech), systém detekuje nárůst MAE a automaticky spustí přetrénování pipeline.

---

## 5. Rychlý tahák implementace v Pythonu (Scikit-learn)

```python
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# POZOR: Vždy striktně y_true jako první, y_pred jako druhý argument!
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  # nebo root_mean_squared_error(y_test, y_pred) v moderním sklearn
r2 = r2_score(y_test, y_pred)

# Výpočet Adjusted R2
n = len(y_test)
k = X_test.shape[1]
adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))

# Výpočet WAPE
wape = np.sum(np.abs(y_test - y_pred)) / np.sum(y_test) * 100

print(f"MAE:      ${mae:,.2f}")
print(f"RMSE:     ${rmse:,.2f}")
print(f"R²:       {r2:.4f}")
print(f"Adj. R²:  {adj_r2:.4f}")
print(f"WAPE:     {wape:.2f} %")
```
