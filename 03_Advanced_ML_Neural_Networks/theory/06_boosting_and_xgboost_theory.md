# Den 3: Boosting a XGBoost – Teorie, Matematika, Scikit-learn API & SOTA 10/2026

> **Cíl modulu:** Detailně prostudovat sekvenční ansámblovou techniku **Boosting**, která kombinuje slabé modely (*Weak Learners*) do jednoho vysoce přesného prediktoru. Rozebrat rozdíly mezi **AdaBoostem**, **Gradient Boostingem** (minimalizace reziduí přes gradient) a algoritmem **XGBoost** (*eXtreme Gradient Boosting*). Krok za krokem matematicky projít výpočet reziduí z přednášky, rozebrat kompletní sadu hyperparametrů a regularizací a doplnit expertní kritiku a srovnání s moderními alternativami (**LightGBM**, **CatBoost**) k **říjnu 2026**.

---

## 1. Originální teorie kurzu (Výtah z prezentací)

### 1.1 Co je Boosting a jak se liší od Baggingu?
V předchozí části jsme viděli, že **Bagging** trénuje mnoho nezávislých, plně vyrostlých stromů paralelně s cílem **drasticky snížit rozptyl (*Variance*)**.

**Boosting volí opačnou filozofii:**
- Vychází ze **slabých modelů (*Weak Learners*)**, typicky velmi mělkých rozhodovacích stromů (např. o hloubce 1 až 3).
- Tyto modely mají nízký rozptyl, ale **vysoké vychýlení (*High Bias*)**.
- Modely se netrénují nezávisle, ale **sekvenčně za sebou**.
- **Klíčový princip:** Každý nový model se učí z chyb a nedostatků modelů předchozích a snaží se je kompenzovat.
- **Výsledek:** Postupným zřetězením dochází k **radikálnímu snížení vychýlení (Bias)** při zachování nízkého rozptylu.

```
Původní dataset
       │
       ▼
   [Model 1] ──► Spočtení chyb / reziduí
       │
       ▼
  Převážená data / Rezidua
       │
       ▼
   [Model 2] ──► Spočtení chyb / reziduí
       │
       ▼
   [Model 3] ──► ... ──► [Model N] ──► Finální vážená predikce
```

---

### 1.2 Dva hlavní přístupy v Boostingu: AdaBoost vs. Gradient Boosting

#### 1. AdaBoost (Adaptive Boosting)
- **Mechanismus vah:** Na začátku mají všechna pozorování stejnou váhu $w_i = \frac{1}{N}$.
- V každém kroku model natrénuje slabý klasifikátor.
- Pozorováním, která byla **klasifikována chybně, je přiřazena vyšší váha**, zatímco správně určeným vzorkům váha klesne.
- V dalším kroku je model nucen zaměřit se na tyto obtížné, problematické případy.
- Samotným modelům je přiřazena váha $\alpha_m$ podle jejich celkové chybovosti (přesnější modely mají v hlasování větší slovo).

#### 2. Gradient Boosting
- **Mechanismus reziduí:** Modely se netrénují na převážených původních datech, ale **přímo na reziduích (chybách)** předchozího prediktoru:
  $$\text{Reziduum } r_i = y_i - \hat{y}_i$$
- První model $F_0$ neprovádí žádné štěpení, ale predikuje **konstantní průměr cílové proměnné**:
  $$F_0(x) = \bar{y}$$
- Další strom $h_1(x)$ se natrénuje tak, aby předpovídal rozdíl $y - F_0$.
- Nová predikce po 1. kroku je:
  $$F_1(x) = F_0(x) + \eta \cdot h_1(x)$$
  kde $\eta$ je rychlost učení (*Learning Rate*).
- Ztrátová funkce $L(y, \hat{y})$ je optimalizována pomocí **gradientního sestupu** (počítá se záporný gradient ztrátové funkce).

---

### 1.3 Matematický příklad ze slajdů krok za krokem (Slajdy 14–17)

Přednáška demonstruje princip na 10 pozorováních s jednou nezávislou proměnnou $X$ a spojitou cílovou proměnnou $Y$:

#### Krok 0: Výchozí konstantní model $F_0$
- Spočteme průměr hodnot $Y$:
  $$F_0 = \frac{82 + 80 + 103 + 118 + 172 + 127 + 204 + 189 + 99 + 166}{10} = \frac{1340}{10} = \mathbf{134}$$
- Spočteme výchozí rezidua $Y - F_0$:
  - Pro $X=5, Y=82 \to 82 - 134 = -52$
  - Pro $X=7, Y=80 \to 80 - 134 = -54$
  - Pro $X=29, Y=204 \to 204 - 134 = +70$

#### Krok 1: První rozhodovací strom $H_1$
- Natrénujeme jednoduchý strom na predikci reziduí $Y - F_0$ na základě $X$.
- Nalezený split: $X \le 23$:
  - Pro $X \le 23$ (4 vzorky: $-52, -54, -31, -16$): průměr $H_1 = \mathbf{-38.25}$.
  - Pro $X > 23$ (6 vzorků: $38, -7, 70, 55, -35, 32$): průměr $H_1 = \mathbf{+25.50}$.
- Nová predikce modelu $F_1 = F_0 + H_1$:
  - Pro $X \le 23$: $F_1 = 134 - 38.25 = \mathbf{95.75}$
  - Pro $X > 23$: $F_1 = 134 + 25.50 = \mathbf{159.50}$
- Spočteme nová rezidua: $Y - F_1$ (chyba se dramaticky zmenšila!).

#### Krok 2: Druhý rozhodovací strom $H_2$
- Natrénujeme další strom na nová rezidua $Y - F_1$.
- Nalezený split: $X \le 34$:
  - Pro $X \le 34$: $H_2 = \mathbf{+6.75} \to F_2 = F_1 + 6.75$
  - Pro $X > 34$: $H_2 = \mathbf{-27.00} \to F_2 = F_1 - 27.00$

#### Krok 3: Třetí rozhodovací strom $H_3$
- Nalezený split: $X \le 28$:
  - Pro $X \le 28$: $H_3 = \mathbf{-10.08} \to F_3 = F_2 - 10.08$
  - Pro $X > 28$: $H_3 = \mathbf{+15.125} \to F_3 = F_2 + 15.125$

#### Výsledek po 3 krocích:
Křivka predikce $F_3$ přesně sleduje reálný tvar dat a rezidua konvergují k nule!

---

### 1.4 Co je XGBoost a proč je tak úspěšný?

**XGBoost (*eXtreme Gradient Boosting*)** vytvořil **Tianqi Chen** v roce 2014. Stal se nejpoužívanějším algoritmem na platformě **Kaggle** a v průmyslu díky klíčovým inovacím:

1. **Vestavěná regularizace (L1 a L2):**  
   Na rozdíl od standardního Gradient Boostingu zahrnuje XGBoost do své účelové funkce penalizaci složitosti stromů:
   $$\text{Obj} = \sum_{i=1}^N L(y_i, \hat{y}_i) + \sum_{m=1}^M \left(\gamma T_m + \frac{1}{2} \lambda \sum_{j=1}^{T_m} w_j^2 + \alpha \sum_{j=1}^{T_m} |w_j|\right)$$
   kde $T_m$ je počet listů a $w$ jsou váhy listů.
2. **Weighted Quantile Sketch:**  
   Algoritmus neprochází všechny možné dělicí body hrubou silou, ale hledá kandidáty štěpení pomocí vážených percentilů, což dramaticky urychluje trénink.
3. **Sparsity Awareness (Přirozená podpora chybějících hodnot):**  
   XGBoost se při tréninku učí výchozí směr (*default direction*) pro chybějící hodnoty v každém uzlu. Není nutné imputovat `NaN`!
4. **Hardwarová optimalizace:**  
   Využití vyrovnávací paměti procesoru (Cache-aware access), out-of-core výpočty pro datasety větší než RAM, plná vícevláknová CPU paralelizace i podpora GPU.

---

### 1.5 Kompletní přehled hyperparametrů v Scikit-learn rozhraní (`XGBRegressor` / `XGBClassifier`)

| Hyperparametr | Výchozí hodnota | Význam & Praktický dopad |
| :--- | :---: | :--- |
| `n_estimators` | `100` | Počet stromů (kol sekvenčního učení). |
| `learning_rate` ($\eta$) | `0.3` | Krok gradientního sestupu. Nižší hodnota (0.01–0.1) vyžaduje více stromů, ale výrazně lépe generalizuje. |
| `max_depth` | `6` | Maximální hloubka stromů. U boostingu se volí spíše menší (3 až 8). |
| `min_child_weight` | `1` | Minimální součet vah hessiánů (pozorování) v listu. Zvýšení působí silně protipřeučení. |
| `gamma` ($\gamma$) | `0` | Minimální snížení ztráty nutné k provedení štěpení. Globální regularizátor stromu. |
| `subsample` | `1.0` | Podíl náhodně vzorkovaných řádků pro každé kolo (stochastický gradient boosting). |
| `colsample_bytree` | `1.0` | Podíl náhodně vybíraných sloupců pro každý strom (obdoba Feature Baggingu). |
| `reg_alpha` ($\alpha$) | `0` | L1 regularizace vah listů (Lasso). Vede k nulování nedůležitých vah. |
| `reg_lambda` ($\lambda$) | `1` | L2 regularizace vah listů (Ridge). Tlumí extrémní váhy listů. |
| `scale_pos_weight` | `1` | Poměr $\frac{N_{neg}}{N_{pos}}$ pro vyvážení nevyvážených binárních tříd. |
| `objective` | `'reg:squarederror'` / `'binary:logistic'` | Ztrátová funkce k minimalizaci. |
| `tree_method` | `'auto'` | Algoritmus tvorby stromů (`'hist'`, `'exact'`, `'approx'`). |

---

### 1.6 Příklad z přednášky: Reality Melbourne (`melb_house_data.csv`)

```python
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

melb_house_df = pd.read_csv("melb_house_data.csv").dropna()
X = melb_house_df.drop("price", axis=1)
y = melb_house_df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)

model = xgb.XGBRegressor(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.2,
    gamma=1,
    random_state=42
)
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

mae_train = mean_absolute_error(y_train, y_pred_train) # 254 450 AUD
mae_test = mean_absolute_error(y_test, y_pred_test)   # 260 891 AUD
```

#### Zhodnocení:
- Rozdíl mezi train chybu (254 450 AUD) a test chybou (260 891 AUD) je minimální $\to$ model **vůbec netrpí přeučením**.
- Nicméně chyba nad 250 000 AUD ukazuje na **vysoké vychýlení (*Underfitting / Bias*)**, způsobené nízkou hloubkou `max_depth=3`, malým počtem stromů `n_estimators=50` a absencí feature engineeringu.

---

## 2. Expertní kritická analýza & Úskalí Boostingu

1. **Citlivost na odlehlé hodnoty a šum (Outlier Sensitivity):**
   - Na rozdíl od Random Forestu (který odlehlé hodnoty zprůměruje a potlačí) se Boosting **snaží za každou cenu opravit největší chyby**.
   - Pokud je v datech extrémní překlep nebo chybná hodnota, Boosting jí přiřadí obrovská rezidua a další stromy budou plýtvat kapacitou na vysvětlení tohoto šumu.
2. **Přeučení při příliš mnoha kolech bez `early_stopping`:**
   - U Random Forestu přidávání stromů nikdy nezvyšuje přeučení (rozptyl asymptoticky klesá).
   - U Boostingu přidávání stromů **může vést k těžkému přeučení**, pokud ztráta na validačních datech začne stoupat. Je povinné používat **Early Stopping** (`early_stopping_rounds=10`).
3. **Vztah mezi `learning_rate` a `n_estimators`:**
   - Snížení rychlosti učení (např. z 0.3 na 0.05) vyžaduje zhruba 3x až 6x více stromů, ale téměř vždy vede k lepšímu a robustnějšímu modelu na testovacích datech.

---

## 3. Moderní SOTA kontext (Stav k 10/2026): Velký souboj na tabulkových datech

K říjnu 2026 tvoří špičku tabulkového machine learningu tři dominantní knihovny:

| Vlastnost | XGBoost | LightGBM (Microsoft) | CatBoost (Yandex) |
| :--- | :--- | :--- | :--- |
| **Způsob růstu stromů** | Depth-wise (po patrech) | **Leaf-wise** (podle největšího zisku) | Symmetric Trees (vyvážené stromy) |
| **Dělení spojitých proměnných** | Exact / Hist / Quantile | **GOSS** (Gradient-based One-Side Sampling) | Minimal Variance Sampling (MVS) |
| **Kategorické proměnné** | One-Hot / Experimental | Integer encoding / Binning | **Nativní Target Encoding bez leakage** |
| **Rychlost tréninku** | Velmi vysoká (XGBoost 2.0+) | **Extrémní** | Vysoká na GPU |
| **Náchylnost k přeučení** | Nízká (L1/L2 regularizace) | Střední (nutno hlídat `num_leaves`) | **Nejmenší (out-of-the-box)** |
| **Vysvětlitelnost (XAI)** | **TreeSHAP nativně** | TreeSHAP nativně | TreeSHAP nativně |

### Moderní doporučení k 10/2026:
- Pro střední a větší tabulková data s mnoha kategoriemi: **CatBoost** nebo **LightGBM**.
- Pro robustní produkční MLOps s přísnou regularizací a integrací do C++ / GPU pipelines: **XGBoost 2.x**.
- Pro GPU trénování na NVIDIA: nastavení `tree_method="hist", device="cuda"`.
