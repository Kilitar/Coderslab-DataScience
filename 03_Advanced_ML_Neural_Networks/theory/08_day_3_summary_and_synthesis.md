# Den 3: Velká syntéza pokročilých modelů strojového učení (Day 3 Summary & Synthesis)

> **Cíl modulu:** Poskytnout ucelené shrnutí třetího dne kurzu **Data Science & Machine Learning: Pokročilé modely strojového učení (Advanced Machine Learning Models)**. Syntetizovat tři základní pilíře: **Bagging** (Random Forest), **Boosting** (XGBoost) a **Neuronové sítě** (Keras/TensorFlow), porovnat jejich matematické předpoklady, mechanismus redukce chyby (Bias vs. Variance), limity a doporučení pro praxi k **říjnu 2026**.

---

## 1. Tři pilíře 3. dne kurzu

```
                                    ┌────────────────────────────────────────────────────────┐
                                    │      DEN 3: ADVANCED MACHINE LEARNING MODELS           │
                                    └────────────────────────────────────────────────────────┘
                                                                 │
                 ┌───────────────────────────────────────────────┼───────────────────────────────────────────────┐
                 │                                               │                                               │
                 ▼                                               ▼                                               ▼
   ┌───────────────────────────┐                   ┌───────────────────────────┐                   ┌───────────────────────────┐
   │    1. BAGGING (Ansámbly)  │                   │   2. BOOSTING (Sekvence)  │                   │   3. NEURONOVÉ SÍTĚ (DL)  │
   ├───────────────────────────┤                   ├───────────────────────────┤                   ├───────────────────────────┤
   │ • Nezávislé hluboké stromy│                   │ • Sekvenční mělké stromy  │                   │ • Biologická inspirace    │
   │ • Bootstrap + Subspace    │                   │ • Učení z chyb a reziduí  │                   │ • Vrstvy neuronů (W, b)   │
   │ • Paralelní průměrování   │                   │ • Gradientní sestup       │                   │ • Feedforward + Backprop  │
   │ • Redukce ROZPTYLU (Var)  │                   │ • Redukce VYCHÝLENÍ (Bias)│                   │ • Nelineární aktivace     │
   │ • Algoritmus: Random Forest│                  │ • Algoritmus: XGBoost     │                   │ • Framework: Keras / TF   │
   └───────────────────────────┘                   └───────────────────────────┘                   └───────────────────────────┘
```

---

### 1.1 Pilíř 1: Bagging (Bootstrap Aggregating) & Random Forest
* **Základní myšlenka:** Mnoho plně vyrostlých stromů má nízké vychýlení (*Low Bias*), ale vysoký rozptyl (*High Variance*) – snadno se přeučí na trénovacích datech. Pokud natrénujeme desítky nezávislých stromů a jejich predikce **zprůměrujeme (regrese)** nebo necháme **hlasovat (klasifikace)**, rozptyl celého ansámblu drasticky klesne.
* **Bootstrap sampling:** Každý strom se trénuje na náhodném výběru s opakováním o stejné velikosti jako původní dataset (cca 63.2 % unikátních pozorování, zbytek tvoří Out-of-Bag vzorky).
* **Random Forest inovace (Breiman, 2001):**
  * V každém uzlu stromu se nehledá nejlepší split ze všech proměnných, ale z **náhodné podmnožiny $m$ příznaků** (typicky $m = \sqrt{p}$ pro klasifikaci a $m = p/3$ pro regresi).
  * Tím se **de-korelují jednotlivé stromy** – dominantní příznak neovládne kořeny všech stromů, což vede k ještě větší redukci rozptylu.
* **Výsledky v kurzu:**
  * Klasifikace srdečních chorob: Testovací přesnost **85.71 %** (překonal samostatný strom se 71.4 %).
  * Regrese cen diamantů: Testovací MAE **268.21 USD** (překonal strom s 358 USD).

---

### 1.2 Pilíř 2: Boosting & XGBoost (eXtreme Gradient Boosting)
* **Základní myšlenka:** Místo nezávislých hlubokých stromů staví boosting na sekvenci **velmi mělkých, slabých modelů (*Weak Learners*)**, které mají vysoké vychýlení (*High Bias*), ale nízký rozptyl.
* **Mechanismus sekvence:**
  * Model začíná konstantní predikcí $F_0 = \bar{y}$.
  * Každý další strom $h_m(x)$ se trénuje přímo na **reziduích (chybách)** předchozího stavu: $r_i = y_i - F_{m-1}(x_i)$.
  * Predikce se aktualizují s krokem učení (*Learning Rate* $\eta$):
    $$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$
  * Dochází k **systematickému stlačování vychýlení (Bias)**.
* **Proč XGBoost (Chen & Guestrin, 2016) dominuje:**
  * **Vestavěná regularizace:** Účelová funkce penalizuje počet listů ($\gamma$) i velikost vah ($L_1$ `reg_alpha`, $L_2$ `reg_lambda`).
  * **Sparsity Awareness:** Naučený výchozí směr štěpení pro chybějící hodnoty (`NaN`).
  * **Hardwarová optimalizace:** Cache-aware bloky, komprese dat a plná vícevláknová CPU/GPU paralelizace.
* **Výsledky v kurzu:**
  * Regrese cen diamantů (SOTA ladění): Testovací MAE **264.79 USD** $\implies$ **Absolutně nejlepší výsledek v celém kurzu!**

---

### 1.3 Pilíř 3: Neuronové sítě (Deep Learning) & Keras
* **Základní myšlenka:** Síť umělých neuronů organizovaných do vrstev. Každý neuron počítá vážený součet vstupů a přičítá bias ($z = Wx + b$), který transformuje nelineární aktivační funkcí ($a = f(z)$).
* **Nezbytnost nelinearity:** Bez nelineární aktivace by libovolně hluboká síť zkolabovala do jediné lineární regrese. Aktivace (ReLU, GELU, Softmax) umožňují síti modelovat libovolně složité nelineární prostory (*Universal Approximation Theorem*).
* **Učící mechanismus (Backpropagation):**
  * **Feedforward:** Data protečou sítí ze vstupu na výstup, kde ztrátová funkce vyhodnotí odchylku od reality.
  * **Backpropagation:** Pomocí **řetízkového pravidla diferenciálního počtu (*Chain Rule*)** se spočítá gradient ztráty vůči každé váze:
    $$\frac{\partial E}{\partial w} = \frac{\partial E}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$$
  * Váhy se aktualizují v protisměru gradientu: $w \leftarrow w - \eta \frac{\partial E}{\partial w}$.
* **Výsledky v kurzu:**
  * MNIST rozpoznávání číslic: **97.50 % přesnost** za 6.4 sekundy na 101 770 vahách.
  * King County nemovitosti: Testovací MAE **78 157 USD** při pouhých 655 parametrech (velikost modelu pod 3 KB).

---

## 2. Velká srovnávací matice pokročilých paradigmat

| Vlastnost / Kritérium | 🌲 Random Forest (Bagging) | 🚀 XGBoost (Boosting) | 🧠 Keras MLP (Neuronové sítě) |
| :--- | :--- | :--- | :--- |
| **Základní stavební jednotka** | Nezávislé hluboké stromy | Sekvenční mělké stromy | Plně propojené vrstvy neuronů |
| **Architektura učení** | Paralelní (nezávislá) | Sekvenční (řetězená na chybách) | Dopředný průchod + Backpropagation |
| **Primární redukce chyby** | **Snížení rozptylu (Variance)** | **Snížení vychýlení (Bias)** | **Aproximace obecných nelinearit** |
| **Počet hyperparametrů k ladění** | Nízký (`n_estimators`, `max_depth`, `min_samples_leaf`) | Střední až vysoký (`learning_rate`, `gamma`, `subsample`, `colsample_bytree`) | Vysoký (vrstvy, neurony, aktivace, lr, batch size, epochy) |
| **Nutnost škálování vstupů ($X$)** | ❌ Není nutné (invariatní k měřítku) | ❌ Není nutné | ⚠️ **Kriticky nutné** (`StandardScaler`) |
| **Škálování cílové proměnné ($y$)** | ❌ Není nutné | ❌ Není nutné | ⚠️ **Nutné u regrese** (prevence exploze gradientu) |
| **Citlivost na odlehlé hodnoty (outliers)** | Velmi nízká (průměr zahladí extrémy) | Střední (může se zacyklit na chybách outlierů) | Vysoká (outlier generuje obrovský gradient) |
| **Velikost natrénovaného modelu** | Velká (desítky MB pro 100 stromů) | Malá až střední (stovky KB) | **Extrémně malá** (jednotky KB pro MLP) |
| **Rychlost inference (predikce)** | Střední ($O(M \cdot \text{depth})$) | Rychlá | **Blesková** (čisté násobení matic) |
| **Doména absolutní dominance** | Tabulková data s šumem a outliery | Strukturovaná tabulková data (Kaggle) | Obraz (CNN), Text/NLP (Transformers), Zvuk |

---

## 3. Rozhodovací strom v praxi (Kdy jaký model nasadit k 10/2026?)

```
                                      MÁTE NOVÝ ML PROBLÉM
                                                │
                     ┌──────────────────────────┴──────────────────────────┐
                     ▼                                                     ▼
           Máte nestrukturovaná data?                             Máte tabulková data?
       (Obrázky, Audio, Volný text)                           (Tabulky, CSV, SQL databáze)
                     │                                                     │
                     ▼                                     ┌───────────────┴───────────────┐
           [ NEURONOVÉ SÍTĚ ]                              ▼                               ▼
       • CNN pro počítačové vidění                 Je dataset malý             Je dataset střední
       • Transformers pro NLP                      a zašuměný? (N < 2 000)     až obří? (N > 5 000)
       • Keras / PyTorch framework                         │                               │
                                                           ▼                               ▼
                                                   [ RANDOM FOREST ]             [ XGBOOST / LIGHTGBM ]
                                                   • Nízké riziko přeučení       • Maximální přesnost
                                                   • Žádné ladění scale          • Regularizace L1/L2
                                                   • Okamžitý baseline           • Rychlá inference
```

---

## 4. Klíčové poznatky a poučení z praktických cvičení Dne 3

1. **Bagging vs. Boosting na malých datech:**
   * Na malém datasetu pacientů se srdcem ($N=212$ v tréninku) Random Forest těsně překonal XGBoost, protože paralelní průměrování stromů je imunní vůči lokálnímu šumu, zatímco sekvenční boosting se snažil na šumu učit.
2. **Klinická metrika v medicíně:**
   * Optimalizace čistě na *Precision* v kardiologii vedla k 8 přehlédnutým infarktům (False Negatives). Správný postup vyžaduje ladění rozhodovacího prahu (*Threshold Tuning* $\tau = 0.35$), které snížilo počet nediagnostikovaných pacientů na 2.
3. **Chyba v parametrech zadání (`min_samples_leaf` v XGBoost):**
   * V jádru XGBoostu parametr `min_samples_leaf` neexistuje – nahrazením za nativní `min_child_weight` a přidáním `colsample_bytree` klesla chyba na diamantech na rekordních **264.79 USD**.
4. **Zploštění Flatten v obrazových datech:**
   * Vrstva `Flatten` u číslic MNIST ničí 2D prostorové vazby pixelů. Pro plnohodnotné vidění jsou nezbytné konvoluční sítě (`Conv2D`).
5. **Standardizace u regresních sítí:**
   * U cen nemovitostí v King County bez normalizace cíle nabývala MSE ztráta hodnot $10^{11}$, což vedlo k okamžité divergenci. Standardizace $X$ i $y$ umožnila síti s pouhými 655 vahami dosáhnout MAE 78 157 USD.
