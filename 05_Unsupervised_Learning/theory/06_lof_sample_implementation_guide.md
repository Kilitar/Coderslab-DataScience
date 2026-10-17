# 🛠️ Průvodce: Vzorová implementace Local Outlier Factor (LOF) v Pythonu

> **Přednáškový základ:** *Local_Outlier_Factor_-_sample_implementation.pdf* (Coders Lab Data Science Track)  
> **Klíčová témata:** Ruční krok za krokem výpočet LOF na 5 bodech ($X_1 \dots X_5$), implementace v Scikit-learn na `make_blobs(n_samples=500, centers=2)`, atribut `negative_outlier_factor_`, interpretace výsledků a bilance výhod i nevýhod.

---

## 1. Pedagogický cíl prezentace

Cílem této vzorové implementace je demystifikovat matematické jádro algoritmu **Local Outlier Factor (LOF)**:
1. **Ruční demonstrace:** Nejprve projít krok za krokem výpočet dosahové vzdálenosti ($RD$), lokální dosahové hustoty ($LRD$) a finálního poměru $LOF$ na jednoduché soustavě 5 bodů bez použití knihoven.
2. **Scikit-learn implementace:** Následně převést koncept do moderního kódu v Pythonu na syntetickém shlukovém datasetu (`make_blobs`, 500 vzorků, 2 centra).
3. **Kritická reflexe:** Rozebrat výhody (detekce v lokálních shlucích, libovolné metriky) a úskalí (výpočetní složitost $\mathcal{O}(N^2)$, nestandardizovaná škála skóre).

---

## 2. Ruční výpočet na 5 bodech krok za krokem (Slajdy 9–13)

Uvažujme 5 bodů v rovině $X_1, X_2, X_3, X_4, X_5$ a zvolme parametr $k = 2$ (2 nejbližší sousedé):

```
         Y ^
           |        X3 (0,2)     X4 (3,2)
           |
           |
           |        X1 (0,0) --- X2 (2,0) --------------- X5 (6,0)
           +---------------------------------------------------> X
```

### Krok 1: $k$-vzdálenost a dosahová vzdálenost (Reachability Distance, RD)
Dosahová vzdálenost mezi bodem $A$ a $B$ je:
$$RD(A, B) = \max\left( K\text{-distance}(B),\, d(A, B) \right)$$

Pro bod $X_1$ a jeho dva nejbližší sousedy $X_2$ a $X_3$:
- $K\text{-distance}(X_2) = 2$ (vzdálenost k jeho 2. sousedovi).
- Eukleidovská vzdálenost $d(X_1, X_2) = 2$.
- $RD(X_1, X_2) = \max(2, 2) = 2$.
- Obdobně: $RD(X_1, X_3) = \max(2, 2) = 2$.

### Krok 2: Lokální dosahová hustota (Local Reachability Density, LRD)
Hodnota $LRD$ pro daný bod je převrácenou hodnotou průměrné dosahové vzdálenosti jeho $k$ sousedů:

$$LRD_k(A) = \frac{1}{\frac{1}{|N_k(A)|} \sum_{X_i \in N_k(A)} RD(A, X_i)}$$

Pro bod $X_1$:
$$LRD_2(X_1) = \frac{1}{\frac{RD(X_1, X_2) + RD(X_1, X_3)}{2}} = \frac{1}{\frac{2 + 2}{2}} = \frac{1}{2} = 0.5$$

Pro body $X_2$ a $X_3$ symetricky vychází:
$$LRD_2(X_2) = 0.5, \quad LRD_2(X_3) = 0.5$$

Pro odlehlejší body $X_4$ a $X_5$:
$$LRD_2(X_4) = \frac{1}{3} \approx 0.3333$$
$$LRD_2(X_5) = \frac{1}{6} \approx 0.1667$$

### Krok 3: Výpočet Local Outlier Factor (LOF)
Faktor $LOF$ je poměr průměrné hustoty sousedů ku hustotě zkoumaného bodu:

$$LOF_k(A) = \frac{\frac{1}{|N_k(A)|} \sum_{X_i \in N_k(A)} LRD_k(X_i)}{LRD_k(A)}$$

1. **Bod $X_1$ (Inlier):**
   $$LOF_2(X_1) = \frac{\frac{LRD_2(X_2) + LRD_2(X_3)}{2}}{LRD_2(X_1)} = \frac{\frac{0.5 + 0.5}{2}}{0.5} = \frac{0.5}{0.5} = \mathbf{1.0}$$
   Hodnota $1.0$ značí naprosto normální bod ležící uvnitř homogenního shluku.
   
2. **Bod $X_4$ (Mírný outlier):**
   $$LOF_2(X_4) = \frac{\frac{0.5 + 0.5}{2}}{\frac{1}{3}} = \frac{0.5}{\frac{1}{3}} = 0.5 \cdot 3 = \mathbf{1.5}$$
   Hodnota $1.5 > 1.0$ indikuje počínající odlehlost.

3. **Bod $X_5$ (Výrazná anomálie):**
   $$LOF_2(X_5) = \frac{\frac{0.5 + \frac{1}{3}}{2}}{\frac{1}{6}} = \frac{\frac{5}{6}}{2} \cdot 6 = \frac{5}{12} \cdot 6 = \mathbf{2.5}$$
   Hodnota $2.5 \gg 1.0$ jasně prokazuje silnou anomálii!

---

## 3. Implementace v Scikit-learn (Slajdy 15–20)

### 1. Generování syntetických dat (`make_blobs`)
Přednáška využívá generátor `make_blobs`:
- $N = 500$ datových bodů
- 2 centra shluků (`centers=2`)
- Směrodatná odchylka shluků $\sigma = 2.0$ (`cluster_std=2.0`)
- Fixovaný random seed: `random_state=42`

```python
from sklearn.datasets import make_blobs

X, labels = make_blobs(
    n_samples=500, centers=2, random_state=42, cluster_std=2.0
)
```

### 2. Vytvoření instance třídy `LocalOutlierFactor`
Klíčové argumenty (slajd 16):
- `n_neighbors=20`: počet nejbližších sousedů pro výpočet hustoty.
- `metric="euclidean"`: vzdálenostní metrika (lze použít i Manhattan či Minkowski).
- `contamination=0.10`: předpokládaný podíl anomálií v datech ($10\,\% = 50$ bodů).

```python
from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
```

### 3. Trénování a predikce (`fit_predict`)
Metoda `fit_predict(X)` provede výpočet a vrátí pole celých čísel:
- `1`: normální pozorování (Inlier).
- `-1`: anomálie / odlehlé pozorování (Outlier).

```python
y_pred = lof.fit_predict(X)
```

### 4. Atribut `negative_outlier_factor_` (Slajd 18)
Scikit-learn ukládá skóre do atributu `negative_outlier_factor_`:
> **Pozor na terminologii:** Slovo *negative* znamená, že hodnoty jsou záporné. Čím je hodnota **menší (více do záporu, např. $-2.7$)**, tím větší je míra anomality! Skutečný teoretický LOF faktor odpovídá kladné hodnotě $-\text{negative\_outlier\_factor\_}$.

```python
lof_scores = lof.negative_outlier_factor_
```

### 5. Vizualizace s barevným odlišením (Slajdy 19–20)
Slajd 19 používá elegantní indexovací trik pro barvy: `marker_colors[(y_pred + 1) // 2]`. 
Pokud $y = -1 \implies (-1+1)//2 = 0 \to$ červená (`"r"`). Pokud $y = 1 \implies (1+1)//2 = 1 \to$ modrá (`"b"`).

```python
import matplotlib.pyplot as plt
import numpy as np

marker_colors = np.array(["r", "b"])
plt.figure(figsize=(9, 6))
plt.scatter(
    X[:, 0],
    X[:, 1],
    s=25,
    color=marker_colors[(y_pred + 1) // 2],
    edgecolors="k" if (y_pred == -1).any() else "none",
)
plt.title("Outliers (red) in dataset")
plt.xlabel("Příznak 1")
plt.ylabel("Příznak 2")
plt.tight_layout()
plt.show()
```

Výsledek: Přesně $50$ bodů ($10\,\%$) ležících na vnějších okrajích obou shluků je označeno červeně jako anomálie.

---

## 4. Výhody a nevýhody metody LOF (Slajdy 23–24)

### ✅ VÝHODY (Advantages):
1. **Schopnost detekovat lokální anomálie:** Na rozdíl od algoritmů zaměřených na globální extrémy dokáže LOF odhalit anomálie skryté v menších či řidších podsestavách dat.
2. **Vysoká flexibilita:** Metoda založená na hustotě spolehlivě funguje v datech libovolných geometrických tvarů (shluky nemusí být kulovité).
3. **Volba metrik vzdálenosti:** Podporuje libovolné metriky z modulu `scipy.spatial.distance` (Eukleidovská, Manhattanova, Čebyševova, Minkowského).

### ❌ NEVÝHODY (Disadvantages):
1. **Výpočetní složitost $\mathcal{O}(N^2)$:** Vyhledávání $k$ nejbližších sousedů pro všechny dvojice bodů je výpočetně náročné. U velkých datasetů ($N > 100\,000$) se efektivita prudce propadá.
2. **Obtížná interpretovatelnost:** Protože LOF je relativní poměr hustot, neexistuje žádná striktně daná univerzální mez (např. 1.2 vs. 1.5 závisí na dimenzi a šumu).
3. **Citlivost na hyperparametr `n_neighbors`:** Příliš malá hodnota vede k falešným poplachům, příliš velká hodnota smaže lokální charakter hustoty.
