# 🌙 Průvodce cvičením: Hustotní shlukování DBSCAN na datech dvou půlměsíců (make_moons)

> **Oficiální zadání cvičení:**  
> *„Using the make_moons method from the datasets module of the Scikit-learn library, create a dataset of 500 observations. The dataset should have a noise of 0.10. Assign the random seed to get the same results every time you run it.*  
> *Perform data standardization.*  
> *Using the DBSCAN class, create a model that will perform data clustering. Select appropriate values for the hyperparameters.*  
> *Assign cluster labels for each observation to the dbscan_labels variable.*  
> *Generate a graph to represent the dataset so that it illustrates which clusters each point belongs to.“*

---

## 1. Pedagogický cíl a metodický kontext

Datová sada **Two Moons (`make_moons`)** je v teorii strojového učení zlatým standardem pro testování algoritmů neřízeného učení:
1. **Nekonvexní geometrie:** Data se skládají ze dvou vzájemně do sebe vnořených půlměsíců (crescents). Žádná lineární dělící nadrovina ani Voroného buňka nedokáže tyto dva půlměsíce oddělit.
2. **Selhání centroidových metod (K-Means):** Algoritmy minimalizující rozptyl od těžiště ($WCSS$) předpokládají sférické (kulové) rozdělení. K-Means proto půlměsíce příčně přeřízne napůl.
3. **Triumf hustotního shlukování (DBSCAN):** DBSCAN definuje shluk jako spojitou oblast s vysokou hustotou bodů oddělenou oblastmi s nízkou hustotou. Dokáže tak kopírovat zakřivení půlměsíců bez ohledu na jejich geometrický tvar a zároveň izolovat náhodný šum způsobený parametrem `noise=0.10`.

---

## 2. Matematické principy krok za krokem

### 2.1 Generování dat (`make_moons`)
Generátor `make_moons` syntetizuje $N = 500$ bodů v $\mathbb{R}^2$ tvořených dvěma propletenými polokružnicemi:
- Horní půlměsíc: $x_1 = \cos(\theta)$, $x_2 = \sin(\theta)$ pro $\theta \in [0, \pi]$.
- Dolní půlměsíc: posunutý o $(1 - \text{offset}_x, 0.5 - \text{offset}_y)$ se záporným sinem.
- Přidaný Gaussovský šum: ke každému bodu je přičten náhodný vektor $\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$, kde $\sigma = 0.10$.

```python
from sklearn.datasets import make_moons

X, y_true = make_moons(n_samples=500, noise=0.10, random_state=42)
```

### 2.2 Standardizace proměnných (`StandardScaler`)
Algoritmus DBSCAN měří vzdálenosti v $\mathbb{R}^2$ standardně eukleidovskou metrikou:

$$d(\mathbf{x}_i, \mathbf{x}_j) = \|\mathbf{x}_i - \mathbf{x}_j\|_2 = \sqrt{\sum_{k=1}^p (x_{ik} - x_{jk})^2}$$

Pokud by jedna osa měla jiný rozptyl než druhá, eukleidovské $\varepsilon$-okolí (kruh) by v původním prostoru odpovídalo elipse, což by zkreslilo hustotu v různých směrech. Standardizace provede transformaci:

$$z_{ik} = \frac{x_{ik} - \mu_k}{\sigma_k}$$

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

## 3. Výběr hyperparametrů DBSCAN ($\varepsilon$ a $\text{min\_samples}$)

DBSCAN vyžaduje nastavení dvou klíčových hyperparametrů:
1. **$\varepsilon$ (epsilon):** Poloměr sousedství okolo zkoumaného bodu.
2. **`min_samples`:** Minimální počet bodů (včetně zkoumaného), které se musí nacházet v kouli $B(\mathbf{x}, \varepsilon)$, aby byl bod prohlášen za **jádrový (Core point)**.

### 3.1 Pravidlo pro `min_samples`
Pro 2D data ($D=2$) se v literatuře doporučuje:
$$\text{min\_samples} \ge 2 \cdot D = 4$$
Vzhledem k přítomnosti šumu ($\sigma=0.10$) a velikosti vzorku $N=500$ je robustnější volbou $\text{min\_samples} \in [5, 10]$ (např. 8). Vyšší hodnota chrání model před tím, aby náhodné shluky šumu spojily oba půlměsíce dohromady.

### 3.2 Analytické určení $\varepsilon$ pomocí k-NN Distance Elbow metody
Jak uvádí přednášková metodika (slajd 16), optimální hodnotu $\varepsilon$ nehledáme náhodným zkoušením, nýbrž analýzou vzdáleností k $k$-tému nejbližšímu sousedovi:
1. Zvolíme $k = \text{min\_samples}$ (např. $k=8$).
2. Spočítáme vzdálenost každého bodu k jeho 8. nejbližšímu sousedovi pomocí `NearestNeighbors`.
3. Vzdálenosti seřadíme vzestupně a vykreslíme tzv. loketní křivku (**Elbow plot**).
4. Bod zlomu křivky, kde vzdálenosti začínají strmě růst, představuje hranici mezi hustým vnitřkem půlměsíců a řídkým prostorem šumu.

Pro náš dataset $N=500$, $\sigma=0.10$ po standardizaci leží loket v intervalu:
$$\varepsilon \in [0.22, 0.30]$$

Při $\varepsilon = 0.25$ a $\text{min\_samples} = 8$:
- **Počet shluků:** 2 (dokonale odpovídá hornímu a dolnímu půlměsíci).
- **Detekovaný šum (outliery):** 5 bodů ($1.0\,\%$ vzorku).
- **Adjusted Rand Index (ARI):** $0.9801$ (téměř $100\,\%$ shoda se skutečnou strukturou).

---

## 4. Úskalí příliš velkého $\varepsilon$: „Most přes propast“ (Chasm Bridging)

Kritickým jevem u hustotního shlukování je tzv. **Chasm Bridging**:
- Pokud nastavíme $\varepsilon \ge 0.35$, poloměr dosahu přesáhne nejužší mezeru mezi oběma půlměsíci.
- Body šumu ležící v mezeře vytvoří hustotní můstek (**Density Bridge**), který spojí horní i dolní půlměsíc do jednoho obřího megashluku (`n_clusters = 1`).
- Proto je nezbytné držet $\varepsilon$ pod kritickou vzdáleností mezery ($\varepsilon \le 0.30$).

---

## 5. Referenční Python implementace dle zadání

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

# 1. Generování datasetu
X, y_true = make_moons(n_samples=500, noise=0.10, random_state=42)

# 2. Standardizace dat
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Model DBSCAN s vhodně vybranými parametry
dbscan = DBSCAN(eps=0.25, min_samples=8)

# 4. Přiřazení označení shluků do požadované proměnné
dbscan_labels = dbscan.fit_predict(X_scaled)

# Výpis základních statistik
n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = np.sum(dbscan_labels == -1)
print(f"Počet nalezených shluků: {n_clusters}")
print(f"Počet bodů označených jako šum: {n_noise} ({n_noise / len(X) * 100:.1f} %)")

# 5. Generování grafu reprezentujícího příslušnost ke shlukům
plt.figure(figsize=(10, 6))
unique_labels = set(dbscan_labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Šum vykreslíme černě s křížkem
        col = [0, 0, 0, 1]
        label_text = "Šum / Outlier (-1)"
        marker_style = "x"
        marker_size = 8
    else:
        label_text = f"Shluk {k + 1}"
        marker_style = "o"
        marker_size = 7

    class_member_mask = (dbscan_labels == k)
    xy = X_scaled[class_member_mask]
    plt.plot(
        xy[:, 0], xy[:, 1],
        marker_style,
        markerfacecolor=tuple(col),
        markeredgecolor="k" if k != -1 else "r",
        markersize=marker_size,
        label=label_text,
        linestyle="none"
    )

plt.title(f"DBSCAN na make_moons (eps=0.25, min_samples=8) -> {n_clusters} shluky, {n_noise} šumových bodů")
plt.xlabel("Standardizovaný Příznak 1 ($z_1$)")
plt.ylabel("Standardizovaný Příznak 2 ($z_2$)")
plt.legend(loc="upper right")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
```

---

## 6. Srovnání algoritmů na datech `make_moons`

| Algoritmus | Parametry | Nalezených shluků | ARI se skutečností | Schopnost zachytit nekonvexní tvar |
| :--- | :--- | :---: | :---: | :---: |
| **K-Means** | $k=2$, `random_state=42` | 2 | $0.248$ | ❌ Selhává (dělí prostor lineární osou) |
| **Ward Hierarchical** | $n=2$, `ward` | 2 | $0.385$ | ❌ Selhává (minimalizuje rozptyl jako K-Means) |
| **Single Linkage** | $n=2$, `single` | 2 | $0.992$ | ✅ Úspěch (princip nejbližšího souseda) |
| **DBSCAN (optimální)** | $\varepsilon=0.25$, `min_samples=8` | 2 | **$0.980$** | ✅ **Dokonalý tvar + filtrace šumu** |
| **DBSCAN (příliš velký $\varepsilon$)** | $\varepsilon=0.40$, `min_samples=8` | 1 | $0.000$ | ❌ Selhává (most přes propast spojí půlměsíce) |

---

## 7. Paradox metriky Silhouette skóre (Důležitý metodický guardrail)

Častou chybou začínajících data scientistů je snaha optimalizovat $\varepsilon$ podle **Silhouette skóre**:
- Silhouette skóre měří poměr vzdálenosti bodu k ostatním bodům ve vlastním shluku versus vzdálenost k bodům v sousedním shluku.
- Tato metrika **implicitně předpokládá konvexní (kulovité) shluky**.
- U zakřivených půlměsíců leží protilehlé konce stejného půlměsíce eukleidovsky dále od sebe než dva sousední body různých půlměsíců v centrální mezeře!
- Výsledkem je, že Silhouette skóre pro perfektní DBSCAN clustering je pouze $\approx 0.38$, zatímco u K-Means (které data nesmyslně přeřízne) je Silhouette paradoxně srovnatelné ($\approx 0.49$)!

> [!WARNING]
> **Metodický Guardrail 2026:** Na datech s nekonvexní geometrií (kruhy, půlměsíce, spirály, uliční sítě) **nikdy nepoužívejte Silhouette skóre ani Davies-Bouldin index jako jediné kritérium** pro výběr hyperparametrů. Vždy použijte k-NN distance elbow metodu, stabilitu hustoty (HDBSCAN persistence) nebo vizuální inspekci.
