# Implementační průvodce: Vzorová implementace K-Means v Scikit-learn

Tento dokument slouží jako detailní implementační průvodce na základě přednáškových materiálů kurzu **`K-Means - an implementation example` (Resources/Day 5 PDF)**. Shrnuje kompletní postup od vygenerování dat, přes výpočet diagnostických křivek (WCSS a Silhouette Score), až po natrénování optimálního modelu a vizualizaci těžišť (centroidů).

---

## 1. Cíl vzorové úlohy

Přednáška demonstruje praktické použití algoritmu **K-Means** na syntetickém shlukovém datasetu vytvořeném pomocí `sklearn.datasets.make_blobs`:
1. Vytvoření 300 (resp. 500) pozorování rozdělených do 4 přirozených sférických shluků v 2D rovině.
2. Vizualizace neoznačených dat bez referenčních návěští.
3. Testování hypotéz pro počet shluků $k \in \{1, 2, \dots, 10\}$.
4. Výpočet vnitroshlukového součtu čtverců vzdáleností (**WCSS – Within-Cluster Sum of Squares**, atribut `.inertia_`).
5. Výpočet průměrného **Silhouette Score** pro $k \ge 2$.
6. Nalezení inflexního bodu loketní metodou (*Elbow Method*) a potvrzení maximem siluetového skóre.
7. Natrénování finálního modelu pro $k = 4$, extrakce souřadnic centroidů (`.cluster_centers_`) a přiřazení návěští (`.labels_`).
8. Zobrazení výsledného rozdělení bodů do barevných shluků s vyznačením těžišť červenými křížky.

---

## 2. Krok za krokem v Pythonu

### Krok 1: Generování a vizualizace neanotovaných dat
```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Vygenerování syntetického datasetu se 4 shluky
data, true_labels = make_blobs(
    n_samples=300,
    centers=4,
    cluster_std=1.0,
    random_state=42
)

# Vykreslení bodů před shlukováním
plt.scatter(data[:, 0], data[:, 1], s=20, color="red")
plt.title("Our dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
```
*Poznámka:* Před vstupem do K-Means model nemá k dispozici `true_labels`. Data vnímá čistě jako množinu souřadnic $[x_1, x_2]$.

---

### Krok 2: Výpočet diagnostických metrik pro $k \in [1, 10]$
V Scikit-learn se pro evaluaci využívají:
* Atribut `kmeans.inertia_`: součet čtverců vzdáleností bodů od jejich příslušného centroidu:
  $$\text{WCSS} = \sum_{j=1}^{k} \sum_{x_i \in S_j} \|x_i - \mu_j\|^2$$
* Funkce `sklearn.metrics.silhouette_score`: měří poměr soudržnosti uvnitř shluku $a(i)$ a vzdálenosti k nejbližšímu sousednímu shluku $b(i)$:
  $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}, \quad s(i) \in [-1, +1]$$

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

inertia_scores = []
silhouette_scores = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)
    inertia_scores.append(kmeans.inertia_)
    if k > 1:
        silhouette_scores.append(silhouette_score(data, kmeans.labels_))
```

---

### Krok 3: Vyhodnocení křivek (Loket & Silueta)

1. **Křivka WCSS (Inertia vs. $k$):**  
   Při přechodu z $k=1$ na $k=4$ dochází k dramatickému poklesu chyby (např. z ~5 000 na ~500). Od $k=4$ dále se křivka oplošťuje a tvoří zřetelný **„loket“ (elbow)**. Další zvyšování $k$ již přináší pouze zanedbatelné zmenšení inertie na úkor nežádoucí fragmentace dat.
2. **Křivka Silhouette Score:**  
   Siluetový koeficient dosahuje pro $k=4$ ostrého globálního maxima ($\approx 0.79$). To jednoznačně potvrzuje, že 4 shluky představují optimální geometrickou konfiguraci.

---

### Krok 4: Natrénování finálního modelu a extrakce parametrů
```python
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(data)

cluster_centers = kmeans.cluster_centers_  # Souřadnice centroidů shape (4, 2)
labels = kmeans.labels_                    # Vektor příslušnosti délky 300
```

---

### Krok 5: Finální vizualizace s centroidy
```python
# Vykreslení bodů obarvených dle nalezených shluků
plt.scatter(data[:, 0], data[:, 1], c=labels, s=50, cmap="viridis")

# Vyznačení centroidů červenými křížky
plt.scatter(
    cluster_centers[:, 0],
    cluster_centers[:, 1],
    c="red",
    marker="x",
    s=200,
    linewidths=3,
    label="Centroids"
)
plt.title(f"Clustering result for k = {optimal_k}")
plt.legend()
plt.show()
```

---

## 3. Parametry třídy `sklearn.cluster.KMeans`

| Parametr | Výchozí hodnota | Popis a doporučení pro praxi |
| :--- | :--- | :--- |
| `n_clusters` | `8` | Počet hledaných shluků ($k$). Vždy nutno určit na základě doménové znalosti, lokte nebo siluety. |
| `init` | `'k-means++'` | Inteligentní inicializace centroidů (Arthur & Vassilvitskii, 2007) – výrazně urychluje konvergenci a brání lokálním extrémům. |
| `n_init` | `'auto'` (10) | Počet nezávislých běhů s různou inicializací. Vybírá se běh s nejnižší inertií. |
| `max_iter` | `300` | Maximální počet EM iterací na jeden běh algoritmu. |
| `tol` | `1e-4` | Prahová relativní tolerance změny polohy těžišť pro ukončení běhu (konvergence). |
| `random_state` | `None` | Pevný seed pro reprodukovatelnost výsledků (doporučeno nastavit např. `42`). |

---

## 4. Výstupy a atributy natrénovaného objektu `KMeans`

* `cluster_centers_`: Matice souřadnic těžišť o rozměrech $(k, d)$.
* `labels_`: Pole celých čísel délky $n$ s indexy shluků ($0, 1, \dots, k-1$).
* `inertia_`: Skalární hodnota minimalizovaného kritéria WCSS.
* `n_iter_`: Skutečný počet iterací, po kterých algoritmus zkonvergoval (obvykle 5–15 iterací).

---

## 5. Expertní komentář a úskalí (Guardrails)

1. **Subjektivita loketní metody:**  
   Jak uvádí slide 19 přednášky, v reálných neideálních datech nemusí být „loket“ zcela ostrý a různí analytici mohou zvolit jiný inflexní bod. Proto je **nezbytné kombinovat WCSS vždy se siluetovou analýzou**, případně Davies-Bouldinovým indexem.
2. **Nutnost škálování u reálných dat:**  
   V této ukázce mají obě osy náhodně podobný rozptyl ($\text{std} = 1.0$), takže neškálování nevedlo k chybě. U reálných tabulkových dat (např. plat v USD vs. věk) je **použití `StandardScaler` striktní nutností**!
3. **Předpoklad kulového tvaru:**  
   Data generovaná přes `make_blobs` jsou ze své definice sférická (izotropní gaussovská). K-Means zde proto dosahuje bezchybného rozdělení. Na protáhlých elipsoidech nebo křivkách by selhal.
