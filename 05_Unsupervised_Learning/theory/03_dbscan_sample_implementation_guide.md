# Implementační průvodce: Vzorová implementace DBSCAN v Scikit-learn

Tento dokument slouží jako detailní implementační průvodce na základě přednáškových materiálů kurzu **`DBSCAN - an implementation example` (Resources/Day 5 PDF)**. Shrnuje postup od vygenerování nekonvexních soustředných kružnic, přes nezbytnou standardizaci, až po ladění hyperparametrů `eps` a `min_samples`, detekci šumu a objektivní volbu parametrů pomocí k-distance grafu.

---

## 1. Cíl vzorové úlohy a selhání K-Means

Přednáška demonstruje fundamentální výhodu hustotního shlukování oproti algoritmům založeným na těžištích (K-Means):
1. **Limity K-Means:** Algoritmus K-Means rozděluje prostor lineárními nadrovinami (Voroného buňky) mezi centroidy. Na soustředných kružnicích (kde vnitřní kruh leží uvnitř vnějšího) K-Means **nevyhnutelně selže** – oba kruhy rozetne napůl.
2. **Přístup DBSCAN:** Algoritmus vyhledává oblasti s vysokou hustotou bodů a propojuje sousedící hustá jádra. Geometrický tvar shluku může být libovolný (kružnice, půlměsíce, spirály).

```
        K-Means (Selhání):                  DBSCAN (Správné rozdělení):
       ╭─────────┬─────────╮                   ╭─────────────────╮
       │ Shluk 1 │ Shluk 2 │                   │  Shluk 1 (Vnější)│
       │  (Levá  │  (Pravá │                   │   ╭─────────╮   │
       │  půlka) │  půlka) │                   │   │ Shluk 2 │   │
       │         │         │                   │   ╰─────────╯   │
       ╰─────────┴─────────╯                   ╰─────────────────╯
```

---

## 2. Dva klíčové hyperparametry DBSCAN

DBSCAN **nevyžaduje zadání počtu shluků $k$**. Místo toho operuje se dvěma geometrickými parametry:

1. **`eps` ($\varepsilon$ – Epsilon):**  
   Maximální poloměr sousedství okolo zkoumaného bodu.  
   * **Příliš malé $\varepsilon$:** Husté oblasti se rozpadnou a většina bodů bude označena jako šum (`-1`).  
   * **Příliš velké $\varepsilon$:** Sousedství pohltí prázdný prostor mezi kruhy a vše se slije do jednoho masivního shluku.
2. **`min_samples` (Minimální počet bodů v sousedství):**  
   Určuje, kolik bodů (včetně zkoumaného) se musí nacházet v okruhu $\varepsilon$, aby byl bod prohlášen za **jádrový bod (Core point)**.  
   * *Pravidlo pro volbu:* Doporučuje se začít s hodnotou $\text{min\_samples} = D + 1$ nebo $2 \times D$, kde $D$ je počet dimenzí (pro 2D data tedy 4 až 8 bodů). Pro zašuměná data se volí vyšší hodnota (10 až 15).

---

## 3. Typologie bodů v DBSCAN

Každé pozorování v datasetu je algoritmem zařazeno do jedné ze tří kategorií:

* **1. Jádrový bod (Core point):**  
  Bod, který má ve svém $\varepsilon$-okolí alespoň `min_samples` sousedů. Tvoří kostru shluku.
* **2. Hraniční bod (Border point):**  
  Bod, který sám nemá dostatek sousedů ($< \text{min\_samples}$), ale leží v $\varepsilon$-okolí nějakého jádrového bodu. Je přiřazen ke shluku tohoto jádra.
* **3. Šum / Odlehlý bod (Noise point / Outlier):**  
  Bod, který není jádrový ani hraniční. DBSCAN ho označí návěštím **`-1`** a **nepřiřadí ho do žádného shluku**.

---

## 4. Kód krok za krokem dle přednášky

### Krok 1: Generování nekonvexních soustředných kružnic
```python
from sklearn.datasets import make_circles
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Generování dat se dvěma soustřednými kružnicemi
data, labels_true = make_circles(
    n_samples=300,
    factor=0.5,    # Poměr vnitřního a vnějšího poloměru
    noise=0.1,     # Gaussovský šum rozptylu bodů
    random_state=42
)

# 2. Povinná standardizace (algoritmy založené na vzdálenostech vyžadují shodné měřítko)
data_scaled = StandardScaler().fit_transform(data)

# 3. Vykreslení neanotovaných vstupních dat
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], color='green', marker='o', s=30)
plt.title("Our dataset (make_circles)")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
```

---

### Krok 2: Fitování DBSCAN a vliv hyperparametrů

Přednáška ukazuje dramatickou citlivost výsledku na volbu parametrů:

#### Varianta ze slajdu 13: `eps=0.4, min_samples=15`
```python
from sklearn.cluster import DBSCAN

dbscan_1 = DBSCAN(eps=0.4, min_samples=15)
dbscan_1.fit(data_scaled)
labels_1 = dbscan_1.labels_

# Detekuje 5 shluků + outliery (-1)
print(f"Unikátní návěští: {set(labels_1)}")
```

#### Varianta ze slajdu 15: `eps=0.25, min_samples=10`
```python
dbscan_2 = DBSCAN(eps=0.25, min_samples=10)
dbscan_2.fit(data_scaled)
labels_2 = dbscan_2.labels_

# Detekuje 2 shluky (vnitřní kruh a segmentovaný vnější kruh) s body šumu
```

#### Optimální konfigurace pro dokonalou separaci: `eps=0.30, min_samples=8`
Při `eps=0.30` a `min_samples=8` algoritmus bezchybně oddělí vnitřní kruh od vnějšího a odfiltruje izolované odlehlé body do třídy `-1`.

---

## 5. Jak objektivně určit `eps`? (k-distance graph)

Jak uvádí slajd 16, volba hyperparametrů nemusí probíhat metodou pokus-omyl. Standardním matematickým nástrojem je **graf vzdáleností k k-tému nejbližšímu sousedovi (k-NN distance graph)**:

```python
from sklearn.neighbors import NearestNeighbors
import numpy as np

# 1. Spočítáme vzdálenost ke k-tému nejbližšímu sousedovi (např. k = min_samples = 5)
k = 5
knn = NearestNeighbors(n_neighbors=k)
knn.fit(data_scaled)
distances, _ = knn.kneighbors(data_scaled)

# 2. Seřadíme vzdálenosti k poslednímu (k-tému) sousedovi vzestupně
sorted_k_distances = np.sort(distances[:, -1])

# 3. Vykreslíme křivku a hledáme bod zlomu (loket)
plt.plot(sorted_k_distances)
plt.title("k-NN Distance Graph pro volbu Epsilonu")
plt.xlabel("Body seřazené dle vzdálenosti")
plt.ylabel(f"{k}-NN Vzdálenost (Eps)")
plt.axhline(y=0.30, color='r', linestyle='--', label='Optimální Eps ~ 0.30')
plt.legend()
plt.show()
```
*Interpretace:* Bod zlomu, kde křivka prudce stoupá, představuje hranici mezi hustým shlukem a řídkým šumem. Tato hodnota na ose $Y$ je optimální volbou pro `eps`.

---

## 6. Limity DBSCAN a přechod na HDBSCAN (2026)

1. **Problém s proměnlivou hustotou shluků:**  
   Protože `eps` je globální konstanta, DBSCAN nedokáže v jednom běhu současně detekovat velmi hustý shluk i řidší shluk.
2. **Kletba dimenzionality (Curse of Dimensionality):**  
   Ve vysokých dimenzích ($d > 20$) jsou vzdálenosti mezi všemi body téměř uniformní (hustota prudce klesá), což činí volbu fixního `eps` prakticky nemožnou.
3. **Moderní řešení:**  
   V moderní praxi se používá **HDBSCAN** (`sklearn.cluster.HDBSCAN`), který integruje hierarchické shlukování s hustotním principem a eliminuje nutnost fixního parametru `eps`.
