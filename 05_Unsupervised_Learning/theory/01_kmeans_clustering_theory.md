# Teoretický rozbor: Neřízené učení – Algoritmus K-Means a Shlukování (Clustering)

Tento dokument slouží jako ucelený teoretický a metodický základ pro téma shlukování v rámci 5. dne kurzu Machine Learning (Session 3: Unsupervised Learning).

---

## 1. Úvod do shlukové analýzy (Cluster Analysis)

Shlukování je fundamentální disciplína **neřízeného strojového učení (Unsupervised Learning)**. Na rozdíl od regrese či klasifikace nemáme k dispozici cílový vektor $y$ (tzv. ground truth návěští). Algoritmus dostává na vstup pouze matici příznaků $X \in \mathbb{R}^{n \times d}$ a jeho úkolem je odhalit latentní (skryté) vzory a rozdělit datové body do $k$ přirozených homogenních skupin (shluků / clusters).

### Základní postuláty dobrého shlukování:
1. **Vysoká vnitroshluková kompaktnost (High Intra-cluster Similarity):** Objekty uvnitř stejného shluku si jsou maximálně podobné (malá vzájemná vzdálenost).
2. **Vysoká mezishluková separabilita (High Inter-cluster Dissimilarity):** Objekty z různých shluků jsou si maximálně odlišné (velká vzájemná vzdálenost).

---

## 2. Algoritmus K-Means (Lloydův algoritmus)

K-Means je nejpoužívanější rozkladový (partiční) shlukovací algoritmus. Rozděluje $n$ pozorování do $k$ disjunktních shluků $S = \{S_1, S_2, \dots, S_k\}$, přičemž každý shluk je reprezentován svým těžištěm – **centroidem** $\mu_j$.

### 2.1 Matematická optimalizační funkce:
Algoritmus minimalizuje **vnitroshlukový součet čtverců vzdáleností** (tzv. **WCSS – Within-Cluster Sum of Squares** neboli **Inertia**):

$$\mathcal{J}(S, \mu) = \sum_{j=1}^{k} \sum_{x_i \in S_j} \|x_i - \mu_j\|^2$$

Kde:
* $x_i$ je datový vektor pozorování.
* $\mu_j = \frac{1}{|S_j|} \sum_{x \in S_j} x$ je těžiště (centroid) shluku $S_j$.
* $\|\cdot\|$ představuje standardní Euklidovskou normu ($L_2$).

### 2.2 Iterační proces krok za krokem:
Optimalizace probíhá pomocí algoritmu **Expectation-Maximization (EM)**:
1. **Inicializace:** Zvolí se počet shluků $k$ a vybere se $k$ počátečních centroidů $\mu_1^{(0)}, \dots, \mu_k^{(0)}$.
2. **Krok přiřazení (Assignment step / Expectation):**  
   Každý bod $x_i$ je přiřazen k nejbližšímu centroidu dle Euklidovské vzdálenosti:
   $$S_j^{(t)} = \left\{ x_i : \|x_i - \mu_j^{(t)}\|^2 \le \|x_i - \mu_{l}^{(t)}\|^2 \quad \forall l = 1, \dots, k \right\}$$
3. **Krok aktualizace (Update step / Maximization):**  
   Pro každý shluk se přepočítá nová poloha těžiště jako aritmetický průměr všech přiřazených bodů:
   $$\mu_j^{(t+1)} = \frac{1}{|S_j^{(t)}|} \sum_{x_i \in S_j^{(t)}} x_i$$
4. **Konvergence:**  
   Kroky 2 a 3 se opakují, dokud se poloha centroidů nemění ($\|\mu^{(t+1)} - \mu^{(t)}\| < \varepsilon$), přiřazení bodů zůstává stabilní nebo je dosažen maximální počet iterací (`max_iter=300`).

---

## 3. Inicializace centroidů: K-Means++ vs Random

Standardní náhodný výběr počátečních centroidů má fatální nevýhodu: snadno uvízne v suboptimálním lokálním minimu, zejména pokud náhodou vybere dva počáteční body ve stejném přirozeném shluku.

### Vylepšení K-Means++ (Arthur & Vassilvitskii, 2007):
V Scikit-learn je výchozím nastavením `init='k-means++'`. Postupuje chytře:
1. První centroid $\mu_1$ je vybrán zcela náhodně z datových bodů.
2. Pro každý zbývající bod $x_i$ se spočte vzdálenost $D(x_i)$ k **nejbližšímu již existujícímu centroidu**.
3. Další centroid je vybrán náhodně, avšak s pravděpodobností úměrnou **čtverci vzdálenosti**:
   $$P(x_i) = \frac{D(x_i)^2}{\sum_{l=1}^n D(x_l)^2}$$
4. Krok 2 a 3 se opakují, dokud není vybráno všech $k$ centroidů.

> **Výsledek:** Počáteční centroidy jsou od sebe maximálně geometricky vzdálené. Výrazně se zrychluje konvergence a snižuje riziko uvíznutí v špatném lokálním extrému. Pro robustnost se v Scikit-learn navíc algoritmus spouští standardně 10× (`n_init=10`) a vybírá se běh s nejnižší inertií.

---

## 4. Volba optimálního počtu shluků $k$

Protože $k$ je hyperparametr zadaný uživatelem, potřebujeme objektivní matematická kritéria pro jeho určení.

### 4.1 Loketní metoda (Elbow Method – WCSS / Inertia):
* S rostoucím $k$ hodnota WCSS monotonně klesá (pro $k = n$ je $\text{WCSS} = 0$).
* Hledáme bod zlomu (tzv. „loket“ / elbow), od kterého už přidávání dalších shluků nepřináší podstatné snížení chyby, ale pouze zbytečně drobí data.

### 4.2 Siluetový koeficient (Silhouette Analysis):
Měří, jak dobře je každý datový bod přiřazen ke svému shluku ve srovnání se sousedními shluky.
Pro bod $i$ definujeme:
* $a(i)$: Průměrná vzdálenost bodu $i$ od všech ostatních bodů v **jeho vlastním shluku** (vnitřní soudržnost).
* $b(i)$: Průměrná vzdálenost bodu $i$ od bodů v **nejbližším sousedním shluku** (separace).

Siluetový koeficient bodu $i$ je dán vzorcem:
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}, \quad s(i) \in [-1, +1]$$

* **$s(i) \approx +1$:** Bod je hluboko uvnitř svého shluku a daleko od sousedů (vynikající přiřazení).
* **$s(i) \approx 0$:** Bod leží přímo na rozhodovací hranici mezi dvěma shluky.
* **$s(i) \approx -1$:** Bod je pravděpodobně přiřazen do špatného shluku (leží blíže k sousednímu shluku).

Celkové **Silhouette Score** datasetu je aritmetickým průměrem $\frac{1}{n} \sum_{i=1}^n s(i)$. Vybírá se $k$, které toto průměrné skóre maximalizuje.

### 4.3 Doplňkové metriky:
* **Davies-Bouldin Index (DBI):** Poměr rozptylu uvnitř shluku ku vzdálenosti mezi těžišti. **Nižší hodnota je lepší**.
* **Calinski-Harabasz Index (Variance Ratio Criterion):** Poměr mezishlukové variance ku vnitroshlukové varianci. **Vyšší hodnota je lepší**.

---

## 5. Kritické zhodnocení a limity K-Means

Při aplikaci v praxi naráží K-Means na zásadní teoretické předpoklady, které v reálných datech často neplatí:

1. **Předpoklad sférického tvaru (Izotropie):**  
   Protože K-Means používá Euklidovskou vzdálenost, implicitně předpokládá, že shluky jsou kulové (sférické) a mají přibližně stejnou velikost a varianci.  
   *Selhává na protáhlých elipsoidních, půlměsícových, spirálových nebo prstencových strukturách.*
2. **Předpoklad stejné hustoty a velikosti:**  
   Pokud má jeden shluk 10 000 bodů a druhý pouze 50 bodů, centroid většího shluku „vysaje“ a zkreslí hranici menšího shluku.
3. **Extrémní citlivost na škálování veličin:**  
   Veličina s velkým číselným rozsahem (např. příjem v USD v desetitisících) zcela přehluší veličinu v jednotkách (např. věk). **Před K-Means je standardizace (`StandardScaler`) povinná!**
4. **Citlivost na odlehlé hodnoty (Outliers):**  
   Centroid je počítán jako aritmetický průměr, který je silně náchylný k vychýlení extrémními hodnotami.

---

## 6. Moderní ML & AI kontext (Stav k 2026)

V moderní praxi se klasický K-Means doplňuje a kombinuje s pokročilými architekturami:

* **MiniBatchKMeans:** Pro datasety s miliony řádků (webové logy, clickstream). Počítá gradienty na náhodných mini-dávkách (SGD princip), což dramaticky zrychluje konvergenci při zachování 98%+ kvality.
* **K-Medoids (PAM – Partitioning Around Medoids):** Centroidem není virtuální průměrný bod, nýbrž reálný zástupce z databáze (medoid) a minimalizuje se manhattanská vzdálenost ($L_1$). Odolný vůči outlierům.
* **Hustotní shlukování (DBSCAN / HDBSCAN):** Nevyžaduje zadání počtu shluků $k$, dokáže detekovat libovolné tvary a automaticky filtruje šum/anomálie.
* **Spherical K-Means na textové embeddingy:** Shlukování s využitím kosinové podobnosti na výstupech z Transformerů (OpenAI text-embedding-3, BERT).
* **XAI pro shlukování (Explainable Clustering):** Použití stromových zástupných modelů (Surrogate Decision Trees) nebo SHAP analýzy k vysvětlení, proč byl daný zákazník zařazen do konkrétního segmentu.
