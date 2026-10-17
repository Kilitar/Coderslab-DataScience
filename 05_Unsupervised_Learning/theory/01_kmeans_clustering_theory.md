# Teoretický rozbor: Neřízené učení – Shluková analýza (Clustering), K-Means & DBSCAN

Tento dokument slouží jako ucelený teoretický a metodický základ pro téma shlukování (Clustering) v rámci 5. dne kurzu Machine Learning (Session 3: Unsupervised Learning). Vychází ze zdrojové přednášky **`Clustering - an introduction` (Resources/Day 5 PDF)** a rozšiřuje ji o rigorózní matematické formulace, algoritmy K-Means a DBSCAN, diagnostické metody a moderní standardy pro rok 2026.

---

## 1. Úvod do shlukové analýzy (Cluster Analysis)

Shlukování je fundamentální disciplína **neřízeného strojového učení (Unsupervised Learning)**. Na rozdíl od úloh s učitelem (regrese či klasifikace) nemáme k dispozici cílový vektor $y$ (tzv. ground truth návěští). Algoritmus dostává na vstup pouze matici příznaků $X \in \mathbb{R}^{n \times d}$ a jeho úkolem je odhalit latentní (skryté) geometrické a pravděpodobnostní struktury a rozdělit datové body do přirozených homogenních skupin (shluků / clusters).

```
Neanotovaná data (Unlabeled data X)
               │
               ▼
   [ Shlukovací algoritmus ]
               │
               ▼
Odhalené přirozené skupiny (Clusters S_1, S_2, ..., S_k)
```

### Základní postuláty dobrého shlukování:
1. **Vysoká vnitroshluková kompaktnost (High Intra-cluster Similarity):** Objekty uvnitř stejného shluku si jsou maximálně podobné (malá vzájemná vzdálenost či vysoká hustota).
2. **Vysoká mezishluková separabilita (High Inter-cluster Dissimilarity):** Objekty z různých shluků jsou si maximálně odlišné (velká vzájemná vzdálenost či oddělení řídkým prostorem).

---

## 2. Dva základní typy shlukování (Types of Clustering)

Dle způsobu přiřazování pozorování do skupin rozlišujeme dva fundamentální přístupy:

### 2.1 Deterministické (Hard / Crisp) shlukování
* **Princip:** Každé pozorování $x_i$ je jednoznačně přiřazeno do **právě jednoho** disjunktního shluku.
* **Míra příslušnosti:** Funkce příslušnosti $\gamma_{ij} \in \{0, 1\}$, přičemž platí $\sum_{j=1}^k \gamma_{ij} = 1$.
* **Příklad z e-shopu:** Máme 5 000 zákazníků měsíčně a rozdělíme je do 5 segmentů. Každý zákazník je zařazen striktně do jedné skupiny (např. *Běžný zákazník* nebo *VIP zákazník*).
* **Zástupci:** K-Means, DBSCAN, K-Medoids, Wardovo hierarchické shlukování.

### 2.2 Fuzzy (Soft / Pravděpodobnostní) shlukování
* **Princip:** Každé pozorování může náležet do **více shluků současně** s určitou mírou příslušnosti či pravděpodobností.
* **Míra příslušnosti:** Funkce příslušnosti $w_{ij} \in [0, 1]$, přičemž součet měr přes všechny shluky je roven 1 ($\sum_{j=1}^k w_{ij} = 1$).
* **Příklad z praxe:** Zákazník ležící na pomezí dvou nákupních profilů může patřit z 60 % do skupiny *Technologičtí nadšenci* a ze 40 % do skupiny *Lovci slev*.
* **Zástupci:** Fuzzy C-Means (FCM), GMM (Gaussian Mixture Models s algoritmem Expectation-Maximization).

---

## 3. Praktické aplikace shlukování v reálném světě

Shlukování nachází uplatnění v široké škále komerčních, vědeckých a průmyslových odvětví:

1. **Analýza trhu a segmentace zákazníků (Market Analysis & Customer Segmentation):**
   * Seskupování zákazníků dle nákupního chování, frekvence nákupů, košíkové afinity a demografie (např. RFM analýza).
   * Umožňuje cílenou marketingovou komunikaci, personalizované slevy a optimalizaci produktového portfolia.
2. **Analýza medicínských obrazů (Medical Image Analysis):**
   * Automatická segmentace tkání a patologických ložisek na snímcích z magnetické rezonance (MRI), výpočetní tomografie (CT) a rentgenu (RTG).
   * Shlukování pixelů či voxelů s podobnou intenzitou a texturou pomáhá detekovat nádory a léze.
3. **Analýza genetických sekvencí a bioinformatika (Genetics & Bioinformatics):**
   * Identifikace funkčně příbuzných genů s podobnými expresními profily v mikročipových analýzách.
   * Fylogenetické shlukování pro objevování nových biologických druhů a studium evolučních vazeb.
4. **Analýza sociálních sítí (Social Media Analysis):**
   * Detekce vlivových komunit (Community Detection), sledování dynamiky šíření informací či dezinformací a identifikace klíčových opinion leaderů (influencerů).
5. **Textová analýza a tématické modelování (Text Analysis & Topic Modeling):**
   * Třídění velkých korpusů neanotovaných dokumentů, novinových článků, právních smluv či zákaznických recenzí do tematických shluků (např. pomocí TF-IDF nebo Transformer embeddingů).
6. **Sportovní a hráčská analytika (Player Analysis):**
   * Analýza herních statistik sportovců (rychlost, úspěšnost přihrávek, naběhané kilometry, střelecké pozice) pro tvorbu vyvážených týmů a optimalizaci tréninkových skupin dle výkonnostních profilů.

---

## 4. Taxonomie shlukovacích algoritmů dle pojetí podobnosti

Algoritmy neřízeného shlukování dělíme do 4 hlavních architektonických rodin:

| Rodina algoritmů | Princip stanovení podobnosti | Typičtí zástupci | Tvary shluků | Citlivost na šum |
| :--- | :--- | :--- | :--- | :--- |
| **Centroid-based (Těžišťové)** | Vzdálenost bodů od centrálního těžiště (centroidu) | K-Means, K-Medoids (PAM), MiniBatchKMeans | Výhradně kulové (sférické / izotropní) | Vysoká (outliery posunují těžiště) |
| **Distribution-based (Distribuční)** | Pravděpodobnost příslušnosti k danému rozdělení | Gaussian Mixture Models (GMM) | Elipsoidní dle kovarianční matice | Střední (modeluje pravděpodobnost šumu) |
| **Density-based (Hustotní)** | Oblasti s vysokou koncentrací bodů oddělené prázdným prostorem | DBSCAN, HDBSCAN, OPTICS | Libovolné nekonvexní tvary (měsíce, prstence) | Nízká (robustní, explicitně filtruje šum) |
| **Hierarchical (Hierarchické)** | Vzájemné párové vzdálenosti bodů/shluků v dendrogramu | Agglomerative (Ward, Complete, Single), Divisive | Různé dle vazebného kritéria (linkage) | Různá dle linkage kritéria |

---

## 5. Algoritmus K-Means (Lloydův algoritmus)

K-Means je nejpoužívanější rozkladový (partiční) shlukovací algoritmus. Rozděluje $n$ pozorování do $k$ disjunktních shluků $S = \{S_1, S_2, \dots, S_k\}$, přičemž každý shluk je reprezentován svým těžištěm – **centroidem** $\mu_j$.

### 5.1 Matematická optimalizační funkce:
Algoritmus minimalizuje **vnitroshlukový součet čtverců vzdáleností** (tzv. **WCSS – Within-Cluster Sum of Squares** neboli **Inertia**):

$$\mathcal{J}(S, \mu) = \sum_{j=1}^{k} \sum_{x_i \in S_j} \|x_i - \mu_j\|^2$$

Kde:
* $x_i$ je datový vektor pozorování ($i = 1, \dots, n$).
* $\mu_j = \frac{1}{|S_j|} \sum_{x \in S_j} x$ je těžiště (centroid) shluku $S_j$.
* $\|\cdot\|$ představuje standardní Euklidovskou normu ($L_2$).

### 5.2 Iterační proces krok za krokem:
Optimalizace probíhá pomocí algoritmu **Expectation-Maximization (EM)**:
1. **Inicializace:** Zvolí se požadovaný počet shluků $k$ a vybere se $k$ počátečních centroidů $\mu_1^{(0)}, \dots, \mu_k^{(0)}$.
2. **Krok přiřazení (Assignment step / Expectation):**  
   Každý bod $x_i$ je přiřazen k nejbližšímu centroidu dle Euklidovské vzdálenosti:
   $$S_j^{(t)} = \left\{ x_i : \|x_i - \mu_j^{(t)}\|^2 \le \|x_i - \mu_{l}^{(t)}\|^2 \quad \forall l = 1, \dots, k \right\}$$
3. **Krok aktualizace (Update step / Maximization):**  
   Pro každý shluk se přepočítá nová poloha těžiště jako aritmetický průměr všech přiřazených bodů:
   $$\mu_j^{(t+1)} = \frac{1}{|S_j^{(t)}|} \sum_{x_i \in S_j^{(t)}} x_i$$
4. **Konvergence:**  
   Kroky 2 a 3 se opakují, dokud se poloha centroidů nemění ($\|\mu^{(t+1)} - \mu^{(t)}\| < \varepsilon$), přiřazení bodů zůstává stabilní nebo je dosažen maximální počet iterací (`max_iter=300`).

---

## 6. Inicializace centroidů: K-Means++ vs Random

Standardní náhodný výběr počátečních centroidů má fatální nevýhodu: snadno uvízne v suboptimálním lokálním minimu, zejména pokud náhodou vybere dva počáteční body ve stejném přirozeném shluku.

### Vylepšení K-Means++ (Arthur & Vassilvitskii, 2007):
V moderním Scikit-learn je výchozím nastavením `init='k-means++'`. Postupuje chytře:
1. První centroid $\mu_1$ je vybrán zcela náhodně z datových bodů.
2. Pro každý zbývající bod $x_i$ se spočte vzdálenost $D(x_i)$ k **nejbližšímu již existujícímu centroidu**.
3. Další centroid je vybrán náhodně, avšak s pravděpodobností úměrnou **čtverci vzdálenosti**:
   $$P(x_i) = \frac{D(x_i)^2}{\sum_{l=1}^n D(x_l)^2}$$
4. Krok 2 a 3 se opakují, dokud není vybráno všech $k$ centroidů.

> **Výsledek:** Počáteční centroidy jsou od sebe maximálně geometricky rozptýlené. Výrazně se zrychluje konvergence a snižuje riziko uvíznutí ve špatném lokálním extrému. Pro robustnost se v Scikit-learn navíc algoritmus spouští standardně 10× (`n_init=10`) a vybírá se běh s nejnižší inertií.

---

## 7. Volba optimálního počtu shluků $k$

Protože $k$ je hyperparametr zadaný uživatelem předem, potřebujeme objektivní matematická kritéria pro jeho určení:

### 7.1 Loketní metoda (Elbow Method – WCSS / Inertia):
* S rostoucím $k$ hodnota WCSS monotonně klesá (pro $k = n$ je $\text{WCSS} = 0$).
* Hledáme bod zlomu (tzv. „loket“ / elbow), od kterého již přidávání dalších shluků nepřináší podstatné snížení chyby, ale pouze zbytečně drobí data.

### 7.2 Siluetový koeficient (Silhouette Analysis):
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

---

## 8. Algoritmus DBSCAN (Hustotní shlukování)

**DBSCAN** je zkratka pro **Density-Based Spatial Clustering of Applications with Noise** (Ester et al., 1996). Na rozdíl od K-Means nevychází z těžišť, ale z lokální hustoty bodů v prostoru.

### 8.1 Dva klíčové parametry:
1. **$\varepsilon$ (Epsilon):** Poloměr sousedství okolo zkoumaného bodu.
2. **`min_samples`:** Minimální počet bodů, které se musí nacházet v $\varepsilon$-okolí, aby byla oblast považována za hustou (jádro shluku).

### 8.2 Tři typy datových bodů v DBSCAN:
* **Jádrový bod (Core point):** Bod, jehož $\varepsilon$-okolí obsahuje alespoň `min_samples` bodů (včetně sebe samého).
* **Hraniční bod (Border point):** Bod, který sám není jádrovým bodem (má méně než `min_samples` sousedů), ale spadá do $\varepsilon$-okolí nějakého jádrového bodu.
* **Šum / Odlehlý bod (Noise point / Outlier):** Bod, který není jádrový ani hraniční. DBSCAN ho označí návěštím `-1` a **nepřiřadí ho do žádného shluku**.

```
         (Core) ── ε ── (Core) ── ε ── (Border)
           │              │
         (Core)         (Core)               [Noise -1]
```

### 8.3 Proč DBSCAN řeší limity K-Means:
1. **Není třeba zadávat počet shluků $k$:** Algoritmus sám identifikuje přirozený počet shluků v datech.
2. **Libovolný tvar shluků:** Detekuje prstence, půlměsíce, spirály a protáhlé struktury, kde K-Means fatálně selhává.
3. **Automatická filtrace anomálií:** Outliery nezkreslují shluky, ale jsou separovány jako šum.

---

## 9. Srovnání K-Means vs DBSCAN

| Kritérium | K-Means | DBSCAN |
| :--- | :--- | :--- |
| **Zadání počtu shluků** | Povinné předem ($k$) | Automatické (určí se dle $\varepsilon$ a `min_samples`) |
| **Geometrie shluků** | Pouze konvexní / kulové (izotropní) | Libovolné nekonvexní tvary (měsíce, spirály) |
| **Robustnost vůči šumu** | Nulová (každý outlier je nuceně přiřazen) | Vynikající (outliery označeny jako `-1`) |
| **Výpočetní náročnost** | $\mathcal{O}(n \cdot k \cdot d \cdot \text{iter})$ (velmi rychlý) | $\mathcal{O}(n \log n)$ se spatial indexy, $\mathcal{O}(n^2)$ bez nich |
| **Problém s různou hustotou** | Zkresluje hranice | Selhává při nestejných hustotách (fixní $\varepsilon$) |
| **Interpretace těžiště** | Jasný centroid $\mu$ | Nemá centroid (shluk je souvislá oblast) |

---

## 10. Expertní kritika a metodická úskalí (Guardrails)

Při aplikaci shlukování v produkční praxi je nutné hlídat několik kritických rizik:

1. **Fatální absence standardizace (`StandardScaler`):**
   * K-Means i DBSCAN počítají Euklidovské vzdálenosti $\sqrt{\sum (x_{ia} - x_{ja})^2}$.
   * Pokud má jedna proměnná rozsah $10^5$ (např. příjem v USD) a druhá jednotky (např. počet dětí), vzdálenost je určena z 99.99 % příjmem. Standardizace na nulový průměr a jednotkový rozptyl je **závazná a povinná**!
2. **Kletba dimenzionality (Curse of Dimensionality):**
   * V prostorech s $d > 50$ (např. po One-Hot kódování či u textu) se vzdálenosti mezi libovolnými dvěma body sbíhají ke stejné hodnotě (koncentrace měr).
   * **Pravidlo:** Před shlukováním vždy aplikovat redukci dimenzionality (PCA, t-SNE nebo UMAP) nebo použít kosinové metriky.
3. **Iluzorní shluky na náhodném šumu:**
   * K-Means rozdělí na $k$ shluků i data vygenerovaná z dokonale uniformního náhodného šumu!
   * Vždy je nutné validovat statistickou významnost (např. Hopkinsův test náhodnosti dat).

---

## 11. Moderní ML & AI kontext (Stav k 2026)

V moderních produkčních architekturách se klasické shlukování rozšiřuje o moderní standardy:

* **HDBSCAN (Hierarchical DBSCAN):**  
  Překonává největší slabinu DBSCAN – neschopnost pracovat s proměnlivou hustotou shluků. Využívá hierarchický strom minimální kostry a stabilní shluky extrahuje automaticky bez nutnosti fixního $\varepsilon$ (plně integrováno v `sklearn.cluster.HDBSCAN`).
* **MiniBatchKMeans & GPU Akcelerace:**  
  Pro datasety s miliony řádků (clickstream, logy) využívá stochastický gradientní sestup na mini-dávkách dat nebo GPU knihovny (cuML, PyTorch), což zrychluje výpočet o 2 až 3 řády.
* **Spherical K-Means na Transformer Embeddingy:**  
  Při práci s moderními vektorovými reprezentacemi textu (OpenAI `text-embedding-3`, BERT) se Euklidovská vzdálenost nahrazuje **kosinovou podobností** na jednotkové sféře.
* **XAI a Explainable Clustering:**  
  Shlukování není černá skříňka. V moderním MLOps se k vysvětlení segmentů nasazují zástupné rozhodovací stromy (Surrogate Decision Trees) nebo SHAP hodnoty, které byznysu objasní, jaké příznaky definují jednotlivé klientské skupiny.
* **LLM-assisted Cluster Labeling:**  
  Moderní pipeline automaticky vezme nejreprezentativnější body (medoidy) a statistické profily shluků, předá je velkému jazykovému modelu (LLM) a ten automaticky vygeneruje sémantické byznysové názvy a marketingové persony.
