# K-Nearest Neighbors (k-NN): Teoretický úvod a geometrické základy klasifikace (09/2026)

> **Zdrojový materiál kurzu:**  
> `K_nearest_neighbors_-_theoretical_introduction.pdf`  
> (Definice klasifikace, algoritmus k-NN, metriky vzdáleností: Euclidean, Manhattan, Minkowski, pravidla pro volbu parametru $k$, příklad klasifikace automobilů, výhody a limity metody).

---

## 1. Co je klasifikační problém?

Ve strojovém učení s učitelem (Supervised Learning) rozlišujeme dva základní typy úloh:
1. **Regrese:** Cílová proměnná $y \in \mathbb{R}$ je **spojitá** (např. tržní cena domu v Seattlu, cena diamantu v USD, teplota, spotřeba paliva).
2. **Klasifikace:** Cílová proměnná $y \in \{C_1, C_2, \dots, C_K\}$ je **diskrétní / kategoriální** (např. druh tučňáka, zda je email SPAM či HAM, zda je pacient zdravý či má patologii páteře).

```text
                     SUPERVISED LEARNING (Učení s učitelem)
                                    |
          +-------------------------+-------------------------+
          |                                                   |
      REGRESE                                            KLASIFIKACE
  Cílová proměnná y je SPOJITÁ                      Cílová proměnná y je KATEGORIÁLNÍ
  (např. 450 000 USD, 3.4 litru)                    (např. Adelie / Chinstrap / Gentoo)
  Hodnotí se odchylky: MAE, RMSE, R²                Hodnotí se shoda: Accuracy, Precision, Recall, F1
```

### Typy klasifikačních úloh:
- **Binární klasifikace ($K = 2$):** Cílová veličina nabývá právě dvou hodnot ($y \in \{0, 1\}$ nebo $\{\text{Negativní}, \text{Pozitivní}\}$).
  - Příklad: detekce podvodu u platební karty, klasifikace vozu (*Fast* vs. *Slow*).
- **Vícetřídní (Multi-class) klasifikace ($K > 2$):** Cílová veličina nabývá jedné z $K$ vzájemně se vylučujících tříd.
  - Příklad: druh tučňáka (*Adelie*, *Chinstrap*, *Gentoo*), čtení ručně psaných číslic 0–9.
- **Vícenálepková (Multi-label) klasifikace:** Každému pozorování může být přiřazeno více štítků současně (např. článek o sportu i ekonomice).

---

## 2. Princip algoritmu K-Nearest Neighbors (k-NN)

Algoritmus k nejbližších sousedů (**k-Nearest Neighbors**, zkráceně **k-NN**) je jedním z nejintuitivnějších a koncepčně nejčistších algoritmů strojového učení. Vychází z fundamentálního předpokladu:

> **Heuristika podobnosti:** Pozorování se stejnými nebo podobnými vlastnostmi (příznaky) leží v příznakovém prostoru blízko sebe a mají tendenci patřit do stejné třídy.

```text
       x₂ ^
          |          O  (Třída B)
          |         /
          |    O   /   O
          |       /
          |     [?] <--- Nový neznámý bod
          |     / \
          |    /   \
          |   X     X   (Třída A)
          |  X
          +--------------------------> x₁
```

### Klíčové vlastnosti k-NN:
1. **Líný model (Lazy Learner / Instance-based Learning):**
   - Na rozdíl od lineární regrese nebo rozhodovacích stromů **neprobíhá žádná explicitní fáze trénování**.
   - Model se nesnaží najít analytickou funkci ani stromovou strukturu pravidel. Pouze si **uloží celý trénovací dataset do paměti** ($\text{Trénovací čas} = O(1)$).
   - Veškerý výpočetní výkon je odložen až do fáze **predikce (inference)**, kdy se pro každý nový dotazovaný bod počítají vzdálenosti ke všem uloženým vzorkům.
2. **Neparametrický model:**
   - Model nepředpokládá žádné konkrétní rozdělení pravděpodobnosti dat ani lineární separabilitu.
   - Počet parametrů modelu neroste pevně s hypotézou, ale je dán samotnou velikostí datové sady.
3. **Přirozená podpora pro vícetřídní klasifikaci:**
   - Algoritmus funguje pro libovolný počet tříd $K$ bez nutnosti převádět problém na One-vs-Rest (OvR) či One-vs-One (OvO).

---

## 3. Výpočet vzdáleností v prostoru příznaků

Základním stavebním kamenem k-NN je definice metriky vzdálenosti $d(\mathbf{x}, \mathbf{y})$ mezi dvěma body $\mathbf{x} = (x_1, x_2, \dots, x_n)$ a $\mathbf{y} = (y_1, y_2, \dots, y_n)$ v $n$-rozměrném prostoru.

### 1. Eukleidovská vzdálenost ($L_2$ norma)
Nejběžnější a výchozí metrika. Představuje geometrickou délku úsečky spojující dva body v prostoru (přímá vzdušná vzdálenost). Vychází z Pythagorovy věty:

$$d_{\text{Euclidean}}(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|_2 = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}$$

- **V 2D rovině:** $d = \sqrt{(x_1 - y_1)^2 + (x_2 - y_2)^2}$
- **Geometrie izočáry (okolí bodu):** Kružnice (v 2D) resp. hypersféra (v $n$-D).

### 2. Manhattanská vzdálenost ($L_1$ norma / City-Block / Taxicab)
Vzdálenost odpovídající pohybu po pravoúhlé uliční síti (jako v newyorském Manhattanu), kde se nelze pohybovat diagonálně:

$$d_{\text{Manhattan}}(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|_1 = \sum_{i=1}^n |x_i - y_i|$$

- **Geometrie izočáry:** Čtverec pootočený o 45° (kosočtverec v 2D) resp. křížový polytop v $n$-D.
- **Výhoda:** V prostorech s vyšší dimenzí nebo s výskytem odlehlých hodnot bývá $L_1$ norma robustnější než $L_2$, protože neumocňuje rozdíly na druhou.

### 3. Minkowského vzdálenost ($L_p$ norma)
Matematické zobecnění obou předchozích metrik řízené reálným parametrem $p \ge 1$:

$$d_{\text{Minkowski}}(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|_p = \left( \sum_{i=1}^n |x_i - y_i|^p \right)^{1/p}$$

- Pro $p = 1$: Odpovídá **Manhattanské vzdálenosti**.
- Pro $p = 2$: Odpovídá **Eukleidovské vzdálenosti**.
- Pro $p \to \infty$: Konverguje k **Čebyševově vzdálenosti** (maximová metrika):  
  $$d_\infty(\mathbf{x}, \mathbf{y}) = \max_{i=1,\dots,n} |x_i - y_i|$$

| Metrika | Scikit-learn zápis | Vzorec | Vhodné použití |
| :--- | :--- | :--- | :--- |
| **Euclidean** | `metric='euclidean'` nebo `metric='minkowski', p=2` | $\sqrt{\sum (x_i - y_i)^2}$ | Standardní spojitá fyzikální data, prostorové souřadnice |
| **Manhattan** | `metric='manhattan'` nebo `metric='minkowski', p=1` | $\sum \|x_i - y_i\|$ | Mřížková data, výskyt outlierů, diskrétní číselné příznaky |
| **Minkowski** | `metric='minkowski', p=p` | $(\sum \|x_i - y_i\|^p)^{1/p}$ | Laditelný hyperparametr pro optimalizaci |

---

## 4. Pravidla pro volbu hyperparametru $k$ (Bias-Variance Tradeoff)

Výsledek klasifikace závisí zcela zásadně na volbě čísla $k$ (počet uvažovaných sousedů). Volba $k$ představuje ukázkový příklad kompromisu mezi vychýlením a rozptylem (**Bias-Variance Tradeoff**).

```text
      k = 1                                      Optimální k                                  k = N
  (Overfitting)                            (Generalizovaný model)                         (Underfitting)
---------------------------------------------------------------------------------------------------------
Vysoká variance                            Vyvážený Bias i Variance                      Vysoký bias
Nulový bias na trainu                      Hladké, ale věrné hranice                     Zcela ignoruje lokální vzory
Hranice kopírují každý šum                 Odolný vůči šumu a chybám                     Predikuje pouze majoritní třídu
```

### 1. Extrém: $k = 1$ (Příliš malé $k$)
- Každý bod v trénovací sadě má za svého nejbližšího souseda sám sebe $\implies$ trénovací přesnost je **100 %**.
- Hranice rozhodování tvoří tzv. **Voronoiovu mozaiku (Voronoi Tessellation)** s ostrými, zubatými přechody.
- Jakýkoli šum, překlep v labelu nebo odlehlá hodnota vytvoří v prostoru izolovaný „ostrov“ jiné třídy.
- **Diagnóza:** Masivní přeučení (Overfitting), špatná generalizace na testovací sadě.

### 2. Extrém: $k = N$ (Příliš velké $k$)
- Pro každý testovací bod se hlasování účastní úplně všechny body z trénovací množiny.
- Model zcela ztrácí schopnost reagovat na lokální strukturu příznaků $X$.
- Výsledná predikce je pro každý bod identická: **vždy vyhrává třída, která má v celém datasetu největší zastoupení (majoritní třída)**.
- **Diagnóza:** Masivní podtrénování (Underfitting).

### 3. Zlatá pravidla pro praktickou volbu $k$:
1. **Pravidlo odmocniny:** Jako výchozí odrazový můstek se často používá heuristika:
   $$k \approx \sqrt{N}$$
   (Kde $N$ je celkový počet trénovacích pozorování).
2. **Volba lichého čísla pro binární klasifikaci:**
   - Pokud řešíme 2 třídy, volíme $k \in \{3, 5, 7, 9, 11, \dots\}$.
   - Liché číslo garantuje, že **nemůže nastat patová situace (remíza / tie vote)** typu 2 vs. 2.
3. **Křížová validace (GridSearchCV):**
   - V profesionální praxi se $k$ nikdy nevolí naslepo. Testuje se mřížka hodnot (např. $k \in [1, 35]$) a vyhodnocuje se validační křivka přesnosti (Accuracy / F1-score).

---

## 5. Hlasovací mechanismy: Uniform vs. Distance

Při určování výsledné třídy pro dotazovaný bod $\mathbf{x}_{\text{new}}$ vybere k-NN jeho $k$ nejbližších sousedů $\mathcal{N}_k(\mathbf{x}_{\text{new}})$.

### A. Prosté majoritní hlasování (`weights='uniform'`)
Každý z $k$ sousedů má stejný hlas o váze 1 bez ohledu na to, zda je od bodu vzdálen 0.1 mm nebo 50 mm:

$$\hat{y} = \arg\max_{c \in \{1,\dots,K\}} \sum_{i \in \mathcal{N}_k(\mathbf{x}_{\text{new}})} \mathbb{I}(y_i = c)$$

### B. Vážení podle převrácené hodnoty vzdálenosti (`weights='distance'`)
Sousedé, kteří jsou k bodu blíže, mají větší vliv na rozhodnutí než sousedé na samém okraji uvažovaného okolí:

$$w_i = \frac{1}{d(\mathbf{x}_{\text{new}}, \mathbf{x}_i)}$$

$$\hat{y} = \arg\max_{c \in \{1,\dots,K\}} \sum_{i \in \mathcal{N}_k(\mathbf{x}_{\text{new}})} w_i \cdot \mathbb{I}(y_i = c)$$

*(V Scikit-learn se v případě, že $d=0$, váha nastaví na maximum).*  
Vážení vzdáleností výrazně pomáhá v situacích s nerovnoměrnou hustotou dat a zmírňuje citlivost na volbu $k$.

---

## 6. Kletba dimenzionality (Curse of Dimensionality)

Největším teoretickým i praktickým nepřítelem k-NN je **vysoký počet dimenzí (příznaků) $d$**:

1. **Exponenciální nárůst objemu prostoru:**
   - Objem hyperkrychle o hraně $L$ v $d$ dimenzích je $L^d$.
   - Abychom zachovali stejnou hustotu datových bodů v prostoru jako v 1D s 10 body, potřebujeme pro 2D $10^2 = 100$ bodů, pro 10D $10^{10} = 10\,000\,000\,000$ bodů!
2. **Fenomén prázdného prostoru (Empty Space Phenomenon):**
   - V reálných datech s desítkami či stovkami proměnných je prostor téměř úplně prázdný.
3. **Kolaps vzdáleností (Distance Concentration):**
   - V prostorech s vysokou dimenzí platí matematická vlastnost:
     $$\lim_{d \to \infty} \frac{d_{\max} - d_{\min}}{d_{\min}} = 0$$
   - Vzdálenost k „nejbližšímu“ sousedovi se stává téměř stejnou jako vzdálenost k „nejvzdálenějšímu“ sousedovi. Koncept *blízkosti* ztrácí svůj rozlišovací smysl.

> **Závěr pro praxi:** Pokud má dataset mnoho příznaků ($d > 20$), k-NN selhává, pokud předem neprovedeme **selekci příznaků** nebo **redukci dimenzionality** (např. PCA – Principal Component Analysis).

---

## 7. Výhody a nevýhody k-NN (Srovnávací matice)

| Výhody (Strengths) | Nevýhody (Weaknesses) |
| :--- | :--- |
| **Intuitivní a snadno vysvětlitelný:** Žádná černá skříňka, logiku rozhodnutí pochopí každý manažer. | **Extrémně pomalá predikce (Inference):** Časová složitost $O(N \cdot d)$ pro každý dotazovaný bod! Při milionech řádků je inference nepoužitelně pomalá. |
| **Nulový trénovací čas:** Dataset se pouze načte do paměti ($O(1)$). | **Obrovské paměťové nároky:** Model musí mít celou trénovací sadu neustále v RAM ($O(N \cdot d)$). |
| **Přirozeně nelineární:** Dokáže modelovat libovolně složité a křivolaké tvary rozhodovacích hranic. | **Kritická citlivost na měřítko proměnných:** Bez standardizace (StandardScaler) proměnná s velkými čísly převálcuje všechny ostatní. |
| **Přirozený multi-class:** Funguje stejně dobře pro 2, 3 i 50 tříd bez nutnosti dekompozice. | **Citlivost na irelevantní šum:** Příznaky bez prediktivní síly zhoršují metriku vzdálenosti. |

---

## 8. Moderní pohled (2026): Kde k-NN dominuje dnes?

Přestože v klasickém tabulárním strojovém učení bývá k-NN často překonán gradientním boostingem (XGBoost, LightGBM, CatBoost), zažil algoritmus v posledních letech **obrovskou renesanci**:

1. **Vektorové databáze & RAG (Retrieval-Augmented Generation):**
   - Moderní AI a velké jazykové modely (LLM) ukládají sémantické embeddingy textů a obrázků do vektorových databází (Pinecone, Qdrant, Chroma, Milvus, pgvector).
   - Vyhledávání relevantního kontextu v dokumentech není nic jiného než **hledání k nejbližších sousedů v prostoru embeddingů** (obvykle přes kosinovou podobnost).
2. **Přibližné vyhledávání nejbližších sousedů (ANN – Approximate Nearest Neighbors):**
   - Aby se obešla pomalá složitost $O(N)$, moderní systémy nepoužívají exaktní k-NN, ale grafové indexy (např. **HNSW – Hierarchical Navigable Small World**), které najdou nejbližší sousedy v čase $O(\log N)$ s 99% přesností.
