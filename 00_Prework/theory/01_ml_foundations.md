# Úvod do Data Science a Strojového Učení (Prework 01)

> **Zdrojový materiál kurzu:**  
> `0-Introduction.txt`, `1-Ml_Python.txt`, `4-Model.txt`, `6-ML_Intro.txt`, `7-ML_Algorithms.txt`  
> (Definice ML a DS, přechod Software 1.0 -> 2.0, 4 paradigmata učení, byznysové využití, moderní kontext 2026)

---

## 1. Co je Data Science a Machine Learning?

**Data Science (Datová věda)** je interdisciplinární obor, který kombinuje **matematiku, statistiku, informatiku a oborovou expertízu (domain knowledge)** za účelem extrakce smysluplných znalostí a prakticky využitelných poznatků z dat.

Uvnitř datové vědy hraje klíčovou roli **Machine Learning (Strojové učení – ML)**:
- Jde o podobor **umělé inteligence (AI)**.
- Zaměřuje se na vývoj algoritmů, které jsou schopny **se automaticky učit a zlepšovat svůj výkon na základě zkušeností (dat)**, aniž by musely být pro každý jednotlivý případ explicitně naprogramovány.

```text
    ┌─────────────────────────────────────────────────────────┐
    │              ARTIFICIAL INTELLIGENCE (AI)               │
    │  ┌───────────────────────────────────────────────────┐  │
    │  │             MACHINE LEARNING (ML)                 │  │
    │  │  ┌─────────────────────────────────────────────┐  │  │
    │  │  │              DEEP LEARNING (DL)             │  │  │
    │  │  │  ┌───────────────────────────────────────┐  │  │  │
    │  │  │  │      Generative AI & LLMs (2026)      │  │  │  │
    │  │  │  └───────────────────────────────────────┘  │  │  │
    │  │  └─────────────────────────────────────────────┘  │  │
    │  └───────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────┘
```

---

## 2. Změna paradigmatu: Software 1.0 vs. Software 2.0

Tradiční způsob vývoje softwaru naráží na limity v momentech, kdy jsou pravidla příliš složitá, subjektivní nebo se mění v čase.

### Tradiční programování (Software 1.0):
Programátor musí pravidla logiky přesně znát a zapsat je do kódu (`if / else` podmínky):
$$\text{Data} + \text{Pravidla (Kód)} \implies \mathbf{Odpov\check{e}di}$$
*Příklad:* Detekce spamu pomocí ručně psaných filtrů – pokud e-mail obsahuje slovo „výhra“ nebo „viagra“, označ jako spam. Spammeři však pravidla rychle obejdou („v-ý-h-r-a“).

### Strojové učení (Software 2.0):
Algoritmus dostane velké množství historických příkladů (vstupní data) a správné odpovědi. Algoritmus sám matematicky odvodí optimální rozhodovací pravidla:
$$\text{Data} + \mathbf{Odpov\check{e}di} \implies \mathbf{Pravidla\ (Model)}$$
*Příklad:* Model analyzuje 500 000 e-mailů, najde skryté statistické závislosti slov a frekvencí a dokáže generalizovat i na e-maily se zkomolenými slovy.

---

## 3. Co je to vlastně „Model“?

V kontextu strojového učení je **model matematickou abstrakcí reálného světa**. Představuje naučenou funkci $f$, která mapuje vstupní charakteristiky na očekávaný výstup:

$$y = f(X) + \epsilon$$

Kde:
- $\mathbf{X}$ = **Příznaky (Features / Prediktory / Vstupy):** Matice nezávislých proměnných popisujících zkoumaný objekt (např. plocha bytu, počet pokojů, lokalita).
- $y$ = **Cílová proměnná (Target / Odezva / Výstup):** Hodnota, kterou se snažíme předpovědět (např. prodejní cena nemovitosti).
- $f$ = **Hypotéza / Funkce modelu:** Vnitřní parametry nalezené algoritmem (např. váhy lineární regrese $\boldsymbol{\beta}$, rozhodovací prahy stromu).
- $\epsilon$ = **Náhodná chyba (Epsilon / Šum):** Neodstranitelná neurčitost reality (faktory, které v datech nemáme k dispozici).

---

## 4. Čtyři základní paradigmata strojového učení

Podle toho, jaká data máme k dispozici a jakým způsobem se algoritmus učí, dělíme strojové učení do čtyř hlavních větví:

### 1. Učení s učitelem (Supervised Learning)
Trénovací data obsahují vstupy $X$ i **správné odpovědi (labely / ground truth $y$)**. Model se učí minimalizovat rozdíl mezi svou predikcí a skutečností.
- **Regrese (Regression):** Cílová proměnná $y$ je spojité číslo.
  - *Příklady:* Odhad tržní ceny diamantu, predikce teploty, odhad doby doručení zásilky.
  - *Algoritmy:* Lineární regrese, Ridge/Lasso, Decision Tree Regressor, Random Forest, XGBoost.
- **Klasifikace (Classification):** Cílová proměnná $y$ je diskrétní kategorie (třída).
  - *Příklady:* Schválení/zamítnutí úvěru (binární), diagnóza typu nádoru (zhoubný/nezhovný), rozpoznání typu vozidla (multiclass).
  - *Algoritmy:* Logistická regrese, k-NN, Rozhodovací stromy, Support Vector Machines (SVM).

### 2. Učení bez učitele (Unsupervised Learning)
Máme k dispozici pouze vstupy $X$, ale **žádné správné odpovědi $y$**. Cílem je odhalit skrytou vnitřní strukturu a vzory v datech.
- **Shlukování (Clustering):** Seskupování podobných objektů (např. zákaznická segmentace pro marketing – $k$-Means, DBSCAN).
- **Redukce dimenzionality:** Zmenšení počtu sloupců při zachování maxima informace (PCA, t-SNE, UMAP).
- **Detekce anomálií:** Hledání neobvyklých transakcí vymykajících se normálnímu chování (Isolation Forest).

### 3. Učení s částečným dohledem (Semi-Supervised Learning)
Kombinace malé množiny přesně anotovaných dat a obrovského množství neoznačených dat. Velmi časté v medicíně, kde je označení snímků lékařem extrémně drahé.

### 4. Zpětnovazební učení (Reinforcement Learning – RL)
Agent interaguje s dynamickým prostředím (Environment). Na základě svých akcí dostává odměny (Rewards) nebo tresty (Penalties). Jeho cílem je maximalizovat kumulativní odměnu v čase.
- *Využití:* Autonomní řízení aut, hraní her (AlphaGo), robotika.
- *Moderní význam (2026):* **RLHF (Reinforcement Learning from Human Feedback)** je klíčová metoda, pomocí které se ladí a zarovnávají moderní velké jazykové modely (ChatGPT, Gemini, Claude).

---

## 5. Proč je Python světovým standardem pro ML?

Python zcela ovládl oblast Data Science a Machine Learning díky unikátní kombinaci faktorů:
1. **Intuitivní a čitelná syntaxe:** Umožňuje vědcům soustředit se na matematiku a hypotézy, nikoli na nízkoúrovňovou správu paměti.
2. **Bohatý a propojený ekosystém:**
   - Manipulace s daty: **NumPy**, **Pandas**, **Polars**.
   - Vizualizace: **Matplotlib**, **Seaborn**, **Plotly**.
   - Statistika a vědecké výpočty: **SciPy**, **Statsmodels**.
   - Klasické strojové učení: **Scikit-learn**, **LightGBM**, **XGBoost**, **CatBoost**.
   - Hluboké učení (Deep Learning): **PyTorch**, **TensorFlow**, **JAX**.
3. **C/C++ a Rust pod kapotou:** Ačkoliv je Python interpretovaný jazyk, kritické výpočty v NumPy a Scikit-learn probíhají ve zkompilovaných knihovnách (BLAS, LAPACK) s plným využitím vícejádrových procesorů a vektorových instrukcí (SIMD / AVX).

---

## 6. Byznysové aplikace v reálné praxi

Strojové učení již dávno není akademickým experimentem, ale kritickou součástí firemní infrastruktury:

| Odvětví | Problém | Typ ML úlohy | Přínos pro byznys |
| :--- | :--- | :--- | :--- |
| **Finanční sektor** | Detekce platebních podvodů (Fraud Detection) | Klasifikace / Anomálie | Záchrana desítek milionů korun v reálném čase |
| **Bankovnictví** | Posuzování rizikovosti žadatelů (Credit Scoring) | Binární klasifikace | Minimalizace nesplacených úvěrů |
| **E-commerce & Streaming** | Doporučovací systémy (Recommenders) | Maticový rozklad / Ranking | Zvýšení konverzního poměru o 20–35 % (Amazon, Netflix) |
| **Maloobchod & Pohostinství** | Dynamická cenotvorba (Dynamic Pricing) | Regrese | Optimalizace marže a vytížení kapacit (Uber, Booking, aerolinky) |
| **Výroba a energetika** | Prediktivní údržba strojů (Predictive Maintenance) | Časové řady / Klasifikace | Prevence drahých neplánovaných odstávek výrobních linek |

---

## 7. Pohled do praxe (2026): Klasické ML vs. GenAI & LLMs

S nástupem generativní AI a velkých jazykových modelů (LLM) se často objevuje otázka: *„Má ještě smysl učit se klasickou regresi, rozhodovací stromy a Scikit-learn?“*

**Jednoznačná odpověď zní: ANO, více než kdy dříve.**

1. **Tabulková data (Tabular Data) vládnou byznysu:**  
   Většina firemních dat (transakce, zákaznické profily, skladové zásoby, lékařské záznamy, telemetrie) jsou strukturované tabulky. V této doméně stromové modely (**XGBoost, LightGBM, CatBoost**) a regularizované lineární modely dlouhodobě překonávají hluboké neuronové sítě i LLM jak v přesnosti, tak v rychlosti a nákladech.
2. **Deterministická přesnost a cena:**  
   Spustit predikci v Scikit-learn trvá **mikrosekundy** a stojí zlomek centu na běžném CPU. Poslat tabulku do velkého LLM trvá sekundy, stojí řádově více a hrozí halucinacemi v číslech.
3. **Regulace a vysvětlitelnost (Explainability / EU AI Act):**  
   V bankovnictví a medicíně musíte přesně doložit, proč byl úvěr zamítnut nebo stanovena konkrétní diagnóza. Lineární regrese, rozhodovací stromy a SHAP analýzy poskytují transparentní matematické zdůvodnění.
