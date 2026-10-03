# Den 3: Ansámblové metody a Bagging (Ensemble Methods & Bagging)

> **Cíl modulu:** Porozumět principu ansámblových metod (**Ensemble Methods**), které kombinují předpovědi více modelů za účelem dosažení vyšší přesnosti, stability a optimálního kompromisu mezi vychýlením a rozptylem (**Bias-Variance Trade-off**). Detailně rozebrat 4 základní rodiny ansámblů (**Voting**, **Stacking**, **Bagging**, **Boosting**) a hloubkově prozkoumat techniku **Bagging** (Bootstrap Aggregating) včetně výběru s opakováním a Out-Of-Bag evaluace.

---

## 1. Proč používáme ansámblové metody?

Během 1. a 2. dne kurzu jsme se seznámili se základními modely strojového učení pro regresi (lineární regrese, polynomy, rozhodovací strom) a klasifikaci (k-NN, logistická regrese, rozhodovací strom, SVM).

V reálné praxi však narážíme na data s vysoce nelineárními vztahy, šumem a vysokou dimenzionalitou. Jeden samostatný model často naráží na své limity:
- **Vysoké vychýlení (*High Bias*):** Model je příliš jednoduchý a nedokáže zachytit strukturu dat $\to$ **podtrénování (*Underfitting*)**.
- **Vysoký rozptyl (*High Variance*):** Model je příliš citlivý na náhodné fluktuace a šum v trénovací sadě $\to$ **přeučení (*Overfitting*)**.

```
Celková chyba = (Vychýlení / Bias)² + Rozptyl / Variance + Neodstranitelný šum (Irreducible Error)
```

Vedle regularizace (Lasso, Ridge) a křížové validace představují **ansámblové metody** nejúčinnější nástroj pro dosažení optimálního poměru mezi vychýlením a rozptylem. 

> **Moudrost davu (Wisdom of the Crowd):**  
> Pokud zkombinujeme předpovědi skupiny nezávislých modelů, jejich individuální náhodné chyby se navzájem vyruší, zatímco správné trendy se posílí.

---

## 2. Přehled 4 základních typů ansámblů

Ansámblové metody dělíme podle toho, **jak jsou jednotlivé modely trénovány** a **jak se kombinují jejich výstupy**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             Ansámblové metody                               │
└─────────────────────────────────────────────────────────────────────────────┘
          │                                                  │
   Paralelní / Nezávislé modely                     Sekvenční modely
          │                                                  │
 ┌────────┴────────┐                                         │
 │                 │                                         │
Voting          Bagging                                   Boosting
(Různé alg.,   (Stejný alg.,                             (Slabé modely
 stejná data)   bootstrap data)                           za sebou)
          │
      Stacking
  (Meta-model kombinuje
   predikce bázových modelů)
```

### 1. Hlasování (Voting)
Technika, která spojuje predikce několika různých algoritmů natrénovaných na stejných trénovacích datech (např. SVM + Logistická regrese + k-NN):
- **V regresi:** Finální předpověď je prostým nebo váženým průměrem předpovědí jednotlivých modelů.
- **V klasifikaci:**
  - **Většinové hlasování (*Majority / Hard Voting*):** Každý model odevzdá hlas pro třídu (0 nebo 1). Vyhrává třída s nejvyšším počtem hlasů.  
    *Příklad ze slajdů:* Pokud 3 modely hlasují pro třídu 0 a 2 modely pro třídu 1, výsledkem je **třída 0** (poměr 3:2).
  - **Vážené hlasování (*Weighted / Soft Voting*):** Modely predikují pravděpodobnosti příslušnosti ke třídám. Tyto pravděpodobnosti se zprůměrují (případně s váhami dle spolehlivosti modelu) a vybere se třída s nejvyšší průměrnou pravděpodobností.

### 2. Stacking (Stacked Generalization)
Pokročilejší alternativa k prostému hlasování:
- Trénuje několik bázových modelů (*Base Models*) na stejných datech.
- Místo prostého průměrování předá jejich predikce tzv. **meta-modelu** (*Meta-Learner / Blender*).
- **Meta-model** (často jednoduchá lineární nebo logistická regrese) se naučí optimální váhy a interakce: zjistí, kterému bázovému modelu věřit v jaké části stavového prostoru.

### 3. Bagging (Bootstrap Aggregating)
Vytvoření mnoha **nezávislých modelů stejného typu** (typicky neprořezaných rozhodovacích stromů):
- Každý model se trénuje na **odlišném náhodném vzorku dat vytvořeném s vracením (Bootstrapping)**.
- Výsledky se agregují průměrováním (regrese) nebo většinovým hlasováním (klasifikace).
- **Hlavní cíl Baggingu:** **Drastické snížení rozptylu (Variance)** bez zvýšení vychýlení.

### 4. Boosting
Sekvenční trénování slabých modelů (*Weak Learners*, např. mělkých stromů – pařezů):
- Modely se netrénují nezávisle, ale **jeden po druhém**.
- Každý další model se zaměřuje na chyby a nepřesnosti předchozího modelu (např. vážením těžkých pozorování v AdaBoostu nebo aproximací gradientu chybové funkce v Gradient Boostingu).
- Vytváří jeden mimořádně silný predikční model.

---

## 3. Detailní rozbor: Co je Bagging (Bootstrap Aggregating)?

Bagging navrhl v roce 1996 americký statistik **Leo Breiman**. Název je zkratkou pro **Bootstrap Aggregating**.

### 4 kroky fungování Baggingu:

```
Původní dataset (N vzorků)
       │
       ├──► Bootstrap vzorek 1 ──► Model 1 (Strom 1) ──► Predikce 1 ──┐
       ├──► Bootstrap vzorek 2 ──► Model 2 (Strom 2) ──► Predikce 2 ──┼──► Agregace ──► Finální predikce
       └──► Bootstrap vzorek M ──► Model M (Strom M) ──► Predikce M ──┘    (Průměr / Hlasování)
```

#### Krok 1: Bootstrapping (Výběr s vracením)
Z původní trénovací množiny o velikosti $N$ náhodně vybereme $N$ pozorování **s vracením (*with replacement*)**.
- To znamená, že některá pozorování mohou být vybrána víckrát (duplikována), zatímco jiná nebudou vybrána vůbec.

> **Matematická vsuvka: Kolik dat zůstane stranou (Out-Of-Bag – OOB)?**  
> Pravděpodobnost, že konkrétní pozorování *nebude* vybráno v jednom tahu, je $1 - \frac{1}{N}$.  
> Pro $N$ nezávislých tahů je pravděpodobnost nevybrání rovna:
> $$P(\text{nevybráno}) = \left(1 - \frac{1}{N}\right)^N$$
> Pro velké $N \to \infty$ platí známá limita:
> $$\lim_{N \to \infty} \left(1 - \frac{1}{N}\right)^N = \frac{1}{e} \approx 0.3679 \approx 36.8\,\%$$
> **Význam:** Každý model v Baggingu je trénován na přibližně **63.2 % unikátních dat**. Zbylých **36.8 % vzorků (tzv. Out-Of-Bag – OOB)** slouží jako vestavěná validační sada bez nutnosti dělení na Train/Validation!

#### Krok 2: Trénování nezávislých modelů
Na každém z vygenerovaných bootstrap vzorků natrénujeme samostatný model (např. hluboký rozhodovací strom).
- Protože jsou modely zcela nezávislé, **trénování lze perfektně paralelizovat** na všechna jádra procesoru (`n_jobs=-1`).

#### Krok 3: Generování predikcí
Každý natrénovaný model vygeneruje předpověď pro nová testovací pozorování.

#### Krok 4: Agregace výsledků
Jednotlivé předpovědi spojíme do jednoho konsensu:
- **V regresi (aritmetický průměr):**  
  *Příklad ze slajdů:* Model 1 vrátí $100$, Model 2 vrátí $150$, Model 3 vrátí $200$.  
  Výsledná predikce:
  $$\hat{y} = \frac{100 + 150 + 200}{3} = 150$$
- **V klasifikaci (většinové hlasování):**  
  Třída, která obdrží největší počet hlasů, se stává finální predikcí ansámblu.

---

## 4. Matematický důkaz: Proč Bagging snižuje rozptyl?

Představme si $M$ nezávislých modelů, z nichž každý má rozptyl chyb $\sigma^2$. Pokud jsou modely ideálně nekorelované, rozptyl jejich průměru je:
$$\text{Var}\left(\frac{1}{M} \sum_{i=1}^M \hat{f}_i(x)\right) = \frac{\sigma^2}{M}$$
Pokud je mezi modely kladná korelace $\rho$ (což v praxi je, protože vzorky pocházejí ze stejného rozdělení), rozptyl se řídí vzorcem:
$$\text{Var}(\text{Ansámbl}) = \rho \sigma^2 + \frac{1 - \rho}{M} \sigma^2$$
Když počet modelů $M$ roste:
- Druhý člen $\frac{1 - \rho}{M} \sigma^2 \to 0$.
- Výsledný rozptyl klesá k teoretickému minimu $\rho \sigma^2$.

Právě snaha ještě více snížit korelaci $\rho$ mezi stromy vedla Lea Breimana k vynálezu **Random Forestu** (náhodný výběr podmnožiny příznaků při každém štěpení).

---

## 5. Shrnutí a srovnání

| Vlastnost | Samotný rozhodovací strom | Bagging (Ansámbl stromů) |
| :--- | :--- | :--- |
| **Vychýlení (Bias)** | Velmi nízké (hluboký strom) | Velmi nízké (stejné jako strom) |
| **Rozptyl (Variance)** | **Extrémně vysoký** (přeučení) | **Výrazně snížený** |
| **Stabilita vůči šumu** | Nízká (drobná změna v datech změní celý strom) | Vysoká (odolný vůči výkyvům) |
| **Interpretovatelnost** | Výborná (snadná vizualizace pravidel) | Horší (černá skříňka $M$ stromů) |
| **Výpočetní náročnost** | Rychlý trénink jednoho stromu | Vyšší, ale plně paralelizovatelný |

V následujících materiálech navážeme na Bagging jeho nejpoužívanějším reprezentantem: **Random Forestem**.
