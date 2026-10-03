# 🧭 Teorie Support Vector Machines (SVM) – Podpůrné vektory, Nadroviny a Jádrový trik

> **Stav k 09/2026** | Zpracováno na základě kurzu *Support vector machine - theoretical introduction* (Coders Lab) a moderních standardů Scikit-learn (1.6+).

---

## 1. Úvod & Filozofie algoritmu

**Support Vector Machine (SVM)**, česky *metoda podpůrných vektorů*, představuje vrchol klasického statistického učení pro klasifikaci a regresi (Vapnik & Červoněnkis, 1995). Zatímco logistická regrese hledá dělící rovinu optimalizací pravděpodobnostní věrohodnosti (Log Loss) a k-NN rozhoduje lokálním hlasováním sousedů, **SVM přistupuje k problému geometricky**:

> **Cíl SVM:**  
> Najít optimální dělící **nadrovinu (hyperplane)**, která nejenže správně oddělí třídy, ale učiní tak s **maximálním možným geometrickým odstupem (maximálním okrajem / marginem)** od nejbližších pozorování obou tříd.

```
                    Třída +1 (Abnormal / Pozitivní)
                           x     x
                              x     [Support Vector]
    -------------------------------------------------- H1: w^T x + b = +1
                ^
                |   MAXIMÁLNÍ OKRAJ (MARGIN) = 2 / ||w||
                v
    ================================================== Optimální nadrovina: w^T x + b = 0
                ^
                |
                v
    -------------------------------------------------- H2: w^T x + b = -1
                      [Support Vector]
                           o     o
                               o     Třída -1 (Normal / Negativní)
```

---

## 2. Podpůrné vektory (Support Vectors) a Okraj (Margin)

### 2.1 Co jsou podpůrné vektory?
**Podpůrné vektory** jsou ty konkrétní datové body z trénovací sady, které leží **nejblíže dělící nadrovině** (přímo na hranicích okraje $H_1$ a $H_2$).

- **Klíčová vlastnost:** Pokud bychom z datasetu odstranili libovolný bod, který *není* podpůrným vektorem, optimální dělící nadrovina by se **vůbec nezměnila**.
- Pozice a orientace nadroviny závisí **výhradně na podpůrných vektorech**. SVM je díky tomu paměťově mimořádně efektivní – po natrénování stačí v paměti držet pouze tyto kritické hraniční vzorky.

### 2.2 Geometrický okraj (Margin)
Okraj představuje pásmo (koridor) oddělující obě třídy.
1. **Hard Margin (Striktní okraj):** Předpokládá dokonalou lineární separabilitu. Uvnitř koridoru nesmí ležet ani jediný datový bod:
   $$y_i (\mathbf{w}^T \mathbf{x}_i + b) \ge 1, \quad \forall i \in \{1, \dots, N\}$$
2. **Soft Margin (Tolerantní okraj):** V reálných datech dochází k překryvu tříd a šumu. Zavádíme tzv. *slack variables* (chybové proměnné) $\xi_i \ge 0$, které povolují určitá narušení okraje.

---

## 3. Co je nadrovina (Hyperplane)?

Nadrovina je podprostor, jehož dimenze je o jedna menší než dimenze celého prostoru ($n - 1$):

| Dimenze prostoru dat ($n$) | Geometrický tvar nadroviny | Matematická reprezentace |
|:---:|:---:|:---|
| **1D** ($n=1$) | **Bod** (0D) | $w_1 x_1 + b = 0 \implies x_1 = -b / w_1$ |
| **2D** ($n=2$) | **Přímka** (1D) | $w_1 x_1 + w_2 x_2 + b = 0$ |
| **3D** ($n=3$) | **Rovina** (2D) | $w_1 x_1 + w_2 x_2 + w_3 x_3 + b = 0$ |
| **$n$-D** ($n > 3$) | **Nadrovina** ($(n-1)$D) | $\mathbf{w}^T \mathbf{x} + b = 0$ |

### 3.1 Hledání optimální nadroviny (Lagrangeovy multiplikátory)
Vzdálenost libovolného bodu $\mathbf{x}$ od nadroviny je dána vzorcem:
$$\text{dist}(\mathbf{x}) = \frac{|\mathbf{w}^T \mathbf{x} + b|}{\|\mathbf{w}\|}$$
Pro body na okraji platí $|\mathbf{w}^T \mathbf{x} + b| = 1$, celková šířka okraje je tedy:
$$\text{Margin} = \frac{2}{\|\mathbf{w}\|}$$
Maximalizace marže $\frac{2}{\|\mathbf{w}\|}$ je ekvivalentní minimalizaci jejího převráceného čtverce:
$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^N \xi_i \quad \text{za podmínek} \quad y_i(\mathbf{w}^T \mathbf{x}_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$

- **Hyperparametr $C$ (Regularizace):**
  - **Velké $C$:** Tvrdý trest za chyby. Úzký okraj, menší tolerance chyb na trénovacích datech $\implies$ riziko **přeučení (overfitting)**.
  - **Malé $C$:** Měkký trest. Širší okraj, model ignoruje šum a odlehlé body $\implies$ vyšší **schopnost generalizace**, ale riziko podučení (underfitting).

---

## 4. Lineárně neseparabilní data a analogie s dekou (The Blanket Analogy)

Většina reálných úloh není v původním prostoru lineárně separabilní (např. data uspořádaná v soustředných kruzích).

> **Analogie s dekou ze slajdů kurzu:**  
> Představte si míčky dvou barev rozprostřené na dece (2D plocha). Žádnou přímkou na dece je nelze oddělit.  
> V okamžiku, kdy deku prudce zvedneme a zatřeseme s ní, míčky vyletí do vzduchu (3D prostor).  
> Když jsou míčky ve vzduchu, můžeme mezi ně vložit rovnou deku jako dělící plochu tak, že červené míčky zůstanou nad dekou a modré pod ní!  
> **Přechod z 2D do 3D umožnil lineární rozdělení pomocí roviny.**

Matematicky: mapujeme vstupní vektor $\mathbf{x} \in \mathbb{R}^n$ do vyšší dimenze pomocí nelineární transformace $\phi(\mathbf{x}) \in \mathbb{R}^m$, kde $m \gg n$.

---

## 5. Jádrový trik (The Kernel Trick) a Jádrové funkce

Přímý výpočet souřadnic $\phi(\mathbf{x})$ ve vysoké dimenzi by byl výpočetně extrémně drahý nebo nemožný (u RBF jádra je dimenze dokonce nekonečná!).

**Jádrový trik (Kernel Trick):**  
Díky duální formulaci optimalizace SVM potřebuje algoritmus **pouze skalární součiny** mezi vektory $\langle \phi(\mathbf{x}), \phi(\mathbf{y}) \rangle$.  
Jádrová funkce $K(\mathbf{x}, \mathbf{y})$ spočítá tento skalární součin přímo z původních souřadnic, **aniž bychom transformaci $\phi$ vůbec explicitně prováděli**:
$$K(\mathbf{x}, \mathbf{y}) = \langle \phi(\mathbf{x}), \phi(\mathbf{y}) \rangle$$

### 5.1 Čtyři základní jádrové funkce

#### 1. Lineární jádro (Linear Kernel)
$$K(\mathbf{x}, \mathbf{y}) = \mathbf{x} \cdot \mathbf{y}$$
- **Kdy použít:** Když jsou data lineárně separabilní, nebo když máme obrovské množství příznaků (např. textová klasifikace, genové exprese, kde $p > n$).
- **Výhoda:** Rychlý trénink, nízká výpočetní náročnost.

#### 2. Polynomiální jádro (Polynomial Kernel)
$$K(\mathbf{x}, \mathbf{y}) = (\mathbf{x} \cdot \mathbf{y} + c)^d$$
- **Parametry:** $d$ (stupeň polynomu, v Scikit-learn `degree`), $c$ (posun, `coef0`).
- **Kdy použít:** Pokud existuje apriorní předpoklad o nelineárních polynomiálních interakcích mezi příznaky.

#### 3. RBF jádro (Radial Basis Function / Gaussovské jádro)
$$K(\mathbf{x}, \mathbf{y}) = \exp\left(-\gamma \|\mathbf{x} - \mathbf{y}\|^2\right)$$
- **Parametr $\gamma$ (Gamma):** Určuje dosah vlivu jednotlivých trénovacích vzorků:
  - **Velká $\gamma$:** Vliv vzorku klesá velmi rychle s rostoucí vzdáleností. Dělící hranice je velmi členitá, obepíná jednotlivé body $\implies$ **vysoké riziko přeučení**.
  - **Malá $\gamma$:** Vliv vzorku sahá daleko. Hranice je hladká a široká $\implies$ vyšší generalizace.
- **Kdy použít:** Výchozí a nejflexibilnější volba v moderním ML (`SVC(kernel='rbf')`). Odpovídá nekonečně-dimenzionálnímu prostoru příznaků.

#### 4. Sigmoidální jádro (Sigmoid / Hyperbolic Tangent)
$$K(\mathbf{x}, \mathbf{y}) = \tanh(\alpha (\mathbf{x} \cdot \mathbf{y}) + c)$$
- **Kdy použít:** Simuluje chování dvouvrstvé dopředné neuronové sítě (MLP). Vyžaduje opatrné ladění parametrů $\alpha$ a $c$.

---

## 6. Vícedenní / Vícetřídní klasifikace (Multiclass SVM)

SVM je principiálně binární klasifikátor. Pro $K$ tříd se v praxi používají dvě strategie:

### 6.1 One-vs-One (OvO)
Vytvoří binární model pro **každou dvojici tříd**.
Počet potřebných dílčích klasifikátorů:
$$N_{\text{models}} = \frac{K(K - 1)}{2}$$
- **Příklad:** Pro 3 druhy tučňáků ($K=3$) vzniknou $\frac{3 \cdot 2}{2} = 3$ modely:
  1. Adelie vs. Chinstrap
  2. Adelie vs. Gentoo
  3. Chinstrap vs. Gentoo
- Predikce probíhá většinovým hlasováním (každý model hlasuje pro jednu třídu).
- **Standard v Scikit-learn:** `sklearn.svm.SVC` implementuje právě strategii OvO (knihovna `libsvm`).

### 6.2 One-vs-Rest (OvR / One-vs-All)
Vytvoří $K$ binárních modelů, kde každý model odděluje danou třídu od všech ostatních spojených dohromady.
- Vhodné pro velké počty tříd.

---

## 7. Výhody, Nevýhody a Porovnání s ostatními modely

| Vlastnost | SVM | k-NN | Logistická regrese | Rozhodovací stromy |
|---|:---:|:---:|:---:|:---:|
| **Hranice rozhodování** | Hladká nadrovina / nelineární (RBF) | Po částech lineární (Voronoi) | Lineární nadrovina (Sigmoida) | Ortogonální schodovitá pravidla |
| **Škálování příznaků** | **KRITICKÉ** (Nutný StandardScaler) | **KRITICKÉ** (Nutný StandardScaler) | Důležité pro regularizaci | **Netřeba** (Invariantní) |
| **Výpočetní náročnost** | $O(N^2)$ až $O(N^3)$ (pomalé na $N > 50\,000$) | $O(N \cdot p)$ při predikci | $O(N \cdot p)$ (velmi rychlé) | $O(N \cdot p \cdot \log N)$ (rychlé) |
| **Vysoká dimenze ($p > N$)** | **Vynikající** | Špatné (Kletba dimenzionality) | Dobré s $L_1$ / $L_2$ | Průměrné |
| **Interpretovatelnost** | Nízká (Černá skříňka u RBF) | Střední (Příklady sousedů) | **Vysoká** (Odds Ratio $\exp(\beta)$) | **Maximální** (Pravidla IF-THEN) |
| **Pravděpodobnostní výstup** | Ne přímo (vyžaduje Plattovo škálování) | Empirické podíly sousedů | **Přímý** ($\sigma(z) \in [0, 1]$) | Podíly v listu |
