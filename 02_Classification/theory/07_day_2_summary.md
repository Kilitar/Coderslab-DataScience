# Den 2: Shrnutí klasifikačních algoritmů a evaluačních metrik

Tento dokument představuje ucelené shrnutí **2. dne kurzu Machine Learning**, který byl věnován fundamentálním algoritmům pro řešení **klasifikačních úloh** s využitím knihovny **Scikit-learn**.

---

## 1. Přehled probíraných klasifikačních rodin

Během druhého dne jsme prozkoumali čtyři zásadní rodiny klasifikačních algoritmů, z nichž každá staví na odlišném matematickém a geometrickém paradigmatu:

```mermaid
graph TD
    ML[Klasifikační úlohy v ML] --> Inst[Instanční učení]
    ML --> Prob[Pravděpodobnostní modely]
    ML --> Rule[Pravidlové / Stromové modely]
    ML --> Geom[Geometrické / Maximální marže]

    Inst --> KNN[k-Nearest Neighbors (k-NN)]
    Prob --> LogReg[Logistická regrese]
    Rule --> DT[Rozhodovací strom (Decision Tree)]
    Geom --> SVM[Support Vector Machine (SVM)]
```

---

## 2. Detailní srovnání algoritmů

| Algoritmus | Matematický princip | Výhody | Nevýhody & Rizika | Klíčové hyperparametry | Kdy použít |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **k-NN** *(k-Nearest Neighbors)* | Metrická blízkost ve stavovém prostoru, většinové hlasování $k$ sousedů. Lazy learning (ukládá celá data). | • Intuitivní, bez trénovací fáze.<br>• Přirozeně nelineární hranice.<br>• Žádné předpoklady o rozdělení dat. | • Výpočetně drahá inference $O(N \cdot D)$.<br>• Trpí prokletím dimenzionality (*Curse of Dimensionality*).<br>• Extrémně citlivý na škálování příznaků. | • `n_neighbors` ($k$)<br>• `weights` (`'uniform'`, `'distance'`)<br>• `metric` (`'euclidean'`, `'manhattan'`) | Menší datasety ($N < 50\,000$), nízká dimenze ($D < 30$), lokální shluky v datech. |
| **Logistická regrese** *(Logistic Regression)* | Lineární kombinace příznaků mapovaná sigmoidální funkcí $\sigma(z) = \frac{1}{1 + e^{-z}}$ na pravděpodobnost $[0, 1]$. | • Kalibrované pravděpodobnosti výstupu.<br>• Vysoká interpretovatelnost koeficientů (Odds Ratios).<br>• Bleskový trénink i predikce. | • Předpokládá lineární separabilitu v logitovém prostoru.<br>• Citlivá na multikolinearitu.<br>• Bez polynomiálních členů nezachytí složité nelinearity. | • `C` (inverzní síla regularizace)<br>• `penalty` (`'l1'`, `'l2'`, `'elasticnet'`)<br>• `solver` (`'lbfgs'`, `'saga'`)<br>• `multi_class` (`'ovr'`, `'multinomial'`) | Medicínské a finanční skóringy, kde je vyžadována interpretovatelnost pravděpodobností. |
| **Rozhodovací strom** *(Decision Tree)* | Rekurzivní binární štěpení příznakového prostoru maximalizující pokles nečistoty (Gini index nebo Entropie). | • Bílý model (white-box) – snadná vizualizace.<br>• Nevyžaduje škálování/normalizaci dat.<br>• Snadno kombinuje numerické i kategorické příznaky. | • Silná náchylnost k přeučení (*overfitting*).<br>• Ortogonální hranice (schůdkovité dělení prostoru).<br>• Vysoká variabilita (malá změna dat změní strom). | • `criterion` (`'gini'`, `'entropy'`)<br>• `max_depth`<br>• `min_samples_split`<br>• `min_samples_leaf`<br>• `ccp_alpha` (prořezávání) | Úlohy vyžadující explicitní rozhodovací pravidla (`if-then`), příprava pro ansámbly (Random Forest, XGBoost). |
| **SVM** *(Support Vector Machine)* | Nalezení optimální nadroviny maximalizující geometrickou marži $\frac{2}{\|\mathbf{w}\|}$ mezi třídami. Podpůrné vektory. | • Robustní vůči odlehlým hodnotám mimo marži.<br>• Efektivní ve vysokých dimenzích ($D > N$).<br>• Jádrový trik (*Kernel Trick*) pro libovolné nelinearity. | • Výpočetní náročnost tréninku $O(N^2)$ až $O(N^3)$.<br>• Černá skříňka (těžká interpretace vah u nelineárních jader).<br>• Přímo neposkytuje pravděpodobnosti (vyžaduje Platt scaling). | • `kernel` (`'linear'`, `'rbf'`, `'poly'`)<br>• `C` (penalizace porušení marže)<br>• `gamma` (dosah vlivu podpůrného vektoru)<br>• `degree` (pro polynom) | Textová klasifikace, bioinformatika, složité nelineární hranice s menším až středním počtem vzorků. |

---

## 3. Evaluační metriky klasifikačních modelů

Správná volba metriky je klíčová – prostá přesnost (*Accuracy*) je při nevyvážených třídách (*Class Imbalance*) zavádějící.

### Matice záměn (Confusion Matrix)
| Skutečnost \ Predikce | Predikováno Negativní ($0$) | Predikováno Pozitivní ($1$) |
| :--- | :---: | :---: |
| **Skutečně Negativní ($0$)** | **TN** (True Negative) | **FP** (False Positive, Chyba I. typu) |
| **Skutečně Pozitivní ($1$)** | **FN** (False Negative, Chyba II. typu) | **TP** (True Positive) |

### Souhrn klíčových metrik
1. **Accuracy (Celková přesnost):**
   $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
   - *Vhodnost:* Vyvážené třídy s rovnoměrnou cenou chyb.
2. **Precision (Přesnost / Kladná prediktivní hodnota):**
   $$\text{Precision} = \frac{TP}{TP + FP}$$
   - *Otázka:* „Když model řekne, že vzorek je pozitivní, jak často má pravdu?“
   - *Kdy maximalizovat:* Minimalizace falešných poplachů (spam filtr, drahá a invazivní léčba).
3. **Recall / Sensitivity (Senzitivita / Úplnost):**
   $$\text{Recall} = \frac{TP}{TP + FN}$$
   - *Otázka:* „Kolik ze skutečně nemocných pacientů model dokázal odhalit?“
   - *Kdy maximalizovat:* Kritické diagnózy, odhalování podvodů (zmeškání pozitivního případu je katastrofální).
4. **F1-Score (Harmonický průměr Precision a Recall):**
   $$\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$
   - *Vhodnost:* Vyvážení mezi přesností a úplností při nerovnováze tříd.
5. **ROC křivka & AUC (Area Under Curve):**
   - Vztah mezi $TPR = \text{Recall}$ a $FPR = \frac{FP}{FP + TN}$ napříč všemi rozhodovacími prahy.
   - Nezávislá na volbě klasifikačního prahu (default $0.5$).
6. **Log Loss (Křížová entropie):**
   $$\text{Log Loss} = -\frac{1}{N}\sum_{i=1}^N \left[ y_i \ln(p_i) + (1 - y_i) \ln(1 - p_i) \right]$$
   - Penalizuje nejen chybné predikce, ale i přehnanou sebevědomost modelu při chybě.

---

## 4. Průvodce výběrem modelu (Decision Guide)

```
Je pro vás zásadní absolutní interpretovatelnost a jednoduchá pravidla?
├── ANO:
│   ├── Potřebujete pravidla typu "když-pak"? ──> Rozhodovací strom (Decision Tree)
│   └── Potřebujete pravděpodobnosti a vliv vah? ──> Logistická regrese (Logistic Regression)
└── NE:
    ├── Máte velmi vysokou dimenzi (texty, geny, D > N) nebo složitou nelineární hranici?
    │   └── ──> Support Vector Machine (SVM) s RBF či lineárním jádrem
    └── Jde o menší tabulková data s přirozenými lokálními shluky?
        └── ──> k-Nearest Neighbors (k-NN)
```

---

## 5. Znalostní kvíz (Otázky & Odpovědi)

1. **Otázka:** Který algoritmus neprovádí žádné explicitní učení parametrů v trénovací fázi a uchovává celou trénovací sadu v paměti?
   - *Správná odpověď:* **k-NN (k-Nearest Neighbors)** – tzv. *lazy learner* (líné učení).

2. **Otázka:** Jaký vliv má volba velmi malého $k$ (např. $k=1$) u algoritmu k-NN?
   - *Správná odpověď:* Model má velmi členitou a nepravidelnou rozhodovací hranici, což vede k **přeučení (overfittingu)** na šum v trénovacích datech.

3. **Otázka:** Pokud je pro nás fatální minout nemocného pacienta (minimalizujeme False Negatives), kterou metriku musíme maximalizovat?
   - *Správná odpověď:* **Recall (Senzitivitu / Úplnost)**.

4. **Otázka:** Co reprezentují podpůrné vektory (Support Vectors) v modelu SVM?
   - *Správná odpověď:* Jsou to **kritické hraniční vzorky**, které leží na hranici marže nebo uvnitř ní a jako jediné určují polohu a sklon oddělující nadroviny.

5. **Otázka:** Proč rozhodovací stromy nevyžadují normalizaci ani standardizaci vstupních příznaků?
   - *Správná odpověď:* Protože podmínka štěpení v uzlu ($x_j \le \theta$) závisí pouze na **monotónním pořadí hodnot** daného příznaku, nikoliv na jejich měřítku.
