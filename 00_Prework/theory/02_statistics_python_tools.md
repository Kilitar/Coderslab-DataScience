# Statistické minimum a Python nástroje (Prework 02)

> **Zdrojový materiál kurzu:**  
> `2-Python_Plotly.txt`, `3-Statistics.txt`, `5-Statistic_Python_Tools.txt`  
> (Deskriptivní statistika, míry polohy a rozptylu, korelace, Anscombův kvartet, Python ekosystém)

---

## 1. Proč je statistika srdcem strojového učení?

Dříve než se vžil termín *Machine Learning*, popisovaly se tyto metody v akademickém prostředí jako **vícerozměrná statistika (multivariate statistics)** nebo **statistické učení**.

Každý algoritmus strojového učení (od lineární regrese až po hluboké sítě) provádí statistický odhad:
- Hledá parametry rozdělení pravděpodobnosti dat.
- Minimalizuje očekávanou ztrátu (Loss Function).
- Odděluje deterministický signál od náhodného šumu.

Bez pochopení základní popisné (deskriptivní) statistiky je datový vědec „slepý“ – snadno podlehne klamu průměrů nebo zamění korelaci za příčinnou souvislost.

---

## 2. Míry polohy (Measures of Central Tendency)

Míry polohy vyjadřují jedním číslem „střed“ nebo typickou hodnotu zkoumaného souboru dat:

### A. Aritmetický průměr (Mean):
$$\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$$
- **Vlastnost:** Matematicky elegantní a snadno derivovatelný.
- **Zásadní slabina:** Je **extrémně citlivý na odlehlé hodnoty (outliery)**. Jediná extrémní hodnota dokáže průměr dramaticky posunout.

### B. Medián (Median – 50. percentil):
Hodnota, která seřazený soubor dělí na dvě stejně početné poloviny. 50 % pozorování je menších nebo rovných mediánu a 50 % větších.
- **Vlastnost:** Je **robustní vůči odlehlým hodnotám**. V šikmých rozděleních (příjmy, ceny nemovitostí) podává nesrovnatelně pravdivější obrázek než průměr.

### C. Modus (Mode):
Nejčastěji se vyskytující hodnota v souboru.
- **Využití:** Jediná míra polohy, kterou lze smysluplně použít i pro kategoriální (nominální) veličiny (např. nejčastější barva očí nebo nejprodávanější značka auta).

```text
       Symetrické rozdělení                Kladně šikmé rozdělení (Příjmy / Ceny)
             Mode                                  Mode
            Median                                Median
             Mean                                    Mean
              │                                        │
           ┌──┴──┐                                  ┌──┴──────┐
        ───┘     └───                            ───┘         └───────────────> Extrémní outliery
     Průměr = Medián = Modus                       Modus < Medián < Průměr
```

### 💼 Příklad z praxe: Mzdy ve startupu
Představme si malou firmu s 10 zaměstnanci:
- 9 juniorních vývojářů bere **40 000 Kč**.
- 1 zakladatel/CEO si vyplácí **1 500 000 Kč**.
- **Aritmetický průměr:** $\frac{9 \times 40\,000 + 1\,500\,000}{10} = \mathbf{186\,000\text{ Kč}}$ (naprosto zkreslené číslo, nikdo z běžných zaměstnanců tolik nemá!).
- **Medián:** $\mathbf{40\,000\text{ Kč}}$ (reálná a férová výpověď o typickém platu).

---

## 3. Míry variability a rozptylu (Measures of Dispersion)

Znalost samotného středu nestačí. Dva soubory mohou mít identický průměr, ale diametrálně odlišnou míru nejistoty:

### A. Rozptyl (Variance):
Průměrná kvadratická odchylka hodnot od jejich průměru:
$$s^2 = \frac{1}{n - 1} \sum_{i=1}^{n} (x_i - \bar{x})^2$$
*(Dělení $n-1$ představuje tzv. Besselovu korekci pro nestranný odhad rozptylu ze vzorku).*

### B. Směrodatná odchylka (Standard Deviation):
Kladná odmocnina z rozptylu:
$$s = \sqrt{s^2}$$
- **Klíčová výhoda:** Je vyjádřena **ve stejných jednotkách** jako původní data (např. v Kč, metrech, sekundách), takže má okamžitou intuitivní interpretaci.

### C. Kvartily a Mezichvartilové rozpětí (IQR):
- **Dolní kvartil ($Q_1$ – 25. percentil):** 25 % dat leží pod touto hodnotou.
- **Medián ($Q_2$ – 50. percentil):** 50 % dat.
- **Horní kvartil ($Q_3$ – 75. percentil):** 75 % dat leží pod touto hodnotou.
- **Mezichvartilové rozpětí (IQR):**
  $$\text{IQR} = Q_3 - Q_1$$
  Měří rozptyl prostředních 50 % dat a je imunní vůči extrémním výkyvům.

### D. Boxplot (Krabicový graf) a detekce outlierů:
Tukeyho pravidlo pro detekci anomálií:
- Dolní mez (Lower Whisker): $\max(\min(x), \, Q_1 - 1.5 \times \text{IQR})$
- Horní mez (Upper Whisker): $\min(\max(x), \, Q_3 + 1.5 \times \text{IQR})$
- Všechny body ležící vně těchto vousů jsou považovány za **statistické odlehlé hodnoty (outliery)**.

---

## 4. Míry závislosti: Korelace a její pasti

Při přípravě prediktorů pro model zkoumáme, jak spolu jednotlivé proměnné souvisejí:

### A. Pearsonův korelační koeficient ($r$):
Měří sílu a směr **čistě lineární závislosti** mezi dvěma spojitými veličinami:
$$r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}} \in [-1, +1]$$

- $r = +1$: Dokonalá přímá lineární závislost (rostoucí).
- $r = 0$: Žádná *lineární* závislost.
- $r = -1$: Dokonalá nepřímá lineární závislost (klesající).

### B. Spearmanův pořadový koeficient ($\rho$):
Počítá Pearsonovu korelaci nikoliv ze samotných čísel, ale z jejich **pořadí (ranks)**.
- Dokáže odhalit libovolnou **monotónní nelineární závislost** (např. exponenciální růst).
- Je robustní vůči extrémním outlierům.

### ⚠️ Zlatá pravidla datového vědce:
1. **Korelace neznamená kauzalita (Correlation is NOT Causation):**  
   Skutečnost, že dvě veličiny v čase statisticky korelují (např. spotřeba zmrzliny a počet utonutí v bazénech), neznamená, že jedna způsobuje druhou. Obvykle existuje skrytá třetí proměnná – tzv. **konfundující faktor** (v tomto případě horké letní počasí).
2. **Anscombův kvartet & Datasaurus Dozen:**  
   V roce 1973 sestavil statistik Francis Anscombe čtyři datasety, které měly naprosto identický průměr, rozptyl i Pearsonovu korelaci ($r = 0.816$), ale při vykreslení představovaly: (1) lineární vztah, (2) dokonalou parabolu, (3) přímku s jedním obřím outlierem a (4) svislý shluk.  
   **Ponaučení:** *Nikdy se nespoléhejte pouze na souhrnná čísla, data musíte VŽDY vizualizovat!*

---

## 5. Python technologický stack pro statistiku

| Knihovna | Klíčové moduly / funkce | Typické využití v kurzu |
| :--- | :--- | :--- |
| **NumPy** | `np.mean()`, `np.median()`, `np.std()`, `np.percentile()` | Nízkoúrovňové bleskové maticové operace a vektorové výpočty |
| **Pandas** | `df.describe()`, `df.corr()`, `df.skew()`, `df.quantile()` | Práce s tabulkovými daty, agregace, groupby analýzy |
| **SciPy** | `scipy.stats.normaltest()`, `scipy.stats.pearsonr()`, `scipy.stats.iqr()` | Testování statistických hypotéz, testy normality, p-hodnoty |
| **Plotly** | `px.histogram()`, `px.box()`, `px.scatter()`, `px.imshow()` | Interaktivní webová vizualizace s zoomem a tooltipy |
| **Seaborn** | `sns.pairplot()`, `sns.heatmap()`, `sns.kdeplot()` | Rychlý publikační statistický průzkum dat v noteboocích |
