# 🚨 Detekce anomálií: Metodika, Matematika & Local Outlier Factor (LOF)

> **Přednáškový základ:** *Anomaly detection - an introduction.pdf* (Coders Lab Data Science Track)  
> **Klíčová témata:** Typologie anomálií (bodové, kontextové, kolektivní), průmyslové domény, statistické limity, Isolation Forest, matematika Local Outlier Factor (LOF), hustotní dosahovost (reachability distance) a nasazení v Scikit-learn.

---

## 1. Úvod do problematiky detekce anomálií

V éře rozhodování založeného na datech (data-driven decisions) závisí úspěch organizací na kvalitě informací vstupujících do analytických a strojových modelů. **Detekce anomálií (Anomaly Detection / Outlier Mining)** je proces identifikace vzácných, neobvyklých nebo odlehlých pozorování, která se signifikantně odchylují od očekávaného chování či většinového vzorce v datasetu.

### Proč je detekce anomálií unikátní disciplínou?
1. **Extrémní vzácnost (Imbalance):** Anomálie tvoří typicky méně než $1\,\%$ (často $<0.1\,\%$) všech pozorování.
2. **Neexistence popisků (Unsupervised povaha):** V reálném provozu nemáme k dispozici trénovací data s označením „podvod“ nebo „závada turbíny“. Modely se musí naučit definovat normálnost samy bez učitele.
3. **Dynamická proměnlivost normálu:** To, co je normální ve všední den v noci, je anomálií v poledne o víkendu (např. zatížení serveru, transakce na účtu).
4. **Vliv na strojové učení:** Přítomnost neidentifikovaných odlehlých hodnot zkresluje ztrátové funkce ($MSE$), posouvá regresní přímky a narušuje rozhodovací hranice klasifikátorů. Očištění dat od šumu je kritickým krokem pro stabilitu celého ML pipelinu.

---

## 2. Taxonomie anomálií (Dělení ze slajdů kurzu)

Přednáška (str. 5–6) dělí anomálie do tří fundamentálních kategorií:

```
                            ┌────────────────────────────────────────┐
                            │         Taxonomie Anomálií             │
                            └────────────────────┬───────────────────┘
                                                 │
         ┌───────────────────────────────────────┼───────────────────────────────────────┐
         │                                       │                                       │
         ▼                                       ▼                                       ▼
┌──────────────────┐                   ┌──────────────────┐                    ┌──────────────────┐
│ Bodové anomálie  │                   │ Kontextové anom. │                    │ Kolektivní anom. │
│ (Global / Point) │                   │  (Conditional)   │                    │   (Collective)   │
└────────┬─────────┘                   └────────┬─────────┘                    └────────┬─────────┘
         │                                       │                                       │
  Extrémní hodnota                       Hodnota je anomální                     Skupina bodů je
  vůči celému vzorku                     pouze v daném kontextu                  podezřelá společně
  (např. převod 10 mil. USD              (např. 35 °C v lednu;                   (např. mikroplatby
  místo běžných 500 USD).                odběr elektřiny v noci).                opakované 100x za minutu).
```

### 1. Bodové / Globální anomálie (Point Anomalies)
Jednotlivé pozorování, které se drasticky odchyluje od zbytku celého datasetu.
- **Příklad:** Převod částky $5\,000\,000\text{ Kč}$ z běžného účtu studenta, kde historicky probíhaly transakce v řádu stokorun.

### 2. Kontextové / Podmíněné anomálie (Contextual Anomalies)
Pozorování, které by v obecném kontextu bylo zcela normální, ale v konkrétním kontextovém rámci (čas, roční období, geografická lokace) představuje anomálii.
- **Kontextové atributy:** Časové razítko, den v týdnu, lokalita, věková skupina.
- **Behaviorální atributy:** Teplota, objem přenesených dat, krevní tlak.
- **Příklad:** Naměřená venkovní teplota $32\,^\circ\text{C}$ je v červenci normální, ale v lednu ve střední Evropě představuje dramatickou anomálii.

### 3. Kolektivní anomálie (Collective Anomalies)
Skupina vzájemně souvisejících datových bodů, které při individuálním posouzení nevykazují anomální chování, ale jako celek tvoří neobvyklou sekvenci či vzorec.
- **Příklad:** 50 plateb po sobě ve výši $1\text{ USD}$ během 30 sekund (útok carding pro ověření funkčnosti ukradené platební karty).

---

## 3. Průmyslové a byznysové aplikace (Slajdy 9–10)

| Doména | Sledovaná data | Povaha anomálie | Byznysový dopad |
| :--- | :--- | :--- | :--- |
| **Finanční sektor** | Bankovní transakce, IP adresy, geolokace | Útoky na účty, praní špinavých peněz (AML), zneužití karet | Okamžité zablokování transakce, prevence ztrát v miliardách USD |
| **Zdravotnictví & IoT** | EKG signál, tep, krevní tlak, glukometry | Arytmie, fibrilace síní, prudký pád vitálních funkcí | Záchrana lidského života, včasné varování na chytrých hodinkách |
| **Prediktivní údržba** | Vibrace turbín, teplota motoru, akustika | Mikropraskliny ložisek, kavitace čerpadla | Zabránění katastrofální havárii a odstávce výrobní linky |
| **Průmyslová výroba** | Kamerový záznam povrchu, laserové skenování | Vady laku, praskliny sváru, odchylky rozměrů | Automatizovaná kontrola kvality bez lidské chyby |
| **Těžební průmysl** | Seismografické senzory, otřesy půdy | Seismické mikrozáchvěvy indikující zával | Okamžitá evakuace dolu, prevence ztrát na životech |
| **ML Data Cleaning** | Trénovací datasety pro predikce | Chyby měření, špatně zadané údaje, přepsané nuly | Zvýšení generalizační přesnosti a stability modelů |

---

## 4. Přehled metodik detekce anomálií (Slajdy 13–15)

Historicky i v moderní praxi se využívají čtyři hlavní myšlenkové proudy:

### 1. Statistické metody (Parametrické rozdělení)
Předpokládají, že normální data pocházejí ze známého pravděpodobnostního rozdělení (např. Gaussovského normálního rozdělení $\mathcal{N}(\mu, \sigma^2)$):
- **Z-skóre:** $z = \frac{x - \mu}{\sigma}$. Hodnoty $|z| > 3$ jsou považovány za anomálie ($3\sigma$ pravidlo).
- **Tukeyho IQR boxplot:** $x < Q_1 - 1.5 \cdot IQR$ nebo $x > Q_3 + 1.5 \cdot IQR$.
- **Limit:** Zcela selhávají vícerozměrně ($D > 2$), při asymetrických distribucích a u multimodálních dat (shluky s různou hustotou).

### 2. Shlukovací metody (Clustering-based)
Využívají algoritmy neřízeného učení k rozdělení prostoru:
- **DBSCAN:** Body, které neleží v $\varepsilon$-okolí žádného jádrového bodu, získají label `-1` (šum/outlier).
- **K-Means:** Vzdálenost bodu k nejbližšímu centroidu $d(\mathbf{x}, \boldsymbol{\mu}_k)$. Body s extrémní vzdáleností jsou anomálie.
- **Limit:** Citlivost na volbu $k$ či globálního $\varepsilon$.

### 3. Isolation Forest (iForest, Liu et al., 2008)
Stromový ansámbl založený na myšlence: *„Anomálie jsou vzácné a v prostoru osamocené, proto je lze izolovat pomocí náhodných řezů mnohem rychleji než normální body.“*
- Algoritmus náhodně vybere příznak a náhodnou dělící hodnotu.
- Normální body vyžadují hluboké stromy (mnoho řezů), anomálie jsou izolovány blízko kořene (malá průměrná hloubka $h(x)$).

### 4. Local Outlier Factor (LOF, Breunig et al., 2000)
Hustotní neparametrická metoda. Hodnotí, nakolik je lokální hustota v okolí bodu nižší než hustota v okolí jeho nejbližších sousedů.
- **Klíčová přednost:** Dokáže spolehlivě odhalit anomálie v datasetech, kde se střídají velmi husté shluky s řídkými shluky (**proměnlivá hustota**).

---

## 5. Matematické základy Local Outlier Factor (LOF)

Algoritmus LOF spočívá ve čtyřech rigorózních matematických krocích:

### Krok 1: $k$-vzdálenost bodu $p$ ($k\text{-distance}(p)$)
Pro dané celé číslo $k \ge 1$ je $k$-vzdálenost bodu $p$, značeno $d_k(p)$, eukleidovská vzdálenost $d(p, o)$ mezi bodem $p$ a objektem $o \in D$ taková, že:
1. Alespoň pro $k$ objektů $o' \in D \setminus \{p\}$ platí $d(p, o') \le d(p, o)$.
2. Nanejvýš pro $k-1$ objektů $o' \in D \setminus \{p\}$ platí $d(p, o') < d(p, o)$.

Množina $k$ nejbližších sousedů bodu $p$ se značí $N_k(p)$.

### Krok 2: Dosahová vzdálenost (Reachability Distance)
Dosahová vzdálenost bodu $p$ vzhledem k referenčnímu bodu $o$ je definována jako:

$$\text{reach-dist}_k(p, o) = \max\left\{ d_k(o),\, d(p, o) \right\}$$

> **Intuitivní význam:** Pokud bod $p$ leží velmi blízko bodu $o$ (uvnitř jeho $k$-sféry), jeho dosahová vzdálenost se uměle „zarazí“ na hodnotě $d_k(o)$. Tím se dramaticky snižuje statistická fluktuace a citlivost na lokální šum uvnitř hustých shluků.

### Krok 3: Lokální dosahová hustota (Local Reachability Density, LRD)
Lokální dosahová hustota bodu $p$ je převrácená hodnota průměrné dosahové vzdálenosti bodu $p$ od jeho $k$ nejbližších sousedů:

$$\text{lrd}_k(p) = \frac{|N_k(p)|}{\sum_{o \in N_k(p)} \text{reach-dist}_k(p, o)}$$

- Pokud má bod $p$ kolem sebe sousedy v těsné blízkosti, průměrná dosahová vzdálenost je malá a $\text{lrd}_k(p)$ je **vysoká**.
- Pokud je bod $p$ osamocen v řídkém prostoru, průměrná dosahová vzdálenost je velká a $\text{lrd}_k(p)$ je **nízká**.

### Krok 4: Výpočet Local Outlier Factor (LOF)
Výsledný faktor LOF bodu $p$ je průměrný poměr lokálních dosahových hustot jeho sousedů ku hustotě samotného bodu $p$:

$$\text{LOF}_k(p) = \frac{\sum_{o \in N_k(p)} \frac{\text{lrd}_k(o)}{\text{lrd}_k(p)}}{|N_k(p)|} = \frac{\frac{1}{|N_k(p)|} \sum_{o \in N_k(p)} \text{lrd}_k(o)}{\text{lrd}_k(p)}$$

---

## 6. Interpretace skóre LOF

| Hodnota $\text{LOF}_k(p)$ | Stav bodu $p$ | Geometrická interpretace |
| :---: | :---: | :--- |
| **$\approx 1.0$** | **Normální bod (Inlier)** | Bod má srovnatelnou hustotu jako jeho sousedé. Leží uvnitř homogenního shluku (ať už hustého či řídkého). |
| **$< 1.0$** | **Vnitřní jádrový bod** | Bod se nachází v ještě hustší oblasti než jeho okolní sousedé (střed hustého jádra). |
| **$1.2 \text{ až } 1.5$** | **Mírný hraniční bod** | Bod leží na vnějším okraji shluku (přechodová zóna). |
| **$> 1.5$** | **Signifikantní anomálie (Outlier)** | Hustota bodu je výrazně nižší než hustota jeho sousedů. Bod je izolován. |

---

## 7. Proč LOF triumfuje nad globálními metodami: Problém různé hustoty

Mějme dataset se dvěma shluky:
- **Shluk $C_1$ (Velmi hustý):** 200 bodů, průměrná vzdálenost mezi sousedy $0.1$.
- **Shluk $C_2$ (Velmi řídký):** 100 bodů, průměrná vzdálenost mezi sousedy $1.5$.
- **Bod $p_1$:** Leží ve vzdálenosti $0.6$ od shluku $C_1$.
- **Bod $p_2$:** Leží uvnitř shluku $C_2$ (běžná vzdálenost $1.4$).

### Výsledek srovnání:
1. **Globální k-NN distance / Z-skóre:**
   - Vzdálenost $p_1$ je $0.6$ (menší než $1.4$!).
   - Globální metoda označí $p_1$ za **normální bod** a body v $C_2$ označí za **anomálie**, což je naprosto nesmyslné!
2. **Local Outlier Factor (LOF):**
   - Pro bod $p_1$: $\text{lrd}(p_1)$ je nízká ($0.6$), ale jeho sousedé v $C_1$ mají $\text{lrd}$ obrovskou ($0.1$). Poměr $\frac{\text{lrd}(\text{sousedi})}{\text{lrd}(p_1)} \approx 6.0 \gg 1$. LOF spolehlivě odhalí $p_1$ jako **lokální anomálii**!
   - Pro body v $C_2$: Mají sice nízkou hustotu, ale jejich sousedé mají také nízkou hustotu. Poměr $\frac{\text{lrd}(\text{sousedi})}{\text{lrd}(p_2)} \approx 1.0$. LOF je správně ponechá jako **normální body**!

---

## 8. Implementace v Scikit-learn (`LocalOutlierFactor`)

V Scikit-learn existují dva základní režimy práce s třídou `LocalOutlierFactor`:

### 1. Outlier Detection (`novelty=False`, výchozí režim)
Model analyzuje zadaný trénovací dataset a označí v něm odlehlá pozorování:
- Metoda: `fit_predict(X)` vrátí pole s hodnotami `1` (inlier) a `-1` (outlier).
- Skóre anomality: atribut `negative_outlier_factor_`. V Scikit-learn je skóre záporné (čím nižší/zápornější číslo, tím větší anomálie). Skutečný $\text{LOF} = -\text{negative\_outlier\_factor\_}$.

```python
from sklearn.neighbors import LocalOutlierFactor

# n_neighbors: typicky 20
# contamination: očekávaný podíl anomálií v datech (např. 0.03 = 3 %)
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.03)
labels = lof.fit_predict(X)  # 1 pro normální body, -1 pro anomálie

# Převedení na kladné LOF skóre dle Breunigovy teorie
lof_scores = -lof.negative_outlier_factor_
```

### 2. Novelty Detection (`novelty=True`)
Model se natrénuje na čistých trénovacích datech a následně predikuje anomálie na **nových, dosud neviděných datech**:
- Vyžaduje nastavení `novelty=True`.
- Umožňuje volat metodu `predict(X_new)` a `decision_function(X_new)`.

---

## 9. Metodické Guardrails pro detekci anomálií 2026

1. **Standardizace proměnných:** Protože LOF vychází z eukleidovských vzdáleností, všechny příznaky musí být standardizovány pomocí `StandardScaler` nebo `RobustScaler` (pokud data již obsahují těžké extrémy).
2. **Volba parametru `n_neighbors`:**
   - Spodní mez: $\text{n\_neighbors} > 10$ (zamezuje falešným detekcím v malých náhodných skupinkách).
   - Horní mez: $\text{n\_neighbors} < N_{\text{min\_cluster}}$ (nesmí být větší než velikost nejmenšího očekávaného shluku). V praxi funguje výborně $\text{n\_neighbors} \in [15, 35]$.
3. **Pravidlo pro `contamination`:**
   - Pokud je `contamination="auto"`, Scikit-learn použije fixní prahovou hodnotu. V byznysu je vždy nutné kontaminaci kalibrovat dle finančních či kapacitních možností týmu (např. kolik podezřelých transakcí denně zvládne tým vyšetřit).
4. **Kombinace s Isolation Forest:** V produkčních pipeline se doporučuje kombinovat LOF (vynikající na lokální hustoty v nízkých dimenzích) s Isolation Forest (vynikající v dimenzích $D > 20$).
