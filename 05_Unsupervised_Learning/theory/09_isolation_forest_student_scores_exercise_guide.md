# Průvodce cvičením: Detekce anomálií studentů pomocí Isolation Forest

Tento dokument detailně rozebírá praktické cvičení zaměřené na detekci odlehlých pozorování v reálném akademickém datasetu `student_scores.csv` (30 641 studentů) pomocí algoritmu **Isolation Forest** v kombinaci s redukcí dimenzionality **PCA**. Speciální důraz je kladen na porovnání citlivosti modelu při předpokladu **2.5 %** vs. **1.0 %** kontaminace dat.

---

## 1. Zadání cvičení a kontext

### Formulace úlohy
1. Načíst dataset `student_scores.csv` do prostředí (Pandas / Google Colab).
2. Importovat potřebné metody z knihoven `pandas` a `scikit-learn`.
3. Vytvořit DataFrame a ponechat pouze tři spojité sloupce s výsledky testů: `MathScore`, `ReadingScore`, `WritingScore`.
4. Provést redukci dimenzionality na dvě dimenze (2D) pro přehlednou vizualizaci (pomocí PCA).
5. Použít třídu `IsolationForest` k detekci anomálií za předpokladu, že anomálie tvoří **2.5 %** všech pozorování (`contamination=0.025`). Zvolit vhodné hodnoty dalších hyperparametrů (`n_estimators=100`, `random_state=42`).
6. Vygenerovat grafické znázornění, které vizuálně odliší běžná pozorování od anomálií.
7. Zjistit a analyzovat, jak se graf a klasifikace změní, pokud předpokládáme podíl anomálií pouze **1.0 %** (`contamination=0.01`).

---

## 2. Architektura dat & Doménový kontext

Dataset `student_scores.csv` obsahuje záznamy o **30 641 studentech**. Tři klíčové předměty jsou hodnoceny na škále od 0 do 100 bodů:
- `MathScore`: Výsledek z matematického testu (průměr $\approx 66.5$, sm. odchylka $\approx 15.3$).
- `ReadingScore`: Výsledek z čtenářské gramotnosti (průměr $\approx 69.4$, sm. odchylka $\approx 14.7$).
- `WritingScore`: Výsledek z písemného projevu (průměr $\approx 68.4$, sm. odchylka $\approx 15.4$).

### Korelační struktura:
Mezi předměty existuje silná lineární závislost:
- Korelace Reading $\leftrightarrow$ Writing je extrémně vysoká ($r \approx 0.95$).
- Korelace Math $\leftrightarrow$ Reading/Writing je rovněž vysoká ($r \approx 0.80 - 0.82$).

Většina studentů vykazuje vyrovnané výsledky ve všech třech oblastech. **Anomálie v tomto kontextu představují:**
1. **Extrémně slabé studenty:** Studenty s celkovým selháním (skóre 0–25 napříč všemi testy).
2. **Extrémně excelentní studenty:** Perfektní skóre 100/100/100 (v Isolation Forest je jakýkoliv izolovaný extrém na okraji distribuce snadno separovatelný mělkými řezy).
3. **Disproporční studenty (Asymetrické profily):** Studenty s těžkým nepoměrem – např. 95 bodů z matematiky, ale pouze 20 bodů z psaní, nebo naopak.

---

## 3. Redukce dimenzionality pomocí PCA: Význam hlavních komponent

Aplikací metody hlavních komponent (`PCA(n_components=2)`) na trojici skórů dosahujeme následujících výsledků:

- **1. hlavní komponenta (PC1):** Vysvětluje **90.56 %** celkového rozptylu!
  - Váhy (koeficienty): Všechny tři předměty mají srovnatelně kladné váhy ($\approx [0.57, 0.58, 0.58]$).
  - **Sémantický význam:** PC1 reprezentuje **celkovou akademickou úroveň studenta** (souhrnný průměrný výkon). Záporné hodnoty značí celkově podprůměrné studenty, vysoké kladné hodnoty značí premianty.
- **2. hlavní komponenta (PC2):** Vysvětluje **7.90 %** celkového rozptylu.
  - Váhy: Kladná váha pro matematiku ($\approx +0.81$) a záporné váhy pro čtení a psaní ($\approx -0.40, -0.42$).
  - **Sémantický význam:** PC2 reprezentuje **asymetrii mezi kvantitativním (matematickým) a verbálním nadáním**. Kladné hodnoty značí studenty s výraznou převahou matematiky nad jazykem; záporné hodnoty značí humanitně orientované studenty.
- **Celkový vysvětlený rozptyl:** PC1 + PC2 pokrývají **98.47 %** veškeré variability v datech! Pouze 1.53 % rozptylu zůstává ve třetí komponentě.

---

## 4. Analýza výsledků: Kontaminace 2.5 % vs. 1.0 %

Při trénování modelu `IsolationForest(n_estimators=100, random_state=42)` na 2D projekci získáváme:

| Metrika / Vlastnost | Model A: Contamination = 2.5 % (0.025) | Model B: Contamination = 1.0 % (0.010) | Posun / Rozdíl |
| :--- | :--- | :--- | :--- |
| **Počet anomálií (Outliers)** | **766 studentů** | **307 studentů** | **-459 studentů (-60 %)** |
| **Počet běžných studentů (Inliers)** | 29 875 studentů | 30 334 studentů | +459 studentů |
| **Rozhodovací offset prahu** | **$\approx -0.1064$** | **$\approx -0.1332$** | Práh je přísnější o **0.0268** |
| **Typ zachycených anomálií** | Extrémní selhání, disproporce i mírné okraje | Pouze nejzávažnější excesy a kolapsy | Zvýšení přesnosti (Precision) |

### Klíčový poznatek o posunu grafu:
1. Při **2.5 %** model ohraničuje širší obálku kolem eliptického mračna bodů. Zachycuje nejen studenty s nulovými body, ale i studenty, kteří mají mírnější disbalanci (např. matematika 80, psaní 50).
2. Při **1.0 %** se filtr dramaticky zúží. Zůstává pouze **307 nejradikálnějších případů** – studenti s celkovým skóre pod 25 bodů nebo studenti s astronomickou asymetrií.
3. Všech 307 anomálií z 1.0% modelu tvoří **podmnožinu** 766 anomálií z 2.5% modelu.

---

## 5. Metodická reflexe: Trénování na 2D (PCA) vs. na původních 3D datech

V zadání je uveden postup: *„Reduce dimensions to two for better visualization. Use the IsolationForest class to detect outliers.“*

V datové praxi je důležité rozumět rozdílu mezi dvěma přístupy:
- **Přístup A (Zadání):** PCA $\to$ Isolation Forest na 2D.
  - *Výhoda:* Snadná interpretace rozhodovacích hranic a přímá shoda mezi 2D grafem a predikcemi modelu.
  - *Nevýhoda:* Ztráta zbývajících 1.53 % rozptylu (3. komponenta), která může obsahovat specifickou jemnou disbalanci mezi čtením a psaním.
- **Přístup B (Průmyslový standard):** Isolation Forest na plných 3D datech $\to$ PCA pouze pro vizualizaci.
  - *Výhoda:* Stromové dělení využívá originální škály a neztrácí žádnou informaci.
  - *Pozorování:* Jelikož PC1 a PC2 pokrývají 98.5 % rozptylu, obě metody mají vysokou shodu na nejextrémnějších anomáliích.

---

## 6. Pedagogické a business implikace

Ve vzdělávacím systému odpovídá volba kontaminace reálnému rozhodnutí vedení školy:
- **Nastavení 1.0 % (Kritická intervence):** Určeno pro okamžitou krizovou pedagogickou pomoc (doučování, sociální šetření pro žáky s celkovým selháním). Kapacity pedagogicko-psychologických poraden jsou omezené, proto je žádoucí nulový počet falešných poplachů.
- **Nastavení 2.5 % (Preventivní screening):** Zahrnuje i žáky se specifickými poruchami učení (např. dysgrafie, kdy je čtení a matematika v normě, ale psaní zaostává).
