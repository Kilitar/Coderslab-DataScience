# 🗺️ Mental Notes: Datová mapa a katalog pro Session 2 (Dny 3, 4 & Homework)

Tento přehled slouží jako interní navigační a analytická mapa pro celou **Session 2** kurzu Data Science (Advanced Machine Learning, Ensemble metody, Neuronové sítě a NLP).

---

## 📅 Blok 1: Den 3 – Pokročilé modely strojového učení (Ensemble Methods)
**Témata Dne 3:** Random Forest, Bagging, Extra Trees, Boosting (AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost), Stacking, Voting a interpretace modelů (Feature Importance, SHAP).

| Datový soubor | Řádky | Sloupce | Typ úlohy | Cílová proměnná (Target) | Klíčové prediktory a kontext |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **`diamonds_preprocessed.csv`** | 53 899 | 10 | **Regrese** | `price` (USD) | `carat`, `cut`, `color`, `clarity`, `depth`, `table`, `x`, `y`, `z`. Velký regresní dataset ideální pro srovnání škálovatelnosti a přesnosti stromových ansámblů (Single Tree vs. Random Forest vs. XGBoost). |
| **`heart_data_normalized.csv`** | 303 | 18 | **Binární klasifikace** | `ahd_yes` (1 = onemocnění srdce, 0 = zdravý) | Kardiologický dataset: `age`, `sex_male`, `chestpain_*`, `restbp`, `chol`, `fbs_yes`, `maxhr`, `oldpeak`, `ca`. Medicínská diagnostika, kde je klíčová metrika Recall/Sensitivity (minimalizace FN) a interpretovatelnost rozhodovacích pravidel. |
| **`kc_house_data_preprocessed.csv`** | 21 613 | 20 | **Regrese** | `price` (USD) | Nemovitosti v King County (Seattle): `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`, `waterfront`, `grade`, `zipcode`, `lat`, `long`. Nelineární geografické a strukturální vztahy – benchmark pro gradient boosting. |

---

## 📅 Blok 2: Den 4 – Zpracování přirozeného jazyka (NLP)
**Témata Dne 4:** Text Preprocessing, NLTK, spaCy, Bag of Words, TF-IDF, Word2Vec, sekvenční modely, klasifikace textu a analýza sentimentu.

| Datový soubor | Řádky | Sloupce | Typ úlohy | Cílová proměnná (Target) | Struktura a kontext |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **`imdb_reviews.csv`** | $\approx 25\,000+$ | Vícesloupcový | **Klasifikace textu / Sentiment** | `sentiment` (`positive` / `negative`) | Surový filmový korpus z IMDB s HTML značkami (`<br />`), interpunkcí a nestejnou délkou recenzí pro výuku end-to-end pipeline. |
| **`imdb_reviews_preprocessed_1.csv`** | 10 000 | 2 | **Klasifikace sentimentu** | `sentiment` | `review_cleaned`: Očištěný text zbavený HTML a speciálních znaků připravený pro tokenizaci a vektorizaci. |
| **`imdb_reviews_lemmatized.csv`** | 10 000 | 2 | **Klasifikace sentimentu** | `sentiment` | `review_lemmatized`: Pokročile předzpracovaný text, kde jsou slova převedena na lemmata (pomocí NLTK/spaCy), ideální pro trénování TF-IDF a klasifikátorů. |

---

## 🏠 Blok 3: Domácí úkoly (Homework Session 2)
Ucelená paleta reálných byznysových úloh z různých domén pokrývající regresi, klasifikaci i NLP:

| Datový soubor | Řádky | Sloupce | Typ úlohy | Cílová proměnná | Doména a analytická výzva |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **`auto_mpg.csv`** | 398 | 9 | **Regrese** | `mpg` (Spotřeba v mílích na galon) | Automobilový průmysl: Vliv válců, objemu motoru, výkonu (`horsepower`), hmotnosti a roku výroby na palivovou efektivitu. Klasický benchmark fyzikálních závislostí. |
| **`calories_exercise_data.csv`** | 15 000 | 9 | **Regrese** | `calories` (Spálené kalorie) | Fitness a zdraví: `gender`, `age`, `height`, `weight`, `duration`, `heart_rate`, `body_temp`. Odhad energetického výdeje z biometrických senzorů. |
| **`car_data.csv`** | 19 237 | 18 | **Regrese** | `price` (Cena ojetého vozu) | Trh ojetých automobilů: Značka, model, rok výroby, typ paliva, stav tachometru, počet airbagů. Řešení kategorických proměnných s vysokou kardinalitou. |
| **`diabetes.csv`** | 768 | 9 | **Klasifikace** | `Outcome` (0/1) | Pima Indians Diabetes: Opakování a benchmark pokročilých modelů proti baseline modelům ze Session 1 (k-NN, LogReg, SVM). |
| **`mcdonalds_reviews.csv`** | 37 131 | 3 | **NLP / Analýza zpětné vazby** | `rating` (1–5 hvězdiček) | Gastronomie: Tisíce reálných zákaznických recenzí řetězce McDonald's. Extrakce stížností (rychlost, teplota jídla, čistota) a vícedenní klasifikace sentimentu. |
| **`sonar.csv`** | 208 | 61 | **Binární klasifikace** | Sloupec 60 (`R` = Skála, `M` = Kovová mina) | Vojenská akustika: 60 frekvenčních pásem sonaru odražených od objektu na dně moře. Extrémní poměr dimenzí vůči počtu pozorování ($p=60, n=208$). |
| **`titanic_data.csv`** | 1 309 | 14 | **Klasifikace** | `survived` (0/1) | Klasická historická data přežití na Titaniku: Chybějící věk, kabiny, socioekonomický status (`pclass`, `fare`). |

---

## 🎯 Strategické "Mental Notes" pro implementaci:
1. **Den 3 (Ansámbly):** 
   - Pro regresi využijeme **Diamonds** a **King County Housing** k jasné demonstraci, proč samotný rozhodovací strom zaostává za Random Forestem a XGBoostem o 10–25 % v $R^2$.
   - Pro kardiologický dataset **Heart Data** (303 pacientů) budeme optimalizovat **Recall** a analyzovat matici záměn (cena přehlédnutého infarktu).
2. **Den 4 (NLP):**
   - Využijeme připravenou trojici IMDB datasetů pro srovnání vlivu jednotlivých fází preprocessingu (surový text vs. čistý text vs. lemmatizovaný text) na výsledné F1-skóre klasifikátoru.
3. **Homework:**
   - Datové sady pokrývají pestré spektrum úloh: od akustického sonaru (vysoká dimenzionalita) přes rozsáhlé texty McDonald's až po kalorie ze sportovních hodinek.
