# 🔬 Data Science & Machine Learning Portfolio (Streamlit App)

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.24+-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![Coders Lab](https://img.shields.io/badge/Vychází_z-Coders_Lab_CZ-00C48C?style=for-the-badge)](https://coderslab.cz/cz/)
[![AI Orchestration](https://img.shields.io/badge/AI_Orchestration-Google_Antigravity-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://deepmind.google/)

> Osobní zápisky z kurzu Coders Lab rozšířené do interaktivního portfolia: regrese, klasifikace, pokročilé ML, NLP, neřízené učení, MLOps a vlastní projekt RAG.

**Veřejná aplikace:** [Data Science & ML Portfolio](https://mldatasciencecoderslab.streamlit.app/) · **Závěrečný projekt:** [ML Projekt Librarian](https://mldatasciencecoderslab.streamlit.app/ml_librarian_lecturer)

---

## 🎓 Uznání autorství, AI orchestrace & Poděkování (Attribution)

### 🏫 Vzdělávací základ & Kurikulum
Základní pedagogická struktura, datové sady a cvičení v tomto projektu vycházejí z materiálů a sylabu akreditovaného kurzu **Data Science / Machine Learning** vzdělávací IT akademie **[Coders Lab Česká republika](https://coderslab.cz/cz/)**.

### 🤖 Realizace, AI orchestrace a role autora
Tento repozitář byl koncipován, architektonicky navržen a softwarově implementován formou **pokročilého párového programování (Pair Programming) a agentní AI orchestrace v prostředí [Google Antigravity (Antigravity IDE)](https://deepmind.google/)**.

- **Synergie člověk + AI:** Uživatel definuje strategické zadání, věcné mantinely a ověřuje pedagogickou integritu; AI agent Antigravity obstarává end-to-end softwarové inženýrství, matematické formulace v KaTeX, konstrukci Scikit-learn pipeline, diagnostiku chyb, generování interaktivních komponent v Plotly/Streamlit a automatické spouštění Jupyter sešitů.
Samostatný závěrečný projekt **ML Librarian** vznikal AI orchestrací ve spolupráci s **GPT 6**, s průběžnou oponenturou **Gemini Argon a Claude 5**. Autor průběžně konzultoval a ověřoval jednotlivé kroky implementačního plánu, testoval aplikaci a rozhodoval o prioritách. Tato spolupráce popisuje proces vývoje, nikoli nezávislou přejímku kvality.

- **Přidaná hodnota nad rámec původních osnov:**
  - Moderní **Streamlit webová aplikace** s interaktivními vizualizacemi v Plotly a hierarchickou navigací.
  - **What-If inferenční simulátory** napojené na reálně natrénované Scikit-learn modely v reálném čase.
  - **Expertní diagnostika:** Gauss-Markovovy teorémy, analýza multikolinearity (VIF), prostorová mapa odchylek v cenách nemovitostí v Seattlu, Cost-Complexity Pruning rozhodovacích stromů a geometrický rozbor metrik k-NN.
  - **Rigorózní metodické rozbory:** eliminace Data Leakage, správná práce s křížovou validací (`GridSearchCV`, `StratifiedKFold`) a ochrana před přetrénováním.

---

## 🌟 O projektu & Architektura aplikace

Streamlit původně sloužil jako můj **záznamník a zápisky z kurzu**. Sdílel jsem jej s lektorem a kolegy; opakované využívání i dalšími účastníky mě potěšilo. Dnes pokrývá všech pět kurzových dnů, domácí úlohy i navazující projekt.

Většina témat propojuje teorii, výsledky zadání, expertní rozbor a Jupyter notebook. Základní členění:

1. **📖 Teoretický základ:** Matematické formulace ($y = \mathbf{X}\boldsymbol{\beta} + \varepsilon$, ztrátové funkce, regularizační pokuty), geometrické interpretace a metodické poznámky.
2. **🎯 Školní výsledky zadání:** Přesné splnění úloh z kurzu včetně natrénování modelů a vyhodnocení základních metrik.
3. **🔬 Expertní analýza & Kritika:** Hlubší vhled za hranice základních zadání (analýza časového posunu u nemovitostí, gemologické paradoxy u diamantů, Rungeův jev u polynomů, prořezávání stromů).

| Část kurzu | Obsah |
|---|---|
| Prework | Základy Data Science, statistika, Python stack, příprava dat a Scikit-learn. |
| Den 1 · regrese | OLS, metriky, regularizace, polynomy, rozhodovací stromy a ansámbly. |
| Den 2 · klasifikace | k-NN, klasifikační metriky, logistická regrese, stromy a SVM. |
| Mezi dny 2 a 3 | Hyperparametry, křížová validace, Grid/Random/Bayes search; úlohy s betonem a diabetem. |
| Den 3 · pokročilé ML | Bagging, Random Forest, boosting/XGBoost, neuronové sítě a diagnostické laboratoře. |
| Den 4 · NLP | Předzpracování, NLTK/spaCy, BoW, TF-IDF, Word2Vec/Gensim, Transformer a BERT. |
| Domácí úlohy Session 2 | Titanic, ceny aut, diabetes, kalorie, sonar, Auto MPG a Word2Vec nad daty McDonald's. |
| Den 5 · neřízené učení | PCA, K-Means, DBSCAN, LOF, Isolation Forest a jejich srovnání. |
| Závěr kurzu | Celkové shrnutí, produkční nasazení/MLOps, další rozvoj a trendy AI. |
| Moderní ML 2026 | Výukové simulátory RAG, srovnání s LoRA a promptingem, agenti a produkční šablona. |
| ML Projekt Librarian | Vlastní RAG nad materiály kurzu, pravý dock a příběh vývoje pro lektora. |

Výukový **RAG simulátor** v části Moderní ML 2026 a **ML Librarian** jsou dva různé moduly: první vysvětluje princip na ukázkové znalostní bázi, druhý vyhledává ve zpracovaných materiálech kurzu.

---

## 📊 Vybrané datasety a výsledky experimentů

Níže jsou vybrané výsledky původních kurzových experimentů, nikoli úplný přehled všech modelů nebo nový nezávislý benchmark. Interpretace závisí na konkrétní přípravě dat, rozdělení a metrice; postupy a další výsledky jsou v příslušných noteboocích a analytických stránkách.

### 1. Nemovitosti v King County, Seattle (`kc_house_data.csv`)
- **21 613 prodejů domů** z let 2014–2015.
- Cíl: Predikce prodejní ceny na základě plochy, lokality, stavu a počtu pokojů.
- **Baseline OLS:** $R^2 \approx 0.695$, $\text{RMSE} \approx 203\,150\text{ USD}$
- **Optimální rozhodovací strom (GridSearchCV):** $R^2 = 0.793$, $\text{RMSE} = 176\,810\text{ USD}$  
  *(Díky schopnosti ohraničit prestižní čtvrti pomocí souřadnic `lat` a `long` strom snížil chybu RMSE o téměř 35 000 USD).*

### 2. Diamanty (`diamonds.csv`)
- **53 908 certifikovaných diamantů** hodnocených dle gemologických standardů 4C (Carat, Cut, Color, Clarity).
- Cíl: Predikce ceny a zachycení nelineárního růstu hodnoty s hmotností.
- **Lineární OLS:** $R^2 = 0.9095$, $\text{RMSE} = 1\,172.5\text{ USD}$
- **Polynom 2. stupně (Kvadratický OLS):** $R^2 = 0.9655$, $\text{RMSE} = 723.7\text{ USD}$
- **Polynom 3. stupně (Surový OLS):** $R^2 = 0.8591$, $\text{RMSE} = 1\,463.0\text{ USD}$ *(Kolaps variance kvůli multikolinearitě 219 členů!)*
- **Polynom 3. stupně + Ridge ($\alpha=100$):** $R^2 = 0.9744$, $\text{RMSE} = 623.2\text{ USD}$ *(Záchrana modelu regularizací)*
- **Optimální rozhodovací strom (GridSearchCV):** $R^2 = 0.9785$, $\text{RMSE} = 587.0\text{ USD}$, $\text{MAE} = 307.3\text{ USD}$  
  *(Dokonale zachycuje psychologické cenové skoky u kulatých karátů `carat >= 1.00`).*

### 3. Antarktičtí tučňáci Palmer Penguins (`penguins_size.csv`)
- **344 tučňáků** tří druhů (*Adelie*, *Chinstrap*, *Gentoo*) ze souostroví Palmer.
- Cíl: Multitřídní klasifikace druhu na základě morfologie (zobák, ploutev, hmotnost, ostrov, pohlaví).
- **Školní baseline (bez škálování, $k=5$):** $\text{Accuracy} \approx 80.6\,\%$ *(Hmotnost v gramech tvořila $99.98\,\%$ celkové vzdálenosti!)*
- **Plný Pipeline s OneHot kódováním ostrova a pohlaví:** $\text{Accuracy} = 99.0\,\%$, $\text{CV} = 100.0\,\%$

### 4. Biomechanika bederní páteře (`lumbar_data.csv`)
- **310 pacientů** (100 zdravých `Normal`, 210 s diagnózou `Abnormal`: výhřez ploténky *Hernia* a posun obratle *Spondylolisthesis*).
- Cíl: Klinická klasifikace stavu páteře na základě 6 anatomických úhlů pánve.
- **Školní model ($k=5$, $L_2$ normalizace):** $\text{Accuracy} = 73.08\,\%$, $\text{Recall} = 80.70\,\%$, $\text{ROC-AUC} = 0.8208$.
- **Optimální model se stratifikací ($k=8$):** $\text{Accuracy} = 88.46\,\%$, $\text{Recall} = 88.68\,\%$.

---

## 🎛️ Hlavní interaktivní funkce (Day 1 Extras)

- **🎛️ What-If simulátor inference:** Kalkulátor cen domů i diamantů volající **skutečně natrénované Scikit-learn pipeliny** v reálném čase.
- **🗺️ Geografická mapa nemovitostí v Seattlu:** Interaktivní mapa 1 500 domů s barevným vyznačením odchylek v odhadu cen ($y - \hat{y}$) odhalující, kde přesně OLS selhává a jak strom lokalizuje bohaté pobřežní oblasti.
- **🔬 Diagnostika Gauss-Markov:** Ověření předpokladů pro vlastnost BLUE (Best Linear Unbiased Estimator) a interaktivní analýza multikolinearity přes VIF faktor.
- **🌲 Cost-Complexity Pruning (`ccp_alpha`):** Vizuální sledování prořezávání rozhodovacího stromu od masivního přeučení ($>36\,000$ listů) až k optimálnímu generalizovanému modelu.
- **🌳 Průvodce výběrem modelu:** Interaktivní rozhodovací diagram v SVG pro volbu správného algoritmu v byznysové praxi.
- **📥 Scikit-learn Cheatsheet:** Kompletní tahák v PDF ke stažení vygenerovaný přes ReportLab.

---

## 🧪 Navazující interaktivní laboratoře

- **Pokročilé ML:** kreslení a rozpoznávání MNIST, 2D neuronová síť, backpropagation, rozměry vrstev, variabilita baggingu, rezidua boostingu a nákladové prahování.
- **NLP:** práce s českým i anglickým textem, tokenizace a lemmatizace, vektorizace, sentiment, slovní vnoření a vysvětlení self-attention.
- **Neřízené učení:** redukce dimenzionality, segmentace zákazníků, hustotní shlukování a detekce odlehlých pozorování.
- **Závěr a MLOps:** přechod od experimentu k nasazení, provozní úskalí, RAG a agentní architektury.

Některé moduly používají předpočtené výsledky nebo uložené modely; výuková simulace není automaticky nový trénink modelu. Dostupnost konkrétních funkcí závisí na příslušných datech a závislostech.

---

## 🚀 Jak aplikaci spustit lokálně (Quickstart)

Repozitář připíná Python **3.11** v `.python-version`. Níže uvedený postup spouští aplikaci; reprodukce všech trénovacích notebooků může vyžadovat další balíčky a vstupní data.

### 1. Klonování repozitáře
```bash
git clone https://github.com/Kilitar/Coderslab-DataScience.git
cd Coderslab-DataScience
```

### 2. Vytvoření virtuálního prostředí
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalace závislostí
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Spuštění Streamlit aplikace
```bash
streamlit run app.py
```
Aplikace se automaticky otevře ve vašem výchozím webovém prohlížeči na adrese `http://localhost:8501`.

---

## 📂 Struktura repozitáře

```text
Coderslab-DataScience/
├── app.py                              # Navigace kurzu a společný dock
├── requirements.txt                    # Závislosti aplikace včetně Librariana
├── .python-version                     # Požadovaná verze Pythonu
├── .streamlit/                         # Konfigurace; skutečné secrets nepatří do Gitu
├── 00_Prework/                         # Úvod, statistika a příprava dat
├── 01_Regression/                      # Regrese a související domácí úlohy
├── 02_Classification/                  # Klasifikace a diagnostika
├── 03_Advanced_ML_Neural_Networks/      # Tuning, ansámbly a neuronové sítě
├── 04_NLP/                             # Textová data, vektorizace a transformery
├── 04_Homework/                        # Navazující domácí úlohy
├── 05_Unsupervised_Learning/            # PCA, shlukování a detekce anomálií
├── 06_Final_Project/                   # Shrnutí, MLOps a moderní ML
├── views/                              # Stránky, laboratoře a prohlížeč notebooků
├── components/ml_librarian/            # Verzovaný export samostatného RAG projektu
├── data/                               # Sdílená data a grafické podklady
├── models/                             # Uložené modelové artefakty
└── docs/                               # Projektové a integrační podklady
```

Konkrétní soubory a jejich návaznosti jsou dostupné v jednotlivých složkách a v navigaci aplikace. Dostupnost autorských kurzových materiálů se řídí jejich vlastními podmínkami.

---

## 🛠️ Použité technologie a knihovny

- **Aplikace a vizualizace:** Streamlit, Plotly, Pandas a NumPy; Matplotlib/Seaborn pro analytické a notebookové podklady.
- **Modelování a diagnostika:** Scikit-learn, XGBoost, SciPy, Statsmodels, Hyperopt a Joblib.
- **NLP:** NLTK, spaCy a navazující výukové moduly pro slovní vnoření a transformery.
- **ML Librarian:** Sentence Transformers, vícejazyčné E5 embeddingy, TF-IDF, cross-encoder a volitelné generování přes Groq API.
- **Příprava materiálů:** pypdf a nbformat; trénovací a rozšiřující notebooky mají vlastní požadavky.
- **Vývoj:** Google Antigravity a AI spolupráce popsaná výše.

Závislosti aplikace určuje [requirements.txt](requirements.txt), rozšíření RAG [requirements-librarian.txt](components/ml_librarian/requirements-librarian.txt).

---

## 📜 Licence & Podmínky užití

Tento projekt je určen pro vzdělávací a studijní účely v rámci studia Data Science. Původní koncepty úloh náleží vzdělávací akademii **[Coders Lab](https://coderslab.cz/cz/)**. Kód implementace aplikace, interaktivní moduly a doplňující analýzy jsou volně k dispozici pod licencí MIT.


## 📚 Integrovaný ML Librarian

Samostatný závěrečný projekt rozšiřuje již existující studijní prostředí. Cílem bylo prakticky použít znalosti z kurzu na vlastní aplikaci místo původního kurzového příkladu a samostatně se naučit další postupy.

### Použití v aplikaci

1. Na kterékoli stránce otevři tlačítko **📚 Librarian** vpravo dole.
2. Zadej otázku česky, anglicky nebo polsky.
3. Ve **V čem hledat** zvol checkboxy typu materiálu a původu zdroje; výchozí výběr zahrnuje vše.
4. Prohlédni nalezené úryvky, jejich původ a relevanci. Odkaz **Otevřít v kurzu** vede na přímo přiřazenou stránku; **Související stránka kurzu** označuje tematickou vazbu, nikoli shodný text.
5. Pokud je dostupná konfigurace a opora v materiálech, lze výslovně odeslat otázku a nejvýše pět úryvků službě Groq a požádat o odpověď s citacemi. Návštěvník nevyplňuje vlastní klíč ani model.

Odeslaný dotaz, výsledky a odpověď se uchovávají v aktuální relaci i při přechodu mezi lekcemi; obnovení stránky může relaci ukončit. Skóre relevance není pravděpodobnost správnosti a doslovná citace sama nezaručuje správný výklad.

### Data a provoz

- Kurzový modul obsahuje ověřovaný runtime balíček se zpracovanými dokumenty a indexem. Při prvním otevření připraví data a připnuté vyhledávací modely; studený start může trvat déle.
- Vyhledávání nevyžaduje Groq API klíč. Generování vyžaduje serverové nastavení provozovatele (`GROQ_API_KEY`, volitelně `GROQ_MODEL`); skutečné hodnoty nepatří do repozitáře.
- Free tier má sdílené limity. Při chybě generování nebo vyčerpání kvóty zůstávají nalezené zdroje dostupné. Úspěšné lokální testy nejsou potvrzením kvality všech veřejných odpovědí.
- Veřejná aplikace **neobsahuje původní kurzová PDF**, pouze jejich zpracované textové úryvky. Originály jsou autorským majetkem Coders Lab a jsou určeny studentům a absolventům kurzu; proto zde není veřejný odkaz na jejich stažení.

### Dokumentace pro lektora

Sekce **36. Projekt: ML Librarian → ML Projekt Librarian** vypráví příběh od záměru přes plánování a experimenty po poučení. Obsahuje původní roadmapu, checklist, implementační plán, slepé uličky, odkazy na kurzové metody, čtyři notebooky ke stažení a odkazy do Google Colab. Výsledky a známá omezení jsou odlišeny od nezávislé přejímky kvality.

Zdrojový vývoj Librariana probíhá v [samostatném repozitáři ML_Librarian](https://github.com/Kilitar/ML_Librarian). Složka `components/ml_librarian/` je verzovaný export, nikoli další nezávislá implementace. Historický integrační postup je v [docs/ML_LIBRARIAN_INTEGRATION.md](docs/ML_LIBRARIAN_INTEGRATION.md); jeho starší části mohou popisovat již nahrazenou navigaci nebo konfiguraci. Aktuální implementaci určují `app.py` a export komponenty.

## 🙏 Poděkování

Děkuji **Coders Lab** za kurz a inspiraci k praktickým experimentům a **Monice** za vedení kurzu a předávání zkušeností. Děkuji také kolegům z kurzu, kteří aplikaci používali a dávali zpětnou vazbu.
