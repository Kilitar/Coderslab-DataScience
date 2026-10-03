# Prework Session 2: Úvod do zpracování přirozeného jazyka (Introduction to NLP)

> **Cíl modulu:** Porozumět základním principům zpracování přirozeného jazyka (**Natural Language Processing – NLP**), výzvám při reprezentaci lidské řeči v počítačovém světě, ucelenému řetězci předzpracování textu (*NLP Pipeline*), technikám vektorizace a moderním průmyslovým aplikacím (od sentimentové analýzy po generativní modely a sémantické vyhledávání).

---

## 1. Co je NLP a proč je klíčovou disciplínou Data Science?

**Zpracování přirozeného jazyka (Natural Language Processing – NLP)** je interdisciplinární podobor informatiky, umělé inteligence a počítačové lingvistiky. Umožňuje počítačům číst, dešifrovat, chápat a smysluplně generovat lidský jazyk.

Za posledních 10 let se NLP stalo jedním z nejrychleji expandujících odvětví AI. Zásluhu na tom má:
1. **Masivní nárůst nestrukturovaných textových dat:** Více než 80 % firemních dat je uloženo v nestrukturované podobě (emaily, zákaznické recenze, smlouvy, technická dokumentace, přepisy hovorů).
2. **Prudký rozvoj hardwaru (GPU/TPU):** Umožnil trénování rozsáhlých neuronových sítí na miliardách slov.
3. **Architektonická revoluce (Transformers, 2017+):** Mechanismus *Self-Attention* překonal omezení starších rekurentních sítí a položil základ moderním velkým jazykovým modelům (LLM jako GPT, Claude, Gemini, LLaMA).

---

## 2. Historický kontext: Jak to všechno začalo?

Pro lidi je jazyk přirozeným nástrojem komunikace, který si osvojují odmala. Z pohledu počítače však přirozený jazyk představuje obrovskou výzvu:

### A. Nekonečná kombinatorika a pravidla
V lidském jazyce existuje prakticky nekonečný počet způsobů, jak poskládat slova do věty. Některé kombinace jsou gramaticky správné a dávají smysl, jiné nikoliv. Člověk to rozpozná intuitivně, ale **počítači nelze jednoduše předat slovník se všemi možnými větami**, protože jazyk je otevřený a neomezený systém.

### B. Analogie s učením dětí
Průkopníci NLP (např. Noam Chomsky v lingvistice a první tvůrci pravidlových systémů v 50. a 60. letech 20. století) navrhli dekompozici:
- Věta se rozloží na jednotlivá slova (atomy jazyka).
- Dítě se nejprve učí jednotlivá izolovaná slova (*„máma“*, *„auto“*), postupně z nich skládá dvouslovná spojení a teprve později ovládne syntax a gramatiku.
- Stejným způsobem postupují i tradiční algoritmy NLP.

### C. Problém polysémie, kontextu a sémantiky
Jedno slovo může mít mnoho zcela odlišných významů v závislosti na kontextu (**polysémie** a **homonymie**):
- Příklad ze zadání: **„Bar“**
  1. *Pohostinské zařízení:* „Šli jsme večer na skleničku do baru.“
  2. *Módní doplněk:* Ozdobná spona držící límeček košile a uzel kravaty (*collar bar*).
  3. *Běžný předmět:* Čokoládová tyčinka (*chocolate bar*), mříže na okně (*window bars*), tabulka mýdla (*bar of soap*).
  4. *Fyzikální jednotka:* Tlak 1 bar.
  5. *Právní význam:* Advokátní komora (*passed the Bar exam*).

Aby počítač dokázal rozlišit význam, nestačí jen syntaktický rozbor slovních druhů (Part-of-Speech – POS), ale je nutné modelovat **sémantiku** (vztah mezi slovy, kontextem a reálným světem).

---

## 3. Zásady budování NLP aplikací: Ucelená pipeline

Vybudování efektivního NLP systému probíhá v systematickém řetězci kroků (*NLP Pipeline*):

```
Surový text ──► Čištění (Regex) ──► Tokenizace ──► Stopwords ──► Normalizace ──► Lemmatizace/Stemming ──► Vektorizace ──► ML Model ──► Produkce
```

### Krok 1: Definice cíle aplikace
Jasná specifikace úlohy:
- Je cílem **klasifikace** (detekce spamu, sentiment recenze)?
- Jde o **extrakci informací** (pojmenované entity NER v lékařských zprávách)?
- Nebo o **generování/transformaci** (strojový překlad, sumarizace textu, chatbot)?

### Krok 2: Volba datových zdrojů
Získání reprezentativního korpusu:
- Scraping webových stránek a recenzí (např. Google Maps, TripAdvisor, Heureka).
- Odborné články, interní firemní databáze CRM, přepisy hovorů zákaznické podpory.
- Anotace dat (Labeling) pro učení s učitelem.

### Krok 3: Předzpracování a standardizace textu (*Text Preprocessing*)
Lidský text je plný překlepů, emotikonů, HTML tagů, různé interpunkce a nekonzistentních tvarů:

1. **Čištění dat (*Data Cleaning*):**
   - Odstranění HTML značek (`<p>`, `<br>`), URL adres a speciálních znaků pomocí regulárních výrazů (**Regex**).
   - Ošetření čísel a interpunkce podle typu úlohy.
2. **Tokenizace (*Tokenization*):**
   - Rozdělení souvislého textu na menší jednotky – **tokeny** (nejčastěji slova, interpunkční znaménka nebo podslova *subwords* jako BPE/WordPiece).
   - *Příklad:* `"Data Science je skvělá!"` $\to$ `["Data", "Science", "je", "skvělá", "!"]`.
3. **Odstranění stop-slov (*Stopwords Removal*):**
   - Slova s vysokou frekvencí výskytu, která nenesou specifický sémantický obsah (např. *a, i, v, na, je, ten, byla, the, and, is, at*).
   - Jejich vyřazením se sníží dimenzionalita slovníku a model se soustředí na nosná klíčová slova.
4. **Normalizace textu (*Normalization*):**
   - Převod na malá písmena (*lower-casing*), odstranění diakritiky (pokud je to žádoucí), unifikace mezer.
   - Zabraňuje tomu, aby počítač vnímal `"Apple"`, `"apple"` a `"APPLE"` jako tři různá slova.
5. **Stemming vs. Lemmatizace:**
   - **Stemming (Ořezávání na kmen):** Heuristický algoritmus (např. *Porter Stemmer*), který mechanicky odsekává koncovky slov. Může vyprodukovat neexistující slova (*„studying“* $\to$ *„studi“*). Je velmi rychlý.
   - **Lemmatizace (Převod na základní slovníkový tvar – lemma):** Lingvisticky podložená metoda využívající slovník a morfologickou analýzu slovního druhu (POS). *„Better“* $\to$ *„good“*, *„běžela“* $\to$ *„běžet“*, *„corpora“* $\to$ *„corpus“*. Je přesnější, ale výpočetně náročnější.

---

## 4. Vektorizace textu: Převod slov na čísla (*Text Vectorization*)

Počítače a algoritmy strojového učení (lineární modely, stromy i hluboké sítě) neumí pracovat přímo s písmeny a řetězci. Text musíme reprezentovat jako číselné vektory $\mathbf{x} \in \mathbb{R}^d$.

| Metoda vektorizace | Princip fungování | Výhody | Nevýhody & Omezení |
| :--- | :--- | :--- | :--- |
| **Bag of Words (BoW / CountVectorizer)** | Počítá četnost výskytu každého slova ze slovníku v daném dokumentu. | Extrémně jednoduché, rychlé, dobře interpretovatelné. | Ignoruje slovosled, vytváří obrovské a řídké (*sparse*) matice. |
| **TF-IDF (Term Frequency – Inverse Document Frequency)** | Váží četnost slova v dokumentu ($TF$) jeho vzácností v celém korpusu ($IDF$). | Potlačuje běžná slova, zvýrazňuje specifická klíčová témata dokumentu. | Stále ignoruje sémantický kontext a synonyma (např. *pes* a *hafík* jsou 2 nezávislé dimenze). |
| **Word2Vec / FastText / GloVe** | Statické husté vektorové vnoření (*Dense Word Embeddings*, např. 300 dimenzí) naučené z kontextových oken. | Zachycuje sémantiku a analogie: $\vec{v}(\text{Král}) - \vec{v}(\text{Muž}) + \vec{v}(\text{Žena}) \approx \vec{v}(\text{Královna})$. | Statické – slovo má pouze jeden vektor bez ohledu na význam ve větě (*banka* finanční vs. říční). |
| **Kontextová vnoření (BERT, RoBERTa, ModernBERT, LLM)** | Dynamická reprezentace generovaná mechanismem *Self-Attention* v hlubokých transformerech. | Vektor slova se dynamicky mění podle celého kontextu věty. Špičkový výkon (SOTA). | Vysoká výpočetní náročnost, nutnost GPU akcelerace pro trénování a inference. |

### Vzorec TF-IDF:
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
Kde:
$$\text{TF}(t, d) = \frac{f_{t, d}}{\sum_{t' \in d} f_{t', d}}, \quad \text{IDF}(t, D) = \ln\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$

---

## 5. Volba algoritmu, evaluace a produkční nasazení

1. **Výběr algoritmu:**
   - *Jednoduché lineární modely:* Multinomiální Naive Bayes nebo Logistická regrese nad TF-IDF (výborný baseline pro spam a sentiment, trénování v řádu sekund).
   - *Gradient Boosting:* XGBoost / LightGBM nad tabulkovými příznaky a TF-IDF.
   - *Sekvenční neuronové sítě:* LSTM, GRU (historicky pro strojový překlad a zachycení dlouhodobých závislostí).
   - *Transformery:* Předtrénované modely z knihovny `transformers` (Hugging Face) fine-tunované na konkrétní doménu.
2. **Evaluace modelu:**
   - Klasifikace: **F1-Score (Macro/Weighted)**, Precision, Recall, Confusion Matrix (zejména při nevyvážených třídách, např. detekce nenávistných projevů).
   - Generování a překlad: **BLEU**, **ROUGE**, **BERTScore**, **Perplexity**.
3. **Produkční nasazení a MLOps:**
   - Integrace do UI: REST API (FastAPI), interaktivní webové rozhraní (Streamlit).
   - Škálování: Asynchronní fronty úloh (Celery/RabbitMQ), optimalizace latence (ONNX Runtime, kvantizace INT8).
   - Monitoring: Sledování *Data Drift* (změna slovní zásoby a slangu uživatelů) a *Concept Drift*.

---

## 6. 11 praktických oblastí využití NLP v reálném světě

Podle učebního plánu kurzu nachází NLP uplatnění v těchto stěžejních odvětvích:

1. **Chatboti a virtuální asistenti:**
   - Konverzační boti odpovídající na dotazy zákazníků (dostupnost zboží, reklamace, stav objednávky).
   - *Příklad:* Asistent Max na webu Orange, zákaznická podpora e-shopů.
2. **Analýza sentimentu (*Sentiment Analysis*):**
   - Automatické určení tónu textu (pozitivní, neutrální, negativní) v recenzích na Trustpilot, Google Reviews či sociálních sítích.
3. **Strojový překlad (*Machine Translation*):**
   - Překlad mezi desítkami jazyků v reálném čase.
   - *Příklad:* Google Translate, DeepL, automatický překlad tweetů na platformě X.
4. **Analýza zákaznické zpětné vazby:**
   - Shlukování témat ve stížnostech zákazníků (např. „studené jídlo“, „pomalá obsluha“ v recenzích restaurací).
5. **Detekce spamu a phishingu:**
   - Analýza textových vzorů a podezřelých odkazů v e-mailech v poštovních klientech (Gmail, Outlook).
6. **Vyhledávání informací a sémantický search (*Information Retrieval & RAG*):**
   - Vektorové srovnání dotazu uživatele s bází dokumentů pomocí kosinové podobnosti ($Cosine\ Similarity$).
   - Uživatel najde správnou odpověď, i když použije jiná slova než autor dokumentu.
7. **Analýza sociálních sítí a trendy:**
   - Sledování nálady kolem značky, včasná detekce PR krizí, identifikace virálních témat.
8. **Rozpoznávání řeči (Speech-to-Text / ASR):**
   - Převod mluveného slova na text.
   - *Příklad:* Apple Siri, Google Voice Search, automatické generování titulků na YouTube.
9. **Automatická sumarizace textu (*Text Summarization*):**
   - *Extraktivní:* Výběr nejreprezentativnějších vět z článku.
   - *Abstraktivní:* Parafrázování a vytvoření stručného souhrnu vlastními slovy pomocí LLM.
10. **Rozpoznávání pojmenovaných entit (Named Entity Recognition – NER):**
    - Detekce a označení jmen osob (`PER`), lokalit (`LOC`), organizací (`ORG`), časových údajů (`DATE`) a měn.
    - Klíčové pro automatické zpracování faktur a lékařských zpráv.
11. **NLP ve vzdělávání:**
    - Automatická kontrola gramatiky (Grammarly), hodnocení slohových prací, adaptivní kvízy a jazykové aplikace (Duolingo).

---

## 7. Shrnutí a pohled do Session 2

Zpracování přirozeného jazyka představuje most mezi lidskou kognicí a výpočetním výkonem počítačů. V průběhu **Session 2 (Dny 3 a 4)** se posuneme od základních statistických metod (TF-IDF a klasifikátorů) k pokročilému modelování sekvencí a práci s reálnými textovými datasety (jako jsou recenze filmů IMDB a zákaznické ohlasy McDonald's).
