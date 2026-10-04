# Den 4: Ucelené shrnutí zpracování přirozeného jazyka (NLP) & Závěr
> **Zdrojový podklad**: Coderslab Data Science – Day 4: *Day 4 summary*  
> **Téma**: Příprava textu, vektorizace, sémantická vnoření, transformery a moderní LLM ekosystém.

---

## 1. Úvod do Dne 4: Proč a jak zpracováváme text?

Počítače a algoritmy strojového učení nedokáží provádět matematické operace přímo se slovy a písmeny. Cílem celého Dne 4 bylo projít kompletní cestu:
1. **Očištění a standardizace surového textu** na konzistentní, bezchybná data.
2. **Transformace textu do číselné reprezentace (vektorizace)**.
3. **Pochopení sémantických vztahů a kontextu** – od jednoduchého počítání slov až po obousměrné transformery.

---

## 2. Čtyři klíčové etapy vývoje NLP v kurzu

### 1. Etapa: Předzpracování textových dat (Text Preprocessing)
* **Tokenizace (Tokenization)**: Rozdělení textu na základní stavební jednotky – tokeny (slova, interpunkce, subwords).
* **Normalizace (Normalization)**: Převod na malá písmena (`lowercase`), odstranění HTML značek (`BeautifulSoup`), čištění regulárními výrazy (`re.sub`).
* **Odstranění stop-slov (Stopwords)**: Vyřazení vysoce frekventovaných, avšak sémanticky prázdných slov (*"the", "is", "a", "a", "že", "s"*).
* **Lemmatizace vs. Stemming**:
  * *Stemming* (Porter): Hrubé ořezání koncovky slova často vedoucí k neexistujícímu kmeni (*"univers"*).
  * *Lemmatizace* (spaCy / WordNet): Převod na gramaticky správný základní tvar (lemma) s využitím slovníku a slovních druhů (*"better"* $\rightarrow$ *"good"*).

---

### 2. Etapa: Klasické frekvenční reprezentace (BoW & TF-IDF)
* **Bag of Words (CountVectorizer)**:
  * Reprezentace textu jako "pytle slov" – vektor četností výskytů.
  * *Omezení*: Ztrácí pořadí slov ve větě, ignoruje gramatiku a produkuje obrovské řídké matice (*sparse matrices*).
* **TF-IDF (TfidfVectorizer)**:
  * Term Frequency – Inverse Document Frequency.
  * Zvyšuje váhu slov, která jsou klíčová pro konkrétní dokument, ale vzácná v celém korpusu.
  * Formula: $\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \log\left(\frac{N}{\text{DF}(t)}\right)$.

---

### 3. Etapa: Distribuovaná sémantická vnoření (Word2Vec)
* Vyvinuto týmem Tomáše Mikolova v Google (2013).
* Přechod od řídkých tisícerozměrných vektorů k **hustým, nízkorozměrným vektorům** (např. 100D až 300D).
* Slovní vnoření zachovávají sémantické vztahy a vektorovou algebru ($\vec{v}(\text{king}) - \vec{v}(\text{man}) + \vec{v}(\text{woman}) \approx \vec{v}(\text{queen})$).
* Dvě trénovací architektury:
  * **CBOW (Continuous Bag of Words)**: Předpovídá cílové slovo z okolního kontextu (rychlejší pro častá slova).
  * **Skip-gram**: Předpovídá okolní kontext z cílového slova (výborné pro vzácná slova).
* Nástroj: Knihovna `gensim`.

---

### 4. Etapa: Transformery & BERT (Deep Learning & Attention)
* **Transformer (Vaswani et al., 2017)**:
  * Odstranil rekurence (RNN/LSTM) a nahradil je mechanismem **Self-Attention**:
    $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
  * Umožnil 100% paralelizaci výpočtů na GPU a přímé modelování vazeb libovolně vzdálených slov ($O(1)$).
* **BERT (Devlin et al., Google 2018)**:
  * Obousměrný enkodér (Bidirectional Encoder).
  * Předtrénován na 3,3 miliardách slov pomocí dvou samo-učících se úloh:
    1. **MLM (Masked Language Model)**: Predikce 15 % maskovaných tokenů `[MASK]` z levého i pravého kontextu.
    2. **NSP (Next Sentence Prediction)**: Binární klasifikace návaznosti dvou vět (`is_next` vs `not_next`).
  * Knihovna: `transformers` (Hugging Face).

---

## 3. Celková srovnávací matice metod reprezentace textu

| Vlastnost | Bag of Words (BoW) | TF-IDF | Word2Vec | BERT |
| :--- | :--- | :--- | :--- | :--- |
| **Typ vektoru** | Řídký (Sparse), $10^4$ dim | Řídký (Sparse), $10^4$ dim | Hustý (Dense), 100–300 dim | Hustý kontextový, 768–1024 dim |
| **Zohlednění pořadí slov** | ❌ Ne | ❌ Ne | ⚠️ Částečně (lokální okno) | ✅ Ano (Poziční kódování) |
| **Zohlednění kontextu** | ❌ Ne | ❌ Ne | ❌ Ne (statický vektor pro slovo) | ✅ Plně obousměrný kontext |
| **Sémantická podobnost** | ❌ Ne | ❌ Ne | ✅ Ano (Kosinová afinita) | ✅ Hluboká sémantika & syntax |
| **Výpočetní náročnost** | Velmi nízká (CPU) | Velmi nízká (CPU) | Střední (CPU / lehká GPU) | Vysoká (GPU / TPU) |
| **Klíčová knihovna** | `scikit-learn` | `scikit-learn` | `gensim` | `transformers` (PyTorch) |

---

## 4. Moderní ekosystém velkých jazykových modelů (LLM)

Technologie představené ve Dni 4 jsou přímým základem současné revoluce v generativní umělé inteligenci:
* **OpenAI & Microsoft**: Rodina modelů **GPT** (GPT-3.5, GPT-4, GPT-4o), na kterých běží **ChatGPT** (Decoder-only architektura).
* **Google**: Modely **BERT**, **T5**, **Bard** a moderní multimodální řada **Gemini**.
* **Meta**: Otevřené modely pro výzkum a komunitu **LLaMA** (Llama-2, Llama-3).
* **Hugging Face**: Globální centrální repozitář desítek tisíc otevřených předtrénovaných modelů a datasetů.
