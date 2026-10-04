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

## 4. Moderní ekosystém velkých jazykových modelů (LLM) – Stav k říjnu 2026

Technologie představené ve Dni 4 jsou přímým základem současné revoluce v generativní umělé inteligenci a autonomních agentech. Zde je přehled klíčových technologických lídrů a jejich milníků až do roku 2026:

* **Google (Google DeepMind)**:
  * *2013–2020*: Word2Vec, Transformer (*Attention Is All You Need*), BERT, T5.
  * *2023–2024*: Příchod **Gemini 1.0 & Gemini 1.5 Pro / Flash** s revolučním multimodálním kontextovým oknem o velikosti 2+ miliony tokenů a otevřená rodina **Gemma / Gemma 2**.
  * *2025–2026*: **Gemini 2.0 & Gemini Ultra** – nativní multimodální reasoning, integrace agentního uvažování do vyhledávání (AI Overviews) a vědecké průlomy (AlphaFold 3).

* **OpenAI & Microsoft**:
  * *2018–2022*: GPT-1 až GPT-3, spuštění fenoménu **ChatGPT** a instrukční ladění (RLHF).
  * *2023–2024*: **GPT-4, GPT-4 Turbo a GPT-4o (Omni)** s nativní hlasovou a vizuální syntézou v reálném čase.
  * *2024–2026*: **OpenAI o1 ("Strawberry"), o3 & o3-mini** – přechod od čistého generování pravděpodobností k **vnitřnímu řetězci uvažování (Chain-of-Thought Reasoning)**, řešení složitých PhD úloh v matematice, fyzice a autonomním programování.

* **Meta & Open-Source komunita**:
  * *2016–2023*: FastText, LLaMA a LLaMA-2, které odstartovaly open-source revoluci.
  * *2024*: **Llama 3 (8B, 70B)** a gigantický model **Llama 3.1 (405B)** se 128k kontextem a podporou vícejazyčnosti, následovaný **Llama 3.2** (multimodalita pro edge i mobilní zařízení).
  * *2025–2026*: **Llama 3.3 & Llama 4** využívající architekturu Mixture of Experts (MoE) a otevřené reasoning modely, které lze provozovat lokálně přes nástroje jako Ollama, vLLM a llama.cpp.

* **Anthropic**:
  * *2024*: **Claude 3 & Claude 3.5 Sonnet / Haiku** – stanovení nového průmyslového standardu pro softwarové inženýrství, logické uvažování a představení průkopnické funkce **Computer Use** (autonomní ovládání PC prostředí).
  * *2025–2026*: Pokročilé bezpečné modely Claude s ústavní AI (Constitutional AI) a plnohodnotní agentní vývojáři.

* **Hugging Face**:
  * Globální domov pro více než milion otevřených modelů, datasetů a knihoven (`transformers`, `diffusers`, `peft`, `accelerate`). Štandardizace formátu `safetensors` a benchmarking na *Open LLM Leaderboard*.
