# Prework Session 2: Knihovny NLTK a spaCy (Introduction to NLTK and spaCy)

> **Cíl modulu:** Seznámit se se dvěma nejvýznamnějšími ekosystémy pro zpracování přirozeného jazyka v Pythonu – **NLTK** (akademický standard a flexibilní lingvistická stavebnice) a **spaCy** (vysokorychlostní průmyslový standard pro produkční nasazení). Pochopit jejich filozofii, klíčové metody, datové struktury a vědět, kdy který nástroj zvolit.

---

## 1. Dva giganti v ekosystému Python NLP

Zatímco základní operace s textem lze řešit regulárními výrazy (`re`), pro komplexní lingvistickou analýzu (tokenizace, POS tagging, lemmatizace, rozpoznávání entit) potřebujeme robustní knihovny.

| Vlastnost / Kritérium | NLTK (Natural Language Toolkit) | spaCy (Industrial-Strength NLP) |
| :--- | :--- | :--- |
| **Vznik & Zaměření** | 2001 (Steven Bird, Edward Loper) – výuka, výzkum, akademie | 2014 (Matthew Honnibal, Ines Montani) – průmysl, produkce |
| **Architektura** | Soubor samostatných algoritmů a funkcí (String $\to$ List) | Objektově orientovaná pipeline (`Text` $\to$ `Doc` $\to$ `Token`) |
| **Implementace** | Čistý Python (pružné, čitelné, ale pomalejší) | Optimalizovaný Cython + C (extrémně rychlé) |
| **Filozofie návrhu** | Nabízí 10 různých algoritmů pro každou úlohu | Nabízí jeden nejlepší optimalizovaný model |
| **Předtrénované modely** | Nutno stahovat separátní datasety (`nltk.download`) | Integrované jazykové balíčky (`en_core_web_sm`, `md`, `lg`) |
| **Multilingual podpora** | Pravidla pro desítky jazyků (omezená ucelenost) | 70+ jazyků (včetně češtiny, polštiny, rumunštiny atd.) |

---

## 2. NLTK (Natural Language Toolkit)

**NLTK** je nejstarší a nejobsáhlejší knihovna pro výzkumníky a studenty. Poskytuje přístup k více než 50 lingvistickým korpusům a lexikálním zdrojům (včetně WordNetu).

### A. Instalace a stažení datových korpusů
```python
import nltk

# NLTK vyžaduje jednorázové stažení konkrétních lexikonů:
nltk.download("punkt_tab")        # Tokenizační pravidla pro věty a slova
nltk.download("averaged_perceptron_tagger") # Určování slovních druhů (POS)
nltk.download("stopwords")        # Seznamy stop-slov pro různé jazyky
nltk.download("wordnet")          # Lexikální databáze synonym a lemat
nltk.download("vader_lexicon")     # Slovník pro analýzu sentimentu
```

### B. Klíčové metody NLTK

#### 1. Tokenizace vět a slov (`word_tokenize`, `sent_tokenize`)
Rozdělí text na jednotlivé tokeny nebo celé věty:
```python
from nltk.tokenize import word_tokenize, sent_tokenize

text = "NLTK is a powerful library for natural language processing. It provides various tools for text analysis."

words = word_tokenize(text)
sentences = sent_tokenize(text)

print("Words:", words)
# ['NLTK', 'is', 'a', 'powerful', 'library', 'for', 'natural', 'language', 'processing', '.', 'It', 'provides', 'various', 'tools', 'for', 'text', 'analysis', '.']

print("Sentences:", sentences)
# ['NLTK is a powerful library for natural language processing.', 'It provides various tools for text analysis.']
```

#### 2. Part-of-Speech Tagging (`pos_tag`)
Přiřadí každému slovu jeho morfologickou kategorii podle standardu Penn Treebank (např. `NN` = podstatné jméno, `JJ` = přídavné jméno, `VBZ` = sloveso ve 3. osobě):
```python
from nltk import pos_tag

tagged_words = pos_tag(words)
print("Tagged words:", tagged_words)
# [('NLTK', 'NNP'), ('is', 'VBZ'), ('a', 'DT'), ('powerful', 'JJ'), ('library', 'NN'), ...]
```

#### 3. Odstranění stop-slov (`stopwords`)
```python
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))
filtered_words = [word for word in words if word.lower() not in stop_words and word.isalnum()]
print("Filtrovaná slova:", filtered_words)
# ['NLTK', 'powerful', 'library', 'natural', 'language', 'processing', 'provides', 'various', 'tools', 'text', 'analysis']
```

#### 4. Kmenování (*Stemming*) vs. Lemmatizace (*Lemmatization*)
```python
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Stemmer (mechanické ořezání přípon):
stemmer = PorterStemmer()
print("Stemming 'beautiful':", stemmer.stem("beautiful")) # 'beauti'
print("Stemming 'studies':", stemmer.stem("studies"))     # 'studi'

# Lemmatizér (slovníkový tvar podle WordNetu):
lemmatizer = WordNetLemmatizer()
print("Lemma 'beautiful':", lemmatizer.lemmatize("beautiful")) # 'beautiful'
print("Lemma 'studies':", lemmatizer.lemmatize("studies"))     # 'study'
print("Lemma 'corpora':", lemmatizer.lemmatize("corpora"))     # 'corpus'
```

#### 5. Analýza sentimentu VADER (`SentimentIntensityAnalyzer`)
Pravidlový a lexikální analyzátor optimalizovaný pro sociální sítě a recenze:
```python
from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()
sentiment = sia.polarity_scores("Antigravity and Python are amazingly fast and enjoyable!")
print("Sentiment:", sentiment)
# {'neg': 0.0, 'neu': 0.444, 'pos': 0.556, 'compound': 0.8122}
```
- `compound` nabývá hodnot od $-1.0$ (extrémně negativní) po $+1.0$ (extrémně pozitivní).

---

## 3. spaCy (Industrial-Strength NLP)

**spaCy** je navržen od základů pro maximální rychlost a praktické využití v produkčním softwaru. Je napsán v Cythonu a operuje s předtrénovanými jazykovými modely.

### A. Jazykové modely v spaCy
spaCy nabízí modely rozdělené dle velikosti:
- `en_core_web_sm` (Small, $\approx 12$ MB): Rychlý, obsahuje základní pipeline bez statických vektorů.
- `en_core_web_md` (Medium, $\approx 40$ MB): Obsahuje 300dimenzionální Word2Vec vektory (vhodné pro sémantickou podobnost).
- `en_core_web_lg` / `trf` (Large / Transformer): Maximální přesnost založená na architektuře RoBERTa.

```python
import spacy

nlp = spacy.load("en_core_web_sm")
```

### B. Objektově orientovaná architektura: `Doc`, `Span`, `Token`
Když zavoláte `doc = nlp(text)`, spaCy v jednom průchodu provede:
1. Tokenizaci,
2. POS Tagging a morfologickou analýzu,
3. Syntaktický strom závislostí (*Dependency Parsing*),
4. Rozpoznávání pojmenovaných entit (*NER*),
5. Lemmatizaci a vektorové přiřazení.

```python
doc = nlp("Apple commits $430 billion in US investments over five years.")
```

#### 1. Procházení tokenů a jejich vlastností
Každý `token` má bohaté atributy:
```python
for token in doc:
    print(f"{token.text:12} | POS: {token.pos_:6} | Lemma: {token.lemma_:10} | Stopword: {token.is_stop}")
```

#### 2. Rozpoznávání pojmenovaných entit (NER – Named Entity Recognition)
spaCy automaticky detekuje entity s jejich znakovým rozsahem a typem:
```python
for entity in doc.ents:
    print(f"Entita: {entity.text:20} | Pozice: ({entity.start_char}, {entity.end_char}) | Typ: {entity.label_}")

# Výstup:
# Entita: Apple                | Pozice: (0, 5)     | Typ: ORG (Organizace)
# Entita: $430 billion         | Pozice: (14, 26)   | Typ: MONEY (Finanční obnos)
# Entita: US                   | Pozice: (30, 32)   | Typ: GPE (Stát/Město)
# Entita: five years           | Pozice: (51, 61)   | Typ: DATE (Časové období)
```

#### 3. Vektorizace slov a sémantická podobnost
Pokud model obsahuje vektory (např. `en_core_web_md`), `token.vector` vrací hustý 300D embedding a `token.vector_norm` jeho eukleidovskou normu:
```python
for token in doc:
    print(token.text, "Norma vektoru:", token.vector_norm)

# Sémantická kosinová podobnost dvou dokumentů nebo slov:
doc1 = nlp("hamburger and pizza")
doc2 = nlp("fast food restaurant")
print("Podobnost:", doc1.similarity(doc2)) # Např. 0.82
```

---

## 4. Velké srovnání: Kdy zvolit NLTK a kdy spaCy?

```
┌─────────────────────────────────────────────────────────────┐
│                       Rozhodovací strom                     │
└─────────────────────────────────────────────────────────────┘
                               │
               Potřebujete model nasadit do produkce?
                               │
                ┌──────────────┴──────────────┐
               ANO                            NE
                │                              │
          Zvolte spaCy           Chcete experimentovat s různými
     (Vysoká rychlost,            algoritmy a ladit gramatiku?
      ucelená pipeline)                        │
                                        ┌──────┴──────┐
                                       ANO            NE
                                        │              │
                                   Zvolte NLTK    Zvolte spaCy
```

### Doporučení:
1. **Použijte NLTK, pokud:**
   - Děláte akademický výzkum, potřebujete srovnat 3 různé stemmery (Porter vs. Snowball vs. Lancaster).
   - Pracujete s netradiční gramatikou (kontextově bezesporné gramatiky CFG, synsety WordNetu).
   - Potřebujete rychlou a jednoduchou analýzu sentimentu pomocí VADER bez nutnosti instalovat velké jazykové modely.
2. **Použijte spaCy, pokud:**
   - Stavíte reálnou produkční webovou nebo mobilní aplikaci.
   - Zpracováváte velké objemy dat (tisíce dokumentů za sekundu díky streamování přes `nlp.pipe()`).
   - Potřebujete spolehlivou extrakci entit (NER) a syntaktický rozbor vět (*Dependency Parsing*).
   - Chcete snadnou integraci s moderními Transformery (Hugging Face / PyTorch).
