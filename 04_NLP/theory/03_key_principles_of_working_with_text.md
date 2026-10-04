# Den 4: Klíčové principy práce s textem (Key Principles of Working with Text)

> **Zdrojový materiál:** Prezentace kurzu `Key_principles_of_working_with_text.pdf` (23 slidů)  
> **Tematický blok:** Blok 2 / Den 4: Natural Language Processing (NLP)  
> **Datum kurzu:** 04.10.2026  
> **Cíl modulu:** Zvládnout systematický proces čištění, unifikace a přípravy nestrukturovaných textových dat před vstupem do algoritmů strojového učení (tokenizace, normalizace, odstraňování diakritiky/akcentů, filtrace stop-slov, stemming a lemmatizace) a porozumět jejich modernímu vývoji v éře velkých jazykových modelů (LLM).

---

## 1. Originální teorie kurzu (Strukturovaný rozbor)

Lidský jazyk je přirozeně neuspořádaný, redundantní a plný stylistických i morfologických variací. Počítače a algoritmy strojového učení však pracují výhradně s čísly a vektory pevné dimenze. Než lze text vektorizovat (např. pomocí Bag-of-Words, TF-IDF či embeddingů), musí projít fází **čištění a unifikace**, která redukuje šum a zmenšuje slovník (*vocabulary size*).

```
Surový text ──► Normalizace (lower/unidecode) ──► Tokenizace ──► Filtrace Stop-slov ──► Lemmatizace / Stemming ──► Vektorizace
```

---

### A. Dělení textu na tokeny (Tokenizace)
* **Definice tokenu:** Token je nejmenší diskrétní syntaktická jednotka textu, se kterou model pracuje. Obvykle odpovídá slovu, interpunkčnímu znaménku, číslu nebo symbolu.
* **Tři základní implementační přístupy:**
  1. **Čistý Python (`str.split()`):**
     * Dělí text podle bílých znaků (mezery, tabulátory, konce řádků).
     * *Nedostatek:* Ponechává interpunkci přilepenou ke slovu (např. `"tokens."` místo `["tokens", "."]`).
  2. **Knihovna NLTK (`word_tokenize` s modelem `punkt_tab`):**
     * Založena na regulárních výrazech a statistickém modelu hranic vět a slov (Punkt tokenizer).
     * Správně odděluje tečky, čárky, uvozovky a zkracovací tečky.
  3. **Knihovna spaCy (`en_core_web_sm`):**
     * Deterministický, jazykově specifický stavový automat založený na lexikálních pravidlech jazyka.
     * Současně vytváří objekt `Doc`, kde každý `Token` nese lingvistické anotace (POS tag, syntaktickou závislost, lemmu).

---

### B. Normalizace slov (Words Normalization)
* **Problém:** Slova `"Learning"` a `"LeaRNinG"` mají pro člověka stejný sémantický význam, ale pro počítač představují zcela odlišné binární řetězce (rozdílné ASCII/Unicode kódy).
* **Řešení:**
  * Převod na malá písmena (`.lower()`).
  * Zmenšení velikosti slovníku na unikátní prvky (`set(tokens_lowercased)`).
  * Zabraňuje explozi dimenzionality matice příznaků (*Curse of Dimensionality*).

---

### C. Odstranění akcentů a diakritiky (Accented Characters)
* **Příklad ze zadání:** `"When did you drink latté at our café?"`
* **Problém:** Pro anglický model jsou slova `"latté"` a `"latte"`, `"café"` a `"cafe"` dvěma cizími termíny, což štěpí jejich frekvenci a zkresluje váhy.
* **Nástroj:** Knihovna `unidecode` (`unidecode.unidecode(sentence)`).
* Převádí libovolný Unicode znak na jeho nejbližší 7bitový ASCII ekvivalent (*ASCII transliteration*).

---

### D. Odstranění stop-slov (Removing Stopwords)
* **Definice:** Stop-slova jsou syntaktická a gramatická slova s vysokou frekvencí výskytu, která však nenesou specifický sémantický obsah (v angličtině např. *the, and, of, to, a, in*).
* **Důvod odstranění:**
  * Redukce šumu v datech.
  * Zmenšení velikosti matice příznaků (ve frekvenčních modelech typu BoW mohou tvořit 30–50 % všech slov v korpusu).
  * Zrychlení trénování a snížení paměťové náročnosti.
* **Implementace:**
  * *Čistý Python:* Vlastní seznam a list comprehension (`[w for w in words if w not in stopwords]`).
  * *NLTK:* `nltk.corpus.stopwords.words("english")`.
  * *spaCy:* Bohatý lingvistický atribut `token.is_stop`.

---

### E. Stemming vs. Lemmatizace (Redukce tvarů na základ)

| Kritérium | Stemming (Kmenování) | Lemmatizace |
| :--- | :--- | :--- |
| **Definice** | Heuristické ořezávání přípon a předpon k získání kmene (*stem*). | Morfologická redukce slova na jeho slovníkový základ (*lemma*). |
| **Platnost slova** | Výsledek **nemusí být skutečné slovo** (např. *examining* $\to$ *examin*). | Výsledek je **vždy platný slovníkový tvar** (*examine*). |
| **Výpočetní náročnost**| Extrémně rychlý (jednoduchá sada pravidel a regulárních výrazů). | Pomalejší (vyžaduje morfologický slovník a analýzu kontextu / POS). |
| **Knihovna** | **NLTK** (`PorterStemmer`, `SnowballStemmer`) | **spaCy** (`token.lemma_`), NLTK (`WordNetLemmatizer`) |
| **Příklad** | `"studies"` $\to$ `"studi"`, `"studying"` $\to$ `"studi"` | `"studies"` $\to$ `"study"`, `"studying"` $\to$ `"study"` |

---

## 2. Kritické zhodnocení a metodická úskalí (Expertní pohled)

Ačkoliv kurzové materiály prezentují tyto techniky jako univerzální a přímočarý recept, v reálné Data Science praxi narážejí na zásadní omezení:

### 1. Kardinální chyba: Ztráta negace a polarity v sentimentové analýze
V obecném seznamu NLTK i spaCy stop-slov se nacházejí záporná slova a modální slovesa:
$$\text{stopwords} \ni \{\text{"not"}, \text{"no"}, \text{"nor"}, \text{"never"}, \text{"neither"}, \text{"cannot"}\}$$

Pokud aplikujeme slepé odstranění stop-slov na sentimentovou recenzi (např. z datasetu IMDb z Dne 4):
* **Původní věta:** *"This film is not good, I would never recommend it."*
* **Po odstranění stop-slov:** *"film good recommend"*
* **Důsledek:** Model vyhodnotí recenzi jako **vysoce pozitivní**, ačkoliv byla silně negativní!
* **Doporučená praxe:** Ze seznamu stop-slov vždy explicitně vyjmout negace (`stopwords - {'not', 'no', 'never'}`), nebo stop-slova neodstraňovat a použít n-gramy či kontextové modely.

### 2. Nedostatky Porterova stemmeru (Over-stemming vs. Under-stemming)
Porterův algoritmus (vznikl v roce 1980) používá kaskádu 5 pevných pravidel. Trpí dvěma typickými chybami:
* **Over-stemming (přílišné oříznutí):** Různá slova s odlišným významem jsou oříznuta na stejný kmen:
  * *"universe"* a *"university"* $\to$ obojí oříznuto na kmen `"univers"`.
  * Model ztratí schopnost rozlišit vesmír od vysoké školy!
* **Under-stemming (nedostatečné oříznutí):** Slova se stejným kořenem skončí s různým kmenem:
  * *"adhere"* $\to$ `"adher"`, *"adhesion"* $\to$ `"adhes"`.

### 3. Úskalí knihovny `unidecode` v multijazyčném prostředí
* `unidecode` je optimalizována na přepis do anglického ASCII, ale ignoruje lingvistické konvence jiných jazyků:
  * Německé přehlásky: *"Mädchen"* se v němčině správně přepisuje jako *"Maedchen"*, ale `unidecode` vytvoří *"Madchen"*.
  * Čeština a slovenština: Odstranění háčků a čárek může zcela změnit význam slova (*"pan"* = titul vs. *"pán"* = vládce; *"zeď"* $\to$ *"zed"*).
* V moderní vícejazyčné praxi (multilingual NLP) se diakritika často ponechává, protože moderní tokenizéry s plnou podporou UTF-8 s ní nemají sebemenší problém.

---

## 3. SOTA Kontext a moderní NLP standardy (Stav k 10/2026)

Techniky popsané v prezentaci reprezentují **tradiční éru statistického NLP (před rokem 2018)**. V moderní praxi s nástupem Transformerů a velkých jazykových modelů (LLM) došlo k zásadnímu posunu paradigmatu:

### A. Subword Tokenizace nahradila dělení na celá slova
Tradiční tokenizace na úrovni slov trpěla problémem **OOV (Out-of-Vocabulary)** – jakékoliv neznámé slovo či překlep dostalo token `<UNK>`. Dnešní standardem jsou subword algoritmy:
1. **BPE (Byte-Pair Encoding):**
   * Používají modely GPT (GPT-4o, ChatGPT), RoBERTa, LLaMA 3.3.
   * Začíná na úrovni jednotlivých bajtů/znaků a postupně spojuje nejčastější dvojice znaků.
   * Slovo *"unhappiness"* rozbije na `["un", "happiness"]` nebo `["un", "happi", "ness"]`.
2. **WordPiece:**
   * Používá BERT a DistilBERT.
   * Maximalizuje věrohodnost jazykového modelu při spojování podčástí slov (s prefixem `##`, např. `["play", "##ing"]`).
3. **SentencePiece / Unigram:**
   * Používají modely T5, Gemma, Mistral.
   * Zpracovává text jako čistý proud bajtů nezávisle na mezerách (vhodné pro jazyky bez mezer jako čínština či japonština).

### B. Proč se v éře LLM stop-slova a stemmery téměř nepoužívají?
Moderní modely typu Transformer využívají mechanismus **Self-Attention**:
* Pozornost počítá vztah mezi **všemi dvojicemi tokenů** ve větě.
* Stop-slova jako zájmena (*"he"*, *"it"*), předložky a spojky tvoří klíčové syntaktické kostry pro pochopení větné vazby (kdo co komu udělal).
* Pokud bychom odstranili stop-slova před vstupem do Transformeru, model by ztratil schopnost porozumět syntaktické závislosti a koreferenci.
* Podobně stemming ničí jemné gramatické informace (časování sloves, pádové koncovky), které LLM využívají pro přesnou sémantickou reprezentaci.

### C. Moderní technologický stack pro předzpracování (2026)
* **Hugging Face `tokenizers`:** Implementováno v jazyce **Rust**, tokenizuje gigabajty textu za sekundy s paralelním zpracováním.
* **`tiktoken`:** Extrémně rychlý BPE tokenizér od OpenAI optimalizovaný pro modely rodiny GPT.
* **spaCy v3.8+ s `spacy-transformers`:** Propojení robustního pipeline systému spaCy s moderními předtrénovanými modely (RoBERTa, DeBERTa) pro špičkovou lemmatizaci s ohledem na hluboký kontext.
