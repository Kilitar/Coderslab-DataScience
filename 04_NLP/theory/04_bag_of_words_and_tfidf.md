# Den 4: Vektorizace textu – Bag of Words (BoW) & TF-IDF

Tento dokument detailně syntetizuje teoretické a implementační podklady z prezentací **IDF - introduction.pdf** a **IDF - implementation.pdf**.

---

## 1. Motivace: Proč převádět text na čísla?
Počítače ani algoritmy strojového učení (Logistická regrese, SVM, Random Forest, Neuronové sítě) neumí přímo interpretovat písmena a řetězce. Abychom mohli aplikovat matematické algoritmy, musíme text převést na **číselné vektory** (vektorová reprezentace dokumentů).

Základní techniky reprezentace:
1. **Bag of Words (BoW) / Count Vectorizer** – frekvenční reprezentace.
2. **TF-IDF (Term Frequency – Inverse Document Frequency)** – vážená relevance termínů v kontextu korpusu.

---

## 2. Bag of Words (BoW): Pytel slov

### Princip metody
Dokument je nahlížen jako neuspořádaná kolekce („pytel“) slov. Zcela se ignoruje slovosled i gramatická struktura vět; zaznamenává se pouze četnost výskytu jednotlivých slov.

### Kroky algoritmu BoW:
1. **Tokenizace:** Rozdělení textu dokumentu na jednotlivé tokeny/slova.
2. **Tvorba slovníku (Vocabulary):** Sestavení množiny všech unikátních slov z celého korpusu dokumentů o velikosti $|V|$.
3. **Vektorizace:** Každý dokument je reprezentován vektorem délky $|V|$, kde hodnota na pozici $i$ udává počet výskytů $i$-tého slova v daném dokumentu.

### Příklad z prezentace (3 dokumenty):
* **Dokument 1:** *"A cat is an animal."* $\rightarrow$ `['cat', 'is', 'animal']`
* **Dokument 2:** *"A dog is also an animal."* $\rightarrow$ `['dog', 'also', 'is', 'animal']`
* **Dokument 3:** *"Cats and dogs are popular animals."* $\rightarrow$ `['cats', 'and', 'dogs', 'it', 'popular', 'animals']`

**Slovník (11 unikátních termínů):**
`['cat', 'is', 'animal', 'dog', 'also', 'cats', 'and', 'dogs', 'it', 'popular', 'animals']`

**Vektory dokumentů:**
* Dokument 1: `[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]`
* Dokument 2: `[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]`
* Dokument 3: `[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]`

### Výhody a limity BoW:
| Výhody | Nevýhody & Rizika |
| :--- | :--- |
| **Jednoduchá implementace** a snadná interpretovatelnost | **Ztráta kontextu:** Ignoruje slovosled (*„není dobrý“* vs. *„dobrý není“*) |
| **Univerzálnost:** Funguje pro libovolný jazyk | **Vysoká dimenzionalita:** Korpus s 1 000 texty může mít 15 000+ dimenzí |
| Dobrý výchozí baseline pro SPAM filtr a sentiment | **Řídkost (Sparsity):** Vektory obsahují převážně nuly (výpočetní a paměťová zátěž) |

---

## 3. TF-IDF: Vážená relevance v kontextu korpusu

### Proč TF-IDF?
V prostém BoW mají nejvyšší váhy slova, která se opakují nejčastěji. Ta však často nenesou specifický sémantický význam pro rozlišení obsahu (např. *film*, *movie*, *time*).
**TF-IDF** penalizuje slova běžná v celém korpusu a odměňuje slova specifická pro daný dokument.

### Matematická definice:
1. **Term Frequency (TF):** Míra frekvence slova v konkrétním dokumentu:
   $$\text{TF}(t, d) = \frac{f_{t, d}}{\sum_{t' \in d} f_{t', d}} = \frac{\text{počet výskytů slova } t \text{ v dokumentu } d}{\text{celkový počet slov v dokumentu } d}$$

2. **Inverse Document Frequency (IDF):** Inverzní frekvence v celém korpusu $D$:
   $$\text{IDF}(t, D) = \log_{10}\left(\frac{|D|}{|\{d \in D : t \in d\}|}\right)$$
   * Pokud se slovo vyskytuje ve **všech** dokumentech: $\text{IDF} = \log_{10}(1) = 0$. Takové slovo dostane nulovou váhu!
   * Pokud je slovo vzácné (např. v 1 z 1 000 dokumentů): $\text{IDF} = \log_{10}(1000/1) = 3$.

3. **TF-IDF skóre:**
   $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

---

## 4. Implementace v Pythonu (Scikit-Learn)

### Scikit-Learn třídy a klíčové parametry:
```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# CountVectorizer (BoW)
bow = CountVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),  # unigramy i bigramy
    max_features=5000,   # limit na nejčastějších 5 000 termínů
)
X_bow = bow.fit_transform(corpus)

# TfidfVectorizer (TF-IDF)
tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    norm="l2",           # Euklidovská normalizace řádků
    smooth_idf=True      # Ochrana před dělením nulou: ln((1+N)/(1+df)) + 1
)
X_tfidf = tfidf.fit_transform(corpus)
```

### Důležitý rozdíl vzorců: Teorie (slidy) vs. Scikit-Learn
* **Teorie ve slidech:** Používá dekadický logaritmus $\log_{10}$ a prosté násobení bez normalizace:
  $$\text{IDF}_{\text{slides}} = \log_{10}\left(\frac{N}{df}\right)$$
* **Scikit-Learn:** Používá přirozený logaritmus $\ln$, vyhlazování `smooth_idf=True` a standardně **L2 normalizaci vektorů**:
  $$\text{IDF}_{\text{sklearn}} = \ln\left(\frac{1 + N}{1 + df}\right) + 1$$
  $$\mathbf{v}_{\text{norm}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2}$$

---

## 5. Klasifikace sentimentu pomocí Logistické regrese
V prezentaci je předveden kompletní end-to-end model:
1. Vektorizace filmových recenzí pomocí BoW (`review_vectors = vectorizer.fit_transform(film_df['review_text'])`).
2. Rozdělení na trénovací a testovací sadu (80/20, `train_test_split`).
3. Trénování `LogisticRegression()`.
4. **Zlepšení přesnosti:**
   * Základní model: $63.6\,\%$ (7 z 11 správně).
   * Vyladění regularizace ($C = 10$, menší penalizace vah): nárůst na **$81.8\,\%$** (9 z 11 správně).
5. **Klíčové poučení z chyb modelu (Error Analysis):**
   * Úspěšně klasifikováno: *„The film's emotional impact stayed with me long after the credits rolled.“* $\rightarrow$ Pozitivní.
   * Chybně klasifikováno: *„The plot twists were predictable and didn't offer any surprises.“* $\rightarrow$ Model predikoval pozitivní, protože viděl slova *surprises*, ale kvůli BoW architektuře nezachytil negaci *didn't offer*.
   * **Řešení:** Zavedení bigramů (`ngram_range=(1, 2)`) nebo ochrana negací.
