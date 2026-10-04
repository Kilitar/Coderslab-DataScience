# Den 4: Word2Vec – Praktická implementace v knihovně Gensim

Tento dokument detailně syntetizuje podklady z prezentace **Word2Vec - sample implementation.pdf**.

---

## 1. Úvod do knihovny Gensim
Knihovna **Gensim** je standardním Python nástrojem pro trénování a správu vektorových reprezentací textu. Kromě samotného **Word2Vec** podporuje také:
* **Doc2Vec:** Vektorová reprezentace celých dokumentů.
* **FastText:** Vektorizace slov založená na pod-slovních n-gramech (subwords).
* **TF-IDF & LSI/LDA:** Téma-analýza (Topic Modeling) a redukce dimenzionality.

---

## 2. Práce s hotovými embeddingy: Třída `KeyedVectors`
Pokud již máme předtrénované vektory (např. z HuggingFace, IPI PAN nebo Google News), nepotřebujeme ukládat celou architekturu neuronové sítě, ale pouze mapování slov na jejich vektory – k tomu slouží objekt **`KeyedVectors`**.

```python
from gensim.models import KeyedVectors
import gensim.downloader as api

# 1. Načtení z textového souboru (formát Word2Vec)
word_vectors = KeyedVectors.load_word2vec_format("word2vec_en.txt", binary=False)

# 2. Stažení populárních předtrénovaných vektorů přes Gensim API
word_vectors_twitter = api.load("glove-twitter-25")
```

### Klíčové metody objektu `KeyedVectors`:
1. **`.most_similar(positive, negative, topn)`**  
   Vrátí $n$ nejvíce podobných slov podle kosinové podobnosti s podporou vektorové aritmetiky:
   ```python
   # King - Man + Woman = Queen
   word_vectors.most_similar(positive=["woman", "king"], negative=["man"], topn=5)
```
2. **`.distance(w1, w2)`**  
   Vrátí kosinovou vzdálenost mezi dvěma slovy ($1 - \text{cosine similarity}$):
   ```python
   dist = word_vectors.distance("city", "nyc")  # Hodnota v intervalu [0, 2]
```
3. **`.distances(word_or_vector, other_words)`**  
   Vrátí pole vzdáleností daného slova ke skupině jiných slov.
4. **`.doesnt_match(words)`**  
   Odhalí slovo, které do dané skupiny významově nezapadá („vetřelec“):
   ```python
   word_vectors.doesnt_match(
       ["car", "truck", "boat", "bike"]
   )  # Vrátí 'boat' (loď nepatří mezi pozemní vozidla)
```
5. **`.get_mean_vector(keys, ignore_missing=True)`**  
   Vypočítá aritmetický průměr vektorů všech zadaných slov dokumentu. Používá se pro **Sentence / Document Embeddings**:
   $$\mathbf{d} = \frac{1}{|d|} \sum_{w \in d} \mathbf{v}_w$$
6. **`.get_vector(key)`**  
   Získá numpy pole embeddingu pro konkrétní zadané slovo.
7. **`.add_vector(key, vector)`**  
   Dynamicky přidá nový vektor pro nově definované slovo do slovníku.

---

## 3. Trénování vlastního modelu Word2Vec

Při trénování vlastního modelu na specifické doméně (např. recenze aplikací Threads, recenze filmů nebo zákaznická podpora) využíváme třídu `Word2Vec`.

### Příprava dat:
Text musí být předzpracován (odstranění znaků, lowercase, stop-slova, lemmatizace) a převeden na **seznam seznamů slov (list of lists)**:
```python
sentences = [
    ["threads", "application", "great", "alternative", "twitter"],
    ["bad", "experience", "crash", "bug", "terrible"],
]
```

### Inicializace a trénování:
```python
from gensim.models import Word2Vec

model = Word2Vec(
    sentences=sentences,  # Vstupní korpus (nebo LineSentence)
    vector_size=200,  # Dimenze embeddingu (obvykle 100-300)
    window=2,  # Kontextové okno (2 slova před a 2 za)
    min_count=3,  # Ignorovat slova s méně než 3 výskyty
    workers=3,  # Počet paralelních vláken CPU
    sg=0,  # 0 = CBOW, 1 = Skip-gram
    epochs=20,  # Počet průchodů datasetem
    alpha=0.025,  # Počáteční učící rychlost (learning rate)
)
```

### Ukládání a načítání modelu:
```python
# Uložení celého modelu (umožňuje budoucí dotrénování)
model.save("word2vec.model")

# Načtení modelu
loaded_model = Word2Vec.load("word2vec.model")

# Dotrénování na nových datech (Fine-tuning / Online learning)
loaded_model.train(
    [["threads", "update", "works", "fine"]], total_examples=1, epochs=1
)

# Export pouze hotových vektorů (KeyedVectors)
model.wv.save("word_vectors.kv")
```
