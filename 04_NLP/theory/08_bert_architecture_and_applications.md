# BERT: Obousměrné reprezentace z Transformerů & Hugging Face
> **Zdrojový podklad**: Coderslab Data Science – Day 4: *BERT*  
> **Klíčový vědecký článek**: Devlin et al. (2018) – *"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"* (Google AI Language)

---

## 1. Co je to BERT?

**BERT** znamená **B**idirectional **E**ncoder **R**epresentations from **T**ransformers (Obousměrné reprezentace z enkodérů architektury Transformer).

### Zásadní rozdíl oproti předchozím modelům:

| Model | Typ kontextu | Popis |
| :--- | :--- | :--- |
| **Word2Vec / GloVe** | Statický (bez kontextu) | Každé slovo má jeden fixní vektor bez ohledu na to, zda jde o *"river bank"* nebo *"investment bank"*. |
| **OpenAI GPT** | Jednosměrný (zleva doprava) | Autoregresivní dekodér. Slovo se může dívat pouze na předchozí slova (důležité pro generování textu). |
| **BERT** | **Skutečně obousměrný (Bidirectional)** | Enkodér transformeru. Slovo je reprezentováno na základě **celého pravého i levého kontextu současně** ve všech vrstvách. |

---

## 2. Architektura modelu

BERT využívá **výhradně bloky Encoderu** z původní architektury Transformer.

### Dvě základní varianty modelu (Google, 2018):
1. **$BERT_{BASE}$**:
   * Počet vrstev (Transformer blocks, $L$): **12**
   * Skrytá dimenze ($H$): **768**
   * Počet hlav pozornosti ($A$): **12**
   * Celkový počet parametrů: **110 milionů**
2. **$BERT_{LARGE}$**:
   * Počet vrstev ($L$): **24**
   * Skrytá dimenze ($H$): **1024**
   * Počet hlav pozornosti ($A$): **16**
   * Celkový počet parametrů: **340 milionů**

### Trénovací korpus:
* **BooksCorpus** (800 milionů slov)
* **Anglická Wikipedia** (2 500 milionů slov)
* Celkem přes **3,3 miliardy slov**.
* Trénováno na 16 až 64 TPU jádrech po dobu 4 dní (96 hodin nepřetržitého běhu).

---

## 3. Vstupní reprezentace: Součet 3 embeddingů

Vstup do BERTu není pouhý text, ale speciálně formátovaná sekvence, kde je každý token reprezentován součtem tří různých vektorů:

$$\mathbf{E}_{token} = \mathbf{E}_{WordPiece} + \mathbf{E}_{Segment} + \mathbf{E}_{Position}$$

```
Vstupní text:    [CLS]     The       dog       barks     [SEP]     It        runs      [SEP]
                  │         │         │         │         │         │         │         │
Token Embed:     E_[CLS]   E_The     E_dog     E_barks   E_[SEP]   E_It      E_runs    E_[SEP]
Segment Embed:   E_A       E_A       E_A       E_A       E_A       E_B       E_B       E_B
Position Embed:  E_0       E_1       E_2       E_3       E_4       E_5       E_6       E_7
```

1. **Token Embeddings (WordPiece)**:
   * Slovník o velikosti cca 30 000 sub-word tokenů.
   * Neznámá nebo vzácná slova jsou rozdělena na podslova s předponou `##` (např. *"playing"* $\rightarrow$ *"play"*, *"##ing"*).
2. **Speciální řídicí tokeny**:
   * **`[CLS]` (Classification)**: Vždy na samém začátku sekvence. Výstupní vektor odpovídající tomuto tokenu v poslední vrstvě slouží jako agregovaná reprezentace celé věty pro klasifikační úlohy.
   * **`[SEP]` (Separator)**: Odděluje dvě různé věty (např. otázku od kontextu nebo větu A od věty B) a ukončuje sekvenci.
3. **Segment Embeddings**:
   * Vektor označující, zda token patří do první věty ($E_A$) nebo druhé věty ($E_B$).
4. **Position Embeddings**:
   * Naučené poziční vektory pro každou pozici (až do maximální délky 512 tokenů).

---

## 4. Dvě předtrénovací úlohy (Self-Supervised Pre-training)

Protože BERT nebyl trénován na ručně anotovaných datech, využil dvě geniální samo-učící se úlohy:

### A. Masked Language Model (MLM) – "Doplňování slov"
Standardní jednosměrné jazykové modely předpovídají další slovo zleva doprava. Pokud by ale BERT viděl obousměrný kontext, predikce dalšího slova by byla triviální (slova by "viděla sama sebe").

**Řešení: Maskování 15 % náhodně vybraných tokenů**:
* **80 % času**: Token je nahrazen speciálním tokenem `[MASK]`.  
  *Příklad:* *"The capital of Poland is [MASK]."* $\rightarrow$ cíl: *"warsaw"*.
* **10 % času**: Token je nahrazen náhodným slovem ze slovníku.  
  *Příklad:* *"The capital of Poland is apple."* $\rightarrow$ cíl: *"warsaw"*. (Učí model nebýt závislý pouze na přítomnosti tokenu `[MASK]`).
* **10 % času**: Token je ponechán beze změny.  
  *Příklad:* *"The capital of Poland is warsaw."* $\rightarrow$ cíl: *"warsaw"*. (Učí model zachovat správnou reprezentaci i pro nemaskovaná slova).

Model minimalizuje Cross-Entropy ztrátu **pouze na těchto 15 % maskovaných pozicích**.

### B. Next Sentence Prediction (NSP) – "Predikce následující věty"
Pro úlohy jako Question Answering (otázky a odpovědi) nebo NLI (přirozené odvozování) nestačí rozumět jednotlivým slovům, model musí chápat vztah mezi dvěma celými větami.

* **50 % případů**: Věta B skutečně následuje za větou A v originálním textu (`label = is_next` / `1`).
* **50 % případů**: Věta B je náhodně vybrána z jiného dokumentu v korpusu (`label = not_next` / `0`).

Binární klasifikace se provádí na výstupním vektoru speciálního tokenu `[CLS]`.

---

## 5. Použití BERT přes knihovnu Hugging Face `transformers`

Knihovna **Hugging Face Transformers** poskytuje jednoduché a standardizované rozhraní pro práci s předtrénovanými modely.

```python
# Instalace knihovny
# !pip install transformers

from transformers import pipeline

# Vytvoření inferenční pipeline pro doplňování maskovaných slov
unmasker = pipeline("fill-mask", model="bert-base-uncased")

# Provedení inference
results = unmasker("The capital of Poland is [MASK].")

for res in results:
    print(f"Token: {res['token_str']:<10} | Pravděpodobnost: {res['score']*100:.2f} %")
```

### Typický výstup:
* `warsaw`: **95.2 %**
* `krakow`: **2.1 %**
* `lodz`: **0.8 %**
* `gdansk`: **0.4 %**

---

## 6. Etika, zkreslení dat (Bias) a stereotypy v jazykových modelech

Protože BERT byl trénován na obrovském množství reálných textů z internetu, knih a encyklopedií, nevyhnutelně **absorboval lidské předsudky a stereotypy**, které se v těchto textech nacházely.

### Experiment s maskováním profesí (Devlin et al., Caliskan et al.):
1. *"The man worked as a [MASK]."*
   * Model preferenčně predikuje profese jako: *lawyer, carpenter, doctor, mechanic, engineer, driver, priest*.
2. *"The woman worked as a [MASK]."*
   * Model preferenčně predikuje profese jako: *nurse, waitress, maid, teacher, secretary, cleaner, dancer*.

Tento jev ukazuje, že neuronové sítě nejsou neutrálními soudci – jsou zrcadlem dat, na kterých byly natrénovány. Vývojáři a datoví vědci proto musí při nasazování modelů do praxe (např. automatické třídění životopisů) zavádět mechanismy detekce a potlačování zkreslení (**Debiasing**, **Fairness AI**).

---

## 7. Praktické uplatnění a Fine-Tuning

BERT přinesl do NLP éru **Transfer Learningu** (přenosem učení):
1. **Předtrénování (Pre-training)**: Extrémně nákladné (tisíce GPU/TPU hodin na terabajtech textu), provádí se jednou velkými laboratořemi (Google, Meta, OpenAI).
2. **Doladění (Fine-Tuning)**: Levné a rychlé (několik minut až hodin na jedné GPU). K výstupu `[CLS]` se přidá jednoduchá lineární klasifikační vrstva a celý model se doučí na specifické doménové úloze:
   * **Sentiment Analysis**: Recenze $\rightarrow$ Pozitivní / Negativní.
   * **Named Entity Recognition (NER)**: Detekce osob, firem, lokalit v textu.
   * **Question Answering (SQuAD)**: Nalezení přesného úseku textu odpovídajícího na otázku.
   * **Sémantická podobnost (Semantic Search)**: Porovnávání embeddingů dokumentů a dotazů.
