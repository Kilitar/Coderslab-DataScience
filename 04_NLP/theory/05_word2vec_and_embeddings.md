# Den 4: Word2Vec & Slovní vnoření (Word Embeddings)

Tento dokument detailně syntetizuje teoretické podklady z prezentace **Word2Vec - introduction.pdf**.

---

## 1. Motivace: Proč opustit Bag of Words a TF-IDF?

Až dosud jsme text reprezentovali pomocí frekvenčních modelů:
* **Bag of Words (BoW):** Vektor výskytů slov.
* **TF-IDF:** Vážený vektor relevancí.

### Zásadní limity BoW a TF-IDF:
1. **Ztráta sémantiky a synonymie:** Každé slovo je samostatná ortogonální dimenze. Slova *pes* a *štěně*, nebo *výborný* a *skvělý* mají v BoW skalární součin rovný nule (jsou na sebe kolmá).
2. **Prokletí dimenzionality (Sparsity):** Vektory mají tisíce až desetitisíce dimenzí a jsou tvořeny z 99 % nulami.
3. **Absence kontextu:** Modely nevnímají, která slova se přirozeně vyskytují vedle sebe.

**Řešení:** **Word Embeddings (Slovní vnoření)** – reprezentace slov pomocí **hustých (dense)** numerických vektorů v nízké dimenzi (typicky 100 až 300 dimenzí), kde sémanticky příbuzná slova leží blízko u sebe.

---

## 2. Vektorový prostor a sémantická aritmetika

V prostoru vnoření jsou vlastnosti slov zakódovány do jednotlivých latentních směrů (např. rod, věk, královská hodnost, gramatický čas).

### Slavná vektorová analogie:
$$\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}} + \mathbf{v}_{\text{woman}} \approx \mathbf{v}_{\text{queen}}$$

Podobně fungují geografické i gramatické relace:
* $\mathbf{v}_{\text{Praha}} - \mathbf{v}_{\text{Česko}} + \mathbf{v}_{\text{Francie}} \approx \mathbf{v}_{\text{Paříž}}$
* $\mathbf{v}_{\text{bigger}} - \mathbf{v}_{\text{big}} + \mathbf{v}_{\text{cold}} \approx \mathbf{v}_{\text{colder}}$

---

## 3. Algoritmus Word2Vec (Tomáš Mikolov et al., Google 2013)

V roce 2013 představil tým z Google vedený českým vědcem **Tomášem Mikolovem** revoluční samo-učící se neuronovou architekturu. Word2Vec využívá princip distribuční sémantiky (*„Slovo poznáš podle společnosti, v níž se pohybuje“* – J. R. Firth).

Model trénuje jednoduchou dvouvrstvou neuronovou síť na velkém neoznačeném textovém korpusu (self-supervised learning). Vlastní embeddingy jsou pak **vnitřní váhy skryté vrstvy**.

Word2Vec definuje dvě doplňující se architektury:

### A. CBOW (Continuous Bag of Words)
* **Princip:** Předpovídá cílové slovo (Target) z jeho okolních kontextových slov (Context).
* **Kontextové okno (Window size):** Počet slov před a za analyzovaným slovem (např. pro okno 1 ve větě *„Machine learning loves only numbers“* bere pro slovo *loves* kontext *learning* a *only*).
* **Vstup:** One-hot vektory kontextových slov.
* **Skrytá vrstva:** Lineární projekce (průměrování embeddingů kontextu, $D$ neuronů).
* **Výstup:** Softmax pravděpodobnostní rozdělení přes celý slovník.
* **Vlastnosti:** Rychlejší trénování, přesnější pro častá slova.

### B. Skip-gram
* **Princip:** Funguje přesně obráceně – z centrálního slova předpovídá jeho okolní kontextová slova.
* **Vlastnosti:** Pomalejší trénování, ale **výrazně lepší pro méně častá a vzácná slova** a menší datasety.

---

## 4. Vizualizace embeddingů (PCA, t-SNE & TensorBoard)

Protože lidé nedokážou přímo vnímat 100- až 300-dimenzionální prostor, využívají se techniky redukce dimenzionality:
1. **PCA (Principal Component Analysis):** Lineární projekce maximalizující rozptyl.
2. **t-SNE (t-Distributed Stochastic Neighbor Embedding):** Nelineární pravděpodobnostní metoda zachovávající lokální shluky (ideální pro 2D/3D vizualizaci).
3. **TensorFlow Embedding Projector (`projector.tensorflow.org`):** Nástroj pro interaktivní 3D rotaci a hledání nejbližších sousedů podle kosinové vzdálenosti:
   $$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$

---

## 5. Výhody a limity Word2Vec

| Výhody | Nevýhody & Omezení |
| :--- | :--- |
| **Sémantická hustá reprezentace:** Zachovává význam a synonyma | **Neřeší polysemii (víceznačnost):** Slovo *koruna* (měna, strom, zub, královská) má pouze **jeden statický vektor** |
| **Kompaktnost:** Nízká dimenze (100–300) místo 50 000+ v BoW | **Problém OOV (Out-of-Vocabulary):** Neznámé slovo mimo trénink nelze zakódovat |
| **Univerzální přenositelnost (Transfer Learning):** Předtrénované vektory pro libovolné modely | **Abstraktní interpretace dimenzí:** Jednotlivé souřadnice nemají přímé lidské pojmenování |
| **Výpočetní efektivita:** Extrémně rychlé trénování díky Hierarchical Softmax / Negative Sampling | Jazyková závislost (nutnost trénovat pro každý jazyk zvlášť) |

---

## 6. Následovníci Word2Vec v NLP

1. **GloVe (Global Vectors, Stanford 2014):** Kombinuje maticovou faktorizaci celých ko-výskytových matic s lokálním oknem Word2Vec.
2. **FastText (Facebook AI / Mikolov 2016):** Reprezentuje slovo jako soubor znakových n-gramů (*subwords*). **Dokáže vytvořit embedding i pro dosud neviděná slova (řeší OOV) a překlepy!**
3. **Kontextuální embeddingy (BERT, RoBERTa, LLM 2018–2026):** Dynamické embeddingy závislé na celé větě (slovo *bank* získá jiný vektor u řeky a jiný u peněz).
