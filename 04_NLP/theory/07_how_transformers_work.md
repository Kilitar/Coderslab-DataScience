# Architektura Transformer a mechanizmus Self-Attention
> **Zdrojový podklad**: Coderslab Data Science – Day 4: *How Transformer Works*  
> **Klíčový vědecký článek**: Vaswani et al. (2017) – *"Attention Is All You Need"* (Google Brain & Google Research)

---

## 1. Motivace: Proč opustit RNN a LSTM?

Před rokem 2017 dominovaly sekvenčnímu zpracování přirozeného jazyka (NLP) rekurentní neuronové sítě (**RNN**) a jejich pokročilé varianty (**LSTM**, **GRU**). Tyto architektury však narážely na dvě zásadní limitace:

1. **Sekvenční zpracování (Bottleneck paralelizace)**:
   * Slova musela být zpracovávána jedno po druhém v čase: $h_t = f(h_{t-1}, x_t)$.
   * Výpočet v čase $t$ nelze provést, dokud není hotov čas $t-1$. To znemožňovalo efektivní paralelizaci na moderních GPU a TPU akcelerátorech.
2. **Ztráta dlouhodobého kontextu (Vanishing Gradient)**:
   * Ačkoliv LSTM zavedlo paměťovou buňku (*cell state*), při dlouhých větách a odstavcích (desítky až stovky slov) docházelo ke ztrátě informací ze začátku sekvence.
   * Veškerý význam předchozího textu se musel vtěsnat do fixně velkého vektoru skrytého stavu $h_t$ (*information bottleneck*).

---

## 2. Přelom: "Attention Is All You Need" (2017)

Transformer zcela odstranil rekurentní smyčky a konvoluce. Celá architektura je postavena výhradně na **mechanismu pozornosti (Self-Attention)** a plně propojených dopředných vrstvách (**Feed-Forward Networks**).

### Klíčové vlastnosti Transformeru:
* **Zpracování celé sekvence najednou (Paralelizace)**: Všechny tokeny věty vstupují do sítě paralelně v jedné maticové operaci.
* **Přímé propojení libovolných slov**: Vzdálenost mezi libovolnými dvěma slovy v textu je $O(1)$ operace (přímá pozornost bez postupného předávání).
* **Flexibilní délka sekvence**: Schopnost přirozeně pracovat se sekvencemi proměnlivé délky.

---

## 3. Celková architektura: Encoder-Decoder

Transformer se skládá ze dvou hlavních větví:
1. **Encoder (Kódovač)**: Přijímá vstupní sekvenci (např. větu v češtině nebo angličtině) a transformuje ji na bohatou matici kontextových reprezentací.
2. **Decoder (Dekodér)**: Využívá reprezentace z encoderu a postupně (autoregresivně) generuje cílovou sekvenci (např. překlad do jiného jazyka).

```
                 VÝSTUP (Pravděpodobnosti slov)
                             ▲
                       [ Linear ]
                             ▲
                       [ Softmax ]
                             ▲
                    ┌─────────────────┐
                    │  DECODER BLOCK  │ x N
                    │ ┌─────────────┐ │
                    │ │Cross-Attn   │◄───────┐
                    │ ├─────────────┤ │       │
                    │ │Masked Attn  │ │       │
                    │ └─────────────┘ │       │
                    └────────┬────────┘       │
                             ▲                │ Kontext
                    Positional + Embeddings   │ z Encoderu
                             ▲                │
                       Cílový text            │
                                              │
                    ┌─────────────────┐       │
                    │  ENCODER BLOCK  │ x N ──┘
                    │ ┌─────────────┐ │
                    │ │Feed-Forward │ │
                    │ ├─────────────┤ │
                    │ │Self-Attn    │ │
                    │ └─────────────┘ │
                    └────────┬────────┘
                             ▲
                    Positional + Embeddings
                             ▲
                       Vstupní text
```

---

## 4. Vstupní vrstva a Poziční kódování (Positional Encoding)

Protože Transformer zpracovává všechna slova současně bez časových kroků, síť sama o sobě **nemá žádné povědomí o pořadí slov ve větě**.
Věty *"Pes kousl člověka"* a *"Člověk kousl psa"* by bez informace o pozici vypadaly pro síť jako identická množina slov.

### Řešení:
K tokenovým embeddingům slov $E(x_i) \in \mathbb{R}^d$ se **přičítá poziční vektor** $PE_{(pos)} \in \mathbb{R}^d$:
$$\text{Input Vector} = \text{Token Embedding} + \text{Positional Encoding}$$

Vaswani et al. navrhli frekvenční sinusové a kosinusové funkce různých frekvencí:
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{\frac{2i}{d_{model}}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{\frac{2i}{d_{model}}}}\right)$$

* $pos$: Index pozice slova ve větě ($0, 1, 2, \dots$).
* $i$: Index dimenze vektoru ($0 \le i < d_{model}/2$).
* **Výhoda**: Model se dokáže snadno naučit relativní posuny, protože pro jakýkoliv fixní offset $k$ lze $PE_{pos+k}$ vyjádřit jako lineární funkci $PE_{pos}$.

---

## 5. Matematika Self-Attention: Dotaz, Klíč, Hodnota (Q, K, V)

Jádrem Transformeru je analogie k vyhledávacím systémům:
* **Query ($Q$)**: "Co hledám?" (vektor dotazu aktuálního slova).
* **Key ($K$)**: "Co nabízím / jaká je má značka?" (vektor klíče každého slova).
* **Value ($V$)**: "Jaký je můj skutečný obsah?" (vektor hodnoty).

Pro každé slovo $x_i$ vytvoříme projekce pomocí naučených vah $W^Q, W^K, W^V$:
$$Q = X W^Q, \quad K = X W^K, \quad V = X W^V$$

### Scaled Dot-Product Attention
Vzorec pro výpočet matice pozornosti:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

1. **Skalární součin $Q K^T$**: Změří afinitu (kompatibilitu) mezi každou dvojicí slov ve větě.
2. **Škálování $\sqrt{d_k}$**: Při vysokých dimenzích (např. $d_k = 64$) by skalární součiny rostly do velkých hodnot, což by zatlačilo funkci softmax do oblastí s extrémně malými gradienty (*saturation*). Dělení odmocninou stabilizuje gradienty.
3. **Softmax**: Převede afinity na pravděpodobnostní váhy ($0$ až $1$, součet řádku je $1$).
4. **Vynásobení $V$**: Výsledný kontextový vektor každého slova je váženým průměrem hodnot $V$ všech ostatních slov.

---

## 6. Multi-Head Attention (Vícehlavá pozornost)

Místo jedné velké pozornosti model rozdělí vektory do $h$ nezávislých "hlav" (např. $h = 8$ nebo $12$ hlav):
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$
kde $\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$.

### Proč více hlav?
Každá hlava se specializuje na **jiný typ lingvistických vztahů**:
* Hlava 1: Syntaktická závislost (sloveso váže podmět).
* Hlava 2: Koreference zájmen (*"it"* $\rightarrow$ *"the animal"*).
* Hlava 3: Místní sousedé (bezprostřední bigramy).
* Hlava 4: Globální téma celého dokumentu.

---

## 7. Decoder a Masked Multi-Head Attention

Dekodér má dvě klíčové odlišnosti:
1. **Causal Masking (Maskovaná pozornost)**:
   * Při generování překladu nesmí model "vidět do budoucnosti".
   * Do matice skalárních součinů před softmaxem se vloží maska $-\infty$ pro všechny budoucí pozice $j > i$.
   * Po softmaxu je váha budoucích slov přesně $0$.
2. **Cross-Attention (Křížová pozornost)**:
   * Query ($Q$) pochází z dekodéru (aktuálně generovaný text).
   * Klíče ($K$) a Hodnoty ($V$) pocházejí z výstupu **Encoderu** (původní zdrojový text).
   * Právě zde probíhá propojení vstupního jazyka a cílového překladu!

---

## 8. Využití a dopad v moderním AI

Architektura Transformer se stala fundamentálním stavebním kamenem moderního AI:
* **Encoder-only** modely (např. **BERT**, RoBERTa): Porozumění textu, klasifikace, extrakce entit (NER), sémantické vyhledávání.
* **Decoder-only** modely (např. **GPT-3, GPT-4, LLaMA**): Generování textu, dialogové systémy, asistence při programování.
* **Encoder-Decoder** modely (např. **T5, BART**): Abstraktivní sumarizace, překlad mezi jazyky.
* **Multimodální modely** (Vision Transformers - ViT, CLIP, Gemini): Zpracování obrazu, audia i videa pomocí stejného mechanismu Self-Attention.
