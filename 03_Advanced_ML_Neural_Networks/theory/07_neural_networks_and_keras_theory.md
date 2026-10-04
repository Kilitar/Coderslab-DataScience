# Den 3: Neuronové sítě a Keras – Teorie, Matematika Backpropagation, Fashion-MNIST & SOTA 10/2026

> **Cíl modulu:** Poskytnout ucelený a hluboký rozbor třetího pilíře Dne 3: **Neuronových sítí (Deep Learning)**. Detailně prostudovat knihovny **TensorFlow a Keras**, anatomii umělého neuronu, matematický aparát **Feedforward** a **Backpropagation** s řetízkovým pravidlem (*Chain Rule*), postavit a analyzovat první model na datasetu **Fashion-MNIST** (101 770 vah) a doplnit kritické principy moderního Deep Learningu k **říjnu 2026** (Keras 3, AdamW, GELU, Vanishing Gradient).

---

## 1. Úvod do TensorFlow a Keras

### 1.1 Co je TensorFlow?
**TensorFlow** je open-source knihovna pro numerické výpočty a strojové učení vyvinutá týmem **Google Brain** a zveřejněná v roce 2015. 
- **C++ jádro s Python rozhraním:** Výpočetně náročné maticové operace běží na vysoce optimalizovaném C++ jádru s přímou podporou hardwarové akcelerace na grafických kartách (**GPU přes CUDA**) a specializovaných procesorech (**TPU – Tensor Processing Units**).
- **Výpočetní tenzory:** Základní datovou strukturou je **tenzor** – vícerozměrné homogenní pole (skalár = 0D tenzor, vektor = 1D, matice = 2D, obrázek se 3 kanály = 3D, dávka obrázků = 4D tenzor `[batch_size, height, width, channels]`).
- **Automatická diferenciace (`tf.GradientTape`):** TensorFlow automaticky zaznamenává všechny operace a počítá přesné symbolické/numerické gradienty pro backpropagation.

---

### 1.2 Co je Keras a jaký je vztah k TensorFlow?
**Keras** vytvořil v roce 2015 **François Chollet** jako minimalistické, modulární a uživatelsky přívětivé vysokoúrovňové rozhraní (High-Level API) pro rychlé experimentování s neuronovými sítěmi.
- **Evoluce:** Původně Keras fungoval jako nadstavba nad různými back-endy (Theano, CNTK, TensorFlow).
- **Integrace do TensorFlow 2.0:** Keras se stal oficiálním a výchozím vysokoúrovňovým rozhraním TensorFlow (`tensorflow.keras`).
- **SOTA 2026 (Keras 3):** Dnešní standard **Keras 3** je multi-backendový – tentýž napsaný model lze bez změny kódu spustit na backendu **TensorFlow**, **PyTorch** nebo **JAX**.

---

### 1.3 Přehled klíčových vrstev v Keras (`tf.keras.layers`)

| Třída vrstvy | Popis & Využití | Klíčové parametry |
| :--- | :--- | :--- |
| `Dense` | **Plně propojená vrstva (Fully Connected).** Každý neuron je spojen se všemi neurony předchozí vrstvy. Používá se pro tabular data a klasifikační hlavy. | `units` (počet neuronů), `activation` |
| `Flatten` | **Zploštění vícerozměrného tenzoru.** Převádí 2D/3D tenzor (např. obrázek $28 \times 28$) do 1D vektoru (784 čísel). Nemá žádné trénovatelné váhy. | `input_shape` |
| `Conv2D` | **2D Konvoluční vrstva (CNN).** Posouvá filtry přes matici pixelů a extrahuje lokální vizuální rysy (hrany, textury, tvary). | `filters`, `kernel_size`, `strides`, `padding` |
| `MaxPooling2D` | **Podvzorkovací vrstva.** Redukuje prostorové rozměry mapy příznaků výběrem maximální hodnoty v okně (např. $2 \times 2$). Zajišťuje prostorovou invarianci. | `pool_size`, `strides` |
| `LSTM` | **Long Short-Term Memory (RNN).** Rekurentní vrstva se stavovými buňkami a branami (forget, input, output) pro sekvence, text a časové řady. | `units`, `return_sequences` |
| `Embedding` | **Vnoření diskrétních tokenů.** Převádí celočíselné indexy slov do hustých vektorů sémantického významu (základ moderního NLP). | `input_dim`, `output_dim`, `input_length` |

---

### 1.4 Životní cyklus modelu v Keras

```
1. Definice architektury ──► 2. Kompilace (.compile) ──► 3. Trénování (.fit) ──► 4. Predikce (.predict)
   Sequential()                 optimizer="adam"            epochs=30                 y_pred
   .add(Flatten)                loss="categorical_..."      batch_size=256
   .add(Dense 128)              metrics=["accuracy"]        validation_split=0.15
   .add(Dense 10)
```

1. **Definice architektury:**
   ```python
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import Flatten, Dense

   model = Sequential([
       Flatten(input_shape=(28, 28)),
       Dense(128, activation="relu"),
       Dense(10, activation="softmax")
   ])
   ```
2. **Kompilace (`.compile`):**
   Nastavuje optimalizační algoritmus (`adam`, `sgd`, `rmsprop`), ztrátovou funkci (`categorical_crossentropy`, `binary_crossentropy`, `mse`) a sledované metriky (`accuracy`).
3. **Trénování (`.fit`):**
   Provádí iterativní cykly (epochy) v dávkách (*batches*). Vrací objekt `history`, který obsahuje historii vývoje ztráty a metrik na trénovací i validační sadě.
4. **Predikce (`.predict`):**
   Vrací tenzor předpovězených pravděpodobností pro jednotlivé třídy.

---

## 2. Jak neuronové sítě pracují a jak se učí?

### 2.1 Anatomie neuronu a vážený součet
Umělý neuron je inspirován biologickou nervovou buňkou:
- **Dendrity:** Přijímají vstupní signály $x_1, x_2, \dots, x_n$.
- **Synapse:** Každý vstup je násoben synaptickou vahou $w_i$, která vyjadřuje důležitost daného vstupu.
- **Soma (Tělo buňky):** Sečte všechny vážené vstupy a přičte práh excitace (**Bias** $b$):
  $$z = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b = \sum_{i=1}^n w_i x_i + b$$
- **Maticový zápis pro vrstvu neuronů:**
  $$Z = W \cdot X + B$$
- **Axon (Výstup):** Výsledek je transformován nelineární aktivační funkcí $f(z)$:
  $$a = f(z) = f(W \cdot X + B)$$

---

### 2.2 Proč jsou nezbytné nelineární aktivační funkce?

> **Základní věta o lineárních transformacích:**  
> Složení libovolného počtu lineárních funkcí je **opět pouze lineární funkce**!

Pokud by aktivační funkce byla identita ($f(z) = z$):
$$a^{(1)} = W_1 x + b_1$$
$$a^{(2)} = W_2 a^{(1)} + b_2 = W_2 (W_1 x + b_1) + b_2 = (W_2 W_1) x + (W_2 b_1 + b_2) = W' x + b'$$

Bez nelineární aktivace by síť o 100 skrytých vrstvách a miliardě parametrů **dokázala modelovat pouze to, co dokáže obyčejná lineární regrese**! Nelineární aktivační funkce (ReLU, Sigmoid, GELU) umožňují síti ohýbat rozhodovací nadroviny a modelovat libovolně složité nelineární vazby (*Universal Approximation Theorem*).

---

### 2.3 Feedforward a Backpropagation

Trénování neuronové sítě probíhá ve dvou fázích:
1. **Dopředný průchod (Feedforward):** Data protékají ze vstupu přes skryté vrstvy až na výstup, kde model vygeneruje predikci $\hat{y}$.
2. **Výpočet ztráty (Loss Function):** Spočte se odchylka mezi predikcí $\hat{y}$ a skutečnou hodnotou $y$.
3. **Zpětné šíření chyby (Backpropagation):** Pomocí **řetízkového pravidla diferenciálního počtu (*Chain Rule*)** se spočte gradient ztrátové funkce podle každé jednotlivé váhy $\frac{\partial E}{\partial w}$ a váhy se upraví ve směru proti gradientu:
   $$w_{\text{new}} = w_{\text{old}} - \eta \cdot \frac{\partial E}{\partial w}$$
   kde $\eta$ je rychlost učení (*Learning Rate*).

---

## 3. Matematický příklad Backpropagation ze slajdů krok za krokem (Slajdy 20–29)

Přednáška demonstruje exaktní matematický výpočet na konkrétní síti:
- **Vstupy:** $x_1 = 3, \quad x_2 = 5$
- **Cílové hodnoty (Ground Truth):** $t_1 = 0.1, \quad t_2 = 0.9$
- **Skrytá vrstva:** 2 neurony ($h_1, h_2$), bias $b_1 = 0.25$, aktivační funkce Sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}}$
  - Váhy do $h_1$: $w_1 = 0.1, \quad w_3 = 0.3$
  - Váhy do $h_2$: $w_2 = 0.2, \quad w_4 = 0.4$
- **Výstupní vrstva:** 2 neurony ($o_1, o_2$), bias $b_2 = 0.4$, aktivační funkce Sigmoid
  - Váhy do $o_1$: $w_5 = 0.5, \quad w_7 = 0.7$
  - Váhy do $o_2$: $w_6 = 0.6, \quad w_8 = 0.8$

```
          w1 = 0.1
  x1 = 3 ─────────► [ h1 ] ──────► w5 = 0.5 ─────► [ o1 ] (predikce out_o1)
     │   w2 = 0.2 ↗      ↘ w6 = 0.6            ↗
     │          ↗          ↘                 ↗
     │         ↗            ↘               ↗
  x2 = 5 ─────          w7 = 0.7
         ↘ w3 = 0.3                ↘
           ↘                       [ o2 ] (predikce out_o2)
             ─────► [ h2 ] ──────► w8 = 0.8
                    (b1=0.25)       (b2=0.4)
```

---

### Krok 1: Dopředný průchod (Feedforward)

#### 1. Výpočet skrytého neuronu $h_1$:
$$\text{sum}_{h_1} = x_1 w_1 + x_2 w_3 + b_1 = 3 \cdot 0.1 + 5 \cdot 0.3 + 0.25 = 0.3 + 1.5 + 0.25 = \mathbf{2.05}$$
$$\text{out}_{h_1} = \sigma(2.05) = \frac{1}{1 + e^{-2.05}} \approx \mathbf{0.886}$$

#### 2. Výpočet skrytého neuronu $h_2$:
$$\text{sum}_{h_2} = x_1 w_2 + x_2 w_4 + b_1 = 3 \cdot 0.2 + 5 \cdot 0.4 + 0.25 = 0.6 + 2.0 + 0.25 = \mathbf{2.85}$$
$$\text{out}_{h_2} = \sigma(2.85) = \frac{1}{1 + e^{-2.85}} \approx \mathbf{0.945}$$

#### 3. Výpočet výstupního neuronu $o_1$:
$$\text{sum}_{o_1} = \text{out}_{h_1} w_5 + \text{out}_{h_2} w_7 + b_2 = 0.886 \cdot 0.5 + 0.945 \cdot 0.7 + 0.4 = 0.443 + 0.6615 + 0.4 = \mathbf{1.505}$$
$$\text{out}_{o_1} = \sigma(1.505) = \frac{1}{1 + e^{-1.505}} \approx \mathbf{0.818}$$

#### 4. Výpočet výstupního neuronu $o_2$:
$$\text{sum}_{o_2} = \text{out}_{h_1} w_6 + \text{out}_{h_2} w_8 + b_2 = 0.886 \cdot 0.6 + 0.945 \cdot 0.8 + 0.4 = 0.5316 + 0.756 + 0.4 = \mathbf{1.687}$$
$$\text{out}_{o_2} = \sigma(1.687) = \frac{1}{1 + e^{-1.687}} \approx \mathbf{0.844}$$

---

### Krok 2: Výpočet celkové chyby sítě (Loss)
Přednáška používá kvadratickou ztrátovou funkci (MSE):
$$E = E_1 + E_2 = \frac{1}{2}(t_1 - \text{out}_{o_1})^2 + \frac{1}{2}(t_2 - \text{out}_{o_2})^2$$
$$E_1 = \frac{1}{2}(0.1 - 0.818)^2 = \frac{1}{2}(-0.718)^2 = \frac{1}{2}(0.5155) \approx \mathbf{0.2578}$$
$$E_2 = \frac{1}{2}(0.9 - 0.844)^2 = \frac{1}{2}(0.056)^2 = \frac{1}{2}(0.0031) \approx \mathbf{0.0016}$$
$$E_{\text{total}} = 0.2578 + 0.0016 = \mathbf{0.2594}$$

---

### Krok 3: Zpětné šíření chyby pro váhu $w_5$ pomocí řetízkového pravidla

Chceme zjistit, jak změna váhy $w_5$ ovlivní celkovou chybu $E$, tedy derivaci $\frac{\partial E}{\partial w_5}$.  
Podle řetízkového pravidla (*Chain Rule*):

$$\frac{\partial E}{\partial w_5} = \frac{\partial E}{\partial \text{out}_{o_1}} \cdot \frac{\partial \text{out}_{o_1}}{\partial \text{sum}_{o_1}} \cdot \frac{\partial \text{sum}_{o_1}}{\partial w_5}$$

Rozložíme si jednotlivé tři složky:

#### Složka 1: Vliv výstupu neuronu na celkovou chybu
$$E = \frac{1}{2}(t_1 - \text{out}_{o_1})^2 + E_2$$
$$\frac{\partial E}{\partial \text{out}_{o_1}} = \frac{1}{2} \cdot 2 \cdot (t_1 - \text{out}_{o_1}) \cdot (-1) = \mathbf{\text{out}_{o_1} - t_1}$$
Dosazení:
$$\frac{\partial E}{\partial \text{out}_{o_1}} = 0.818 - 0.1 = \mathbf{0.718}$$

#### Složka 2: Vliv váženého součtu na výstup aktivace (derivace Sigmoidu)
Derivace funkce Sigmoid $\sigma(z)$ je $\sigma(z)(1 - \sigma(z))$:
$$\frac{\partial \text{out}_{o_1}}{\partial \text{sum}_{o_1}} = \text{out}_{o_1} \cdot (1 - \text{out}_{o_1}) = 0.818 \cdot (1 - 0.818) = 0.818 \cdot 0.182 = \mathbf{0.1489}$$

#### Složka 3: Vliv váhy $w_5$ na vážený součet
$$\text{sum}_{o_1} = \text{out}_{h_1} w_5 + \text{out}_{h_2} w_7 + b_2$$
$$\frac{\partial \text{sum}_{o_1}}{\partial w_5} = \text{out}_{h_1} = \mathbf{0.886}$$

#### Celkový gradient pro váhu $w_5$:
$$\frac{\partial E}{\partial w_5} = 0.718 \cdot 0.1489 \cdot 0.886 = \mathbf{0.0947}$$

---

### Krok 4: Aktualizace váhy (Gradient Descent Step)
Při rychlosti učení $\eta = 0.3$:
$$w_{5,\text{new}} = w_5 - \eta \cdot \frac{\partial E}{\partial w_5} = 0.5 - 0.3 \cdot 0.0947 = 0.5 - 0.0284 = \mathbf{0.4716}$$

*(Váha $w_5$ klesla z 0.5 na cca 0.47, protože neuron $o_1$ predikoval příliš vysokou hodnotu 0.818 oproti cílové 0.1!)*

---

## 4. První praktická neuronová síť: Fashion-MNIST

Prezentace `First_neural_network.pdf` demonstruje kompletní implementaci klasifikátoru módních doplňků na datasetu **Fashion-MNIST**:

### 4.1 Dataset a 10 tříd
- 70 000 obrázků $28 \times 28$ pixelů ve stupních šedi (hodnoty pixelů 0 až 255).
- 60 000 trénovacích vzorků, 10 000 testovacích vzorků.
- Třídy: `[0: T-shirt/top, 1: Trouser, 2: Pullover, 3: Dress, 4: Coat, 5: Sandal, 6: Shirt, 7: Sneaker, 8: Bag, 9: Ankle boot]`.

### 4.2 Kód architektury v Keras
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

# Načtení a normalizace
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

# Architektura modelu
model = Sequential([
    Flatten(input_shape=(28, 28)),          # Vstup: 784 neuronů
    Dense(128, activation="relu"),          # Skrytá vrstva: 128 neuronů
    Dense(10, activation="softmax")         # Výstupní vrstva: 10 tříd
])

# Kompilace
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

### 4.3 Analýza parametrů (`model.summary()`)
- **Vstupní vrstva (`Flatten`):** $28 \times 28 = 784$ vstupů. Trénovatelných parametrů = **0**.
- **Skrytá vrstva (`Dense 128`):** $(784 \text{ vstupů} \times 128 \text{ neuronů}) + 128 \text{ biasů} = 100\,352 + 128 = \mathbf{100\,480}$ vah.
- **Výstupní vrstva (`Dense 10`):** $(128 \text{ vstupů} \times 10 \text{ neuronů}) + 10 \text{ biasů} = 1\,280 + 10 = \mathbf{1\,290}$ vah.
- **Celkový počet trénovatelných vah:** $\mathbf{101\,770}$ parametrů.

### 4.4 Trénovací dynamika a overfitting
- Při trénování na 30 epoch s `batch_size=256` a `validation_split=0.15`:
  - **Epochy 1–15:** Validační přesnost prudce roste z cca 82 % na 88 %, loss klesá.
  - **Epochy 20–25:** Validační přesnost kulminuje kolem **88.5–89.0 %** a validační loss dosahuje minima (~0.32).
  - **Epochy 25–30:** Trénovací přesnost dále roste k 95 %, ale validační loss začíná mírně narůstat $\to$ **nástup přeučení (Overfitting)**. Ideální bod pro ukončení trénování (*Early Stopping*) je kolem epochy 22.

---

## 5. Expertní kritika & SOTA standardy v Deep Learningu (10/2026)

### 5.1 Vanishing Gradient Problém (Proč Sigmoid selhává v hlubokých sítích)
V našem matematickém příkladu jsme použili funkci Sigmoid. Derivace Sigmoidu má své absolutní maximum $\sigma'(0) = 0.25$.
- Pokud má síť 5 vrstev, gradient se v řetízku násobí faktorem:
  $$\prod_{l=1}^5 \sigma'(z_l) \le (0.25)^5 = 0.000976$$
- Gradient v prvních vrstvách doslova **vymizí (Vanishing Gradient)** a síť se přestane učit!
- **SOTA řešení:** Pro skryté vrstvy se používá **ReLU** ($\text{derivace} = 1$ pro $z > 0$) nebo moderní hladká aktivace **GELU** (*Gaussian Error Linear Unit*, standard v transformerech a moderních MLP) či **Swish/SiLU**.

### 5.2 Evoluce optimalizátorů: Od SGD k AdamW
- **SGD (Stochastic Gradient Descent):** Pomalá konvergence, náchylný k uvíznutí v mělkých sedlových bodech.
- **Adam (Adaptive Moment Estimation, 2014):** Kombinuje momentum (1. moment gradientu) a adaptivní škálování rychlosti učení podle rozptylu (2. moment). Rychlý a robustní.
- **SOTA 2026 – AdamW (Loshchilov & Hutter):** Standardní Adam chybně aplikoval $L_2$ regularizaci na adaptivní momenty. **AdamW** odděluje váhový rozpad (*Decoupled Weight Decay*), což radikálně zlepšuje generalizaci.

### 5.3 Moderní regularizace neuronových sítí
1. **Dropout (Srivastava et al., 2014):** Při každém kroku trénování náhodně vypne procento neuronů (např. 20–50 %). Zabraňuje neuronům v nežádoucí ko-adaptaci.
2. **Batch Normalization (Ioffe & Szegedy, 2015):** Normalizuje aktivace v rámci dávky ($\mu=0, \sigma=1$). Urychluje konvergenci a stabilizuje trénink.
3. **Early Stopping:** Callback zastavující trénování, jakmile validační loss přestane po $k$ epoch klesat (`patience=5`), a obnovující nejlepší nalezené váhy (`restore_best_weights=True`).
