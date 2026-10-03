# 🧠 Prework Session 2: Úvod do neuronových sítí (Introduction to Neural Networks)

Tento teoretický materiál slouží jako příprava na **Session 2 (Den 3)** kurzu Data Science / Machine Learning. Pokrývá biologické základy, matematický model umělého neuronu, architekturu vrstev, detailní přehled aktivačních funkcí (včetně derivací a mizejícího gradientu) a typologii architektur (Perceptron, MLP, CNN, RNN).

---

## 1. Co jsou neuronové sítě a biologická analogie

**Umělé neuronové sítě (Artificial Neural Networks – ANN)** jsou třídou algoritmů strojového učení inspirovaných strukturou a funkcí nervové soustavy živých organismů:
- Lidský mozek obsahuje přibližně **86 miliard neuronů** propojených až **$10^{15}$ synaptickými spoji**.
- **Biologický neuron** přijímá elektrochemické vzruchy přes **dendrity**, vyhodnocuje je v **buněčném těle (soma)** a při překročení akčního potenciálu vysílá impuls přes **axon** na další neurony přes **synaptické štěrbiny**.
- **Umělý neuron** je zjednodušeným matematickým modelem: přijímá numerické vstupy ($x_i$), násobí je váhami ($w_i$), přičítá práh (**bias** $b$) a výsledný signál propouští přes **aktivační funkci** $f(z)$.

### Srovnávací analogie:
| Biologický neuron | Umělý neuron (ANN) | Matematický ekvivalent | Funkce v modelu |
| :--- | :--- | :--- | :--- |
| **Dendrity** | Vstupy ($x_1, x_2, \dots, x_n$) | Vstupní vektor $\mathbf{x} \in \mathbb{R}^n$ | Přenášejí hodnoty rysů (např. výkon motoru, věk) |
| **Synapse** | Váhy ($w_1, w_2, \dots, w_n$) | Vektor vah $\mathbf{w} \in \mathbb{R}^n$ | Určují sílu a polaritu vlivu (důležitost rysu) |
| **Soma (Tělo buňky)** | Vážený součet + Bias | $z = \sum_{i=1}^n w_i x_i + b = \mathbf{w}^T \mathbf{x} + b$ | Agregace všech vstupních signálů |
| **Akční potenciál** | Aktivační funkce | $a = f(z)$ | Nelineární transformace rozhodující o excitaci |
| **Axon** | Výstup neuronu | $y = a$ | Výsledný signál odeslaný dalším vrstvám |

---

## 2. Matematický aparát jednoho neuronu

Výpočet uvnitř každého neuronu probíhá striktně ve dvou po sobě jdoucích krocích:

### Krok 1: Lineární kombinace (Vážený součet + Bias)
$$z = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b = \sum_{i=1}^n w_i x_i + b = \mathbf{w}^T \mathbf{x} + b$$
- **Váhy ($w_i$):** Určují sklon rozhodovací nadroviny. Kladná váha posiluje excitaci, záporná váha působí inhibičně.
- **Bias ($b$):** Posunuje rozhodovací hranici v prostoru nezávisle na vstupech (analogicky k absolutnímu členu v lineární regresi $\beta_0$). Bez biasu by každá rozhodovací nadrovina musela procházet počátkem souřadnic $[0, 0, \dots, 0]$.

### Krok 2: Nelineární aktivace
$$a = f(z)$$
- Pokud by neuron neměl nelineární aktivační funkci ($f(z) = z$), celá vícevrstvá síť by zkolabovala do jediné lineární transformace:
  $$f_2(f_1(\mathbf{x})) = \mathbf{W}_2 (\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2 = (\mathbf{W}_2 \mathbf{W}_1)\mathbf{x} + (\mathbf{W}_2 \mathbf{b}_1 + \mathbf{b}_2) = \mathbf{W}_{new} \mathbf{x} + \mathbf{b}_{new}$$
  *Libovolně hluboká lineární síť nedokáže modelovat nic složitějšího než obyčejná lineární regrese!*

---

## 3. Detailní rozbor 6 klíčových aktivačních funkcí

### 1. Lineární funkce (Identity)
- **Vzorec:** $f(z) = z$
- **Derivace:** $f'(z) = 1$
- **Rozsah:** $(-\infty, \infty)$
- **Vlastnosti:** Používá se téměř výhradně ve **výstupní vrstvě regresních sítí** (např. odhad spojité ceny nemovitosti či pevnosti betonu). Ve skrytých vrstvách je nepoužitelná, protože znemožňuje učení nelineárních vztahů.

### 2. Sigmoida (Logistic Sigmoid)
- **Vzorec:** $\sigma(z) = \frac{1}{1 + e^{-z}}$
- **Derivace:** $\sigma'(z) = \sigma(z)(1 - \sigma(z))$
- **Rozsah:** $(0, 1)$
- **Vlastnosti:**
  - Mapuje libovolné reálné číslo na pravděpodobnostní rozsah $(0, 1)$.
  - Běžná ve výstupní vrstvě pro **binární klasifikaci**.
  - **Slabina (Mizející gradient / Vanishing Gradient):** Maximální hodnota derivace je pouhých $0.25$ (při $z=0$). Pro $|z| > 4$ klesá derivace k nule. Při zpětném šíření chyby (backpropagation) přes více vrstev se derivace násobí a gradient exponenciálně mizí k nule $\rightarrow$ síť se přestává učit.
  - Výstup není vycentrován kolem nuly (vždy kladný $\rightarrow$ zpomaluje optimalizaci gradientním sestupem).

### 3. Hyperbolický tangens (Tanh)
- **Vzorec:** $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} = \frac{2}{1 + e^{-2z}} - 1$
- **Derivace:** $\tanh'(z) = 1 - \tanh^2(z)$
- **Rozsah:** $(-1, 1)$
- **Vlastnosti:**
  - Vycentrován kolem nuly (Zero-centered: průměrný výstup je blízko 0), což urychluje konvergenci oproti sigmoitě.
  - Maximální derivace je $1.0$ (při $z=0$).
  - Stále trpí **saturací a mizejícím gradientem** pro velké kladné či záporné hodnoty $|z| > 3$.

### 4. ReLU (Rectified Linear Unit)
- **Vzorec:** $\text{ReLU}(z) = \max(0, z)$
- **Derivace:**
  $$\text{ReLU}'(z) = \begin{cases} 1 & \text{pro } z > 0 \\ 0 & \text{pro } z < 0 \end{cases}$$
- **Rozsah:** $[0, \infty)$
- **Vlastnosti:**
  - **Průmyslový standard pro skryté vrstvy.**
  - Výpočetně extrémně levná (pouhé porovnání `x > 0 ? x : 0` místo drahé exponenciály).
  - Nesaturuje v kladné oblasti $\rightarrow$ derivace je konstantně $1$, což eliminuje mizející gradient pro kladné aktivace.
  - **Slabina (Dying ReLU):** Pokud dojde k velkému zápornému posunu vah, neuron může pro všechny trénovací vzorky vstoupit do oblasti $z < 0$. Jeho derivace je trvale $0$, váhy se nikdy neaktualizují a neuron efektivně "zemře".

### 5. Leaky ReLU
- **Vzorec:** $\text{LeakyReLU}(z) = \max(\alpha z, z)$ (typicky $\alpha = 0.01$)
- **Derivace:**
  $$\text{LeakyReLU}'(z) = \begin{cases} 1 & \text{pro } z > 0 \\ \alpha & \text{pro } z < 0 \end{cases}$$
- **Rozsah:** $(-\infty, \infty)$
- **Vlastnosti:**
  - Řeší problém "Dying ReLU" zavedením malého sklonu $\alpha$ pro záporné hodnoty.
  - Gradient pro záporné hodnoty je nenulový ($\alpha = 0.01$), takže i "spící" neurony se mohou v průběhu učení zotavit.

### 6. Softmax
- **Vzorec:** $\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}} \quad \text{pro } i = 1, \dots, K$
- **Rozsah:** $(0, 1)$, přičemž platí $\sum_{i=1}^K \text{Softmax}(z_i) = 1.0$
- **Vlastnosti:**
  - Používá se výhradně ve **výstupní vrstvě pro vícetřídní klasifikaci (Multiclass Classification)**.
  - Transformuje vektor reálných čísel (tzv. **logits**) na platné rozdělení pravděpodobnosti příslušnosti do jednotlivých $K$ tříd.

---

## 4. Architektura sítě a numerický průchod příkladem (Cena auta)

Uvažujme třívrstvou síť pro predikci ceny automobilu ze zadání:
- **Vstupy ($n=2$):** Výkon motoru $x_1 = 300\ \text{HP}$, Věk vozidla $x_2 = 2\ \text{roky}$.
- **Skrytá vrstva ($m=3$ neurony):** Lineární aktivace.
- **Výstupní vrstva ($1$ neuron):** Lineární aktivace predikující cenu.

### Počet parametrů:
- Spojení vstup $\to$ skrytá vrstva: $2 \times 3 = 6$ vah.
- Biasy skryté vrstvy: $3$ biasy.
- Spojení skrytá vrstva $\to$ výstup: $3 \times 1 = 3$ váhy.
- Bias výstupního neuronu: $1$ bias.
- **Celkový počet trénovatelných parametrů:** $6 + 3 + 3 + 1 = 13$ parametrů.

### Průchod signálu (Forward Pass):
Nechť jsou váhy a biasy nastaveny následovně:
1. **Neuron 1 skryté vrstvy:**
   $$z_1 = (w_{11} \cdot 300) + (w_{21} \cdot 2) + b_1 \implies a_1 = f(z_1)$$
2. **Neuron 2 skryté vrstvy:**
   $$z_2 = (w_{12} \cdot 300) + (w_{22} \cdot 2) + b_2 \implies a_2 = f(z_2)$$
3. **Neuron 3 skryté vrstvy:**
   $$z_3 = (w_{13} \cdot 300) + (w_{23} \cdot 2) + b_3 \implies a_3 = f(z_3)$$
4. **Výstupní neuron:**
   $$\hat{y} = (v_1 \cdot a_1) + (v_2 \cdot a_2) + (v_3 \cdot a_3) + b_{out}$$

---

## 5. Typologie základních architektur neuronových sítí

1. **Jednovrstvý perceptron (Single-Layer Perceptron – SLP):**
   - Vstup přímo napojen na výstupní neurony (žádná skrytá vrstva).
   - Dokáže oddělit pouze lineárně separabilní problémy (slavné selhání na problému XOR, Minsky & Papert 1969).
2. **Vícevrstvý perceptron (Multilayer Perceptron – MLP):**
   - Obsahuje 1 nebo více skrytých vrstev s nelineárními aktivacemi.
   - **Univerzální aproximační teorém (Cybenko, 1989):** MLP s jedinou skrytou vrstvou a nelineární aktivací dokáže s libovolnou přesností aproximovat jakoukoliv spojitou funkci na kompaktní množině.
3. **Konvoluční sítě (Convolutional Neural Networks – CNN):**
   - Využívají konvoluční filtry s váhovým sdílením (weight sharing) a pooling pro prostorovou invarianci.
   - Doména: Počítačové vidění (Computer Vision), klasifikace medicínských snímků (CT, MRI), rozpoznávání objektů.
4. **Rekurentní sítě (Recurrent Neural Networks – RNN, LSTM, GRU):**
   - Obsahují zpětné vazby a vnitřní skrytý stav (paměť), který uchovává kontext předchozích kroků v čase.
   - Doména: Zpracování sekvenčních dat, časové řady, přirozený jazyk (NLP), rozpoznávání řeči.
5. **Moderní stav (10/2026): Transformery & Attention mechanismus:**
   - Nahradily tradiční RNN díky plné paralelizaci a mechanismu Self-Attention (např. BERT, GPT, LLaMA, Vision Transformers ViT).
