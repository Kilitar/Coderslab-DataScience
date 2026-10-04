"""
Den 3: Neuronové sítě a Keras – Teorie, Matematika & SOTA 10/2026
================================================================
Interaktivní výukový modul pro třetí část teorie Dne 3 (Sekce Neural Networks):
1. Anatomie umělého neuronu & přehled vrstev Keras (Dense, Flatten, Conv2D, LSTM, Embedding).
2. Živá matematická simulace Backpropagation ze slajdů kurzu (vstupy x1=3, x2=5, řetízkové pravidlo pro váhu w5).
3. První síť na Fashion-MNIST: Rozklad 101 770 vah a křivky učení (loss & accuracy přes 30 epoch).
4. Interaktivní laboratoř aktivačních funkcí (Sigmoid, ReLU, LeakyReLU, Tanh, GELU a jejich derivace).
5. Expertní kritika & Deep Learning SOTA 2026 (Vanishing gradient, Keras 3, AdamW, Dropout).
6. Interaktivní vědomostní kvíz.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def render_neural_networks_theory_view():
    st.title("🧠 Den 3: Neuronové sítě & Keras – Teorie, Matematika & SOTA 2026")
    st.markdown(
        r"""
        **Třetí pilíř Dne 3:** Vstup do světa hlubokého učení (**Deep Learning**). 
        Od biologické inspirace umělého neuronu, přes přesný matematický aparát **Backpropagation** 
        s řetízkovým pravidlem (*Chain Rule*), až po první praktickou síť pro klasifikaci oblečení 
        **Fashion-MNIST** v knihovně **TensorFlow / Keras**.
        """
    )

    # Horní KPI karty
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Základní stavební kámen", "Perceptron / Neuron", delta="z = Wx + b + aktivace")
    with k2:
        st.metric("Mechanismus učení", "Backpropagation", delta="Řetízkové pravidlo (Chain Rule)")
    with k3:
        st.metric("Fashion-MNIST model", "101 770 vah", delta="784 -> 128 -> 10 neuronů")
    with k4:
        st.metric("SOTA Standard 2026", "Keras 3 + AdamW", delta="TF, PyTorch a JAX multi-backend")

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🧠 Neuron & Keras API",
        "📐 Matematika Backpropagation (Slajdy)",
        "👗 První síť: Fashion-MNIST",
        "📈 Laboratoř aktivací & Gradient",
        "🔬 Expertní kritika & SOTA 2026",
        "📝 Vědomostní kvíz"
    ])

    # =========================================================================
    # TAB 1: NEURON & KERAS API
    # =========================================================================
    with tab1:
        st.subheader("1. Anatomie umělého neuronu a vrstvy v Keras")
        st.markdown(
            r"""
            Umělý neuron přijímá vstupy $x_1, x_2, \dots, x_n$, vynásobí je synaptickými vahami $w_i$, 
            sečte s prahem excitace (**Bias** $b$) a výsledek prožene nelineární **aktivační funkcí** $f(z)$:
            
            $$z = \sum_{i=1}^n w_i x_i + b = W^T x + b \quad \Longrightarrow \quad a = f(z)$$
            """
        )

        c_lay1, c_lay2 = st.columns(2)
        with c_lay1:
            st.info(
                r"""
                #### 🧱 Základní vrstvy (`tf.keras.layers`):
                - **`Dense(units, activation)`:** Plně propojená vrstva (Fully Connected). Každý neuron je spojen se všemi v předešlé vrstvě.
                - **`Flatten()`:** Převede vícerozměrný tenzor (např. 2D obrázek $28 \times 28$) na 1D vektor (784 čísel). Nemá žádné váhy.
                - **`Conv2D(filters, kernel_size)`:** Konvoluční vrstva pro zpracování obrazu (detekce hran, textur a objektů).
                """
            )
        with c_lay2:
            st.success(
                r"""
                #### 🔄 Pokročilé a sekvenční vrstvy:
                - **`MaxPooling2D(pool_size)`:** Zmenšuje prostorové rozměry obrazu výběrem maxima (zajišťuje prostorovou invarianci).
                - **`LSTM(units)`:** Rekurentní buňky s vnitřní pamětí pro text, audio a časové řady.
                - **`Embedding(input_dim, output_dim)`:** Převádí diskrétní slova do hustých sémantických vektorů.
                """
            )

        st.markdown("---")
        st.subheader("Životní cyklus modelu v Keras")
        st.code(
            r'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

# 1. Definice architektury
model = Sequential([
    Flatten(input_shape=(28, 28)),          # Vstupní vrstva
    Dense(128, activation="relu"),          # Skrytá vrstva
    Dense(10, activation="softmax")         # Výstupní vrstva (10 tříd)
])

# 2. Kompilace (výběr optimalizátoru, ztráty a metriky)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 3. Trénování modelu (v dávkách / batches)
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=256,
    validation_split=0.15
)

# 4. Predikce a vyhodnocení
y_pred = model.predict(X_test)
test_loss, test_acc = model.evaluate(X_test, y_test)
            ''',
            language="python"
        )

    # =========================================================================
    # TAB 2: MATEMATIKA BACKPROPAGATION
    # =========================================================================
    with tab2:
        st.subheader("2. Exaktní matematický příklad ze slajdů kurzu (Slajdy 20–29)")
        st.markdown(
            r"""
            Přednáška detailně demonstruje výpočet na malé síti se **2 vstupy, 2 skrytými neurony a 2 výstupními neurony**:
            - **Vstupy:** $x_1 = 3, \quad x_2 = 5$
            - **Cílové pravděpodobnosti:** $t_1 = 0.1, \quad t_2 = 0.9$
            - **Aktivace:** Sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}}$, Biasy: $b_1 = 0.25, \quad b_2 = 0.4$
            """
        )

        st.markdown("#### Krok 1: Dopředný průchod (Feedforward)")
        c_ff1, c_ff2 = st.columns(2)
        with c_ff1:
            st.markdown(
                r"""
                **Skrytá vrstva ($h_1, h_2$):**
                - $\text{sum}_{h_1} = 3(0.1) + 5(0.3) + 0.25 = \mathbf{2.05} \implies \text{out}_{h_1} = \sigma(2.05) \approx \mathbf{0.886}$
                - $\text{sum}_{h_2} = 3(0.2) + 5(0.4) + 0.25 = \mathbf{2.85} \implies \text{out}_{h_2} = \sigma(2.85) \approx \mathbf{0.945}$
                """
            )
        with c_ff2:
            st.markdown(
                r"""
                **Výstupní vrstva ($o_1, o_2$):**
                - $\text{sum}_{o_1} = 0.886(0.5) + 0.945(0.7) + 0.4 = \mathbf{1.505} \implies \text{out}_{o_1} \approx \mathbf{0.818}$
                - $\text{sum}_{o_2} = 0.886(0.6) + 0.945(0.8) + 0.4 = \mathbf{1.687} \implies \text{out}_{o_2} \approx \mathbf{0.844}$
                """
            )

        st.markdown(
            r"""
            **Celková kvadratická chyba sítě ($E = E_1 + E_2$):**
            $$E = \frac{1}{2}(t_1 - \text{out}_{o_1})^2 + \frac{1}{2}(t_2 - \text{out}_{o_2})^2 = \frac{1}{2}(0.1 - 0.818)^2 + \frac{1}{2}(0.9 - 0.844)^2 \approx \mathbf{0.2594}$$
            """
        )

        st.markdown("---")
        st.subheader("Krok 2: Zpětné šíření chyby pro váhu $w_5$ (Řetízkové pravidlo)")
        st.markdown(
            r"""
            Hledáme gradient $\frac{\partial E}{\partial w_5}$. Podle **řetízkového pravidla (*Chain Rule*)**:
            
            $$\frac{\partial E}{\partial w_5} = \underbrace{\frac{\partial E}{\partial \text{out}_{o_1}}}_{\text{1. Složka}} \cdot \underbrace{\frac{\partial \text{out}_{o_1}}{\partial \text{sum}_{o_1}}}_{\text{2. Složka}} \cdot \underbrace{\frac{\partial \text{sum}_{o_1}}{\partial w_5}}_{\text{3. Složka}}$$
            """
        )

        d1 = 0.818 - 0.1
        d2 = 0.818 * (1.0 - 0.818)
        d3 = 0.886
        grad_w5 = d1 * d2 * d3

        c_bp1, c_bp2, c_bp3 = st.columns(3)
        with c_bp1:
            st.metric("1. Složka: dE / d(out_o1)", f"{d1:.3f}", delta="out_o1 - t1 (0.818 - 0.1)")
        with c_bp2:
            st.metric("2. Složka: d(out_o1) / d(sum)", f"{d2:.4f}", delta="Derivace Sigmoidu")
        with c_bp3:
            st.metric("3. Složka: d(sum) / d(w5)", f"{d3:.3f}", delta="Aktivace neuronu h1")

        st.info(f"**Výsledný gradient:** $\\frac{{\\partial E}}{{\\partial w_5}} = {d1:.3f} \\times {d2:.4f} \\times {d3:.3f} = \\mathbf{{{grad_w5:.4f}}}$")

        st.markdown("---")
        st.subheader("Interaktivní simulátor aktualizace váhy $w_5$")
        eta_sim = st.slider("Rychlost učení (Learning Rate $\\eta$):", min_value=0.05, max_value=1.0, value=0.3, step=0.05)
        new_w5 = 0.5 - eta_sim * grad_w5
        st.write(
            f"""
            Původní váha: **$w_5 = 0.5$**  
            Aktualizační pravidlo: **$w_{{5,\\text{{new}}}} = w_5 - \\eta \\cdot \\frac{{\\partial E}}{{\\partial w_5}} = 0.5 - {eta_sim} \\times {grad_w5:.4f} = \\mathbf{{{new_w5:.5f}}}$**
            """
        )
        st.caption("Jelikož neuron o1 predikoval příliš vysokou hodnotu (0.818 oproti 0.1), váha w5 logicky klesá!")

    # =========================================================================
    # TAB 3: PRVNÍ SÍŤ: FASHION-MNIST
    # =========================================================================
    with tab3:
        st.subheader("3. První praktická neuronová síť: Fashion-MNIST (101 770 parametrů)")
        st.markdown(
            r"""
            Prezentace `First_neural_network.pdf` demonstruje model klasifikující obrázky oblečení o rozlišení $28 \times 28$ pixelů 
            do 10 tříd (trička, kalhoty, svetry, šaty, kabáty, sandály, košile, tenisky, tašky, kotníkové boty).
            """
        )

        st.markdown("#### Rozpad architektury a počet trénovatelných vah")
        param_table = pd.DataFrame([
            {"Vrstva": "1. Flatten (Vstup)", "Výstupní tvar": "(None, 784)", "Výpočet vah": "Pouze zploštění 28x28 na 784", "Parametry": 0},
            {"Vrstva": "2. Dense (Skrytá ReLU)", "Výstupní tvar": "(None, 128)", "Výpočet vah": "784 vstupů x 128 neuronů + 128 biasů", "Parametry": 100480},
            {"Vrstva": "3. Dense (Výstup Softmax)", "Výstupní tvar": "(None, 10)", "Výpočet vah": "128 vstupů x 10 neuronů + 10 biasů", "Parametry": 1290},
            {"Vrstva": "CELKEM", "Výstupní tvar": "-", "Výpočet vah": "100 480 + 1 290", "Parametry": 101770}
        ])
        st.table(param_table)

        st.markdown("---")
        st.subheader("Dynamika učení přes 30 epoch (Křivky ztráty a přesnosti)")
        st.markdown(
            r"""
            Níže je simulace typického průběhu trénování modelu z přednášky (`batch_size=256`, `validation_split=0.15`). 
            Všimněte si, kde nastává **bod přeučení (Overfitting)**!
            """
        )

        epochs = np.arange(1, 31)
        # Simulované realistické křivky dle Keras výstupu
        train_acc = 0.82 + 0.13 * (1 - np.exp(-epochs / 6.0))
        val_acc = 0.81 + 0.08 * (1 - np.exp(-epochs / 4.0)) - 0.0008 * np.maximum(0, epochs - 20)**1.5
        train_loss = 0.55 * np.exp(-epochs / 7.0) + 0.15
        val_loss = 0.52 * np.exp(-epochs / 5.0) + 0.30 + 0.003 * np.maximum(0, epochs - 22)**1.4

        fig_curves = go.Figure()
        fig_curves.add_trace(go.Scatter(x=epochs, y=train_acc, mode="lines", name="Trénovací přesnost (Train Accuracy)", line=dict(color="#3b82f6", width=2)))
        fig_curves.add_trace(go.Scatter(x=epochs, y=val_acc, mode="lines+markers", name="Validační přesnost (Val Accuracy)", line=dict(color="#10b981", width=2)))
        fig_curves.add_trace(go.Scatter(x=epochs, y=train_loss, mode="lines", name="Trénovací ztráta (Train Loss)", line=dict(color="#f59e0b", dash="dash")))
        fig_curves.add_trace(go.Scatter(x=epochs, y=val_loss, mode="lines+markers", name="Validační ztráta (Val Loss)", line=dict(color="#ef4444", dash="dash")))

        fig_curves.add_vline(x=22, line_dash="dot", line_color="purple", annotation_text="Ideální Early Stopping (Epocha 22)", annotation_position="top left")
        fig_curves.update_layout(
            title="Vývoj metrik Fashion-MNIST modelu v průběhu 30 epoch",
            xaxis_title="Epocha (Epoch)",
            yaxis_title="Hodnota metriky / ztráty",
            height=450,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_curves)

        st.warning(
            r"""
            **Klíčový poznatek z grafu:**  
            Okolo epochy 20–25 validační ztráta dosahuje minima (~0.32) a validační přesnost stagnuje kolem 88.8 %. 
            Po 25. epoše se model začíná biflovat trénovací data nazpaměť (trénovací accuracy stoupá k 95 %, ale validační loss roste) $\to$ **přeučení**.
            """
        )

    # =========================================================================
    # TAB 4: LABORATOŘ AKTIVACÍ & GRADIENT
    # =========================================================================
    with tab4:
        st.subheader("4. Interaktivní laboratoř aktivačních funkcí & Vanishing Gradient")
        st.markdown(
            r"""
            Prozkoumejte tvar aktivačních funkcí a jejich derivací. 
            Právě derivace aktivační funkce určuje, jak dobře se gradient šíří zpět do hlubokých vrstev sítě.
            """
        )

        act_choice = st.selectbox("Vyberte aktivační funkci k vizualizaci:", ["Sigmoid", "ReLU", "LeakyReLU (alpha=0.1)", "Tanh", "GELU (Gaussian Error Linear Unit)"])

        x_vals = np.linspace(-5, 5, 400)
        if act_choice == "Sigmoid":
            y_act = 1.0 / (1.0 + np.exp(-x_vals))
            y_der = y_act * (1.0 - y_act)
            desc = "Max derivace je pouze 0.25! Při 5 vrstvách je gradient utlumen faktorem (0.25)^5 = 0.00097 -> Vanishing Gradient!"
        elif act_choice == "ReLU":
            y_act = np.maximum(0, x_vals)
            y_der = np.where(x_vals > 0, 1.0, 0.0)
            desc = "Derivace pro x > 0 je přesně 1.0 -> žádné tlumení gradientu! Pozor na 'Dying ReLU' pro x < 0."
        elif act_choice == "LeakyReLU (alpha=0.1)":
            y_act = np.where(x_vals > 0, x_vals, 0.1 * x_vals)
            y_der = np.where(x_vals > 0, 1.0, 0.1)
            desc = "Řeší problém 'Dying ReLU' malým nenulovým sklonem pro záporné hodnoty."
        elif act_choice == "Tanh":
            y_act = np.tanh(x_vals)
            y_der = 1.0 - y_act**2
            desc = "Symetrická kolem nuly (-1 až 1). Max derivace je 1.0, ale v saturaci také trpí na vanishing gradient."
        else: # GELU
            from scipy.stats import norm
            y_act = x_vals * norm.cdf(x_vals)
            y_der = norm.cdf(x_vals) + x_vals * norm.pdf(x_vals)
            desc = "Hladká nelinearita – moderní standard v Transformer modelech (BERT, GPT, ViT) a moderních sítích."

        fig_act = go.Figure()
        fig_act.add_trace(go.Scatter(x=x_vals, y=y_act, mode="lines", name=f"Aktivace f(x) - {act_choice}", line=dict(color="#3b82f6", width=3)))
        fig_act.add_trace(go.Scatter(x=x_vals, y=y_der, mode="lines", name="Derivace f'(x)", line=dict(color="#ef4444", width=2, dash="dash")))
        fig_act.update_layout(
            title=f"Průběh funkce {act_choice} a její derivace",
            xaxis_title="Vnitřní potenciál (z)",
            yaxis_title="Hodnota",
            height=400,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_act)
        st.info(desc)

    # =========================================================================
    # TAB 5: EXPERTNÍ KRITIKA & SOTA 2026
    # =========================================================================
    with tab5:
        st.subheader("5. Deep Learning SOTA 2026 & Expertní analýza")
        
        c_e1, c_e2 = st.columns(2)
        with c_e1:
            st.markdown(
                r"""
                #### 🚀 Keras 3 (Next-Gen Multi-Backend)
                - Dnešní standard **Keras 3** už není vázán pouze na TensorFlow.
                - Stejný model můžete trénovat a nasazovat na **TensorFlow**, **PyTorch** i **JAX**.
                - Zajišťuje absolutní přenositelnost kódu bez Vendor Lock-inu.
                
                #### ⚙️ Moderní optimalizátory (Adam vs. AdamW)
                - Klasický **Adam** aplikoval L2 regularizaci přímo na gradienty, což v kombinaci s adaptivním učením vedlo k degradaci vah.
                - **AdamW (Loshchilov & Hutter):** Odděluje váhový rozpad (*Decoupled Weight Decay*). V roce 2026 je AdamW zlatým standardem pro trénování hlubokých sítí.
                """
            )
        with c_e2:
            st.markdown(
                r"""
                #### 🛡️ Moderní regularizační arzenál:
                1. **Dropout (Srivastava et al.):** Náhodné nulování neuronů během trénování (např. 20–30 %). Nutí neurony nespoléhat se jeden na druhého.
                2. **Batch Normalization (BatchNorm) vs. LayerNorm:**
                   - BatchNorm normalizuje přes dávku (vhodné pro konvoluční sítě CNN).
                   - LayerNorm normalizuje přes příznaky (vhodné pro NLP a sekvence).
                3. **Learning Rate Warmup & Cosine Annealing:**
                   - Postupný náběh rychlosti učení a následný pokles podle kosinového rozpadu předchází nestabilitě na počátku tréninku.
                """
            )

    # =========================================================================
    # TAB 6: VĚDOMOSTNÍ KVÍZ
    # =========================================================================
    with tab6:
        st.subheader("6. Interaktivní vědomostní kvíz: Neuronové sítě & Keras")
        
        q1 = st.radio(
            "1. Co by se stalo, kdybychom v neuronové síti nepoužili žádnou nelineární aktivační funkci?",
            [
                "Síť by se nedokázala zkompilovat.",
                "Síť s libovolným počtem vrstev by degenerovala na pouhou obyčejnou lineární regresi.",
                "Síť by měla nekonečnou kapacitu a okamžitě by se přeučila."
            ],
            key="q_nn_1"
        )
        if q1 == "Síť s libovolným počtem vrstev by degenerovala na pouhou obyčejnou lineární regresi.":
            st.success("Správně! Složení libovolného počtu lineárních transformací je opět pouze lineární transformace.")

        q2 = st.radio(
            "2. Proč je u klasifikačního modelu Fashion-MNIST celkem 101 770 parametrů?",
            [
                "Vstupní vrstva Flatten má 100 000 parametrů.",
                "Mezi 784 vstupy a 128 skrytými neurony je 100 352 vah + 128 biasů, a na výstupu 1 280 vah + 10 biasů.",
                "Každý pixel obrázku má vlastní neuron v každé z 10 tříd."
            ],
            key="q_nn_2"
        )
        if q2 == "Mezi 784 vstupy a 128 skrytými neurony je 100 352 vah + 128 biasů, a na výstupu 1 280 vah + 10 biasů.":
            st.success("Správně! (784 x 128 + 128) + (128 x 10 + 10) = 100 480 + 1 290 = 101 770 parametrů.")


if __name__ == "__main__":
    render_neural_networks_theory_view()
