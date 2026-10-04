"""
Den 3 Extras: 🔢 Krok za krokem: Backpropagation kalkulačka (s erraty ze slidů)
==============================================================================
- Síť 2-2-2 se sigmoidální aktivací a MSE loss přesně dle přednášky (How neural networks work and learn).
- Přesná reprodukce forward passu i backward passu s libovolnými parametry.
- Interaktivní ladění vah, biasů a learning rate.
- Detailní errata a rozbor tiskových chyb ve slidech Coderslab kurzu (str. 22-28).
- Simulace více epoch s grafem poklesu chyby (loss curve).
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Backpropagation kalkulačka", page_icon="🔢", layout="wide")

st.markdown("""
# 🔢 Krok za krokem: Backpropagation kalkulačka
### Interaktivní průchod sítí 2-2-2, výpočet gradientů a odhalení tiskových chyb ve slidech
""")

st.info("""
Tato interaktivní kalkulačka přesně modeluje **příklad z oficiálních přednáškových slidů kurzu** 
(*How neural networks work and learn*, str. 18–29). Můžeš si krok za krokem projít dopředný průchod (Forward pass), 
zpětné šíření chyby (Backpropagation), řetízkové pravidlo pro všechny váhy i biasy a podívat se, jak síť konverguje napříč epochami.
""")

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

def d_sigmoid(out):
    return out * (1.0 - out)

# --- Konfigurace parametrů sítě v hlavní ploše (nezasahuje do navigace v levém docku) ---
with st.expander("⚙️ Počáteční parametry sítě 2-2-2 (přesně dle přednášky)", expanded=True):
    col_p1, col_p2, col_p3 = st.columns(3)
    
    with col_p1:
        st.markdown("#### 1. Vstupy & Cíle")
        c1, c2 = st.columns(2)
        with c1:
            x1 = st.number_input("Vstup x₁:", value=3.0, step=0.5)
            t1 = st.number_input("Cíl t₁ (Class 0):", value=0.1, step=0.05, min_value=0.0, max_value=1.0)
        with c2:
            x2 = st.number_input("Vstup x₂:", value=5.0, step=0.5)
            t2 = st.number_input("Cíl t₂ (Class 1):", value=0.9, step=0.05, min_value=0.0, max_value=1.0)
        
        st.markdown("#### 4. Hyperparametry učení")
        lr = st.slider("Learning Rate (η):", min_value=0.01, max_value=1.0, value=0.30, step=0.01)
        update_biases = st.checkbox("Aktualizovat i biasy (slidy to přeskočily)", value=False)

    with col_p2:
        st.markdown("#### 2. Váhy: Vstup $\\to$ Skrytá vrstva")
        c_w1, c_w2 = st.columns(2)
        with c_w1:
            w1 = st.number_input("w₁ (x₁→h₁):", value=0.10, step=0.05)
            w3 = st.number_input("w₃ (x₂→h₁):", value=0.30, step=0.05)
        with c_w2:
            w2 = st.number_input("w₂ (x₁→h₂):", value=0.20, step=0.05)
            w4 = st.number_input("w₄ (x₂→h₂):", value=0.40, step=0.05)
        b1 = st.number_input("Bias b₁ (pro h₁ i h₂):", value=0.25, step=0.05)

    with col_p3:
        st.markdown("#### 3. Váhy: Skrytá $\\to$ Výstupní vrstva")
        c_w3, c_w4 = st.columns(2)
        with c_w3:
            w5 = st.number_input("w₅ (h₁→o₁):", value=0.50, step=0.05)
            w7 = st.number_input("w₇ (h₂→o₁):", value=0.70, step=0.05)
        with c_w4:
            w6 = st.number_input("w₆ (h₁→o₂):", value=0.60, step=0.05)
            w8 = st.number_input("w₈ (h₂→o₂):", value=0.80, step=0.05)
        b2 = st.number_input("Bias b₂ (pro o₁ i o₂):", value=0.40, step=0.05)

# --- 1. KROK: FORWARD PASS ---
sum_h1 = x1 * w1 + x2 * w3 + b1
out_h1 = sigmoid(sum_h1)

sum_h2 = x1 * w2 + x2 * w4 + b1
out_h2 = sigmoid(sum_h2)

sum_o1 = out_h1 * w5 + out_h2 * w7 + b2
out_o1 = sigmoid(sum_o1)

sum_o2 = out_h1 * w6 + out_h2 * w8 + b2
out_o2 = sigmoid(sum_o2)

# Ztráta (MSE): E = 1/2 * (t - y)^2
e1 = 0.5 * ((t1 - out_o1) ** 2)
e2 = 0.5 * ((t2 - out_o2) ** 2)
total_loss = e1 + e2

# --- 2. KROK: BACKPROPAGATION ---
# Výstupní vrstva
# dE/dout_o1 = (out_o1 - t1)
# dout_o1/dsum_o1 = out_o1 * (1 - out_o1)
# delta_o1 = (out_o1 - t1) * out_o1 * (1 - out_o1)
delta_o1 = (out_o1 - t1) * d_sigmoid(out_o1)
delta_o2 = (out_o2 - t2) * d_sigmoid(out_o2)

grad_w5 = delta_o1 * out_h1
grad_w7 = delta_o1 * out_h2
grad_w6 = delta_o2 * out_h1
grad_w8 = delta_o2 * out_h2

grad_b2 = delta_o1 + delta_o2

# Skrytá vrstva (řetízkové pravidlo přes oba výstupy)
# delta_h1 = (delta_o1 * w5 + delta_o2 * w6) * out_h1 * (1 - out_h1)
# delta_h2 = (delta_o1 * w7 + delta_o2 * w8) * out_h2 * (1 - out_h2)
delta_h1 = (delta_o1 * w5 + delta_o2 * w6) * d_sigmoid(out_h1)
delta_h2 = (delta_o1 * w7 + delta_o2 * w8) * d_sigmoid(out_h2)

grad_w1 = delta_h1 * x1
grad_w3 = delta_h1 * x2
grad_w2 = delta_h2 * x1
grad_w4 = delta_h2 * x2

grad_b1 = delta_h1 + delta_h2

# Aktualizace vah
w5_new = w5 - lr * grad_w5
w7_new = w7 - lr * grad_w7
w6_new = w6 - lr * grad_w6
w8_new = w8 - lr * grad_w8

w1_new = w1 - lr * grad_w1
w3_new = w3 - lr * grad_w3
w2_new = w2 - lr * grad_w2
w4_new = w4 - lr * grad_w4

# --- VIZUALIZACE A REPORT ---
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("Predikce out(o₁)", f"{out_o1:.4f}", help=f"Cíl t₁ = {t1}")
col_m2.metric("Predikce out(o₂)", f"{out_o2:.4f}", help=f"Cíl t₂ = {t2}")
col_m3.metric("Chyba E = E₁ + E₂", f"{total_loss:.4f}", help="Skutečná hodnota 1/2*(t1-y1)^2 + 1/2*(t2-y2)^2")
col_m4.metric("Klíčový gradient ∂E/∂w₅", f"{grad_w5:.4f}", delta=f"Nové w₅: {w5_new:.4f}")

st.divider()

# Sekce Errata
with st.expander("🔬 DŮLEŽITÉ: Errata a odhalení tiskových chyb ve slidech kurzu", expanded=True):
    st.markdown(r"""
    Pokud jsi četl slidy *How neural networks work and learn* (str. 22–28) a snažil ses spočítat vzorce ručně, 
    pravděpodobně ses zasekl na nesrovnalostech. Zde je přesný rozbor, **proč čísla ve slidech nesedí a co v nich autor překlepl**:
    
    1. **Chyba ve výpočtu celkové ztráty (str. 23):**
       * *Text ve slidech:* $E = \frac{1}{2}(0.1 - 0.818)^2 + \frac{1}{2}(0.9 - 0.844)^2 = 0.516 + 0.554 = 1.071$.
       * *Matematická realita:* 
         * $(-0.718)^2 = 0.5155$. Po vynásobení $\frac{1}{2}$ je $E_1 = \mathbf{0.2578}$ (autor zapomněl podělit dvěma).
         * $(0.9 - 0.844)^2 = (0.056)^2 = \mathbf{0.003136}$. Hodnota $0.554$ ve slidech je nesmyslný překlep.
         * Skutečná celková ztráta je $E = 0.2578 + 0.0016 = \mathbf{0.2594}$, nikoliv $1.071$!
    
    2. **Záměna indexu při výpočtu gradientu $\partial E / \partial w_5$ (str. 27):**
       * *Vzorec:* $\frac{\partial E}{\partial w_5} = (\text{out}_{o1} - t_1) \cdot [\text{out}_{o1}(1 - \text{out}_{o1})] \cdot \mathbf{\text{out}_{h1}}$.
       * *Čísla ve slidech:* $(0.818 - 0.1) \cdot (0.818 \cdot (1 - 0.818)) \cdot \mathbf{0.818} = 0.0874$.
       * *Realita:* Autor dosadil na konci $\text{out}_{o1} = 0.818$ místo $\text{out}_{h1} = \mathbf{0.886}$.
       * Správný gradient je $(0.718) \cdot (0.14888) \cdot (0.886) = \mathbf{0.0947}$.
    
    3. **Tisková chyba v aktualizaci váhy (str. 28):**
       * *Text ve slidech:* $w_{5\text{new}} = 0.5 - 0.3 \cdot \mathbf{0.0238} = 0.5 - 0.02622 = 0.47378$.
       * *Rozpor:* Hodnota $0.0238$ se v celém dokumentu předtím vůbec nevyskytuje. Číslo $0.02622$ je ve skutečnosti $0.3 \times \mathbf{0.0874}$ (autor udělal překlep v zápisu čísla $0.0874$).
    
    > **Shrnutí:** Náš kalkulátor níže počítá **matematicky správné hodnoty** a umožňuje ti vidět čistou rigorózní derivaci bez těchto tiskových šumů.
    """)

# --- TABULKY: FORWARD A BACKWARD PASS ---
tab_fwd, tab_bwd, tab_sim = st.tabs([
    "1️⃣ Dopředný průchod (Forward Pass)",
    "2️⃣ Zpětné šíření a gradienty (Backward Pass)",
    "📈 Simulace tréninku (Multi-Epoch)",
])

with tab_fwd:
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.subheader("Skrytá vrstva (Neurony h₁ a h₂)")
        df_hidden = pd.DataFrame([
            {
                "Neuron": "h₁",
                "Lineární suma (sum)": f"3·{w1:.2f} + 5·{w3:.2f} + {b1:.2f} = {sum_h1:.4f}",
                "Aktivace Sigmoid": f"1 / (1 + e^{{-{sum_h1:.4f}}})",
                "Výstup (out)": f"{out_h1:.4f}",
                "Hodnota ve slidech": "0.886",
            },
            {
                "Neuron": "h₂",
                "Lineární suma (sum)": f"3·{w2:.2f} + 5·{w4:.2f} + {b1:.2f} = {sum_h2:.4f}",
                "Aktivace Sigmoid": f"1 / (1 + e^{{-{sum_h2:.4f}}})",
                "Výstup (out)": f"{out_h2:.4f}",
                "Hodnota ve slidech": "0.945",
            },
        ])
        st.dataframe(df_hidden, hide_index=True, width="stretch")

    with col_f2:
        st.subheader("Výstupní vrstva (Neurony o₁ a o₂)")
        df_out = pd.DataFrame([
            {
                "Neuron": "o₁ (Class 0)",
                "Lineární suma (sum)": f"{out_h1:.3f}·{w5:.2f} + {out_h2:.3f}·{w7:.2f} + {b2:.2f} = {sum_o1:.4f}",
                "Výstup (out)": f"{out_o1:.4f}",
                "Cíl (target)": f"{t1:.2f}",
                "Chyba E₁": f"{e1:.6f}",
                "Hodnota ve slidech": "0.818",
            },
            {
                "Neuron": "o₂ (Class 1)",
                "Lineární suma (sum)": f"{out_h1:.3f}·{w6:.2f} + {out_h2:.3f}·{w8:.2f} + {b2:.2f} = {sum_o2:.4f}",
                "Výstup (out)": f"{out_o2:.4f}",
                "Cíl (target)": f"{t2:.2f}",
                "Chyba E₂": f"{e2:.6f}",
                "Hodnota ve slidech": "0.844",
            },
        ])
        st.dataframe(df_out, hide_index=True, width="stretch")

with tab_bwd:
    st.subheader("Aktualizace všech vah sítě (1. epocha, $\\eta = 0.3$)")
    
    weights_summary = [
        {
            "Váha": "w₅ (h₁ → o₁)",
            "Původní": f"{w5:.4f}",
            "Gradient ∂E/∂w": f"{grad_w5:.5f}",
            "Posun (-η·grad)": f"{-lr * grad_w5:.5f}",
            "Nová váha": f"{w5_new:.5f}",
            "Slidy (s překlepem)": "0.47378",
        },
        {
            "Váha": "w₇ (h₂ → o₁)",
            "Původní": f"{w7:.4f}",
            "Gradient ∂E/∂w": f"{grad_w7:.5f}",
            "Posun (-η·grad)": f"{-lr * grad_w7:.5f}",
            "Nová váha": f"{w7_new:.5f}",
            "Slidy (s překlepem)": "Nezmíněno",
        },
        {
            "Váha": "w₆ (h₁ → o₂)",
            "Původní": f"{w6:.4f}",
            "Gradient ∂E/∂w": f"{grad_w6:.5f}",
            "Posun (-η·grad)": f"{-lr * grad_w6:.5f}",
            "Nová váha": f"{w6_new:.5f}",
            "Slidy (s překlepem)": "Nezmíněno",
        },
        {
            "Váha": "w₈ (h₂ → o₂)",
            "Původní": f"{w8:.4f}",
            "Gradient ∂E/∂w": f"{grad_w8:.5f}",
            "Posun (-η·grad)": f"{-lr * grad_w8:.5f}",
            "Nová váha": f"{w8_new:.5f}",
            "Slidy (s překlepem)": "Nezmíněno",
        },
        {
            "Váha": "w₁ (x₁ → h₁)",
            "Původní": f"{w1:.4f}",
            "Gradient ∂E/∂w": f"{grad_w1:.5f}",
            "Posun (-η·grad)": f"{-lr * grad_w1:.5f}",
            "Nová váha": f"{w1_new:.5f}",
            "Slidy (s překlepem)": "Nezmíněno",
        },
        {
            "Váha": "w₃ (x₂ → h₁)",
            "Původní": f"{w3:.4f}",
            "Gradient ∂E/∂w": f"{grad_w3:.5f}",
            "Posun (-η·grad)": f"{-lr * grad_w3:.5f}",
            "Nová váha": f"{w3_new:.5f}",
            "Slidy (s překlepem)": "Nezmíněno",
        },
        {
            "Váha": "w₂ (x₁ → h₂)",
            "Původní": f"{w2:.4f}",
            "Gradient ∂E/∂w": f"{grad_w2:.5f}",
            "Posun (-η·grad)": f"{-lr * grad_w2:.5f}",
            "Nová váha": f"{w2_new:.5f}",
            "Slidy (s překlepem)": "Nezmíněno",
        },
        {
            "Váha": "w₄ (x₂ → h₂)",
            "Původní": f"{w4:.4f}",
            "Gradient ∂E/∂w": f"{grad_w4:.5f}",
            "Posun (-η·grad)": f"{-lr * grad_w4:.5f}",
            "Nová váha": f"{w4_new:.5f}",
            "Slidy (s překlepem)": "Nezmíněno",
        },
    ]
    st.dataframe(pd.DataFrame(weights_summary), hide_index=True, width="stretch")

with tab_sim:
    st.subheader("Simulace konvergence po 500 epochách")
    st.caption("Sleduj, jak se hodnota chyby snižuje s každou iterací gradientního sestupu.")
    
    epochs = st.slider("Počet simulovaných epoch:", min_value=10, max_value=1000, value=250, step=10)
    
    # Rychlá simulace
    curr_w1, curr_w2, curr_w3, curr_w4 = w1, w2, w3, w4
    curr_w5, curr_w6, curr_w7, curr_w8 = w5, w6, w7, w8
    curr_b1, curr_b2 = b1, b2
    
    losses = []
    p_o1_hist = []
    p_o2_hist = []
    
    for _ in range(epochs):
        # fwd
        sh1 = x1 * curr_w1 + x2 * curr_w3 + curr_b1
        oh1 = sigmoid(sh1)
        sh2 = x1 * curr_w2 + x2 * curr_w4 + curr_b1
        oh2 = sigmoid(sh2)
        
        so1 = oh1 * curr_w5 + oh2 * curr_w7 + curr_b2
        oo1 = sigmoid(so1)
        so2 = oh1 * curr_w6 + oh2 * curr_w8 + curr_b2
        oo2 = sigmoid(so2)
        
        cur_loss = 0.5 * ((t1 - oo1) ** 2 + (t2 - oo2) ** 2)
        losses.append(cur_loss)
        p_o1_hist.append(oo1)
        p_o2_hist.append(oo2)
        
        # bwd
        d_o1 = (oo1 - t1) * d_sigmoid(oo1)
        d_o2 = (oo2 - t2) * d_sigmoid(oo2)
        
        gw5 = d_o1 * oh1
        gw7 = d_o1 * oh2
        gw6 = d_o2 * oh1
        gw8 = d_o2 * oh2
        
        d_h1 = (d_o1 * curr_w5 + d_o2 * curr_w6) * d_sigmoid(oh1)
        d_h2 = (d_o1 * curr_w7 + d_o2 * curr_w8) * d_sigmoid(oh2)
        
        gw1 = d_h1 * x1
        gw3 = d_h1 * x2
        gw2 = d_h2 * x1
        gw4 = d_h2 * x2
        
        # update
        curr_w5 -= lr * gw5
        curr_w7 -= lr * gw7
        curr_w6 -= lr * gw6
        curr_w8 -= lr * gw8
        
        curr_w1 -= lr * gw1
        curr_w3 -= lr * gw3
        curr_w2 -= lr * gw2
        curr_w4 -= lr * gw4
        
        if update_biases:
            curr_b2 -= lr * (d_o1 + d_o2)
            curr_b1 -= lr * (d_h1 + d_h2)

    fig_sim = go.Figure()
    fig_sim.add_trace(go.Scatter(y=losses, mode="lines", name="Celková ztráta E", line=dict(color="#ef4444", width=2.5)))
    fig_sim.add_trace(go.Scatter(y=p_o1_hist, mode="lines", name=f"Predikce o₁ (Cíl {t1})", line=dict(color="#3b82f6", dash="dot")))
    fig_sim.add_trace(go.Scatter(y=p_o2_hist, mode="lines", name=f"Predikce o₂ (Cíl {t2})", line=dict(color="#10b981", dash="dot")))
    
    fig_sim.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(title="Epocha"),
        yaxis=dict(title="Hodnota"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig_sim, width="stretch")
    
    c_end1, c_end2, c_end3 = st.columns(3)
    c_end1.metric("Počáteční ztráta E", f"{losses[0]:.4f}")
    c_end2.metric(f"Konečná ztráta po {epochs} epochách", f"{losses[-1]:.6f}")
    c_end3.metric("Úbytek chyby", f"{(1 - losses[-1] / losses[0]) * 100:.1f} %")
