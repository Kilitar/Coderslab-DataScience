"""
Prework Session 2: Úvod do neuronových sítí (Introduction to Neural Networks)
============================================================================
Interaktivní výukový modul pro přípravu na Den 3 (Session 2):
1. Biologická inspirace vs. matematický model umělého neuronu.
2. Interaktivní Neuron Playground (vážený součet, bias a nelineární aktivace).
3. Interaktivní srovnávač 6 aktivačních funkcí a jejich derivací (demonstrace mizejícího gradientu).
4. Interaktivní krokový simulátor sítě ze zadání (předpověď ceny auta se 13 parametry).
5. Typologie architektur: Perceptron, MLP, CNN, RNN a moderní Transformery.
6. Interaktivní vědomostní kvíz k ověření pochopení.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def render_neural_networks_intro_view():
    st.title("🧠 Prework Session 2: Úvod do neuronových sítí")
    st.markdown(
        "**Příprava na Den 3 (Session 2):** Seznámení se základním stavebním kamenem hlubokého učení (Deep Learning) – "
        "od biologické analogie přes matematický model umělého neuronu až po aktivační funkce, architekturu vrstev "
        "a problém mizejícího gradientu."
    )

    # Horní KPI karty
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Základní stavební jednotka", "Neuron (Node)", delta="Soma + Axon + Synapse")
    with c2:
        st.metric("Počet neuronů v mozku", "≈ 86 miliard", delta="Až 10¹⁵ synapsí")
    with c3:
        st.metric("Klíč k nelinearitě", "Aktivační funkce", delta="ReLU / Sigmoid / Tanh")
    with c4:
        st.metric("Učení neuronu", "Minimalizace ztráty", delta="Zpětné šíření (Backprop)")

    st.markdown("---")

    # Záložky modulu
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔬 Biologický vs. Umělý neuron",
        "🎛️ Neuron Playground",
        "📈 Průzkumník aktivačních funkcí",
        "🚗 Příklad sítě: Cena auta (13 vah)",
        "🏛️ Typologie architektur & Kvíz"
    ])

    # TAB 1: Biologie vs Matematika
    with tab1:
        st.subheader("1. Biologická inspirace a princip umělého neuronu")
        st.markdown(
            "Lidský mozek se skládá z přibližně 86 miliard neuronů. "
            "Umělé neuronové sítě (**Artificial Neural Networks – ANN**) vyvinuté v základu Warrenem McCullochem "
            "a Walterem Pittsem (1943) převádějí tento biologický zázrak do zjednodušeného, ale mocného matematického modelu."
        )

        col_bio1, col_bio2 = st.columns([1, 1])
        with col_bio1:
            st.markdown("##### 🧬 Biologický neuron:")
            st.write(
                "- **Dendrity:** Vstupní rozvětvená vlákna přijímající chemické a elektrické signály od sousedních buněk.\n"
                "- **Soma (Buněčné tělo):** Jádro buňky, které sčítá příchozí potenciály.\n"
                "- **Akční potenciál:** Pokud součet překročí biochemický práh excitace, buňka vystřelí impuls.\n"
                "- **Axon:** Dlouhý výstupní výběžek vedoucí vzruch dál.\n"
                "- **Synapse:** Spojení s proměnlivou propustností (posiluje se učením – tzv. Hebbovo pravidlo)."
            )

        with col_bio2:
            st.markdown("##### 💻 Umělý neuron (Matematický model):")
            st.write(
                r"""
                - **Vstupy ($x_1, \dots, x_n$):** Číselné hodnoty prediktorů (např. výkon motoru, věk, glykémie).
                - **Váhy ($w_1, \dots, w_n$):** Multiplikátory důležitosti každého vstupu (analogie síly synapse).
                - **Vážený součet + Bias ($z$):** Lineární kombinace $z = \sum w_i x_i + b = \mathbf{w}^T \mathbf{x} + b$.
                - **Aktivační funkce ($f(z)$):** Nelineární transformace rozhodující o výstupu.
                - **Výstup ($a$):** Výsledné číslo předané dalším vrstvám sítě.
                """
            )

        # Plotly interaktivní schéma proudění signálu v neuronu
        st.markdown("##### 🔄 Interaktivní model toku signálu jedním neuronem")
        fig_flow = go.Figure(go.Sankey(
            node=dict(
                pad=20,
                thickness=25,
                line=dict(color="black", width=0.5),
                label=["Vstup x₁", "Vstup x₂", "Bias b", "Váha w₁", "Váha w₂", "Suma ∑ (Soma)", "Aktivace f(z)", "Výstup a"],
                color=["#60a5fa", "#60a5fa", "#f59e0b", "#93c5fd", "#93c5fd", "#3b82f6", "#10b981", "#8b5cf6"]
            ),
            link=dict(
                source=[0, 1, 3, 4, 2, 5, 6],
                target=[3, 4, 5, 5, 5, 6, 7],
                value=[10, 10, 10, 10, 5, 25, 25],
                color=["#bfdbfe", "#bfdbfe", "#93c5fd", "#93c5fd", "#fde68a", "#a7f3d0", "#ddd6fe"]
            )
        ))
        fig_flow.update_layout(
            title_text="Proudění dat v umělém neuronu: Vstupy → Vahy → Sumace → Nelineární aktivace → Výstup",
            height=320,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_flow, width="stretch")

    # TAB 2: Neuron Playground
    with tab2:
        st.subheader("2. Interaktivní Neuron Playground")
        st.markdown(
            "Vyzkoušejte si, jak změna vstupů, vah a biasu mění vnitřní stav neuronu $z$ "
            "a výslednou aktivaci $a = f(z)$."
        )

        col_p_in, col_p_res = st.columns([1, 1])

        with col_p_in:
            st.markdown("##### ⚙️ Nastavení parametrů neuronu:")
            c_in1, c_in2 = st.columns(2)
            with c_in1:
                in_x1 = st.slider("Vstup x₁", -5.0, 5.0, 2.0, 0.1)
                in_w1 = st.slider("Váha w₁", -3.0, 3.0, 0.8, 0.1)
            with c_in2:
                in_x2 = st.slider("Vstup x₂", -5.0, 5.0, -1.0, 0.1)
                in_w2 = st.slider("Váha w₂", -3.0, 3.0, -1.2, 0.1)

            in_bias = st.slider("Bias (posun b)", -5.0, 5.0, 0.5, 0.1)
            act_choice = st.selectbox(
                "Zvolte aktivační funkci f(z):",
                ["Lineární (f(z) = z)", "Sigmoida (Logistic)", "Tanh (Hyperbolický tangens)", "ReLU (Rectified Linear)", "Leaky ReLU (α = 0.1)"]
            )

        # Výpočty
        z_score = (in_w1 * in_x1) + (in_w2 * in_x2) + in_bias

        if "Lineární" in act_choice:
            act_val = z_score
            func_name = "Linear"
        elif "Sigmoida" in act_choice:
            act_val = 1.0 / (1.0 + np.exp(-z_score))
            func_name = "Sigmoid"
        elif "Tanh" in act_choice:
            act_val = np.tanh(z_score)
            func_name = "Tanh"
        elif "ReLU" in act_choice and "Leaky" not in act_choice:
            act_val = max(0.0, z_score)
            func_name = "ReLU"
        else:
            act_val = z_score if z_score > 0 else 0.1 * z_score
            func_name = "Leaky ReLU"

        with col_p_res:
            st.markdown("##### 📊 Stav a výstup neuronu:")
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Vážený součet (z)", f"{z_score:+.3f}", delta=f"w₁x₁ + w₂x₂ + b")
            with m2:
                st.metric("Výstupní aktivace a = f(z)", f"{act_val:+.3f}", delta=func_name)

            st.write(
                r"""
                **Krokový rozpis výpočtu:**
                """
            )
            st.code(
                f"1. w1 * x1 = ({in_w1:.2f}) * ({in_x1:.2f}) = {in_w1 * in_x1:+.3f}\n"
                f"2. w2 * x2 = ({in_w2:.2f}) * ({in_x2:.2f}) = {in_w2 * in_x2:+.3f}\n"
                f"3. Lineární suma: z = {in_w1 * in_x1:+.3f} + ({in_w2 * in_x2:+.3f}) + ({in_bias:+.2f}) = {z_score:+.3f}\n"
                f"4. Aktivace: a = {func_name}({z_score:+.3f}) = {act_val:+.4f}"
            )

        # Graf polohy neuronu na křivce aktivace
        z_grid = np.linspace(-6, 6, 200)
        if "Lineární" in act_choice:
            y_grid = z_grid
        elif "Sigmoida" in act_choice:
            y_grid = 1.0 / (1.0 + np.exp(-z_grid))
        elif "Tanh" in act_choice:
            y_grid = np.tanh(z_grid)
        elif "ReLU" in act_choice and "Leaky" not in act_choice:
            y_grid = np.maximum(0, z_grid)
        else:
            y_grid = np.where(z_grid > 0, z_grid, 0.1 * z_grid)

        fig_act_point = go.Figure()
        fig_act_point.add_trace(go.Scatter(
            x=z_grid, y=y_grid, mode="lines", name=f"{func_name} křivka",
            line=dict(color="#2563eb", width=3)
        ))
        fig_act_point.add_trace(go.Scatter(
            x=[z_score], y=[act_val], mode="markers+text", name="Pracovní bod neuronu",
            text=[f"  z={z_score:+.2f}, a={act_val:+.2f}"],
            textposition="top left",
            marker=dict(size=14, color="#ef4444", symbol="diamond")
        ))
        fig_act_point.update_layout(
            title=f"Pracovní bod na vybrané aktivační funkci ({func_name})",
            xaxis_title="Vážený součet z = wᵀx + b",
            yaxis_title="Aktivace a = f(z)",
            height=350,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_act_point, width="stretch")

    # TAB 3: Aktivační funkce a derivace
    with tab3:
        st.subheader("3. Průzkumník aktivačních funkcí a mizející gradient (Vanishing Gradient)")
        st.markdown(
            "Aktivační funkce vnášejí do sítě **nelinearitu**. Bez nich by celá síť bez ohledu na počet vrstev "
            "kolabovala do jediné lineární regrese. "
            "Zásadní vlastností je však také jejich **derivace $f'(z)$**, která se používá při zpětném šíření chyby (Backpropagation)."
        )

        show_derivative = st.checkbox("Zobrazit také derivaci funkce f'(z) (klíčové pro Backpropagation)", value=True)

        z_axis = np.linspace(-5, 5, 250)

        # Výpočty funkcí a derivací
        # 1. Linear
        f_lin = z_axis
        df_lin = np.ones_like(z_axis)

        # 2. Sigmoid
        f_sig = 1.0 / (1.0 + np.exp(-z_axis))
        df_sig = f_sig * (1.0 - f_sig)

        # 3. Tanh
        f_tanh = np.tanh(z_axis)
        df_tanh = 1.0 - f_tanh**2

        # 4. ReLU
        f_relu = np.maximum(0, z_axis)
        df_relu = np.where(z_axis > 0, 1.0, 0.0)

        # 5. Leaky ReLU
        f_lrelu = np.where(z_axis > 0, z_axis, 0.1 * z_axis)
        df_lrelu = np.where(z_axis > 0, 1.0, 0.1)

        col_f1, col_f2 = st.columns([1, 1])

        with col_f1:
            fig_funcs = go.Figure()
            fig_funcs.add_trace(go.Scatter(x=z_axis, y=f_sig, mode="lines", name="Sigmoid", line=dict(width=2.5)))
            fig_funcs.add_trace(go.Scatter(x=z_axis, y=f_tanh, mode="lines", name="Tanh", line=dict(width=2.5)))
            fig_funcs.add_trace(go.Scatter(x=z_axis, y=f_relu, mode="lines", name="ReLU", line=dict(width=2.5)))
            fig_funcs.add_trace(go.Scatter(x=z_axis, y=f_lrelu, mode="lines", name="Leaky ReLU (α=0.1)", line=dict(dash="dot", width=2.5)))
            fig_funcs.update_layout(
                title="Srovnání tvaru aktivačních funkcí f(z)",
                xaxis_title="Vstupní hodnota z",
                yaxis_title="f(z)",
                height=400,
                margin=dict(l=10, r=10, t=50, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_funcs, width="stretch")

        with col_f2:
            fig_derivs = go.Figure()
            fig_derivs.add_trace(go.Scatter(x=z_axis, y=df_sig, mode="lines", name="Sigmoid' (Max = 0.25)", line=dict(width=2.5)))
            fig_derivs.add_trace(go.Scatter(x=z_axis, y=df_tanh, mode="lines", name="Tanh' (Max = 1.0)", line=dict(width=2.5)))
            fig_derivs.add_trace(go.Scatter(x=z_axis, y=df_relu, mode="lines", name="ReLU' (0 nebo 1)", line=dict(width=2.5)))
            fig_derivs.add_trace(go.Scatter(x=z_axis, y=df_lrelu, mode="lines", name="Leaky ReLU' (0.1 nebo 1)", line=dict(dash="dot", width=2.5)))
            fig_derivs.update_layout(
                title="Derivace f'(z) – Odhalení mizejícího gradientu",
                xaxis_title="Vstupní hodnota z",
                yaxis_title="Derivace f'(z)",
                height=400,
                margin=dict(l=10, r=10, t=50, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_derivs, width="stretch")

        st.warning(
            "⚠️ **Proč je ReLU králem moderního Deep Learningu?**\n\n"
            "- U **Sigmoidy** je maximální možná derivace pouhých **0.25** (a pro $|z| > 3$ padá k 0). "
            "Při zpětném šíření přes 10 vrstev je gradient utlumen faktorem $0.25^{10} \\approx 10^{-6}$, "
            "což vede k **úplnému zamrznutí učení (Vanishing Gradient)**.\n"
            "- U **ReLU** je derivace pro všechna kladná čísla konstantně **1.0** $\\implies$ "
            "gradient prochází libovolným počtem vrstev bez oslabení!"
        )

    # TAB 4: Příklad Cena auta
    with tab4:
        st.subheader("4. Detailní průchod příkladem ze zadání kurzu: Předpověď ceny auta")
        st.markdown(
            "V zadání kurzu je představen konkrétní model sítě: \n"
            "- **Vstupy:** $x_1 = 300\\ \\text{HP}$ (výkon), $x_2 = 2\\ \\text{roky}$ (stáří).\n"
            "- **Architektura:** Vstupní vrstva (2 neurony) $\\to$ 1 skrytá vrstva (3 neurony) $\\to$ Výstupní neuron (1 cena).\n"
            "- **Celkem 13 parametrů:** $2 \\times 3 = 6$ vah skryté vrstvy + 3 biasy + $3 \\times 1 = 3$ váhy výstupu + 1 bias výstupu."
        )

        col_car_ctrl, col_car_vis = st.columns([1, 1])

        with col_car_ctrl:
            st.markdown("##### 🚗 Vstupní data vozidla:")
            car_hp = st.slider("Výkon motoru (HP)", 50, 600, 300, 10)
            car_age = st.slider("Stáří vozidla (roky)", 0, 25, 2, 1)

            st.markdown("##### ⚙️ Nastavení vah sítě (ukázkové parametry):")
            # Defaultní rozumné váhy pro demonstraci
            w11, w21, b1 = 45.0, -1500.0, 5000.0
            w12, w22, b2 = 35.0, -800.0, 2000.0
            w13, w23, b3 = 20.0, -2200.0, 8000.0
            v1, v2, v3, b_out = 0.45, 0.35, 0.20, 10000.0

            st.caption("Přednastavené váhy odrážejí logiku: vyšší výkon zvyšuje cenu (+), stáří cenu sráží (-).")

        # Výpočty jednotlivých neuronů
        z1 = (w11 * car_hp) + (w21 * car_age) + b1
        z2 = (w12 * car_hp) + (w22 * car_age) + b2
        z3 = (w13 * car_hp) + (w23 * car_age) + b3

        # S lineární aktivací ze zadání
        a1, a2, a3 = z1, z2, z3
        predicted_price = (v1 * a1) + (v2 * a2) + (v3 * a3) + b_out

        with col_car_vis:
            st.markdown("##### 💰 Výsledná predikce sítě:")
            st.metric("Odhadnutá cena automobilu", f"{predicted_price:,.0f} USD".replace(",", " "))

            st.markdown("##### 🔢 Výpočty v jednotlivých uzlech:")
            st.write(
                f"- **Neuron 1 skryté vrstvy:**\n"
                f"  $z_1 = ({w11} \\times {car_hp}) + ({w21} \\times {car_age}) + {b1} = {z1:,.0f}$\n\n"
                f"- **Neuron 2 skryté vrstvy:**\n"
                f"  $z_2 = ({w12} \\times {car_hp}) + ({w22} \\times {car_age}) + {b2} = {z2:,.0f}$\n\n"
                f"- **Neuron 3 skryté vrstvy:**\n"
                f"  $z_3 = ({w13} \\times {car_hp}) + ({w23} \\times {car_age}) + {b3} = {z3:,.0f}$\n\n"
                f"- **Výstupní neuron (finální agregace):**\n"
                f"  $\\hat{{y}} = ({v1} \\times {a1:,.0f}) + ({v2} \\times {a2:,.0f}) + ({v3} \\times {a3:,.0f}) + {b_out} = \\mathbf{{{predicted_price:,.0f}\\ \\text{{USD}}}}$"
            )

        # Interaktivní graf topologie sítě v Plotly
        st.markdown("##### 🕸️ Vizuální topologie sítě (2 vstupy → 3 skryté neurony → 1 výstup)")
        edge_x = []
        edge_y = []

        # Pozice uzlů
        nodes_pos = {
            "HP (x₁)": (0, 0.7),
            "Age (x₂)": (0, 0.3),
            "Neuron 1 (h₁)": (1, 0.8),
            "Neuron 2 (h₂)": (1, 0.5),
            "Neuron 3 (h₃)": (1, 0.2),
            "Cena (y)": (2, 0.5)
        }

        # Hrany vstup -> skrytá
        in_nodes = ["HP (x₁)", "Age (x₂)"]
        hid_nodes = ["Neuron 1 (h₁)", "Neuron 2 (h₂)", "Neuron 3 (h₃)"]
        out_node = "Cena (y)"

        for i_n in in_nodes:
            for h_n in hid_nodes:
                edge_x.extend([nodes_pos[i_n][0], nodes_pos[h_n][0], None])
                edge_y.extend([nodes_pos[i_n][1], nodes_pos[h_n][1], None])

        # Hrany skrytá -> výstup
        for h_n in hid_nodes:
            edge_x.extend([nodes_pos[h_n][0], nodes_pos[out_node][0], None])
            edge_y.extend([nodes_pos[h_n][1], nodes_pos[out_node][1], None])

        fig_net = go.Figure()
        fig_net.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1.5, color="#cbd5e1"),
            hoverinfo="none",
            mode="lines"
        ))

        node_x = [pos[0] for pos in nodes_pos.values()]
        node_y = [pos[1] for pos in nodes_pos.values()]
        node_text = list(nodes_pos.keys())
        node_colors = ["#60a5fa", "#60a5fa", "#3b82f6", "#3b82f6", "#3b82f6", "#10b981"]

        fig_net.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            marker=dict(size=28, color=node_colors, line=dict(width=2, color="#1e293b")),
            hoverinfo="text"
        ))

        fig_net.update_layout(
            title="Architektura sítě: 9 vah + 4 biasy = 13 trénovatelných parametrů",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=350,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_net, width="stretch")

    # TAB 5: Typologie architektur a kvíz
    with tab5:
        st.subheader("5. Typologie architektur a znalostní sebekontrola")
        st.markdown(
            "Přehled 4 základních rodin neuronových sítí a jejich specializace v praxi:"
        )

        t_col1, t_col2 = st.columns(2)
        with t_col1:
            st.info(
                "#### 1. Perceptron & Vícevrstvý perceptron (MLP)\n"
                "- **Struktura:** Plně propojené dopředné vrstvy (Dense / Fully Connected).\n"
                "- **Využití:** Tabulární data, regrese a klasifikace (např. medicínská diagnostika).\n"
                "- **Univerzální aproximátor:** Již 1 skrytá vrstva s nelinearitou dokáže aproximovat jakoukoliv spojitou funkci."
            )
            st.success(
                "#### 3. Rekurentní sítě (RNN, LSTM, GRU)\n"
                "- **Struktura:** Vnitřní zpětné vazby uchovávající skrytý stav v čase (paměť).\n"
                "- **Využití:** Časové řady, zvuk, přirozený jazyk (NLP).\n"
                "- **Moderní evoluce:** Od roku 2023–2026 nahrazovány **Transformery** s Attention mechanismem."
            )

        with t_col2:
            st.warning(
                "#### 2. Konvoluční sítě (CNN)\n"
                "- **Struktura:** Konvoluční filtry se sdílenými vahami a pooling vrstvy.\n"
                "- **Využití:** Zpracování obrazu, detekce objektů, analýza lékařských snímků (rentgen, MRI).\n"
                "- **Výhoda:** Prostorová invariance (rozpozná kočku kdekoliv v záběru)."
            )
            st.error(
                "#### 4. Moderní stav 2026: Transformery & Foundation Models\n"
                "- **Struktura:** Self-Attention mechanismus bez rekurence, plná paralelizace.\n"
                "- **Využití:** LLM (GPT, LLaMA), Vision Transformers (ViT), multimodální AI."
            )

        st.markdown("---")

        # Interaktivní kvíz
        st.subheader("📝 Rychlý vědomostní kvíz k ověření pochopení")

        q1 = st.radio(
            "1. Co by se stalo, kdybychom ve všech vrstvách sítě použili pouze lineární aktivační funkci f(z) = z?",
            [
                "Síť by se učila rychleji a přesněji.",
                "Celá vícevrstvá síť by zkolabovala do jediné lineární regrese bez schopnosti modelovat nelineární vztahy.",
                "Síť by byla náchylná k explozi gradientu.",
                "Výstup sítě by byl vždy roven nule."
            ]
        )

        q2 = st.radio(
            "2. Jaká je hlavní výhoda aktivační funkce ReLU oproti Sigmoidě ve skrytých vrstvách?",
            [
                "ReLU je ohraničená v intervalu (-1, 1).",
                "ReLU má nulový gradient pro kladná čísla.",
                "ReLU nesaturuje v kladné oblasti a její derivace je 1, což eliminuje problém mizejícího gradientu (Vanishing Gradient).",
                "ReLU je hladká a má všude spojité derivace."
            ]
        )

        q3 = st.radio(
            "3. Kolik celkem trénovatelných parametrů (vah a biasů) má síť se 2 vstupy, 3 neurony ve skryté vrstvě a 1 výstupním neuronem?",
            [
                "6 parametrů",
                "9 parametrů",
                "13 parametrů (6 vah skryté vrstvy + 3 biasy + 3 váhy výstupu + 1 bias výstupu)",
                "15 parametrů"
            ]
        )

        if st.button("Vyhodnotit kvíz", width="stretch"):
            score = 0
            if "zkolabovala do jediné lineární regrese" in q1:
                score += 1
            if "eliminuje problém mizejícího gradientu" in q2:
                score += 1
            if "13 parametrů" in q3:
                score += 1

            if score == 3:
                st.balloons()
                st.success(f"🎉 Skvěle! 3 ze 3 správně! Máte perfektní teoretický základ pro Session 2 (Den 3).")
            else:
                st.warning(f"Získali jste {score} ze 3 bodů. Projděte si záložky s aktivačními funkcemi a příkladem auta.")


render_neural_networks_intro_view()
