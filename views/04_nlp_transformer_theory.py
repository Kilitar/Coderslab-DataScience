"""
Den 4: NLP – Architektura Transformer & Mechanismus Self-Attention
==================================================================
Podklad: Resources/Day 4 PDF/How_Transformer_works.pdf
Klíčový článek: "Attention Is All You Need" (Vaswani et al., 2017)
Precomputed: 04_NLP/data/bert_precomputed.json
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.title("⚡ Architektura Transformer & Self-Attention")
st.caption(
    "Revoluční architektura 'Attention Is All You Need' (Google 2017), která nahradila rekurentní sítě (RNN/LSTM), "
    "umožnila masivní paralelizaci a stala se základem modelů BERT, GPT a moderních LLM."
)

base_dir = Path(__file__).resolve().parent.parent
json_path = base_dir / "04_NLP" / "data" / "bert_precomputed.json"


@st.cache_data
def load_bert_data():
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


bert_data = load_bert_data()

# Horní KPI karty
c1, c2, c3, c4 = st.columns(4)
c1.metric("Publikováno", "2017", delta="Google Brain & Research")
c2.metric("Časová složitost vazby", "O(1)", delta="Přímá pozornost bez kroků")
c3.metric("Klíčový vzorec", "Scaled Dot-Product", delta="softmax(QK^T / √d_k)V")
c4.metric("Paralelizace", "100 %", delta="Zpracování celé věty naráz")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "🏛️ 1. Architektura: Encoder & Decoder",
    "🔍 2. Mechanika Self-Attention (Q, K, V)",
    "📐 3. Poziční kódování (Positional Encoding)",
    "🛡️ 4. Decoder & Maskovaná pozornost (Causal Masking)"
])

# ==============================================================================
# TAB 1: ARCHITEKTURA ENCODER & DECODER
# ==============================================================================
with tab1:
    st.subheader("Od sekvenčních RNN k plně paralelnímu Transformeru")

    st.markdown(
        r"""
        V dřívějších sekvenčních modelech (**RNN**, **LSTM**) bylo nutné procházet text slovo za slovem v čase: 
        $$h_t = f(h_{t-1}, x_t)$$. 
        Tento přístup trpěl dvěma zásadními problémy:
        1. **Nemožnost efektivní paralelizace na GPU** (výpočet kroku $t$ musel čekat na $t-1$).
        2. **Informační hrdlo (Information Bottleneck)** a mizející gradient při dlouhých větách.

        **Transformer** veškeré rekurence odstranil. Celou větu zpracovává současně v jediném maticovém násobení.
        """
    )

    col_arch1, col_arch2 = st.columns([1, 1])

    with col_arch1:
        st.markdown(
            """
            <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(59, 130, 246, 0.4); border-radius: 10px; padding: 18px; margin-bottom: 15px;">
                <h4 style="color: #60a5fa; margin-top: 0;">🟦 Větev Kódovače (Encoder)</h4>
                <p><strong>Úkol:</strong> Zakódovat vstupní text do bohaté matice kontextových reprezentací.</p>
                <ul>
                    <li><strong>Vstup:</strong> Tokeny + Poziční kódování (Positional Encoding).</li>
                    <li><strong>Multi-Head Self-Attention:</strong> Každé slovo se dívá na všechna ostatní slova ve větě.</li>
                    <li><strong>Add & Norm:</strong> Reziduální spojení (skip connections) + Layer Normalization.</li>
                    <li><strong>Feed-Forward Network (MLP):</strong> Dvě plně propojené vrstvy aplikované nezávisle na každou pozici.</li>
                    <li>Výstup je předán do další vrstvy Encoderu nebo přímo do Decoderu.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_arch2:
        st.markdown(
            """
            <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(168, 85, 247, 0.4); border-radius: 10px; padding: 18px; margin-bottom: 15px;">
                <h4 style="color: #c084fc; margin-top: 0;">🟪 Větev Dekodéru (Decoder)</h4>
                <p><strong>Úkol:</strong> Postupně generovat cílovou sekvenci (např. překlad z jednoho jazyka do druhého).</p>
                <ul>
                    <li><strong>Masked Multi-Head Attention:</strong> Zabraňuje modelu vidět budoucí slova během tréninku (Causal Mask).</li>
                    <li><strong>Cross-Attention:</strong> Dotazy ($Q$) jdou z dekodéru, Klíče ($K$) a Hodnoty ($V$) z Encoderu!</li>
                    <li><strong>Feed-Forward Network (MLP):</strong> Zpracování zkombinovaných vektorů.</li>
                    <li><strong>Linear + Softmax:</strong> Projekce do velikosti celého cílového slovníku a pravděpodobnosti dalších slov.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("##### 📊 Srovnání architektur: RNN vs LSTM vs Transformer")
    comp_df = pd.DataFrame([
        {
            "Architektura": "Vanilla RNN",
            "Paralelizace trénování": "❌ Žádná (sekvenční O(N))",
            "Dlouhodobý kontext": "❌ Špatný (mizející gradient)",
            "Složitost propojení slov": "O(N)",
            "Vhodnost pro LLM": "❌ Zastaralé"
        },
        {
            "Architektura": "LSTM / GRU",
            "Paralelizace trénování": "❌ Žádná (sekvenční O(N))",
            "Dlouhodobý kontext": "⚠️ Střední (paměťová buňka cell state)",
            "Složitost propojení slov": "O(N)",
            "Vhodnost pro LLM": "⚠️ Omezená"
        },
        {
            "Architektura": "Transformer",
            "Paralelizace trénování": "✅ Plná (O(1) paralelní matice)",
            "Dlouhodobý kontext": "✅ Excelentní (přímá Self-Attention)",
            "Složitost propojení slov": "O(1)",
            "Vhodnost pro LLM": "⭐ Globální standard (BERT, GPT, Claude)"
        }
    ])
    st.dataframe(comp_df, hide_index=True, width="stretch")


# ==============================================================================
# TAB 2: MECHANIKA SELF-ATTENTION (Q, K, V)
# ==============================================================================
with tab2:
    st.subheader("Matematika mechanismu Self-Attention")

    st.markdown(
        r"""
        V analogii s vyhledáváním má každé slovo 3 role reprezentované vektory:
        * **Dotaz (Query $Q$)**: Co dané slovo právě hledá v kontextu věty.
        * **Klíč (Key $K$)**: Čím se slovo prezentuje ostatním (jaká je jeho značka).
        * **Hodnota (Value $V$)**: Samotný obsah informace, kterou slovo předává dál.

        $$Q = X W^Q, \quad K = X W^K, \quad V = X W^V$$

        ### Klíčový vzorec Scaled Dot-Product Attention:
        $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
        """
    )

    st.info(
        "💡 **Proč dělíme $\\sqrt{d_k}$?** Pokud mají vektory vysokou dimenzi (např. $d_k = 64$), "
        "skalární součin $Q K^T$ nabývá velkých absolutních hodnot. Funkce softmax by pak byla extrémně strmá "
        "(sytila by se do 0 nebo 1) a gradienty při zpětném šíření by vymizely. Dělení odmocninou stabilizuje rozptyl na 1."
    )

    st.markdown("##### 🔬 Interaktivní simulátor vah pozornosti (Attention Heatmap)")
    st.caption("Prohlédněte si, jak různé hlavy pozornosti (Attention Heads) v Transformeru zachycují syntaktické vazby a zájmena:")

    if bert_data and "attention_simulation" in bert_data:
        attn_info = bert_data["attention_simulation"]
        toks = attn_info["tokens"]

        head_choice = st.radio(
            "Vyberte specializaci hlavy pozornosti (Attention Head):",
            [
                "Hlava A: Řešení zájmenné koreference ('it' -> 'animal')",
                "Hlava B: Místní syntaktičtí sousedé (bigramy a trigramy)"
            ],
            horizontal=True
        )

        matrix = attn_info["coreference_head"] if "Hlava A" in head_choice else attn_info["syntactic_head"]

        fig_attn = px.imshow(
            matrix,
            x=toks,
            y=toks,
            labels=dict(x="Klíč (Key Token)", y="Dotaz (Query Token)", color="Váha pozornosti"),
            color_continuous_scale="Blues",
            title=f"Matice vah pozornosti Softmax(QK^T / √d_k) – {head_choice.split(':')[0]}"
        )
        fig_attn.update_layout(
            template="plotly_dark",
            height=480,
            margin=dict(l=40, r=40, t=50, b=40)
        )
        st.plotly_chart(fig_attn, width="stretch")

        if "Hlava A" in head_choice:
            st.success(
                "🔎 **Analýza Winogradova schématu:** Podívejte se na řádek slova **'it'** na ose Y. "
                "Pozornost směřuje nejsilněji na slovo **'animal'** (65 %) a **'tired'** (15 %). "
                "Transformer tak automaticky vyřešil, že zájmeno 'it' odkazuje na zvíře, nikoliv na ulici!"
            )
        else:
            st.info(
                "🔎 **Syntaktická hlava:** Pozornost se soustředí na diagonálu a bezprostřední sousedy slova "
                "(levý a pravý kontext), což umožňuje zachytit větnou skladbu."
            )


# ==============================================================================
# TAB 3: POZIČNÍ KÓDOVÁNÍ (POSITIONAL ENCODING)
# ==============================================================================
with tab3:
    st.subheader("Proč a jak funguje Poziční kódování (Positional Encoding)?")

    st.markdown(
        r"""
        Protože Self-Attention vyhodnocuje vztah každého slova ke každému nezávisle na čase, 
        je maticový násobek **permutančně invariantní**. Bez doplňkové informace by věta:
        > *"Pes kousl člověka."* a *"Člověk kousl psa."*
        
        vyprodukovala pro síť naprosto stejné množiny vektorů!

        ### Sinusoidální poziční formule (Vaswani et al.):
        $$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{\frac{2i}{d_{model}}}}\right)$$
        $$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{\frac{2i}{d_{model}}}}\right)$$
        kde $pos$ je index slova ve větě ($0, 1, 2, \dots$) a $i$ je index dimenze vektoru.
        """
    )

    if bert_data and "positional_encoding_sample" in bert_data:
        pe_matrix = np.array(bert_data["positional_encoding_sample"])

        fig_pe = px.imshow(
            pe_matrix,
            labels=dict(x="Dimenze vektoru (0 až 63)", y="Pozice slova ve větě (0 až 49)", color="Hodnota PE"),
            color_continuous_scale="Viridis",
            title="Vzory sinusoidálního pozičního kódování (50 pozic x 64 dimenzí)"
        )
        fig_pe.update_layout(template="plotly_dark", height=420, margin=dict(l=30, r=30, t=50, b=30))
        st.plotly_chart(fig_pe, width="stretch")

        st.markdown(
            """
            * **Nízké dimenze (vlevo):** Vlny mají velmi vysokou frekvenci (rychle kmitají), což kóduje informace o bezprostředních sousedech.
            * **Vysoké dimenze (vpravo):** Vlny mají nízkou frekvenci (dlouhé periody), což pomáhá udržet povědomí o globální pozici v dlouhém odstavci.
            * **Lineární transformovatelnost:** Pro jakýkoliv posun $k$ lze vektor $PE_{pos+k}$ vypočítat lineární rotací $PE_{pos}$.
            """
        )

        st.markdown("##### 📈 Průběh křivek pro vybrané dimenze:")
        pos_axis = np.arange(len(pe_matrix))
        fig_lines = go.Figure()
        for d_idx in [0, 4, 16, 32]:
            fig_lines.add_trace(go.Scatter(
                x=pos_axis,
                y=pe_matrix[:, d_idx],
                mode="lines+markers",
                name=f"Dimenze {d_idx} (frekvence = 1/{10000**(d_idx/64):.1f})"
            ))
        fig_lines.update_layout(
            template="plotly_dark",
            title="Srovnání frekvencí sinusoid napříč pozicemi slov",
            xaxis_title="Pozice slova (Token Position)",
            yaxis_title="Hodnota PE",
            height=350,
            margin=dict(l=30, r=30, t=50, b=30)
        )
        st.plotly_chart(fig_lines, width="stretch")


# ==============================================================================
# TAB 4: DECODER & MASKED ATTENTION
# ==============================================================================
with tab4:
    st.subheader("Causal Masking v Decoderu: Proč nesmí model vidět do budoucnosti?")

    st.markdown(
        r"""
        V úlohách generování textu (např. strojový překlad, ChatGPT) model předpovídá slova postupně zleva doprava:
        $$P(w_t \mid w_1, w_2, \dots, w_{t-1})$$.

        Během trénování však předáváme do dekodéru **celou cílovou větu najednou**, abychom využili paralelizaci na GPU. 
        Kdybychom nepoužili maskování, slovo na pozici $t$ by se v mechanismu Self-Attention jednoduše podívalo na slovo $t+1$ 
        a trénink by zdegeneroval v pouhé opisování!
        """
    )

    col_m1, col_m2 = st.columns([1, 1])

    with col_m1:
        st.markdown("##### 🎭 Maskovací matice (Causal Mask):")
        mask_words = ["<START>", "Film", "byl", "velmi", "dobrý"]
        n_m = len(mask_words)
        mask_matrix = np.tril(np.ones((n_m, n_m)))

        fig_mask = px.imshow(
            mask_matrix,
            x=mask_words,
            y=mask_words,
            labels=dict(x="Klíč (Dostupné slovo)", y="Generované slovo", color="Povoleno"),
            color_continuous_scale=[[0, "#dc2626"], [1, "#16a34a"]],
            title="Kauzální trojúhelníková maska (1 = Povoleno, 0 = Zablokováno)"
        )
        fig_mask.update_layout(template="plotly_dark", height=380, margin=dict(l=30, r=30, t=50, b=30))
        st.plotly_chart(fig_mask, width="stretch")

    with col_m2:
        st.markdown("##### ⚙️ Jak maska funguje v matematice:")
        st.markdown(
            r"""
            1. Vypočte se matice skalárních součinů:
               $$S = \frac{Q K^T}{\sqrt{d_k}}$$
            2. Všem pozicím $j > i$ (budoucí slova) se přiřadí hodnota $-\infty$:
               $$S_{masked} = S + M, \quad M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}$$
            3. Po aplikaci funkce $\text{softmax}$:
               $$e^{-\infty} = 0$$
            4. Váha budoucího slova je **přesně rovna 0 %**! Informace z budoucnosti nemůže do výpočtu uniknout.
            """
        )
        st.success(
            "🛡️ **Výsledek:** Trénink probíhá pro celou větu paralelně za zlomek sekundy, "
            "ale model je přísně nucen generovat slova pouze z dosud známého kontextu!"
        )
