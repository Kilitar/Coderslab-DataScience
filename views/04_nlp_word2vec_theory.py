"""
Den 4: NLP – Teorie: Word2Vec & Slovní vnoření (Word Embeddings)
=================================================================
Syntéza materiálů z Word2Vec_-_introduction.pdf
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.title("📖 Den 4: Word2Vec – Slovní vnoření & Sémantický prostor")
st.caption(
    "Od řídkých frekvenčních matic k hustým sémantickým vektorům: Principy Word2Vec (Tomáš Mikolov, Google 2013), "
    "architektury CBOW a Skip-gram, vektorová aritmetika a vizualizace sémantického prostoru."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("1. Dimenze embeddingu", "100–300", delta="Místo 50 000+ v BoW", delta_color="normal")
c2.metric("2. Typ matice", "Dense (Hustá)", delta="Žádné zbytečné nuly", delta_color="normal")
c3.metric("3. Sémantická blízkost", "Kosinová podobnost", delta=r"cos(\theta) \in [-1, 1]", delta_color="off")
c4.metric("4. Vektorová algebra", "Analogické relace", delta="King - Man + Woman = Queen", delta_color="off")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "🧬 1. Co jsou Word Embeddings?",
    "🧠 2. Architektury: CBOW vs. Skip-gram",
    "🌌 3. Interaktivní Embedding Explorer (CZ / EN)",
    "⚖️ 4. GloVe, FastText & SOTA 2026"
])

with tab1:
    st.subheader("1. Proč přejít od Bag of Words k Word Embeddings?")
    st.markdown(
        """
        V modelech Bag of Words a TF-IDF je každé slovo reprezentováno jako **nezávislá ortogonální dimenze** (One-Hot kódování).
        To znamená, že pro počítač jsou slova *„vynikající“* a *„skvělý“* stejně odlišná jako *„vynikající“* a *„traktor“*.
        Jejich skalární součin je vždy nulový.
        
        **Word Embeddings (Slovní vnoření)** tento problém překonávají tím, že mapují slova do **spojitého vektorového prostoru**, 
        kde vzdálenost a úhel mezi vektory přímo vyjadřují **sémantickou podobnost**.
        """
    )

    col_comp1, col_comp2 = st.columns(2)
    with col_comp1:
        st.markdown("##### ❌ Tradiční přístup (One-Hot / BoW):")
        st.markdown(
            r"""
            * **Dimenze:** Rovna velikosti slovníku $|V|$ (např. 50 000+ čísel).
            * **Řídkost:** 99.9 % prvků jsou nuly.
            * **Podobnost:** $\mathbf{v}_{\text{pes}} \cdot \mathbf{v}_{\text{štěně}} = 0$ (nulová sémantická vazba).
            """
        )

    with col_comp2:
        st.markdown("##### 🌟 Word Embeddings (Word2Vec):")
        st.markdown(
            r"""
            * **Dimenze:** Nízká a pevná (např. 100 až 300 čísel).
            * **Hustota:** Každý prvek je reálné číslo nesoucí latentní význam.
            * **Podobnost:** $\cos(\mathbf{v}_{\text{pes}}, \mathbf{v}_{\text{štěně}}) \approx 0.88$ (vysoká afinita).
            """
        )

    st.markdown("---")
    st.markdown("#### Geometrický příklad ze slidů: Věk vs. Rod (Gender vs. Age)")
    st.markdown("Jednoduchý 2D řez ilustrující, jak embeddingový prostor kóduje sémantické koncepty:")

    demo_words_2d = pd.DataFrame([
        {"Slovo": "boy (chlapec)", "Pohlaví (Gender)": 1.0, "Věk (Age)": 2.0, "Kategorie": "Mužský rod"},
        {"Slovo": "man (muž)", "Pohlaví (Gender)": 1.0, "Věk (Age)": 7.0, "Kategorie": "Mužský rod"},
        {"Slovo": "grandfather (dědeček)", "Pohlaví (Gender)": 1.0, "Věk (Age)": 10.0, "Kategorie": "Mužský rod"},
        {"Slovo": "girl (dívka)", "Pohlaví (Gender)": 9.0, "Věk (Age)": 2.0, "Kategorie": "Ženský rod"},
        {"Slovo": "woman (žena)", "Pohlaví (Gender)": 9.0, "Věk (Age)": 7.0, "Kategorie": "Ženský rod"},
        {"Slovo": "infant (nemluvně)", "Pohlaví (Gender)": 5.0, "Věk (Age)": 0.5, "Kategorie": "Neutrální"},
        {"Slovo": "child (dítě)", "Pohlaví (Gender)": 5.0, "Věk (Age)": 3.0, "Kategorie": "Neutrální"},
        {"Slovo": "adult (dospělý)", "Pohlaví (Gender)": 5.0, "Věk (Age)": 7.0, "Kategorie": "Neutrální"}
    ])

    fig_demo2d = px.scatter(
        demo_words_2d,
        x="Pohlaví (Gender)",
        y="Věk (Age)",
        text="Slovo",
        color="Kategorie",
        title="2D projekce sémantického prostoru: Pohlaví vs. Věk",
        template="plotly_dark"
    )
    fig_demo2d.update_traces(textposition="top right", marker=dict(size=12))
    fig_demo2d.update_layout(height=380)
    st.plotly_chart(fig_demo2d, width="stretch")

    st.markdown("---")
    st.markdown("#### Slavná vektorová aritmetika: Relace Král - Královna")
    st.markdown(
        r"""
        Jeden z nejpozoruhodnějších objevů Word2Vec je, že vztahy mezi koncepty odpovídají **vektorovým posunům**:
        """
    )
    st.latex(r"""\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}} + \mathbf{v}_{\text{woman}} \approx \mathbf{v}_{\text{queen}}""")
    st.markdown(
        r"""
        * Odečtením vektoru $\mathbf{v}_{\text{man}}$ od $\mathbf{v}_{\text{king}}$ odstraníme maskulinní rys a zůstane abstraktní koncept „královské hodnosti“.
        * Následným přičtením $\mathbf{v}_{\text{woman}}$ získáme bod v prostoru, jehož nejbližším sousedem je slovo **„queen“**!
        """
    )

with tab2:
    st.subheader("2. Architektury Word2Vec: CBOW vs. Skip-gram (Tomáš Mikolov, Google 2013)")
    st.markdown(
        """
        Word2Vec byl vyvinut v roce 2013 týmem z Google pod vedením českého vědce **Tomáše Mikolova**. 
        Model využívá princip **distribuční sémantiky**: slova vyskytující se v podobných kontextech mají podobný význam.
        
        Word2Vec definuje dvě komplementární architektury neuronových sítí:
        """
    )

    col_arch1, col_arch2 = st.columns(2)
    with col_arch1:
        st.info("A. CBOW (Continuous Bag of Words)")
        st.markdown(
            r"""
            * **Úkol sítě:** Předpovědět **centrální slovo (Target)** na základě jeho **okolního kontextu (Context)**.
            * **Příklad:** Ve větě *„Machine learning loves only numbers“* při kontextovém okně 1 síť z kontextu `[learning, only]` odhaduje cílové slovo `loves`.
            * **Vstup:** One-hot vektory kontextových slov $\rightarrow$ průměrování ve skryté vrstvě $\rightarrow$ Softmax pravděpodobnost přes celý slovník.
            * **Výhody:** Rychlejší trénování, vyšší přesnost pro často se opakující slova.
            """
        )

    with col_arch2:
        st.success("B. Skip-gram")
        st.markdown(
            r"""
            * **Úkol sítě:** Přesný opak CBOW – na základě **jednoho centrálního slova** předpovídá jeho **okolní kontext**.
            * **Příklad:** Z centrálního slova `loves` síť předpovídá pravděpodobnost výskytu slov `learning` a `only` v jeho okolí.
            * **Schéma:** Síť otočená o 180 stupňů oproti CBOW.
            * **Výhody:** Vynikající na menších datasetech a **výrazně lepší pro vzácná a méně častá slova**.
            """
        )

    st.markdown("---")
    st.markdown("#### Srovnávací matice: Kdy zvolit CBOW a kdy Skip-gram?")
    comp_cbow_sg = pd.DataFrame([
        {
            "Kritérium": "Směr predikce",
            "CBOW": "Kontext -> Centrální slovo",
            "Skip-gram": "Centrální slovo -> Kontext"
        },
        {
            "Kritérium": "Rychlost trénování",
            "CBOW": "⚡ Velmi rychlé (průměruje kontext)",
            "Skip-gram": "🐢 Pomalejší (trénuje na každé dvojici)"
        },
        {
            "Kritérium": "Chování u vzácných slov",
            "CBOW": "Slabší (častá slova přehluší vzácná)",
            "Skip-gram": "🌟 Vynikající (reprezentuje i zřídka se vyskytující slova)"
        },
        {
            "Kritérium": "Vhodnost pro velikost dat",
            "CBOW": "Ideální pro obrovské korpusy (miliardy slov)",
            "Skip-gram": "Ideální pro menší a střední korpusy"
        }
    ])
    st.dataframe(comp_cbow_sg, hide_index=True, width="stretch")

with tab3:
    st.subheader("3. Interaktivní Embedding Explorer (CZ / EN)")
    st.markdown(
        "Vyzkoušejte si chování sémantického prostoru. Zvolte jazykový režim a tematický shluk slov, "
        "nebo spočítejte kosinovou podobnost a vektorovou analogii."
    )

    lang_tab3 = st.radio("Jazykový režim exploreru:", ["🇨🇿 Čeština", "🇬🇧 Angličtina"], horizontal=True)

    # Definice syntetického, ale geometricky věrného embeddingového prostoru
    if "Čeština" in lang_tab3:
        embeddings_db = {
            # Královská rodina
            "král": np.array([0.92, 0.85, 0.12, 0.88]),
            "královna": np.array([0.89, 0.84, 0.88, 0.90]),
            "princ": np.array([0.72, 0.40, 0.15, 0.82]),
            "princezna": np.array([0.70, 0.38, 0.86, 0.84]),
            "muž": np.array([0.15, 0.70, 0.10, 0.10]),
            "žena": np.array([0.14, 0.68, 0.90, 0.12]),
            "chlapec": np.array([0.10, 0.25, 0.12, 0.08]),
            "dívka": np.array([0.11, 0.24, 0.89, 0.09]),
            # Geografie
            "praha": np.array([-0.80, 0.10, 0.50, 0.92]),
            "česko": np.array([-0.85, 0.08, 0.48, 0.50]),
            "paříž": np.array([-0.75, 0.15, 0.52, 0.95]),
            "francie": np.array([-0.82, 0.12, 0.50, 0.52]),
            "berlín": np.array([-0.78, 0.11, 0.51, 0.93]),
            "německo": np.array([-0.84, 0.09, 0.49, 0.51]),
            # Zvířata
            "pes": np.array([0.30, -0.75, 0.40, 0.20]),
            "štěně": np.array([0.28, -0.85, 0.42, 0.15]),
            "vlk": np.array([0.35, -0.70, 0.38, 0.25]),
            "kočka": np.array([0.25, -0.78, 0.60, 0.18]),
            "kotě": np.array([0.22, -0.88, 0.62, 0.12])
        }
    else:
        embeddings_db = {
            # Royalty
            "king": np.array([0.92, 0.85, 0.12, 0.88]),
            "queen": np.array([0.89, 0.84, 0.88, 0.90]),
            "prince": np.array([0.72, 0.40, 0.15, 0.82]),
            "princess": np.array([0.70, 0.38, 0.86, 0.84]),
            "man": np.array([0.15, 0.70, 0.10, 0.10]),
            "woman": np.array([0.14, 0.68, 0.90, 0.12]),
            "boy": np.array([0.10, 0.25, 0.12, 0.08]),
            "girl": np.array([0.11, 0.24, 0.89, 0.09]),
            # Geography
            "paris": np.array([-0.75, 0.15, 0.52, 0.95]),
            "france": np.array([-0.82, 0.12, 0.50, 0.52]),
            "london": np.array([-0.76, 0.14, 0.51, 0.94]),
            "england": np.array([-0.83, 0.11, 0.49, 0.51]),
            "rome": np.array([-0.74, 0.16, 0.53, 0.93]),
            "italy": np.array([-0.81, 0.13, 0.51, 0.50]),
            # Animals
            "dog": np.array([0.30, -0.75, 0.40, 0.20]),
            "puppy": np.array([0.28, -0.85, 0.42, 0.15]),
            "wolf": np.array([0.35, -0.70, 0.38, 0.25]),
            "cat": np.array([0.25, -0.78, 0.60, 0.18]),
            "kitten": np.array([0.22, -0.88, 0.62, 0.12])
        }

    def cos_sim(u, v):
        return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

    col_proj, col_sim = st.columns([3, 2])

    with col_proj:
        st.markdown("##### 🌌 3D Sémantický prostor embeddingů:")
        pts = []
        for word, vec in embeddings_db.items():
            pts.append({
                "Slovo": word,
                "Dim 1 (Královský status)": vec[0],
                "Dim 2 (Věk & Pozice)": vec[1],
                "Dim 3 (Rod & Ženskost)": vec[2]
            })
        df_pts = pd.DataFrame(pts)

        fig_3d = px.scatter_3d(
            df_pts,
            x="Dim 1 (Královský status)",
            y="Dim 2 (Věk & Pozice)",
            z="Dim 3 (Rod & Ženskost)",
            text="Slovo",
            color="Dim 1 (Královský status)",
            color_continuous_scale="Viridis",
            template="plotly_dark"
        )
        fig_3d.update_traces(marker=dict(size=6), textposition="top center")
        fig_3d.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_3d, width="stretch")

    with col_sim:
        st.markdown("##### 📐 Výpočet kosinové podobnosti:")
        word_list = list(embeddings_db.keys())
        w1_choice = st.selectbox("První slovo:", word_list, index=0)
        w2_choice = st.selectbox("Druhé slovo:", word_list, index=1)

        sim_val = cos_sim(embeddings_db[w1_choice], embeddings_db[w2_choice])
        st.metric(f"Podobnost: {w1_choice} vs {w2_choice}", f"{sim_val:.4f}")
        st.progress(float(max(0.0, min(1.0, sim_val))), text=f"Kosinus úhlu: {sim_val:.2f}")

        st.markdown("---")
        st.markdown("##### 🧮 Vektorová analogie:")
        if "Čeština" in lang_tab3:
            st.markdown(r"$$\mathbf{v}_{\text{král}} - \mathbf{v}_{\text{muž}} + \mathbf{v}_{\text{žena}} = \mathbf{v}_{\text{target}}$$")
            v_target = embeddings_db["král"] - embeddings_db["muž"] + embeddings_db["žena"]
        else:
            st.markdown(r"$$\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}} + \mathbf{v}_{\text{woman}} = \mathbf{v}_{\text{target}}$$")
            v_target = embeddings_db["king"] - embeddings_db["man"] + embeddings_db["woman"]

        sims = {w: cos_sim(v_target, v) for w, v in embeddings_db.items()}
        sorted_sims = sorted(sims.items(), key=lambda x: x[1], reverse=True)
        df_target_sims = pd.DataFrame(sorted_sims[:5], columns=["Slovo", "Podobnost"])
        st.dataframe(df_target_sims, hide_index=True, width="stretch")

with tab4:
    st.subheader("4. Evoluce embeddingů: GloVe, FastText a Moderní LLM (10/2026)")
    st.markdown(
        """
        Word2Vec způsobil v roce 2013 revoluci, ale měl několik zásadních slabin, 
        které vedly k vývoji dalších generací jazykových modelů:
        """
    )

    lim1, lim2, lim3 = st.columns(3)
    with lim1:
        st.error("1. Polysemie (Víceznačnost)")
        st.markdown(
            """
            * Ve Word2Vec má slovo *„koruna“* (strom, měna, královská, zub) **pouze jeden jediný statický vektor**.
            * Výsledný vektor je jen průměrem všech těchto kontextů, což vede k nepřesnostem.
            """
        )

    with lim2:
        st.warning("2. Problém OOV (Out-of-Vocabulary)")
        st.markdown(
            """
            * Pokud slovo nebylo v trénovacím korpusu (např. překlep *„amaziiing“* nebo odborný neologismus), Word2Vec ho **neumí zakódovat**.
            """
        )

    with lim3:
        st.info("3. Kontextová slepota")
        st.markdown(
            """
            * Vektor slova je po dotrénování **statický** a nemění se v závislosti na větě, ve které právě vystupuje.
            """
        )

    st.markdown("---")
    st.markdown("#### Následníci Word2Vec v moderním NLP:")
    evo_table = pd.DataFrame([
        {
            "Technologie": "GloVe (Stanford 2014)",
            "Klíčová inovace": "Global Vectors – maticová faktorizace celého korpusu ko-výskytů slov (spojuje lokální okno s globální statistikou).",
            "Řeší OOV?": "❌ Ne",
            "Kontextuální?": "❌ Statický"
        },
        {
            "Technologie": "FastText (FAIR / Mikolov 2016)",
            "Klíčová inovace": "Znakové n-gramy (subwords). Každé slovo je součtem svých částí (např. <ap, app, ple, le>).",
            "Řeší OOV?": "✅ Ano (poskládá z n-gramů)",
            "Kontextuální?": "❌ Statický"
        },
        {
            "Technologie": "BERT & RoBERTa (2018–2022)",
            "Klíčová inovace": "Obousměrný Transformer (Self-Attention). Vektor slova je dynamicky počítán podle celé okolní věty.",
            "Řeší OOV?": "✅ Ano (WordPiece)",
            "Kontextuální?": "✅ Plně dynamický"
        },
        {
            "Technologie": "Moderní LLM (10/2026)",
            "Klíčová inovace": "LLaMA 3.3, GPT-4o, Claude 3.5: Generativní autoregresní modely s obrovským latentním prostorem (dim 4096+).",
            "Řeší OOV?": "✅ Ano (BPE tiktoken)",
            "Kontextuální?": "🌟 Absolutní sémantická hloubka"
        }
    ])
    st.dataframe(evo_table, hide_index=True, width="stretch")
