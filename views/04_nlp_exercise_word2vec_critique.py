"""
Den 4: NLP – Cvičení Word2Vec: Expertní analýza & Kritika řešení
================================================================
Téma: Husté sémantické embeddingy vs. Sparse TF-IDF, Semantic Averaging a SOTA 10/2026
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.title("🔬 Expertní analýza: Husté embeddingy vs. Řídké matice & Sémantické průměrování")
st.caption(
    "Hloubkový rozbor výsledků Word2Vec: Jak je možné, že 100 hustých čísel dosáhne 85 % přesnosti jako 10 000 čísel v BoW? "
    "Kde selhává prosté průměrování vektorů a jak moderní transformery v roce 2026 řeší Sentence Embeddings."
)

tab1, tab2, tab3 = st.tabs([
    "📊 1. Velké srovnání: BoW vs. TF-IDF vs. Word2Vec",
    "⚠️ 2. Slabiny prostého průměrování (Mean Vector)",
    "🚀 3. SOTA 10/2026: Od Word2Vec k Sentence-Transformers"
])

with tab1:
    st.subheader("1. Velké srovnání reprezentací textu na IMDb datasetu")
    st.markdown(
        """
        Během Dne 4 jsme otestovali tři fundamentální generace reprezentace textových dat na identickém datasetu 10 000 filmových recenzí:
        """
    )

    comp_summary = pd.DataFrame([
        {
            "Generace NLP": "1. Bag of Words (1954+)",
            "Dimenze": "10 000 (Sparse)",
            "Přesnost (Accuracy)": "85.75 %",
            "F1-skóre": "0.858",
            "Paměťová náročnost": "Střední",
            "Klíčová výhoda": "Jednoduchost a interpretovatelnost vah",
            "Hlavní limit": "Absence sémantiky a synonymie"
        },
        {
            "Generace NLP": "2. TF-IDF (1972+)",
            "Dimenze": "10 000 (Sparse)",
            "Přesnost (Accuracy)": "87.30 %",
            "F1-skóre": "0.873",
            "Paměťová náročnost": "Střední",
            "Klíčová výhoda": "Penalizace obecných slov, L2 normalizace délky",
            "Hlavní limit": "Ortogonalita slov (žádná příbuznost)"
        },
        {
            "Generace NLP": "3. Word2Vec (2013+)",
            "Dimenze": "100 (Dense)",
            "Přesnost (Accuracy)": "84.65 %",
            "F1-skóre": "0.847",
            "Paměťová náročnost": "Minimální (100 čísel!)",
            "Klíčová výhoda": "Sémantická příbuznost, 100x komprese prostoru",
            "Hlavní limit": "Ztráta syntaxe při prostém průměrování"
        }
    ])
    st.dataframe(comp_summary, hide_index=True, width="stretch")

    st.markdown("---")
    st.markdown("#### Zázrak 100násobné komprese:")
    st.markdown(
        r"""
        Všimněte si fascinujícího zjištění:
        * **TF-IDF matice:** Potřebuje $10\,000$ dimenzí k dosažení přesnosti $87.3\,\%$.
        * **Word2Vec matice:** Potřebuje pouhých **$100$ dimenzí** k dosažení přesnosti **$84.7\,\%$**!
        * To znamená, že **100 hustých latentních dimenzí v sobě nese téměř tolik diskriminační informace jako 10 000 frekvenčních sloupců**. 
        Pro mobilní aplikace a mikroprocesory je to naprosto klíčová úspora paměti i propustnosti.
        """
    )

with tab2:
    st.subheader("2. Skrytá úskalí prostého průměrování vektorů (Mean Vector)")
    st.markdown(
        r"""
        V zadání jsme použili standardní postup:
        $$\mathbf{d} = \frac{1}{|d|} \sum_{w \in d} \mathbf{v}_w$$
        Ačkoliv je tento přístup rychlý a jednoduchý, v produkčním NLP naráží na 3 zásadní limity:
        """
    )

    u1, u2, u3 = st.columns(3)
    with u1:
        st.error("1. Sémantické rozmělnění")
        st.markdown(
            """
            * Pokud zprůměrujete 300 slov dlouhé recenze, vektor zkonverguje k průměrnému „šedému těžišti“ celého jazyka.
            * Dlouhé recenze se v prostoru začnou slévat dohromady a ztrácejí specifické nuance.
            """
        )

    with u2:
        st.warning("2. Stejná váha pro všechna slova")
        st.markdown(
            """
            * Běžné slovo jako *film* nebo *time* má v průměru naprosto stejnou váhu jako klíčové sémantické slovo *masterpiece* nebo *catastrophic*.
            * **Lepší řešení:** TF-IDF vážený průměr slovních vektorů!
            """
        )

    with u3:
        st.info("3. Ztráta slovosledu a logiky")
        st.markdown(
            """
            * Věty *„This was not good, it was bad“* a *„This was not bad, it was good“* 
            mají při aritmetickém průměru vektorů **naprosto identický výsledný vektor**!
            """
        )

    st.markdown("---")
    st.markdown("#### Matematické vylepšení: TF-IDF vážené vnoření dokumentu:")
    st.latex(r"""\mathbf{d}_{\text{weighted}} = \frac{\sum_{w \in d} \text{TF-IDF}(w, d) \cdot \mathbf{v}_w}{\sum_{w \in d} \text{TF-IDF}(w, d)}""")
    st.caption("Tento hybridní přístup spojuje to nejlepší z obou světů: sémantiku Word2Vec a relevanční filtraci TF-IDF.")

with tab3:
    st.subheader("3. Stav SOTA v říjnu 2026: Proč Sentence-Transformers nahradily Word2Vec")
    st.markdown(
        """
        Jak se v moderním světě řeší reprezentace celých vět a dokumentů dnes?
        """
    )

    st.markdown(
        """
        #### 1. Sentence-BERT & Bi-Encodery (Dense Retrieval & RAG)
        * Místo průměrování statických slov se využívá model trénovaný na siamských transformerových sítích (např. `all-MiniLM-L6-v2`, `bge-large-en`, `text-embedding-3`).
        * **Mean Pooling s pozorností (Attention Pooling):** Model bere v úvahu kontext každého slova ve větě.
        * Slovo *apple* má jiný embedding v kontextu *„I ate a sweet apple“* a jiný v *„Apple released new iPhone“*.

        #### 2. RAG & Vektorové databáze (Chroma, Qdrant, Pinecone)
        * V roce 2026 tvoří moderní Dense Embeddings páteř vyhledávání v korpusech pro velké jazykové modely (RAG architektury).
        * Přesnost vyhledávání v sémantickém prostoru přesahuje 95 % bez jakékoliv manuální lemmatizace či stemmingu.
        """
    )

    # Srovnání radarový graf
    fig_comp = go.Figure()
    categories = ["Paměťová efektivita", "Rychlost trénování", "Sémantická příbuznost", "Zachování slovosledu", "Odolnost vůči polysemii"]

    fig_comp.add_trace(go.Scatterpolar(
        r=[70, 95, 20, 10, 10],
        theta=categories,
        fill="toself",
        name="TF-IDF (1972)"
    ))
    fig_comp.add_trace(go.Scatterpolar(
        r=[95, 85, 75, 20, 25],
        theta=categories,
        fill="toself",
        name="Word2Vec (Mikolov 2013)"
    ))
    fig_comp.add_trace(go.Scatterpolar(
        r=[85, 40, 98, 95, 95],
        theta=categories,
        fill="toself",
        name="Sentence-Transformers (SOTA 2026)"
    ))

    fig_comp.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        template="plotly_dark",
        title="Evoluce schopností textových reprezentací",
        height=420
    )
    st.plotly_chart(fig_comp, width="stretch")
