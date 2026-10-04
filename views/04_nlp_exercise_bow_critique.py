"""
Den 4: NLP – Cvičení BoW: Expertní analýza & Kritika řešení
============================================================
Téma: Limity Bag of Words, Zipfův zákon, paměťová náročnost a SOTA standardy (10/2026)
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.title("🔬 Expertní analýza: Bag of Words, Zipfův zákon & Paměťové pasti (10/2026)")
st.caption(
    "Hloubkový rozbor klasifikace textu pomocí Bag of Words: Jak funguje parametr max_features, "
    "proč je .toarray() v produkci nebezpečné a jak moderní modely překonaly limity unigramů."
)

tab1, tab2, tab3 = st.tabs([
    "📐 1. max_features, Zipfův zákon & Truncation",
    "💾 2. Paměťová past: Dense vs. Sparse matice",
    "🚀 3. Proč 86 % nestačí: Negace, N-gramy & SOTA 2026"
])

with tab1:
    st.subheader("1. Parametr max_features a Zipfův zákon v textových datech")
    st.markdown(
        r"""
        V zadání jsme použili `CountVectorizer(max_features=10000)`. Proč zrovna 10 000 a jak to ovlivní model?
        
        V přirozeném jazyce platí **Zipfův zákon**: frekvence libovolného slova je nepřímo úměrná jeho pořadí v žebříčku četností:
        """
    )
    st.latex(r"""f(r) \propto \frac{1}{r^s} \quad (s \approx 1)""")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📊 Důsledky pro slovník (Vocabulary):")
        st.markdown(
            """
            * **Dlouhý chvost (Long Tail):** Celý IMDb korpus může obsahovat přes 60 000 unikátních slov. Avšak 70 % z nich se objeví pouze **jednou nebo dvakrát** (tzv. *hapax legomena* – překlepy, exotická jména, chyby OCR).
            * **Filtrování šumu:** Nastavením `max_features=10000` model zahodí tyto náhodné, statisticky nevýznamné výskyty a ponechá jádro jazyka.
            * **Regulace dimenzionality:** Zamezuje přetrénování (overfitting) lineárního klasifikátoru na vzácných slovech.
            """
        )

    with col2:
        # Simulace Zipfova rozdělení
        ranks = list(range(1, 101))
        freqs = [10000 / r for r in ranks]
        fig_zipf = go.Figure()
        fig_zipf.add_trace(go.Scatter(x=ranks, y=freqs, mode="lines+markers", line=dict(color="#50E3C2", width=3)))
        fig_zipf.update_layout(
            title="Zipfův zákon: Extrémní pokles frekvence slov",
            xaxis_title="Pořadí slova dle četnosti (Rank)",
            yaxis_title="Frekvence výskytu",
            template="plotly_dark",
            height=300
        )
        st.plotly_chart(fig_zipf, width="stretch")

with tab2:
    st.subheader("2. Produkční past: Proč je .toarray() kritická chyba paměti")
    st.markdown(
        r"""
        V mnoha tutoriálech a zadáních se uvádí: `df_bow = pd.DataFrame(X.toarray(), columns=...)`. 
        Zatímco na malém školním datasetu to projde, v **reálné produkci je to nejrychlejší cesta k pádu na Out-Of-Memory (OOM)**.
        """
    )

    st.error("💥 Výpočet paměti pro převod na Dense matici:")
    st.markdown(
        r"""
        * Mějme korpus o velikosti $N = 100\,000$ dokumentů a slovník $M = 50\,000$ slov.
        * Pokud vytvoříme dense matici v `float64` (8 bajtů):
        $$\text{Paměť} = 100\,000 \times 50\,000 \times 8 \text{ bajtů} = 40\,000\,000\,000 \text{ bajtů} \approx \mathbf{40\text{ GB RAM!}}$$
        * Přitom skutečná matice textu má **řídkost (sparsity) přes 99.2 %** (99.2 % prvků jsou nuly!).
        """
    )

    st.success("✅ Správné řešení: Compressed Sparse Row (CSR Matrix)")
    st.markdown(
        """
        * `scikit-learn` vrací výstup `fit_transform()` jako formát `scipy.sparse.csr_matrix`.
        * Ukládá pouze nenulové hodnoty a jejich indexy.
        * Místo **40 GB** zabere v RAM pouhých **~300 MB** (úspora více než 99 % paměti)!
        * `LogisticRegression` i lineární SVM v scikit-learn plně podporují práci přímo se sparse maticemi bez nutnosti převodu do numpy pole.
        """
    )

with tab3:
    st.subheader("3. Proč přesnost 86 % nestačí: Limity BoW a co přinesl rok 2026")
    st.markdown(
        """
        Dosáhli jsme úctyhodné testovací přesnosti **85.75 %**. Kde jsou však nepřekročitelné limity přístupu Bag of Words?
        """
    )

    w1, w2, w3 = st.columns(3)
    with w1:
        st.markdown("##### 1. Absence slovosledu")
        st.caption("„Not good, absolutely bad“ vs. „Not bad, absolutely good“ mají v unigramovém BoW naprosto totožnou vektorovou reprezentaci!")

    with w2:
        st.markdown("##### 2. Absence synonymie")
        st.caption("Slova „film“, „movie“, „picture“ a „flick“ jsou pro BoW zcela nezávislé ortogonální dimenze bez vzájemného vztahu.")

    with w3:
        st.markdown("##### 3. Pevný slovník (OOV)")
        st.caption("Jakékoli slovo mimo horních 10 000 (nebo nové neologismy a překlepy) je zcela ignorováno.")

    st.markdown("---")
    st.markdown("#### Evoluce reprezentace textu v NLP (Přehled technologií):")
    evo_data = {
        "Éra & Technologie": [
            "1. Bag of Words (1954+)",
            "2. TF-IDF (1972+)",
            "3. N-gram BoW (1990+)",
            "4. Word2Vec / GloVe (2013)",
            "5. Sentence Transformers (2019)",
            "6. Moderní LLM (10/2026)"
        ],
        "Princip reprezentace": [
            "Počty výskytů slov (Sparse)",
            "Vážené výskyty s penalizací (Sparse)",
            "Dvojice a trojice sousedních slov",
            "Statické husté vektory (Dense embeddings, dim 300)",
            "Kontextuální větné embeddingy (dim 768 / 1536)",
            "Generativní self-attention (LLaMA 3.3, GPT-4o)"
        ],
        "Zachování kontextu": ["Žádné", "Žádné", "Lokální (2-3 slova)", "Sémantická podobnost", "Vysoké (celá věta)", "Kompletní hluboký kontext"],
        "Běžná přesnost IMDb": ["~85 %", "~88 %", "~89 %", "~88 %", "~93 %", "> 96 %"]
    }
    st.dataframe(pd.DataFrame(evo_data), hide_index=True, width="stretch")
