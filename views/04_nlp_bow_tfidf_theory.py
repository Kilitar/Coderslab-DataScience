"""
Den 4: NLP – Teorie: Reprezentace textu – Bag of Words (BoW) & TF-IDF
======================================================================
Syntéza materiálů:
- IDF_-_introduction.pdf
- IDF_-_implementation.pdf
"""

import math
import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.title("📖 Den 4: Reprezentace textu – Bag of Words (BoW) & TF-IDF")
st.caption(
    "Od textových řetězců k numerickým vektorům: Matematické principy Bag of Words, výpočet vah TF-IDF, "
    "analýza řídkosti (sparsity) a klasifikace sentimentu pomocí Logistické regrese."
)

# Rychlé shrnutí klíčových pojmů
c1, c2, c3, c4 = st.columns(4)
c1.metric("1. Bag of Words", "Frekvence slov", delta="Ignoruje pořadí", delta_color="off")
c2.metric("2. TF (Term Frequency)", "Lokální váha", delta="Výskyty / Délka textu", delta_color="off")
c3.metric("3. IDF (Inverse Doc Freq)", "Globální penalizace", delta=r"log(N / df)", delta_color="off")
c4.metric("4. Vektorová matice", "Sparse Matrix", delta="Sparsity často > 95 %", delta_color="off")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "🎒 1. Bag of Words (BoW)",
    "⚖️ 2. TF-IDF & Matematika",
    "🧪 3. Interaktivní Vektorizační Laboratoř",
    "🎬 4. Case Study: Sentiment & Logistická regrese"
])

with tab1:
    st.subheader("1. Bag of Words (BoW) – Model „Pytle slov“")
    st.markdown(
        """
        Počítače neumí přímo interpretovat text. Abychom mohli na text aplikovat klasifikační algoritmy, 
        musíme převést každý dokument na **vektor čísel**. 
        
        Nejjednodušší přístup je **Bag of Words**: dokument vnímáme jako neuspořádaný pytel slov, 
        kde ignorujeme gramatiku a slovosled, a zaznamenáváme pouze **četnost výskytů**.
        """
    )

    st.markdown("#### 3 základní kroky implementace BoW:")
    k1, k2, k3 = st.columns(3)
    with k1:
        st.info("1. Tokenizace")
        st.markdown("Rozdělení textu každého dokumentu na jednotlivé tokeny a normalizace (např. lowercase).")
    with k2:
        st.success("2. Tvorba slovníku (Vocabulary)")
        st.markdown(r"Sestavení množiny všech unikátních slov ze všech dokumentů o celkové velikosti $|V|$.")
    with k3:
        st.warning("3. Vektorizace (Počítání)")
        st.markdown(r"Každý dokument je reprezentován vektorem délky $|V|$, kde $v_i$ je počet výskytů $i$-tého slova.")

    st.markdown("---")
    st.markdown("#### Ukázka z přednášky (3 vzorové dokumenty):")
    sample_docs = [
        "A cat is an animal.",
        "A dog is also an animal.",
        "Cats and dogs are popular animals."
    ]

    col_docs, col_vec = st.columns([1, 1])
    with col_docs:
        for idx, doc in enumerate(sample_docs, 1):
            st.markdown(f"**Dokument {idx}:** `{doc}`")

    # BoW výpočet pro ukázku
    vec_bow_demo = CountVectorizer(lowercase=True)
    X_demo = vec_bow_demo.fit_transform(sample_docs)
    vocab_demo = vec_bow_demo.get_feature_names_out()
    df_demo_bow = pd.DataFrame(X_demo.toarray(), columns=vocab_demo, index=[f"Dokument {i}" for i in range(1, 4)])

    with col_vec:
        st.markdown(f"**Velikost slovníku:** `{len(vocab_demo)} unikátních slov`")
        st.caption(f"Slovník: {list(vocab_demo)}")

    st.markdown("##### Výsledná matice četností (Document-Term Matrix):")
    st.dataframe(df_demo_bow, width="stretch")

    st.markdown("---")
    st.markdown("#### Výhody a nevýhody BoW:")
    col_pro, col_con = st.columns(2)
    with col_pro:
        st.markdown("##### 🟢 Výhody:")
        st.markdown(
            """
            * **Jednoduchá implementace a interpretace:** Každý koeficient lineárního modelu má jasný význam.
            * **Univerzálnost:** Lze použít pro libovolný jazyk bez znalosti gramatických vazeb.
            * Skvělý baseline pro SPAM filtry a základní klasifikaci témat.
            """
        )
    with col_con:
        st.markdown("##### 🔴 Nevýhody & Úskalí:")
        st.markdown(
            """
            * **Ztráta kontextu:** Ignoruje pořadí slov (*„není dobrý“* vs. *„dobrý není“*).
            * **Vysoká dimenzionalita (High Dimensionality):** Při korpusu 1 000 recenzí může mít slovník 15 000+ unikátních slov.
            * **Řídkost matice (Data Sparsity):** Většina buněk jsou nuly, což zatěžuje paměť i výpočetní čas.
            """
        )

with tab2:
    st.subheader("2. TF-IDF (Term Frequency – Inverse Document Frequency)")
    st.markdown(
        """
        V prostém Bag of Words mají nejvyšší hodnoty slova, která se nejčastěji opakují. 
        Tato slova jsou však často obecná a nenesou specifickou informaci pro rozlišení obsahu.
        
        **TF-IDF** řeší tento problém tím, že kombinuje:
        1. **Lokální frekvenci (TF):** Jak často je slovo v tomto konkrétním textu.
        2. **Globální vzácnost (IDF):** Penalizuje slova, která se vyskytují téměř ve všech dokumentech.
        """
    )

    st.markdown("#### Matematický rozpad vzorců:")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown("##### 1. Term Frequency (TF)")
        st.latex(r"""\text{TF}(t, d) = \frac{\text{počet výskytů } t \text{ v } d}{\text{celkový počet slov v } d}""")
        st.caption("Čím častěji se slovo v textu objeví, tím je lokálně důležitější.")

    with m2:
        st.markdown("##### 2. Inverse Doc Frequency (IDF)")
        st.latex(r"""\text{IDF}(t, D) = \log_{10}\left(\frac{|D|}{df_t}\right)""")
        st.caption("Pokud je slovo ve všech dokumentech, log(1) = 0 a váha klesne na nulu!")

    with m3:
        st.markdown("##### 3. Celkové TF-IDF")
        st.latex(r"""\text{TF-IDF} = \text{TF} \times \text{IDF}""")
        st.caption("Vysokou váhu získají slova, která jsou častá v daném dokumentu, ale vzácná v ostatních.")

    st.markdown("---")
    st.markdown("#### Krok za krokem: Výpočet pro slova „cat“ a „animal“")
    st.markdown(
        """
        Mějme 2 dokumenty z přednášky (po odstranění stop-slov):
        * **Dokument 1:** *„cat is animal“* (3 slova)
        * **Dokument 2:** *„dog also is animal“* (4 slova)
        """
    )

    c_calc1, c_calc2 = st.columns(2)
    with c_calc1:
        st.info("🐱 Slovo: „cat“")
        st.markdown(
            r"""
            * **Výskyt v Dok 1:** 1x ze 3 slov $\rightarrow \text{TF} = \frac{1}{3} \approx 0.333$
            * **Výskyt v korpusu:** V 1 dokumentu ze 2 $\rightarrow df = 1$
            * **IDF:** $\log_{10}\left(\frac{2}{1}\right) = \log_{10}(2) \approx 0.301$
            * **Výsledné TF-IDF:**
            $$\text{TF-IDF} = 0.333 \times 0.301 = \mathbf{0.100}$$
            *(Slovo „cat“ nese rozlišovací schopnost!)*
            """
        )

    with c_calc2:
        st.warning("🐾 Slovo: „animal“")
        st.markdown(
            r"""
            * **Výskyt v Dok 1:** 1x ze 3 slov $\rightarrow \text{TF} = \frac{1}{3} \approx 0.333$
            * **Výskyt v korpusu:** Ve 2 dokumentech ze 2 $\rightarrow df = 2$
            * **IDF:** $\log_{10}\left(\frac{2}{2}\right) = \log_{10}(1) = \mathbf{0.000}$
            * **Výsledné TF-IDF:**
            $$\text{TF-IDF} = 0.333 \times 0.000 = \mathbf{0.000}$$
            *(Protože je slovo všude, má nulovou rozlišovací hodnotu!)*
            """
        )

    st.markdown("---")
    st.markdown("#### 💡 Důležitý rozdíl: Vzorec ve slidech vs. Scikit-Learn `TfidfVectorizer`")
    st.markdown(
        r"""
        Všimněte si, že v praxi knihovna **Scikit-learn** používá mírně odlišný a robustnější vzorec:
        1. **Přirozený logaritmus $\ln$** místo dekadického $\log_{10}$.
        2. **Smooth IDF (ochrana před nulou):** 
           $$\text{IDF}_{\text{sklearn}} = \ln\left(\frac{1 + N}{1 + df}\right) + 1$$
        3. **L2 Normalizace řádků (vektorů):** Každý řádek je znormalizován tak, aby $\|\mathbf{v}\|_2 = 1$. Tím se eliminuje zkreslení délkou dokumentu!
        """
    )

with tab3:
    st.subheader("🧪 Interaktivní Vektorizační Laboratoř")
    st.markdown("Zadejte vlastní texty nebo vyberte připravené ukázky a sledujte, jak se vygeneruje Document-Term matice.")

    lang_tab3 = st.radio("Jazykový režim:", ["🇬🇧 Angličtina (Prezentace & Filmy)", "🇨🇿 Čeština (Morfologie & Diakritika)"], horizontal=True)

    default_corpus_en = (
        "The movie was great and full of action.\n"
        "The movie was terrible and completely boring.\n"
        "Great acting and wonderful action scenes in this movie."
    )
    default_corpus_cs = (
        "Tento film byl naprosto skvělý a plný akce.\n"
        "Tento film byl hrozný a neuvěřitelně nudný.\n"
        "Skvělí herci a nádherné akční scény v tomto filmu."
    )

    init_text = default_corpus_cs if "Čeština" in lang_tab3 else default_corpus_en
    user_corpus_raw = st.text_area("Zadejte dokumenty (jeden dokument na jeden řádek):", value=init_text, height=120)

    docs_list = [line.strip() for line in user_corpus_raw.split("\n") if line.strip()]

    if len(docs_list) >= 2:
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            method_choice = st.selectbox("Metoda vektorizace:", ["TF-IDF (TfidfVectorizer)", "Bag of Words (CountVectorizer)"])
        with col_p2:
            ngram_choice = st.selectbox("N-gram rozsah:", ["Unigramy (1, 1)", "Unigramy + Bigramy (1, 2)", "Pouze Bigramy (2, 2)"])
        with col_p3:
            stop_choice = st.checkbox("Odstranit stop-slova", value=True)

        ngram_val = (1, 1) if "1, 1" in ngram_choice else ((1, 2) if "1, 2" in ngram_choice else (2, 2))
        
        # Stopwords
        if stop_choice:
            if "Čeština" in lang_tab3:
                try:
                    from czech_nlp import CZECH_STOPWORDS
                    custom_stops = list(CZECH_STOPWORDS)
                except Exception:
                    custom_stops = None
            else:
                custom_stops = "english"
        else:
            custom_stops = None

        try:
            if "TF-IDF" in method_choice:
                vectorizer = TfidfVectorizer(
                    ngram_range=ngram_val,
                    stop_words=custom_stops,
                    token_pattern=r"(?u)\b\w+\b"
                )
            else:
                vectorizer = CountVectorizer(
                    ngram_range=ngram_val,
                    stop_words=custom_stops,
                    token_pattern=r"(?u)\b\w+\b"
                )

            X_mat = vectorizer.fit_transform(docs_list)
            features = vectorizer.get_feature_names_out()
            df_res = pd.DataFrame(
                X_mat.toarray(),
                columns=features,
                index=[f"Dokument {i+1}" for i in range(len(docs_list))]
            )

            # Zobrazení statistik
            s1, s2, s3 = st.columns(3)
            s1.metric("Počet dokumentů", len(docs_list))
            s2.metric("Počet unikátních příznaků (sloupců)", len(features))
            sparsity = 1.0 - (np.count_nonzero(X_mat.toarray()) / float(X_mat.toarray().size))
            s3.metric("Řídkost matice (Sparsity)", f"{sparsity * 100:.1f} %")

            st.markdown("##### 📊 Výsledná Document-Term Matice:")
            st.dataframe(df_res.round(3), width="stretch")

            # Interaktivní Heatmapa
            fig_heat = px.imshow(
                df_res,
                labels=dict(x="Příznaky (Slova / N-gramy)", y="Dokumenty", color="Hodnota"),
                x=features,
                y=df_res.index,
                color_continuous_scale="Viridis",
                title=f"Tepelná mapa vah: {method_choice}",
                template="plotly_dark"
            )
            fig_heat.update_layout(height=350)
            st.plotly_chart(fig_heat, width="stretch")

        except Exception as e:
            st.error(f"Chyba při vektorizaci: {e}")
    else:
        st.warning("Pro výpočet zadejte alespoň 2 dokumenty (každý na nový řádek).")

with tab4:
    st.subheader("🎬 Case Study ze slidů: Klasifikace sentimentu pomocí BoW a Logistické regrese")
    st.markdown(
        """
        V prezentaci `IDF - implementation.pdf` je předveden klasifikační model:
        1. Vstup: Text filmové recenze (`review_text`).
        2. Vektorizace: Pomocí `CountVectorizer()`.
        3. Model: `LogisticRegression()`.
        4. Target: Pozitivní (1) vs. Negativní (0).
        """
    )

    st.markdown("#### 📈 Vliv ladění hyperparametru regularizace ($C$):")
    res_data = {
        "Konfigurace modelu": ["1. Baseline model (výchozí parametry)", "2. Vyladěný model (Regularizace C = 10)"],
        "Správně klasifikováno": ["7 z 11 recenzí", "9 z 11 recenzí"],
        "Testovací přesnost (Accuracy)": ["63.6 %", "81.8 %"],
        "Zlepšení": ["Základ", "+18.2 %"]
    }
    st.dataframe(pd.DataFrame(res_data), hide_index=True, width="stretch")

    st.markdown("---")
    st.markdown("#### 🔬 Hloubková analýza chyb (Error Analysis) – Kde BoW selhává?")
    st.markdown(
        """
        Prezentace demonstruje typický limit unigramového Bag of Words na dvou konkrétních větách z testovací sady:
        """
    )

    col_pos, col_neg = st.columns(2)
    with col_pos:
        st.success("✅ Správně rozpoznaná pozitivní recenze:")
        st.markdown(
            """
            > *„The film's emotional impact stayed with me long after the credits rolled.“*
            
            * **Predikce modelu:** Pozitivní (1)
            * **Skutečnost:** Pozitivní (1)
            * **Proč fungovalo:** Slova jako *emotional impact* mají v korpusu silnou pozitivní korelaci.
            """
        )

    with col_neg:
        st.error("❌ Chybně klasifikovaná negativní recenze (False Positive):")
        st.markdown(
            """
            > *„The plot twists were predictable and didn't offer any surprises.“*
            
            * **Predikce modelu:** Pozitivní (1)
            * **Skutečnost:** Negativní (0)
            * **Proč model selhal?**
              1. Slovo **„surprises“** má v trénovacích datech pozitivní váhu.
              2. Slovo **„didn't“** bylo buď vyřazeno jako stop-slovo, nebo jako unigram neovlivnilo slovo *surprises*.
              3. V modelu BoW se informace o vazbě *didn't offer surprises* **ztratila**!
            """
        )

    st.markdown(
        """
        > [!TIP]
        > **Jak tuto chybu v praxi odstranit?**
        > 1. **Použít Bigramy:** `CountVectorizer(ngram_range=(1, 2))` vytvoří token `didn't offer` a `offer surprises`, které zachovají kontext.
        > 2. **Chránit negace:** Přepsat negace přímo do slovníku (např. spojením `not_good`).
        > 3. **Přejít na Subword BPE / Transformer:** Pozornostní mechanismus (Self-Attention) propojí negaci s příslovcem bez ohledu na vzdálenost ve větě.
        """
    )
