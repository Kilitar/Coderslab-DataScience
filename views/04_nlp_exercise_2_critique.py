"""
Den 4: NLP – Cvičení 2: Expertní analýza & Kritika řešení
==========================================================
Téma: Morfologická lemmatizace vs. Klasický Stemming vs. SOTA Subword BPE (10/2026)
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.title("🔬 Expertní analýza: Lemmatizace, Limity & SOTA Standardy (10/2026)")
st.caption(
    "Hloubkový rozbor lemmatizačních technik: Proč spaCy překonalo Porterův stemmer, jaká jsou skrytá úskalí lemmatizace "
    "a proč moderní LLM v roce 2026 lemmatizaci zcela opustily ve prospěch Subword BPE tokenizérů."
)

tab1, tab2, tab3 = st.tabs([
    "💡 1. Anatomie úspěchu & Proč Lemmatizace",
    "⚠️ 2. Skrytá úskalí a limity lemmatizace",
    "🚀 3. SOTA 10/2026: Subword BPE & LLM revoluce"
])

with tab1:
    st.subheader("1. Anatomie úspěchu: Proč lemmatizace překonává stemming")
    st.markdown(
        """
        Zatímco **Stemming** (např. Porterův nebo Snowball algoritmus) používá heuristická pravidla pro ořezávání koncovek 
        bez znalosti gramatiky (často vytváří neexistující patvary jako *uni* pro *universities* nebo *studi* pro *studying*), 
        **Lemmatizace** provádí skutečnou **morfologickou analýzu**.
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🏆 Hlavní výhody Lemmatizace:")
        st.markdown(
            """
            1. **Skutečná slova (Valid Vocabulary):** 
               * Výsledkem je vždy validní slovníkový tvar (lemma), který lze snadno interpretovat v byznys reportech i topic modelingu (LDA).
            2. **Zvládání nepravidelných tvarů:**
               * Stemmer nikdy nespojí *went* a *go*, nebo *better* a *good*. Lemmatizér díky slovníku ano.
            3. **Kontextuální disambiguace (PoS Tagging):**
               * Slovo *meeting* může být sloveso (*they are meeting*) s lemmatem *meet*, nebo podstatné jméno (*the meeting was cancelled*) s lemmatem *meeting*. Pokročilé lemmatizéry toto rozlišují.
            """
        )

    with col2:
        st.markdown("#### ⚖️ Přímé srovnání redukce tvarů:")
        comp_data = {
            "Surové slovo": ["wolves", "running", "better", "was", "went", "universities", "flies"],
            "Porter Stemmer": ["wolv", "run", "better", "wa", "went", "univers", "fli"],
            "spaCy Lemma": ["wolf", "run", "good", "be", "go", "university", "fly"],
            "Zachován smysl?": ["✅ Ano", "✅ Ano", "🌟 Dokonale (sjednoceno)", "🌟 Dokonale (sjednoceno)", "🌟 Dokonale (sjednoceno)", "✅ Ano", "✅ Ano"]
        }
        st.dataframe(pd.DataFrame(comp_data), hide_index=True, width="stretch")

    st.markdown("---")
    st.markdown("#### Matematický dopad na matici termínů (BoW / TF-IDF):")
    st.latex(r"""\text{Sparsity} = 1 - \frac{\text{Počet nenulových prvků v matici } X}{N \times |V|}""")
    st.markdown(
        """
        Zmenšením velikosti slovníku $|V|$ o **20 až 30 %** dochází ke:
        * Zmenšení paměťové náročnosti reprezentace dokumentů.
        * Zmírnění problému *prokletí dimenzionality* (Curse of Dimensionality) pro lineární klasifikátory (Logistická regrese, Lineární SVM, Naive Bayes).
        * Zvýšení frekvence jednotlivých termínů, což stabilizuje výpočet IDF vah.
        """
    )

with tab2:
    st.subheader("2. Skrytá úskalí a limity lemmatizace v reálné praxi")
    st.markdown(
        """
        Ačkoliv je lemmatizace podstatně čistší než stemming, v reálných produkčních systémech přináší specifická úskalí:
        """
    )

    u1, u2, u3 = st.columns(3)
    with u1:
        st.error("🐢 1. Výpočetní bottleneck")
        st.markdown(
            r"""
            * Stemmer běží v čase $\mathcal{O}(1)$ na úrovni řetězcových operací.
            * Lemmatizace vyžaduje tokenizaci, slovníkový lookup a často i morfologický rozbor (Part-of-Speech tagging).
            * Při zpracování 1 000 000 dokumentů může lemmatizace trvat **desítky minut až hodiny**, zatímco regex či subwords zaberou sekundy.
            """
        )

    with u2:
        st.warning("⚠️ 2. Ztráta sémantických nuancí")
        st.markdown(
            """
            * Převod *better* → *good* sjednotí frekvence, ale smaže fakt, že zákazník v recenzi prováděl **srovnání**.
            * Převod *news* → *new* (častá chyba naivních lemmatizérů) zcela změní význam podstatného jména na přídavné jméno.
            * Sloučení slovesných časů stírá rozdíl mezi minulostí (*I loved this brand*) a přítomností (*I love this brand*).
            """
        )

    with u3:
        st.info("❓ 3. Problém OOV (Out-of-Vocabulary)")
        st.markdown(
            """
            * Jak lemmatizovat neologismy, překlepy (*amaziiing*), odborné zkratky (*Kubernetes*, *PyTorch*), hashtags nebo emojis?
            * Pokud slovo není ve slovníku lemmatizátoru, zůstává nezměněno, což vede k nekonzistentnímu prostoru příznaků.
            """
        )

with tab3:
    st.subheader("3. SOTA Standard 10/2026: Proč moderní LLM lemmatizaci nepoužívají?")
    st.markdown(
        """
        V éře velkých jazykových modelů (LLM jako LLaMA 3.3, GPT-4o, Claude 3.5 Sonnet) a moderních transformerů 
        se **klasická lemmatizace a stemming v produkci již nepoužívají**. Proč?
        """
    )

    st.markdown(
        """
        #### 1. Subword Tokenizace (BPE & SentencePiece)
        * Místo slovníkového osekávání pracují moderní tokenizéry s proměnlivými podslovními fragmenty (Subword Units).
        * Častá slova tvoří jeden token (`apple`), vzácná nebo flexivní slova se bezeztrátově rozdělí (`un`, `believ`, `able`).
        * **Výsledek:** Slovník je pevně ohraničený (např. 128 000 tokenů u LLaMA 3), ale model **nemá žádná OOV slova** a zvládne libovolný jazyk, kód i překlepy.

        #### 2. Kontextuální vnoření (Contextual Dense Embeddings)
        * V architektuře Transformer se každé slovo mapuje do spojitého vektorového prostoru (např. dimenze 4096).
        * Pozornostní mechanismus (Self-Attention) sám o sobě rozpozná, že *wolves* a *wolf* sdílí stejný sémantický koncept, aniž by musel kdokoli text destruktivně zjednodušovat.
        * Model si navíc pamatuje, že šlo o množné číslo, což je klíčové pro gramatickou správnost generování.
        """
    )

    # Srovnávací radar/barchart
    categories = ["Rychlost zpracování", "Gramatická přesnost", "Odolnost vůči OOV", "Podpora vícejazyčnosti", "Zachování kontextu"]
    fig_comp = go.Figure()

    fig_comp.add_trace(go.Bar(
        name="Stemming (Porter)",
        x=categories,
        y=[95, 30, 40, 25, 20],
        marker_color="#F5A623"
    ))
    fig_comp.add_trace(go.Bar(
        name="Lemmatizace (spaCy)",
        x=categories,
        y=[45, 85, 50, 60, 65],
        marker_color="#4A90E2"
    ))
    fig_comp.add_trace(go.Bar(
        name="Subword BPE (SOTA LLM 2026)",
        x=categories,
        y=[90, 95, 99, 95, 100],
        marker_color="#50E3C2"
    ))

    fig_comp.update_layout(
        title="Srovnání přístupů k reprezentaci slov (Klasické NLP vs. SOTA 10/2026)",
        barmode="group",
        yaxis_title="Výkonnostní skóre (0–100)",
        template="plotly_dark",
        height=420
    )
    st.plotly_chart(fig_comp, width="stretch")

    st.markdown(
        """
        > [!TIP]
        > **Závěrečné doporučení pro datového vědce:**
        > * Pokud stavíte rychlý baseline model s **TF-IDF a Logistickou regresí** na menším korpusu: **Lemmatizace pomocí spaCy je ideální volba**.
        > * Pokud stavíte embeddingový model, RAG pipeline nebo fine-tunujete Transformer: **Nelemmatizujte! Ponechte původní surový text a svěřte jej nativnímu Subword BPE tokenizéru daného modelu.**
        """
    )
