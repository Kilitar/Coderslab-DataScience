"""
Prework Session 2: Úvod do zpracování přirozeného jazyka (Introduction to NLP)
=============================================================================
Interaktivní výukový modul pro přípravu na Session 2 (Dny 3 a 4):
1. Základní principy NLP, lingvistická úskalí a polysémie (příklad slova 'bar').
2. Interaktivní simulátor textové pipeline (Čištění, Tokenizace, Stopwords, Normalizace, Lemmatizace).
3. Hloubkový průzkumník: Stemming vs. Lemmatizace (srovnání algoritmů na reálných slovech).
4. Vektorizace textu: Bag of Words, TF-IDF kalkulátor a interaktivní sémantická mapa Word2Vec embeddings.
5. 11 praktických oblastí využití NLP v byznysu a průmyslu s interaktivními ukázkami.
6. Interaktivní ověřovací kvíz.
"""

import re
import math
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def render_nlp_intro_view():
    st.title("🗣️ Prework Session 2: Úvod do NLP (Natural Language Processing)")
    st.markdown(
        "**Příprava na Dny 3 a 4 (Session 2):** Seznámení s principy strojového zpracování lidské řeči – "
        "od lingvistických výzev přes systematický řetězec předzpracování textu (*Preprocessing Pipeline*), "
        "techniky vektorizace (TF-IDF, Word2Vec) až po 11 klíčových praktických aplikací moderní AI."
    )

    # Horní KPI karty
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Základní jednotka textu", "Token (Slovo / Podslovo)", delta="Tokenizace")
    with c2:
        st.metric("Nestrukturovaná data", "> 80 % firemních dat", delta="E-maily, recenze, smlouvy")
    with c3:
        st.metric("Klíčový krok pro ML", "Vektorizace", delta="BoW / TF-IDF / Embeddings")
    with c4:
        st.metric("Moderní architektura", "Transformers / LLM", delta="Self-Attention mechanismus")

    st.markdown("---")

    # Záložky modulu
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔄 Pipeline předzpracování textu",
        "✂️ Stemming vs. Lemmatizace",
        "📐 Vektorizace & Sémantické embeddings",
        "🏢 11 Praktických aplikací NLP",
        "📝 Vědomostní kvíz"
    ])

    # =========================================================================
    # TAB 1: INTERAKTIVNÍ NLP PIPELINE SIMULÁTOR
    # =========================================================================
    with tab1:
        st.subheader("1. Systematický řetězec předzpracování textu (NLP Pipeline)")
        st.markdown(
            "Počítače nerozumí slovům intuitivně jako lidé. Abychom z nestrukturovaného textu získali data vhodná "
            "pro algoritmy strojového učení, musíme projít standardizovaným procesem transformace."
        )

        preset = st.selectbox(
            "Vyberte ukázkový text pro simulaci pipeline nebo zadejte vlastní:",
            [
                "Zákaznická recenze (E-shop): 'Skvělý produkt! Rychlé doručení do 24 hodin, ale balení bylo mírně poškozené... 5/5 hvězd.'",
                "Hotelová stížnost (Hotelnictví): 'The hotel rooms were NOT clean at all! Bed sheets were dirty and the staff was extremely unhelpful.'",
                "Lékařská zpráva (Medicína): 'Pacient si stěžuje na bolesti hlavy a zvýšenou teplotu 38.5 °C. Předepsán Paralen 500 mg 3x denně.'"
            ]
        )

        user_text = st.text_area("Vstupní surový text:", value=preset, height=85)

        st.markdown("##### ⚙️ Nastavení kroků pipeline:")
        col_opt1, col_opt2, col_opt3, col_opt4 = st.columns(4)
        with col_opt1:
            opt_lower = st.checkbox("Normalizace (Lowercase)", value=True)
        with col_opt2:
            opt_clean = st.checkbox("Odstranit interpunkci a čísla", value=True)
        with col_opt3:
            opt_stop = st.checkbox("Odstranit stop-slova", value=True)
        with col_opt4:
            opt_stem = st.radio("Základní tvar:", ["Ponechat původní", "Jednoduchý Stemming", "Lemmatizace"], index=2)

        # Implementace kroků
        current_text = user_text

        # 1. Normalizace
        if opt_lower:
            step1_text = current_text.lower()
        else:
            step1_text = current_text

        # 2. Čištění
        if opt_clean:
            step2_text = re.sub(r"[^\w\s]", " ", step1_text)
            step2_text = re.sub(r"\d+", " ", step2_text)
            step2_text = re.sub(r"\s+", " ", step2_text).strip()
        else:
            step2_text = step1_text

        # 3. Tokenizace
        raw_tokens = step2_text.split() if step2_text else []

        # 4. Stopwords
        cz_en_stopwords = {
            "a", "i", "v", "na", "se", "si", "je", "byl", "byla", "bylo", "ale", "do", "k", "o", "po", "ze",
            "the", "and", "is", "was", "were", "at", "in", "on", "to", "for", "with", "all", "not", "but"
        }
        if opt_stop:
            filtered_tokens = [t for t in raw_tokens if t.lower() not in cz_en_stopwords]
        else:
            filtered_tokens = raw_tokens

        # 5. Kmen / Lemma
        stem_map = {
            "produkt": "produkt", "doručení": "doruč", "hodin": "hodin", "balení": "bal",
            "poškozené": "poškozen", "hvězd": "hvězd", "hotel": "hotel", "rooms": "room",
            "clean": "clean", "bed": "bed", "sheets": "sheet", "dirty": "dirt",
            "staff": "staff", "extremely": "extrem", "unhelpful": "unhelp", "pacient": "pacient",
            "stěžuje": "stěž", "bolesti": "bolest", "hlavy": "hlav", "teplotu": "teplot",
            "předepsán": "předeps", "paralen": "paralen", "denně": "den"
        }
        lemma_map = {
            "produkt": "produkt", "doručení": "doručení", "hodin": "hodina", "balení": "balení",
            "poškozené": "poškozený", "hvězd": "hvězda", "hotel": "hotel", "rooms": "room",
            "clean": "clean", "bed": "bed", "sheets": "sheet", "dirty": "dirty",
            "staff": "staff", "extremely": "extremely", "unhelpful": "unhelpful", "pacient": "pacient",
            "stěžuje": "stěžovat", "bolesti": "bolest", "hlavy": "hlava", "teplotu": "teplota",
            "předepsán": "předepsat", "paralen": "paralen", "denně": "denně"
        }

        if opt_stem == "Jednoduchý Stemming":
            final_tokens = [stem_map.get(t.lower(), t[:5] if len(t) > 5 else t) for t in filtered_tokens]
        elif opt_stem == "Lemmatizace":
            final_tokens = [lemma_map.get(t.lower(), t) for t in filtered_tokens]
        else:
            final_tokens = filtered_tokens

        st.markdown("---")
        st.subheader("📊 Výsledky jednotlivých fází pipeline:")

        pipe_cols = st.columns(4)
        with pipe_cols[0]:
            st.metric("1. Hrubý počet znaků", f"{len(user_text)} znaků")
        with pipe_cols[1]:
            st.metric("2. Počet tokenů (slov)", f"{len(raw_tokens)}")
        with pipe_cols[2]:
            st.metric("3. Počet po stop-slovech", f"{len(filtered_tokens)}", delta=f"{len(filtered_tokens) - len(raw_tokens)} slov")
        with pipe_cols[3]:
            st.metric("4. Finální unifikované tokeny", f"{len(final_tokens)}")

        col_tokens, col_chart = st.columns([1, 1])
        with col_tokens:
            st.markdown("**Finální seznam tokenů připravený pro model:**")
            if final_tokens:
                token_df = pd.DataFrame({
                    "Index": range(1, len(final_tokens) + 1),
                    "Výsledný token": final_tokens,
                    "Původní slovo": filtered_tokens[:len(final_tokens)],
                    "Délka (znaky)": [len(t) for t in final_tokens]
                })
                st.dataframe(token_df, width="stretch", hide_index=True)
            else:
                st.info("Žádné tokeny nezbyly po aplikaci filtrů.")

        with col_chart:
            if final_tokens:
                st.markdown("**Rozdělení délek výsledných tokenů:**")
                fig_lens = px.histogram(
                    x=[len(t) for t in final_tokens],
                    nbins=10,
                    labels={"x": "Délka tokenu (znaky)", "count": "Četnost"},
                    color_discrete_sequence=["#3b82f6"]
                )
                fig_lens.update_layout(
                    margin=dict(l=20, r=20, t=30, b=20),
                    height=280,
                    showlegend=False,
                    xaxis_title="Počet znaků v tokenu",
                    yaxis_title="Četnost"
                )
                st.plotly_chart(fig_lens, width="stretch")

        with st.expander("🔍 Zobrazit lingvistickou výzvu: Polysémie a kontext (příklad slova 'Bar')"):
            st.markdown(r"""
            V kurzu je zmíněna klíčová otázka: **Proč počítači nestačí pouhý slovník všech existujících slov a vět?**
            
            1. **Nekonečná generativní schopnost:** Lidský jazyk umožňuje tvořit nekonečné množství nových vět, metafor a neologismů.
            2. **Polysémie (víceznačnost):** Jedno slovo nabývá zcela odlišných významů:
               - *„Šli jsme po práci do baru na drink.“* $\to$ Gastronomický podnik.
               - *„Zlatá spona na kravatě (collar bar).“* $\to$ Módní doplněk.
               - *„Mříže na vězeňském okně (window bars).“* $\to$ Fyzická kovová překážka.
               - *„Čokoládová tyčinka (chocolate bar).“* $\to$ Cukrovinka.
               - *„Tlakoměr ukázal 2.5 bar.“* $\to$ Fyzikální jednotka tlaku.
            
            Proto moderní NLP nespoléhá na fixní slovníky, ale učí se **statistické a kontextové vztahy z obrovských korpusů textu**.
            """)

    # =========================================================================
    # TAB 2: STEMMING VS. LEMMATIZACE
    # =========================================================================
    with tab2:
        st.subheader("2. Převod slov na základní tvar: Stemming vs. Lemmatizace")
        st.markdown(
            "Cílem obou technik je snížit dimenzionalitu slovníku a sjednotit různé tvary téhož slova "
            "(např. *běžel, běžela, běželi, běhají* $\to$ základní koncept). Způsob, jakým toho dosahují, se však zásadně liší."
        )

        col_st1, col_st2 = st.columns([1, 1])
        with col_st1:
            st.info(
                """
                #### ✂️ Stemming (Kmenování)
                - **Mechanismus:** Heuristický algoritmus založený na sadě pravidel (např. *Porter Stemmer*, *Snowball*), 
                  který mechanicky odřezává běžné přípony a koncovky.
                - **Výsledek:** Často vyprodukuje **neexistující slovo** (kmen), které není ve slovníku (tzv. *stem*).
                - **Rychlost:** Extrémně rychlý (jen řetězcové operace).
                - **Slovní druhy:** Ignoruje kontext i slovní druh (nerozlišuje sloveso od podstatného jména).
                """
            )

        with col_st2:
            st.success(
                """
                #### 📖 Lemmatizace (Slovníkový tvar)
                - **Mechanismus:** Lingvistická analýza využívající rozsáhlý slovník (např. *WordNet*, *spaCy*, *MorfFlex* pro češtinu) 
                  a určení slovního druhu (**Part-of-Speech – POS tagger**).
                - **Výsledek:** Vždy platné **slovníkové heslo (lemma)**.
                - **Rychlost:** Pomalejší, vyžaduje gramatickou a morfologickou analýzu.
                - **Kontext:** Dokáže rozlišit sloveso *„meeting“* (lemma *„meet“*) od podstatného jména *„meeting“* (lemma *„meeting“*).
                """
            )

        st.markdown("---")
        st.subheader("🧪 Interaktivní porovnání na problematických slovech")

        test_words_data = [
            {"Původní slovo": "studies", "Jazyk": "EN", "Porter Stemmer": "studi", "Lemmatizace": "study", "Komentář": "Stemmer zkrátil 'ies' na 'i' (neplatné slovo), Lemma našla infinitiv."},
            {"Původní slovo": "studying", "Jazyk": "EN", "Porter Stemmer": "studi", "Lemmatizace": "study", "Komentář": "Stemmer i Lemmatizér sjednotily tvar."},
            {"Původní slovo": "better", "Jazyk": "EN", "Porter Stemmer": "better", "Lemmatizace": "good", "Komentář": "Stemmer selhává u nepravidelného stupňování. Lemmatizér ví, že základ je 'good'!"},
            {"Původní slovo": "went", "Jazyk": "EN", "Porter Stemmer": "went", "Lemmatizace": "go", "Komentář": "Nepravidelné minulé sloveso: Stemmer ponechává 'went', Lemma vrací 'go'."},
            {"Původní slovo": "corpora", "Jazyk": "EN", "Porter Stemmer": "corpora", "Lemmatizace": "corpus", "Komentář": "Množné číslo latinského původu: Lemma správně detekuje 'corpus'."},
            {"Původní slovo": "wolves", "Jazyk": "EN", "Porter Stemmer": "wolv", "Lemmatizace": "wolf", "Komentář": "Změna koncovky f -> ves: Stemmer tvoří neexistující 'wolv'."},
            {"Původní slovo": "letěla", "Jazyk": "CZ", "Porter Stemmer": "let", "Lemmatizace": "letět", "Komentář": "V češtině stemmer odsekne koncovku, lemma vrátí infinitiv slovesa."}
        ]
        df_comparison = pd.DataFrame(test_words_data)
        st.dataframe(df_comparison, width="stretch", hide_index=True)

        # Plotly porovnání vlastností
        st.markdown("##### ⚖️ Srovnání klíčových vlastností v praxi:")
        comp_metrics = pd.DataFrame({
            "Kritérium": ["Rychlost zpracování", "Gramatická správnost", "Zachování sémantiky", "Výpočetní nenáročnost"],
            "Stemming": [95, 45, 60, 95],
            "Lemmatizace": [55, 95, 90, 50]
        })
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            x=comp_metrics["Kritérium"],
            y=comp_metrics["Stemming"],
            name="Stemming (Kmenování)",
            marker_color="#f59e0b"
        ))
        fig_comp.add_trace(go.Bar(
            x=comp_metrics["Kritérium"],
            y=comp_metrics["Lemmatizace"],
            name="Lemmatizace (Základní tvar)",
            marker_color="#10b981"
        ))
        fig_comp.update_layout(
            barmode="group",
            height=300,
            margin=dict(l=20, r=20, t=30, b=20),
            yaxis_title="Skóre vlastnosti (0–100)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_comp, width="stretch")

    # =========================================================================
    # TAB 3: VEKTORIZACE & SÉMANTICKÉ EMBEDDINGS
    # =========================================================================
    with tab3:
        st.subheader("3. Vektorizace textu: Převod slov na čísla")
        st.markdown(
            "Algoritmy strojového učení nedokáží pracovat s písmeny. Musíme text převést na čísla. "
            "Vývoj prošel od jednoduchého počítání výskytů (Bag of Words) přes vážený TF-IDF až po moderní hustá sémantická vnoření (**Word Embeddings**)."
        )

        sub_tab_bow, sub_tab_tfidf, sub_tab_w2v = st.tabs([
            "📦 Bag of Words (BoW)",
            "📊 TF-IDF Kalkulátor",
            "🌌 Word2Vec Sémantické Embeddings"
        ])

        with sub_tab_bow:
            st.markdown("#### Bag of Words (Pytel slov)")
            st.write(
                "Nejjednodušší přístup: Vytvoříme slovník všech unikátních slov v korpusu. "
                "Každý dokument reprezentujeme vektorem o velikosti celého slovníku, kde hodnota udává, kolikrát se dané slovo v dokumentu vyskytlo."
            )
            st.warning("⚠️ **Nevýhoda BoW:** Ignoruje slovosled (např. *'kočka loví myš'* má stejný vektor jako *'myš loví kočku'*) a vytváří obrovské řídké matice plné nul.")

        with sub_tab_tfidf:
            st.markdown("#### TF-IDF (Term Frequency – Inverse Document Frequency)")
            st.write(
                "TF-IDF řeší zásadní problém: Slova jako *'auto'* v automobilových recenzích mají vysokou frekvenci, "
                "ale nenesou rozlišovací hodnotu. TF-IDF proto penalizuje slova, která se vyskytují téměř ve všech dokumentech."
            )
            st.latex(r"\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \ln\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right)")

            # Interaktivní ukázka na miniaturním korpusu 3 dokumentů
            doc_corpus = [
                "data science je super obor",
                "machine learning a data science",
                "data science pomáhá byznysu růst"
            ]
            st.caption("Příklad miniaturního korpusu o 3 dokumentech:")
            for i, d in enumerate(doc_corpus, 1):
                st.code(f"Dokument D{i}: {d}")

            words_sample = ["data", "science", "obor", "byznysu", "learning"]
            tfidf_demo_data = [
                {"Slovo": "data", "TF v D1": "1 / 5 = 0.20", "Výskyt v dokumentech": "3 ze 3", "IDF váha": "Nízká (běžné slovo)", "TF-IDF skóre": "0.20 (utlumeno)"},
                {"Slovo": "science", "TF v D1": "1 / 5 = 0.20", "Výskyt v dokumentech": "3 ze 3", "IDF váha": "Nízká (běžné slovo)", "TF-IDF skóre": "0.20 (utlumeno)"},
                {"Slovo": "obor", "TF v D1": "1 / 5 = 0.20", "Výskyt v dokumentech": "1 ze 3", "IDF váha": "Vysoká (unikátní)", "TF-IDF skóre": "0.78 (zvýrazněno!)"},
                {"Slovo": "learning", "TF v D1": "0 / 5 = 0.00", "Výskyt v dokumentech": "1 ze 3", "IDF váha": "Vysoká", "TF-IDF skóre": "0.00 (není v D1)"}
            ]
            st.dataframe(pd.DataFrame(tfidf_demo_data), width="stretch", hide_index=True)

        with sub_tab_w2v:
            st.markdown("#### Word2Vec a Vektorový prostor sémantiky")
            st.write(
                "V roce 2013 Tomáš Mikolov představil model **Word2Vec**. Místo obřích řídkých vektorů přiřazuje každému slovu "
                "hustý vektor (např. 100 až 300 čísel) naučený z kontextových oken. "
                "Slova s podobným významem končí blízko u sebe v mnohorozměrném prostoru!"
            )
            st.success(
                r"""
                ✨ **Slavná vektorová aritmetika sémantiky:**
                $$\vec{v}(\text{Král}) - \vec{v}(\text{Muž}) + \vec{v}(\text{Žena}) \approx \vec{v}(\text{Královna})$$
                """
            )

            # Interaktivní 2D sémantická mapa (PCA projekce)
            embeddings_points = pd.DataFrame([
                {"slovo": "král", "x": 8.2, "y": 7.5, "kategorie": "Panovníci / Hierarchie"},
                {"slovo": "královna", "x": 8.0, "y": 9.2, "kategorie": "Panovníci / Hierarchie"},
                {"slovo": "princ", "x": 6.8, "y": 7.0, "kategorie": "Panovníci / Hierarchie"},
                {"slovo": "princezna", "x": 6.7, "y": 8.8, "kategorie": "Panovníci / Hierarchie"},
                {"slovo": "muž", "x": 3.5, "y": 3.0, "kategorie": "Lidé / Rodina"},
                {"slovo": "žena", "x": 3.3, "y": 4.8, "kategorie": "Lidé / Rodina"},
                {"slovo": "chlapec", "x": 2.2, "y": 2.8, "kategorie": "Lidé / Rodina"},
                {"slovo": "dívka", "x": 2.1, "y": 4.6, "kategorie": "Lidé / Rodina"},
                {"slovo": "pes", "x": -5.5, "y": -4.0, "kategorie": "Zvířata"},
                {"slovo": "štěně", "x": -6.5, "y": -4.8, "kategorie": "Zvířata"},
                {"slovo": "kočka", "x": -4.8, "y": -2.5, "kategorie": "Zvířata"},
                {"slovo": "kotě", "x": -5.9, "y": -3.2, "kategorie": "Zvířata"},
                {"slovo": "počítač", "x": -6.0, "y": 8.0, "kategorie": "Technologie"},
                {"slovo": "software", "x": -7.2, "y": 7.5, "kategorie": "Technologie"},
                {"slovo": "algoritmus", "x": -5.5, "y": 9.2, "kategorie": "Technologie"},
                {"slovo": "python", "x": -6.8, "y": 9.0, "kategorie": "Technologie"}
            ])

            fig_embed = px.scatter(
                embeddings_points,
                x="x",
                y="y",
                color="kategorie",
                text="slovo",
                title="Interaktivní projekce sémantického prostoru Word2Vec (2D vnoření)",
                color_discrete_sequence=["#ef4444", "#3b82f6", "#10b981", "#8b5cf6"]
            )
            fig_embed.update_traces(
                textposition="top center",
                marker=dict(size=14, line=dict(width=1, color="black"))
            )
            fig_embed.update_layout(
                height=450,
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis_title="Sémantická dimenze 1",
                yaxis_title="Sémantická dimenze 2",
                legend_title="Sémantická kategorie"
            )
            st.plotly_chart(fig_embed, width="stretch")
            st.caption("Všimněte si, že vektor posunu od 'muž' k 'žena' (směrem nahoru na ose Y) je téměř identický s posunem od 'král' ke 'královna'!")

    # =========================================================================
    # TAB 4: 11 PRAKTICKÝCH APLIKACÍ NLP V PRAXI
    # =========================================================================
    with tab4:
        st.subheader("4. 11 praktických oblastí využití NLP v byznysu a průmyslu")
        st.markdown(
            "Technologie NLP transformovaly fungování moderních firem. Vyberte kteroukoli z 11 oblastí "
            "prozkoumejte architekturu a vyzkoušejte interaktivní mini-simulátor:"
        )

        app_selection = st.selectbox(
            "Zvolte oblast NLP aplikace k prozkoumání:",
            [
                "1. Chatboti a virtuální asistenti",
                "2. Analýza sentimentu (Sentiment Analysis)",
                "3. Strojový překlad (Machine Translation)",
                "4. Analýza zákaznické zpětné vazby",
                "5. Detekce spamu a phishingu",
                "6. Vyhledávání informací a sémantický search (RAG)",
                "7. Analýza sociálních sítí a trendy",
                "8. Rozpoznávání řeči (Speech-to-Text / ASR)",
                "9. Generování souhrnů textu (Summarization)",
                "10. Rozpoznávání pojmenovaných entit (NER)",
                "11. NLP ve vzdělávání a e-learningu"
            ]
        )

        st.markdown("---")

        if "1. Chatboti" in app_selection:
            st.markdown("### 🤖 1. Chatboti a virtuální asistenti")
            st.write(
                "**Účel:** Automatická obsluha zákazníků v reálném čase, odpovídání na dotazy k produktům, skladové dostupnosti a reklamacím.\n"
                "**Reálný příklad z praxe:** Asistent Max na webu telekomunikačního operátora Orange, bankovní chatboti (např. George v ČS).\n"
                "**Architektura:** NLU (Intent Recognition + Entity Slot Filling) $\\to$ Dialog Manager $\\to$ RAG nad firemní znalostní bází $\\to$ Generování odpovědi."
            )
            sim_q = st.text_input("Zeptejte se virtuálního asistenta:", value="Dobrý den, jaká je záruční doba na zakoupený notebook?")
            if st.button("Simulovat odpověď asistenta", width="stretch"):
                st.success("🤖 **Asistent Max:** Standardní záruční doba na veškerou spotřební elektroniku činí 24 měsíců. V případě nákupu na IČO je záruka 12 měsíců. Přejete si pomoci s reklamací?")

        elif "2. Analýza sentimentu" in app_selection:
            st.markdown("### ❤️ 2. Analýza sentimentu (Sentiment Analysis)")
            st.write(
                "**Účel:** Automatické hodnocení tónu a emocí v zákaznických recenzích (pozitivní, neutrální, negativní).\n"
                "**Reálný příklad:** Monitorování recenzí na Trustpilot, Google Maps, Heurece či Amazonu pro okamžitou reakci na nespokojené zákazníky.\n"
                "**Architektura:** Preprocessing $\\to$ TF-IDF / RoBERTa fine-tuned na sentiment $\\to$ Klasifikační Softmax hlava."
            )
            user_rev = st.text_area("Zadejte text recenze:", "Jídlo bylo vynikající a obsluha velice milá, ale na stůl jsme museli čekat přes 40 minut.")
            # Jednoduchý slovníkový odhad pro simulaci
            pos_words = ["vynikající", "milá", "skvělý", "super", "perfektní", "chutné"]
            neg_words = ["čekat", "hrozné", "studené", "drahé", "špatné", "poškozené"]
            p_cnt = sum(1 for w in pos_words if w in user_rev.lower())
            n_cnt = sum(1 for w in neg_words if w in user_rev.lower())
            score = 50 + (p_cnt - n_cnt) * 25
            score = max(5, min(95, score))

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                title={"text": "Predikovaný index spokojenosti (0 % = Negativní, 100 % = Pozitivní)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#3b82f6"},
                    "steps": [
                        {"range": [0, 40], "color": "#fee2e2"},
                        {"range": [40, 60], "color": "#fef3c7"},
                        {"range": [60, 100], "color": "#d1fae5"}
                    ]
                }
            ))
            fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_gauge, width="stretch")

        elif "3. Strojový překlad" in app_selection:
            st.markdown("### 🌐 3. Strojový překlad (Machine Translation)")
            st.write(
                "**Účel:** Automatický převod textu z jednoho jazyka do druhého se zachováním gramatiky a idiomů.\n"
                "**Reálný příklad:** Google Translate, DeepL, automatický překlad vícejazyčných příspěvků na síti X (Twitter).\n"
                "**Architektura:** Encoder-Decoder Transformery (např. T5, MarianMT, NLLB-200) s křížovou pozorností (*Cross-Attention*)."
            )
            col_tr1, col_tr2 = st.columns([1, 1])
            with col_tr1:
                st.info("**Zdrojový text (Čeština):**\n\n'Neuronové sítě představují klíč k moderní umělé inteligenci.'")
            with col_tr2:
                st.success("**Strojový překlad (Angličtina):**\n\n'Neural networks represent the key to modern artificial intelligence.'")

        elif "4. Analýza zákaznické" in app_selection:
            st.markdown("### 🍽️ 4. Analýza zákaznické zpětné vazby")
            st.write(
                "**Účel:** Agregace tisíců komentářů z recenzních portálů a extrakce konkrétních témat (Topic Modeling).\n"
                "**Reálný příklad:** Řetězce restaurací (např. McDonald's) analyzují recenze: Která pobočka má studené hranolky a kde vázne personál?\n"
                "**Architektura:** BERTopic, LDA (Latent Dirichlet Allocation) nebo LLM extrakce strukturovaných JSON položek."
            )
            st.markdown("**Ukázka detekovaných témat v 1 500 recenzích:**")
            topic_df = pd.DataFrame({
                "Téma stížnosti / pochvaly": ["Rychlost obsluhy", "Čistota toalet a stolů", "Chuť a teplota pokrmů", "Ceny a porce", "Chování personálu"],
                "Zastoupení (%)": [35, 24, 18, 13, 10],
                "Převažující sentiment": ["Spíše negativní", "Negativní", "Vysoce pozitivní", "Neutrální", "Pozitivní"]
            })
            st.dataframe(topic_df, width="stretch", hide_index=True)

        elif "5. Detekce spamu" in app_selection:
            st.markdown("### 🛡️ 5. Detekce spamu a phishingu")
            st.write(
                "**Účel:** Klasifikace příchozích zpráv v poštovních schránkách (Gmail, Outlook) a filtrace podvodných či nevyžádaných sdělení.\n"
                "**Architektura:** Naive Bayes, Support Vector Machines (SVM) nebo BERT nad předzpracovaným textem a metadaty odesílatele."
            )
            sample_mail = st.selectbox(
                "Otestujte zprávu:",
                [
                    "URGENT: Your account has been suspended! Click here to claim your 1,000,000 USD prize immediately!",
                    "Ahoj Petře, posílám ti zápis z dnešní schůzky ohledně rozpočtu na příští kvartál. Dej mi vědět, zda vše sedí."
                ]
            )
            is_spam = "prize" in sample_mail.lower() or "urgent" in sample_mail.lower()
            prob = 98.4 if is_spam else 1.2
            st.metric("Pravděpodobnost SPAMu", f"{prob:.1f} %", delta="POZOR: Detekován Phishing" if is_spam else "Bezpečná zpráva")

        elif "6. Vyhledávání informací" in app_selection:
            st.markdown("### 🔍 6. Vyhledávání informací & Sémantický search (RAG)")
            st.write(
                "**Účel:** Vyhledávání na základě významu a záměru uživatele, nikoliv pouhé shody klíčových slov.\n"
                r"**Princip:** Dotaz uživatele se převede na vektor vnoření a pomocí kosinové podobnosti ($Cosine\ Similarity$) se vyhledají nejbližší dokumenty."
            )
            st.code("Uživatel hledá: 'jak se zbavit bolesti hlavy bez léků'\nVýsledek sémantického vyhledávače: Článek s názvem 'Dostatečná hydratace a odpočinek při migréně' (Aniž by článek obsahoval slovo 'léky'!)")

        elif "7. Analýza sociálních" in app_selection:
            st.markdown("### 📱 7. Analýza sociálních sítí a trendy")
            st.write(
                "**Účel:** Průběžné sledování zmínek o značce, analýza reputace v reálném čase a včasné varování před PR krizí.\n"
                "**Metody:** Sledování hashtagů, frekvenční analýza n-gramů, grafové modely šíření informací."
            )

        elif "8. Rozpoznávání řeči" in app_selection:
            st.markdown("### 🎙️ 8. Rozpoznávání řeči (Speech-to-Text / ASR)")
            st.write(
                "**Účel:** Převod akustického audiosignálu na text pro hlasové ovládání zařízení nebo přepis nahrávek.\n"
                "**Příklady:** Apple Siri, OpenAI Whisper, Google Speech API, přepis soudních líčení a hovorů call center."
            )

        elif "9. Generování souhrnů" in app_selection:
            st.markdown("### 📑 9. Automatická sumarizace textu (Summarization)")
            st.write(
                "**Dva základní přístupy:**\n"
                "- **Extraktivní sumarizace:** Algoritmus vybere a sestaví nejdůležitější původní věty z textu (např. TextRank).\n"
                "- **Abstraktivní sumarizace:** Moderní LLM (GPT, Gemini) text pochopí a napíše zbrusu nový výstižný souhrn vlastními slovy."
            )

        elif "10. Rozpoznávání pojmenovaných" in app_selection:
            st.markdown("### 🏷️ 10. Rozpoznávání pojmenovaných entit (Named Entity Recognition – NER)")
            st.write(
                "**Účel:** Automatická detekce a typování specifických objektů v textu: Osobnost (`PER`), Organizace (`ORG`), Místo (`LOC`), Datum (`DATE`), Peněžní částka (`MONEY`).\n"
                "**Využití:** Automatické vytěžování faktur, smluv a anonymizace citlivých osobních údajů (GDPR)."
            )
            st.markdown(
                """
                **Ukázka označkované věty:**  
                *„[Elon Musk]<sub>PER</sub> navštívil včera [Berlín]<sub>LOC</sub>, aby jednal s představiteli [Tesla Motors]<sub>ORG</sub> o investici [2.5 miliardy EUR]<sub>MONEY</sub>.“*
                """
            )

        elif "11. NLP ve vzdělávání" in app_selection:
            st.markdown("### 🎓 11. NLP ve vzdělávání a e-learningu")
            st.write(
                "**Účel:** Automatická kontrola gramatiky (Grammarly), adaptivní výuka cizích jazyků (Duolingo), "
                "poloautomatické opravování testů a personalizované studijní plány podle chybovosti studenta."
            )

    # =========================================================================
    # TAB 5: VĚDOMOSTNÍ KVÍZ
    # =========================================================================
    with tab5:
        st.subheader("📝 Vědomostní kvíz k ověření pochopení NLP")

        q1 = st.radio(
            "1. Co v oblasti NLP vyjadřuje pojem 'Tokenizace'?",
            [
                "Šifrování textu pomocí privátního klíče.",
                "Rozdělení souvislého textu na menší jednotky – tokeny (nejčastěji slova či podslova).",
                "Odstranění diakritiky z českých slov.",
                "Převod textu na malá písmena."
            ]
        )

        q2 = st.radio(
            "2. Jaký je hlavní rozdíl mezi Stemmingem a Lemmatizací?",
            [
                "Stemming je lingvisticky přesnější než Lemmatizace.",
                "Lemmatizace mechanicky uřezává koncovky, zatímco Stemming používá slovník.",
                "Stemming mechanicky uřezává koncovky podle pravidel (často vytvoří neexistující slovo), zatímco Lemmatizace využívá slovník a vrací platné heslo (lemma).",
                "Mezi Stemmingem a Lemmatizací není v praxi žádný rozdíl."
            ]
        )

        q3 = st.radio(
            "3. Proč je v NLP metoda TF-IDF lepší než pouhé počítání četností slov (Bag of Words)?",
            [
                "Protože TF-IDF zmenšuje velikost textu o 90 %.",
                "Protože TF-IDF penalizuje slova, která se běžně vyskytují ve všech dokumentech, a vyzdvihuje slova specifická pro dané téma.",
                "Protože TF-IDF funguje pouze na neuronových sítích.",
                "Protože TF-IDF automaticky překládá text do angličtiny."
            ]
        )

        q4 = st.radio(
            "4. Proč nelze počítači předat pouze slovník všech vět a slov v jazyce?",
            [
                "Protože lidský jazyk umožňuje tvořit nekonečné kombinace vět a slova mají kontextové významy (polysémie, např. slovo 'bar').",
                "Protože počítač neumí číst textové soubory větší než 1 MB.",
                "Protože gramatika všech jazyků je zcela náhodná bez pravidel.",
                "Protože počítače umí pracovat výhradně se zvukovým záznamem."
            ]
        )

        if st.button("Vyhodnotit kvíz NLP", width="stretch"):
            score = 0
            if "Rozdělení souvislého textu na menší jednotky" in q1:
                score += 1
            if "Stemming mechanicky uřezává koncovky" in q2:
                score += 1
            if "penalizuje slova, která se běžně vyskytují" in q3:
                score += 1
            if "nekonečné kombinace vět a slova mají kontextové významy" in q4:
                score += 1

            if score == 4:
                st.balloons()
                st.success("🎉 Skvěle! 4 ze 4 správně! Máte dokonalý přehled o základech NLP a jste připraveni na Session 2.")
            else:
                st.warning(f"Získali jste {score} ze 4 bodů. Projděte si záložky s pipeline a sémantickou vektorizací.")


render_nlp_intro_view()
