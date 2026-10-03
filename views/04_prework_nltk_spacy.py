"""
Prework Session 2: Knihovny NLTK a spaCy (Introduction to NLTK and spaCy)
========================================================================
Interaktivní výukový modul pro přípravu na Session 2 (Dny 3 a 4):
1. Praktické vyzkoušení stěžejních metod knihovny NLTK (Tokenizace, POS Tagging, Stopwords, Stemming, Lemmatizace, VADER Sentiment).
2. Objektová pipeline knihovny spaCy (en_core_web_sm, token.pos_, token.lemma_, token.is_stop, doc.ents NER, vektory).
3. Přímé srovnání NLTK vs. spaCy (architektura, filozofie, rychlost a benchmarky).
4. Head-to-Head srovnávač na vlastním textu (jak obě knihovny řeší kontrakce, měny a interpunkci).
5. Interaktivní vědomostní kvíz.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import os
import warnings
from pathlib import Path

# Bezpečná a tichá inicializace NLTK zdrojů
@st.cache_resource
def init_nltk():
    try:
        import nltk
        # Použijeme explicitně privátní domovský adresář ~/nltk_data pro eliminaci UserWarning na Streamlit Cloud
        nltk_data_dir = Path.home() / "nltk_data"
        nltk_data_dir.mkdir(parents=True, exist_ok=True)
        data_dir_str = str(nltk_data_dir)
        if data_dir_str not in nltk.data.path:
            nltk.data.path.insert(0, data_dir_str)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for res_name in [
                "punkt",
                "punkt_tab",
                "averaged_perceptron_tagger",
                "averaged_perceptron_tagger_eng",
                "stopwords",
                "wordnet",
                "vader_lexicon"
            ]:
                nltk.download(res_name, download_dir=data_dir_str, quiet=True)
        return True
    except Exception:
        return False

NLTK_AVAILABLE = init_nltk()

try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk import pos_tag
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.sentiment import SentimentIntensityAnalyzer
except Exception:
    NLTK_AVAILABLE = False

# Import spaCy
try:
    import spacy
    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False


@st.cache_resource
def get_spacy_model():
    if not SPACY_AVAILABLE:
        return None
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        try:
            # Pokus o stažení za běhu
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], capture_output=True)
            return spacy.load("en_core_web_sm")
        except Exception:
            return None


def render_nltk_spacy_view():
    st.title("🛠️ Prework Session 2: Knihovny NLTK a spaCy")
    st.markdown(
        "**Příprava na Dny 3 a 4 (Session 2):** Prozkoumání dvou pilířů zpracování textu v Pythonu – "
        "akademické stavebnice **NLTK** a vysokorychlostní průmyslové pipeline **spaCy**."
    )

    # Horní KPI karty
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("NLTK (Vznik 2001)", "50+ korpusů & algoritmů", delta="Akademický standard")
    with c2:
        st.metric("spaCy (Vznik 2014)", "Optimalizovaný Cython", delta="Až 20× rychlejší")
    with c3:
        st.metric("Architektura NLTK", "String in / List out", delta="Sada funkcí")
    with c4:
        st.metric("Architektura spaCy", "Doc → Span → Token", delta="Objektová pipeline")

    st.markdown("---")

    # Záložky modulu
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📚 NLTK v akci (Laboratoř)",
        "⚡ spaCy v akci (Objektová pipeline)",
        "⚖️ NLTK vs. spaCy: Srovnání & Benchmarky",
        "🔬 Head-to-Head analyzátor",
        "📝 Vědomostní kvíz"
    ])

    # =========================================================================
    # TAB 1: NLTK V AKCI
    # =========================================================================
    with tab1:
        st.subheader("1. Knihovna NLTK: Funkcionální nástroje pro lingvistiku")
        st.markdown(
            "NLTK (*Natural Language Toolkit*) přistupuje k textu jako k sérii transformací. "
            "Každá operace je reprezentována samostatným modulem."
        )

        preset_nltk = st.selectbox(
            "Vyberte ukázkový text pro NLTK nebo zadejte vlastní:",
            [
                "NLTK is a powerful library for natural language processing. It provides various tools for text analysis.",
                "The movie was absolutely fantastic! Brilliant acting, breathtaking visuals, but the ending felt a bit rushed.",
                "Apple reported record quarterly revenue of $94.8 billion, beating Wall Street expectations."
            ]
        )
        nltk_input = st.text_area("Vstupní text pro NLTK:", value=preset_nltk, height=75)

        if st.button("Spustit NLTK analýzu", width="stretch", key="btn_run_nltk"):
            if not NLTK_AVAILABLE:
                st.error("Knihovna NLTK není v prostředí dostupná.")
            else:
                st.markdown("#### 🔍 Výsledky zpracování v NLTK:")

                # 1. Tokenizace vět a slov
                sentences = sent_tokenize(nltk_input)
                words = word_tokenize(nltk_input)

                t_col1, t_col2 = st.columns(2)
                with t_col1:
                    st.write(f"**Větná tokenizace (`sent_tokenize`):** {len(sentences)} vět")
                    for idx, s in enumerate(sentences, 1):
                        st.info(f"**Věta {idx}:** {s}")

                with t_col2:
                    st.write(f"**Slovní tokenizace (`word_tokenize`):** {len(words)} tokenů")
                    st.code(str(words))

                st.markdown("---")

                # 2. Stopwords a POS Tagging
                stop_words = set(stopwords.words("english"))
                try:
                    tagged = pos_tag(words)
                except Exception:
                    try:
                        nltk.download("averaged_perceptron_tagger_eng", quiet=True)
                        nltk.download("averaged_perceptron_tagger", quiet=True)
                        tagged = pos_tag(words)
                    except Exception:
                        tagged = [(w, "NNP" if w and w[0].isupper() else ("VB" if w.endswith("ing") or w.endswith("ed") else "NN")) for w in words]

                pos_col, filter_col = st.columns([1, 1])
                with pos_col:
                    st.write("**Part-of-Speech Tagging (`pos_tag`):**")
                    pos_df = pd.DataFrame(tagged, columns=["Token", "Penn Treebank Tag"])
                    
                    # Přehledný český popis nejběžnějších tagů
                    tag_desc = {
                        "NN": "Podstatné jméno (jednotné)", "NNS": "Podstatné jméno (množné)",
                        "NNP": "Vlastní jméno (jednotné)", "NNPS": "Vlastní jméno (množné)",
                        "JJ": "Přídavné jméno", "JJR": "Přídavné jméno (2. stupeň)", "JJS": "Přídavné jméno (3. stupeň)",
                        "VB": "Sloveso (základní)", "VBD": "Sloveso (minulý čas)", "VBG": "Sloveso (gerundium/-ing)",
                        "VBZ": "Sloveso (3. os. přítomný)", "RB": "Příslovce", "DT": "Člen (Determinátor)",
                        "IN": "Předložka / Spojka", "PRP": "Zájmeno", "CD": "Číslovka"
                    }
                    pos_df["Význam tagu"] = pos_df["Penn Treebank Tag"].map(lambda t: tag_desc.get(t, "Ostatní syntaktický prvek"))
                    st.dataframe(pos_df.head(10), width="stretch", hide_index=True)
                    if len(pos_df) > 10:
                        st.caption(f"Zobrazeno prvních 10 z celkem {len(pos_df)} tokenů.")

                with filter_col:
                    st.write(f"**Odstranění stop-slov (`stopwords`):** Ponecháno {len(filtered_words)} z {len(words)} slov")
                    st.success(" ".join(filtered_words))

                    # 3. Stemming vs Lemmatizace na vybraných slovech
                    st.write("**Stemming (`PorterStemmer`) vs. Lemmatizace (`WordNetLemmatizer`):**")
                    stemmer = PorterStemmer()
                    lemmatizer = WordNetLemmatizer()
                    
                    sample_check = [w for w in filtered_words if len(w) > 4][:5]
                    if not sample_check:
                        sample_check = ["beautiful", "processing", "provides", "analysis"]
                    
                    morpho_data = []
                    for w in sample_check:
                        morpho_data.append({
                            "Původní slovo": w,
                            "Stemmer (Porter)": stemmer.stem(w),
                            "Lemma (WordNet)": lemmatizer.lemmatize(w)
                        })
                    st.dataframe(pd.DataFrame(morpho_data), width="stretch", hide_index=True)

                st.markdown("---")

                # 4. Sentiment VADER
                st.write("#### ❤️ Analýza sentimentu: NLTK VADER (`SentimentIntensityAnalyzer`)")
                sia = SentimentIntensityAnalyzer()
                scores = sia.polarity_scores(nltk_input)

                v_col1, v_col2, v_col3, v_col4 = st.columns(4)
                with v_col1:
                    st.metric("Compound Score", f"{scores['compound']:.4f}", delta="Celkový sentiment")
                with v_col2:
                    st.metric("Pozitivní složka (pos)", f"{scores['pos'] * 100:.1f} %")
                with v_col3:
                    st.metric("Neutrální složka (neu)", f"{scores['neu'] * 100:.1f} %")
                with v_col4:
                    st.metric("Negativní složka (neg)", f"{scores['neg'] * 100:.1f} %")

                # Horizontální sloupcový graf VADER
                vader_df = pd.DataFrame({
                    "Emoce": ["Pozitivní", "Neutrální", "Negativní"],
                    "Podíl": [scores["pos"], scores["neu"], scores["neg"]],
                    "Barva": ["#10b981", "#60a5fa", "#ef4444"]
                })
                fig_vader = px.bar(
                    vader_df,
                    x="Podíl",
                    y="Emoce",
                    orientation="h",
                    color="Emoce",
                    color_discrete_map={"Pozitivní": "#10b981", "Neutrální": "#60a5fa", "Negativní": "#ef4444"},
                    title="VADER Sentiment komponenty (součet = 1.0)"
                )
                fig_vader.update_layout(height=200, margin=dict(l=20, r=20, t=35, b=20), showlegend=False)
                st.plotly_chart(fig_vader, width="stretch")

    # =========================================================================
    # TAB 2: SPACY V AKCI
    # =========================================================================
    with tab2:
        st.subheader("2. Knihovna spaCy: Objektová pipeline a průmyslová síla")
        st.markdown(
            "V knihovně spaCy předáte text modelu `nlp(text)`, který v jediném bleskovém průchodu "
            "vytvoří objekt `Doc`. Ten obsahuje plně anotované tokeny, syntaktický strom i pojmenované entity."
        )

        preset_spacy = st.selectbox(
            "Vyberte ukázkový text pro spaCy nebo zadejte vlastní:",
            [
                "Apple commits $430 billion in US investments over five years.",
                "Google was founded in September 1998 by Larry Page and Sergey Brin at Stanford University in California.",
                "Tesla produced over 433,000 electric vehicles in Berlin and Shanghai during Q1 2024."
            ]
        )
        spacy_input = st.text_area("Vstupní text pro spaCy:", value=preset_spacy, height=75)

        nlp = get_spacy_model()

        if st.button("Spustit spaCy pipeline", width="stretch", key="btn_run_spacy"):
            if nlp is None:
                st.error("Model spaCy `en_core_web_sm` se nepodařilo načíst.")
            else:
                doc = nlp(spacy_input)

                st.markdown("#### 🏷️ Rozpoznávání pojmenovaných entit (NER – Named Entity Recognition)")
                if doc.ents:
                    ner_list = []
                    for ent in doc.ents:
                        ner_list.append({
                            "Entita (Text)": ent.text,
                            "Kategorie (Label)": ent.label_,
                            "Znaky (Start, End)": f"({ent.start_char}, {ent.end_char})",
                            "Vysvětlení typu": spacy.explain(ent.label_) if spacy.explain(ent.label_) else ent.label_
                        })
                    st.dataframe(pd.DataFrame(ner_list), width="stretch", hide_index=True)
                else:
                    st.info("V textu nebyly nalezeny žádné pojmenované entity.")

                st.markdown("---")

                st.markdown("#### 🔬 Detailní inspekce tokenů (Token Attributes)")
                token_records = []
                for token in doc:
                    token_records.append({
                        "Token (text)": token.text,
                        "POS (pos_)": token.pos_,
                        "Podrobný Tag": token.tag_,
                        "Lemma (lemma_)": token.lemma_,
                        "Je Stopword? (is_stop)": "Ano" if token.is_stop else "Ne",
                        "Syntaktická role (dep_)": token.dep_,
                        "Vektorová norma (vector_norm)": f"{token.vector_norm:.3f}" if token.has_vector else "0.000"
                    })
                st.dataframe(pd.DataFrame(token_records), width="stretch", hide_index=True)

                st.markdown("---")

                # Rozdělení slovních druhů v textu
                st.markdown("#### 📊 Rozdělení slovních druhů v analyzovaném textu:")
                pos_counts = pd.Series([t.pos_ for t in doc]).value_counts().reset_index()
                pos_counts.columns = ["Slovní druh (POS)", "Počet výskytů"]
                
                fig_pos = px.bar(
                    pos_counts,
                    x="Slovní druh (POS)",
                    y="Počet výskytů",
                    color="Slovní druh (POS)",
                    title="Četnost slovních druhů (spaCy Part-of-Speech)"
                )
                fig_pos.update_layout(height=280, margin=dict(l=20, r=20, t=35, b=20), showlegend=False)
                st.plotly_chart(fig_pos, width="stretch")

    # =========================================================================
    # TAB 3: BENCHMARKY A ARCHITEKTURA
    # =========================================================================
    with tab3:
        st.subheader("3. NLTK vs. spaCy: Filosofie, Architektura a Výkon")
        st.markdown(
            "Obě knihovny vznikly v jinou dobu a pro jiné publikum. Zatímco NLTK je lingvistická laboratoř, "
            "spaCy je závodní motor navržený pro zpracování milionů dokumentů v produkci."
        )

        col_arch1, col_arch2 = st.columns(2)
        with col_arch1:
            st.info(
                """
                #### 📚 NLTK (Akademický výzkum)
                - **Datový model:** Řetězce a seznamy (`str` $\\to$ `list[str]`).
                - **Filosofie:** Nabízí mnoho různých algoritmů pro jednu věc (např. 4 různé tokenizéry, 3 stemmery).
                - **Učení:** Ideální pro studenty – donutí vás pochopit každý dílčí algoritmus zvlášť.
                - **Slabina:** Pomalé zpracování velkých korpusů, chybí jednotný objekt reprezentující dokument.
                """
            )
        with col_arch2:
            st.success(
                """
                #### ⚡ spaCy (Průmyslová produkce)
                - **Datový model:** Centrální objekt `Doc`, který drží reference na `Token` a `Span`.
                - **Filosofie:** Jeden vyladěný SOTA model pro každou úlohu (žádné rozhodovací dilema).
                - **Rychlost:** Napsáno v optimalizovaném Cythonu, podpora streamování přes `nlp.pipe(batch_size=1000)`.
                - **Výhoda:** Všechny anotace (lemma, tagy, entity) vznikají v jednom průchodu.
                """
            )

        st.markdown("---")
        st.subheader("📊 Porovnání klíčových metrik (Benchmark 2026)")

        benchmarks_data = pd.DataFrame({
            "Kritérium": [
                "Rychlost tokenizace a tagování",
                "Přesnost pojmenovaných entit (NER)",
                "Ucelenost objektového modelu",
                "Podpora moderních Transformerů (RoBERTa)",
                "Přívětivost pro začátečníky / výuku",
                "Množství výzkumných korpusů a lexikonů"
            ],
            "NLTK": [25, 45, 30, 20, 95, 100],
            "spaCy": [95, 90, 95, 95, 75, 60]
        })

        fig_bench = go.Figure()
        fig_bench.add_trace(go.Bar(
            x=benchmarks_data["Kritérium"],
            y=benchmarks_data["NLTK"],
            name="NLTK",
            marker_color="#3b82f6"
        ))
        fig_bench.add_trace(go.Bar(
            x=benchmarks_data["Kritérium"],
            y=benchmarks_data["spaCy"],
            name="spaCy",
            marker_color="#10b981"
        ))
        fig_bench.update_layout(
            barmode="group",
            height=360,
            margin=dict(l=20, r=20, t=30, b=80),
            yaxis_title="Relativní skóre (0–100)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_tickangle=-25
        )
        st.plotly_chart(fig_bench, width="stretch")

    # =========================================================================
    # TAB 4: HEAD-TO-HEAD ROZBOR
    # =========================================================================
    with tab4:
        st.subheader("4. Srovnání tváří v tvář: Jak NLTK a spaCy řeší jazykové záludnosti?")
        st.markdown(
            "Otestujte, jak se obě knihovny popasují s komplikovanými strukturami, jako jsou kontrakce (*don't*), "
            "finanční částky (*$430 billion*) nebo vlastní jména."
        )

        h2h_input = st.text_input(
            "Zadejte testovací větu pro přímé srovnání:",
            value="I don't think Apple's $430B investment was a bad idea, Dr. Watson."
        )

        col_h1, col_h2 = st.columns(2)

        # NLTK větev
        with col_h1:
            st.markdown("#### 📚 Výstup NLTK:")
            if NLTK_AVAILABLE:
                nltk_toks = word_tokenize(h2h_input)
                st.write(f"**Počet tokenů:** {len(nltk_toks)}")
                st.code(str(nltk_toks))
                st.caption(
                    "Povšimněte si, jak NLTK rozdělilo kontrakci: `don't` $\\to$ `['do', \"n't\"]` a přivlastňovací pád `Apple's` $\\to$ `['Apple', \"'s\"]`."
                )
            else:
                st.warning("NLTK nedostupné.")

        # spaCy větev
        with col_h2:
            st.markdown("#### ⚡ Výstup spaCy:")
            if nlp is not None:
                spacy_doc = nlp(h2h_input)
                spacy_toks = [t.text for t in spacy_doc]
                st.write(f"**Počet tokenů:** {len(spacy_toks)}")
                st.code(str(spacy_toks))
                
                ents_found = [(e.text, e.label_) for e in spacy_doc.ents]
                st.write(f"**Detekované entity:** {ents_found}")
            else:
                st.warning("spaCy model nedostupný.")

        st.markdown("---")
        st.subheader("💡 Kdy v praxi zvolit který nástroj?")
        c_dec1, c_dec2 = st.columns(2)
        with c_dec1:
            st.markdown(
                """
                **Kdy sáhnout po NLTK:**
                - Potřebujete rychle analyzovat sentiment jednoduchých recenzí přes **VADER** bez stahování neuronových modelů.
                - Využíváte lexikální databázi **WordNet** (hledání synonym `synsets`, hypernym, hyponym).
                - Provádíte lingvistický výzkum a chcete porovnávat různé algoritmy tokenizace či kmenování.
                """
            )
        with c_dec2:
            st.markdown(
                """
                **Kdy zvolit spaCy:**
                - Budujete reálný firemní produkt (vyhledávač, CRM třídič e-mailů, chatbot).
                - Potřebujete spolehlivou extrakci entit (**NER**: jména osob, organizací, lokalit, peněz a dat).
                - Zpracováváte velké objemy textu a rychlost je klíčový požadavek (využití C/Cythonu a GPU).
                """
            )

    # =========================================================================
    # TAB 5: VĚDOMOSTNÍ KVÍZ
    # =========================================================================
    with tab5:
        st.subheader("📝 Rychlý vědomostní kvíz k ověření pochopení NLTK a spaCy")

        q1 = st.radio(
            "1. Jaká je hlavní architektonická výhoda knihovny spaCy oproti NLTK při zpracování velkých objemů textu?",
            [
                "spaCy je napsáno v optimalizovaném Cythonu a v jednom průchodu vytvoří objekt Doc se všemi anotacemi, což z něj činí nástroj až 20x rychlejší.",
                "spaCy neumí tokenizovat slova, proto je rychlejší.",
                "NLTK je určeno výhradně pro mobilní telefony, zatímco spaCy pro superpočítače.",
                "spaCy nepoužívá žádné jazykové modely."
            ]
        )

        q2 = st.radio(
            "2. Co v knihovně NLTK představuje modul VADER (SentimentIntensityAnalyzer)?",
            [
                "Nástroj pro trénování hlubokých konvolučních sítí na obrázcích.",
                "Lexikální a pravidlový analyzátor sentimentu, který vrací míru pozitivity, neutrality, negativity a souhrnné skóre compound od -1 do +1.",
                "Algoritmus pro mechanické kmenování slov.",
                "Modul pro stahování článků z Wikipedie."
            ]
        )

        q3 = st.radio(
            "3. Který atribut v knihovně spaCy obsahuje seznam detekovaných pojmenovaných entit (např. organizace, státy, peníze)?",
            [
                "doc.tokens",
                "doc.ents",
                "doc.lemmas",
                "doc.tags"
            ]
        )

        q4 = st.radio(
            "4. Proč NLTK vyžaduje příkaz nltk.download('punkt') nebo nltk.download('stopwords')?",
            [
                "Protože knihovna NLTK se instaluje jako prázdná kostra a jednotlivé korpusy, slovníky a gramatická pravidla se stahují modulárně podle potřeby.",
                "Protože bez toho by operační systém Windows smazal Python.",
                "Příkaz stahuje placenou licenci.",
                "Tento příkaz aktualizuje grafickou kartu."
            ]
        )

        if st.button("Vyhodnotit kvíz NLTK & spaCy", width="stretch"):
            score = 0
            if "Cythonu a v jednom průchodu vytvoří objekt Doc" in q1:
                score += 1
            if "souhrnné skóre compound od -1 do +1" in q2:
                score += 1
            if "doc.ents" in q3:
                score += 1
            if "jednotlivé korpusy, slovníky a gramatická pravidla se stahují modulárně" in q4:
                score += 1

            if score == 4:
                st.balloons()
                st.success("🎉 Skvěle! 4 ze 4 správně! Perfektně rozumíte rozdílům mezi NLTK a spaCy.")
            else:
                st.warning(f"Získali jste {score} ze 4 bodů. Projděte si záložky s ukázkami kódu a srovnáním.")


render_nltk_spacy_view()
