"""
Den 4: NLP – Klíčové principy práce s textem (Teorie, Interaktivní Laboratoř & SOTA 10/2026)
=============================================================================================
Podle přednášky: Key_principles_of_working_with_text.pdf (23 slidů)
Modul pokrývá:
1. Úvod do NLP a příprava nestrukturovaných textových dat.
2. Dělení textu na tokeny (Pure Python vs. NLTK vs. spaCy).
3. Normalizace, lowercasing a odstranění akcentů (unidecode).
4. Odstranění stop-slov a riziko ztráty negace / sémantiky v sentimentu.
5. Stemming (Porter) vs. Lemmatizace (spaCy morfologický analyzátor).
6. Živá interaktivní textová laboratoř s Plotly grafem redukce korpusu.
7. Expertní analýza & Moderní standardy 10/2026 (Subword tokenizace BPE/WordPiece, proč LLM nepotřebují stopwords, Rust tokenizéry).
8. Interaktivní vědomostní kvíz.
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re

# Import českého NLP modulu
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NLP_DIR = os.path.join(BASE_DIR, "04_NLP")
if NLP_DIR not in sys.path:
    sys.path.append(NLP_DIR)

try:
    from czech_nlp import czech_stem, czech_lemmatize, CZECH_STOPWORDS, CZECH_NEGATIONS, is_czech_negation
    CZECH_NLP_AVAILABLE = True
except Exception:
    CZECH_NLP_AVAILABLE = False

# Bezpečný import NLP knihoven s fallbacky
try:
    import unidecode
    def clean_accents(text: str) -> str:
        return unidecode.unidecode(text)
except ImportError:
    import unicodedata
    def clean_accents(text: str) -> str:
        return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords as nltk_stopwords
    from nltk.stem import PorterStemmer
    try:
        nltk_stop_set = set(nltk_stopwords.words("english"))
    except Exception:
        nltk_stop_set = {"the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with", "is", "was", "this", "that"}
    stemmer = PorterStemmer()
    NLTK_AVAILABLE = True
except Exception:
    NLTK_AVAILABLE = False
    nltk_stop_set = {"the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with", "is", "was", "this", "that"}
    stemmer = None

try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = None
except Exception:
    nlp = None


st.set_page_config(page_title="Den 4: Klíčové principy práce s textem", page_icon="📖", layout="wide")

st.markdown("""
# 📖 Den 4: Klíčové principy práce s textem (NLP)
### Od surových řetězců k tokenům, lemmatizaci a moderní reprezentaci jazyka
""")

st.info(r"""
**Vítej ve 4. dni kurzu Data Science & Machine Learning!** Dnes opouštíme čistě tabulková čísla a vstupujeme do světa 
nestrukturovaných textových dat (**Natural Language Processing – NLP**). V tomto modulu se seznámíme se základním 
řetězcem předzpracování (*NLP Preprocessing Pipeline*): **tokenizací, normalizací, odstraňováním stop-slov a redukcí 
tvarů (stemming vs. lemmatizace)**, a prozkoumáme, jak tyto koncepty fungují v klasickém ML i v moderní éře velkých 
jazykových modelů (LLM 10/2026).
""")

# Horní KPI karty
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Základní atom textu", "Token", delta="Slovo / Znak / Subword")
with c2:
    st.metric("Normalizace znaků", ".lower() + unidecode", delta="Eliminace akcentů & velikostí")
with c3:
    st.metric("Redukce tvarů", "Lemmatizace > Stemming", delta="Slovníkový základ vs. ořezání")
with c4:
    st.metric("SOTA Standard 10/2026", "Subword BPE & LLM", delta="Zachování větného kontextu")

st.divider()

# Záložky modulu
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔤 1. Tokenizace & Čištění",
    "🛑 2. Stop-slova & Šum",
    "🌿 3. Stemming vs. Lemmatizace",
    "🧪 4. Interaktivní Textová Laboratoř",
    "🔬 5. Expertní kritika & SOTA 2026",
    "📝 6. Vědomostní kvíz"
])

# =========================================================================
# TAB 1: TOKENIZACE & ZÁKLADNÍ ČIŠTĚNÍ
# =========================================================================
with tab1:
    st.subheader("1. Co je token a jak dělit věty na části?")
    st.markdown(r"""
    Počítač nerozumí větám jako celku. Prvním krokem každého textového algoritmu je **tokenizace** – rozdělení souvislého textu 
    na nejmenší diskrétní jednotky (**tokeny**). V klasickém NLP je tokenem obvykle samostatné slovo, interpunkční znaménko nebo číslo.
    """)

    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        st.markdown("""
        #### 🐍 1. Čistý Python (`split()`)
        - **Princip:** Dělí text podle bílých znaků (mezery, tabulátory).
        - **Výhoda:** Blesková rychlost, nulové závislosti.
        - **Nevýhoda:** Interpunkce zůstává přilepená ke slovům (`"tokens."` $\to$ jedno slovo s tečkou).
        ```python
        sentence = "This is a sentence."
        tokens = sentence.split()
        # ['This', 'is', 'a', 'sentence.']
        ```
        """)
    with col_t2:
        st.markdown("""
        #### 📚 2. Knihovna NLTK (`word_tokenize`)
        - **Princip:** Statistický a regexový model hranic vět a slov (*Punkt Tokenizer*).
        - **Výhoda:** Správně odděluje interpunkci, uvozovky a zkracovací tečky.
        - **Nevýhoda:** Vyžaduje stažení datových balíčků (`punkt_tab`).
        ```python
        from nltk.tokenize import word_tokenize
        tokens = word_tokenize(sentence)
        # ['This', 'is', 'a', 'sentence', '.']
        ```
        """)
    with col_t3:
        st.markdown("""
        #### ⚡ 3. Knihovna spaCy (`Doc` pipeline)
        - **Princip:** Jazykově specifický deterministický stavový automat.
        - **Výhoda:** Současně tvoří lingvistický model – tokeny mají POS tagy, lemmy i závislosti.
        - **Nevýhoda:** Větší paměťová náročnost a režie při načítání modelu.
        ```python
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(sentence)
        tokens = [t.text for t in doc]
        ```
        """)

    st.divider()
    st.subheader("2. Normalizace slov & Odstranění akcentů (`unidecode`)")
    
    col_n1, col_n2 = st.columns(2)
    with col_n1:
        st.markdown(r"""
        #### 🔠 Převod na malá písmena (`.lower()`)
        V češtině i angličtině nesou slova `"Learning"`, `"learning"` a `"LeaRNinG"` totožný význam. Bez normalizace velikosti písmen by model každou variantu chápal jako **zcela odlišný příznak ve slovníku**, což vede k:
        1. **Explozi dimenzionality** matice příznaků (*Curse of Dimensionality*).
        2. **Roztříštění frekvencí** a horší statistické spolehlivosti.
        
        ```python
        words = ["Learning", "LeaRNinG"]
        tokens_low = [w.lower() for w in words]
        unique_tokens = set(tokens_low) # {'learning'}
        ```
        """)
    with col_n2:
        st.markdown(r"""
        #### ☕ Odstranění akcentů a diakritiky (`unidecode`)
        Přednáška uvádí příklad ze slajdu 12:  
        *„When did you drink latté at our café?“*
        
        Pro anglický klasifikátor jsou slova `"latté"` a `"café"` s čárkami neznámá nebo rozdělená od standardních tvarů `"latte"` a `"cafe"`. Knihovna `unidecode` převádí libovolný Unicode znak na jeho 7bitový ASCII ekvivalent:
        
        ```python
        import unidecode
        sentence = "When did you drink latté at our café?"
        clean = unidecode.unidecode(sentence)
        # "When did you drink latte at our cafe?"
        ```
        """)


# =========================================================================
# TAB 2: STOP-SLOVA & ŠUM
# =========================================================================
with tab2:
    st.subheader("🛑 Co jsou stop-slova a proč se v klasickém NLP odstraňují?")
    st.markdown(r"""
    **Stop-slova (Stopwords)** jsou nejčastěji se vyskytující slova v jazyce (spojky, předložky, zájmena, pomocná slovesa), 
    která slouží ke stavbě větné struktury, ale v tradičních modelech typu *Bag of Words* nenesou specifický věcný obsah.
    
    Příklady v angličtině: `"the"`, `"and"`, `"of"`, `"to"`, `"a"`, `"in"`, `"at"`, `"is"`.
    """)

    col_sw1, col_sw2 = st.columns(2)
    with col_sw1:
        st.success("""
        #### 🎯 Výhody odstranění stop-slov:
        1. **Drastická redukce velikosti slovníku:** Stop-slova tvoří v běžném korpusu 30 až 50 % celkového počtu výskytů slov.
        2. **Úspora paměti a výpočetního času:** Matice BoW / TF-IDF má o desítky tisíc sloupců méně.
        3. **Zvýraznění sémantických pojmů:** Model se soustředí na podstatná jména, slovesa a přídavná jména nesoucí téma dokumentu.
        """)
    with col_sw2:
        st.error(r"""
        #### 🚨 Kardinální riziko: Ztráta negace a sentimentu!
        V obecných seznamech stop-slov v NLTK i spaCy se nacházejí záporná slova:  
        `"not"`, `"no"`, `"never"`, `"nor"`, `"neither"`, `"without"`.
        
        **Co se stane při slepém odstranění stop-slov z recenze?**
        - Původní recenze: *"This movie is **not good**, I would **never** recommend it."*
        - Po filtru stop-slov: *"movie **good** recommend"*
        - **Výsledek:** Klasifikátor vyhodnotí film jako **jednoznačně pozitivní**, ačkoliv byl absolutně zkritizován!
        """)

    st.markdown("---")
    st.subheader("Srovnání stop-slov: NLTK vs. spaCy")
    
    c_list1, c_list2 = st.columns(2)
    with c_list1:
        st.markdown(f"**NLTK seznam stop-slov ({len(nltk_stop_set)} slov):**")
        st.caption("Kompaktní seznam nejběžnějších gramatických slov.")
        sample_nltk = sorted(list(nltk_stop_set))[:25]
        st.code(", ".join(sample_nltk) + " ...")
    with c_list2:
        spacy_stops = nlp.Defaults.stop_words if nlp is not None else set()
        st.markdown(f"**spaCy seznam stop-slov ({len(spacy_stops)} slov):**")
        st.caption("Rozšířený seznam včetně zkrácených tvarů (n't, 'll, 've, 're).")
        sample_spacy = sorted(list(spacy_stops))[:25] if spacy_stops else ["(spaCy načteno)"]
        st.code(", ".join(sample_spacy) + " ...")


# =========================================================================
# TAB 3: STEMMING VS. LEMMATIZACE
# =========================================================================
with tab3:
    st.subheader("🌿 Redukce tvarů: Stemming vs. Lemmatizace")
    st.markdown(r"""
    V přirozeném jazyce se slova ohýbají (časování, skloňování, plurály, stupňování). Aby model chápal tvary 
    *„studying“*, *„studied“* a *„studies“* jako tentýž koncept, používáme dvě odlišné techniky:
    """)

    col_cmp1, col_cmp2 = st.columns(2)
    with col_cmp1:
        st.info(r"""
        ### ✂️ Stemming (Kmenování)
        - **Metoda:** Heuristické pravidlové ořezávání přípon a předpon (*Rule-based suffix stripping*).
        - **Algoritmus:** Martin Porter (1980) – **PorterStemmer**, Snowball.
        - **Klíčová vlastnost:** Výsledný kmen (*stem*) **nemusí být skutečné gramatické slovo**!
        - Příklad: `"examining"` $\to$ `"examin"`, `"university"` $\to$ `"univers"`.
        - **Rychlost:** Bleskově rychlý (pouze regulární pravidla).
        """)
    with col_cmp2:
        st.success(r"""
        ### 📖 Lemmatizace (Morfologický základ)
        - **Metoda:** Lingvistická analýza s využitím slovníku (*WordNet*) a kontextu (Part-of-Speech – slovní druh).
        - **Algoritmus:** spaCy lemmatizer, NLTK WordNetLemmatizer.
        - **Klíčová vlastnost:** Výsledné lemma je **vždy platný slovníkový tvar** (infinitive pro slovesa, 1. pád sg. pro podstatná jména).
        - Příklad: `"examining"` $\to$ `"examine"`, `"better"` $\to$ `"good"`.
        - **Rychlost:** Výpočetně náročnější (vyžaduje morfologické tabulky a POS tagging).
        """)

    st.markdown("---")
    st.subheader("Přímé srovnání chování na typických případech")

    tab3_lang = st.radio("Vyber jazyk pro srovnávací tabulku:", ["🇨🇿 Čeština", "🇬🇧 Angličtina"], horizontal=True)

    if tab3_lang == "🇨🇿 Čeština":
        st.markdown(r"""
        **Český jazyk je silně flektivní:** Podstatná a přídavná jména mají 7 pádů v jednotném i množném čísle a slovesa se časují podle osoby, čísla, času a rodu. 
        Anglický Porter stemmer na češtině zcela selže, protože předpokládá anglické přípony (*-ing, -ed, -s*). 
        Níže vidíš srovnání **českého kmenovače (Savoy/Dolamic)** a **českého morfologického lemmatizátoru**:
        """)
        comparison_words_cs = [
            "koně", "žluťoučký", "úpěl", "ďábelské", "ódy", "novém", "hradě", 
            "chladného", "večera", "byli", "dělali", "lepší", "lidem", "kočky"
        ]
        comp_rows_cs = []
        for w in comparison_words_cs:
            stem_val = czech_stem(w)
            lemma_val = czech_lemmatize(w)
            note = ""
            if w in ["koně", "byli", "lepší", "lidem"]:
                note = "Nepravidelný / supletivní tvar – Lemmatizátor vrátil základní lemma!"
            elif stem_val != lemma_val:
                note = f"Stemmer ořízl pádovou koncovku na '{stem_val}', Lemmatizátor vrátil 1. pád '{lemma_val}'"
            else:
                note = "Základní tvar (shoda)"
            comp_rows_cs.append({
                "Původní české slovo": w,
                "Český Stem (Savoy)": stem_val,
                "České Lemma": lemma_val,
                "Lingvistické vysvětlení": note
            })
        st.dataframe(pd.DataFrame(comp_rows_cs), hide_index=True, width="stretch")
    else:
        comparison_words = [
            "examining", "examination", "studies", "studying", "better", 
            "feet", "wolves", "meeting", "universe", "university"
        ]
        comp_rows = []
        for w in comparison_words:
            stem_val = stemmer.stem(w) if stemmer else w
            lemma_val = nlp(w)[0].lemma_ if nlp else w
            note = ""
            if w in ["better", "feet", "wolves"]:
                note = "Nepravidelný tvar – Stemmer selhal, Lemmatizátor uspěl!"
            elif w in ["universe", "university"]:
                note = "Over-stemming: Porter ořezal obě různá slova na stejný kmen 'univers'!"
            elif stem_val != lemma_val:
                note = f"Stemmer vytvořil neplatné slovo '{stem_val}'"
            else:
                note = "Obě metody shodné"
                
            comp_rows.append({
                "Původní slovo": w,
                "Stemming (Porter)": stem_val,
                "Lemmatizace (spaCy)": lemma_val,
                "Lingvistický komentář": note
            })
        st.dataframe(pd.DataFrame(comp_rows), hide_index=True, width="stretch")


# =========================================================================
# TAB 4: INTERAKTIVNÍ TEXTOVÁ LABORATOŘ (ČEŠTINA & ANGLIČTINA)
# =========================================================================
with tab4:
    st.subheader("🧪 Interaktivní Textová Laboratoř (Bilingual NLP Lab)")
    st.markdown("Vyzkoušej si celou pipeline předzpracování krok za krokem v **češtině i angličtině**.")

    col_lang, col_opt, col_diac = st.columns([1.3, 1.4, 1.8])
    with col_lang:
        lang_mode = st.radio("Jazykový režim (Language Mode):", ["🇨🇿 Čeština", "🇬🇧 Angličtina"], horizontal=True)
    with col_opt:
        keep_negations = st.checkbox(
            "Chránit negace před smazáním", 
            value=True, 
            help="V ČJ chrání 'ne-', 'ani', 'nikdy', 'žádný' atd., v AJ chrání 'not', 'no', 'never' atd."
        )
    with col_diac:
        if lang_mode == "🇨🇿 Čeština":
            diac_mode = st.radio("Diakritika v textu:", ["Zachovat UTF-8 (č, š, ž...)", "Odstranit (unidecode)"], horizontal=True)
        else:
            diac_mode = "Odstranit (unidecode)"

    sample_texts_cs = {
        "Ukázka 1: Klasická testovací věta": "Příliš žluťoučký kůň úpěl ďábelské ódy na novém hradě za chladného večera.",
        "Ukázka 2: Recenze s českou negací": "Tento film nebyl vůbec dobrý, herci nepředvedli žádný výkon a rozhodně bych ho nikomu nedoporučoval!",
        "Ukázka 3: Morfologie & Plurály": "Lidé a jejich kočky i věrní psi sledovali v Praze lepší film o starých městech a velkých hradech.",
        "Ukázka 4: Diakritika vs unidecode": "Včera jsme šli na výbornou kávu a čerstvý koláč do naší oblíbené kavárny.",
        "Ukázka 5: Vlastní český text": ""
    }

    sample_texts_en = {
        "Ukázka 1: Přednáška (Akcenty & Café)": "When did you drink latté at our café? This is a sample sentence to split into tokens.",
        "Ukázka 2: Sentiment recenze (Kritická negace)": "This movie was not good at all, the acting was terrible and I would never recommend watching it!",
        "Ukázka 3: Morfologie & Plurály": "The striped wolves were studying the running feet of examined mice better than universities.",
        "Ukázka 4: Vlastní anglický text": ""
    }

    curr_samples = sample_texts_cs if lang_mode == "🇨🇿 Čeština" else sample_texts_en
    preset_choice = st.selectbox("Vyber přednastavený text nebo zvol vlastní:", list(curr_samples.keys()))
    default_val = curr_samples[preset_choice] if "Vlastní" not in preset_choice else ("Zadej vlastní text..." if lang_mode == "🇨🇿 Čeština" else "Enter your own text...")
    user_text = st.text_area("Vstupní text k analýze:", value=default_val, height=100)

    if user_text.strip():
        # Rozlišení zpracování dle jazykového režimu
        if lang_mode == "🇨🇿 Čeština":
            # 1. Diakritika
            if diac_mode == "Odstranit (unidecode)":
                text_clean = clean_accents(user_text)
                words_raw = re.findall(r"\b[A-Za-z0-9_]+\b", text_clean)
            else:
                text_clean = user_text
                words_raw = re.findall(r"\b[A-Za-zÁ-ž0-9_]+\b", text_clean)
                
            words_lower = [w.lower() for w in words_raw]
            
            # 2. Stop-slova ČJ
            effective_stopwords = set(CZECH_STOPWORDS)
            if keep_negations:
                words_no_stop = [w for w in words_lower if (w not in effective_stopwords) or is_czech_negation(w)]
                removed_stopwords = [w for w in words_lower if (w in effective_stopwords) and not is_czech_negation(w)]
            else:
                words_no_stop = [w for w in words_lower if w not in effective_stopwords]
                removed_stopwords = [w for w in words_lower if w in effective_stopwords]
                
            # 3. Český Stemming vs Lemmatizace
            pipeline_table = []
            for w in words_no_stop:
                s_val = czech_stem(w)
                l_val = czech_lemmatize(w)
                pipeline_table.append({
                    "Normalizované slovo": w,
                    "Český Stem (Savoy)": s_val,
                    "České Lemma": l_val,
                    "Je shodné?": "✅ Ano" if s_val == l_val else "❌ Ne (Rozdíl)"
                })
        else:
            # 1. Unidecode pro AJ
            text_clean = clean_accents(user_text)
            words_raw = re.findall(r"\b[A-Za-z0-9_]+\b", text_clean)
            words_lower = [w.lower() for w in words_raw]
            
            # 2. Stop-slova AJ
            effective_stopwords = set(nltk_stop_set)
            if keep_negations:
                effective_stopwords -= {"not", "no", "never", "nor", "neither", "without"}
                
            words_no_stop = [w for w in words_lower if w not in effective_stopwords]
            removed_stopwords = [w for w in words_lower if w in effective_stopwords]
            
            # 3. Anglický Stemming vs Lemmatizace
            pipeline_table = []
            for w in words_no_stop:
                s_val = stemmer.stem(w) if stemmer else w
                l_val = nlp(w)[0].lemma_ if nlp else w
                pipeline_table.append({
                    "Normalizované slovo": w,
                    "Stem (Porter)": s_val,
                    "Lemma (spaCy)": l_val,
                    "Je shodné?": "✅ Ano" if s_val == l_val else "❌ Ne (Rozdíl)"
                })
            
        # Zobrazení metrik pipeline
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Počet surových slov", len(words_raw))
        m2.metric("Odstraněno stop-slov", len(removed_stopwords))
        m3.metric("Zbylých sémantických tokenů", len(words_no_stop))
        lemma_key = "České Lemma" if lang_mode == "🇨🇿 Čeština" else "Lemma (spaCy)"
        m4.metric("Unikátních lemat", len(set([row[lemma_key] for row in pipeline_table])))

        st.markdown(f"#### Detailní rozpad tokenů v pipeline ({lang_mode}):")
        df_pipe = pd.DataFrame(pipeline_table)
        st.dataframe(df_pipe, hide_index=True, width="stretch")

        # Interaktivní graf redukce velikosti korpusu
        stem_key = "Český Stem (Savoy)" if lang_mode == "🇨🇿 Čeština" else "Stem (Porter)"
        step_names = ["1. Surový text", "2. Unikátní slova", "3. Po vyjmutí Stop-slov", "4. Unikátní Stemy", "5. Unikátní Lemmata"]
        step_counts = [
            len(words_raw),
            len(set(words_lower)),
            len(set(words_no_stop)),
            len(set([r[stem_key] for r in pipeline_table])),
            len(set([r[lemma_key] for r in pipeline_table]))
        ]
        
        fig = go.Figure(go.Bar(
            x=step_names,
            y=step_counts,
            text=step_counts,
            textposition="auto",
            marker_color=["#4A90E2", "#50E3C2", "#F5A623", "#D0021B", "#9013FE"]
        ))
        fig.update_layout(
            title=f"📉 Redukce počtu unikátních termínů v jednotlivých krocích čištění ({lang_mode})",
            xaxis_title="Krok zpracování",
            yaxis_title="Počet unikátních prvků",
            template="plotly_dark",
            height=380
        )
        st.plotly_chart(fig, width="stretch")


# =========================================================================
# TAB 5: EXPERTNÍ KRITIKA & SOTA 10/2026
# =========================================================================
with tab5:
    st.subheader("🔬 Expertní kritika & SOTA standardy v NLP k říjnu 2026")
    st.markdown(r"""
    Postupy prezentované v materiálech kurzu odpovídají **klasickému statistickému NLP (před rokem 2018)**. 
    V moderní éře hlubokého učení a velkých jazykových modelů (LLM) došlo v oblasti předzpracování textu k radikální revoluci.
    """)

    c_sota1, c_sota2 = st.columns(2)
    with c_sota1:
        st.markdown(r"""
        #### 1. Proč LLM (GPT-4, Claude, Gemini) nepoužívají stop-slova a stemmery?
        - **Mechanismus Self-Attention:** Transformativní modely se učí vztahy mezi **všemi slovy** ve větě. Stop-slova (zájmena, předložky, spojky) tvoří klíčovou syntaktickou kostru, která určuje, *kdo udělal co komu*.
        - **Ztráta gramatiky:** Odstranění stop-slov ničí schopnost modelu chápat závislosti a negace.
        - **Kmenování ničí jemné nuance:** Tvary jako *„running“* (probíhající děj) vs. *„run“* (obecný fakt) mají pro neuronovou síť odlišný sémantický význam, který by stemmer smazal.
        """)
    with c_sota2:
        st.markdown(r"""
        #### 2. Subword Tokenizace nahradila dělení na celá slova
        Stará tokenizace na úrovni celých slov narážela na problém **Out-of-Vocabulary (OOV)** – neznámá slova skončila jako token `<UNK>`. Moderní modely dělí text na **pod-slova (subwords)**:
        - **BPE (Byte-Pair Encoding):** Používá GPT rodina a LLaMA. Frekvenčně slučuje nejčastější dvojice znaků/bajtů. Slovo *"unhappiness"* $\to$ `["un", "happiness"]`.
        - **WordPiece:** Používá BERT a DistilBERT (prefix `##`).
        - **SentencePiece / Unigram:** Používají modely T5, Gemma a Mistral.
        """)

    st.markdown("---")
    st.subheader("Srovnání vývoje NLP nástrojů v čase")

    era_data = [
        {"Éra": "Před 2015 (Tradiční NLP)", "Tokenizace": "Regulární výrazy, NLTK split", "Slovník": "Celá slova (50k - 200k), OOV problém", "Čištění": "Agresivní stopwords + Porter Stemmer", "Reprezentace": "Bag-of-Words, TF-IDF"},
        {"Éra": "2018–2022 (Transformer boom)", "Tokenizace": "WordPiece (BERT), BPE (GPT-2)", "Slovník": "Subwordy (30k - 50k tokenů)", "Čištění": "Žádný stemming, stop-slova zachována", "Reprezentace": "Kontextové embeddingy"},
        {"Éra": "2024–2026 (Moderní SOTA LLMs)", "Tokenizace": "Byte-level BPE v Rustu (tiktoken, HF)", "Slovník": "Velké subword slovníky (100k - 256k)", "Čištění": "Plné UTF-8 bez změn, zachování formátu", "Reprezentace": "Decodery (RoPE, FlashAttention-3)"}
    ]
    st.dataframe(pd.DataFrame(era_data), hide_index=True, width="stretch")


# =========================================================================
# TAB 6: VĚDOMOSTNÍ KVÍZ
# =========================================================================
with tab6:
    st.subheader("📝 Vědomostní kvíz: Prověř své znalosti práce s textem")
    
    q1 = st.radio(
        "1. Jaký je zásadní rozdíl mezi Stemmingem a Lemmatizací?",
        [
            "Stemming bere v úvahu kontext a gramatiku, zatímco lemmatizace pouze uřezává konce slov.",
            "Stemming heuristicky ořezává koncovky (výsledek nemusí být platné slovo), zatímco lemmatizace vrací platný slovníkový tvar na základě morfologie.",
            "Stemming se používá pouze pro češtinu, zatímco lemmatizace výhradně pro angličtinu.",
            "Mezi nimi není žádný rozdíl, jedná se o synonyma."
        ],
        index=None
    )
    if q1:
        if "Stemming heuristicky ořezává" in q1:
            st.success("✅ Správně! Stemming používá rychlá pravidla ořezávání (např. 'examining' -> 'examin'), zatímco lemmatizace vrací gramatický základ ('examine').")
        else:
            st.error("❌ Špatně. Správná odpověď je, že stemmer pouze ořezává konce a nemusí vytvořit platné slovo, zatímco lemmatizátor vrací platný slovníkový tvar.")

    st.markdown("---")
    q2 = st.radio(
        "2. Proč může být slepé odstranění stop-slov nebezpečné při analýze sentimentu recenzí?",
        [
            "Protože stop-slova zabírají příliš mnoho místa v paměti.",
            "Protože odstraněním slov jako 'not', 'no' nebo 'never' ztratíme negaci a negativní recenze se mohou jevit jako pozitivní.",
            "Protože stop-slova způsobují přetrénování rozhodovacích stromů.",
            "Protože v angličtině žádná stop-slova neexistují."
        ],
        index=None
    )
    if q2:
        if "ztratíme negaci" in q2:
            st.success("✅ Přesně tak! Věta 'This is not good' se po odstranění stop-slov změní na 'good', což model svede k opačnému sentimentu.")
        else:
            st.error("❌ Špatně. Hlavním rizikem je ztráta negací jako 'not', 'no', 'never'.")

    st.markdown("---")
    q3 = st.radio(
        "3. K čemu slouží knihovna `unidecode`?",
        [
            "K překladu textu z jednoho cizího jazyka do druhého.",
            "K převodu nestandardních Unicode znaků a diakritiky (např. 'latté', 'café') na jejich nejbližší ASCII ekvivalenty ('latte', 'cafe').",
            "K šifrování hesel uživatelů před uložením do databáze.",
            "K trénování velkých jazykových modelů."
        ],
        index=None
    )
    if q3:
        if "ASCII ekvivalenty" in q3:
            st.success("✅ Správně! Unidecode normalizuje Unicode znaky na 7bitový ASCII text pro snazší zpracování v modelech.")
        else:
            st.error("❌ Špatně. Unidecode provádí ASCII transliteraci Unicode znaků a akcentů.")
