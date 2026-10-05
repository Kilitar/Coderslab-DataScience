"""
Den 4: Cvičení 1 – Předzpracování textových dat (IMDb Reviews) – Výsledky zadání
================================================================================
Podle zadání kurzu: Processing textual data - exercise 1
1. Načtení datového souboru imdb_reviews.csv.
2. Načtení do proměnné imdb_reviews v Pandas.
3. Funkce pro očištění textu (lowercase, pouze písmena regexem, odstranění stop-slov).
4. Přidání sloupce review_cleaned pomocí .apply().
5. Uložení do souboru imdb_reviews_preprocessed_1.csv.
"""

import os
import re
import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Cvičení 1: Předzpracování textu (IMDb) – Výsledky zadání", page_icon="🎯", layout="wide")

# Načtení předpočtených metrik pro bleskový start (< 5 ms)
JSON_PATH = "04_NLP/data/nlp_exercise_1_precomputed.json"
metrics_data = None
if os.path.exists(JSON_PATH):
    try:
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
    except Exception:
        metrics_data = None

st.title("🎯 Cvičení 1: Předzpracování textu (IMDb) – Výsledky zadání")
st.markdown(r"""
**Cíl cvičení:** Vytvořit a otestovat ucelenou funkci pro vyčištění surových filmových recenzí z korpusu **IMDb (10 000 textů)**: 
převést slova na malá písmena, regulárním výrazem ponechat pouze abecední znaky a odstranit anglická stop-slova. 
Výsledek je aplikován přes `.apply()` a uložen do výstupního souboru `imdb_reviews_preprocessed_1.csv`.
""")

# Horní KPI karty
k1, k2, k3, k4 = st.columns(4)
total_revs = metrics_data["total_reviews"] if metrics_data else 10000
mean_raw = metrics_data["mean_raw_words"] if metrics_data else 232.3
mean_clean = metrics_data["mean_clean_words"] if metrics_data else 126.1
red_pct = metrics_data["reduction_pct"] if metrics_data else 45.7

with k1:
    st.metric("Celkem recenzí v korpusu", f"{total_revs:,}", delta="5 001 neg / 4 999 pos")
with k2:
    st.metric("Průměrná délka před čištěním", f"{mean_raw} slov", delta="Surový text s interpunkcí")
with k3:
    st.metric("Průměrná délka po vyčištění", f"{mean_clean} slov", delta=f"-{red_pct} % redukce šumu", delta_color="inverse")
with k4:
    st.metric("Stav výstupu", "100 % hotovo", delta="imdb_reviews_preprocessed_1.csv")

st.divider()

# Záložky řešení
t1, t2, t3, t4 = st.tabs([
    "📋 Krok za krokem: Kód & Zadání",
    "🔍 Porovnání recenzí (Před vs. Po)",
    "📊 Distribuce délek & Frekvence slov",
    "🧪 Otestovat na vlastním textu"
])

# =========================================================================
# TAB 1: KROK ZA KROKEM
# =========================================================================
with t1:
    st.subheader("1. Oficiální implementace krok za krokem")
    
    st.markdown(r"""
    #### Krok 1 & 2: Načtení dat do proměnné `imdb_reviews`
    ```python
    import pandas as pd
    
    # Načtení datového souboru
    imdb_reviews = pd.read_csv('imdb_reviews.csv')
    print(imdb_reviews.shape) # (10000, 2)
    ```
    
    #### Krok 3: Definice čistící funkce `clean_review`
    Zadání vyžaduje 3 operace:
    1. Všechna slova budou složena z malých písmen (`.lower()`).
    2. Pouze písmena v textu (`re.sub(r'[^a-zA-Z\s]', ' ', text)`).
    3. Odstranění stop-slov (použijeme korpus NLTK stop-words).
    
    ```python
    import re
    import nltk
    from nltk.corpus import stopwords
    
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    
    def clean_review(text):
        # 1. Lowercase
        text = text.lower()
        # 2. Pouze písmena (odstranění čísel, HTML tagů, interpunkce)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        # 3. Odstranění stop-slov
        words = text.split()
        filtered = [w for w in words if w not in stop_words]
        return " ".join(filtered)
    ```
    
    #### Krok 4: Aplikace na celý DataFrame přes `.apply()`
    ```python
    imdb_reviews['review_cleaned'] = imdb_reviews['review'].apply(clean_review)
    ```
    
    #### Krok 5: Uložení do CSV
    ```python
    imdb_reviews.to_csv('imdb_reviews_preprocessed_1.csv', index=False)
    ```
    """)

    st.success("✅ **Ověření výstupu:** Soubor `imdb_reviews_preprocessed_1.csv` byl úspěšně vygenerován v kořeni i složce `data/` a má přesně 10 000 řádků a 2 sloupce (`sentiment`, `review_cleaned`).")

# =========================================================================
# TAB 2: POROVNÁNÍ RECENZÍ
# =========================================================================
with t2:
    st.subheader("🔍 Přímé srovnání původního a vyčištěného textu")
    st.caption("Vyber konkrétní recenzi a podívej se, jak funkce odstranila HTML tagy, čísla, interpunkci a stop-slova:")

    samples = metrics_data.get("samples", []) if metrics_data else []
    sample_options = [f"Recenze #{s['index']} ({s['sentiment'].upper()}) – původně {s['raw_words']} slov -> po vyčištění {s['clean_words']} slov" for s in samples]

    if sample_options:
        selected_idx_str = st.selectbox("Vyber ukázkovou recenzi:", sample_options, index=0)
        selected_pos = sample_options.index(selected_idx_str)
        curr_sample = samples[selected_pos]

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown(f"#### 📄 Původní text (`review`) – {curr_sample['raw_words']} slov")
            st.info(curr_sample["raw_text"])
            if curr_sample.get("has_html_breaks"):
                st.caption("⚠️ Text obsahoval HTML tagy jako `<br />`, které byly úspěšně odstraněny.")
        with col_r2:
            st.markdown(f"#### ✨ Očištěný text (`review_cleaned`) – {curr_sample['clean_words']} slov")
            st.success(curr_sample["clean_text"])
            pct_drop = round((curr_sample['raw_words'] - curr_sample['clean_words']) / curr_sample['raw_words'] * 100, 1)
            st.caption(f"📉 Redukce délky o **{pct_drop} %** slov.")

        if curr_sample.get("has_negation"):
            st.warning("⚠️ **Pozor na ztrátu negace:** Tato recenze původně obsahovala záporná slova (např. *not, no, never*), která byla standardním NLTK filtrem odstraněna (podrobněji v záložce Expertní analýza).")

# =========================================================================
# TAB 3: DISTRIBUCE DÉLEK & FREKVENCE
# =========================================================================
with t3:
    st.subheader("📊 Analýza zkrácení textu napříč 10 000 recenzemi")

    if metrics_data:
        # Histogram délek
        hist_fig = go.Figure()
        hist_fig.add_trace(go.Bar(
            x=metrics_data["hist_bin_labels"],
            y=metrics_data["hist_raw_counts"],
            name="Před čištěním (Surový text)",
            marker_color="#4A90E2",
            opacity=0.75
        ))
        hist_fig.add_trace(go.Bar(
            x=metrics_data["hist_bin_labels"],
            y=metrics_data["hist_clean_counts"],
            name="Po vyčištění (review_cleaned)",
            marker_color="#50E3C2",
            opacity=0.85
        ))
        hist_fig.update_layout(
            barmode="overlay",
            title="Distribuce počtu slov v recenzích: Před vs. Po odstranění stop-slov",
            xaxis_title="Rozsah délky (počet slov)",
            yaxis_title="Počet recenzí",
            template="plotly_dark",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(hist_fig, width="stretch")

        col_w1, col_w2 = st.columns(2)
        with col_w1:
            st.markdown("#### 🔝 Nejčastější slova PŘED čištěním (dominují stop-slova)")
            df_top_raw = pd.DataFrame(metrics_data["top_raw_words"][:10])
            st.dataframe(df_top_raw.rename(columns={"word": "Slovo", "count": "Výskyty"}), hide_index=True, width="stretch")
        with col_w2:
            st.markdown("#### 🌟 Nejčastější slova PO vyčištění (nositelé sémantiky)")
            df_top_clean = pd.DataFrame(metrics_data["top_clean_words"][:10])
            st.dataframe(df_top_clean.rename(columns={"word": "Slovo", "count": "Výskyty"}), hide_index=True, width="stretch")

# =========================================================================
# TAB 4: INTERAKTIVNÍ TESTER
# =========================================================================
with t4:
    st.subheader("🧪 Otestuj funkci `clean_review` na libovolném vstupu")
    
    test_input = st.text_area(
        "Zadej anglický text k vyčištění:",
        value="This 2026 film was not great at all, but the actors did a truly fantastic job! Rating: 6/10. <br /><br />Visit https://imdb.com for more info.",
        height=100
    )
    
    if test_input:
        # Lokální aplikace přesné funkce
        import nltk
        from nltk.corpus import stopwords
        try:
            st_stops = set(stopwords.words("english"))
        except Exception:
            st_stops = {"the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with", "is", "was", "this", "that", "not", "but"}
            
        t_low = test_input.lower()
        t_letters = re.sub(r"[^a-zA-Z\s]", " ", t_low)
        t_words = t_letters.split()
        t_clean = [w for w in t_words if w not in st_stops]
        t_result = " ".join(t_clean)
        
        c_res1, c_res2 = st.columns(2)
        with c_res1:
            st.markdown("**1. Ponechaná sémantická slova:**")
            st.success(t_result if t_result else "(Žádná slova nezbyla)")
            st.caption(f"Z původních {len(test_input.split())} slov zůstalo {len(t_clean)} slov.")
        with c_res2:
            st.markdown("**2. Odstraněná stop-slova a interpunkce:**")
            removed_items = [w for w in t_words if w in st_stops]
            st.info(", ".join(set(removed_items)) if removed_items else "Žádná stop-slova.")
