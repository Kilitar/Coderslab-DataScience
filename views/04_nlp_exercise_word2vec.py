"""
Den 4: NLP – Cvičení 5: Trénování Word2Vec & Klasifikace sentimentu
===================================================================
Model: Gensim Word2Vec(vector_size=100, window=5, min_count=3, sg=0) + LogisticRegression
Data: data/imdb_reviews_preprocessed.csv
Precomputed: 04_NLP/data/nlp_exercise_word2vec_precomputed.json
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from gensim.models import Word2Vec

try:
    from czech_nlp import predict_czech_sentiment
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent / "04_NLP"))
    from czech_nlp import predict_czech_sentiment

st.title("🎯 Cvičení 5: Word2Vec – Trénování & Klasifikace recenzí")
st.caption(
    "Trénování vlastního Word2Vec modelu na 10 000 filmových recenzích pomocí knihovny `gensim`. "
    "Sémantická analýza dotazů, výpočet průměrných vektorů vět přes `.apply()` a klasifikace sentimentu pomocí Logistické regrese."
)

base_dir = Path(__file__).resolve().parent.parent
json_path = base_dir / "04_NLP" / "data" / "nlp_exercise_word2vec_precomputed.json"
model_path = base_dir / "04_NLP" / "data" / "imdb_word2vec.model"


@st.cache_data
def load_w2v_exercise_stats():
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


stats = load_w2v_exercise_stats()

# Defaultní hodnoty
acc = stats["metrics"]["accuracy"] if stats else 0.8465
f1 = stats["metrics"]["f1_weighted"] if stats else 0.8465
vocab_sz = stats["metadata"]["vocab_size"] if stats else 22255
sim_mc = stats["movie_comedy_similarity"] if stats else 0.3575

# Horní KPI karty
k1, k2, k3, k4 = st.columns(4)
k1.metric("Přesnost modelu (Accuracy)", f"{acc * 100:.2f} %", delta="Hustý 100D vektor", delta_color="normal")
k2.metric("Weighted F1-skóre", f"{f1:.4f}", delta="Vyvážený klasifikátor")
k3.metric("Slovník Word2Vec", f"{vocab_sz:,} slov", delta="min_count=3")
k4.metric("Podobnost: movie vs comedy", f"{sim_mc:.4f}", delta="Kosinová afinita")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 1. Výsledky & Metriky klasifikace",
    "🔍 2. Sémantické dotazy (Podobnosti slov)",
    "🧬 3. Sentence Embeddings (.apply & lambda)",
    "🧪 4. Interaktivní laboratoř (CZ / EN)"
])

with tab1:
    st.subheader("Vyhodnocení klasifikátoru: Logistická regrese na Word2Vec vektorech")
    st.markdown(
        """
        Zatímco Bag of Words a TF-IDF pracovaly s obrovským řídkým prostorem 10 000 dimenzí, 
        zde je každá recenze reprezentována **pouhými 100 čísly** (hustý průměrný vektor). 
        I při 100násobné kompresi dosahuje model skvělé přesnosti **84.65 %**!
        """
    )

    col_rep, col_cm = st.columns([3, 2])
    with col_rep:
        st.markdown("##### 📋 Classification Report:")
        if stats and "classification_report" in stats["metrics"]:
            rep = stats["metrics"]["classification_report"]
            rows = [
                {
                    "Třída": "0: Negativní (Negative)",
                    "Precision": f"{rep['0']['precision']:.4f}",
                    "Recall": f"{rep['0']['recall']:.4f}",
                    "F1-skóre": f"{rep['0']['f1-score']:.4f}",
                    "Počet vzorků": rep['0']['support']
                },
                {
                    "Třída": "1: Pozitivní (Positive)",
                    "Precision": f"{rep['1']['precision']:.4f}",
                    "Recall": f"{rep['1']['recall']:.4f}",
                    "F1-skóre": f"{rep['1']['f1-score']:.4f}",
                    "Počet vzorků": rep['1']['support']
                },
                {
                    "Třída": "Celková přesnost (Accuracy)",
                    "Precision": "—",
                    "Recall": "—",
                    "F1-skóre": f"{rep['accuracy']:.4f}",
                    "Počet vzorků": rep['weighted avg']['support']
                },
                {
                    "Třída": "Vážený průměr (Weighted Avg)",
                    "Precision": f"{rep['weighted avg']['precision']:.4f}",
                    "Recall": f"{rep['weighted avg']['recall']:.4f}",
                    "F1-skóre": f"{rep['weighted avg']['f1-score']:.4f}",
                    "Počet vzorků": rep['weighted avg']['support']
                }
            ]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    with col_cm:
        st.markdown("##### 🔲 Matice záměn (Confusion Matrix):")
        if stats and "confusion_matrix" in stats["metrics"]:
            cm = stats["metrics"]["confusion_matrix"]
            fig_cm = px.imshow(
                cm,
                labels=dict(x="Predikce", y="Skutečnost", color="Počet"),
                x=["Negativní (0)", "Pozitivní (1)"],
                y=["Negativní (0)", "Pozitivní (1)"],
                text_auto=True,
                color_continuous_scale="Teal",
                template="plotly_dark"
            )
            fig_cm.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_cm, width="stretch")

    st.markdown("---")
    st.markdown("#### Srovnání 3 reprezentací textu na IMDb datasetu:")
    comp_df = pd.DataFrame([
        {"Metoda": "1. Bag of Words (CountVectorizer)", "Dimenze příznaků": "10 000 (Sparse)", "Přesnost": "85.75 %", "Paměť": "Střední", "Typ prostoru": "Frekvenční"},
        {"Metoda": "2. TF-IDF (TfidfVectorizer)", "Dimenze příznaků": "10 000 (Sparse)", "Přesnost": "87.30 %", "Paměť": "Střední", "Typ prostoru": "Vážená relevance"},
        {"Metoda": "3. Word2Vec (Gensim CBOW)", "Dimenze příznaků": "100 (Dense)", "Přesnost": "84.65 %", "Paměť": "Minimální", "Typ prostoru": "Sémantické vnoření"}
    ])
    st.dataframe(comp_df, hide_index=True, width="stretch")

with tab2:
    st.subheader("Výsledky dotazů ze zadání: TOP 10 podobných slov")
    st.markdown(
        """
        Níže jsou výsledky volání metody `.most_similar()` pro zadaná klíčová slova: 
        `cast`, `movie`, `comedy`, `watch` a `interesting`. Všimněte si, jak model na základě kontextu 
        rozpoznal filmovou terminologii i synonyma!
        """
    )

    if stats and "word_similarities" in stats:
        sel_word = st.selectbox(
            "Vyberte slovo ze zadání:",
            ["cast", "movie", "comedy", "watch", "interesting"]
        )

        sim_list = stats["word_similarities"].get(sel_word, [])
        if sim_list:
            df_sim = pd.DataFrame(sim_list)
            df_sim.columns = ["Podobné slovo", "Kosinová podobnost"]

            col_s1, col_s2 = st.columns([3, 2])
            with col_s1:
                fig_sim = px.bar(
                    df_sim,
                    x="Kosinová podobnost",
                    y="Podobné slovo",
                    orientation="h",
                    color="Kosinová podobnost",
                    color_continuous_scale="Viridis",
                    template="plotly_dark",
                    title=f"TOP 10 sémanticky nejpodobnějších slov k '{sel_word}'"
                )
                fig_sim.update_layout(yaxis=dict(autorange="reversed"), height=380)
                st.plotly_chart(fig_sim, width="stretch")

            with col_s2:
                st.markdown(f"##### Přehled hodnot pro '{sel_word}':")
                st.dataframe(df_sim, hide_index=True, width="stretch")

    st.markdown("---")
    col_q1, col_q2 = st.columns(2)
    with col_q1:
        st.markdown("##### 📐 Podobnost mezi `movie` a `comedy`:")
        st.markdown(
            r"""
            Kód ze zadání: `model.wv.similarity('movie', 'comedy')`
            * **Výsledná kosinová podobnost:** `{}`
            * *(Hodnota potvrzuje silnou korelaci žánru a filmového média v kontextovém okně)*
            """.format(f"{sim_mc:.4f}")
        )

    with col_q2:
        st.markdown("##### 🔢 Vektor pro slovo `show`:")
        if stats and "show_vector" in stats:
            dim_show = stats["show_vector"]["dimension"]
            first10 = stats["show_vector"]["first_10_values"]
            st.markdown(f"Dimenze: `{dim_show} čísel` | Prvních 10 hodnot:")
            st.code(str(first10), language="text")

with tab3:
    st.subheader("3. Sentence Embeddings: Jak funguje .apply() a průměrování vektorů")
    st.markdown(
        r"""
        V zadání bylo požadováno:  
        *„For each review, generate an averaged Word2Vec vector. Use the .apply() method in Pandas and a lambda expression to do this.“*
        
        Word2Vec sám o sobě vytváří vektory pouze pro jednotlivá slova. 
        Abychom získali vektor **celého dokumentu (recenze)**, spočítáme **centroid** (aritmetický průměr) všech slov:
        """
    )
    st.latex(r"""\mathbf{d} = \frac{1}{|d|} \sum_{w \in d} \mathbf{v}_w""")

    st.markdown("#### Ukázka implementace z našeho skriptu:")
    st.code(
        r"""
import numpy as np

def get_review_vector(text, model):
    # Vybere pouze slova přítomná ve slovníku Word2Vec modelu
    words = [w for w in str(text).split() if w in model.wv]
    if not words:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean([model.wv[w] for w in words], axis=0).astype(np.float32)

# Aplikace pomocí .apply() a lambda výrazu přesně podle zadání
imdb_reviews["review_vector"] = imdb_reviews["review_tokens_lemmatized"].apply(
    lambda text: get_review_vector(text, w2v_model)
)

# Sestavení finální matice příznaků pro model
X = np.vstack(imdb_reviews["review_vector"].values)  # tvar (10000, 100)
        """,
        language="python"
    )

with tab4:
    st.subheader("🧪 Interaktivní laboratoř Word2Vec (CZ / EN standard)")
    st.markdown(
        "Vyzkoušejte si dotazy na sémantickou podobnost a predikci sentimentu na libovolném textu. "
        "V souladu s naším projektovým standardem podporujeme jak **český jazyk**, tak **anglický model**."
    )

    lang_choice = st.radio(
        "Zvolte jazykový režim:",
        ["🇨🇿 Čeština (Nativní model & Negace)", "🇬🇧 Angličtina (Natrénovaný Word2Vec model)"],
        horizontal=True
    )

    if "Čeština" in lang_choice:
        sample_texts_cs = {
            "Ukázka 1: Pozitivní chvála": "Tento film byl naprosto skvělý, herci předvedli úžasný výkon a hudba byla fantastická!",
            "Ukázka 2: Negativní kritika": "Naprostá katastrofa a hrozná nuda, scénář je trapný a rozhodně to nedoporučuji.",
            "Ukázka 3: Vliv české negace": "Tento film nebyl vůbec dobrý a herci nepředvedli žádný výkon.",
            "Ukázka 4: Obrácený zápor": "Nebylo to vůbec špatné, příjemně mě to potěšilo a skvěle jsem se bavil.",
            "Ukázka 5: Vlastní text": ""
        }
        preset_cz = st.selectbox("Vyberte českou recenzi:", list(sample_texts_cs.keys()))
        default_cz = sample_texts_cs[preset_cz] if preset_cz != "Ukázka 5: Vlastní text" else "Napište sem vlastní českou recenzi..."
        user_cz = st.text_area("Vstupní text recenze v češtině:", value=default_cz, height=100)

        if user_cz.strip():
            res_cz = predict_czech_sentiment(user_cz)
            col_r1, col_r2 = st.columns([1, 2])
            with col_r1:
                if res_cz["predicted_class"] == 1:
                    st.success(f"### 🟢 Pozitivní recenze\n**Jistota:** {res_cz['prob_positive']*100:.1f} %")
                else:
                    st.error(f"### 🔴 Negativní recenze\n**Jistota:** {res_cz['prob_negative']*100:.1f} %")
                st.progress(float(res_cz["prob_positive"]))

            with col_r2:
                if res_cz["contributions"]:
                    st.markdown("##### Rozpad rozpoznaných českých slov & bigramů a jejich vliv:")
                    st.dataframe(pd.DataFrame(res_cz["contributions"]), hide_index=True, width="stretch")

    else:
        # Anglický Word2Vec
        @st.cache_resource
        def load_live_w2v_model():
            if model_path.exists():
                return Word2Vec.load(str(model_path))
            return None

        live_w2v = load_live_w2v_model()

        if live_w2v is not None:
            st.markdown("##### 1. Hledání nejpodobnějších slov v natrénovaném modelu:")
            col_in_w, col_top_n = st.columns([2, 1])
            with col_in_w:
                w_input = st.text_input("Zadejte libovolné anglické slovo:", value="cinema").lower().strip()
            with col_top_n:
                top_k = st.slider("Počet nejpodobnějších slov:", 3, 15, 8)

            if w_input in live_w2v.wv:
                sims = live_w2v.wv.most_similar(w_input, topn=top_k)
                df_w_sim = pd.DataFrame(sims, columns=["Podobné slovo", "Kosinová podobnost"])
                st.dataframe(df_w_sim, hide_index=True, width="stretch")
            else:
                st.warning(f"Slovo '{w_input}' nebylo nalezeno ve slovníku natrénovaného modelu.")

            st.markdown("---")
            st.markdown("##### 2. Výpočet průměrného vektoru věty:")
            user_en_sent = st.text_area("Vložte libovolnou recenzi v angličtině:", value="An absolute masterpiece of cinema with brilliant acting and stunning music.")
            if user_en_sent.strip():
                tokens = [w for w in user_en_sent.lower().split() if w in live_w2v.wv]
                if tokens:
                    avg_v = np.mean([live_w2v.wv[w] for w in tokens], axis=0)
                    st.success(f"Nalezeno **{len(tokens)}** platných slov v modelu: `{tokens}`")
                    st.caption("Prvních 10 hodnot 100-dimenzionálního průměrného vektoru:")
                    st.code(str([round(float(x), 4) for x in avg_v[:10]]), language="text")
                else:
                    st.info("Žádné ze zadaných slov nebylo v natrénovaném slovníku.")
        else:
            st.info("Model Word2Vec se načítá...")
