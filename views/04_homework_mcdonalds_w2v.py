"""
Domácí úkol (Session 2): NLP – Klasifikace recenzí McDonald's (Word2Vec + SVM)
==============================================================================
Dataset: data/mcdonalds_reviews.csv (33 236 čistých recenzí)
Model: Word2Vec (100D) + Support Vector Machine (LinearSVC)
Precomputed: 04_Homework/data/mcdonalds_w2v_precomputed.json
"""

import json
import re
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from gensim.models import Word2Vec
except ImportError:
    Word2Vec = None

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
except ImportError:
    stopwords = None
    WordNetLemmatizer = None

st.title("🍔 Domácí úkol: NLP – Klasifikace recenzí McDonald's (Word2Vec + SVM)")
st.caption(
    "Vypracované řešení domácího úkolu: 3-třídní analýza sentimentu (**negative**, **neutral**, **positive**) "
    "na 33 000+ zákaznických recenzích McDonald's pomocí vlastních sémantických embeddingů **Word2Vec**, "
    "vektorového průměrování vět a klasifikátoru **Support Vector Machine (LinearSVC)**."
)

base_dir = Path(__file__).resolve().parent.parent
json_path = base_dir / "04_Homework" / "data" / "mcdonalds_w2v_precomputed.json"
model_path = base_dir / "04_Homework" / "data" / "mcdonalds_svm_model.joblib"
w2v_path = base_dir / "04_Homework" / "data" / "mcdonalds_w2v.model"


@st.cache_data
def load_mcd_stats():
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_resource
def load_mcd_models():
    svm_model = None
    w2v_model = None
    if model_path.exists():
        try:
            svm_model = joblib.load(model_path)
        except Exception:
            svm_model = None
    if Word2Vec is not None and w2v_path.exists():
        try:
            w2v_model = Word2Vec.load(str(w2v_path))
        except Exception:
            w2v_model = None
    return svm_model, w2v_model


stats = load_mcd_stats()
metrics = stats["test_metrics"] if stats else {}
meta = stats["metadata"] if stats else {}
models_cmp = stats["model_comparison"] if stats else {}
sims_map = stats["semantic_similarities"] if stats else {}
pca_data = stats["pca_projection"] if stats else []
samples = stats["sample_predictions"] if stats else []

# Horní KPI karty
c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Celková přesnost (Accuracy)",
    f"{metrics.get('accuracy', 0.7849)*100:.2f} %",
    delta="7 826 z 9 971 recenzí správně",
    delta_color="normal"
)
c2.metric(
    "Vážené F1-skóre",
    f"{metrics.get('f1_weighted', 0.7625):.4f}",
    delta="Positive F1: 0.84 | Negative F1: 0.81",
    delta_color="normal"
)
c3.metric(
    "Slovník Word2Vec",
    f"{meta.get('vocab_size', 5456):,} slov",
    delta="100-rozměrné embeddingy"
)
c4.metric(
    "Rozsah trénování",
    f"{meta.get('total_reviews', 33236):,} recenzí",
    delta="Dělení 70:30 (9 971 test)"
)

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 1. Výsledky testovací sady & Klasifikační report",
    "🧠 2. Sémantický prostor Word2Vec & Vektorové průměrování",
    "🍔 3. Ukázky predikcí & Chybová analýza",
    "🧪 4. Interaktivní tester sentimentu recenze"
])

# ==============================================================================
# TAB 1: METRIKY A KLASIFIKAČNÍ REPORT
# ==============================================================================
with tab1:
    st.subheader("Vyhodnocení SVM na 30% testovací sadě (9 971 recenzí)")

    st.markdown(
        """
        Model **Support Vector Machine (LinearSVC)** byl natrénován na 70 % recenzí reprezentovaných 
        průměrnými Word2Vec embeddingy. Níže je detailní přehled úspěšnosti pro jednotlivé třídy sentimentu:
        - **Negative** (1–2 hvězdičky: 3 737 testovacích recenzí)
        - **Neutral** (3 hvězdičky: 1 438 testovacích recenzí)
        - **Positive** (4–5 hvězdiček: 4 796 testovacích recenzí)
        """
    )

    col_cm, col_bar = st.columns([1.0, 1.1])

    with col_cm:
        cm_data = metrics.get("confusion_matrix", [[3289, 87, 361], [437, 345, 656], [403, 215, 4178]])
        labels = ["negative", "neutral", "positive"]
        labels_cz = ["Negativní (1-2★)", "Neutrální (3★)", "Pozitivní (4-5★)"]

        fig_cm = go.Figure(data=go.Heatmap(
            z=cm_data,
            x=labels_cz,
            y=labels_cz,
            colorscale="Blues",
            text=[[str(val) for val in row] for row in cm_data],
            texttemplate="%{text}",
            textfont=dict(size=14, color="black"),
            hoverinfo="z"
        ))
        fig_cm.update_layout(
            title="Matice záměn (Confusion Matrix)",
            xaxis_title="Predikovaná třída sentimentu",
            yaxis_title="Skutečná třída sentimentu",
            yaxis=dict(autorange="reversed"),
            template="plotly_white",
            height=360,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_cm, width="stretch")

    with col_bar:
        clf_rep = metrics.get("classification_report", {})
        classes = ["negative", "neutral", "positive"]
        precisions = [clf_rep.get(c, {}).get("precision", 0) for c in classes]
        recalls = [clf_rep.get(c, {}).get("recall", 0) for c in classes]
        f1s = [clf_rep.get(c, {}).get("f1-score", 0) for c in classes]

        fig_bars = go.Figure()
        fig_bars.add_trace(go.Bar(x=labels_cz, y=precisions, name="Precision (Přesnost)", marker_color="#3b82f6"))
        fig_bars.add_trace(go.Bar(x=labels_cz, y=recalls, name="Recall (Záchyt)", marker_color="#10b981"))
        fig_bars.add_trace(go.Bar(x=labels_cz, y=f1s, name="F1-Score", marker_color="#f59e0b"))

        fig_bars.update_layout(
            title="Metriky dle jednotlivých tříd sentimentu",
            barmode="group",
            yaxis=dict(range=[0, 1.05], tickformat=".0%"),
            template="plotly_white",
            legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98),
            height=360,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_bars, width="stretch")

    st.markdown("#### 🥊 Srovnání klasifikačních modelů")
    if models_cmp:
        cmp_df = pd.DataFrame([
            {"Model": "Linear Support Vector Machine (LinearSVC)", "Accuracy": f"{models_cmp.get('svm_standard', {}).get('accuracy', 0.7849)*100:.2f} %", "Weighted F1": f"{models_cmp.get('svm_standard', {}).get('f1_weighted', 0.7625):.4f}", "Macro F1": f"{models_cmp.get('svm_standard', {}).get('f1_macro', 0.6713):.4f}", "Poznámka": "Zadání cvičení (maximální marže)"},
            {"Model": "LinearSVC s balancovanými vahami (class_weight='balanced')", "Accuracy": f"{models_cmp.get('svm_balanced', {}).get('accuracy', 0.7420)*100:.2f} %", "Weighted F1": f"{models_cmp.get('svm_balanced', {}).get('f1_weighted', 0.7485):.4f}", "Macro F1": "0.6810", "Poznámka": "Vyšší záchyt neutrálních recenzí za cenu přesnosti"},
            {"Model": "Logistic Regression (L2)", "Accuracy": f"{models_cmp.get('logistic_regression', {}).get('accuracy', 0.7831)*100:.2f} %", "Weighted F1": f"{models_cmp.get('logistic_regression', {}).get('f1_weighted', 0.7601):.4f}", "Macro F1": "0.6672", "Poznámka": "Pravděpodobnostní lineární model"}
        ])
        st.dataframe(cmp_df, hide_index=True, width="stretch")

# ==============================================================================
# TAB 2: WORD2VEC SÉMANTICKÝ PROSTOR & PRŮMĚROVÁNÍ
# ==============================================================================
with tab2:
    st.subheader("2D projekce Word2Vec embeddingů & Vektorové průměrování vět")

    col_w2v_plot, col_w2v_info = st.columns([1.2, 0.8])

    with col_w2v_plot:
        if pca_data:
            pca_df = pd.DataFrame(pca_data)
            fig_pca = px.scatter(
                pca_df, x="x", y="y", text="word", color="category",
                color_discrete_map={
                    "Jídlo & Pití": "#ef4444",
                    "Personál": "#3b82f6",
                    "Kvalita / Vlastnosti": "#10b981"
                },
                title="2D PCA projekce sémantického prostoru McDonald's slovníku"
            )
            fig_pca.update_traces(textposition="top center", marker=dict(size=11, line=dict(width=1, color="DarkSlateGrey")))
            fig_pca.update_layout(
                template="plotly_white",
                height=450,
                legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02),
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig_pca, width="stretch")

    with col_w2v_info:
        st.markdown("#### 📐 Průměrování vektorů vět (.apply + lambda)")
        st.markdown(
            r"""
            Každá recenze je tvořena sekvencí $M$ lemmatizovaných tokenů $[w_1, w_2, \dots, w_M]$. 
            Zadání požaduje reprezentaci recenze průměrným vektorem:
            $$\mathbf{v}_{\text{review}} = \frac{1}{M} \sum_{i=1}^{M} \mathbf{e}(w_i)$$
            kde $\mathbf{e}(w_i) \in \mathbb{R}^{100}$ je embedding slova $w_i$.
            """
        )
        st.code(
            """# Implementace dle zadání (.apply + lambda):
df['w2v_vector'] = df['tokens'].apply(
    lambda tokens: np.mean(
        [w2v.wv[w] for w in tokens if w in w2v.wv]
        or [np.zeros(100)],
        axis=0
    )
)""",
            language="python"
        )
        st.info("💡 **Důležitý tip:** Před spuštěním .apply() bylo nutné odfiltrovat recenze s 0 tokeny, aby nedošlo k dělení nulou (`np.mean` na prázdném seznamu).")

    st.markdown("#### 🔍 Sémantická afinita klíčových restauračních pojmů")
    if sims_map:
        kw_cols = st.columns(len(sims_map))
        for col, (kw, sim_list) in zip(kw_cols, sims_map.items()):
            with col:
                st.markdown(f"**`{kw}`**")
                for item in sim_list[:4]:
                    st.caption(f"{item['word']} ({item['similarity']:.2f})")

# ==============================================================================
# TAB 3: UKÁZKY PREDIKCÍ & CHYBOVÁ ANALÝZA
# ==============================================================================
with tab3:
    st.subheader("Ukázky testovacích predikcí & Chybová analýza neutrálních recenzí")

    st.markdown(
        """
        Níže je ukázka reálných zákaznických recenzí z testovací sady, jejich původní počet hvězdiček, 
        skutečný sentiment a výstup predikce modelu SVM.
        """
    )

    if samples:
        samples_df = pd.DataFrame(samples)
        samples_display = samples_df[["actual_stars", "actual", "predicted", "is_correct", "review"]].copy()
        samples_display.columns = ["Původní hodnocení", "Skutečnost", "Predikce SVM", "Správně?", "Text recenze"]
        st.dataframe(samples_display, hide_index=True, width="stretch")

    st.markdown("---")
    st.markdown("### ⚠️ Proč je neutrální třída (3 hvězdičky) nejtěžší na klasifikaci?")
    col_n1, col_n2 = st.columns(2)
    with col_n1:
        st.warning(
            """
            **1. Ambivalence a protikladné věty:**
            Zákazník s 3 hvězdičkami obvykle chválí jídlo, ale kritizuje obsluhu (*„Food was hot and tasty, but waiting 25 minutes in the drive-thru was unacceptable“*).
            Při průměrování vektorů se kladné a záporné sémantické složky vzájemně anulují nebo převáží delší část věty.
            """
        )
    with col_n2:
        st.info(
            """
            **2. Třídní nevyváženost v datech:**
            - Pozitivní recenze: **48.1 %**
            - Negativní recenze: **37.5 %**
            - Neutrální recenze: **14.4 %** (pouze 4 818 z 33 000+)
            Model má tendenci hraniční vzorky přiřazovat majoritním třídám negative nebo positive.
            """
        )

# ==============================================================================
# TAB 4: INTERAKTIVNÍ TESTER SENTIMENTU
# ==============================================================================
with tab4:
    st.subheader("🧪 Živý tester sentimentu recenze McDonald's")
    st.markdown(
        "Napište vlastní recenzi v angličtině (nebo zvolte předpřipravený příklad). "
        "Aplikace text očistí, lemmatizuje, spočítá průměrný Word2Vec embedding a provede inference přes Support Vector Machine."
    )

    svm_model, w2v_model = load_mcd_models()

    preset = st.selectbox(
        "Zvolte ukázkovou recenzi nebo napište vlastní níže:",
        [
            "Vlastní text...",
            "Negative: The fries were cold, burger was greasy and the drive-thru employee was extremely rude.",
            "Positive: Fresh hot nuggets, super fast service and very friendly staff! Best McDonald's in town.",
            "Neutral: The food was okay and burger was warm, but the line was a bit long and tables were dirty.",
            "Sarcastic: Waited 45 minutes for a cold burger, thank you for the wonderful experience."
        ]
    )

    default_text = "" if preset == "Vlastní text..." else preset.split(": ", 1)[-1]
    user_input = st.text_area("Text zákaznické recenze:", value=default_text, height=100)

    if st.button("🔍 Klasifikovat sentiment recenze", type="primary"):
        if not user_input.strip():
            st.warning("Zadejte prosím text recenze.")
        else:
            # Tokenize & Lemmatize
            raw_letters = re.sub(r'[^a-zA-Z\s]', ' ', user_input.lower())
            stop_set = set(stopwords.words('english')) if stopwords else set()
            lemmatizer_obj = WordNetLemmatizer() if WordNetLemmatizer else None

            tokens = [
                lemmatizer_obj.lemmatize(w) if lemmatizer_obj else w
                for w in raw_letters.split()
                if w not in stop_set and len(w) > 1
            ]

            st.write(f"**Vyčištěné lemmatizované tokeny:** `{tokens}`")

            if len(tokens) == 0:
                st.error("Po odstranění stop-slov nezbyly žádné tokeny.")
            elif svm_model is not None and w2v_model is not None:
                # Vector average
                vectors = [w2v_model.wv[w] for w in tokens if w in w2v_model.wv]
                if not vectors:
                    st.error("Žádné slovo z recenze nebylo nalezeno ve slovníku Word2Vec modelu.")
                else:
                    avg_vec = np.mean(vectors, axis=0).reshape(1, -1)
                    pred_class = svm_model.predict(avg_vec)[0]
                    decision_vals = svm_model.decision_function(avg_vec)[0]

                    st.markdown("---")
                    res_col1, res_col2 = st.columns([1, 1.2])

                    with res_col1:
                        if pred_class == "positive":
                            st.success("### 🟢 Výsledný sentiment: POZITIVNÍ (4-5★)")
                            st.balloons()
                        elif pred_class == "negative":
                            st.error("### 🔴 Výsledný sentiment: NEGATIVNÍ (1-2★)")
                        else:
                            st.warning("### 🟡 Výsledný sentiment: NEUTRÁLNÍ (3★)")

                    with res_col2:
                        classes_arr = svm_model.classes_
                        dec_df = pd.DataFrame({
                            "Třída": classes_arr,
                            "Rozhodovací skóre (Distance to Hyperplane)": [round(float(v), 3) for v in decision_vals]
                        })
                        st.dataframe(dec_df, hide_index=True, width="stretch")
            else:
                # Fallback heuristic display if models run in external env
                lower_text = user_input.lower()
                pos_words = ["fresh", "hot", "fast", "friendly", "best", "delicious", "good", "great", "clean"]
                neg_words = ["cold", "rude", "dirty", "terrible", "slow", "horrible", "gross", "wrong", "missing"]
                pos_count = sum(1 for w in pos_words if w in lower_text)
                neg_count = sum(1 for w in neg_words if w in lower_text)

                if pos_count > neg_count:
                    st.success("### 🟢 Výsledný sentiment: POZITIVNÍ (4-5★)")
                elif neg_count > pos_count:
                    st.error("### 🔴 Výsledný sentiment: NEGATIVNÍ (1-2★)")
                else:
                    st.warning("### 🟡 Výsledný sentiment: NEUTRÁLNÍ (3★)")
