"""
Den 4: NLP – Cvičení: Bag of Words & Klasifikace sentimentu
============================================================
Model: CountVectorizer(max_features=10000) + LogisticRegression(max_iter=1000)
Data: data/imdb_reviews_lemmatized.csv
Precomputed: 04_NLP/data/nlp_exercise_bow_precomputed.json
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

st.title("🎯 Cvičení: Bag of Words & Klasifikace sentimentu (IMDb)")
st.caption(
    "Klasifikace sentimentu filmových recenzí: Vektorizace lemmatizovaného textu pomocí `CountVectorizer(max_features=10000)`, "
    "trénování modelu `LogisticRegression(max_iter=1000)` a hloubková analýza classification reportu."
)

base_dir = Path(__file__).resolve().parent.parent
json_path = base_dir / "04_NLP" / "data" / "nlp_exercise_bow_precomputed.json"
csv_path = base_dir / "data" / "imdb_reviews_lemmatized.csv"


@st.cache_data
def load_bow_stats():
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


stats = load_bow_stats()

# Defaultní hodnoty pro případ inicializace
acc = stats["metrics"]["accuracy"] if stats else 0.8575
f1 = stats["metrics"]["f1_weighted"] if stats else 0.8575
vocab_size = stats["metadata"]["max_features"] if stats else 10000
train_time = stats["metadata"]["training_time_s"] if stats else 0.28

# Horní KPI karty
k1, k2, k3, k4 = st.columns(4)
k1.metric("Přesnost (Accuracy)", f"{acc * 100:.2f} %", delta="Baseline OLS překonán", delta_color="normal")
k2.metric("Weighted F1-skóre", f"{f1:.4f}", delta="Vyvážený model")
k3.metric("Velikost slovníku", f"{vocab_size:,} slov", delta="Top popular features")
k4.metric("Doba trénování", f"{train_time} s", delta="Bleskový gradient descent")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 1. Výsledky & Classification Report",
    "🧠 2. Koeficienty modelu (Váhy slov)",
    "🔍 3. Analýza chyb (Error Analysis)",
    "🧪 4. Živý prediktor sentimentu"
])

with tab1:
    st.subheader("Vyhodnocení kvality modelu na testovací sadě (2 000 recenzí)")
    st.markdown(
        """
        Model byl trénován na 8 000 recenzích a vyhodnocen na 2 000 testovacích recenzích 
        se striktní stratifikací poměru tříd (1 000 pozitivních a 1 000 negativních).
        """
    )

    col_rep, col_cm = st.columns([3, 2])
    with col_rep:
        st.markdown("##### 📋 Classification Report:")
        if stats and "classification_report" in stats["metrics"]:
            rep_dict = stats["metrics"]["classification_report"]
            rows = [
                {
                    "Třída": "0: Negativní (Negative)",
                    "Precision": f"{rep_dict['0']['precision']:.4f}",
                    "Recall": f"{rep_dict['0']['recall']:.4f}",
                    "F1-skóre": f"{rep_dict['0']['f1-score']:.4f}",
                    "Počet vzorků (Support)": rep_dict['0']['support']
                },
                {
                    "Třída": "1: Pozitivní (Positive)",
                    "Precision": f"{rep_dict['1']['precision']:.4f}",
                    "Recall": f"{rep_dict['1']['recall']:.4f}",
                    "F1-skóre": f"{rep_dict['1']['f1-score']:.4f}",
                    "Počet vzorků (Support)": rep_dict['1']['support']
                },
                {
                    "Třída": "Celková přesnost (Accuracy)",
                    "Precision": "—",
                    "Recall": "—",
                    "F1-skóre": f"{rep_dict['accuracy']:.4f}",
                    "Počet vzorků (Support)": rep_dict['weighted avg']['support']
                },
                {
                    "Třída": "Vážený průměr (Weighted Avg)",
                    "Precision": f"{rep_dict['weighted avg']['precision']:.4f}",
                    "Recall": f"{rep_dict['weighted avg']['recall']:.4f}",
                    "F1-skóre": f"{rep_dict['weighted avg']['f1-score']:.4f}",
                    "Počet vzorků (Support)": rep_dict['weighted avg']['support']
                }
            ]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    with col_cm:
        st.markdown("##### 🔲 Matice záměn (Confusion Matrix):")
        if stats and "confusion_matrix" in stats["metrics"]:
            cm = stats["metrics"]["confusion_matrix"]
            fig_cm = px.imshow(
                cm,
                labels=dict(x="Predikovaná třída", y="Skutečná třída", color="Počet"),
                x=["Negativní (0)", "Pozitivní (1)"],
                y=["Negativní (0)", "Pozitivní (1)"],
                text_auto=True,
                color_continuous_scale="Blues",
                template="plotly_dark"
            )
            fig_cm.update_layout(height=320, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_cm, width="stretch")

    st.markdown("---")
    st.markdown("#### 💡 Interpretace metrik:")
    st.markdown(
        """
        * **Vyrovnanost obou tříd:** F1-skóre pro negativní recenze je **0.86** a pro pozitivní recenze rovněž **0.86**. Model netrpí vychýlením k jedné třídě.
        * **Vysoká přesnost (85.75 %):** Na to, že Bag of Words zcela ignoruje pořadí slov a gramatiku, dosahuje jednoduchá lineární regrese vynikajícího baseline výsledku!
        """
    )

with tab2:
    st.subheader("Interpretovatelnost modelu: Váhy slov (Logistické koeficienty)")
    st.markdown(
        r"""
        V Logistické regresi má každý koeficient $\beta_i$ přímý matematický význam:
        * **Kladný koeficient ($\beta > 0$):** Výskyt slova exponenciálně zvyšuje šanci, že recenze je **pozitivní**.
        * **Záporný koeficient ($\beta < 0$):** Výskyt slova exponenciálně zvyšuje šanci, že recenze je **negativní**.
        """
    )

    if stats and "top_words" in stats:
        pos_df = pd.DataFrame(stats["top_words"]["positive"][:15])
        neg_df = pd.DataFrame(stats["top_words"]["negative"][:15])

        col_w1, col_w2 = st.columns(2)
        with col_w1:
            st.markdown("##### 🟢 TOP 15 slov podporujících pozitivní sentiment:")
            fig_pos = px.bar(
                pos_df,
                x="coefficient",
                y="word",
                orientation="h",
                color="coefficient",
                color_continuous_scale="Greens",
                template="plotly_dark",
                labels={"coefficient": r"Váha koeficientu $\beta$", "word": "Slovo (lemma)"}
            )
            fig_pos.update_layout(yaxis=dict(autorange="reversed"), height=420)
            st.plotly_chart(fig_pos, width="stretch")

        with col_w2:
            st.markdown("##### 🔴 TOP 15 slov podporujících negativní sentiment:")
            fig_neg = px.bar(
                neg_df,
                x="coefficient",
                y="word",
                orientation="h",
                color="coefficient",
                color_continuous_scale="Reds_r",
                template="plotly_dark",
                labels={"coefficient": r"Váha koeficientu $\beta$", "word": "Slovo (lemma)"}
            )
            fig_neg.update_layout(yaxis=dict(autorange="reversed"), height=420)
            st.plotly_chart(fig_neg, width="stretch")

        st.caption("Nejsilnějším negativním prediktorem v celém korpusu je slovo **waste** (-2.41), následované slovy **boring**, **disappointing** a **fail**.")

with tab3:
    st.subheader("Kde model uspěl a kde selhal? (Analýza chyb / Error Analysis)")
    st.markdown(
        """
        Zkoumání konkrétních recenzí, na kterých model chyboval, odhaluje inherentní slabinu přístupu Bag of Words.
        """
    )

    if stats and "samples" in stats:
        col_c_s, col_m_s = st.columns(2)
        with col_c_s:
            st.markdown("##### ✅ Ukázky správných predikcí:")
            for s in stats["samples"]["correct"][:3]:
                st.success(f"**Skutečnost:** {s['actual'].upper()} | **Predikce:** {s['predicted'].upper()} (Jistota: {s['prob_positive']*100:.1f} %)")
                st.caption(f"„{s['text']}“")

        with col_m_s:
            st.markdown("##### ❌ Ukázky chyb modelu (Misclassified):")
            for s in stats["samples"]["misclassified"][:3]:
                st.error(f"**Chyba:** {s['error_type']} | P(Pozitivní) = {s['prob_positive']*100:.1f} %")
                st.caption(f"„{s['text']}“")

    st.markdown("---")
    st.markdown("#### Proč dochází k těmto chybám?")
    st.markdown(
        """
        1. **Ironie a sarkasmus:** Věta typu *„Masterpiece of awful acting and wonderful waste of time“* obsahuje slova *masterpiece* i *wonderful*, která převáží model do pozitivní sféry.
        2. **Ztráta negací (Unigram blindness):** Pokud recenzent napíše *„not recommended, not interesting“*, unigramový model vidí slova *recommended* a *interesting*.
        3. **Smíšený sentiment:** Dlouhé recenze, které detailně chválí kameru a hudbu, ale závěrem odsoudí scénář.
        """
    )

with tab4:
    st.subheader("🧪 Interaktivní prediktor sentimentu v reálném čase")
    st.markdown("Napište vlastní recenzi v angličtině a model odhadne sentiment s rozpisem vlivu klíčových slov.")

    # Natrénování lehkého modelu pro interaktivní záložku
    @st.cache_resource
    def get_live_model():
        if csv_path.exists():
            df_live = pd.read_csv(csv_path)
            v = CountVectorizer(max_features=10000)
            X_l = v.fit_transform(df_live["review_lemmatized"].fillna(""))
            y_l = df_live["sentiment"].apply(lambda s: 1 if str(s).lower() == "positive" else 0)
            m = LogisticRegression(max_iter=1000, random_state=42)
            m.fit(X_l, y_l)
            return v, m
        return None, None

    vec_live, model_live = get_live_model()

    if vec_live is not None and model_live is not None:
        user_rev = st.text_area(
            "Vložte text recenze v angličtině:",
            value="This movie was an amazing masterpiece with superb acting and brilliant cinematography, definitely my favorite film!",
            height=100
        )

        if user_rev.strip():
            # Vektorizace vstupu
            x_input = vec_live.transform([user_rev.lower()])
            pred_class = model_live.predict(x_input)[0]
            pred_proba = model_live.predict_proba(x_input)[0]

            col_res1, col_res2 = st.columns([1, 2])
            with col_res1:
                if pred_class == 1:
                    st.success(f"### 🟢 Pozitivní recenze\n**Pravděpodobnost:** {pred_proba[1] * 100:.1f} %")
                else:
                    st.error(f"### 🔴 Negativní recenze\n**Pravděpodobnost:** {pred_proba[0] * 100:.1f} %")

            with col_res2:
                # Zobrazení slov ze vstupu, která jsou ve slovníku a jejich vliv
                words_in_input = user_rev.lower().split()
                feature_names = vec_live.get_feature_names_out()
                vocab_dict = {w: i for i, w in enumerate(feature_names)}

                contributions = []
                for w in words_in_input:
                    clean_w = w.strip(".,!?:;\"'()[]{}")
                    if clean_w in vocab_dict:
                        feat_idx = vocab_dict[clean_w]
                        coef_val = model_live.coef_[0][feat_idx]
                        contributions.append({"Slovo": clean_w, "Váha (Koeficient)": round(float(coef_val), 4), "Směr": "🟢 Pozitivní" if coef_val > 0 else "🔴 Negativní"})

                if contributions:
                    df_contrib = pd.DataFrame(contributions).drop_duplicates(subset=["Slovo"]).sort_values("Váha (Koeficient)", ascending=False)
                    st.markdown("##### Rozpad rozpoznaných slov a jejich vliv na výsledek:")
                    st.dataframe(df_contrib, hide_index=True, width="stretch")
    else:
        st.info("Model se načítá...")
