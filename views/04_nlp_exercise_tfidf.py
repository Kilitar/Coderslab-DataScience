"""
Den 4: NLP – Cvičení 4: TF-IDF & Srovnání modelů (Logistic Regression vs. SVM)
==============================================================================
Model: TfidfVectorizer(max_features=10000) + LogisticRegression & LinearSVC
Data: data/imdb_reviews_preprocessed.csv
Precomputed: 04_NLP/data/nlp_exercise_tfidf_precomputed.json
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

try:
    from czech_nlp import predict_czech_sentiment
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent / "04_NLP"))
    from czech_nlp import predict_czech_sentiment

st.title("🎯 Cvičení 4: TF-IDF & Srovnání modelů (LR vs. SVM)")
st.caption(
    "Klasifikace sentimentu pomocí vážení TF-IDF (Term Frequency – Inverse Document Frequency). "
    "Přímé porovnání Logistické regrese a Support Vector Machine (LinearSVC) a bilingvální prediktor v reálném čase."
)

base_dir = Path(__file__).resolve().parent.parent
json_path = base_dir / "04_NLP" / "data" / "nlp_exercise_tfidf_precomputed.json"
csv_path = base_dir / "data" / "imdb_reviews_preprocessed.csv"
if not csv_path.exists():
    csv_path = base_dir / "data" / "imdb_reviews_lemmatized.csv"


@st.cache_data
def load_tfidf_stats():
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


stats = load_tfidf_stats()

# Defaultní hodnoty
acc_lr = stats["logistic_regression"]["accuracy"] if stats else 0.8730
acc_svm = stats["svm"]["accuracy"] if stats else 0.8660
f1_lr = stats["logistic_regression"]["f1_weighted"] if stats else 0.8729
f1_svm = stats["svm"]["f1_weighted"] if stats else 0.8660

# Horní KPI karty
k1, k2, k3, k4 = st.columns(4)
k1.metric("Logistická regrese (TF-IDF)", f"{acc_lr * 100:.2f} %", delta="+1.55 % vs. BoW", delta_color="normal")
k2.metric("Support Vector Machine (LinearSVC)", f"{acc_svm * 100:.2f} %", delta="Maximum Margin", delta_color="normal")
k3.metric("F1-skóre (LR / SVM)", f"{f1_lr:.3f} / {f1_svm:.3f}", delta="Vyvážený sentiment")
k4.metric("Dimenze slovníku", "10 000 termínů", delta="TF-IDF L2 norm", delta_color="off")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 1. Výsledky & Srovnání LR vs. SVM",
    "⚖️ 2. BoW vs. TF-IDF (Proč TF-IDF vyhrává)",
    "🧠 3. Interpretace vah slov (LR vs. SVM)",
    "🧪 4. Živý prediktor sentimentu (CZ / EN)"
])

with tab1:
    st.subheader("Srovnání kvality klasifikace: Logistická regrese vs. LinearSVC")
    st.markdown(
        """
        Oba modely byly vyhodnoceny na identické testovací sadě 2 000 recenzí (1 000 pozitivních a 1 000 negativních) 
        s TF-IDF reprezentací textu.
        """
    )

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("##### 📈 Souhrnná tabulka metrik:")
        df_comp_table = pd.DataFrame([
            {
                "Model": "1. Logistická regrese (TF-IDF)",
                "Přesnost (Accuracy)": f"{acc_lr * 100:.2f} %",
                "Precision": f"{stats['logistic_regression']['precision_weighted']:.4f}" if stats else "0.873",
                "Recall": f"{stats['logistic_regression']['recall_weighted']:.4f}" if stats else "0.873",
                "F1-skóre": f"{f1_lr:.4f}",
                "Čas trénování": f"{stats['logistic_regression']['train_time_s']} s" if stats else "0.10 s"
            },
            {
                "Model": "2. LinearSVC (SVM)",
                "Přesnost (Accuracy)": f"{acc_svm * 100:.2f} %",
                "Precision": f"{stats['svm']['precision_weighted']:.4f}" if stats else "0.866",
                "Recall": f"{stats['svm']['recall_weighted']:.4f}" if stats else "0.866",
                "F1-skóre": f"{f1_svm:.4f}",
                "Čas trénování": f"{stats['svm']['train_time_s']} s" if stats else "0.04 s"
            }
        ])
        st.dataframe(df_comp_table, hide_index=True, width="stretch")

    with col_t2:
        st.markdown("##### 📊 Srovnání přesnosti (Bar Chart):")
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name="Accuracy",
            x=["Bag of Words (LR)", "TF-IDF (SVM)", "TF-IDF (LR)"],
            y=[85.75, acc_svm * 100, acc_lr * 100],
            text=["85.75 %", f"{acc_svm * 100:.2f} %", f"{acc_lr * 100:.2f} %"],
            textposition="outside",
            marker_color=["#F5A623", "#4A90E2", "#50E3C2"]
        ))
        fig_bar.update_layout(
            yaxis=dict(range=[80, 92]),
            template="plotly_dark",
            height=280,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_bar, width="stretch")

    st.markdown("---")
    st.markdown("#### Matice záměn (Confusion Matrices):")
    col_cm1, col_cm2 = st.columns(2)
    with col_cm1:
        st.markdown("##### 🔲 Logistická regrese:")
        if stats and "confusion_matrix" in stats["logistic_regression"]:
            cm_lr = stats["logistic_regression"]["confusion_matrix"]
            fig_cm_lr = px.imshow(
                cm_lr,
                labels=dict(x="Predikce LR", y="Skutečnost", color="Počet"),
                x=["Negativní (0)", "Pozitivní (1)"],
                y=["Negativní (0)", "Pozitivní (1)"],
                text_auto=True,
                color_continuous_scale="Blues",
                template="plotly_dark"
            )
            fig_cm_lr.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_cm_lr, width="stretch")

    with col_cm2:
        st.markdown("##### 🔲 Support Vector Machine (LinearSVC):")
        if stats and "confusion_matrix" in stats["svm"]:
            cm_svm = stats["svm"]["confusion_matrix"]
            fig_cm_svm = px.imshow(
                cm_svm,
                labels=dict(x="Predikce SVM", y="Skutečnost", color="Počet"),
                x=["Negativní (0)", "Pozitivní (1)"],
                y=["Negativní (0)", "Pozitivní (1)"],
                text_auto=True,
                color_continuous_scale="Purples",
                template="plotly_dark"
            )
            fig_cm_svm.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_cm_svm, width="stretch")

with tab2:
    st.subheader("Proč TF-IDF dosahuje vyšší přesnosti než Bag of Words?")
    st.markdown(
        """
        Zatímco Bag of Words v minulém cvičení dosáhl přesnosti **85.75 %**, 
        stejná Logistická regrese na TF-IDF vektorech stoupla na **87.30 %** (nárůst o **1.55 procentního bodu**). 
        Proč je TF-IDF efektivnější?
        """
    )

    c_r1, c_r2, c_r3 = st.columns(3)
    with c_r1:
        st.info("1. Potlačení frekventovaného šumu")
        st.markdown(
            """
            * V BoW mají slova jako *movie*, *film*, *scene* obrovské hodnoty (často 10+ v jedné recenzi), přestože nenesou žádný sentiment.
            * **IDF váha** tato obecná filmová slova dramaticky utlumí, takže nezastiňují skutečné hodnotící přívlastky.
            """
        )

    with c_r2:
        st.success("2. L2 Normalizace délky dokumentu")
        st.markdown(
            r"""
            * Dlouhá recenze (500 slov) má v BoW přirozeně 5x větší hodnoty vektorů než krátká recenze (100 slov).
            * TF-IDF standardně normalizuje euklidovskou délku vektoru na $\|\mathbf{v}\|_2 = 1$. 
            * Dlouhé i krátké recenze tak mají **stejnou celkovou energii** a model není zkreslen délkou textu.
            """
        )

    with c_r3:
        st.warning("3. Vyšší separabilita prostoru")
        st.markdown(
            """
            * V geometrickém prostoru vytvořeném TF-IDF leží pozitivní a negativní recenze v kompaktnějších shlucích.
            * Lineární dělící nadrovina (ať už v Logistické regresi nebo SVM) dokáže najít zřetelnější a stabilnější rozhodovací hranici.
            """
        )

with tab3:
    st.subheader("Interpretace vah: Která slova rozhodují v LR a SVM?")
    st.markdown(
        """
        Srovnání klíčových slovních vah ukazuje, že oba algoritmy se shodují na nejsilnějších polaritních termínech:
        """
    )

    if stats and "top_features" in stats:
        lr_pos = pd.DataFrame(stats["top_features"]["lr_positive"][:12])
        svm_pos = pd.DataFrame(stats["top_features"]["svm_positive"][:12])
        lr_neg = pd.DataFrame(stats["top_features"]["lr_negative"][:12])
        svm_neg = pd.DataFrame(stats["top_features"]["svm_negative"][:12])

        col_pos_comp, col_neg_comp = st.columns(2)
        with col_pos_comp:
            st.markdown("##### 🟢 TOP pozitivní slova (LR vs. SVM):")
            df_pos_table = pd.DataFrame({
                "Pořadí": list(range(1, 13)),
                "Slovo (LR)": lr_pos["word"],
                "Váha LR": lr_pos["weight"],
                "Slovo (SVM)": svm_pos["word"],
                "Váha SVM": svm_pos["weight"]
            })
            st.dataframe(df_pos_table, hide_index=True, width="stretch")

        with col_neg_comp:
            st.markdown("##### 🔴 TOP negativní slova (LR vs. SVM):")
            df_neg_table = pd.DataFrame({
                "Pořadí": list(range(1, 13)),
                "Slovo (LR)": lr_neg["word"],
                "Váha LR": lr_neg["weight"],
                "Slovo (SVM)": svm_neg["word"],
                "Váha SVM": svm_neg["weight"]
            })
            st.dataframe(df_neg_table, hide_index=True, width="stretch")

with tab4:
    st.subheader("🧪 Interaktivní prediktor sentimentu (CZ / EN standard)")
    st.markdown(
        "Otestujte klasifikaci sentimentu na libovolném textu. V souladu s naším projektovým standardem "
        "můžete přepínat mezi **českým jazykem** a **anglickým IMDb modelem** s možností volby mezi **Logistickou regresí** a **SVM**."
    )

    lang_choice = st.radio(
        "Zvolte jazykový režim pro analýzu sentimentu:",
        ["🇨🇿 Čeština (Nativní model – Negace & Morfologie)", "🇬🇧 Angličtina (TF-IDF model – LR & SVM)"],
        horizontal=True
    )

    sample_texts_cs = {
        "Ukázka 1: Pozitivní chvála": "Tento film byl naprosto skvělý, herci předvedli úžasný výkon a hudba byla fantastická!",
        "Ukázka 2: Negativní kritika": "Naprostá katastrofa a hrozná nuda, scénář je trapný a rozhodně to nedoporučuji.",
        "Ukázka 3: Vliv české negace": "Tento film nebyl vůbec dobrý a herci nepředvedli žádný výkon.",
        "Ukázka 4: Obrácený zápor": "Nebylo to vůbec špatné, příjemně mě to potěšilo a skvěle jsem se bavil.",
        "Ukázka 5: Vlastní český text": ""
    }

    sample_texts_en = {
        "Ukázka 1: Pozitivní mistrovské dílo": "This movie was an amazing masterpiece with superb acting and brilliant cinematography, definitely my favorite film!",
        "Ukázka 2: Negativní zklamání": "Total waste of time, boring plot, terrible acting and disappointing ending.",
        "Ukázka 3: Jemná ironie & negace": "The plot twists were predictable and did not offer any surprises.",
        "Ukázka 4: Vlastní anglický text": ""
    }

    if "Čeština" in lang_choice:
        preset_cz = st.selectbox("Vyberte ukázkovou českou recenzi:", list(sample_texts_cs.keys()))
        default_cz = sample_texts_cs[preset_cz] if preset_cz != "Ukázka 5: Vlastní český text" else "Napište sem vlastní českou recenzi..."
        user_text_cz = st.text_area("Vstupní recenze v češtině:", value=default_cz, height=100)

        if user_text_cz.strip():
            res_cz = predict_czech_sentiment(user_text_cz)
            pred_class = res_cz["predicted_class"]
            p_pos = res_cz["prob_positive"]
            p_neg = res_cz["prob_negative"]

            col_res1, col_res2 = st.columns([1, 2])
            with col_res1:
                if pred_class == 1:
                    st.success(f"### 🟢 Pozitivní recenze\n**Jistota:** {p_pos * 100:.1f} %")
                else:
                    st.error(f"### 🔴 Negativní recenze\n**Jistota:** {p_neg * 100:.1f} %")

                st.progress(float(p_pos), text=f"P(Pozitivní): {p_pos*100:.1f}% | P(Negativní): {p_neg*100:.1f}%")

            with col_res2:
                contribs = res_cz["contributions"]
                if contribs:
                    st.markdown("##### Rozpad rozpoznaných českých slov & bigramů a jejich vliv:")
                    st.dataframe(pd.DataFrame(contribs), hide_index=True, width="stretch")
                else:
                    st.info("Ve vstupu nebyla nalezena žádná polaritní slova ze slovníku.")

    else:
        # Anglický režim: Možnost volby modelu LR vs SVM
        @st.cache_resource
        def get_live_tfidf_models():
            if csv_path.exists():
                df_l = pd.read_csv(csv_path)
                col_name = "review_tokens_lemmatized" if "review_tokens_lemmatized" in df_l.columns else "review_lemmatized"
                v = TfidfVectorizer(max_features=10000)
                X_mat = v.fit_transform(df_l[col_name].fillna(""))
                y_arr = df_l["sentiment"].apply(lambda s: 1 if str(s).lower() == "positive" else 0)
                m_lr = LogisticRegression(max_iter=1000, random_state=42).fit(X_mat, y_arr)
                m_svm = LinearSVC(random_state=42).fit(X_mat, y_arr)
                return v, m_lr, m_svm
            return None, None, None

        vec_en, lr_live, svm_live = get_live_tfidf_models()

        if vec_en is not None and lr_live is not None and svm_live is not None:
            col_m_choice, col_preset = st.columns([1, 2])
            with col_m_choice:
                chosen_algo = st.selectbox("Algoritmus klasifikace:", ["Logistická regrese (s pravděpodobností)", "LinearSVC (Support Vector Machine)"])
            with col_preset:
                preset_en = st.selectbox("Vyberte ukázkovou anglickou recenzi:", list(sample_texts_en.keys()))

            default_en = sample_texts_en[preset_en] if preset_en != "Ukázka 4: Vlastní anglický text" else "Type your own English review here..."
            user_text_en = st.text_area("Vstupní recenze v angličtině:", value=default_en, height=100)

            if user_text_en.strip():
                x_in = vec_en.transform([user_text_en.lower()])
                
                # Predikce obou modelů pro srovnání
                p_lr = lr_live.predict(x_in)[0]
                proba_lr = lr_live.predict_proba(x_in)[0]
                p_svm = svm_live.predict(x_in)[0]
                decision_svm = svm_live.decision_function(x_in)[0]

                col_res1, col_res2 = st.columns([1, 2])
                with col_res1:
                    if "Logistická" in chosen_algo:
                        if p_lr == 1:
                            st.success(f"### 🟢 Pozitivní recenze (LR)\n**Pravděpodobnost:** {proba_lr[1] * 100:.1f} %")
                        else:
                            st.error(f"### 🔴 Negativní recenze (LR)\n**Pravděpodobnost:** {proba_lr[0] * 100:.1f} %")
                        st.progress(float(proba_lr[1]), text=f"P(Pozitivní): {proba_lr[1]*100:.1f}% | P(Negativní): {proba_lr[0]*100:.1f}%")
                    else:
                        if p_svm == 1:
                            st.success(f"### 🟢 Pozitivní recenze (SVM)\n**Vzdálenost od nadroviny:** +{decision_svm:.2f}")
                        else:
                            st.error(f"### 🔴 Negativní recenze (SVM)\n**Vzdálenost od nadroviny:** {decision_svm:.2f}")

                    st.caption(f"Shoda modelů: {'✅ Shodují se' if p_lr == p_svm else '⚠️ Rozcházejí se v predikci'}")

                with col_res2:
                    words_in_input = user_text_en.lower().split()
                    feat_names = vec_en.get_feature_names_out()
                    vocab_map = {w: i for i, w in enumerate(feat_names)}

                    active_coefs = lr_live.coef_[0] if "Logistická" in chosen_algo else svm_live.coef_[0]

                    contributions = []
                    for w in words_in_input:
                        clean_w = w.strip(".,!?:;\"'()[]{}")
                        if clean_w in vocab_map:
                            idx = vocab_map[clean_w]
                            weight = active_coefs[idx]
                            contributions.append({
                                "Slovo": clean_w,
                                "TF-IDF Váha": round(float(weight), 4),
                                "Směr": "🟢 Pozitivní" if weight > 0 else "🔴 Negativní"
                            })

                    if contributions:
                        df_contrib = pd.DataFrame(contributions).drop_duplicates(subset=["Slovo"]).sort_values("TF-IDF Váha", ascending=False)
                        st.markdown("##### Rozpad rozpoznaných anglických slov a jejich váhy:")
                        st.dataframe(df_contrib, hide_index=True, width="stretch")
                    else:
                        st.info("Ve vstupu nebyla nalezena žádná slova ze slovníku modelu.")
        else:
            st.info("Modely se načítají...")
