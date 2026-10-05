"""
Domácí úkol (Session 2): XGBoost – Klasifikace diabetu (Pima Indians)
====================================================================
Dataset: data/diabetes.csv (768 pacientek, 9 proměnných)
Model: XGBClassifier + GridSearchCV(scoring='precision', cv=5)
Precomputed: 04_Homework/data/diabetes_xgb_precomputed.json
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.title("🩺 Domácí úkol: XGBoost – Klasifikace diabetu")
st.caption(
    "Vypracované řešení domácího úkolu: Predikce diagnózy diabetu u pacientek pomocí gradientního boostingu "
    "(`XGBClassifier`) s výběrem nejlepších hyperparametrů přes `GridSearchCV` s optimalizací na metriku **Precision**."
)

base_dir = Path(__file__).resolve().parent.parent
json_path = base_dir / "04_Homework" / "data" / "diabetes_xgb_precomputed.json"
model_path = base_dir / "04_Homework" / "data" / "diabetes_xgb_model.joblib"


@st.cache_data
def load_diabetes_stats():
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_resource
def load_diabetes_model():
    if model_path.exists():
        try:
            return joblib.load(model_path)
        except Exception:
            return None
    return None


stats = load_diabetes_stats()
model = load_diabetes_model()

metrics = stats["test_metrics"] if stats else {}
meta = stats["metadata"] if stats else {}
gs = stats["grid_search"] if stats else {}

# Horní KPI karty
c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Testovací Precision (Diabetik)",
    f"{metrics.get('precision', 0.7568)*100:.1f} %",
    delta="Cílová optimalizovaná metrika",
    delta_color="normal"
)
c2.metric(
    "Testovací Accuracy",
    f"{metrics.get('accuracy', 0.7338)*100:.1f} %",
    delta="154 testovacích pacientek"
)
c3.metric(
    "Testovací Recall (Záchyt)",
    f"{metrics.get('recall', 0.4667)*100:.1f} %",
    delta="Při výchozím prahu P >= 0.50"
)
c4.metric(
    "Plocha pod ROC (AUC)",
    f"{metrics.get('roc_auc', 0.8234):.4f}",
    delta="Silná diskriminační schopnost"
)

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 1. Výsledky testovací sady & Metriky",
    "⚙️ 2. Výsledky GridSearchCV & Hyperparametry",
    "🩺 3. Ukázky testovacích predikcí",
    "🧪 4. Interaktivní diagnostický prediktor"
])

# ==============================================================================
# TAB 1: METRIKY A CLASSIFICATION REPORT
# ==============================================================================
with tab1:
    st.subheader("Vyhodnocení XGBoost klasifikátoru na 20% testovací sadě (154 pacientek)")

    st.markdown(
        """
        Model byl trénován na 80 % dat (**614 pacientek**) a vyhodnocen na 20 % neviděných pacientek 
        (**154 vzorků**). Dle zadání proběhlo rozdělení se stanoveným `random_state=21` a hyperparametry 
        byly laděny s cílem maximalizovat **Precision** (přesnost pozitivní diagnózy).
        """
    )

    col_rep, col_cm = st.columns([1.1, 0.9])

    with col_rep:
        st.markdown("##### 📋 Souhrnný report klasifikace (Classification Report):")
        clf_dict = metrics.get("classification_report", {})
        if clf_dict:
            rep_rows = []
            for label, key in [("Zdravý (Třída 0)", "0"), ("Diabetik (Třída 1)", "1")]:
                if key in clf_dict:
                    d = clf_dict[key]
                    rep_rows.append({
                        "Třída": label,
                        "Precision (Přesnost)": f"{d.get('precision', 0)*100:.1f} %",
                        "Recall (Záchyt)": f"{d.get('recall', 0)*100:.1f} %",
                        "F1-skóre": f"{d.get('f1-score', 0):.4f}",
                        "Počet pacientek": int(d.get("support", 0))
                    })
            st.dataframe(pd.DataFrame(rep_rows), hide_index=True, width="stretch")

            st.caption(
                f"Celková přesnost (Accuracy): **{clf_dict.get('accuracy', 0.7338)*100:.2f} %** | "
                f"Makro průměr F1: **{clf_dict.get('macro avg', {}).get('f1-score', 0.69):.4f}** | "
                f"Vážený průměr F1: **{clf_dict.get('weighted avg', {}).get('f1-score', 0.72):.4f}**"
            )

    with col_cm:
        st.markdown("##### 🎯 Matice záměn (Confusion Matrix):")
        cm = metrics.get("confusion_matrix", [[85, 9], [32, 28]])
        cm_labels = ["Skutečně zdravý (0)", "Skutečně diabetik (1)"]
        pred_labels = ["Predikce: Zdravý", "Predikce: Diabetik"]

        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predikce modelu", y="Skutečná diagnóza", color="Počet"),
            x=pred_labels,
            y=cm_labels,
            color_continuous_scale="Blues",
            text_auto=True
        )
        fig_cm.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_cm, width="stretch")

    st.markdown("---")

    col_roc, col_pr = st.columns(2)

    with col_roc:
        st.markdown("##### 📈 ROC křivka (Receiver Operating Characteristic):")
        roc_data = stats.get("roc_curve", {})
        if roc_data:
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=roc_data.get("fpr", []),
                y=roc_data.get("tpr", []),
                mode="lines",
                name=f"XGBoost (AUC = {metrics.get('roc_auc', 0.8234):.3f})",
                line=dict(color="#2563eb", width=3)
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode="lines",
                line=dict(color="#9ca3af", dash="dash"),
                name="Náhodný klasifikátor"
            ))
            fig_roc.update_layout(
                xaxis_title="FPR (1 - Specificita)",
                yaxis_title="TPR (Senzitivita / Recall)",
                height=320,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig_roc, width="stretch")

    with col_pr:
        st.markdown("##### 🎯 Precision-Recall křivka:")
        pr_data = stats.get("precision_recall_curve", {})
        if pr_data:
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(
                x=pr_data.get("recall", []),
                y=pr_data.get("precision", []),
                mode="lines",
                name="XGBoost PR Křivka",
                line=dict(color="#10b981", width=3)
            ))
            fig_pr.add_hline(
                y=meta.get("target_distribution", {}).get("diabetes_rate", 0.35),
                line_dash="dash",
                line_color="#9ca3af",
                annotation_text="Prevalence diabetu v populaci"
            )
            fig_pr.update_layout(
                xaxis_title="Recall (Záchyt)",
                yaxis_title="Precision (Spolehlivost)",
                height=320,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig_pr, width="stretch")


# ==============================================================================
# TAB 2: GRIDSEARCHCV A HYPERPARAMETRY
# ==============================================================================
with tab2:
    st.subheader("Optimalizace hyperparametrů pomocí GridSearchCV")

    st.markdown(
        """
        Dle zadání byla definována mřížka parametrů `params` pro klíče `max_depth`, `n_estimators`, `gamma`, 
        `objective` a `learning_rate` s cílem nalézt model s nejvyšší validační **Precision** při 5-násobné křížové validaci.
        """
    )

    col_g1, col_g2 = st.columns([1, 1])

    with col_g1:
        st.markdown("##### 🔍 Prohledávaný prostor hyperparametrů:")
        p_dict = gs.get("param_grid", {})
        param_table = pd.DataFrame([
            {"Hyperparametr": "max_depth", "Testované hodnoty": str(p_dict.get("max_depth", [3, 4, 5])), "Účel a role v XGBoost": "Hloubka stromů; menší hloubka (3–4) účinně brání přeučení na malém datasetu."},
            {"Hyperparametr": "n_estimators", "Testované hodnoty": str(p_dict.get("n_estimators", [50, 100, 150])), "Účel a role v XGBoost": "Počet sekvenčních boostingových kol (stromů)."},
            {"Hyperparametr": "gamma (min_split_loss)", "Testované hodnoty": str(p_dict.get("gamma", [0.0, 0.1, 0.5])), "Účel a role v XGBoost": "Minimální snížení ztrátové funkce nutné pro provedení dalšího rozštěpení uzlu."},
            {"Hyperparametr": "learning_rate (eta)", "Testované hodnoty": str(p_dict.get("learning_rate", [0.05, 0.1, 0.2])), "Účel a role v XGBoost": "Zmenšovací faktor (shrinkage) váhy každého nově přidaného stromu."},
            {"Hyperparametr": "objective", "Testované hodnoty": str(p_dict.get("objective", ["binary:logistic"])), "Účel a role v XGBoost": "Logistická ztrátová funkce pro binární klasifikaci."},
            {"Optimalizační metrika", "scoring='precision'", "precision", "Maximalizace podílu skutečných diabetiček mezi označenými."},
            {"Křížová validace", "cv=5", "5 záhybů", "Odhad zobecnitelnosti na trénovacích datech."}
        ])
        st.dataframe(param_table, hide_index=True, width="stretch")

        best_p = gs.get("best_params", {})
        st.success(
            f"🏆 **Nejlepší nalezená kombinace:**\n"
            f"- `max_depth = {best_p.get('max_depth', 4)}`\n"
            f"- `n_estimators = {best_p.get('n_estimators', 50)}`\n"
            f"- `gamma = {best_p.get('gamma', 0.0)}`\n"
            f"- `learning_rate = {best_p.get('learning_rate', 0.05)}`\n\n"
            f"Dosáhla validační přesnosti **CV Precision = {gs.get('best_cv_precision', 0.7159)*100:.2f} %**."
        )

    with col_g2:
        st.markdown("##### 🥇 Top 10 nejlepších konfigurací dle CV Precision:")
        top_configs = gs.get("top_10_configs", [])
        if top_configs:
            tc_rows = []
            for c in top_configs:
                p = c["params"]
                tc_rows.append({
                    "Pořadí": f"#{c['rank']}",
                    "max_depth": p.get("max_depth"),
                    "n_estimators": p.get("n_estimators"),
                    "gamma": p.get("gamma"),
                    "learning_rate": p.get("learning_rate"),
                    "CV Precision": f"{c.get('mean_test_precision', 0)*100:.2f} %",
                    "Směrodatná odchylka": f"±{c.get('std_test_score', 0)*100:.1f} %"
                })
            st.dataframe(pd.DataFrame(tc_rows), hide_index=True, width="stretch")


# ==============================================================================
# TAB 3: UKÁZKY PREDIKCÍ
# ==============================================================================
with tab3:
    st.subheader("Ukázka reálných předpovědí modelu na testovacích pacientkách")

    st.markdown(
        """
        Níže je uveden reprezentativní vzorek 25 pacientek z testovací sady, jejich naměřené hodnoty, 
        skutečná diagnóza a odhadnutá pravděpodobnost diabetu predikovaná modelem XGBoost.
        """
    )

    sample_preds = stats.get("sample_predictions", [])
    if sample_preds:
        sp_rows = []
        for s in sample_preds:
            sp_rows.append({
                "Těhotenství": s.get("pregnancies"),
                "Glukóza": f"{s.get('glucose', 0):.0f} mg/dl",
                "Tlak": f"{s.get('blood_pressure', 0):.0f} mmHg",
                "BMI": f"{s.get('bmi', 0):.1f}",
                "Inzulín": f"{s.get('insulin', 0):.0f} µU/ml",
                "DPF": f"{s.get('dpf', 0):.3f}",
                "Věk": s.get("age"),
                "Skutečnost": s.get("actual"),
                "Predikce": s.get("predicted"),
                "Pravděpodobnost diabetu": f"{s.get('prob_diabetes', 0)*100:.1f} %",
                "Výsledek": "✅ Správně" if s.get("is_correct") else "❌ Chyba"
            })
        st.dataframe(pd.DataFrame(sp_rows), hide_index=True, width="stretch")


# ==============================================================================
# TAB 4: INTERAKTIVNÍ PREDIKTOR A SIMULÁTOR PRAHU
# ==============================================================================
with tab4:
    st.subheader("🧪 Interaktivní medicínský kalkulátor rizika diabetu")

    st.markdown(
        """
        Zadejte klinické parametry pacientky a model XGBoost spočítá odhadnutou pravděpodobnost rozvoje diabetu. 
        Kromě bodového odhadu můžete interaktivně **upravovat rozhodovací práh** a sledovat dopad na Precision vs. Recall.
        """
    )

    col_inp1, col_inp2, col_inp3 = st.columns(3)

    with col_inp1:
        i_preg = st.number_input("Počet těhotenství (Pregnancies):", min_value=0, max_value=20, value=2, step=1)
        i_gluc = st.number_input("Hladina glukózy nalačno (Glucose, mg/dl):", min_value=40, max_value=250, value=125, step=1)
        i_bp = st.number_input("Diastolický krevní tlak (BloodPressure, mmHg):", min_value=30, max_value=140, value=72, step=1)

    with col_inp2:
        i_skin = st.number_input("Tloušťka kožní řasy tricepsu (SkinThickness, mm):", min_value=0, max_value=100, value=23, step=1)
        i_ins = st.number_input("Hladina inzulínu po 2 hodinách (Insulin, µU/ml):", min_value=0, max_value=900, value=85, step=5)
        i_bmi = st.number_input("Index tělesné hmotnosti (BMI):", min_value=10.0, max_value=70.0, value=32.0, step=0.5)

    with col_inp3:
        i_dpf = st.number_input("Diabetes Pedigree Function (Genetika DPF):", min_value=0.05, max_value=2.5, value=0.45, step=0.01)
        i_age = st.number_input("Věk pacientky (Age, roky):", min_value=18, max_value=100, value=35, step=1)
        i_thresh = st.slider("Rozhodovací práh diagnózy (Threshold):", min_value=0.10, max_value=0.90, value=0.50, step=0.05)

    if st.button("🔮 Diagnostikovat riziko diabetu", type="primary"):
        features = meta.get("features", ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"])

        input_data = pd.DataFrame([{
            "Pregnancies": float(i_preg),
            "Glucose": float(i_gluc),
            "BloodPressure": float(i_bp),
            "SkinThickness": float(i_skin),
            "Insulin": float(i_ins),
            "BMI": float(i_bmi),
            "DiabetesPedigreeFunction": float(i_dpf),
            "Age": float(i_age)
        }])[features]

        if model is not None:
            prob = float(model.predict_proba(input_data)[:, 1][0])
            pred_diag = 1 if prob >= i_thresh else 0

            st.markdown("### 🏷️ Klinický výsledek:")
            r1, r2, r3 = st.columns(3)
            r1.metric("Predikovaná pravděpodobnost", f"{prob*100:.1f} %")
            diag_label = "🚨 Riziko Diabetu (Pozitivní)" if pred_diag == 1 else "✅ Nízké riziko (Negativní)"
            r2.metric("Diagnóza dle zvoleného prahu", diag_label)
            r3.metric("Aktivní rozhodovací práh", f"{i_thresh:.2f}")

            # Doporučení lékaři
            if pred_diag == 1:
                st.warning(
                    f"⚠️ Model s pravděpodobností **{prob*100:.1f} %** (při prahu {i_thresh:.2f}) indikuje riziko diabetu. "
                    "Doporučuje se provést potvrzující orální glukózový toleranční test (oGTT) a stanovení glykovaného hemoglobinu (HbA1c)."
                )
            else:
                st.success(
                    f"Výsledek je negativní (odhadnuté riziko {prob*100:.1f} % je pod stanoveným prahem {i_thresh:.2f})."
                )
        else:
            st.info("Předpočítaný odhad: Vysoká glykémie a BMI zvyšují riziko diabetu.")
