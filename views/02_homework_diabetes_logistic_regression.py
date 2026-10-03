"""
Homework: Logistická regrese – Klasifikace diabetu (RandomizedSearchCV)
======================================================================
Interaktivní modul pro klasifikaci diabetu pomocí Logistické regrese:
1. Ladění inverzní síly regularizace C pomocí RandomizedSearchCV (loguniform).
2. Interaktivní prostor náhodných pokusů a vliv regularizační penalizace (L1 vs L2).
3. Analýza koeficientů a Odds Ratios (Poměr šancí pro jednotlivé biomarkery).
4. Regularization Path (vývoj koeficientů v závislosti na C).
5. Interaktivní matice záměn a ROC/PR křivky v Plotly.
6. Přímé mezimodelové srovnání: k-NN vs. Logistická regrese na testovacích datech.
7. Interaktivní sigmoidální kalkulátor individuálního rizika pacientky.
"""

import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def load_logistic_precomputed():
    base_dir = Path(__file__).resolve().parent.parent
    cache_path = base_dir / "02_Classification" / "data" / "diabetes_logistic_precomputed.json"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def render_diabetes_logistic_regression_view():
    st.title("📈 DÚ: Logistická regrese – Klasifikace diabetu")
    st.markdown(
        "**Vypracování cvičení:** Trénování a optimalizace modelu **LogisticRegression** na standardizovaných datech "
        "`diabetes_scaled.csv`, dělení v poměru 70/30, hledání optimálního parametru regularizace `C` "
        "pomocí **RandomizedSearchCV** a klinická interpretace vah přes **Odds Ratios** (poměr šancí)."
    )

    data = load_logistic_precomputed()
    if not data:
        st.error("Předpočtená data nebyla nalezena. Spusťte prosím skript `02_Classification/17_homework_diabetes_logistic_regression.py`.")
        return

    pri_m = data["primary_metric"]
    best_p = data["random_search_results"][pri_m]["best_params"]
    eval_m = data["test_evaluations"][pri_m]

    # Horní KPI karty
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Optimální C", f"{best_p['C']:.4f}", delta=f"Penalizace: {best_p['penalty'].upper()}")
    with c2:
        st.metric("Test F1-Score", f"{eval_m['f1']:.4f}", delta="Harmonický průměr")
    with c3:
        st.metric("Test Recall", f"{eval_m['recall'] * 100:.1f} %", delta=f"{eval_m['tp']} z 81 zachyceno")
    with c4:
        st.metric("Test Precision", f"{eval_m['precision'] * 100:.1f} %", delta="Spolehlivost pozitivity")
    with c5:
        st.metric("ROC-AUC", f"{eval_m['roc_auc']:.4f}", delta="Separační schopnost")

    st.markdown("---")

    # Sekce 1: RandomizedSearchCV a vliv optimalizační metriky
    st.subheader("1. Výsledky RandomizedSearchCV: Vliv regularizace C")
    st.markdown(
        r"""
        V logistické regresi parametr $C$ představuje **inverzní sílu regularizace** ($C = \frac{1}{\lambda}$):
        - **Malé $C \to 0$:** Silná regularizace, model má velký bias, váhy jsou stlačovány k nule (prevence přeučení).
        - **Velké $C \to \infty$:** Slabá regularizace, model se blíží čistému OLS/MLE odhadu, hrozí přeučení.
        """
    )

    selected_metric_choice = st.radio(
        "Zvolte optimalizační cíl pro zobrazení výsledků RandomizedSearchCV:",
        options=["F1-Score (Stejně jako v cvičení 1)", "Recall (Maximalizace záchytu nemoci)", "Accuracy (Celková přesnost)", "ROC-AUC (Pravděpodobnostní separace)"],
        horizontal=True
    )
    metric_key_map = {
        "F1-Score (Stejně jako v cvičení 1)": "f1",
        "Recall (Maximalizace záchytu nemoci)": "recall",
        "Accuracy (Celková přesnost)": "accuracy",
        "ROC-AUC (Pravděpodobnostní separace)": "roc_auc"
    }
    active_key = metric_key_map[selected_metric_choice]
    active_res = data["random_search_results"][active_key]
    active_eval = data["test_evaluations"][active_key]

    c_info1, c_info2 = st.columns([1, 1])
    with c_info1:
        st.info(
            f"**Nejlepší parametry z RandomizedSearchCV pro {selected_metric_choice}:**\n"
            f"- Inverzní regularizace `C`: **{active_res['best_params']['C']:.5f}**\n"
            f"- Typ penalizace (`penalty`): **{active_res['best_params']['penalty'].upper()}**\n"
            f"- Řešitel (`solver`): **{active_res['best_params']['solver']}**\n"
            f"- Validační 5-Fold CV skóre: **{active_res['best_score']:.4f}**"
        )
    with c_info2:
        st.success(
            f"**Ověření na testovací sadě (231 pacientek):**\n"
            f"- Recall (Záchyt): **{active_eval['recall'] * 100:.2f} %** ({active_eval['tp']} z 81 diabetiček)\n"
            f"- Precision: **{active_eval['precision'] * 100:.2f} %**\n"
            f"- Celková Accuracy: **{active_eval['accuracy'] * 100:.2f} %**\n"
            f"- F1-Score: **{active_eval['f1']:.4f}** | ROC-AUC: **{active_eval['roc_auc']:.4f}**"
        )

    # Interaktivní Plotly Scatter náhodných pokusů z RandomizedSearchCV
    trials_df = pd.DataFrame(active_res["trials"])
    trials_df["log10_C"] = np.log10(trials_df["C"])

    fig_trials = px.scatter(
        trials_df,
        x="C",
        y="mean_test_score",
        color="penalty",
        symbol="penalty",
        log_x=True,
        hover_data=["trial", "C", "penalty", "solver", "mean_train_score"],
        labels={"C": "Inverzní regularizace C (Log-škála)", "mean_test_score": f"Validační CV skóre ({active_key.upper()})"},
        title=f"Průzkum 100 náhodných konfigurací RandomizedSearchCV ({active_key.upper()})"
    )
    # Zvýraznění optimálního bodu
    fig_trials.add_vline(
        x=active_res["best_params"]["C"],
        line_dash="dash",
        line_color="#dc2626",
        annotation_text=f"Nejlepší C = {active_res['best_params']['C']:.4f}",
        annotation_position="top left"
    )
    fig_trials.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_trials, width="stretch")

    st.markdown("---")

    # Sekce 2: Klinická interpretace vah a Odds Ratios (Poměr šancí)
    st.subheader("2. Klinická interpretace modelu: Váhy koeficientů & Odds Ratios")
    st.markdown(
        r"""
        V logistické regresi má každý koeficient přímý medicínský význam. Pravděpodobnost diabetu je modelována vztahem:
        $$\ln\left(\frac{P}{1 - P}\right) = \beta_0 + \sum_{i=1}^p \beta_i x_i$$
        Hodnota **Odds Ratio (Poměr šancí)** je rovna $\text{OR}_i = e^{\beta_i}$:
        - $\text{OR} > 1$: Nárůst biomarkeru o 1 směrodatnou odchylku **zvyšuje šanci** na diabetes.
        - $\text{OR} = 1$: Biomarker nemá na šanci žádný vliv.
        - $\text{OR} < 1$: Biomarker snižuje šanci na diabetes.
        """
    )

    df_fi = pd.DataFrame(data["feature_importance"])
    col_fi1, col_fi2 = st.columns([1, 1])

    with col_fi1:
        fig_or = px.bar(
            df_fi.sort_values("coefficient", ascending=True),
            x="coefficient",
            y="feature",
            orientation="h",
            color="direction",
            color_discrete_map={"Zvyšuje riziko (+)": "#ef4444", "Snižuje riziko (-)": "#3b82f6"},
            text="coefficient",
            title="Váhy koeficientů (Log-Odds) standardizovaných příznaků"
        )
        fig_or.update_traces(texttemplate="%{text:+.3f}", textposition="outside")
        fig_or.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_or, width="stretch")

    with col_fi2:
        st.markdown("##### 🔬 Přehled multiplikativního vlivu (Odds Ratios):")
        display_fi = df_fi[["feature", "coefficient", "odds_ratio", "direction"]].copy()
        display_fi.columns = ["Biomarker", "Koeficient (w)", "Odds Ratio (eʷ)", "Vliv na diabetes"]
        display_fi["Odds Ratio (eʷ)"] = display_fi["Odds Ratio (eʷ)"].map(lambda x: f"{x:.3f}x")
        display_fi["Koeficient (w)"] = display_fi["Koeficient (w)"].map(lambda x: f"{x:+.4f}")
        st.dataframe(display_fi, width="stretch", hide_index=True)

        st.caption(
            "💡 **Klíčové zjištění:** Glukóza má zdaleka nejvyšší váhu ($w = +1.107, \\text{OR} = 3.03\\times$). "
            "Každý nárůst glukózy o $1\\sigma$ ztrojnásobuje šanci pacientky na diagnózu diabetu. "
            "Druhým nejvýznamnějším faktorem je BMI ($w = +0.680, \\text{OR} = 1.97\\times$)."
        )

    st.markdown("---")

    # Sekce 3: Regularization Path (Vývoj koeficientů)
    st.subheader("3. Regularization Path: Smršťování vah při změně C")
    st.markdown(
        "Níže uvedený interaktivní graf demonstruje, jak síla regularizace ($C$) postupně stlačuje váhy příznaků k nule. "
        "Při velmi silné regularizaci ($C < 0.01$) přežívají pouze nejsilnější klinické biomarkery (`glucose` a `bmi`)."
    )

    reg_path_df = pd.DataFrame(data["regularization_path"])
    fig_path = go.Figure()

    feature_cols = data["dataset_info"]["feature_names"]
    colors = px.colors.qualitative.Plotly

    for idx, feat in enumerate(feature_cols):
        fig_path.add_trace(go.Scatter(
            x=reg_path_df["C"],
            y=reg_path_df[f"w_{feat}"],
            mode="lines",
            name=feat,
            line=dict(width=2, color=colors[idx % len(colors)])
        ))

    fig_path.update_layout(
        title="Vývoj koeficientů logistické regrese v závislosti na inverzní regularizaci C",
        xaxis_type="log",
        xaxis_title="Inverzní síla regularizace C (Logaritmická osa)",
        yaxis_title="Váha koeficientu (w)",
        height=450,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig_path.add_vline(
        x=best_p["C"],
        line_width=2,
        line_dash="dot",
        line_color="#dc2626",
        annotation_text=f"Grid/Random Optimum C={best_p['C']:.4f}",
        annotation_position="top left"
    )
    st.plotly_chart(fig_path, width="stretch")

    st.markdown("---")

    # Sekce 4: Matice záměn (Confusion Matrix)
    st.subheader("4. Matice záměn (Confusion Matrix) na testovací sadě")
    col_cm1, col_cm2 = st.columns([1, 1])

    with col_cm1:
        cm_vals = active_eval["confusion_matrix"]
        z_text = [[f"{cm_vals[0][0]} ({cm_vals[0][0]/150*100:.1f} %)<br>Správně negativní (TN)",
                   f"{cm_vals[0][1]} ({cm_vals[0][1]/150*100:.1f} %)<br>Falešná pozitivita (FP)"],
                  [f"{cm_vals[1][0]} ({cm_vals[1][0]/81*100:.1f} %)<br>Kritická chyba (FN)",
                   f"{cm_vals[1][1]} ({cm_vals[1][1]/81*100:.1f} %)<br>Správně zachyceno (TP)"]]

        fig_cm = px.imshow(
            cm_vals,
            x=["Predikce: Zdravá (0)", "Predikce: Diabetes (1)"],
            y=["Skutečnost: Zdravá (0)", "Skutečnost: Diabetes (1)"],
            color_continuous_scale="Blues",
            title=f"Matice záměn LogReg (C={active_res['best_params']['C']:.4f})"
        )
        fig_cm.update_traces(text=z_text, texttemplate="%{text}")
        fig_cm.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_cm, width="stretch")

    with col_cm2:
        st.markdown("##### 🩺 Srovnání klasifikačního chování:")
        st.write(
            f"- **Celkem testováno:** 231 pacientek (150 zdravých, 81 diabetiček).\n"
            f"- **Úspěšný záchyt (TP):** Model zachytil **{active_eval['tp']} diabetiček** ze 81 "
            f"(Recall = **{active_eval['recall']*100:.1f} %**).\n"
            f"- **Přehlédnuté diagnózy (FN):** **{active_eval['fn']} diabetiček** nebylo rozpoznáno.\n"
            f"- **Falešné poplachy (FP):** **{active_eval['fp']} zdravých žen** bylo označeno jako podezřelé.\n"
            f"- **Specificita:** **{active_eval['tn']/150*100:.1f} %** zdravých žen správně vyloučeno."
        )

    st.markdown("---")

    # Sekce 5: Přímé srovnání: k-NN vs. Logistická regrese
    st.subheader("5. Mezimodelové srovnání: k-NN vs. Logistická regrese")
    st.markdown(
        "Oba modely byly trénovány a testovány na **identickém rozdělení 70/30** se stejnou normalizací dat. "
        "Zde je jejich přímé porovnání:"
    )

    comp = data.get("model_comparison", None)
    if comp:
        df_comp = pd.DataFrame([
            {
                "Model": comp["knn"]["model_name"],
                "Accuracy (%)": f"{comp['knn']['accuracy'] * 100:.2f} %",
                "Recall (Záchyt) (%)": f"{comp['knn']['recall'] * 100:.2f} %",
                "Precision (%)": f"{comp['knn']['precision'] * 100:.2f} %",
                "F1-Score": f"{comp['knn']['f1']:.4f}",
                "ROC-AUC": f"{comp['knn']['roc_auc']:.4f}",
                "TP (Zachyceno)": f"{comp['knn']['tp']} / 81",
                "FN (Uniklo)": f"{comp['knn']['fn']}"
            },
            {
                "Model": comp["logreg"]["model_name"],
                "Accuracy (%)": f"{comp['logreg']['accuracy'] * 100:.2f} %",
                "Recall (Záchyt) (%)": f"{comp['logreg']['recall'] * 100:.2f} %",
                "Precision (%)": f"{comp['logreg']['precision'] * 100:.2f} %",
                "F1-Score": f"{comp['logreg']['f1']:.4f}",
                "ROC-AUC": f"{comp['logreg']['roc_auc']:.4f}",
                "TP (Zachyceno)": f"{comp['logreg']['tp']} / 81",
                "FN (Uniklo)": f"{comp['logreg']['fn']}"
            }
        ])
        st.dataframe(df_comp, width="stretch", hide_index=True)

        # Plotly porovnání metrik
        metrics_names = ["Accuracy", "Recall", "Precision", "F1-Score", "ROC-AUC"]
        knn_scores = [comp["knn"]["accuracy"], comp["knn"]["recall"], comp["knn"]["precision"], comp["knn"]["f1"], comp["knn"]["roc_auc"]]
        lr_scores = [comp["logreg"]["accuracy"], comp["logreg"]["recall"], comp["logreg"]["precision"], comp["logreg"]["f1"], comp["logreg"]["roc_auc"]]

        fig_compare = go.Figure(data=[
            go.Bar(name=comp["knn"]["model_name"], x=metrics_names, y=knn_scores, marker_color="#3b82f6"),
            go.Bar(name=comp["logreg"]["model_name"], x=metrics_names, y=lr_scores, marker_color="#10b981")
        ])
        fig_compare.update_layout(
            barmode="group",
            title="Přímé srovnání metrik na testovací sadě (k-NN vs. Logistic Regression)",
            yaxis_title="Hodnota metriky (0.0 - 1.0)",
            height=400,
            margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_compare, width="stretch")

    st.markdown("---")

    # Sekce 6: Interaktivní sigmoidální kalkulátor pacientky
    st.subheader("6. Interaktivní kalkulátor individuálního rizika (Sigmoida)")
    st.markdown(
        "Zadejte hodnoty pacientky a sledujte, jak logistická regrese transformuje lineární kombinaci "
        "$z = \\beta_0 + \\sum \\beta_i x_i$ přes sigmoidální funkci $\\sigma(z)$ na pravděpodobnost výskytu diabetu."
    )

    with st.form("patient_calc_form"):
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            p_glu = st.slider("Glukóza (mg/dl)", 50, 220, 135)
            p_bmi = st.slider("BMI (kg/m²)", 18.0, 60.0, 33.5, step=0.1)
        with k2:
            p_age = st.slider("Věk (roky)", 21, 81, 42)
            p_preg = st.slider("Počet těhotenství", 0, 17, 3)
        with k3:
            p_dpf = st.slider("Diabetes pedigree", 0.07, 2.50, 0.65, step=0.01)
            p_bp = st.slider("Tlak krve (mm Hg)", 40, 130, 78)
        with k4:
            p_skin = st.slider("Kožní řasa (mm)", 7, 99, 29)
            p_ins = st.slider("Inzulin (mIU/ml)", 14, 846, 140)

        calc_submitted = st.form_submit_button("Vypočítat pravděpodobnost diabetu", width="stretch")

    # Odhad pravděpodobnosti
    # Načteme průměry a rozptyly ze souboru pro správnou standardizaci
    prep_cache_path = base_dir / "02_Classification" / "data" / "diabetes_preprocessing_precomputed.json"
    if prep_cache_path.exists():
        with open(prep_cache_path, "r", encoding="utf-8") as f:
            prep_data = json.load(f)
        means = prep_data["scaler_means"]
        scales = prep_data["scaler_scales"]

        inputs = {
            "pregnancies": p_preg,
            "glucose": p_glu,
            "blood_pressure": p_bp,
            "skin_thickness": p_skin,
            "insulin": p_ins,
            "bmi": p_bmi,
            "diabetes_pedigree_function": p_dpf,
            "age": p_age
        }

        # Standardizované z-skóre
        z_inputs = {k: (inputs[k] - means[k]) / scales[k] for k in inputs}
        
        # Lineární kombinace z = intercept + sum(w_i * x_i)
        coef_map = {item["feature"]: item["coefficient"] for item in data["feature_importance"]}
        z_val = data["intercept"] + sum(coef_map[k] * z_inputs[k] for k in z_inputs)
        prob = 1.0 / (1.0 + np.exp(-z_val))

        p_col1, p_col2, p_col3 = st.columns(3)
        with p_col1:
            st.metric("Lineární skóre (z)", f"{z_val:+.3f}", delta="Před aktivací sigmoidy")
        with p_col2:
            st.metric("Odhadnutá pravděpodobnost P", f"{prob * 100:.1f} %", delta="Pozitivní nález" if prob >= 0.5 else "Negativní nález")
        with p_col3:
            st.metric("Doporučená diagnóza", "DIABETES (1)" if prob >= 0.5 else "ZDRAVÁ (0)")

        # Sigmoidální křivka v Plotly s vyznačeným bodem pacientky
        z_range = np.linspace(-6, 6, 200)
        sigmoid_vals = 1.0 / (1.0 + np.exp(-z_range))

        fig_sig = go.Figure()
        fig_sig.add_trace(go.Scatter(
            x=z_range, y=sigmoid_vals, mode="lines", name="Logistická křivka σ(z)",
            line=dict(color="#2563eb", width=3)
        ))
        # Hranice 0.5
        fig_sig.add_hline(y=0.5, line_dash="dash", line_color="#94a3b8", annotation_text="Rozhodovací práh (0.5)")
        # Bod pacientky
        fig_sig.add_trace(go.Scatter(
            x=[z_val], y=[prob], mode="markers+text", name="Profil pacientky",
            text=[f"Pacientka: {prob * 100:.1f} % (z={z_val:+.2f})"],
            textposition="top left",
            marker=dict(size=14, color="#ef4444", symbol="star")
        ))
        fig_sig.update_layout(
            title="Umístění pacientky na logistické sigmoidální funkci",
            xaxis_title="Lineární kombinace vstupů z = wᵀx + b",
            yaxis_title="Predikovaná pravděpodobnost P(diabetes = 1)",
            height=380,
            margin=dict(l=10, r=10, t=50, b=10)
        )
        st.plotly_chart(fig_sig, width="stretch")


if __name__ == "__main__":
    st.set_page_config(page_title="LogReg: Diabetes", layout="wide")
    render_diabetes_logistic_regression_view()
