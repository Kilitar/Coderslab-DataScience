"""
Homework: Support Vector Machine (SVM) – Klasifikace diabetu (Hyperopt)
======================================================================
Interaktivní analytický modul pro klasifikaci diabetu pomocí SVM:
1. Bayesovská optimalizace hyperparametrů (kernel, C, gamma) přes Hyperopt (TPE).
2. Interaktivní konvergenční křivka a hyperparametrický prostor zkoumaných bodů.
3. Analýza podpůrných vektorů (Support Vectors) a matice záměn na testovacích datech.
4. Interaktivní ROC a Precision-Recall křivky.
5. FINÁLNÍ VELKÉ SROVNÁNÍ: k-NN vs. Logistická regrese vs. SVM na stejné testovací sadě.
6. Vyhodnocení otázky ze zadání: Který ze tří modelů podal nejlepší výkon?
"""

import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def load_svm_precomputed():
    base_dir = Path(__file__).resolve().parent.parent
    cache_path = base_dir / "02_Classification" / "data" / "diabetes_svm_precomputed.json"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def render_diabetes_svm_view():
    st.title("⚔️ DÚ: Support Vector Machine – Klasifikace diabetu")
    st.markdown(
        "**Vypracování cvičení:** Trénování algoritmu **Support Vector Machine (SVC)** na standardizovaných datech "
        "`diabetes_scaled.csv`, split 70/30, Bayesovská optimalizace hyperparametrů (`kernel`, `C`, `gamma`, `degree`) "
        "pomocí knihovny **Hyperopt (TPE)** na stejné metrice **F1-Score** a závěrečné velké srovnání všech 3 modelů."
    )

    data = load_svm_precomputed()
    if not data:
        st.error("Předpočtená data nebyla nalezena. Spusťte prosím skript `02_Classification/18_homework_diabetes_svm.py`.")
        return

    best_p = data["best_params"]
    te = data["test_evaluation"]

    # Horní KPI karty
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Optimální Jádro", f"{best_p['kernel'].upper()}", delta=f"C={best_p['C']:.2f}")
    with c2:
        st.metric("Podpůrné vektory", f"{te['n_support_vectors']}", delta="Z 537 trénovacích bodů")
    with c3:
        st.metric("Test F1-Score", f"{te['f1']:.4f}", delta="Validační CV: " + f"{data['best_cv_f1']:.4f}")
    with c4:
        st.metric("Test Recall (Záchyt)", f"{te['recall'] * 100:.1f} %", delta=f"{te['tp']} z 81 zachyceno")
    with c5:
        st.metric("ROC-AUC", f"{te['roc_auc']:.4f}", delta="Nejvyšší ze 3 modelů")

    st.markdown("---")

    # Sekce 1: Bayesovská optimalizace (Hyperopt & TPE)
    st.subheader("1. Bayesovská optimalizace hyperparametrů (Hyperopt / TPE)")
    st.markdown(
        "Místo vyčerpávající mřížky (Grid Search) nebo náhodného vzorkování (Random Search) byl použit "
        "**Bayesovský algoritmus TPE (Tree of Parzen Estimators)** ze slajdů kurzu. "
        "TPE modeluje rozdělení pravděpodobnosti $P(x|y)$ slibných hyperparametrů a postupně koncentruje pokusy do oblastí s nejvyšším F1-skóre."
    )

    t1, t2 = st.columns([1, 1])
    with t1:
        st.info(
            f"**Nalezené optimální hyperparametry (100 iterací TPE):**\n"
            f"- Typ jádra (`kernel`): **{best_p['kernel'].upper()}**\n"
            f"- Inverzní regularizace `C`: **{best_p['C']:.4f}**\n"
            f"- Koeficient jádra `gamma`: **{best_p['gamma']:.6f}**\n"
            f"- Stupeň polynomu (`degree`): **{best_p['degree']}**\n"
            f"- Nejlepší validační 5-Fold F1: **{data['best_cv_f1']:.4f}**"
        )
    with t2:
        st.success(
            f"**Výsledky na testovací sadě (231 pacientek):**\n"
            f"- Accuracy: **{te['accuracy'] * 100:.2f} %**\n"
            f"- Recall (Záchyt): **{te['recall'] * 100:.2f} %** ({te['tp']} z 81 diabetiček)\n"
            f"- Precision: **{te['precision'] * 100:.2f} %**\n"
            f"- F1-Score: **{te['f1']:.4f}**\n"
            f"- ROC-AUC: **{te['roc_auc']:.4f}**"
        )

    # Interaktivní graf konvergence TPE v Plotly
    conv_df = pd.DataFrame(data["convergence"])
    fig_conv = go.Figure()
    fig_conv.add_trace(go.Scatter(
        x=conv_df["iteration"], y=conv_df["trial_f1"],
        mode="markers", name="Jednotlivé pokusy (Trial F1)",
        marker=dict(color="#94a3b8", size=6, opacity=0.7)
    ))
    fig_conv.add_trace(go.Scatter(
        x=conv_df["iteration"], y=conv_df["best_f1"],
        mode="lines", name="Nejlepší nalezené F1 (Kumulativní maximum)",
        line=dict(color="#2563eb", width=3)
    ))
    fig_conv.update_layout(
        title="Konvergenční křivka Bayesovské optimalizace (Hyperopt TPE – 100 iterací)",
        xaxis_title="Iterace optimalizace",
        yaxis_title="Validační F1-Score",
        height=380,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_conv, width="stretch")

    st.markdown("---")

    # Sekce 2: Matice záměn a analýza podpůrných vektorů
    st.subheader("2. Matice záměn (Confusion Matrix) a geometrie podpůrných vektorů")
    col_m1, col_m2 = st.columns([1, 1])

    with col_m1:
        cm_vals = te["confusion_matrix"]
        z_text = [[f"{cm_vals[0][0]} ({cm_vals[0][0]/150*100:.1f} %)<br>Správně negativní (TN)",
                   f"{cm_vals[0][1]} ({cm_vals[0][1]/150*100:.1f} %)<br>Falešná pozitivita (FP)"],
                  [f"{cm_vals[1][0]} ({cm_vals[1][0]/81*100:.1f} %)<br>Kritická chyba (FN)",
                   f"{cm_vals[1][1]} ({cm_vals[1][1]/81*100:.1f} %)<br>Správně zachyceno (TP)"]]

        fig_cm = px.imshow(
            cm_vals,
            x=["Predikce: Zdravá (0)", "Predikce: Diabetes (1)"],
            y=["Skutečnost: Zdravá (0)", "Skutečnost: Diabetes (1)"],
            color_continuous_scale="Blues",
            title=f"Matice záměn SVC ({best_p['kernel'].upper()} jádro)"
        )
        fig_cm.update_traces(text=z_text, texttemplate="%{text}")
        fig_cm.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_cm, width="stretch")

    with col_m2:
        st.markdown("##### 📐 Geometrický rozbor podpůrných vektorů:")
        st.write(
            f"- **Celkem trénovacích vzorků:** 537 pacientek.\n"
            f"- **Aktivní podpůrné vektory (Support Vectors):** **{te['n_support_vectors']}** "
            f"({te['n_support_vectors']/537*100:.1f} % dat leží na okraji marginu nebo uvnitř něj).\n"
            f"  - Třída 0 (Zdravé): **{te['support_per_class'][0]}** podpůrných vektorů.\n"
            f"  - Třída 1 (Diabetičky): **{te['support_per_class'][1]}** podpůrných vektorů.\n\n"
            f"💡 **Vysvětlení:** Velký podíl podpůrných vektorů (~52 %) potvrzuje, že třídy diabetu a zdravých žen "
            f"se v prostoru 8 standardizovaných příznaků **výrazně překrývají**. "
            f"RBF jádro vytváří zakřivenou separační nadrovinu, která efektivně separuje jádro populace, "
            f"ale na okrajích má tendenci k opatrnější klasifikaci."
        )

    st.markdown("---")

    # Sekce 3: ROC a Precision-Recall křivky
    st.subheader("3. ROC a Precision-Recall křivky pro optimalizované SVM")
    c_roc1, c_roc2 = st.columns([1, 1])

    with c_roc1:
        roc_d = data["roc_curve"]
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=roc_d["fpr"], y=roc_d["tpr"],
            mode="lines", name=f"SVM RBF (AUC = {te['roc_auc']:.4f})",
            line=dict(color="#2563eb", width=3)
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines", name="Náhodný odhad (AUC = 0.50)",
            line=dict(color="#94a3b8", dash="dash")
        ))
        fig_roc.update_layout(
            title="ROC Křivka – Schopnost pravděpodobnostní separace",
            xaxis_title="False Positive Rate (1 - Specificita)",
            yaxis_title="True Positive Rate (Recall)",
            height=380,
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_roc, width="stretch")

    with c_roc2:
        pr_d = data["pr_curve"]
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(
            x=pr_d["recall"], y=pr_d["precision"],
            mode="lines", name="Precision-Recall křivka",
            line=dict(color="#10b981", width=3)
        ))
        fig_pr.update_layout(
            title="Precision-Recall Křivka pro diagnostiku",
            xaxis_title="Recall (Záchyt)",
            yaxis_title="Precision",
            height=380,
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_pr, width="stretch")

    st.markdown("---")

    # Sekce 4: VELKÉ SROVNÁNÍ VŠECH 3 MODELŮ A VERDIKT
    st.subheader("4. Velké srovnání všech tří modelů: Který fungoval nejlépe?")
    st.markdown(
        "Všechny tři algoritmy (**k-NN, Logistická regrese, SVM**) byly natrénovány na **naprosto identickém rozdělení dat (70/30)** "
        "se stejnou normalizací a laděny na stejnou optimalizační metriku (**F1-Score**)."
    )

    all_models = data["all_models_comparison"]
    df_all = pd.DataFrame([
        {
            "Model": m["model_name"],
            "Accuracy": f"{m['accuracy'] * 100:.2f} %",
            "Recall (Záchyt)": f"{m['recall'] * 100:.2f} %",
            "Precision": f"{m['precision'] * 100:.2f} %",
            "F1-Score": f"{m['f1']:.4f}",
            "ROC-AUC": f"{m['roc_auc']:.4f}",
            "TP (Zachyceno)": f"{m['tp']} / 81",
            "FN (Uniklo)": f"{m['fn']}"
        }
        for m in all_models
    ])
    st.dataframe(df_all, width="stretch", hide_index=True)

    # Interaktivní porovnání v Plotly
    metric_labels = ["Accuracy", "Recall (Záchyt)", "Precision", "F1-Score", "ROC-AUC"]
    fig_comp_all = go.Figure()

    model_colors = ["#3b82f6", "#10b981", "#8b5cf6"]
    for idx, m in enumerate(all_models):
        scores = [m["accuracy"], m["recall"], m["precision"], m["f1"], m["roc_auc"]]
        fig_comp_all.add_trace(go.Bar(
            name=m["model_name"],
            x=metric_labels,
            y=scores,
            marker_color=model_colors[idx % len(model_colors)]
        ))

    fig_comp_all.update_layout(
        barmode="group",
        title="Komplexní mezimodelové srovnání na testovací sadě (231 pacientek)",
        yaxis_title="Hodnota metriky (0.0 - 1.0)",
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_comp_all, width="stretch")

    # Inženýrsko-medicínský verdikt na položenou otázku
    st.markdown("### 🏆 Závěrečný verdikt: Který model fungoval nejlépe?")
    
    col_v1, col_v2, col_v3 = st.columns(3)
    with col_v1:
        st.success(
            "#### 🥇 Celkový vítěz metrik: k-NN (k=17)\n"
            "- **Nejvyšší F1-Score: 0.6081**\n"
            "- **Nejvyšší celková přesnost: 74.89 %**\n"
            "- **Nejvyšší záchyt diabetiček: 55.56 % (45 z 81)**\n"
            "- **Verdikt:** Vícerozměrné shlukování pacientek podle eukleidovské vzdálenosti "
            "nejlépe vystihlo lokální strukturu populace Pima Indians."
        )
    with col_v2:
        st.info(
            "#### 🥈 Vítěz pro klinickou praxi: Logistická regrese\n"
            "- **F1-Score: 0.5850 | Accuracy: 73.59 %**\n"
            "- **Plná interpretovatelnost:** Lékař přesně ví, proč model rozhodl (Odds Ratios).\n"
            "- Glukóza ztrojnásobuje šanci ($\text{OR} = 3.03$), BMI zdvojnásobuje ($\text{OR} = 1.97$).\n"
            "- **Verdikt:** V medicíně je často preferována před k-NN kvůli transparentnosti a absenci 'černé skříňky'."
        )
    with col_v3:
        st.warning(
            "#### 🥉 Třetí místo: SVM (RBF jádro)\n"
            "- **Nejvyšší ROC-AUC: 0.8371**\n"
            "- Zaostává v Recallu při prahu 0.5 (pouze 46.91 %, zachytilo 38 z 81 diabetiček).\n"
            "- Trpí velkým překryvem tříd (52 % bodů jsou podpůrné vektory).\n"
            "- **Verdikt:** Skvělý potenciál separace pravděpodobností, ale vyžaduje posun rozhodovacího prahu dolů."
        )


if __name__ == "__main__":
    st.set_page_config(page_title="SVM: Diabetes", layout="wide")
    render_diabetes_svm_view()
