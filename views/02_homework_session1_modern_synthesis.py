"""
Syntéza Session 1 (Dny 1 a 2) & SOTA Benchmark k 10/2026
=========================================================
Tento modul přináší ucelený nadstavbový pohled nad rámec kurzu:
1. Velká rekapitulace obou úloh Session 1: Regrese (Beton) a Klasifikace (Diabetes).
2. Srovnání školních modelů (k-NN, LogReg, SVM) s moderními standardy (XGBoost, HistGradientBoosting, Random Forest).
3. Permutační důležitost příznaků (Permutation Importance).
4. Klinická nákladová matice (Cost Matrix: Penalizace 500 USD za FN vs. 50 USD za FP).
5. 6 klíčových technologických posunů v Data Science od roku 2020 k 10/2026.
"""

import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def load_synthesis_precomputed():
    base_dir = Path(__file__).resolve().parent.parent
    cache_path = base_dir / "02_Classification" / "data" / "session1_modern_benchmark_precomputed.json"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def render_session1_modern_synthesis_view():
    st.title("🚀 Syntéza Session 1 & SOTA Benchmark 2026")
    st.markdown(
        "**Závěrečný expertní modul Session 1 (Dny 1 a 2):** "
        "Ucelené srovnání všech vypracovaných modelů regrese i klasifikace, porovnání školních algoritmů "
        "s moderním průmyslovým standardem k **říjnu 2026** (XGBoost, HistGradientBoosting, Optuna, MICE, Conformal Prediction) "
        "a kalkulace klinické nákladové matice."
    )

    data = load_synthesis_precomputed()
    if not data:
        st.error("Předpočtená data nebyla nalezena. Spusťte prosím skript `02_Classification/19_homework_session1_modern_benchmark_2026.py`.")
        return

    # Horní KPI karty
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Nejlepší Regresní model", "Rozhodovací strom", delta="R² = 84.8 %, MAE = 4.39 MPa")
    with c2:
        st.metric("Školní vítěz Klasifikace", "k-NN (k=17)", delta="F1 = 0.6081, Acc = 74.9 %")
    with c3:
        st.metric("Moderní SOTA Vítěz", "XGBoost Classifier", delta="F1 = 0.6309, Acc = 76.2 %")
    with c4:
        st.metric("Nejvyšší Záchyt (Recall)", "HistGradientBoosting", delta="59.3 % (48 z 81 diabetiček)")

    st.markdown("---")

    # Záložky pro přehlednou navigaci
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Srovnání klasifikace (Diabetes)",
        "🧱 Srovnání regrese (Beton)",
        "💰 Klinická nákladová matice",
        "🧠 6 paradigmat moderního ML (10/2026)"
    ])

    # TAB 1: Klasifikace (Diabetes)
    with tab1:
        st.subheader("Mezimodelový benchmark: Školní algoritmy vs. Moderní SOTA (2026)")
        st.markdown(
            "Všechny modely byly vyhodnoceny na **naprosto identické testovací sadě 231 pacientek** (81 diabetiček) "
            "se standardizovanými vstupy a z-score normalizací."
        )

        clf_data = data["classification_comparison"]
        df_clf_table = pd.DataFrame([
            {
                "Model": m["name"],
                "Kategorie": m["type"],
                "Accuracy": f"{m['accuracy'] * 100:.2f} %",
                "Recall (Záchyt)": f"{m['recall'] * 100:.2f} %",
                "Precision": f"{m['precision'] * 100:.2f} %",
                "F1-Score": f"{m['f1']:.4f}",
                "ROC-AUC": f"{m['roc_auc']:.4f}",
                "Zachyceno (TP)": f"{m['tp']} / 81",
                "Uniklo (FN)": f"{m['fn']}"
            }
            for m in clf_data
        ])
        st.dataframe(df_clf_table, width="stretch", hide_index=True)

        # Plotly grouped bar chart
        st.markdown("##### 📈 Grafické porovnání výkonu modelů")
        metric_keys = ["Accuracy", "Recall", "Precision", "F1-Score", "ROC-AUC"]
        fig_bar = go.Figure()

        palette = ["#94a3b8", "#64748b", "#475569", "#2563eb", "#10b981", "#f59e0b"]
        for idx, m in enumerate(clf_data):
            vals = [m["accuracy"], m["recall"], m["precision"], m["f1"], m["roc_auc"]]
            fig_bar.add_trace(go.Bar(
                name=m["name"],
                x=metric_keys,
                y=vals,
                marker_color=palette[idx % len(palette)]
            ))

        fig_bar.update_layout(
            barmode="group",
            title="Porovnání školních a moderních klasifikátorů na testovací sadě",
            yaxis_title="Hodnota metriky (0.0 - 1.0)",
            height=430,
            margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_bar, width="stretch")

        # Permutation Feature Importance
        st.markdown("##### 🎯 Permutační důležitost příznaků (Permutation Feature Importance)")
        st.markdown(
            "Permutační metoda měří pokles přesnosti modelu, když hodnoty daného sloupce náhodně promícháme. "
            "Na rozdíl od vah logistické regrese je tato metoda nezávislá na architektuře a postihuje i nelineární interakce."
        )

        df_perm = pd.DataFrame(data["permutation_importance"])
        fig_perm = px.bar(
            df_perm.sort_values("importance_mean", ascending=True),
            x="importance_mean",
            y="feature",
            error_x="importance_std",
            orientation="h",
            color="importance_mean",
            color_continuous_scale="Viridis",
            labels={"importance_mean": "Pokles přesnosti (Mean Accuracy Drop)", "feature": "Biomarker"},
            title="Důležitost biomarkerů pro SOTA Gradient Boosting model"
        )
        fig_perm.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_perm, width="stretch")

    # TAB 2: Regrese (Beton)
    with tab2:
        st.subheader("Souhrnná syntéza regresní části: Pevnost betonu (Concrete)")
        st.markdown(
            "V regresní části homeworku jsme postupovali od jednoduchého OLS modelu přes regularizaci "
            "až po nelineární rozhodovací stromy:"
        )

        c_reg = data.get("concrete_regression_summary")
        if c_reg:
            df_reg_table = pd.DataFrame([
                {
                    "Model": m["name"],
                    "R² skóre (Test)": f"{m['r2'] * 100:.2f} %",
                    "RMSE (Chyba)": f"{m['rmse']:.3f} MPa",
                    "MAE (Abs. chyba)": f"{m['mae']:.3f} MPa",
                    "Princip a ladění": m["approach"]
                }
                for m in c_reg["models"]
            ])
            st.dataframe(df_reg_table, width="stretch", hide_index=True)

            # Plotly vizualizace skoku mezi lineárními modely a stromy
            fig_reg = go.Figure()
            m_names = [m["name"] for m in c_reg["models"]]
            r2_vals = [m["r2"] * 100 for m in c_reg["models"]]
            mae_vals = [m["mae"] for m in c_reg["models"]]

            fig_reg.add_trace(go.Bar(
                name="Koeficient determinace R² (%)",
                x=m_names,
                y=r2_vals,
                marker_color="#2563eb",
                text=[f"{v:.1f} %" for v in r2_vals],
                textposition="outside"
            ))
            fig_reg.add_trace(go.Bar(
                name="Střední absolutní chyba MAE (MPa)",
                x=m_names,
                y=mae_vals,
                marker_color="#ef4444",
                text=[f"{v:.2f} MPa" for v in mae_vals],
                textposition="outside"
            ))

            fig_reg.update_layout(
                barmode="group",
                title="Srovnání lineárních vs. stromových modelů na datech betonu",
                height=400,
                margin=dict(l=10, r=10, t=50, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_reg, width="stretch")

            st.info(
                "💡 **Hlavní závěr z regrese:** Pevnost betonu závisí na složitých nelineárních chemických interakcích "
                "(hydratace slínku s věkem zrání v logaritmickém měřítku). Lineární modely (OLS, ElasticNet) "
                "narazily na strop $R^2 \\approx 56\\ %$. Rozhodovací strom s hloubkou 7–9 dokázal tyto nelinearity zachytit "
                "a skokově zvýšil $R^2$ na **84.8 %** (snížení chyby MAE na polovinu: z 8.6 MPa na 4.4 MPa)."
            )

    # TAB 3: Klinická nákladová matice
    with tab3:
        st.subheader("Optimalizace klinické nákladové matice (Cost Matrix Analysis)")
        st.markdown(
            "V reálné nemocniční praxi **nejsou chyby symetrické**. "
            "Při zhodnocení celkových nákladů na screeningovou kampaň zohledňujeme:\n"
            "- **Náklad na FN (Přehlédnutý diabetes):** 500 USD (pozdní léčba komplikací, hospitalizace).\n"
            "- **Náklad na FP (Falešný poplach):** 50 USD (opakovaný krevní test oGTT u obvodního lékaře).\n"
            "- **Náklad na TP (Včasný záchyt):** 20 USD (zahájení včasné dietní a pohybové intervence)."
        )

        cost_list = data["cost_matrix_comparison"]
        df_cost = pd.DataFrame([
            {
                "Model": c["name"],
                "Celkové odhadnuté náklady (USD)": f"{c['total_cost']:,} USD",
                "Ztráta z uniklých diabetiček (FN)": f"{c['cost_fn']:,} USD ({c['fn_count']} pacientek)",
                "Náklady na zbytečné testy (FP)": f"{c['cost_fp']:,} USD ({c['fp_count']} pacientek)"
            }
            for c in cost_list
        ])
        st.dataframe(df_cost, width="stretch", hide_index=True)

        # Plotly skládaný sloupcový graf nákladů
        fig_cost = go.Figure()
        c_names = [c["name"] for c in cost_list]
        cost_fns = [c["cost_fn"] for c in cost_list]
        cost_fps = [c["cost_fp"] for c in cost_list]

        fig_cost.add_trace(go.Bar(
            name="Ztráta z FN (přehlédnuté diagnózy)",
            x=c_names,
            y=cost_fns,
            marker_color="#dc2626"
        ))
        fig_cost.add_trace(go.Bar(
            name="Náklady na FP (zbytečné kontroly)",
            x=c_names,
            y=cost_fps,
            marker_color="#f59e0b"
        ))

        fig_cost.update_layout(
            barmode="stack",
            title="Celkové ekonomické a zdravotní ztráty modelů na 231 pacientkách",
            yaxis_title="Odhadované celkové náklady (USD)",
            height=420,
            margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_cost, width="stretch")

        st.caption(
            "💡 **Klinický postřeh:** Modely s vyšším záchytem (HistGradientBoosting s 48 TP a k-NN s 45 TP) "
            "generují výrazně nižší celkové škody, protože zabraňují drahým hospitalizacím pacientů s rozvinutými komplikacemi."
        )

    # TAB 4: 6 paradigmat moderního ML k 10/2026
    with tab4:
        st.subheader("6 klíčových technologických posunů v Machine Learning (10/2026)")
        st.markdown(
            "Jak se posunula praxe v průmyslu od historických sylabů k moderním standardům současnosti?"
        )

        for p in data["modern_paradigms_2026"]:
            with st.expander(f"📌 {p['topic']}", expanded=True):
                col_p1, col_p2 = st.columns([1, 1])
                with col_p1:
                    st.markdown("**Školní/Historický postup:**")
                    st.write(p["legacy"])
                with col_p2:
                    st.markdown("**Moderní standard (10/2026):**")
                    st.write(p["modern_2026"])
                st.success(f"**Doporučení pro praxi:** {p['recommendation']}")


if __name__ == "__main__":
    st.set_page_config(page_title="Syntéza Session 1: SOTA 2026", layout="wide")
    render_session1_modern_synthesis_view()
