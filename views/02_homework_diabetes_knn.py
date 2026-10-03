"""
Homework: K-Nearest Neighbors (k-NN) na škálovaných datech diabetu
=================================================================
Interaktivní analytický modul pro klasifikaci diabetu pomocí algoritmu k-NN:
1. Volba optimalizační metriky pro GridSearchCV (medicínská diskuse: Recall vs F1 vs Accuracy).
2. Interaktivní Grid Search 2D heatmapa a hyperparametrický prostor (k vs. metrika vzdálenosti).
3. Interaktivní K-Sweep křivky (Train vs Test Accuracy, Recall, F1) odhalující přeučení při k=1.
4. Interaktivní matice záměn (Confusion Matrix) v Plotly s medicínskou interpretací.
5. Interaktivní ROC křivka a Precision-Recall profil.
6. Interaktivní simulátor klinického rozhodovacího prahu (Threshold Tuning).
"""

import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def load_knn_precomputed():
    base_dir = Path(__file__).resolve().parent.parent
    cache_path = base_dir / "02_Classification" / "data" / "diabetes_knn_precomputed.json"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def render_diabetes_knn_view():
    st.title("🔍 DÚ: K-Nearest Neighbors – Klasifikace diabetu")
    st.markdown(
        "**Vypracování cvičení:** Trénování algoritmu **k-NN** na standardizovaném datasetu `diabetes_scaled.csv`, "
        "dělení v poměru 70/30, systematické ladění hyperparametrů (`n_neighbors`, `metric`, `weights`) přes **GridSearchCV** "
        "a hloubkové zhodnocení efektivity na testovací sadě s ohledem na medicínský kontext."
    )

    data = load_knn_precomputed()
    if not data:
        st.error("Předpočtená data nebyla nalezena. Spusťte prosím skript `02_Classification/16_homework_diabetes_knn.py`.")
        return

    # Základní informace o modelu
    pri_metric = data.get("primary_metric", "f1")
    eval_m = data["test_evaluations"][pri_metric]
    best_p = data["grid_results"][pri_metric]["best_params"]

    # Horní KPI karty
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Optimální k", f"{best_p['n_neighbors']}", delta=f"Metrika: {best_p['metric']}")
    with c2:
        st.metric("Test F1-Score", f"{eval_m['f1']:.4f}", delta="Harmonický průměr")
    with c3:
        st.metric("Test Recall (Záchyt)", f"{eval_m['recall'] * 100:.1f} %", delta=f"{eval_m['tp']} z 81 diabetiček")
    with c4:
        st.metric("Test Precision", f"{eval_m['precision'] * 100:.1f} %", delta="Spolehlivost pozitivity")
    with c5:
        st.metric("ROC-AUC", f"{eval_m['roc_auc']:.4f}", delta="Separační schopnost")

    st.markdown("---")

    # Sekce 1: Medicínská úvaha o volbě optimalizační metriky
    st.subheader("1. Volba optimalizační metriky v medicínském screeningu")
    st.markdown(
        "Při diagnostice diabetu **nejsou obě chyby rovnocenné**: \n"
        "- **Falešně negativní (FN):** Nemocná žena je označena jako zdravá $\\rightarrow$ **kritické riziko**, "
        "nedostane včasnou intervenci, hrozí rozvoj diabetické retinopatie, neuropatie a selhání ledvin.\n"
        "- **Falešně pozitivní (FP):** Zdravá žena je označena jako diabetička $\\rightarrow$ menší riziko, "
        "bude odeslána na kontrolní orální glukózový toleranční test (oGTT).\n\n"
        "**Závěr:** Optimalizace na pouhou **Accuracy** je zavádějící, protože model může dosahovat vysoké přesnosti "
        "triviální preferencí většinové třídy (65 % zdravých). Pro medicínský screening volíme **F1-Score** nebo **Recall**."
    )

    # Přepínač optimalizační metriky
    selected_metric_choice = st.radio(
        "Vyberte optimalizační cíl pro zobrazení parametrů GridSearchCV:",
        options=["F1-Score (Doporučeno)", "Recall (Maximalizace záchytu)", "ROC-AUC (Pravděpodobnostní separace)", "Accuracy (Celková přesnost)"],
        horizontal=True
    )
    metric_key_map = {
        "F1-Score (Doporučeno)": "f1",
        "Recall (Maximalizace záchytu)": "recall",
        "ROC-AUC (Pravděpodobnostní separace)": "roc_auc",
        "Accuracy (Celková přesnost)": "accuracy"
    }
    active_metric_key = metric_key_map[selected_metric_choice]
    active_grid = data["grid_results"][active_metric_key]
    active_eval = data["test_evaluations"][active_metric_key]

    c_box1, c_box2 = st.columns([1, 1])
    with c_box1:
        st.info(
            f"**Nejlepší parametry pro {selected_metric_choice}:**\n"
            f"- Počet sousedů (`n_neighbors`): **{active_grid['best_params']['n_neighbors']}**\n"
            f"- Metrika vzdálenosti (`metric`): **{active_grid['best_params']['metric']}**\n"
            f"- Váhy sousedů (`weights`): **{active_grid['best_params']['weights']}**\n"
            f"- Validační 5-Fold CV skóre: **{active_grid['best_score']:.4f}**"
        )
    with c_box2:
        st.success(
            f"**Výsledky na testovací sadě (231 pacientek):**\n"
            f"- Recall (Záchyt): **{active_eval['recall'] * 100:.2f} %** ({active_eval['tp']} z 81)\n"
            f"- Precision: **{active_eval['precision'] * 100:.2f} %**\n"
            f"- Accuracy: **{active_eval['accuracy'] * 100:.2f} %**\n"
            f"- F1-Score: **{active_eval['f1']:.4f}** | ROC-AUC: **{active_eval['roc_auc']:.4f}**"
        )

    st.markdown("---")

    # Sekce 2: Interaktivní 2D Grid Search Heatmapa
    st.subheader("2. Interaktivní mřížka hyperparametrů: Vliv $k$ a metriky vzdálenosti")
    st.markdown(
        "Níže uvedená interaktivní heatmapa zobrazuje průměrné validační skóre z 5-násobné křížové validace "
        "pro všechny testované kombinace počtu sousedů ($k \\in [1, 35]$) a metrik vzdálenosti."
    )

    # Sestavení dat pro heatmapu z cv_results
    params_list = active_grid["cv_results"]["params"]
    scores_list = active_grid["cv_results"]["mean_test_score"]

    heatmap_rows = []
    for p, s in zip(params_list, scores_list):
        if p.get("weights") == "uniform":
            heatmap_rows.append({
                "metric": p["metric"],
                "n_neighbors": p["n_neighbors"],
                "score": s
            })

    df_hm = pd.DataFrame(heatmap_rows)
    pivot_hm = df_hm.pivot(index="metric", columns="n_neighbors", values="score")

    fig_grid_hm = px.imshow(
        pivot_hm,
        labels=dict(x="Počet sousedů (k)", y="Metrika vzdálenosti", color=f"CV {active_metric_key.upper()}"),
        x=pivot_hm.columns,
        y=pivot_hm.index,
        color_continuous_scale="Viridis",
        title=f"Validace mřížky GridSearchCV pro metriku: {active_metric_key.upper()} (weights='uniform')"
    )
    fig_grid_hm.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_grid_hm, use_container_width=True)

    st.markdown("---")

    # Sekce 3: K-Sweep křivky – Analýza přeučení (Overfitting vs Underfitting)
    st.subheader("3. Analýza přeučení v k-NN: Křivky metrik napříč $k \\in [1, 35]$")
    st.markdown(
        "Klíčový koncept algoritmu k-NN: \n"
        "- Pro malé $k=1$: model je **extrémně přeučený (Overfitting)**. Na trénovací sadě má 100 % přesnost, "
        "ale na testovací sadě prudce klesá a je náchylný k šumu.\n"
        "- Se zvyšujícím se $k$: rozhodovací hranice se vyhlazuje, roste bias a klesá variance. "
        "Optimum se nachází v intervalu $k \\in [15, 20]$."
    )

    df_sweep = pd.DataFrame(data["k_sweep"])

    selected_dist = st.selectbox(
        "Vyberte metriku vzdálenosti pro zobrazení průběhu křivek:",
        options=["euclidean", "manhattan", "minkowski"],
        index=0
    )
    df_filtered_sweep = df_sweep[df_sweep["metric"] == selected_dist]

    fig_curves = go.Figure()
    fig_curves.add_trace(go.Scatter(
        x=df_filtered_sweep["k"], y=df_filtered_sweep["train_acc"],
        mode="lines+markers", name="Trénovací přesnost (Train Accuracy)",
        line=dict(color="#ef4444", width=2, dash="dash")
    ))
    fig_curves.add_trace(go.Scatter(
        x=df_filtered_sweep["k"], y=df_filtered_sweep["test_acc"],
        mode="lines+markers", name="Testovací přesnost (Test Accuracy)",
        line=dict(color="#3b82f6", width=3)
    ))
    fig_curves.add_trace(go.Scatter(
        x=df_filtered_sweep["k"], y=df_filtered_sweep["test_f1"],
        mode="lines+markers", name="Testovací F1-Score",
        line=dict(color="#10b981", width=2)
    ))
    fig_curves.add_trace(go.Scatter(
        x=df_filtered_sweep["k"], y=df_filtered_sweep["test_recall"],
        mode="lines+markers", name="Testovací Recall (Záchyt)",
        line=dict(color="#f59e0b", width=2)
    ))

    # Zvýraznění optimálního k
    fig_curves.add_vline(
        x=best_p["n_neighbors"],
        line_width=2,
        line_dash="dot",
        line_color="#8b5cf6",
        annotation_text=f"GridSearchCV Optimum (k={best_p['n_neighbors']})",
        annotation_position="top left"
    )

    fig_curves.update_layout(
        title=f"Průběh výkonu k-NN na testovacích datech pro {selected_dist.capitalize()} vzdálenost",
        xaxis_title="Počet sousedů (k)",
        yaxis_title="Hodnota metriky (0.0 - 1.0)",
        height=450,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_curves, use_container_width=True)

    st.markdown("---")

    # Sekce 4: Interaktivní Matice záměn (Confusion Matrix)
    st.subheader("4. Matice záměn (Confusion Matrix) a medicínská interpretace")
    col_cm1, col_cm2 = st.columns([1, 1])

    with col_cm1:
        cm_vals = active_eval["confusion_matrix"]
        cm_labels = [["Skutečně Zdravé (TN)", "Falešný poplach (FP)"],
                     ["Uniklé diabetičky (FN)", "Správně zachycené (TP)"]]

        z_text = [[f"{cm_vals[0][0]} ({cm_vals[0][0]/150*100:.1f} %)<br>Správně negativní",
                   f"{cm_vals[0][1]} ({cm_vals[0][1]/150*100:.1f} %)<br>Falešná pozitivita"],
                  [f"{cm_vals[1][0]} ({cm_vals[1][0]/81*100:.1f} %)<br>Kritická chyba (FN)",
                   f"{cm_vals[1][1]} ({cm_vals[1][1]/81*100:.1f} %)<br>Úspěšný záchyt (TP)"]]

        fig_cm = px.imshow(
            cm_vals,
            x=["Predikce: Zdravá (0)", "Predikce: Diabetes (1)"],
            y=["Skutečnost: Zdravá (0)", "Skutečnost: Diabetes (1)"],
            color_continuous_scale="Blues",
            title=f"Matice záměn na testovací sadě (k={active_grid['best_params']['n_neighbors']})"
        )
        fig_cm.update_traces(text=z_text, texttemplate="%{text}")
        fig_cm.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_cm2:
        st.markdown("##### 🩺 Lékařský rozbor klasifikačních chyb:")
        st.write(
            f"- **Skutečně zdravé pacientky (Celkem 150):**\n"
            f"  - **{active_eval['tn']}** správně identifikováno jako negativní (**Specificita = {active_eval['tn']/150*100:.1f} %**).\n"
            f"  - **{active_eval['fp']}** falešných poplachů (FP). Pacientky podstoupí doplňkový test.\n\n"
            f"- **Pacientky s diabetem (Celkem 81):**\n"
            f"  - **{active_eval['tp']}** úspěšně zachyceno (**Senzitivita/Recall = {active_eval['recall']*100:.1f} %**).\n"
            f"  - **{active_eval['fn']}** diabetiček model přehlédl (**False Negatives**).\n\n"
            f"💡 **Doporučení pro klinickou praxi:** Standardní rozhodovací práh $0.5$ přehlíží {active_eval['fn']} diabetiček. "
            f"V medicíně se práh snižuje (např. na $0.35$), aby se zvýšil záchyt (Recall) i za cenu vyššího počtu FP."
        )

    st.markdown("---")

    # Sekce 5: Interaktivní ROC a Precision-Recall křivky
    st.subheader("5. ROC křivka a separace pravděpodobností")
    c_roc1, c_roc2 = st.columns([1, 1])

    with c_roc1:
        roc_d = data["roc_curve"]
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=roc_d["fpr"], y=roc_d["tpr"],
            mode="lines", name=f"k-NN (AUC = {eval_m['roc_auc']:.4f})",
            line=dict(color="#2563eb", width=3)
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines", name="Náhodný odhad (AUC = 0.50)",
            line=dict(color="#94a3b8", dash="dash")
        ))
        fig_roc.update_layout(
            title="ROC Křivka (Receiver Operating Characteristic)",
            xaxis_title="False Positive Rate (1 - Specificita)",
            yaxis_title="True Positive Rate (Recall / Senzitivita)",
            height=380,
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with c_roc2:
        pr_d = data["pr_curve"]
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(
            x=pr_d["recall"], y=pr_d["precision"],
            mode="lines", name="Precision-Recall křivka",
            line=dict(color="#10b981", width=3)
        ))
        fig_pr.update_layout(
            title="Precision-Recall Křivka (Vhodná pro mírně nevyvážená data)",
            xaxis_title="Recall (Záchyt)",
            yaxis_title="Precision (Přesnost)",
            height=380,
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    st.markdown("---")

    # Sekce 6: Interaktivní simulátor klinického rozhodovacího prahu
    st.subheader("6. Interaktivní simulátor rozhodovacího prahu (Threshold Tuning)")
    st.markdown(
        "Vyzkoušejte si, jak změna pravděpodobnostního prahu pro označení pacientky za diabetičku "
        "ovlivní poměr záchytu (Recall) a falešných poplachů (FP)."
    )

    col_sim_ctrl, col_sim_res = st.columns([1, 2])

    with col_sim_ctrl:
        sim_threshold = st.slider(
            "Rozhodovací práh pravděpodobnosti (Threshold)",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.05,
            help="Při snížení prahu pod 0.5 model označí za diabetičku každou ženu s menší mírou podezření."
        )

    # Výpočet dynamických metrik pro zvolený práh na testovací sadě
    # Simulujeme z ROC dat interpolaci
    # Použijeme data z testovací sady
    tpr_idx = np.argmin(np.abs(np.array(roc_d["thresholds"]) - sim_threshold))
    dyn_tpr = roc_d["tpr"][tpr_idx]
    dyn_fpr = roc_d["fpr"][tpr_idx]

    dyn_tp = int(round(dyn_tpr * 81))
    dyn_fn = 81 - dyn_tp
    dyn_fp = int(round(dyn_fpr * 150))
    dyn_tn = 150 - dyn_fp

    dyn_rec = dyn_tp / (dyn_tp + dyn_fn) if (dyn_tp + dyn_fn) > 0 else 0
    dyn_prec = dyn_tp / (dyn_tp + dyn_fp) if (dyn_tp + dyn_fp) > 0 else 0
    dyn_f1 = (2 * dyn_prec * dyn_rec) / (dyn_prec + dyn_rec) if (dyn_prec + dyn_rec) > 0 else 0

    with col_sim_res:
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.metric("Nový Recall (Záchyt)", f"{dyn_rec * 100:.1f} %", delta=f"{dyn_tp} z 81 zachyceno")
        with sc2:
            st.metric("Nová Precision", f"{dyn_prec * 100:.1f} %", delta=f"{dyn_fp} falešných poplachů")
        with sc3:
            st.metric("Výsledné F1-Score", f"{dyn_f1:.4f}")

        st.caption(
            f"Při prahu **{sim_threshold:.2f}**: Úspěšně odhaleno **{dyn_tp}** diabetiček ze 81, "
            f"přehlédnuto pouze **{dyn_fn}** žen (FN), vygenerováno **{dyn_fp}** kontrolních testů (FP)."
        )


if __name__ == "__main__":
    st.set_page_config(page_title="k-NN: Diabetes", layout="wide")
    render_diabetes_knn_view()
