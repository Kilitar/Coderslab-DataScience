import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# 1. Načtení předpočtených dat
base_dir = Path(__file__).resolve().parent.parent
json_path = base_dir / "03_Advanced_ML_Neural_Networks" / "data" / "heart_rf_exercise_1_precomputed.json"

st.title("🎯 Cvičení 1: Nemoc srdce – Random Forest Klasifikace")
st.caption("Splnění všech kroků zadání kurzu: Načtení heart_data_normalized.csv, split 70:30, GridSearchCV s metrikou precision, nejlepší hyperparametry a classification_report.")

if not json_path.exists():
    st.error("Předpočtená data `heart_rf_exercise_1_precomputed.json` nebyla nalezena. Spusťte nejprve výpočetní skript.")
    st.stop()

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

meta = data["metadata"]
head_10 = pd.DataFrame(data["head_10"])
dt_base = data["baseline_decision_tree"]
rf_base = data["baseline_default_rf"]
gs_info = data["grid_search"]
opt_model = data["optimal_model_test"]
rep_dict = opt_model["classification_report_dict"]

# =============================================================================
# HORNÍ KPI KARTY
# =============================================================================
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Testovací Precision (Nemoc)", f"{opt_model['precision'] * 100:.2f} %", delta=f"{(opt_model['precision'] - dt_base['precision']) * 100:+.2f} % vs Strom")
with k2:
    st.metric("Testovací Accuracy", f"{opt_model['accuracy'] * 100:.2f} %", delta=f"{(opt_model['accuracy'] - dt_base['accuracy']) * 100:+.2f} % vs Strom")
with k3:
    st.metric("Testovací F1-Score", f"{opt_model['f1']:.4f}", delta=f"{opt_model['f1'] - dt_base['f1']:+.4f} vs Strom")
with k4:
    best_p = gs_info["best_params"]
    st.metric("Optimální les", f"{best_p['n_estimators']} stromů", delta=f"max_depth={best_p['max_depth']}, leaf={best_p['min_samples_leaf']}")

st.divider()

# =============================================================================
# KROK 1 & 2: NAČTENÍ DAT A STRUKTURA DATASETU
# =============================================================================
st.subheader("1. & 2. Načtení dat a kontrola pacientů (`head(10)`)")
st.markdown(
    r"""
    Dataset kardiologických pacientů (`heart_data_normalized.csv`) obsahuje **303 pacientů** a **17 normalizovaných prediktorů**:
    - **Demografie & Anamnéza:** `age`, `sex_male`, `chestpain_...` (3 dummy sloupce pro typ bolesti na hrudi).
    - **Klinická měření:** klidový tlak (`restbp`), cholesterol (`chol`), lačný cukr (`fbs_yes`), klidové EKG (`restecg_...`), maximální tepová frekvence (`maxhr`), zátěžová angina pectoris (`exang_yes`).
    - **Diagnostika zátěže & cév:** ST deprese (`oldpeak`), sklon ST segmentu (`slope`), počet zasažených hlavních cév (`ca`), talasémie (`thal_normal`, `thal_reversable`).
    - **Cílová proměnná `ahd_yes`:** 0 = zdravý pacient (164), 1 = přítomnost onemocnění srdce (139).
    """
)
st.dataframe(head_10, width="stretch", hide_index=True)
st.caption(
    f"Celkem: {meta['dataset_shape'][0]} pacientů | Trénovací sada (70 %): {meta['train_shape'][0]} | "
    f"Testovací sada (30 %): {meta['test_shape'][0]} (`random_state=42`)."
)

st.divider()

# =============================================================================
# KROK 3 AŽ 7: GRIDSEARCHCV OPTIMALIZACE
# =============================================================================
st.subheader("3.–7. Hyperparametrická optimalizace přes GridSearchCV (`scoring='precision'`)")
st.markdown(
    r"""
    Dle zadání byla definována mřížka hyperparametrů `params` a natrénován `GridSearchCV` s 5násobnou křížovou validací (CV=5) 
    s cílem maximalizovat **Precision** (přesnost pozitivní predikce):
    """
)

col_grid_code, col_grid_best = st.columns([1.1, 0.9])
with col_grid_code:
    st.code(
        """
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

rf_classifier = RandomForestClassifier(random_state=42, n_jobs=-1)

params = {
    'max_depth': [3, 5, 7, None],
    'min_samples_leaf': [1, 2, 4],
    'n_estimators': [50, 100, 150]
}

grid_search = GridSearchCV(
    estimator=rf_classifier,
    param_grid=params,
    scoring='precision',
    cv=5,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
        """,
        language="python"
    )

with col_grid_best:
    st.markdown("#### 🏆 Vítězná konfigurace hyperparametrů:")
    st.success(
        f"""
        - **`max_depth`:** `{best_p['max_depth']}` (mělčí stromy omezují přeučení a zvyšují přesnost)
        - **`min_samples_leaf`:** `{best_p['min_samples_leaf']}`
        - **`n_estimators`:** `{best_p['n_estimators']}`
        - **Průměrná validační Precision (CV=5):** **`{gs_info['best_cv_precision'] * 100:.2f} %`**
        """
    )
    st.info("Mřížka prohledala celkem **36 kombinací** (4 × 3 × 3) napříč 180 natrénovanými stromy.")

st.markdown("##### 📋 Přehled nejlepších výsledků křížové validace (Top 10):")
df_top = pd.DataFrame(gs_info["top_results"]).head(10)
df_top.rename(columns={
    "param_max_depth": "max_depth",
    "param_min_samples_leaf": "min_samples_leaf",
    "param_n_estimators": "n_estimators",
    "mean_test_precision": "Mean CV Precision",
    "std_test_precision": "Std CV Precision",
    "rank_test_score": "Pořadí"
}, inplace=True)
df_top["Mean CV Precision"] = df_top["Mean CV Precision"].apply(lambda x: f"{x * 100:.2f} %")
df_top["Std CV Precision"] = df_top["Std CV Precision"].apply(lambda x: f"± {x * 100:.2f} %")
st.dataframe(df_top, width="stretch", hide_index=True)

st.divider()

# =============================================================================
# KROK 8, 9 & 10: VÝSLEDKY NA TESTOVACÍ SADĚ
# =============================================================================
st.subheader("8.–10. Evaluace na testovací sadě: Classification Report & Matice záměn")

col_cm, col_rep = st.columns([1, 1.1])

with col_cm:
    st.markdown("#### Matice záměn (Confusion Matrix)")
    cm = np.array(opt_model["confusion_matrix"])
    fig_cm = px.imshow(
        cm,
        labels=dict(x="Predikovaný stav", y="Skutečný stav", color="Počet pacientů"),
        x=["Zdravý (0)", "Nemoc srdce (1)"],
        y=["Zdravý (0)", "Nemoc srdce (1)"],
        text_auto=True,
        color_continuous_scale="Blues"
    )
    fig_cm.update_layout(height=320, margin=dict(l=30, r=30, t=30, b=30))
    st.plotly_chart(fig_cm, width="stretch")
    st.caption(
        f"✅ **True Negatives:** {cm[0, 0]} | ❌ **False Positives:** {cm[0, 1]} | "
        f"❌ **False Negatives:** {cm[1, 0]} | ✅ **True Positives:** {cm[1, 1]}"
    )

with col_rep:
    st.markdown("#### Detailní `classification_report` na testu (91 pacientů)")
    st.code(opt_model["classification_report_str"], language="text")

    st.markdown("#### Přehledové metriky pro pozitivní třídu (Nemoc srdce):")
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.metric("Precision (Zadání)", f"{opt_model['precision'] * 100:.2f} %", help="36 / (36 + 6) = 85.71 %")
    with mc2:
        st.metric("Recall (Záchyt)", f"{opt_model['recall'] * 100:.2f} %", help="36 / (36 + 7) = 83.72 %")
    with mc3:
        st.metric("F1-Score", f"{opt_model['f1']:.4f}")

st.divider()

# =============================================================================
# SROVNÁNÍ: ROZHODOVACÍ STROM VS DEFAULT RF VS OPTIMALIZOVANÝ RF
# =============================================================================
st.subheader("📊 Srovnání: Samostatný strom vs. Výchozí les vs. Optimalizovaný les")

df_comp = pd.DataFrame({
    "Metrika": ["Testovací Accuracy", "Testovací Precision (Nemoc)", "Testovací Recall (Záchyt)", "Testovací F1-Score"],
    "Samostatný strom (Overfitting)": [dt_base["accuracy"] * 100, dt_base["precision"] * 100, dt_base["recall"] * 100, dt_base["f1"] * 100],
    "Výchozí Random Forest (100 stromů)": [rf_base["accuracy"] * 100, rf_base["precision"] * 100, rf_base["recall"] * 100, rf_base["f1"] * 100],
    "Optimalizovaný RF (GridSearchCV)": [opt_model["accuracy"] * 100, opt_model["precision"] * 100, opt_model["recall"] * 100, opt_model["f1"] * 100]
})

fig_comp = go.Figure()
fig_comp.add_trace(go.Bar(
    x=df_comp["Metrika"],
    y=df_comp["Samostatný strom (Overfitting)"],
    name="Samostatný strom",
    marker_color="#ef4444"
))
fig_comp.add_trace(go.Bar(
    x=df_comp["Metrika"],
    y=df_comp["Výchozí Random Forest (100 stromů)"],
    name="Výchozí Random Forest",
    marker_color="#3b82f6"
))
fig_comp.add_trace(go.Bar(
    x=df_comp["Metrika"],
    y=df_comp["Optimalizovaný RF (GridSearchCV)"],
    name="Optimalizovaný RF (GridSearchCV)",
    marker_color="#10b981"
))
fig_comp.update_layout(
    barmode="group",
    height=360,
    margin=dict(l=30, r=30, t=35, b=30),
    yaxis_title="Hodnota metriky (%)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_comp, width="stretch")

st.info(
    "💡 **Pozorování z benchmarku:** Samostatný rozhodovací strom dosáhl Precision pouze **69.77 %** a chyboval ve 26 případech. "
    "Random Forest díky paralelnímu ansámblu a redukci rozptylu skokově zvýšil Precision na **85.71 %** a celkovou Accuracy na **85.71 %**."
)
