import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# 1. Načtení předpočtených dat
base_dir = Path(__file__).resolve().parent.parent
json_path = base_dir / "03_Advanced_ML_Neural_Networks" / "data" / "heart_xgb_exercise_1_precomputed.json"

st.title("🎯 Cvičení 1: Nemoc srdce – XGBoost Klasifikace")
st.caption("Splnění všech kroků zadání kurzu: Načtení heart_data_normalized.csv, split 70:30, GridSearchCV s parametry max_depth, n_estimators, gamma, objective a learning_rate a classification_report.")

if not json_path.exists():
    st.error("Předpočtená data `heart_xgb_exercise_1_precomputed.json` nebyla nalezena. Spusťte nejprve výpočetní skript.")
    st.stop()

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

meta = data["metadata"]
head_10 = pd.DataFrame(data["head_10"])
dt_base = data["baseline_decision_tree"]
rf_base = data["baseline_random_forest"]
gs_info = data["grid_search"]
opt_model = data["optimal_model_test"]
rep_dict = opt_model["classification_report_dict"]
best_p = gs_info["best_params"]

# =============================================================================
# HORNÍ KPI KARTY
# =============================================================================
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric(
        "Testovací Precision (Nemoc)",
        f"{opt_model['precision'] * 100:.2f} %",
        delta=f"{(opt_model['precision'] - dt_base['precision']) * 100:+.2f} % vs Strom"
    )
with k2:
    st.metric(
        "Testovací Recall (Záchyt)",
        f"{opt_model['recall'] * 100:.2f} %",
        delta=f"{(opt_model['recall'] - dt_base['recall']) * 100:+.2f} % vs Strom"
    )
with k3:
    st.metric(
        "Testovací Accuracy",
        f"{opt_model['accuracy'] * 100:.2f} %",
        delta=f"{(opt_model['accuracy'] - dt_base['accuracy']) * 100:+.2f} % vs Strom"
    )
with k4:
    st.metric(
        "Vítězný XGBoost model",
        f"{best_p['n_estimators']} stromů",
        delta=f"depth={best_p['max_depth']}, gamma={best_p['gamma']}, lr={best_p['learning_rate']}"
    )

st.divider()

# =============================================================================
# KROK 1 & 2: NAČTENÍ DAT A STRUKTURA DATASETU
# =============================================================================
st.subheader("1. & 2. Načtení dat a kontrola prvních 10 pacientů (`head(10)`)")
st.markdown(
    r"""
    Dataset pacientů se srdečním onemocněním (`heart_data_normalized.csv`) obsahuje **303 pozorování** a **17 normalizovaných prediktorů**:
    - **Demografické a klinické ukazatele:** věk (`age`), pohlaví (`sex_male`), typ bolesti na hrudi (`chestpain_...`), klidový tlak (`restbp`), cholesterol (`chol`), lačný cukr (`fbs_yes`), klidové EKG (`restecg_...`), maximální tepová frekvence (`maxhr`), zátěžová angina (`exang_yes`).
    - **Cévní a zátěžové nálezy:** ST deprese (`oldpeak`), sklon ST segmentu (`slope`), počet zasažených cév fluoroskopií (`ca`), talasémie (`thal_normal`, `thal_reversable`).
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
# KROK 3 AŽ 7: GRIDSEARCHCV S XGBOOST MODELU
# =============================================================================
st.subheader("3.–7. Hyperparametrická optimalizace přes GridSearchCV (`scoring='precision'`)")
st.markdown(
    r"""
    Vytvořena instance `xgb_classifier = xgb.XGBClassifier(objective='binary:logistic')` a prohledána mřížka 
    hyperparametrů dle zadání kurzu:
    """
)

col_grid_code, col_grid_best = st.columns([1.1, 0.9])
with col_grid_code:
    st.code(
        """
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

xgb_classifier = xgb.XGBClassifier(
    objective='binary:logistic',
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)

params = {
    'max_depth': [3, 5, 7],
    'n_estimators': [50, 100, 150],
    'gamma': [0, 0.5, 1.0],
    'objective': ['binary:logistic'],
    'learning_rate': [0.05, 0.1, 0.2]
}

grid_search = GridSearchCV(
    estimator=xgb_classifier,
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
        - **`max_depth`:** `{best_p['max_depth']}`
        - **`n_estimators`:** `{best_p['n_estimators']}`
        - **`gamma`:** `{best_p['gamma']}` (silná regularizace tlumící zbytečné štěpení)
        - **`learning_rate`:** `{best_p['learning_rate']}`
        - **`objective`:** `{best_p['objective']}`
        - **Validační CV Precision (5-fold):** **`{gs_info['best_cv_precision'] * 100:.2f} %`**
        """
    )
    st.info("GridSearchCV otestoval celkem **81 kombinací** hyperparametrů v 405 kolech křížové validace.")

st.markdown("##### 📋 Přehled nejlepších výsledků křížové validace (Top 10):")
df_top_xgb = pd.DataFrame(gs_info["top_results"]).head(10)
df_top_xgb.rename(columns={
    "param_max_depth": "max_depth",
    "param_n_estimators": "n_estimators",
    "param_gamma": "gamma",
    "param_learning_rate": "learning_rate",
    "mean_test_precision": "Mean CV Precision",
    "std_test_precision": "Std CV Precision",
    "rank_test_score": "Pořadí"
}, inplace=True)
df_top_xgb["Mean CV Precision"] = df_top_xgb["Mean CV Precision"].apply(lambda x: f"{x * 100:.2f} %")
df_top_xgb["Std CV Precision"] = df_top_xgb["Std CV Precision"].apply(lambda x: f"± {x * 100:.2f} %")
st.dataframe(df_top_xgb, width="stretch", hide_index=True)

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
        color_continuous_scale="Oranges"
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
        st.metric("Precision (Požadavek zadání)", f"{opt_model['precision'] * 100:.2f} %", help="35 / (35 + 8) = 81.40 %")
    with mc2:
        st.metric("Recall (Záchyt)", f"{opt_model['recall'] * 100:.2f} %", help="35 / (35 + 8) = 81.40 %")
    with mc3:
        st.metric("F1-Score", f"{opt_model['f1']:.4f}")

st.divider()

# =============================================================================
# VELKÉ SROVNÁNÍ: ROZHODOVACÍ STROM VS. RANDOM FOREST VS. XGBOOST
# =============================================================================
st.subheader("📊 Velký souboj na datech srdce: Decision Tree vs. Random Forest vs. XGBoost")

df_battle = pd.DataFrame({
    "Metrika": ["Testovací Accuracy", "Testovací Precision (Nemoc)", "Testovací Recall (Záchyt)", "Testovací F1-Score"],
    "Samostatný strom CART": [dt_base["accuracy"] * 100, dt_base["precision"] * 100, dt_base["recall"] * 100, dt_base["f1"] * 100],
    "Random Forest (Bagging)": [rf_base["accuracy"] * 100, rf_base["precision"] * 100, rf_base["recall"] * 100, rf_base["f1"] * 100],
    "XGBoost (Boosting)": [opt_model["accuracy"] * 100, opt_model["precision"] * 100, opt_model["recall"] * 100, opt_model["f1"] * 100]
})

fig_battle = go.Figure()
fig_battle.add_trace(go.Bar(
    x=df_battle["Metrika"],
    y=df_battle["Samostatný strom CART"],
    name="Samostatný strom (Overfitting)",
    marker_color="#ef4444"
))
fig_battle.add_trace(go.Bar(
    x=df_battle["Metrika"],
    y=df_battle["Random Forest (Bagging)"],
    name="Random Forest (1. cvičení)",
    marker_color="#3b82f6"
))
fig_battle.add_trace(go.Bar(
    x=df_battle["Metrika"],
    y=df_battle["XGBoost (Boosting)"],
    name="XGBoost (Tento model)",
    marker_color="#f97316"
))
fig_battle.update_layout(
    barmode="group",
    height=360,
    margin=dict(l=30, r=30, t=35, b=30),
    yaxis_title="Hodnota metriky (%)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_battle, width="stretch")

st.info(
    "💡 **Pozorování ze srovnání:** Na takto malém datasetu (212 trénovacích pacientů) dosahuje **Random Forest (85.71 %)** o něco vyšší "
    "přesnosti než **XGBoost (81.40 %)**. Je to přirozené: paralelní Bagging těží z masivní redukce rozptylu průměrováním, "
    "zatímco sekvenční Boosting je na malých datech citlivější na náhodné fluktuace a šum."
)
