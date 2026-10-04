import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# 1. Načtení předpočtených dat
base_dir = Path(__file__).resolve().parent.parent
json_path = base_dir / "03_Advanced_ML_Neural_Networks" / "data" / "diamonds_rf_exercise_2_precomputed.json"

st.title("🎯 Cvičení 2: Ceny diamantů – Random Forest Regrese")
st.caption("Splnění všech kroků zadání kurzu: Načtení diamonds_preprocessed.csv, split 70:30, RandomizedSearchCV s metrikou mean_squared_error, nejlepší hyperparametry a výpočet testovací MAE.")

if not json_path.exists():
    st.error("Předpočtená data `diamonds_rf_exercise_2_precomputed.json` nebyla nalezena. Spusťte nejprve výpočetní skript.")
    st.stop()

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

meta = data["metadata"]
head_10 = pd.DataFrame(data["head_10"])
dt_base = data["baseline_decision_tree"]
rs_info = data["random_search"]
opt_model = data["optimal_model_test"]
samples = pd.DataFrame(data["sample_predictions"])
best_p = rs_info["best_params"]

# =============================================================================
# HORNÍ KPI KARTY
# =============================================================================
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric(
        "Testovací MAE (Chyba)",
        f"{opt_model['mae']:.2f} USD",
        delta=f"-{opt_model['improvement_mae_usd']:.2f} USD (-{opt_model['improvement_mae_pct']:.1f} %)",
        delta_color="normal"
    )
with k2:
    st.metric(
        "Testovací RMSE",
        f"{opt_model['rmse']:.2f} USD",
        delta=f"-{dt_base['rmse'] - opt_model['rmse']:.2f} USD vs Strom",
        delta_color="normal"
    )
with k3:
    st.metric(
        "Koeficient determinace (R²)",
        f"{opt_model['r2'] * 100:.2f} %",
        delta=f"{(opt_model['r2'] - dt_base['r2']) * 100:+.2f} % vs Strom"
    )
with k4:
    st.metric(
        "Vítězný les (Random Search)",
        f"{best_p['n_estimators']} stromů",
        delta=f"max_depth={best_p['max_depth']}, leaf={best_p['min_samples_leaf']}"
    )

st.divider()

# =============================================================================
# KROK 1 & 2: NAČTENÍ DAT A STRUKTURA DATASETU
# =============================================================================
st.subheader("1. & 2. Načtení dat a kontrola prvních 10 diamantů (`head(10)`)")
st.markdown(
    r"""
    Předzpracovaný dataset diamantů (`diamonds_preprocessed.csv`) obsahuje **53 899 pozorování** a **9 prediktorů**:
    - **Fyzické rozměry a hmotnost:** `carat` (hmotnost), `depth` (procentuální hloubka), `table` (šířka tabulky), `x`, `y`, `z` (rozměry v mm).
    - **Kvalitativní kódované znaky:** `cut` (kvalita brusu: 0–4), `color` (barva: 0–6), `clarity` (čistota: 0–7).
    - **Cílová proměnná `price`:** tržní cena v amerických dolarech (USD) s rozsahem od 326 USD do 18 823 USD.
    """
)
st.dataframe(head_10, width="stretch", hide_index=True)
st.caption(
    f"Celkem: {meta['dataset_shape'][0]:,} diamantů | Trénovací sada (70 %): {meta['train_shape'][0]:,} | "
    f"Testovací sada (30 %): {meta['test_shape'][0]:,} (`random_state=42`)."
)

st.divider()

# =============================================================================
# KROK 3 AŽ 7: RANDOMIZEDSEARCHCV OPTIMALIZACE
# =============================================================================
st.subheader("3.–7. Hyperparametrická optimalizace přes RandomizedSearchCV (`scoring='mean_squared_error'`)")
st.markdown(
    r"""
    Na 37 729 trénovacích pozorováních byl aplikován `RandomizedSearchCV` s 3násobnou křížovou validací (CV=3), 
    který náhodně vzorkoval kombinace z definovaného prostoru hyperparametrů:
    """
)

col_rs_code, col_rs_best = st.columns([1.1, 0.9])
with col_rs_code:
    st.code(
        """
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

rf_regressor = RandomForestRegressor(random_state=42, n_jobs=-1)

params = {
    'max_depth': [10, 15, 20, None],
    'min_samples_leaf': [1, 2, 4],
    'n_estimators': [50, 100, 150]
}

random_search = RandomizedSearchCV(
    estimator=rf_regressor,
    param_distributions=params,
    scoring='neg_mean_squared_error',
    n_iter=8,
    cv=3,
    random_state=42,
    n_jobs=-1
)
random_search.fit(X_train, y_train)
        """,
        language="python"
    )

with col_rs_best:
    st.markdown("#### 🏆 Vítězná konfigurace hyperparametrů:")
    st.success(
        f"""
        - **`max_depth`:** `{best_p['max_depth']}`
        - **`min_samples_leaf`:** `{best_p['min_samples_leaf']}`
        - **`n_estimators`:** `{best_p['n_estimators']}`
        - **Nejlepší CV RMSE (křížová validace):** **`{rs_info['best_cv_rmse']:.2f} USD`**
        """
    )
    st.info("RandomizedSearchCV prozkoumal 8 náhodných kombinací parametrů za necelých 20 sekund díky paralelnímu běhu (`n_jobs=-1`).")

st.markdown("##### 📋 Přehled prozkoumaných kombinací hyperparametrů:")
df_rs_table = pd.DataFrame(rs_info["top_results"])
df_rs_table.rename(columns={
    "param_max_depth": "max_depth",
    "param_min_samples_leaf": "min_samples_leaf",
    "param_n_estimators": "n_estimators",
    "mean_cv_rmse": "CV RMSE (USD)",
    "rank_test_score": "Pořadí"
}, inplace=True)
st.dataframe(df_rs_table, width="stretch", hide_index=True)

st.divider()

# =============================================================================
# KROK 8 & 9: VÝSLEDKY NA TESTOVACÍ SADĚ A VÝPOČET MAE
# =============================================================================
st.subheader("8. & 9. Evaluace na testovací sadě: Výpočet průměrné absolutní chyby (MAE)")

c_res1, c_res2, c_res3 = st.columns(3)
with c_res1:
    st.metric("Průměrná absolutní chyba (MAE)", f"{opt_model['mae']:.2f} USD", help="Požadovaná metrika zadání")
with c_res2:
    st.metric("Kvadratická chyba (RMSE)", f"{opt_model['rmse']:.2f} USD")
with c_res3:
    st.metric("Vysvětlený rozptyl (R²)", f"{opt_model['r2'] * 100:.2f} %")

st.markdown("---")
st.subheader("📊 Velké historické srovnání modelů na datasetu diamantů")
st.caption("Srovnání modelů vyzkoušených napříč kurzem na stejných testovacích datech diamantů:")

df_hist = pd.DataFrame({
    "Model": ["Lineární regrese OLS (Den 1)", "Polynomiální regrese stupeň 2 (Den 1)", "Rozhodovací strom CART (Den 1)", "Random Forest (Den 3)"],
    "MAE (USD)": [740.0, 500.0, dt_base["mae"], opt_model["mae"]],
    "RMSE (USD)": [1120.0, 890.0, dt_base["rmse"], opt_model["rmse"]],
    "R² skóre (%)": [92.0, 95.0, dt_base["r2"] * 100, opt_model["r2"] * 100]
})

fig_comp_diam = go.Figure()
fig_comp_diam.add_trace(go.Bar(
    x=df_hist["Model"],
    y=df_hist["MAE (USD)"],
    name="MAE (USD) – nižší je lepší",
    text=[f"{v:.1f} USD" for v in df_hist["MAE (USD)"]],
    textposition="auto",
    marker_color=["#94a3b8", "#64748b", "#ef4444", "#10b981"]
))
fig_comp_diam.update_layout(
    title="Vývoj střední absolutní chyby (MAE) napříč modely diamantů",
    yaxis_title="Střední absolutní chyba MAE (USD)",
    height=340,
    margin=dict(l=30, r=30, t=35, b=30)
)
st.plotly_chart(fig_comp_diam, width="stretch")

st.divider()

# =============================================================================
# UKÁZKA PREDIKCÍ NA 10 TESTOVACÍCH DIAMANTECH
# =============================================================================
st.subheader("💎 Ukázka predikcí na 10 náhodných testovacích diamantech")
st.caption("Porovnání skutečné tržní ceny, predikce samostatného stromu a predikce Random Forestu:")
st.dataframe(samples, width="stretch", hide_index=True)
st.info(
    f"💡 **Shrnutí výsledku cvičení:** Random Forest srazil průměrnou absolutní chybu na pouhých **{opt_model['mae']:.2f} USD** "
    f"(zlepšení o **{opt_model['improvement_mae_usd']:.2f} USD** neboli **{opt_model['improvement_mae_pct']:.1f} %** oproti jednomu stromu). "
    f"Model vysvětluje **{opt_model['r2']*100:.2f} %** veškeré cenové variability diamantů."
)
