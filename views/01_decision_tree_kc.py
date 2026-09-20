from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.title("🎯 Cvičení 8: Rozhodovací strom v regresi (Nemovitosti King County)")
st.caption("Praktická aplikace DecisionTreeRegressor na surová data kc_house_data.csv: Boj s přetrénováním, ladění hyperparametrů a nalezení optimální rovnováhy.")

# Načtení dat (s cache pro bleskový běh)
@st.cache_data
def load_kc_data():
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "01_Regression" / "data" / "kc_house_data.csv"
    df = pd.read_csv(data_path)
    df_clean = df.drop(columns=["id"])
    if "date" in df_clean.columns:
        df_clean["yr_sold"] = df_clean["date"].str[:4].astype(int)
        df_clean["month_sold"] = df_clean["date"].str[4:6].astype(int)
        df_clean = df_clean.drop(columns=["date"])
    return df_clean

df_kc = load_kc_data()

tab_summary, tab_sim, tab_feat = st.tabs([
    "📋 Výsledky 4 iterací zadání",
    "🧪 Interaktivní laboratoř stromu",
    "📊 Důležitost příznaků (Feature Importance)"
])

# =============================================================================
# TAB 1: SOUHRN VŠECH 4 ITERACÍ
# =============================================================================
with tab_summary:
    st.markdown("### 📈 Postupná evoluce modelu: Od přeučeného stromu k optimu")
    st.markdown(r"""
    Zadání vyžaduje inicializovat výchozí model, vyhodnotit metriky a následně **opakovat kroky ladění, dokud nezískáme optimální model**.  
    Zde je přesné srovnání 4 klíčových vývojových fází:
    """)

    # Předem spočtené benchmarkové metriky
    summary_data = [
        {
            "Fáze / Model": "1. Výchozí neomezený strom (Baseline)",
            "Hyperparametry": "max_depth=None, leaf=1 (Default)",
            "Trénovací R²": 1.0000,
            "Testovací R²": 0.7048,
            "Overfitting Gap": 0.2952,
            "Test MAE (USD)": 104496,
            "Test RMSE (USD)": 211262,
            "Hloubka": 35,
            "Počet listů": 16683,
            "Hodnocení": "❌ Extrémní přeučení"
        },
        {
            "Fáze / Model": "2. Manuální prořezání",
            "Hyperparametry": "max_depth=8, min_leaf=20, split=40",
            "Trénovací R²": 0.8053,
            "Testovací R²": 0.7318,
            "Overfitting Gap": 0.0735,
            "Test MAE (USD)": 104844,
            "Test RMSE (USD)": 201368,
            "Hloubka": 8,
            "Počet listů": 145,
            "Hodnocení": "⚠️ Velký pokles variance"
        },
        {
            "Fáze / Model": "3. Optimální strom (GridSearchCV)",
            "Hyperparametry": "max_depth=14, min_leaf=5, split=40",
            "Trénovací R²": 0.8889,
            "Testovací R²": 0.7932,
            "Overfitting Gap": 0.0957,
            "Test MAE (USD)": 90502,
            "Test RMSE (USD)": 176810,
            "Hloubka": 14,
            "Počet listů": 1240,
            "Hodnocení": "✅ Globální optimum"
        },
        {
            "Fáze / Model": "4. Optimální strom + Feature Engineering",
            "Hyperparametry": "max_depth=14 + house_age, is_renovated",
            "Trénovací R²": 0.8897,
            "Testovací R²": 0.7959,
            "Overfitting Gap": 0.0938,
            "Test MAE (USD)": 90118,
            "Test RMSE (USD)": 175647,
            "Hloubka": 14,
            "Počet listů": 1285,
            "Hodnocení": "🏆 Nejlepší generalizace"
        }
    ]
    df_summary = pd.DataFrame(summary_data)

    st.dataframe(
        df_summary.style.format({
            "Trénovací R²": "{:.4f}",
            "Testovací R²": "{:.4f}",
            "Overfitting Gap": "{:.4f}",
            "Test MAE (USD)": "{:,.0f}",
            "Test RMSE (USD)": "{:,.0f}",
            "Počet listů": "{:,}"
        }),
        width="stretch"
    )

    # 4 KPI karty pro Model 3 (GridSearch Optimum)
    st.markdown("#### 🏆 Metriky vítězného modelu (Model 3 z GridSearchCV):")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Testovací R²", "0.7932", delta="+0.0884 vs. Baseline")
    with k2:
        st.metric("Testovací RMSE", "176 810 USD", delta="-34 452 USD vs. Baseline", delta_color="inverse")
    with k3:
        st.metric("Testovací MAE", "90 502 USD", delta="-13 994 USD vs. Baseline", delta_color="inverse")
    with k4:
        st.metric("Overfitting Gap (Δ R²)", "0.0957", delta="-0.1995 vs. Baseline", delta_color="inverse")

    st.info(
        "💡 **Klíčový poznatek z iterací:** "
        "Neomezený strom si data zapamatoval do hloubky 35 pater s 16 683 listy (téměř v každém listu byl jediný dům). "
        "Teprve omezením `min_samples_split=40` a `min_samples_leaf=5` strom přestal reagovat na jednotlivé extrémy "
        "a vytvořil robustní zóny, čímž **snížil chybu předpovědi o 34 500 USD na každé nemovitosti**."
    )

# =============================================================================
# TAB 2: INTERAKTIVNÍ SIMULÁTOR STROMU NA KING COUNTY
# =============================================================================
with tab_sim:
    st.markdown("### 🧪 Živá simulace: Vyzkoušejte si vliv hyperparametrů na reálných datech")

    with st.container(border=True):
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            sel_depth = st.slider("Maximální hloubka stromu (max_depth):", min_value=2, max_value=22, value=12)
        with col_s2:
            sel_leaf = st.slider("Minimální počet vzorků v listu (min_samples_leaf):", min_value=1, max_value=50, value=10)

    # Příprava dat pro živý trénink
    X_live = df_kc.drop(columns=["price"])
    y_live = df_kc["price"]
    X_tr, X_te, y_tr, y_te = train_test_split(X_live, y_live, test_size=0.20, random_state=42)

    # Živý trénink rychlého modelu
    tree_live = DecisionTreeRegressor(
        max_depth=sel_depth,
        min_samples_leaf=sel_leaf,
        min_samples_split=max(sel_leaf * 2, 10),
        random_state=42
    )
    tree_live.fit(X_tr, y_tr)

    p_tr = tree_live.predict(X_tr)
    p_te = tree_live.predict(X_te)

    live_tr_r2 = r2_score(y_tr, p_tr)
    live_te_r2 = r2_score(y_te, p_te)
    live_te_mae = mean_absolute_error(y_te, p_te)
    live_te_rmse = np.sqrt(mean_squared_error(y_te, p_te))
    gap = live_tr_r2 - live_te_r2

    # Metriky
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    with m_col1:
        st.metric("Počet listů", f"{tree_live.get_n_leaves():,}")
    with m_col2:
        st.metric("Trénovací R²", f"{live_tr_r2:.4f}")
    with m_col3:
        st.metric("Testovací R²", f"{live_te_r2:.4f}")
    with m_col4:
        st.metric("Testovací MAE", f"{live_te_mae:,.0f} USD")
    with m_col5:
        st.metric("Overfitting Gap", f"{gap:.4f}", delta="Přeučeno" if gap > 0.15 else "Vyváženo", delta_color="inverse" if gap > 0.15 else "normal")

    # Scatter graf skutečnost vs. predikce (pro přehlednost vzorek 1000 bodů z testu)
    sample_idx = np.random.RandomState(42).choice(len(y_te), size=min(1000, len(y_te)), replace=False)
    y_sample_true = y_te.iloc[sample_idx]
    y_sample_pred = p_te[sample_idx]

    fig_scat = go.Figure()
    fig_scat.add_trace(go.Scatter(
        x=y_sample_true, y=y_sample_pred,
        mode="markers", name="Testovací nemovitosti",
        marker=dict(size=6, color="#38BDF8", opacity=0.6, line=dict(color="#0284C7", width=0.5))
    ))
    fig_scat.add_trace(go.Scatter(
        x=[0, 3500000], y=[0, 3500000],
        mode="lines", name="Ideální predikce (y = ŷ)",
        line=dict(color="#EF4444", width=2, dash="dash")
    ))

    fig_scat.update_layout(
        title=f"Skutečné vs. Predikované ceny nemovitostí (max_depth={sel_depth}, min_samples_leaf={sel_leaf})",
        xaxis_title="Skutečná cena nemovitosti (USD)",
        yaxis_title="Predikovaná cena stromem (USD)",
        xaxis=dict(range=[0, 3500000]),
        yaxis=dict(range=[0, 3500000]),
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_scat, width="stretch")

# =============================================================================
# TAB 3: FEATURE IMPORTANCE
# =============================================================================
with tab_feat:
    st.markdown("### 📊 Které příznaky rozhodují o ceně v King County nejvíce?")
    st.markdown(r"""
    Rozhodovací strom měří důležitost příznaků pomocí **MDI (Mean Decrease in Impurity)** – 
    součtu poklesu kvadratické chyby MSE napříč všemi uzly, kde byl daný příznak vybrán k dělení.
    """)

    feat_series = pd.Series(tree_live.feature_importances_, index=X_live.columns).sort_values(ascending=True)

    fig_feat = go.Figure(go.Bar(
        x=feat_series.values,
        y=feat_series.index,
        orientation="h",
        marker=dict(
            color=feat_series.values,
            colorscale="Blues",
            line=dict(color="#0284C7", width=1)
        )
    ))
    fig_feat.update_layout(
        title="Důležitost příznaků (Feature Importance) v rozhodovacím stromu",
        xaxis_title="Relativní důležitost (MDI, součet = 1.0)",
        yaxis_title="Příznak nemovitosti",
        height=540
    )
    st.plotly_chart(fig_feat, width="stretch")

    st.success(
        "🧠 **Dominantní faktory:** "
        "1. **`sqft_living` & `grade`** tvoří přes 50 % celkového rozhodování. "
        "2. **`lat` (zeměpisná šířka)** je 3. nejdůležitějším prediktorem – odráží bonitní severní a pobřežní části Seattlu/Bellevue. "
        "Právě díky schopnosti stromu dělit prostor podle souřadnic dosahuje strom výrazně lepšího výsledku než základní lineární regrese!"
    )
