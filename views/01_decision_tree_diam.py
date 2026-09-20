from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.title("💎 Cvičení 9: Rozhodovací strom v regresi (Ceny diamantů)")
st.caption("Praktická aplikace DecisionTreeRegressor na diamonds.csv: Rychlostní wrapper funkce, iterativní ladění hyperparametrů, boj s přetrénováním a srovnání s polynomem.")

# Načtení dat diamonds.csv s ordonálním zakódováním
@st.cache_data
def load_diamonds_data():
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "01_Regression" / "data" / "diamonds.csv"
    df = pd.read_csv(data_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Ordinal encoding standardních stupnic 4Cs
    cut_map = {"Fair": 1, "Good": 2, "Very Good": 3, "Premium": 4, "Ideal": 5}
    color_map = {"J": 1, "I": 2, "H": 3, "G": 4, "F": 5, "E": 6, "D": 7}
    clarity_map = {"I1": 1, "SI2": 2, "SI1": 3, "VS2": 4, "VS1": 5, "VVS2": 6, "VVS1": 7, "IF": 8}

    df["cut"] = df["cut"].map(cut_map).fillna(3).astype(int)
    df["color"] = df["color"].map(color_map).fillna(4).astype(int)
    df["clarity"] = df["clarity"].map(clarity_map).fillna(3).astype(int)

    # Odstranění fyzikálně nemožných rozměrů
    df = df[(df["x"] > 0) & (df["y"] > 0) & (df["z"] > 0) & (df["y"] < 30) & (df["z"] < 30)].copy()
    return df

df_diam = load_diamonds_data()

tab_summary, tab_sim, tab_feat = st.tabs([
    "📋 Výsledky 4 iterací zadání",
    "🧪 Interaktivní laboratoř stromu",
    "📊 Důležitost příznaků (Feature Importance)"
])

# =============================================================================
# TAB 1: SOUHRN VŠECH 4 ITERACÍ
# =============================================================================
with tab_summary:
    st.markdown("### 📈 Postupná evoluce modelu: Od 36 000 listů k elegantnímu optimu")
    st.markdown(r"""
    Zadání vyžaduje sestavit wrapper funkci `train_and_evaluate_tree(...)` a **opakovat ladění hyperparametrů, dokud nezískáme optimální model**.  
    Zde je detailní srovnání 4 klíčových iterací na datasetu diamantů:
    """)

    # Předem spočtené benchmarkové metriky
    summary_data = [
        {
            "Iterace / Model": "1. Výchozí neomezený strom (Baseline)",
            "Hyperparametry": "max_depth=None, leaf=1 (Default)",
            "Trénovací R²": 1.0000,
            "Testovací R²": 0.9681,
            "Overfitting Gap": 0.0319,
            "Test MAE (USD)": 353,
            "Test RMSE (USD)": 715,
            "Hloubka": 36,
            "Počet listů": 36792,
            "Hodnocení": "❌ Gigantický přeučený strom (36k listů)"
        },
        {
            "Iterace / Model": "2. Bezpečné prořezání (Pruned)",
            "Hyperparametry": "max_depth=6, min_leaf=20",
            "Trénovací R²": 0.9539,
            "Testovací R²": 0.9559,
            "Overfitting Gap": -0.0020,
            "Test MAE (USD)": 516,
            "Test RMSE (USD)": 841,
            "Hloubka": 6,
            "Počet listů": 63,
            "Hodnocení": "⚠️ Velmi rychlý, ale příliš zjednodušený"
        },
        {
            "Iterace / Model": "3. Vyvážený strom (Balanced)",
            "Hyperparametry": "max_depth=10, min_leaf=10",
            "Trénovací R²": 0.9800,
            "Testovací R²": 0.9770,
            "Overfitting Gap": 0.0030,
            "Test MAE (USD)": 321,
            "Test RMSE (USD)": 607,
            "Hloubka": 10,
            "Počet listů": 712,
            "Hodnocení": "✨ Výborná generalizace"
        },
        {
            "Iterace / Model": "4. Optimální strom (GridSearchCV)",
            "Hyperparametry": "max_depth=13, min_leaf=6, split=25",
            "Trénovací R²": 0.9846,
            "Testovací R²": 0.9787,
            "Overfitting Gap": 0.0059,
            "Test MAE (USD)": 308,
            "Test RMSE (USD)": 584,
            "Hloubka": 13,
            "Počet listů": 1840,
            "Hodnocení": "🏆 Globální optimum (R² = 97.9%)"
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

    # 4 KPI karty pro Model 4 (GridSearch Optimum)
    st.markdown("#### 🏆 Metriky vítězného modelu (Model 4 z GridSearchCV):")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Testovací R²", "0.9787", delta="+0.0106 vs. Baseline")
    with k2:
        st.metric("Testovací RMSE", "584.0 USD", delta="-131.5 USD vs. Baseline", delta_color="inverse")
    with k3:
        st.metric("Testovací MAE", "307.5 USD", delta="-45.5 USD vs. Baseline", delta_color="inverse")
    with k4:
        st.metric("Redukce počtu listů", "1 840 vs. 36 792", delta="-95.0 % uzlů", delta_color="normal")

    st.info(
        "💡 **Klíčové zjištění úlohy:**  \n"
        "Výchozí strom dosáhl sice slušného $R^2 = 0.9681$, ale vytvořil **36 792 listů** pro 43 134 trénovacích vzorků. "
        "V průměru měl každý list pouze 1.17 diamantu!  \n"
        "Po optimalizaci hyperparametrů na `max_depth=13, min_samples_leaf=6, min_samples_split=25` klesl počet listů na pouhých **1 840**, "
        "zatímco chyba předpovědi **RMSE klesla ze 715.5 USD na 584.0 USD** a $R^2$ stouplo na rekordních **97.87 %**!"
    )

# =============================================================================
# TAB 2: INTERAKTIVNÍ SIMULÁTOR STROMU NA DIAMANTECH
# =============================================================================
with tab_sim:
    st.markdown("### 🧪 Živá simulace: Wrapper funkce v praxi")
    st.markdown("Vyzkoušejte si, jak změna hyperparametrů přímo ovlivňuje tvar predikcí a velikost stromu:")

    with st.container(border=True):
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            sel_depth = st.slider("Maximální hloubka stromu (max_depth):", min_value=2, max_value=20, value=10)
        with col_s2:
            sel_leaf = st.slider("Minimální vzorky v listu (min_samples_leaf):", min_value=1, max_value=50, value=10)
        with col_s3:
            sel_split = st.slider("Minimální vzorky pro dělení (min_samples_split):", min_value=2, max_value=80, value=20)

    # Příprava dat pro živý trénink
    X_live = df_diam.drop(columns=["price"])
    y_live = df_diam["price"]
    X_tr, X_te, y_tr, y_te = train_test_split(X_live, y_live, test_size=0.20, random_state=42)

    # Živý trénink rychlého modelu
    tree_live = DecisionTreeRegressor(
        max_depth=sel_depth,
        min_samples_leaf=sel_leaf,
        min_samples_split=sel_split,
        random_state=42
    )
    tree_live.fit(X_tr, y_tr)

    p_tr = tree_live.predict(X_tr)
    p_te = tree_live.predict(X_te)

    live_tr_r2 = r2_score(y_tr, p_tr)
    live_te_r2 = r2_score(y_te, p_te)
    live_te_mae = mean_absolute_error(y_te, p_te)
    live_te_mse = mean_squared_error(y_te, p_te)
    live_te_rmse = np.sqrt(live_te_mse)
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
        st.metric("Testovací RMSE", f"{live_te_rmse:,.1f} USD")
    with m_col5:
        st.metric("Overfitting Gap", f"{gap:.4f}", delta="Přeučeno" if gap > 0.05 else "Skvělé", delta_color="inverse" if gap > 0.05 else "normal")

    # Scatter graf skutečnost vs. predikce (vzorek 1000 bodů z testu)
    sample_idx = np.random.RandomState(42).choice(len(y_te), size=min(1000, len(y_te)), replace=False)
    y_sample_true = y_te.iloc[sample_idx]
    y_sample_pred = p_te[sample_idx]

    fig_scat = go.Figure()
    fig_scat.add_trace(go.Scatter(
        x=y_sample_true, y=y_sample_pred,
        mode="markers", name="Testovací diamanty (vzorek 1 000)",
        marker=dict(size=5, color="#10B981", opacity=0.6, line=dict(color="#059669", width=0.5))
    ))
    fig_scat.add_trace(go.Scatter(
        x=[0, 19000], y=[0, 19000],
        mode="lines", name="Ideální predikce (y = ŷ)",
        line=dict(color="#EF4444", width=2, dash="dash")
    ))

    fig_scat.update_layout(
        title=f"Skutečné vs. Predikované ceny diamantů (max_depth={sel_depth}, min_samples_leaf={sel_leaf})",
        xaxis_title="Skutečná cena diamantu (USD)",
        yaxis_title="Predikovaná cena stromem (USD)",
        xaxis=dict(range=[0, 19000]),
        yaxis=dict(range=[0, 19000]),
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_scat, width="stretch")

# =============================================================================
# TAB 3: FEATURE IMPORTANCE
# =============================================================================
with tab_feat:
    st.markdown("### 📊 Které vlastnosti diamantu určují jeho cenu nejvíce?")
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
            colorscale="Viridis",
            line=dict(color="#047857", width=1)
        )
    ))
    fig_feat.update_layout(
        title="Důležitost příznaků (Feature Importance) v rozhodovacím stromu",
        xaxis_title="Relativní důležitost (MDI, součet = 1.0)",
        yaxis_title="Příznak diamantu",
        height=480
    )
    st.plotly_chart(fig_feat, width="stretch")

    st.success(
        "🧠 **Dominantní faktory:** "
        "1. **`y` (šířka) a `carat` (hmotnost)** představují dohromady přes **75 % veškeré predikční síly**. "
        "2. **`clarity` (čistota)** a **`color` (barva)** jsou dalšími významnými korektory ceny. "
        "3. Příznaky jako `cut`, `depth` a `table` mají naopak minimální vliv na konečnou cenu surového diamantu."
    )
