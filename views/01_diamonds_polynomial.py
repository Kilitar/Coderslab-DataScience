import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline

st.title("🔷 Cvičení 7: Polynomiální regrese (Diamonds Dataset)")
st.caption("Analýza nelineárních závislostí a interakcí mezi 4C parametry diamantu. Srovnání stupně 1 (OLS), stupně 2 (Kvadratický), stupně 3 (Kubický) a regularizovaného polynomu Ridge.")

# =============================================================================
# CACHE VÝPOČTŮ (OKAMŽITÉ NAČTENÍ Z PŘEDPOČÍTANÝCH DAT)
# =============================================================================
@st.cache_data
def load_diamonds_poly_precomputed():
    base_dir = Path(__file__).resolve().parent.parent
    precomputed_path = base_dir / "01_Regression" / "data" / "diamonds_poly_precomputed.json"

    with open(precomputed_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

@st.cache_resource
def get_trained_pipelines():
    """Rychlý cache modelů pro interaktivní kalkulátor ceny diamantu (< 0.3s)."""
    base_dir = Path(__file__).resolve().parent.parent
    csv_path = base_dir / "01_Regression" / "data" / "diamonds_preprocessed.csv"
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["price"])
    y = df["price"]

    p1 = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LinearRegression())
    ]).fit(X, y)

    p2 = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("lr", LinearRegression())
    ]).fit(X, y)

    p3_ols = Pipeline([
        ("poly", PolynomialFeatures(degree=3, include_bias=False)),
        ("scaler", StandardScaler()),
        ("lr", LinearRegression())
    ]).fit(X, y)

    p3_ridge = Pipeline([
        ("poly", PolynomialFeatures(degree=3, include_bias=False)),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=100.0, random_state=42))
    ]).fit(X, y)

    return p1, p2, p3_ols, p3_ridge, list(X.columns)

data = load_diamonds_poly_precomputed()
metrics_table = data["metrics_table"]
top_features = data["top_features_deg2"]
sample_data = data["sample_plot_data"]

# =============================================================================
# TOP KPI METRICS
# =============================================================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="🏆 Vítězný model",
        value="Stupeň 2 (Kvadratický)",
        delta="54 příznaků"
    )

with col2:
    st.metric(
        label="Testovací R²",
        value=f"{data['winner_r2']:.4f}",
        delta="+0.0560 vs OLS"
    )

with col3:
    st.metric(
        label="Testovací MAE",
        value=f"{data['winner_mae']:.1f} USD",
        delta="-43.3 % chyba vs OLS",
        delta_color="inverse"
    )

with col4:
    st.metric(
        label="Testovací RMSE",
        value=f"{data['winner_rmse']:.1f} USD",
        delta="-38.3 % chyba vs OLS",
        delta_color="inverse"
    )

st.markdown("---")

# =============================================================================
# ZÁLOŽKY
# =============================================================================
t1, t2, t3, t4 = st.tabs([
    "📊 Srovnání stupňů polynomu",
    "🔍 Nejdůležitější interakce (Stupeň 2)",
    "💎 Interaktivní kalkulátor ceny diamantu",
    "💡 Odpovědi na zadání cvičení"
])

# -----------------------------------------------------------------------------
# TAB 1: SROVNÁNÍ MODELŮ
# -----------------------------------------------------------------------------
with t1:
    st.markdown("### 📋 Porovnání modelů na testovací sadě (Test Split)")

    df_metrics = pd.DataFrame(metrics_table)
    df_metrics["MSE_Test"] = df_metrics["Test_RMSE"] ** 2

    disp_df = df_metrics[[
        "Name", "Degree", "Features_Count",
        "Train_R2", "Test_R2", "Overfitting_Delta_R2",
        "Test_MAE", "Test_RMSE", "MSE_Test", "Overfitting_Risk"
    ]].rename(columns={
        "Name": "Model",
        "Degree": "Stupeň",
        "Features_Count": "Příznaky",
        "Train_R2": "Trénovací R²",
        "Test_R2": "Testovací R²",
        "Overfitting_Delta_R2": "Δ R² (Overfit Gap)",
        "Test_MAE": "Test MAE (USD)",
        "Test_RMSE": "Test RMSE (USD)",
        "MSE_Test": "Test MSE (USD²)",
        "Overfitting_Risk": "Stav přeučení"
    })
    disp_df["Stupeň"] = disp_df["Stupeň"].astype(str)

    st.dataframe(
        disp_df.style.format({
            "Trénovací R²": "{:.4f}",
            "Testovací R²": "{:.4f}",
            "Δ R² (Overfit Gap)": "{:+.4f}",
            "Test MAE (USD)": "{:,.2f} USD",
            "Test RMSE (USD)": "{:,.2f} USD",
            "Test MSE (USD²)": "{:,.0f}",
        }),
        width="stretch"
    )

    st.markdown("#### 🎯 Skutečná vs. Predikovaná cena (Ukázka testovací sady)")
    st.caption("Porovnání predikcí jednotlivých modelů vůči ideální diagonále ($y = \\hat{y}$).")

    sample_df = pd.DataFrame({
        "Skutečná cena (USD)": sample_data["actual"],
        "Stupeň 1 (Lineární OLS)": sample_data["pred_deg1"],
        "Stupeň 2 (Kvadratický OLS)": sample_data["pred_deg2"],
        "Stupeň 3 (Kubický OLS)": sample_data["pred_deg3"],
        "Stupeň 3 + Ridge (α=100)": sample_data["pred_ridge3"],
        "HistGradientBoosting": sample_data["pred_hgb"]
    })

    selected_model = st.selectbox(
        "Zvolte model k vykreslení:",
        options=[
            "Stupeň 2 (Kvadratický OLS)",
            "Stupeň 1 (Lineární OLS)",
            "Stupeň 3 (Kubický OLS)",
            "Stupeň 3 + Ridge (α=100)",
            "HistGradientBoosting"
        ],
        index=0
    )

    fig_scatter = px.scatter(
        sample_df,
        x="Skutečná cena (USD)",
        y=selected_model,
        opacity=0.6,
        color_discrete_sequence=["#1D3557" if "2" in selected_model else ("#E63946" if "3 (" in selected_model else "#2A9D8F")],
        title=f"Skutečná cena vs. Predikce: {selected_model}",
        labels={"x": "Skutečná cena (USD)", "y": "Predikce modelu (USD)"}
    )

    # Ideální diagonální linie
    max_val = max(sample_df["Skutečná cena (USD)"].max(), sample_df[selected_model].max())
    min_val = min(sample_df["Skutečná cena (USD)"].min(), sample_df[selected_model].min())
    fig_scatter.add_shape(
        type="line", line=dict(dash="dash", color="gray", width=2),
        x0=0, y0=0, x1=max_val, y1=max_val
    )

    fig_scatter.update_layout(height=520)
    st.plotly_chart(fig_scatter, width="stretch")

    if "3 (" in selected_model:
        st.warning(
            "⚠️ Všimněte si chování Kubického polynomu (Stupeň 3 OLS): zatímco většina bodů sedí těsně, "
            "v okrajových oblastech dochází k extrémním výkyvům (body vystřelují daleko od diagonály). "
            "To je přímý důsledek Rungeova jevu a kolinearity kubických mocnin!"
        )

# -----------------------------------------------------------------------------
# TAB 2: INTERAKCE STUPNĚ 2
# -----------------------------------------------------------------------------
with t2:
    st.markdown("### 🔍 Nejsilnější polynomiální členy a interakce (Stupeň 2)")
    st.markdown(r"""
    Kvadratický polynom rozšiřuje původních 9 příznaků na **54 proměnných** ($\binom{9+2}{2} = 54$).
    Níže je 15 nejvýznamnějších členů seřazených podle absolutní hodnoty standardizovaného regresního koeficientu $\beta$.
    """)

    top_df = pd.DataFrame(top_features)
    top_df["importance"] = top_df["coef"].abs()
    top_df["coefficient"] = top_df["coef"]

    fig_imp = px.bar(
        top_df.sort_values(by="importance", ascending=True),
        x="importance",
        y="feature",
        orientation="h",
        color="coefficient",
        color_continuous_scale="Blues",
        title="Top 15 nejdůležitějších členů kvadratického modelu",
        labels={"importance": "Absolutní váha (|β|)", "feature": "Polynomiální člen", "coefficient": "Koeficient β"},
        height=500
    )
    st.plotly_chart(fig_imp, width="stretch")

    st.markdown(r"""
    #### 💎 Gemologická interpretace výsledků:
    1. **`carat^2` (Kvadratický karát):** 
       - Cena diamantů neroste lineárně, ale podle mocninného zákona ("Indian Diamond Price Rule": $\text{Cena} \propto \text{Hmotnost}^2$). 
       - Dvoukarátový diamant je vzácnější než dva jednokarátové dohromady, proto je jeho cena více než dvojnásobná.
    2. **Interakce `carat * clarity` a `carat * color`:**
       - U malého diamantu (0.3 ct) nehraje barva a čistota tak dramatickou roli v absolutní ceně.
       - U velkého diamantu (2.0 ct) je však rozdíl mezi čistotou IF a SI2 v řádu desítek tisíc USD. Model stupně 2 tuto synergii zachycuje přesně díky součinu $x_i \cdot x_j$.
    3. **Interakce rozměrů `x * y` a `x * z`:**
       - Tvoří aproximaci objemu diamantu. Objem v kombinaci s karátem dává modelu informaci o hustotě a proporcích výbrusu.
    """)

# -----------------------------------------------------------------------------
# TAB 3: KALKULÁTOR CENY DIAMANTU
# -----------------------------------------------------------------------------
with t3:
    st.markdown("### 💎 Interaktivní kalkulátor ceny diamantu")
    st.caption("Nastavte parametry diamantu a porovnejte predikce lineárního OLS, kvadratického polynomu, kubického OLS a kubického Ridge.")

    p1, p2, p3_ols, p3_ridge, feat_cols = get_trained_pipelines()

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        carat_val = st.slider("Hmotnost v karátech (Carat)", min_value=0.2, max_value=3.5, value=1.0, step=0.05)
        cut_val = st.slider("Kvalita brusu (Cut: 1=Fair, 5=Ideal)", min_value=1, max_value=5, value=4, step=1)
        color_val = st.slider("Barva (Color: 1=J nejhorší, 7=D nejlepší)", min_value=1, max_value=7, value=5, step=1)

    with col_b:
        clarity_val = st.slider("Čistota (Clarity: 1=I1, 8=IF)", min_value=1, max_value=8, value=5, step=1)
        depth_val = st.slider("Hloubka v % (Depth)", min_value=55.0, max_value=70.0, value=61.8, step=0.1)
        table_val = st.slider("Šířka tabulky v % (Table)", min_value=50.0, max_value=70.0, value=57.0, step=0.5)

    with col_c:
        # Přibližné rozměry pro daný karát pro realistický vstup
        approx_dim = (carat_val * 140) ** (1/3)
        x_val = st.number_input("Délka x (mm)", min_value=3.0, max_value=10.5, value=float(np.round(approx_dim, 2)), step=0.1)
        y_val = st.number_input("Šířka y (mm)", min_value=3.0, max_value=10.5, value=float(np.round(approx_dim, 2)), step=0.1)
        z_val = st.number_input("Hloubka z (mm)", min_value=2.0, max_value=7.0, value=float(np.round(approx_dim * 0.62, 2)), step=0.1)

    # Sestavení vzorku
    input_row = pd.DataFrame([{
        "carat": carat_val,
        "cut": cut_val,
        "color": color_val,
        "clarity": clarity_val,
        "depth": depth_val,
        "table": table_val,
        "x": x_val,
        "y": y_val,
        "z": z_val
    }])[feat_cols]

    # Predikce
    pred_1 = float(p1.predict(input_row)[0])
    pred_2 = float(p2.predict(input_row)[0])
    pred_3 = float(p3_ols.predict(input_row)[0])
    pred_3_ridge = float(p3_ridge.predict(input_row)[0])

    st.markdown("---")
    st.markdown("#### 💰 Výsledné odhadované ceny:")

    res_col1, res_col2, res_col3, res_col4 = st.columns(4)

    with res_col1:
        st.metric(
            label="Lineární OLS (Stupeň 1)",
            value=f"{pred_1:,.0f} USD"
        )
        st.caption("Předpokládá plochý nárůst.")

    with res_col2:
        st.metric(
            label="Kvadratický OLS (Stupeň 2)",
            value=f"{pred_2:,.0f} USD"
        )
        st.caption("Optimální model s interakcemi.")

    with res_col3:
        st.metric(
            label="Kubický OLS (Stupeň 3)",
            value=f"{pred_3:,.0f} USD"
        )
        if abs(pred_3 - pred_2) > 1500 or pred_3 < 0:
            st.caption("⚠️ Náchylný k extrapolacím!")
        else:
            st.caption("Nebezpečí kolísání.")

    with res_col4:
        st.metric(
            label="Kubický + Ridge (α=100)",
            value=f"{pred_3_ridge:,.0f} USD"
        )
        st.caption("Stabilizováno L2 penalizací.")

    if pred_3 < 0 or abs(pred_3) > 35000:
        st.error(
            f"🚨 **Extrémní anomálie kubického OLS modelu:** Pro zadané hodnoty predikoval {pred_3:,.0f} USD! "
            "To přesně demonstruje, proč v praxi nelze používat vysoké stupně polynomu bez regularizace."
        )

# -----------------------------------------------------------------------------
# TAB 4: ODPOVĚDI NA ZADÁNÍ CVIČENÍ
# -----------------------------------------------------------------------------
with t4:
    st.markdown("### 💡 Vyhodnocení cvičení dle zadání")

    st.markdown(r"""
    #### 1. Volba vhodného stupně polynomu (Select the appropriate degree):
    - **Jednoznačným vítězem je stupeň 2 (kvadratický polynom).**
    - Stupeň 2 zvýšil testovací $R^2$ z **0.9095 na 0.9655** (+0.056) a srazil průměrnou absolutní chybu (MAE) z **784.78 USD na 445.14 USD** (pokles o 43.3 %!).
    - Rozdíl mezi trénovacím a testovacím $R^2$ je u stupně 2 pouhých **0.0002** – model tedy netrpí žádným přeučením a skvěle generalizuje.

    #### 2. Proč je stupeň 3 bez regularizace nevhodný?
    - Ačkoliv trénovací $R^2$ u stupně 3 stoupne na **0.9780**, testovací $R^2$ propadne na **0.8591** a RMSE vystřelí na **1,463 USD** (výrazně horší než jednoduchý lineární model!).
    - Dochází ke kombinatorické explozi počtu parametrů (z 9 na 219) a multikolinearitě mocnin, což vede k vysokému rozptylu (tzv. **Rungeův jev**).

    #### 3. Srovnávací přehled metrik:
    """)

    summary_cards = pd.DataFrame([
        {"Metrika": "R² (Koeficient determinace)", "Stupeň 1 (OLS)": "0.9095", "Stupeň 2 (Kvadratický)": "0.9655 (Nejlepší)", "Stupeň 3 (Kubický)": "0.8591 (Kolaps)"},
        {"Metrika": "MAE (Střední absolutní chyba)", "Stupeň 1 (OLS)": "784.78 USD", "Stupeň 2 (Kvadratický)": "445.14 USD (-43 %)", "Stupeň 3 (Kubický)": "353.35 USD (Zkresleno)"},
        {"Metrika": "RMSE (Odmocnina MSE)", "Stupeň 1 (OLS)": "1,172.53 USD", "Stupeň 2 (Kvadratický)": "723.74 USD (-38 %)", "Stupeň 3 (Kubický)": "1,463.05 USD (+25 %)"},
        {"Metrika": "MSE (Střední kvadratická chyba)", "Stupeň 1 (OLS)": "1,374,832", "Stupeň 2 (Kvadratický)": "523,799", "Stupeň 3 (Kubický)": "2,140,511"}
    ])

    st.dataframe(summary_cards, width="stretch", hide_index=True)
