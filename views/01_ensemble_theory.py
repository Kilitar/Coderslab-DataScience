from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

base_dir = Path(__file__).resolve().parent.parent
theory_md_path = base_dir / "01_Regression" / "theory" / "07_ensemble_methods_theory.md"

st.title("🌲 Teorie 7: Ansámblové metody (Random Forest & Gradient Boosting)")
st.caption("Od jednoho nestabilního stromu k moudrosti lesa a síle boostingu: Bagging, Feature Subspacing, Sekvenční korekce reziduí a porovnání špičkových GBDT knihoven.")

tab1, tab2, tab3, tab4 = st.tabs([
    "📖 Teoretický rozbor",
    "⚔️ Živý souboj: Strom vs. Random Forest vs. GBDT",
    "🔍 Rentgen Boostingu: Postupná oprava reziduí",
    "📊 Významnost příznaků (Feature Importance)"
])

# =============================================================================
# TAB 1: TEORETICKÝ TEXT Z MARKDOWNU
# =============================================================================
with tab1:
    if theory_md_path.exists():
        with open(theory_md_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.warning("Dokument 07_ensemble_methods_theory.md nebyl nalezen.")

# =============================================================================
# TAB 2: ŽIVÝ SOUBOJ MODELŮ
# =============================================================================
with tab2:
    st.markdown("### ⚔️ Živý souboj: 1 Strom vs. Random Forest vs. Gradient Boosting")
    st.markdown(r"""
    Sledujte na stejném nelineárním datasetu se šumem, jak se liší chování a predikční křivky tří generací stromových algoritmů:
    1. **Jediný rozhodovací strom:** Tvoří hrubé schody a snadno se přeučí na náhodném šumu.
    2. **Random Forest (Bagging):** Zprůměrováním desítek nezávislých stromů schody vyhladí a dramaticky srazí testovací chybu.
    3. **Gradient Boosting (GBDT):** Postupně koriguje chyby a dosahuje nejvyšší přesnosti při správně zvoleném kroku učení.
    """)

    with st.container(border=True):
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            dataset_type = st.selectbox(
                "Experimentální dataset:",
                ["Nelineární vlna se šumem", "Parabola (y = x² + šum)", "Komplexní skoková funkce"]
            )
        with col_c2:
            n_trees = st.slider("Počet stromů v ansámblu (n_estimators):", min_value=10, max_value=250, value=80, step=10)
        with col_c3:
            tree_depth = st.slider("Max. hloubka stromů (max_depth):", min_value=1, max_value=8, value=3)

    # Generování dat
    np.random.seed(42)
    n_samples = 160
    X_raw = np.sort(np.random.uniform(0.5, 10.0, n_samples))

    if "vlna" in dataset_type.lower():
        y_clean = 2.5 * np.sin(X_raw) + 0.5 * X_raw
        y_noisy = y_clean + np.random.normal(0, 0.6, n_samples)
    elif "parabola" in dataset_type.lower():
        y_clean = 0.18 * (X_raw - 5.0) ** 2 + 1.2
        y_noisy = y_clean + np.random.normal(0, 0.45, n_samples)
    else:
        y_clean = np.where(X_raw < 3.5, 2.0, np.where(X_raw < 7.0, 5.5, 3.2))
        y_noisy = y_clean + np.random.normal(0, 0.4, n_samples)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_raw.reshape(-1, 1), y_noisy, test_size=0.25, random_state=42
    )

    # Fit 3 modelů
    # 1. Single Decision Tree
    single_tree = DecisionTreeRegressor(max_depth=tree_depth, random_state=42).fit(X_tr, y_tr)
    # 2. Random Forest
    rf_model = RandomForestRegressor(n_estimators=n_trees, max_depth=tree_depth, random_state=42).fit(X_tr, y_tr)
    # 3. Gradient Boosting
    gb_model = GradientBoostingRegressor(n_estimators=n_trees, max_depth=min(tree_depth, 4), learning_rate=0.08, random_state=42).fit(X_tr, y_tr)

    # Predikce pro jemnou osu
    x_eval = np.linspace(0.5, 10.0, 400).reshape(-1, 1)
    pred_tree_plot = single_tree.predict(x_eval)
    pred_rf_plot = rf_model.predict(x_eval)
    pred_gb_plot = gb_model.predict(x_eval)

    # Metriky
    te_r2_tree = r2_score(y_te, single_tree.predict(X_te))
    te_r2_rf = r2_score(y_te, rf_model.predict(X_te))
    te_r2_gb = r2_score(y_te, gb_model.predict(X_te))

    te_mse_tree = mean_squared_error(y_te, single_tree.predict(X_te))
    te_mse_rf = mean_squared_error(y_te, rf_model.predict(X_te))
    te_mse_gb = mean_squared_error(y_te, gb_model.predict(X_te))

    # KPI srovnání
    kpi_m1, kpi_m2, kpi_m3 = st.columns(3)
    with kpi_m1:
        st.metric("1. Samostatný strom", f"R²: {te_r2_tree:.3f}", delta=f"MSE: {te_mse_tree:.3f}", delta_color="inverse")
    with kpi_m2:
        st.metric("2. Random Forest", f"R²: {te_r2_rf:.3f}", delta=f"Δ R²: {te_r2_rf - te_r2_tree:+.3f}")
    with kpi_m3:
        st.metric("3. Gradient Boosting", f"R²: {te_r2_gb:.3f}", delta=f"Δ R²: {te_r2_gb - te_r2_tree:+.3f}")

    # Plotly graf srovnání
    fig_comp = go.Figure()
    # Body
    fig_comp.add_trace(go.Scatter(
        x=X_tr.flatten(), y=y_tr,
        mode="markers", name="Trénovací data",
        marker=dict(size=7, color="#38BDF8", opacity=0.7)
    ))
    fig_comp.add_trace(go.Scatter(
        x=X_te.flatten(), y=y_te,
        mode="markers", name="Neznámá testovací data",
        marker=dict(size=8, color="#F59E0B", symbol="diamond", opacity=0.85)
    ))
    # Křivky
    fig_comp.add_trace(go.Scatter(
        x=x_eval.flatten(), y=pred_tree_plot,
        mode="lines", name=f"1 Strom (Hloubka {tree_depth})",
        line=dict(color="#EF4444", width=2, dash="dot")
    ))
    fig_comp.add_trace(go.Scatter(
        x=x_eval.flatten(), y=pred_rf_plot,
        mode="lines", name=f"Random Forest ({n_trees} stromů)",
        line=dict(color="#10B981", width=3)
    ))
    fig_comp.add_trace(go.Scatter(
        x=x_eval.flatten(), y=pred_gb_plot,
        mode="lines", name=f"Gradient Boosting ({n_trees} kroků)",
        line=dict(color="#A855F7", width=3)
    ))

    fig_comp.update_layout(
        title="Porovnání predikčních křivek na testovacích datech",
        xaxis_title="Vstupní prediktor X",
        yaxis_title="Cílová veličina Y",
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_comp, width="stretch")

    st.info(
        "💡 **Klíčový vizuální poznatek:** "
        "Všimněte si, jak **Random Forest (zelená)** průměrováním desítek posunutých bootstrapových stromů přirozeně zaobluje ostré rohy schodů, "
        "zatímco **Gradient Boosting (fialová)** se dokáže těsně přizpůsobit i složitým zákrutám funkce bez toho, aby se nechal zmást lokálními šumovými body."
    )

# =============================================================================
# TAB 3: RENTGEN BOOSTINGU (KROKOVÁ KOREKCE REZIDUÍ)
# =============================================================================
with tab3:
    st.markdown("### 🔍 Rentgen Boostingu: Jak strom za stromem maže chyby")
    st.markdown(r"""
    Zde můžete nahlédnout přímo „pod kapotu“ algoritmu **Gradient Boosting**.  
    Každý nový strom se netrénuje na původní hodnoty $Y$, ale na **rozdíl mezi realitou a dosavadním součtem předchozích stromů (tzv. rezidua)**:
    $$r_{im} = y_i - \hat{y}_{m-1}(x_i)$$
    """)

    with st.container(border=True):
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            step_m = st.slider("Zvolte počet aproximačních kroků (Stromů m):", min_value=1, max_value=40, value=3)
        with col_b2:
            lr_val = st.select_slider(
                "Rychlost učení (Learning Rate η):",
                options=[0.02, 0.05, 0.1, 0.3, 0.8],
                value=0.1,
                help="Menší learning rate vyžaduje více stromů, ale dosahuje hladšího a přesnějšího řešení."
            )

    # Simulace postupného trénování po jednotlivých stromech
    np.random.seed(99)
    x_b = np.sort(np.random.uniform(1.0, 9.0, 120))
    y_true_b = 3.0 * np.sin(x_b) + 0.3 * (x_b - 5)**2
    y_b = y_true_b + np.random.normal(0, 0.45, 120)

    # Natrénujeme model přesně do zvoleného kroku
    gb_step = GradientBoostingRegressor(
        n_estimators=step_m,
        learning_rate=lr_val,
        max_depth=2, # Mělké stromy (stumps / malé větve)
        random_state=42
    )
    gb_step.fit(x_b.reshape(-1, 1), y_b)

    x_dense = np.linspace(1.0, 9.0, 300).reshape(-1, 1)
    y_dense_pred = gb_step.predict(x_dense)
    current_preds = gb_step.predict(x_b.reshape(-1, 1))
    current_residuals = y_b - current_preds

    # Vykreslení dvou panelů nad sebou
    fig_gb_xray = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=(
            f"1. Celková predikce ansámblu po {step_m} krocích (Aproximace Y)",
            f"2. Zbývající nevysvětlená rezidua (Chyba, na kterou se vrhne další strom)"
        )
    )

    # Panel 1: Data a predikce
    fig_gb_xray.add_trace(go.Scatter(
        x=x_b, y=y_b,
        mode="markers", name="Původní data Y",
        marker=dict(size=7, color="#38BDF8", opacity=0.7)
    ), row=1, col=1)

    fig_gb_xray.add_trace(go.Scatter(
        x=x_dense.flatten(), y=y_dense_pred,
        mode="lines", name=f"Součet {step_m} stromů (η={lr_val})",
        line=dict(color="#A855F7", width=3)
    ), row=1, col=1)

    # Panel 2: Rezidua
    fig_gb_xray.add_trace(go.Scatter(
        x=x_b, y=current_residuals,
        mode="markers", name="Aktuální rezidua (y - ŷ)",
        marker=dict(size=7, color="#EF4444", symbol="x")
    ), row=2, col=1)

    fig_gb_xray.add_hline(y=0.0, line_dash="dash", line_color="gray", row=2, col=1)

    fig_gb_xray.update_layout(
        height=580,
        xaxis2_title="Prediktor X",
        yaxis_title="Hodnota Y",
        yaxis2_title="Chyba (Reziduální Y)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_gb_xray, width="stretch")

    res_std = np.std(current_residuals)
    st.caption(
        f"📊 **Směrodatná odchylka reziduí po {step_m} krocích:** `{res_std:.4f}`. "
        "Sledujte, jak se s posunem posuvníku vpravo spodní body (rezidua) zplošťují kolem nuly a ztrácejí jakýkoliv tvar – stává se z nich neuspořádaný náhodný šum, což značí dokonalé vytěžení signálu."
    )

# =============================================================================
# TAB 4: FEATURE IMPORTANCE
# =============================================================================
with tab4:
    st.markdown("### 📊 Důležitost příznaků (Feature Importance): Moudrost lesa")
    st.markdown(r"""
    Jednou z nejcennějších vlastností ansámblů je schopnost spolehlivě určit, **které proměnné mají na výsledek skutečný vliv a které jsou pouhý balast**.
    
    Zatímco jeden strom má tendenci favorizovat jeden dominantní prediktor a ostatní ignorovat, **Random Forest díky náhodnému vzorkování příznaků v uzlech (`max_features`)** dává férovou šanci všem signálům.
    """)

    # Vytvoření syntetického datasetu s 5 příznaky
    np.random.seed(42)
    n_fi = 250
    f1 = np.random.uniform(10, 50, n_fi) # Silný lineární
    f2 = np.random.uniform(1, 10, n_fi)   # Silný nelineární
    f3 = np.random.uniform(0, 100, n_fi)  # Slabá interakce
    noise1 = np.random.normal(0, 1, n_fi) # Čistý šum
    noise2 = np.random.uniform(50, 100, n_fi) # Čistý šum

    # Cílová proměnná
    y_fi = 4.0 * f1 + 2.5 * (f2 ** 2) + 0.5 * (f1 * f3 / 50.0) + np.random.normal(0, 15, n_fi)

    X_fi = pd.DataFrame({
        "1. Plocha bytu (Silný signál)": f1,
        "2. Lokalita rating (Silný nelineární)": f2,
        "3. Patro / Výtah (Slabý signál)": f3,
        "4. Náhodný šum A (Balast)": noise1,
        "5. Náhodný šum B (Balast)": noise2
    })

    # Modely
    m_tree = DecisionTreeRegressor(max_depth=4, random_state=42).fit(X_fi, y_fi)
    m_rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42).fit(X_fi, y_fi)
    m_gb = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42).fit(X_fi, y_fi)

    df_importances = pd.DataFrame({
        "Příznak": X_fi.columns,
        "1 Rozhodovací strom": m_tree.feature_importances_,
        "Random Forest (100 stromů)": m_rf.feature_importances_,
        "Gradient Boosting (GBDT)": m_gb.feature_importances_
    })

    fig_fi = go.Figure()
    fig_fi.add_trace(go.Bar(
        x=df_importances["Příznak"], y=df_importances["1 Rozhodovací strom"],
        name="1 Rozhodovací strom", marker_color="#EF4444"
    ))
    fig_fi.add_trace(go.Bar(
        x=df_importances["Příznak"], y=df_importances["Random Forest (100 stromů)"],
        name="Random Forest", marker_color="#10B981"
    ))
    fig_fi.add_trace(go.Bar(
        x=df_importances["Příznak"], y=df_importances["Gradient Boosting (GBDT)"],
        name="Gradient Boosting", marker_color="#A855F7"
    ))

    fig_fi.update_layout(
        title="Srovnání Feature Importance (MDI – Mean Decrease in Impurity)",
        barmode="group",
        xaxis_title="Vstupní prediktor",
        yaxis_title="Relativní důležitost (Součet = 1.0)",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_fi, width="stretch")

    st.success(
        "🎯 **Co z grafu vyčíst pro praxi:** "
        "Všimněte si, jak oba ansámblové modely (Random Forest i Boosting) spolehlivě stlačily důležitost náhodného šumu téměř na nulu, "
        "zatímco jediný strom mohl náhodnému šumu přisoudit falešnou důležitost kvůli náhodnému rozdělení v listu."
    )
