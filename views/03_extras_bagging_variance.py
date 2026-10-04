"""
Den 3 Extras: 🌲 Bagging & Random Forest: Variance & Correlation Lab
===================================================================
- Matematický rozbor Breimanova vzorce redukce variance v ansámblech:
  Var(f_avg) = rho * sigma^2 + ((1 - rho) / B) * sigma^2
- Živá 1D simulace na zašuměné funkci (sinusovka):
  Jednotlivé stromy trénované na bootstrap vzorcích (tenké čáry) vs. průměr ansámblu (tlustá křivka).
- Vliv parametru max_features na dekorelaci stromů (rho) v Random Forest.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

st.set_page_config(page_title="Bagging & Random Forest Variance Lab", page_icon="🌲", layout="wide")

st.markdown("""
# 🌲 Bagging & Random Forest: Variance & Correlation Lab
### Proč ansámbl snižuje rozptyl a jak de-korelace stromů zachraňuje přesnost
""")

st.info("""
Tato laboratoř vizualizuje teoretické jádro **Baggingu a Random Forestu**. 
Pochopíš, proč pouhé průměrování stromů naráží na strop daný korelací $\\rho$ a proč Leo Breiman 
přidal k baggingu náhodný výběr příznaků (`max_features`), který stromy de-koreluje.
""")

tab_math, tab_sim, tab_feat = st.tabs([
    "📐 1. Breimanův vzorec variance",
    "🧪 2. 1D Živá simulace (Bootstrap stromy)",
    "🔀 3. Vliv max_features na korelaci",
])

# =============================================================================
# TAB 1: BREIMANŮV VZOREC
# =============================================================================
with tab_math:
    st.subheader("Matematická podstata redukce rozptylu")
    st.markdown(r"""
    Nechť máme ansámbl $B$ stromů, kde každý strom má rozptyl $\sigma^2$ a průměrná párová korelace mezi stromy je $\rho \in [0, 1]$.
    Rozptyl průměrované předpovědi $\bar{f}(x) = \frac{1}{B} \sum_{b=1}^B f_b(x)$ je přesně roven:
    
    $$\text{Var}(\bar{f}) = \mathbf{\rho \cdot \sigma^2} + \mathbf{\frac{1 - \rho}{B} \cdot \sigma^2}$$
    
    * **První člen $\rho \sigma^2$ (Neredukovatelná složka):** Nezávisí na počtu stromů $B$! Ani kdybychom měli 1 000 000 stromů, rozptyl neklesne pod tuto mez.
    * **Druhý člen $\frac{1 - \rho}{B} \sigma^2$ (Redukovatelná složka):** Pro $B \to \infty$ klesá k nule.
    """)
    
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        sigma2_val = st.slider("Rozptyl 1 stromu (σ²):", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
    with col_c2:
        rho_val = st.slider("Párová korelace stromů (ρ):", min_value=0.0, max_value=0.95, value=0.40, step=0.05)
    with col_c3:
        b_max = st.slider("Maximální počet stromů v grafu (B):", min_value=10, max_value=200, value=100, step=10)

    # Výpočet křivky
    b_range = np.arange(1, b_max + 1)
    var_total = rho_val * sigma2_val + ((1.0 - rho_val) / b_range) * sigma2_val
    var_irred = np.full_like(b_range, rho_val * sigma2_val, dtype=float)

    fig_var = go.Figure()
    fig_var.add_trace(go.Scatter(x=b_range, y=var_total, mode="lines", name="Celkový rozptyl Var(f_avg)", line=dict(color="#3b82f6", width=3)))
    fig_var.add_trace(go.Scatter(x=b_range, y=var_irred, mode="lines", name=f"Asymptotický strop ρ·σ² = {rho_val * sigma2_val:.2f}", line=dict(color="#ef4444", dash="dash", width=2)))
    
    fig_var.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(title="Počet stromů v ansámblu (B)"),
        yaxis=dict(title="Rozptyl předpovědi Var", range=[0, sigma2_val * 1.05]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig_var, use_container_width=True)

    cm1, cm2, cm3 = st.columns(3)
    cm1.metric("Počáteční rozptyl (1 strom)", f"{sigma2_val:.2f}")
    cm2.metric(f"Rozptyl při B = {b_max} stromech", f"{var_total[-1]:.3f}", delta=f"{(var_total[-1] - sigma2_val):.2f}")
    cm3.metric("Minimální dosažitelný rozptyl (B → ∞)", f"{rho_val * sigma2_val:.3f}")

# =============================================================================
# TAB 2: 1D ŽIVÁ SIMULACE (BOOTSTRAP)
# =============================================================================
with tab_sim:
    st.subheader("1D Simulace: Jednotlivé stromy vs. Bagging průměr")
    st.caption("Sleduj, jak každý jednotlivý hluboký strom divoce osciluje kolem šumu, ale jejich průměr se zklidní do hladké křivky.")
    
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        n_pts = st.slider("Počet bodů dat:", min_value=20, max_value=120, value=60, step=10)
        noise_std = st.slider("Šum (Gaussian noise):", min_value=0.05, max_value=0.6, value=0.25, step=0.05)
    with col_p2:
        n_trees_sim = st.slider("Počet bootstrap stromů:", min_value=3, max_value=50, value=20, step=1)
        tree_max_depth = st.slider("Max depth jednotlivých stromů:", min_value=2, max_value=12, value=6, step=1)
    with col_p3:
        sim_seed = st.number_input("Seed simulace:", value=42, step=1)

    # Generování dat
    rng_sim = np.random.RandomState(int(sim_seed))
    X_raw = np.sort(rng_sim.uniform(-3, 3, size=n_pts))
    y_clean = np.sin(X_raw) + 0.5 * np.cos(2 * X_raw)
    y_noisy = y_clean + rng_sim.normal(0, noise_std, size=n_pts)

    X_grid = np.linspace(-3, 3, 300).reshape(-1, 1)
    
    # Trénování jednotlivých bootstrap stromů
    tree_preds = []
    for i in range(n_trees_sim):
        boot_idx = rng_sim.choice(n_pts, size=n_pts, replace=True)
        t = DecisionTreeRegressor(max_depth=tree_max_depth, random_state=int(sim_seed) + i)
        t.fit(X_raw[boot_idx].reshape(-1, 1), y_noisy[boot_idx])
        tree_preds.append(t.predict(X_grid))
    
    tree_preds = np.array(tree_preds)  # shape (n_trees, 300)
    ensemble_mean = tree_preds.mean(axis=0)

    fig_sim = go.Figure()
    # Datové body
    fig_sim.add_trace(go.Scatter(x=X_raw, y=y_noisy, mode="markers", name="Trénovací data (se šumem)", marker=dict(color="#64748b", size=8)))
    # Skutečná křivka
    fig_sim.add_trace(go.Scatter(x=X_grid.ravel(), y=np.sin(X_grid.ravel()) + 0.5 * np.cos(2 * X_grid.ravel()), mode="lines", name="Pravá funkce (True signal)", line=dict(color="#10b981", width=2, dash="dot")))
    
    # Jednotlivé stromy (průhledné tenké čáry)
    for i in range(min(n_trees_sim, 20)):
        fig_sim.add_trace(go.Scatter(
            x=X_grid.ravel(), y=tree_preds[i], mode="lines",
            name="Jednotlivé stromy" if i == 0 else None,
            line=dict(color="rgba(239, 68, 68, 0.25)", width=1),
            showlegend=(i == 0),
            hoverinfo="skip",
        ))
    
    # Průměr ansámblu
    fig_sim.add_trace(go.Scatter(x=X_grid.ravel(), y=ensemble_mean, mode="lines", name=f"Ansámbl ({n_trees_sim} stromů)", line=dict(color="#2563eb", width=4)))

    fig_sim.update_layout(
        height=450,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(title="x"),
        yaxis=dict(title="y"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig_sim, use_container_width=True)

# =============================================================================
# TAB 3: VLIV MAX_FEATURES NA KORELACI
# =============================================================================
with tab_feat:
    st.subheader("Random Forest de-korelace: Proč omezujeme počet příznaků?")
    st.markdown("""
    Při obyčejném Baggingu mají stromy tendenci volit na kořenových uzlech **tytéž dominantní prediktory**. 
    Tím pádem jsou si stromy velmi podobné a korelace $\\rho$ mezi nimi je vysoká (např. 0.7 až 0.9).
    
    **Random Forest řešení:** Při každém rozštěpení strom smí vybírat pouze z náhodné podmnožiny $m$ příznaků (typicky $\\sqrt{p}$ u klasifikace nebo $p/3$ u regrese). 
    Tím se stromy přinutí hledat alternativní závislosti a **korelace $\\rho$ prudce klesá**, což srazí asymptotický strop variance $\\rho \\sigma^2$!
    """)
    
    st.dataframe(pd.DataFrame([
        {"Metoda": "Rozhodovací strom (1x)", "Příznaky na split": "Všechny (p)", "Bootstrap vzorky": "Ne (100% dat)", "Korelace ρ": "N/A", "Rozptyl (Variance)": "Extrémně vysoký"},
        {"Metoda": "Bagging (B stromů)", "Příznaky na split": "Všechny (p)", "Bootstrap vzorky": "Ano (s opakováním)", "Korelace ρ": "Vysoká (0.6 - 0.8)", "Rozptyl (Variance)": "Snížený na ρ·σ²"},
        {"Metoda": "Random Forest (B stromů)", "Příznaky na split": "Náhodná podmnožina m < p", "Bootstrap vzorky": "Ano (s opakováním)", "Korelace ρ": "Nízká (0.2 - 0.4)", "Rozptyl (Variance)": "Minimální možný"},
    ]), hide_index=True, use_container_width=True)
