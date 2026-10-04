"""
Den 3 Extras: 📈 Gradient Boosting: Krok za krokem simulátor & animátor
======================================================================
- Interaktivní krokovač 1D Gradient Boostingu (F_0 -> strom 1 -> strom 2 ... -> F_M).
- Horní graf: Kumulativní predikce ansámblu F_m(x) vs skutečná data a pravá funkce.
- Dolní graf: Aktuální rezidua r_m a slabý model h_m(x), který je fituje.
- Zkoumání vlivu learning rate (shrinkage η) a hloubky stromu (depth 1 = stump).
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.tree import DecisionTreeRegressor

st.set_page_config(page_title="Gradient Boosting Simulátor", page_icon="📈", layout="wide")

st.markdown("""
# 📈 Gradient Boosting: Krok za krokem simulátor
### Sleduj, jak se sekvence slabých stromů postupně učí z vlastních chyb (reziduí)
""")

st.info("""
V kurzu jsme se seznámili s metaforou: **„Student a učitel“** (*Boosting introduction*, str. 4). 
Zatímco Bagging trénuje všechny stromy nezávisle vedle sebe (paralelně), **Boosting staví stromy za sebou (sekvenčně)**. 
Každý nový strom se nesnaží předpovídat $y$, ale **chybu (reziduum)**, kterou zanechaly všechny předchozí stromy dohromady!
""")

# --- Ovládací prvky v postranním panelu ---
with st.sidebar:
    st.header("⚙️ Konfigurace Boostingu")
    
    dataset_type = st.selectbox(
        "Tvar 1D funkce:",
        ["Kombinovaná sinusovka & schod", "Polynom 3. stupně", "Dva skoky (Step function)"],
        index=0,
    )
    
    n_points = st.slider("Počet datových bodů:", min_value=30, max_value=150, value=70, step=10)
    noise_lvl = st.slider("Šum (Noise):", min_value=0.0, max_value=0.4, value=0.15, step=0.05)
    
    st.divider()
    st.header("🌲 Hyperparametry stromů")
    max_stages = st.slider("Celkový počet stromů (M):", min_value=5, max_value=50, value=25, step=5)
    learning_rate = st.slider("Learning rate / Shrinkage (η):", min_value=0.05, max_value=1.0, value=0.30, step=0.05)
    tree_depth = st.radio("Hloubka jednotlivých stromů (max_depth):", [1, 2, 3], index=1, horizontal=True,
                          help="Depth 1 = Decision Stump (nejjednodušší slabý model).")

    seed = st.number_input("Random Seed:", value=42, step=1)

# Generování dat
rng = np.random.RandomState(int(seed))
X = np.sort(rng.uniform(-3, 3, size=n_points))

if dataset_type == "Kombinovaná sinusovka & schod":
    y_true = np.sin(X) + (X > 0) * 0.8
elif dataset_type == "Polynom 3. stupně":
    y_true = 0.1 * (X ** 3) - 0.3 * X
else:
    y_true = np.where(X < -1, -1.0, np.where(X < 1, 0.0, 1.0))

y = y_true + rng.normal(0, noise_lvl, size=n_points)
X_grid = np.linspace(-3, 3, 300).reshape(-1, 1)

# --- Výpočet kroků Boostingu ---
# Krok 0: F_0 = mean(y)
f0_val = np.mean(y)
F_grid = np.full(300, f0_val)
F_train = np.full(n_points, f0_val)

stages_F = [F_grid.copy()]
stages_residuals = [y - F_train]
stages_trees = [None]
stages_tree_preds_grid = [np.zeros(300)]
mse_history = [np.mean((y - F_train) ** 2)]

for m in range(1, max_stages + 1):
    res = y - F_train
    tree = DecisionTreeRegressor(max_depth=tree_depth, random_state=int(seed) + m)
    tree.fit(X.reshape(-1, 1), res)
    
    # Predikce stromu na mřížce i na trénovacích bodech
    h_grid = tree.predict(X_grid)
    h_train = tree.predict(X.reshape(-1, 1))
    
    # Update s learning rate (shrinkage)
    F_grid = F_grid + learning_rate * h_grid
    F_train = F_train + learning_rate * h_train
    
    stages_F.append(F_grid.copy())
    stages_residuals.append(y - F_train)
    stages_trees.append(tree)
    stages_tree_preds_grid.append(h_grid)
    mse_history.append(np.mean((y - F_train) ** 2))

# --- Interaktivní posuvník aktuálního kroku ---
st.subheader("🕹️ Interaktivní časová osa: Krok po kroku")
col_s1, col_s2 = st.columns([3, 1])
with col_s1:
    curr_step = st.slider("Zvol aktuální krok boostingového ansámblu (m):", min_value=0, max_value=max_stages, value=1, step=1)
with col_s2:
    st.metric(
        f"Chyba MSE (Krok {curr_step})",
        f"{mse_history[curr_step]:.4f}",
        delta=f"{(mse_history[curr_step] - mse_history[0]):.4f} vs F₀" if curr_step > 0 else None,
        delta_color="inverse",
    )

# --- GRAFY ---
col_g1, col_g2 = st.columns([1.6, 1.0])

with col_g1:
    # Horní graf: Ansámbl F_m(x)
    fig_top = go.Figure()
    # Skutečná data
    fig_top.add_trace(go.Scatter(x=X, y=y, mode="markers", name="Trénovací data y", marker=dict(color="#64748b", size=7)))
    # Pravá funkce
    if dataset_type == "Kombinovaná sinusovka & schod":
        true_grid = np.sin(X_grid.ravel()) + (X_grid.ravel() > 0) * 0.8
    elif dataset_type == "Polynom 3. stupně":
        true_grid = 0.1 * (X_grid.ravel() ** 3) - 0.3 * X_grid.ravel()
    else:
        true_grid = np.where(X_grid.ravel() < -1, -1.0, np.where(X_grid.ravel() < 1, 0.0, 1.0))
    fig_top.add_trace(go.Scatter(x=X_grid.ravel(), y=true_grid, mode="lines", name="Pravý signál", line=dict(color="#10b981", dash="dot", width=1.5)))
    
    # Model F_m(x)
    fig_top.add_trace(go.Scatter(
        x=X_grid.ravel(), y=stages_F[curr_step], mode="lines",
        name=f"Ansámbl F_{curr_step}(x)",
        line=dict(color="#2563eb", width=3.5),
    ))
    
    fig_top.update_layout(
        title=f"1. Kumulativní predikce ansámblu $F_{{{curr_step}}}(x)$ (Aproximace cíle y)",
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(title="x"),
        yaxis=dict(title="y"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig_top, use_container_width=True)

    # Dolní graf: Rezidua a nový strom h_m(x)
    if curr_step > 0:
        prev_res = stages_residuals[curr_step - 1]
        cur_tree_pred = stages_tree_preds_grid[curr_step]
        
        fig_bot = go.Figure()
        fig_bot.add_trace(go.Scatter(
            x=X, y=prev_res, mode="markers",
            name=f"Rezidua $r_{{{curr_step-1}}} = y - F_{{{curr_step-1}}}$",
            marker=dict(color="#ef4444", size=6),
        ))
        fig_bot.add_trace(go.Scatter(
            x=X_grid.ravel(), y=cur_tree_pred, mode="lines",
            name=f"Slabý strom $h_{{{curr_step}}}(x)$ fitující rezidua",
            line=dict(color="#f97316", width=2.5),
        ))
        fig_bot.update_layout(
            title=f"2. Co se právě učí: Strom č. {curr_step} fituje rezidua $r_{{{curr_step-1}}}$",
            height=280,
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis=dict(title="x"),
            yaxis=dict(title="Reziduum"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_bot, use_container_width=True)
    else:
        st.info("💡 V kroku 0 je predikce $F_0(x)$ rovna pouhému průměru všech hodnot ($\bar{y}$). Posuň slider na krok 1.")

with col_g2:
    st.subheader("📉 Pokles chyby MSE napříč stromy")
    fig_mse = go.Figure()
    fig_mse.add_trace(go.Scatter(
        x=list(range(max_stages + 1)),
        y=mse_history,
        mode="lines+markers",
        name="MSE",
        line=dict(color="#6366f1", width=2),
        marker=dict(size=4),
    ))
    # Zvýraznění aktuálního bodu
    fig_mse.add_trace(go.Scatter(
        x=[curr_step],
        y=[mse_history[curr_step]],
        mode="markers",
        name="Aktuální krok",
        marker=dict(color="#ef4444", size=10, symbol="diamond"),
    ))
    fig_mse.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(title="Krok boostingu (m)"),
        yaxis=dict(title="Trénovací MSE"),
        showlegend=False,
    )
    st.plotly_chart(fig_mse, use_container_width=True)

    st.markdown("""
    #### 🎓 Klíčové poznatky:
    1. **Role Shrinkage ($\\eta$):** Pokud nastavíš $\\eta = 1.0$, model se učí agresivně a rychle přeučí šum. S malým $\\eta = 0.1$ jsou kroky opatrné a model má mnohem lepší generalizaci.
    2. **Hloubka stromu (Max Depth):**
       * Hloubka 1 (Stump): Dělá pouze 1 řez (dvě konstantní hladiny). Dokáže modelovat jen aditivní efekty.
       * Hloubka 2-3: Umožňuje interakce mezi proměnnými.
    """)
