"""
Den 3 Extras: 🧠 Interaktivní 2D hřiště neuronových sítí (NN Playground)
======================================================================
- Interaktivní trénování MLPClassifier přímo v aplikaci na 2D datasetech:
  Moons, Circles, XOR a Archimédova spirála.
- Nastavení architektury (1 nebo 2 skryté vrstvy, 2 až 32 neuronů na vrstvu).
- Volba aktivační funkce (ReLU, Tanh, Logistic/Sigmoid).
- Nastavení optimalizátoru (Adam, SGD), learning rate a regularizace L2 (alpha).
- 2D vizualizace rozhodovací hranice (Plotly contour / decision boundary) a ztrátové křivky (loss curve).
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import make_circles, make_moons
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

st.set_page_config(page_title="2D Hřiště neuronových sítí", page_icon="🧠", layout="wide")

st.markdown("""
# 🧠 Interaktivní 2D hřiště neuronových sítí
### Prozkoumej, jak hloubka, neurony a aktivace ohýbají rozhodovací hranici
""")

st.info("""
**Proč toto hřiště existuje?** V přednáškách a cvičeních jsme viděli neuronové sítě na velkých tabulkách (KC Housing) 
a obrázcích (MNIST). Tady máš možnost vidět **přesně totéž ve 2D prostoru**: jak jednotlivé vrstvy a nelineární aktivace 
ohýbají rozhodovací plochu kolem dat a proč lineární modely selhávají na problémech typu XOR nebo Spirála.
""")

# --- Generování datasetů ---
def generate_dataset(dataset_type: str, n_samples: int = 300, noise: float = 0.15, random_state: int = 42):
    rng = np.random.RandomState(random_state)
    if dataset_type == "Dva půlměsíce (Moons)":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    elif dataset_type == "Soustředné kružnice (Circles)":
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=random_state)
    elif dataset_type == "XOR (Čtyři kvadranty)":
        raw_x = rng.uniform(-2, 2, size=(n_samples, 2))
        y = ((raw_x[:, 0] > 0) ^ (raw_x[:, 1] > 0)).astype(int)
        # přidat lehký šum
        raw_x += rng.normal(0, noise, size=raw_x.shape)
        X = raw_x
    elif dataset_type == "Dvojitá spirála (Spiral)":
        n_per_arm = n_samples // 2
        # Rameno 0
        theta0 = np.sqrt(rng.rand(n_per_arm)) * 2.5 * np.pi
        r0 = 2 * theta0 + rng.randn(n_per_arm) * (noise * 5)
        x0 = np.column_stack([r0 * np.cos(theta0), r0 * np.sin(theta0)])
        y0 = np.zeros(n_per_arm, dtype=int)
        # Rameno 1
        theta1 = np.sqrt(rng.rand(n_per_arm)) * 2.5 * np.pi
        r1 = 2 * theta1 + rng.randn(n_per_arm) * (noise * 5)
        x1 = np.column_stack([-r1 * np.cos(theta1), -r1 * np.sin(theta1)])
        y1 = np.ones(n_per_arm, dtype=int)
        X = np.vstack([x0, x1]) / 10.0
        y = np.concatenate([y0, y1])
    else:
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    return X, y

# --- Postranní panel / Ovládací prvky ---
with st.sidebar:
    st.header("⚙️ Konfigurace úlohy")
    dataset_name = st.selectbox(
        "Vyber 2D dataset:",
        ["Dva půlměsíce (Moons)", "Soustředné kružnice (Circles)", "XOR (Čtyři kvadranty)", "Dvojitá spirála (Spiral)"],
        index=0,
    )
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        n_samples = st.slider("Počet bodů", min_value=100, max_value=600, value=300, step=50)
    with col_s2:
        noise_lvl = st.slider("Úroveň šumu", min_value=0.02, max_value=0.35, value=0.15, step=0.02)

    st.divider()
    st.header("🏗️ Architektura sítě (MLP)")
    n_layers = st.radio("Počet skrytých vrstev:", [1, 2], horizontal=True)
    layer1_neurons = st.slider("Neurony ve vrstvě 1", min_value=2, max_value=48, value=12, step=2)
    if n_layers == 2:
        layer2_neurons = st.slider("Neurony ve vrstvě 2", min_value=2, max_value=48, value=8, step=2)
        hidden_sizes = (layer1_neurons, layer2_neurons)
    else:
        hidden_sizes = (layer1_neurons,)

    activation = st.selectbox(
        "Aktivační funkce:",
        ["relu", "tanh", "logistic"],
        index=0,
        format_func=lambda x: {
            "relu": "ReLU – max(0, z) (Moderní standard)",
            "tanh": "Tanh – Hyperbolický tangens (-1..1)",
            "logistic": "Logistic / Sigmoid – 1/(1+e^-z)",
        }[x],
    )

    st.divider()
    st.header("⚡ Hyperparametry tréninku")
    solver = st.selectbox("Optimalizátor (Solver):", ["adam", "sgd"], index=0)
    lr_init = st.select_slider(
        "Počáteční Learning Rate (η):",
        options=[0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5],
        value=0.03,
    )
    l2_alpha = st.select_slider(
        "L2 Regularizace (alpha):",
        options=[1e-5, 1e-4, 1e-3, 0.01, 0.1, 1.0],
        value=1e-4,
    )
    max_iter = st.slider("Max epoch tréninku:", min_value=30, max_value=500, value=200, step=20)
    seed = st.number_input("Random Seed:", min_value=0, max_value=9999, value=42, step=1)

# Generování dat a split
X, y = generate_dataset(dataset_name, n_samples=n_samples, noise=noise_lvl, random_state=int(seed))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=int(seed))

# Spočítání počtu vah
# Vstup má 2 dimenze, výstup má 1 logit (binární klasifikace)
if len(hidden_sizes) == 1:
    n_params = (2 * hidden_sizes[0] + hidden_sizes[0]) + (hidden_sizes[0] * 1 + 1)
else:
    n_params = (2 * hidden_sizes[0] + hidden_sizes[0]) + (hidden_sizes[0] * hidden_sizes[1] + hidden_sizes[1]) + (hidden_sizes[1] * 1 + 1)

# Trénování modelu
clf = MLPClassifier(
    hidden_layer_sizes=hidden_sizes,
    activation=activation,
    solver=solver,
    learning_rate_init=lr_init,
    alpha=l2_alpha,
    max_iter=max_iter,
    random_state=int(seed),
)

with st.spinner("Trénuji neuronovou síť..."):
    clf.fit(X_train, y_train)

# Metriky
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_prob_test = clf.predict_proba(X_test)[:, 1]

acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)
loss_test = log_loss(y_test, y_prob_test)

# --- Horní lišta metrik ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Přesnost na testu (Accuracy)", f"{acc_test * 100:.1f} %", delta=f"{(acc_test - acc_train) * 100:+.1f} % vs train")
c2.metric("Trénovací přesnost", f"{acc_train * 100:.1f} %")
c3.metric("Počet parametrů (vah)", f"{n_params} vah", help="Zahrnuje matice vah W i biasy b napříč všemi vrstvami.")
c4.metric("Odtrénováno epoch", f"{clf.n_iter_} / {max_iter}", help="Pokud je hodnota menší než max, síť zkonvergovala předčasně dle tolerance.")

st.divider()

# --- Grafy: Rozhodovací hranice a Ztrátová křivka ---
col_left, col_right = st.columns([1.3, 1.0])

with col_left:
    st.subheader("🗺️ Rozhodovací plocha v 2D prostoru")
    
    # Meshgrid pro decision boundary
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 120), np.linspace(y_min, y_max, 120))
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # Predikce pravděpodobnosti pro třídu 1
    probs_grid = clf.predict_proba(grid)[:, 1].reshape(xx.shape)
    
    fig_boundary = go.Figure()
    
    # Konturová plocha pravděpodobností
    fig_boundary.add_trace(
        go.Contour(
            x=np.linspace(x_min, x_max, 120),
            y=np.linspace(y_min, y_max, 120),
            z=probs_grid,
            colorscale="RdBu_r",
            zmin=0.0,
            zmax=1.0,
            opacity=0.75,
            contours=dict(start=0.1, end=0.9, size=0.1, showlines=False),
            colorbar=dict(title="P(y=1)", len=0.8),
            hoverinfo="skip",
        )
    )
    
    # Hranice P = 0.5 (Decision boundary)
    fig_boundary.add_trace(
        go.Contour(
            x=np.linspace(x_min, x_max, 120),
            y=np.linspace(y_min, y_max, 120),
            z=probs_grid,
            contours=dict(start=0.5, end=0.5, coloring="none"),
            line=dict(color="black", width=3, dash="solid"),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    
    # Trénovací body (kroužky)
    fig_boundary.add_trace(
        go.Scatter(
            x=X_train[y_train == 0, 0],
            y=X_train[y_train == 0, 1],
            mode="markers",
            name="Train: Třída 0",
            marker=dict(color="#ef4444", size=8, line=dict(color="white", width=1)),
        )
    )
    fig_boundary.add_trace(
        go.Scatter(
            x=X_train[y_train == 1, 0],
            y=X_train[y_train == 1, 1],
            mode="markers",
            name="Train: Třída 1",
            marker=dict(color="#3b82f6", size=8, line=dict(color="white", width=1)),
        )
    )
    
    # Testovací body (kosočtverce)
    fig_boundary.add_trace(
        go.Scatter(
            x=X_test[y_test == 0, 0],
            y=X_test[y_test == 0, 1],
            mode="markers",
            name="Test: Třída 0",
            marker=dict(color="#b91c1c", symbol="diamond", size=10, line=dict(color="white", width=1.5)),
        )
    )
    fig_boundary.add_trace(
        go.Scatter(
            x=X_test[y_test == 1, 0],
            y=X_test[y_test == 1, 1],
            mode="markers",
            name="Test: Třída 1",
            marker=dict(color="#1d4ed8", symbol="diamond", size=10, line=dict(color="white", width=1.5)),
        )
    )
    
    fig_boundary.update_layout(
        height=500,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(title="Příznak $x_1$"),
        yaxis=dict(title="Příznak $x_2$"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig_boundary, width="stretch")

with col_right:
    st.subheader("📉 Průběh trénování (Loss curve)")
    if hasattr(clf, "loss_curve_"):
        loss_curve = clf.loss_curve_
        fig_loss = go.Figure()
        fig_loss.add_trace(
            go.Scatter(
                x=list(range(1, len(loss_curve) + 1)),
                y=loss_curve,
                mode="lines",
                name="Trénovací ztráta (Loss)",
                line=dict(color="#6366f1", width=2.5),
            )
        )
        fig_loss.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(title="Epocha (iterace)"),
            yaxis=dict(title="Ztráta (Cross-Entropy Loss)"),
        )
        st.plotly_chart(fig_loss, width="stretch")
    else:
        st.write("Optimalizátor nezaznamenal křivku ztráty.")

    st.subheader("🔍 Architektura vrstev")
    layer_info = []
    layer_info.append({"Vrstva": "Vstupní vrstva", "Typ": "Input", "Dimenze": "2 neurony", "Aktivace": "Lineární"})
    for idx, size in enumerate(hidden_sizes, 1):
        layer_info.append({
            "Vrstva": f"Skrytá vrstva {idx}",
            "Typ": "Dense",
            "Dimenze": f"{size} neuronů",
            "Aktivace": activation.upper(),
        })
    layer_info.append({"Vrstva": "Výstupní vrstva", "Typ": "Output", "Dimenze": "1 neuron (Softmax/Sigmoid)", "Aktivace": "Logistic"})
    st.dataframe(pd.DataFrame(layer_info), hide_index=True, width="stretch")

st.divider()

# --- Pedagogické experimenty & výzvy ---
st.subheader("🧪 4 experimenty, které si musíš vyzkoušet:")

exp1, exp2 = st.columns(2)
with exp1:
    st.markdown("""
    #### 1. Proč 1 neuron nestačí na XOR?
    * Vyber dataset **XOR (Čtyři kvadranty)**.
    * Nastav **1 skrytou vrstvu** se **2 neurony**.
    * Podívej se na rozhodovací plochu: 2 neurony vytvoří dvě přímky, které dokážou ohraničit křížový vzor. 
    * *Zkus snížit neurony na 1:* síť dokáže vést jen jedinou rovnou přímku a přesnost spadne na 50 % (čistý tip).
    """)

    st.markdown("""
    #### 2. Vanishing Gradient u Sigmoidu vs. ReLU
    * Vyber dataset **Dvojitá spirála (Spiral)**.
    * Zvol **2 skryté vrstvy** (např. 24 a 16 neuronů).
    * Přepni aktivaci na **Logistic (Sigmoid)** s malým learning rate `0.005` $\\to$ všimni si, jak se ztráta téměř nepohne (mizející gradient).
    * Přepni na **ReLU** se stejnými parametry $\\to$ síť bleskově prorazí a začne kreslit spirálové laloky.
    """)

with exp2:
    st.markdown("""
    #### 3. Tvar rozhodovací hranice: Lomené čáry vs. Hladké křivky
    * Porovnej **ReLU** a **Tanh** na datasetu **Dva půlměsíce (Moons)**:
      * **ReLU** skládá po částech lineární plochy $\\to$ výsledná hranice je mnohoúhelníková (lomená).
      * **Tanh / Sigmoid** má všude spojité hladké derivace $\\to$ výsledná hranice je krásně zaoblená.
    """)

    st.markdown("""
    #### 4. Vliv regularizace L2 (Alpha) na přeučení
    * Nastav vysoký počet neuronů (např. 32 a 24) a nízký šum.
    * Zvol $\\alpha = 10^{-5}$ $\\to$ hranice bude detailně obtékat každý jednotlivý odlehlý bod (overfitting).
    * Zvyš $\\alpha = 1.0$ $\\to$ váhy se penalizují, hranice se zjednoduší a vyhladí (vyšší bias, nižší variance).
    """)
