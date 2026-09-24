import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.title("🐧 Cvičení 2: Klasifikace tučňáků (k-NN) na Palmer Penguins")
st.caption("Interaktivní analýza modelu k nejbližších sousedů, rozhodovacích hranic v 2D prostoru, vlivu škálování a What-If predikce druhu s inspekcí sousedů.")

base_dir = Path(__file__).resolve().parent.parent
data_dir = base_dir / "02_Classification" / "data"
csv_path = data_dir / "penguins_size.csv"
json_path = data_dir / "penguins_knn_precomputed.json"


@st.cache_data
def load_penguins_data():
    df = pd.read_csv(csv_path)
    clean_df = df.copy()
    clean_df = clean_df[clean_df["sex"] != "."]
    clean_df.dropna(inplace=True)
    return df, clean_df


@st.cache_data
def load_precomputed_results():
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_resource
def get_trained_full_pipeline():
    _, clean_df = load_penguins_data()
    X = clean_df.drop("species", axis=1)
    y = clean_df["species"]

    num_cols = ["culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g"]
    cat_cols = ["island", "sex"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
        ]
    )

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("knn", KNeighborsClassifier(n_neighbors=5, weights="uniform", p=2)),
    ])
    pipeline.fit(X, y)
    return pipeline, clean_df


raw_df, clean_df = load_penguins_data()
precomputed = load_precomputed_results()
full_pipeline, cached_clean_df = get_trained_full_pipeline()

# =============================================================================
# METRICKÉ KARTY
# =============================================================================
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        "Školní baseline",
        f"{precomputed['school_acc']*100:.1f} %",
        help="Model ze slajdů kurzu: k=5, neškálováno, vyhozen ostrov a pohlaví.",
    )
with col2:
    st.metric(
        "StandardScaler (k=5)",
        f"{precomputed['scaling_results']['StandardScaler (Z-score)']*100:.1f} %",
        f"+{(precomputed['scaling_results']['StandardScaler (Z-score)'] - precomputed['school_acc'])*100:.1f} p.b.",
        help="Stejné příznaky a k=5, ale s odstraněním dominance gramů.",
    )
with col3:
    st.metric(
        "Optimální k (k=6)",
        "100.0 %",
        "+19.4 p.b.",
        help="Nejvyšší přesnost na testovací sadě při k=6 s normalizací.",
    )
with col4:
    st.metric(
        "Plný Pipeline (CV)",
        f"{precomputed['full_cv_acc']*100:.1f} %",
        "Včetně ostrova & pohlaví",
        help="5-násobná StratifiedKFold validace s GridSearchCV.",
    )

st.markdown("---")

tab1, tab2, tab3 = st.tabs([
    "🗺️ 2D Rozhodovací hranice (Boundary Map)",
    "📊 Diagnostika: Škálování & Křivka k",
    "🎛️ What-If Klasifikátor tučňáka (XAI)",
])

# =============================================================================
# TAB 1: 2D ROZHODOVACÍ HRANICE
# =============================================================================
with tab1:
    st.subheader("Interaktivní simulátor rozhodovacích hranic k-NN")
    st.markdown(
        "Zvolte libovolné dva příznaky a sledujte, jak se v reálném čase mění tvar rozhodovacích hranic "
        "v závislosti na parametru $k$, metrice vzdálenosti a **především na tom, zda zapnete standardizaci**!"
    )

    fcol1, fcol2, fcol3, fcol4 = st.columns(4)
    with fcol1:
        feat_x = st.selectbox(
            "Osa X (Příznak 1):",
            ["culmen_length_mm", "flipper_length_mm", "body_mass_g", "culmen_depth_mm"],
            index=0,
        )
    with fcol2:
        feat_y = st.selectbox(
            "Osa Y (Příznak 2):",
            ["culmen_depth_mm", "flipper_length_mm", "body_mass_g", "culmen_length_mm"],
            index=0,
        )
    with fcol3:
        param_k = st.slider("Počet sousedů (k):", min_value=1, max_value=35, value=5, step=2)
    with fcol4:
        use_scaling = st.toggle("Aktivovat StandardScaler", value=True)

    c_metric1, c_metric2 = st.columns(2)
    with c_metric1:
        metric_choice = st.radio("Metrika vzdálenosti:", ["Eukleidovská (L2)", "Manhattanská (L1)"], horizontal=True)
    with c_metric2:
        weights_choice = st.radio("Vážení sousedů:", ["Rovnoměrné (uniform)", "Dle vzdálenosti (1/d)"], horizontal=True)

    if feat_x == feat_y:
        st.warning("Zvolte prosím dva různé příznaky pro osu X a Y.")
    else:
        # Příprava dat pro 2D klasifikátor
        p = 1 if "Manhattanská" in metric_choice else 2
        w = "distance" if "vzdálenosti" in weights_choice else "uniform"

        X_2d = clean_df[[feat_x, feat_y]].values
        y_2d = clean_df["species"].values
        species_unique = sorted(list(set(y_2d)))
        class_to_int = {cls_name: i for i, cls_name in enumerate(species_unique)}
        int_to_class = {i: cls_name for i, cls_name in enumerate(species_unique)}
        y_2d_int = np.array([class_to_int[label] for label in y_2d])

        # Trénování modelu s/bez škálování
        if use_scaling:
            pipe_2d = Pipeline([
                ("scaler", StandardScaler()),
                ("knn", KNeighborsClassifier(n_neighbors=param_k, p=p, weights=w)),
            ])
        else:
            pipe_2d = KNeighborsClassifier(n_neighbors=param_k, p=p, weights=w)

        pipe_2d.fit(X_2d, y_2d_int)

        # Mřížka pro kontury
        x_min, x_max = X_2d[:, 0].min() - 0.05 * (X_2d[:, 0].max() - X_2d[:, 0].min()), X_2d[:, 0].max() + 0.05 * (X_2d[:, 0].max() - X_2d[:, 0].min())
        y_min, y_max = X_2d[:, 1].min() - 0.05 * (X_2d[:, 1].max() - X_2d[:, 1].min()), X_2d[:, 1].max() + 0.05 * (X_2d[:, 1].max() - X_2d[:, 1].min())

        grid_steps = 100
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, grid_steps),
            np.linspace(y_min, y_max, grid_steps),
        )
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = pipe_2d.predict(grid_points).reshape(xx.shape)

        # Vytvoření Plotly grafu
        fig_boundary = go.Figure()

        # 1. Konturová vrstva oblastí
        colorscale = [[0.0, "rgba(99, 110, 250, 0.25)"], [0.5, "rgba(239, 85, 59, 0.25)"], [1.0, "rgba(0, 204, 150, 0.25)"]]
        fig_boundary.add_trace(go.Contour(
            x=np.linspace(x_min, x_max, grid_steps),
            y=np.linspace(y_min, y_max, grid_steps),
            z=Z,
            showscale=False,
            colorscale=colorscale,
            contours=dict(start=0, end=2, size=1),
            line=dict(width=1, color="rgba(100, 100, 100, 0.4)"),
            hoverinfo="skip",
        ))

        # 2. Skutečná pozorování (Scatter)
        color_map = {"Adelie": "#636EFA", "Chinstrap": "#EF553B", "Gentoo": "#00CC96"}
        for sp in species_unique:
            mask = clean_df["species"] == sp
            fig_boundary.add_trace(go.Scatter(
                x=clean_df.loc[mask, feat_x],
                y=clean_df.loc[mask, feat_y],
                mode="markers",
                name=sp,
                marker=dict(size=8, color=color_map[sp], line=dict(width=1, color="white")),
                hovertemplate=f"<b>Druh: {sp}</b><br>{feat_x}: %{{x:.1f}}<br>{feat_y}: %{{y:.1f}}<extra></extra>",
            ))

        fig_boundary.update_layout(
            title=f"Rozhodovací hranice k-NN pro k={param_k} ({'StandardScaler ZAPNUT' if use_scaling else 'BEZ ŠKÁLOVÁNÍ - DEFEKT VZDÁLENOSTÍ!'})",
            xaxis_title=feat_x,
            yaxis_title=feat_y,
            template="plotly_white",
            height=540,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_boundary, use_container_width=True)

        if not use_scaling and ("body_mass_g" in [feat_x, feat_y]):
            st.error(
                "🚨 **Pozor na efekt neškálování:** Podívejte se, jak se hranice deformovaly na rovnoběžné horizontální/vertikální pruhy! "
                "Hmotnost `body_mass_g` v tisících gramů zcela přehlušila druhou proměnnou měřenou v milimetrech. Zapněte přepínač *Aktivovat StandardScaler* pro záchranu modelu."
            )

# =============================================================================
# TAB 2: DIAGNOSTIKA & VLIV K
# =============================================================================
with tab2:
    st.subheader("Hluboká srovnávací diagnostika")
    dcol1, dcol2 = st.columns(2)

    with dcol1:
        st.markdown("#### Srovnání metod předzpracování (k=5)")
        scaling_dict = precomputed["scaling_results"]
        fig_scaling = px.bar(
            x=list(scaling_dict.keys()),
            y=[v * 100 for v in scaling_dict.values()],
            color=list(scaling_dict.keys()),
            color_discrete_sequence=["#e74c3c", "#3498db", "#2ecc71", "#f39c12"],
            labels={"x": "Metoda škálování", "y": "Testovací Accuracy (%)"},
            title="Přesnost modelu k-NN dle metody škálování",
        )
        fig_scaling.update_yaxes(range=[70, 102])
        fig_scaling.update_layout(showlegend=False, template="plotly_white", height=380)
        st.plotly_chart(fig_scaling, use_container_width=True)

    with dcol2:
        st.markdown("#### Konfúzní matice plného modelu (Pipeline)")
        cm_data = np.array(precomputed["confusion_matrix"])
        classes = precomputed["classes"]
        fig_cm = px.imshow(
            cm_data,
            labels=dict(x="Predikovaná třída", y="Skutečná třída", color="Počet"),
            x=classes,
            y=classes,
            text_auto=True,
            color_continuous_scale="Blues",
            title=f"Konfúzní matice na testovacích datech (Accuracy = {precomputed['full_test_acc']*100:.1f} %)",
        )
        fig_cm.update_layout(height=380, template="plotly_white")
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("#### Bias-Variance Tradeoff: Křivka závislosti přesnosti na počtu sousedů $k$")
    k_vals = precomputed["k_range"]
    fig_k = go.Figure()
    fig_k.add_trace(go.Scatter(
        x=k_vals,
        y=[v * 100 for v in precomputed["test_scores_unscaled"]],
        mode="lines+markers",
        name="Neškálováno (Test Accuracy)",
        line=dict(color="#e74c3c", dash="dash"),
    ))
    fig_k.add_trace(go.Scatter(
        x=k_vals,
        y=[v * 100 for v in precomputed["train_scores_scaled"]],
        mode="lines",
        name="StandardScaler (Train Accuracy)",
        line=dict(color="#2980b9", dash="dot"),
    ))
    fig_k.add_trace(go.Scatter(
        x=k_vals,
        y=[v * 100 for v in precomputed["test_scores_scaled"]],
        mode="lines+markers",
        name="StandardScaler (Test Accuracy)",
        line=dict(color="#27ae60", width=3),
    ))
    fig_k.add_vline(
        x=precomputed["best_k_scaled"],
        line_width=2,
        line_dash="dot",
        line_color="green",
        annotation_text=f"Optimum k={precomputed['best_k_scaled']}",
    )
    fig_k.update_layout(
        xaxis_title="Počet sousedů (k)",
        yaxis_title="Accuracy (%)",
        template="plotly_white",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_k, use_container_width=True)

# =============================================================================
# TAB 3: WHAT-IF SIMULÁTOR INFERENCE S INSPEKCÍ SOUSEDŮ (.kneighbors)
# =============================================================================
with tab3:
    st.subheader("🎛️ What-If klasifikátor tučňáka s vysvětlitelným AI (XAI)")
    st.markdown(
        "Zadejte biometrické a geografické parametry tučňáka. Plně natrénovaný produkční Pipeline "
        "určí nejpravděpodobnější druh a pomocí metody `.kneighbors()` vám **zobrazí $k$ nejbližších reálných tučňáků** "
        "z Antarktidy, na základě kterých model rozhodl!"
    )

    wcol1, wcol2, wcol3 = st.columns(3)
    with wcol1:
        in_cul_len = st.slider("Délka zobáku (culmen_length_mm):", min_value=30.0, max_value=60.0, value=45.0, step=0.5)
        in_cul_dep = st.slider("Výška zobáku (culmen_depth_mm):", min_value=12.0, max_value=22.0, value=17.5, step=0.5)
    with wcol2:
        in_flip_len = st.slider("Délka ploutve (flipper_length_mm):", min_value=170, max_value=235, value=200, step=1)
        in_mass = st.slider("Hmotnost těla (body_mass_g):", min_value=2500, max_value=6500, value=4200, step=50)
    with wcol3:
        in_island = st.selectbox("Ostrov odchytu (island):", ["Biscoe", "Dream", "Torgersen"])
        in_sex = st.selectbox("Pohlaví (sex):", ["MALE", "FEMALE"])
        in_k = st.slider("Počet sousedů k pro vysvětlení:", min_value=1, max_value=11, value=5, step=2)

    input_df = pd.DataFrame([{
        "culmen_length_mm": in_cul_len,
        "culmen_depth_mm": in_cul_dep,
        "flipper_length_mm": in_flip_len,
        "body_mass_g": in_mass,
        "island": in_island,
        "sex": in_sex,
    }])

    pred_species = full_pipeline.predict(input_df)[0]
    pred_proba = full_pipeline.predict_proba(input_df)[0]
    classes = full_pipeline.classes_

    st.markdown("### Výsledek predikce modelu:")
    pcol1, pcol2 = st.columns([1, 2])
    with pcol1:
        st.success(f"### 🐧 Druh: **{pred_species}**")
        for cls_name, prob in zip(classes, pred_proba):
            st.write(f"Pravděpodobnost **{cls_name}**: `{prob * 100:.1f} %`")
            st.progress(float(prob))

    with pcol2:
        # Inspekce .kneighbors() přes transformovaný prostor
        preprocessor = full_pipeline.named_steps["prep"]
        knn_estimator = full_pipeline.named_steps["knn"]

        X_all = cached_clean_df.drop("species", axis=1)
        X_all_trans = preprocessor.transform(X_all)
        input_trans = preprocessor.transform(input_df)

        distances, indices = knn_estimator.kneighbors(input_trans, n_neighbors=in_k)

        st.markdown(f"#### 🔍 {in_k} nejbližších sousedů z datasetu (XAI vysvětlení):")
        neighbors_df = cached_clean_df.iloc[indices[0]].copy()
        neighbors_df.insert(0, "Vzdálenost (d)", np.round(distances[0], 3))
        st.dataframe(
            neighbors_df[["Vzdálenost (d)", "species", "island", "culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g", "sex"]],
            hide_index=True,
            use_container_width=True,
        )
