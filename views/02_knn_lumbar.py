import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer

st.title("🏥 Cvičení 1: Diagnostika bederní páteře (k-NN)")
st.caption("Biomedicínská klasifikace patologií páteře (Hernia & Spondylolisthesis) na základě 6 biomechanických parametrů pánve.")

base_dir = Path(__file__).resolve().parent.parent
data_dir = base_dir / "02_Classification" / "data"
raw_csv_path = data_dir / "lumbar_data.csv"
norm_csv_path = data_dir / "lumbar_normalized_df.csv"
json_path = data_dir / "lumbar_knn_precomputed.json"


@st.cache_data
def load_lumbar_data():
    raw_df = pd.read_csv(raw_csv_path)
    norm_df = pd.read_csv(norm_csv_path)
    return raw_df, norm_df


@st.cache_data
def load_precomputed():
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_resource
def get_trained_lumbar_models():
    _, norm_df = load_lumbar_data()
    X = norm_df.drop("class", axis=1)
    y = norm_df["class"]

    model_k5 = KNeighborsClassifier(n_neighbors=5).fit(X, y)
    model_opt = KNeighborsClassifier(n_neighbors=8).fit(X, y)
    return model_k5, model_opt, X.columns.tolist()


raw_df, norm_df = load_lumbar_data()
precomputed = load_precomputed()
model_k5, model_opt, feature_cols = get_trained_lumbar_models()

# =============================================================================
# METRICKÉ KARTY
# =============================================================================
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        "Celkem pacientů",
        f"{len(raw_df)}",
        "210 patologických / 100 zdravých",
        help="310 diagnostikovaných pacientů s 6 parametry pánve a páteře.",
    )
with col2:
    st.metric(
        "Školní testovací přesnost",
        f"{precomputed['school_metrics']['accuracy']*100:.1f} %",
        f"k = {precomputed['chosen_k']}",
        help="Základní 75/25 rozdělení bez stratifikace.",
    )
with col3:
    st.metric(
        "Senzitivita (Recall)",
        f"{precomputed['school_metrics']['recall']*100:.1f} %",
        "Záchyt patologií",
        help="Procento správně odhalených patologických stavů z celkového počtu nemocných.",
    )
with col4:
    st.metric(
        "Stratifikovaný model (k=8)",
        f"{precomputed['strat_metrics']['accuracy']*100:.1f} %",
        f"Recall {precomputed['strat_metrics']['recall']*100:.1f} %",
        help="Optimální model s vyváženým rozdělením tříd.",
    )

st.markdown("---")

tab1, tab2, tab3 = st.tabs([
    "📊 Diagnostika & Analýza chyb",
    "🔬 Biomechanický průzkum (2D/3D)",
    "🎛️ What-If Diagnostický kalkulátor pacienta",
])

# =============================================================================
# TAB 1: DIAGNOSTIKA & CHYBY
# =============================================================================
with tab1:
    st.subheader("Klinická diagnostika a matice záměn")
    st.markdown(
        "V medicínských úlohách není celková přesnost (*Accuracy*) jediným ani nejdůležitějším kritériem. "
        "Klíčová je **Senzitivita (Recall)**, protože nerozpoznaná patologie páteře (tzv. *False Negative*) "
        "může vést k chronickému poškození zdraví pacienta."
    )

    dcol1, dcol2 = st.columns(2)
    with dcol1:
        cm_data = np.array(precomputed["confusion_matrix"])
        fig_cm = px.imshow(
            cm_data,
            labels=dict(x="Predikovaný stav", y="Skutečný stav", color="Počet"),
            x=["0 (Normal)", "1 (Abnormal)"],
            y=["0 (Normal)", "1 (Abnormal)"],
            text_auto=True,
            color_continuous_scale="Blues",
            title=f"Konfúzní matice testovací sady (k=5, N_test={cm_data.sum()})",
        )
        fig_cm.update_layout(height=380, template="plotly_white")
        st.plotly_chart(fig_cm, width="stretch")

    with dcol2:
        tn, fp = cm_data[0]
        fn, tp = cm_data[1]
        st.markdown("#### Rozpad klinického rizika:")
        st.success(f"✅ **True Positives (Správně zachycená patologie): {tp} pacientů**")
        st.info(f"✅ **True Negatives (Správně potvrzeno zdraví): {tn} pacientů**")
        st.warning(f"⚠️ **False Positives (Falešný poplach – zdravý poslán na MRI): {fp} pacientů**")
        st.error(f"🚨 **False Negatives (Kritická chyba – přehlédnutá patologie!): {fn} pacientů**")

        st.caption(
            f"Poměr falešně negativních nálezů: Pouze **{fn / (fn + tp) * 100:.1f} %** nemocných pacientů "
            "nebylo modelem správně zachyceno při základním nastavení."
        )

    st.markdown("#### Vliv počtu sousedů k na diagnostickou přesnost")
    k_vals = precomputed["k_values"]
    fig_k = go.Figure()
    fig_k.add_trace(go.Scatter(
        x=k_vals,
        y=[s * 100 for s in precomputed["test_scores_unstrat"]],
        mode="lines+markers",
        name="Základní split (Test Acc)",
        line=dict(color="#e74c3c", dash="dash"),
    ))
    fig_k.add_trace(go.Scatter(
        x=k_vals,
        y=[s * 100 for s in precomputed["test_scores_strat"]],
        mode="lines+markers",
        name="Stratifikovaný split (Test Acc)",
        line=dict(color="#2980b9", width=3),
    ))
    fig_k.add_vline(x=5, line_width=2, line_dash="dot", line_color="orange", annotation_text="Školní k=5")
    fig_k.add_vline(x=precomputed["best_k_strat"], line_width=2, line_dash="dot", line_color="green", annotation_text=f"Optimum k={precomputed['best_k_strat']}")
    fig_k.update_layout(
        xaxis_title="Počet sousedů (k)",
        yaxis_title="Accuracy (%)",
        template="plotly_white",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_k, width="stretch")

# =============================================================================
# TAB 2: BIOMECHANICKÝ PRŮZKUM
# =============================================================================
with tab2:
    st.subheader("Vzájemné vztahy biomechanických parametrů")
    st.markdown(
        "Prozkoumejte rozložení pacientů podle jednotlivých úhlů páteře. Všimněte si silné korelace "
        "mezi `pelvic_incidence` a `sacral_slope` (platí anatomický vztah $\\text{Incidence} \\approx \\text{Tilt} + \\text{Slope}$) "
        "a jak vysoký `degree_spondylolisthesis` jasně odděluje těžké patologie."
    )

    bcol1, bcol2 = st.columns(2)
    with bcol1:
        fx = st.selectbox("Osa X:", feature_cols, index=0)
    with bcol2:
        fy = st.selectbox("Osa Y:", feature_cols, index=5)

    fig_scatter = px.scatter(
        raw_df,
        x=fx,
        y=fy,
        color="class",
        color_discrete_map={"Normal": "#2ecc71", "Hernia": "#e67e22", "Spondylolisthesis": "#e74c3c"},
        labels={"class": "Diagnóza"},
        title=f"Vztah {fx} vs. {fy} podle diagnózy pacienta",
        hover_data=feature_cols,
    )
    fig_scatter.update_layout(template="plotly_white", height=500)
    st.plotly_chart(fig_scatter, width="stretch")

# =============================================================================
# TAB 3: WHAT-IF KALKULÁTOR PACIENTA
# =============================================================================
with tab3:
    st.subheader("🎛️ What-If simulátor diagnózy pacienta s vysvětlením sousedů")
    st.markdown(
        "Zadejte naměřené biomechanické parametry z rentgenu pacienta. Algoritmus k-NN normalizuje "
        "hodnoty $L_2$ normou a porovná profil pacienta s databází 310 dřívějších klinických případů."
    )

    pcol1, pcol2, pcol3 = st.columns(3)
    with pcol1:
        in_pi = st.slider("Pánevní index (pelvic_incidence):", min_value=25.0, max_value=130.0, value=60.0, step=0.5)
        in_pt = st.slider("Sklon pánve (pelvic_tilt):", min_value=-10.0, max_value=50.0, value=17.5, step=0.5)
    with pcol2:
        in_lla = st.slider("Úhel bederní lordózy (lumbar_lordosis_angle):", min_value=15.0, max_value=130.0, value=50.0, step=0.5)
        in_ss = st.slider("Sklon kosti křížové (sacral_slope):", min_value=10.0, max_value=125.0, value=42.0, step=0.5)
    with pcol3:
        in_pr = st.slider("Poloměr pánve (pelvic_radius):", min_value=65.0, max_value=165.0, value=118.0, step=0.5)
        in_ds = st.slider("Stupeň spondylolistézy (degree_spondylolisthesis):", min_value=-15.0, max_value=200.0, value=15.0, step=0.5)

    user_k = st.slider("Počet referenčních sousedů (k):", min_value=1, max_value=15, value=5, step=2)

    # Příprava vstupních dat a normalizace L2
    input_raw = np.array([[in_pi, in_pt, in_lla, in_ss, in_pr, in_ds]])
    input_norm = Normalizer(norm="l2").transform(input_raw)
    input_norm_df = pd.DataFrame(input_norm, columns=feature_cols)

    pred_class_code = model_k5.predict(input_norm_df)[0]
    pred_prob = model_k5.predict_proba(input_norm_df)[0]

    st.markdown("### Výsledek predikce modelu:")
    res_col1, res_col2 = st.columns([1, 2])
    with res_col1:
        if pred_class_code == 1:
            st.error("### ⚠️ Závěr: **ABNORMAL (Patologický nález)**")
            st.write(f"Pravděpodobnost patologie: `{pred_prob[1]*100:.1f} %`")
            st.progress(float(pred_prob[1]))
            st.caption("Doporučeno podrobné zobrazení MRI pro vyloučení hernie disku nebo spondylolistézy.")
        else:
            st.success("### ✅ Závěr: **NORMAL (Fyziologický stav)**")
            st.write(f"Pravděpodobnost normálního stavu: `{pred_prob[0]*100:.1f} %`")
            st.progress(float(pred_prob[0]))
            st.caption("Biomechanické parametry odpovídají zdravému kontrolnímu vzorku.")

    with res_col2:
        # Metoda .kneighbors() pro vysvětlení sousedů
        X_all_norm = norm_df.drop("class", axis=1).values
        distances, indices = model_k5.kneighbors(input_norm, n_neighbors=user_k)

        st.markdown(f"#### 🔍 {user_k} nejbližších klinických případů (XAI vysvětlení):")
        neighbors_raw = raw_df.iloc[indices[0]].copy()
        neighbors_raw.insert(0, "Vzdálenost (d)", np.round(distances[0], 4))
        st.dataframe(
            neighbors_raw[["Vzdálenost (d)", "class", "pelvic_incidence", "pelvic_tilt", "lumbar_lordosis_angle", "sacral_slope", "pelvic_radius", "degree_spondylolisthesis"]],
            hide_index=True,
            width="stretch",
        )
