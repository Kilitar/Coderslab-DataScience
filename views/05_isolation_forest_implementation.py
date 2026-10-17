"""
Den 5: Neřízené učení – Vzorová implementace Isolation Forest v Scikit-learn
=============================================================================
Syntéza materiálů z přednášky Isolation_Forest_-_sample_implementation.pdf:
1. Generování shlukových dat make_blobs (1 000 bodů, centers=1, cluster_std=3.0, random_state=42).
2. Vzorová implementace IsolationForest(n_estimators=100, contamination=0.05).
3. Vizualizace detekovaných anomálií a izočar rozhodovací funkce (.decision_function()).
4. Interaktivní laboratoř parametrů (What-If simulátor pro contamination a n_estimators).
5. Porovnávací benchmark na datech s proměnlivou hustotou (iForest vs. LOF) a bilance silných/slabých stránek.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import streamlit as st

st.set_page_config(page_title="Implementace: Isolation Forest", page_icon="🛠️", layout="wide")

st.title("🛠️ Vzorová implementace: Isolation Forest v Scikit-learn")
st.caption(
    "Oficiální vzorová implementace z přednášky kurzu (Isolation Forest - sample implementation). "
    "Generování shluku make_blobs (1 000 bodů, 5 % anomálií), 100 stromů, analýza spojité funkce .decision_function(), "
    "2D kontury rozhodovací plochy a vyhodnocení silných i slabých stránek algoritmu."
)


@st.cache_data
def load_isolation_forest_data():
    candidates_json = [
        Path("05_Unsupervised_Learning/data/isolation_forest_precomputed.json"),
        Path(__file__).resolve().parent.parent / "05_Unsupervised_Learning/data/isolation_forest_precomputed.json",
    ]
    for p in candidates_json:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    if isinstance(d, dict) and "points" in d and "metrics" in d:
                        return d
            except Exception:
                pass

    # Fallback on-the-fly výpočet (< 40 ms)
    X, _ = make_blobs(n_samples=1000, centers=1, cluster_std=3.0, random_state=42)
    iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    preds = iso.fit_predict(X)
    dec = iso.decision_function(X)

    pts = []
    for i in range(len(X)):
        pts.append({
            "id": i,
            "x_1": float(round(X[i, 0], 4)),
            "x_2": float(round(X[i, 1], 4)),
            "is_anomaly": int(preds[i]),
            "decision_function": float(round(dec[i], 4)),
            "anomaly_score": float(round(-iso.score_samples(X[i:i+1])[0], 4))
        })

    x_min, x_max = X[:, 0].min() - 2, X[:, 0].max() + 2
    y_min, y_max = X[:, 1].min() - 2, X[:, 1].max() + 2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
    zz = iso.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    return {
        "dataset_meta": {"n_samples": 1000, "contamination": 0.05, "n_estimators": 100},
        "metrics": {
            "n_anomalies": int(np.sum(preds == -1)),
            "n_inliers": int(np.sum(preds == 1)),
            "min_decision": float(round(np.min(dec), 4)),
            "max_decision": float(round(np.max(dec), 4)),
            "mean_decision": float(round(np.mean(dec), 4)),
            "threshold_offset": float(round(iso.offset_, 4))
        },
        "points": pts,
        "contour": {
            "x": np.linspace(x_min, x_max, 50).tolist(),
            "y": np.linspace(y_min, y_max, 50).tolist(),
            "z": zz.tolist()
        },
        "multi_density_benchmark": []
    }


data = load_isolation_forest_data()
metrics = data["metrics"]
pts_df = pd.DataFrame(data["points"])

# Horní přehledové metriky
m_col1, m_col2, m_col3, m_col4 = st.columns(4)
m_col1.metric("Celkem pozorování", f"{len(pts_df):,}".replace(",", " "))
m_col2.metric("Běžná pozorování (Inliers)", f"{metrics['n_inliers']} bodů", delta=f"{metrics['n_inliers'] / len(pts_df) * 100:.1f} %")
m_col3.metric("Detekované anomálie (Outliers)", f"{metrics['n_anomalies']} bodů", delta=f"{metrics['n_anomalies'] / len(pts_df) * 100:.1f} %", delta_color="inverse")
m_col4.metric("Práh decision_function", "0.000", help="Záporné hodnoty značí anomálii, kladné inlier.")

tab_slides, tab_func, tab_lab, tab_crit = st.tabs([
    "🎯 Výsledky ze slajdů (5 %)",
    "📊 Rozhodovací funkce & Skóre",
    "🎛️ Živá laboratoř parametrů",
    "⚖️ Výhody, nevýhody & Multi-Density"
])

# -----------------------------------------------------------------------------
# TAB 1: Výsledky ze slajdů
# -----------------------------------------------------------------------------
with tab_slides:
    st.subheader("🎯 Vzorová implementace ze slajdů 14–19")
    st.markdown(
        r"""
        V prezentaci byl vygenerován syntetický shluk (`make_blobs`, $N=1\,000$, $\text{centers}=1$, $\text{std}=3.0$) 
        a natrénován model `IsolationForest(n_estimators=100, contamination=0.05, random_state=42)`.
        """
    )

    g_col1, g_col2 = st.columns(2)

    with g_col1:
        st.markdown("**1. Klasifikace bodů (Odpovídá slajdu 18):**")
        fig_scatter = px.scatter(
            pts_df,
            x="x_1",
            y="x_2",
            color=pts_df["is_anomaly"].map({1: "Běžné body (950)", -1: "Anomálie (50)"}),
            color_discrete_map={"Běžné body (950)": "#3b82f6", "Anomálie (50)": "#ef4444"},
            hover_data={"x_1": ":.2f", "x_2": ":.2f", "decision_function": ":.3f", "id": True},
            labels={"x_1": "x_1", "x_2": "x_2", "color": "Klasifikace"},
            title="Dataset with outliers detected by Isolation Forest"
        )
        fig_scatter.update_traces(marker=dict(size=7, opacity=0.65))
        fig_scatter.for_each_trace(
            lambda t: t.update(marker=dict(size=11, line=dict(width=1.5, color="black"))) if "Anomálie" in t.name else None
        )
        fig_scatter.update_layout(height=460, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_scatter, width="stretch")

    with g_col2:
        st.markdown("**2. Konturová mapa rozhodovací plochy (Decision Boundary):**")
        contour_info = data.get("contour", {})
        if contour_info and "x" in contour_info:
            fig_cont = go.Figure()
            # Contour
            fig_cont.add_trace(
                go.Contour(
                    x=contour_info["x"],
                    y=contour_info["y"],
                    z=contour_info["z"],
                    colorscale="Reds_r",
                    contours=dict(start=-0.25, end=0.0, size=0.04),
                    opacity=0.45,
                    showscale=False,
                    name="Anomální zóna"
                )
            )
            # Inliers
            inliers = pts_df[pts_df["is_anomaly"] == 1]
            outliers = pts_df[pts_df["is_anomaly"] == -1]
            fig_cont.add_trace(
                go.Scatter(
                    x=inliers["x_1"],
                    y=inliers["x_2"],
                    mode="markers",
                    marker=dict(color="#3b82f6", size=6, opacity=0.55),
                    name="Inliers (950)"
                )
            )
            fig_cont.add_trace(
                go.Scatter(
                    x=outliers["x_1"],
                    y=outliers["x_2"],
                    mode="markers",
                    marker=dict(color="#ef4444", size=10, line=dict(width=1.2, color="black")),
                    name="Outliers (50)"
                )
            )
            fig_cont.update_layout(
                title="Rozhodovací plocha v 2D (Červené stínování = anomální zóna)",
                xaxis_title="x_1",
                yaxis_title="x_2",
                height=460,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_cont, width="stretch")

    st.markdown("---")
    st.markdown("#### Ukázkový kód z přednášky:")
    st.code(
        r"""from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest
import pandas as pd
import matplotlib.pyplot as plt

# 1. Generování dat
data, labels = make_blobs(n_samples=1000, centers=1, cluster_std=3, random_state=42)

# 2. Inicializace a trénování modelu (slajd 16)
isolation_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
preds = isolation_forest.fit_predict(data)

# 3. Zpracování výsledků do DataFrame (slajd 17)
blobs_df = pd.DataFrame(data, columns=["x_1", "x_2"])
blobs_df["is_anomaly"] = preds
blobs_df["anomaly_score"] = isolation_forest.decision_function(data)

# 4. Vizualizace (slajd 18)
normal_points_df = blobs_df[blobs_df["is_anomaly"] == 1]
anomaly_points_df = blobs_df[blobs_df["is_anomaly"] == -1]

plt.scatter(normal_points_df["x_1"], normal_points_df["x_2"], color="blue")
plt.scatter(anomaly_points_df["x_1"], anomaly_points_df["x_2"], color="red")
plt.title("Dataset with outliers detected by Isolation Forest")
plt.show()
""",
        language="python"
    )

# -----------------------------------------------------------------------------
# TAB 2: Rozhodovací funkce & Skóre
# -----------------------------------------------------------------------------
with tab_func:
    st.subheader("📊 Analýza spojité funkce .decision_function() (Slajd 20)")
    st.markdown(
        r"""
        Jak uvádí slajd 20: *„The number of outliers is directly affected by the contamination hyperparameter... 
        We could also manipulate their number by setting the cutoff point ourselves based on the anomaly score. 
        We can use the `.decision_function()` method.“*
        
        - V Scikit-learn je hodnota `decision_function` posunuta tak, že **0.0 odpovídá rozhodovacímu prahu**:
          - $\text{decision\_function} < 0 \implies \text{Anomálie (Outlier, označeno -1)}$
          - $\text{decision\_function} \ge 0 \implies \text{Běžné pozorování (Inlier, označeno +1)}$
        """
    )

    fig_hist = go.Figure()
    fig_hist.add_trace(
        go.Histogram(
            x=pts_df["decision_function"],
            nbinsx=40,
            marker_color="#3b82f6",
            opacity=0.7,
            name="Distribuce decision_function"
        )
    )
    fig_hist.add_vline(
        x=0.0,
        line_width=3,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text="Rozhodovací práh (0.0)",
        annotation_position="top right"
    )
    fig_hist.update_layout(
        title="Rozdělení skóre decision_function() v populaci 1 000 bodů",
        xaxis_title="Hodnota decision_function (záporné = anomálie, kladné = inliers)",
        yaxis_title="Počet bodů v intervalu",
        height=380,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_hist, width="stretch")

    st.markdown("#### Tabulka 50 detekovaných anomálií (seřazeno od nejextrémnějších):")
    outliers_table = pts_df[pts_df["is_anomaly"] == -1].sort_values(by="decision_function", ascending=True)
    st.dataframe(
        outliers_table[["id", "x_1", "x_2", "decision_function", "anomaly_score"]],
        width="stretch",
        hide_index=True
    )

# -----------------------------------------------------------------------------
# TAB 3: Živá laboratoř parametrů
# -----------------------------------------------------------------------------
with tab_lab:
    st.subheader("🎛️ Živá laboratoř: Interaktivní ladění hyperparametrů v reálném čase")
    st.markdown(
        """
        Vyzkoušejte si, jak se mění rozhodovací plocha a počet anomálií při změně kontaminace a počtu stromů:
        """
    )

    l_col1, l_col2 = st.columns(2)
    with l_col1:
        live_contam = st.slider(
            "Předpokládaná kontaminace (contamination %)",
            min_value=1.0,
            max_value=15.0,
            value=5.0,
            step=1.0,
            help="Procento bodů, které model a priori považuje za anomálie."
        ) / 100.0
    with l_col2:
        live_trees = st.slider(
            "Počet stromů v lese (n_estimators)",
            min_value=10,
            max_value=200,
            value=100,
            step=10,
            help="Vyšší počet stromů snižuje stochastický rozptyl náhodných řezů."
        )

    # Rychlý on-the-fly fit na 1000 bodech (< 25 ms)
    X_live = pts_df[["x_1", "x_2"]].values
    live_model = IsolationForest(n_estimators=live_trees, contamination=live_contam, random_state=42)
    live_preds = live_model.fit_predict(X_live)
    live_scores = live_model.decision_function(X_live)

    live_df = pts_df.copy()
    live_df["live_is_anomaly"] = (live_preds == -1)
    live_df["live_score"] = live_scores
    live_n_outliers = int(np.sum(live_preds == -1))

    r_c1, r_c2, r_c3 = st.columns(3)
    r_c1.metric("Detekované anomálie", f"{live_n_outliers} bodů", delta=f"{live_n_outliers / len(live_df) * 100:.1f} %")
    r_c2.metric("Běžná pozorování", f"{len(live_df) - live_n_outliers} bodů")
    r_c3.metric("Nejextrémnější skóre", f"{np.min(live_scores):.4f}")

    fig_live = px.scatter(
        live_df,
        x="x_1",
        y="x_2",
        color=live_df["live_is_anomaly"].map({False: "Inlier", True: "Anomálie"}),
        color_discrete_map={"Inlier": "#3b82f6", "Anomálie": "#e11d48"},
        hover_data={"x_1": ":.2f", "x_2": ":.2f", "live_score": ":.3f", "id": True},
        labels={"x_1": "x_1", "x_2": "x_2", "color": "Stav"},
        title=f"Živý model: Kontaminace = {live_contam * 100:.1f} %, Stromy = {live_trees}"
    )
    fig_live.update_traces(marker=dict(size=7, opacity=0.65))
    fig_live.for_each_trace(
        lambda t: t.update(marker=dict(size=11, line=dict(width=1.5, color="black"))) if t.name == "Anomálie" else None
    )
    fig_live.update_layout(height=520, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_live, width="stretch")

# -----------------------------------------------------------------------------
# TAB 4: Výhody, nevýhody & Multi-Density test
# -----------------------------------------------------------------------------
with tab_crit:
    st.subheader("⚖️ Bilance výhod a nevýhod (Slajdy 21–22) & Multi-Density test")

    b_col1, b_col2 = st.columns(2)
    with b_col1:
        with st.container(border=True):
            st.markdown("### ✅ Výhody (Advantages ze slajdu 22)")
            st.markdown(
                """
                - **Efektivita (Efficiency):** V porovnání s metodami založenými na vzdálenostech (k-NN, LOF) je iForest vysoce efektivní i při velkém počtu dimenzí.
                - **Škálovatelnost (Scalability):** Vynikající pro masivní datasety – staví mělké stromy na malých vzorcích (subsampling $\\psi = 256$).
                - **Nevyžaduje normalizaci:** Odpadá předpoklad normality i nutnost transformace příznaků (StandardScaler).
                """
            )
    with b_col2:
        with st.container(border=True):
            st.markdown("### ⚠️ Nevýhody (Disadvantages ze slajdu 22)")
            st.markdown(
                """
                - **Náhodnost (Randomness):** Výsledky se mohou mezi běhy mírně lišit kvůli náhodnému výběru os a řezů (nutnost zafixovat `random_state`).
                - **Problém se shluky anomálií:** Pokud anomálie vytvoří vlastní husté shluky, iForest je nedokáže snadno oddělit (vyžadují mnoho řezů).
                - **Axiální omezení:** Řezy jsou rovnoběžné s osami, což může selhávat u šikmých nelineárních závislostí.
                """
            )

    st.markdown("---")
    st.subheader("🔬 Srovnávací experiment: Kde triumfuje LOF a kde Isolation Forest?")
    st.markdown(
        """
        Co se stane, pokud máme **dva shluky s dramaticky odlišnou hustotou** (jeden hustý se std=0.8 a druhý řídký se std=2.5)?
        """
    )

    multi_pts = data.get("multi_density_benchmark", [])
    if multi_pts:
        multi_df = pd.DataFrame(multi_pts)
        c_m1, c_m2 = st.columns(2)
        with c_m1:
            fig_if = px.scatter(
                multi_df,
                x="x_1",
                y="x_2",
                color=multi_df["iforest_anomaly"].map({False: "Inlier", True: "Anomálie"}),
                color_discrete_map={"Inlier": "#3b82f6", "Anomálie": "#ef4444"},
                title="Isolation Forest (Preferuje globální okraj prostoru)"
            )
            fig_if.update_traces(marker=dict(size=7))
            fig_if.for_each_trace(lambda t: t.update(marker=dict(size=11, line=dict(width=1.5, color="black"))) if t.name == "Anomálie" else None)
            fig_if.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_if, width="stretch")

        with c_m2:
            fig_lof = px.scatter(
                multi_df,
                x="x_1",
                y="x_2",
                color=multi_df["lof_anomaly"].map({False: "Inlier", True: "Anomálie"}),
                color_discrete_map={"Inlier": "#3b82f6", "Anomálie": "#f59e0b"},
                title="Local Outlier Factor (Detekuje lokální anomálie uvnitř obou shluků)"
            )
            fig_lof.update_traces(marker=dict(size=7))
            fig_lof.for_each_trace(lambda t: t.update(marker=dict(size=11, line=dict(width=1.5, color="black"))) if t.name == "Anomálie" else None)
            fig_lof.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_lof, width="stretch")

        st.info(
            "💡 **Klíčový poznatek:** Isolation Forest vyhodnotil jako anomálie převážně body z řídkého shluku, "
            "protože jsou v globálním měřítku dále od sebe. LOF naopak správně nalezl lokální anomálie "
            "v poměru k hustotě každého z obou shluků!"
        )
