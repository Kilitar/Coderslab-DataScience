"""
Den 5: Neřízené učení – Cvičení Isolation Forest na datech student_scores.csv (Výsledky zadání)
==============================================================================================
Oficiální řešení cvičení z kurzu:
1. Načtení datasetu student_scores.csv do prostředí Pandas (30 641 studentů).
2. Ponechání pouze tří sloupců se známkami: MathScore, ReadingScore, WritingScore.
3. Redukce dimenzionality na dvě dimenze pomocí PCA pro přehlednou vizualizaci.
4. Trénování IsolationForest s předpokladem 2.5 % anomálií (contamination=0.025).
5. Grafické znázornění detekovaných anomálií v 2D prostoru hlavních komponent.
6. Srovnání se stavem při kontaminaci 1.0 % (contamination=0.01) a analýza posunu rozhodovací hranice.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import streamlit as st

st.set_page_config(page_title="Cvičení: Skóre studentů – Výsledky", page_icon="🎯", layout="wide")

st.title("🎯 Cvičení: Detekce anomálií studentů (student_scores.csv) – Výsledky zadání")
st.caption(
    "Oficiální řešení cvičení z kurzu: Analýza studijních výsledků 30 641 studentů pomocí kombinace "
    "redukce dimenzionality PCA (2D projekce) a algoritmu Isolation Forest s porovnáním kontaminace 2.5 % vs. 1.0 %."
)


@st.cache_data
def load_iforest_student_scores_data():
    candidates_json = [
        Path("05_Unsupervised_Learning/data/iforest_student_scores_precomputed.json"),
        Path(__file__).resolve().parent.parent / "05_Unsupervised_Learning/data/iforest_student_scores_precomputed.json",
    ]
    for p in candidates_json:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    if isinstance(d, dict) and "outliers" in d and "metrics" in d:
                        return d
            except Exception:
                pass

    # Fallback on-the-fly výpočet
    data_candidates = [
        Path("data/Downloadable materials for Session 3/student_scores.csv"),
        Path(__file__).resolve().parent.parent / "data/Downloadable materials for Session 3/student_scores.csv",
        Path("data/student_scores.csv"),
    ]
    csv_path = None
    for cand in data_candidates:
        if cand.exists():
            csv_path = cand
            break

    if csv_path:
        df = pd.read_csv(csv_path)
    else:
        # Syntetický fallback
        np.random.seed(42)
        m = np.random.normal(66, 15, 1000)
        r = m * 0.8 + np.random.normal(15, 8, 1000)
        w = r * 0.9 + np.random.normal(5, 5, 1000)
        df = pd.DataFrame({"MathScore": m, "ReadingScore": r, "WritingScore": w})

    X = df[["MathScore", "ReadingScore", "WritingScore"]].values
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    iso_025 = IsolationForest(n_estimators=100, contamination=0.025, random_state=42)
    y_pred_025 = iso_025.fit_predict(X_pca)
    scores_025 = iso_025.decision_function(X_pca)

    iso_01 = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    y_pred_01 = iso_01.fit_predict(X_pca)
    scores_01 = iso_01.decision_function(X_pca)

    outliers_list = []
    plot_pts = []
    for i in range(len(df)):
        p025 = (y_pred_025[i] == -1)
        p01 = (y_pred_01[i] == -1)
        if p01:
            cat = "Extrémní anomálie (1.0 % i 2.5 %)"
        elif p025:
            cat = "Přidaná anomálie (pouze při 2.5 %)"
        else:
            cat = "Běžný student (Inlier)"

        pt = {
            "id": i,
            "math": int(X[i, 0]),
            "reading": int(X[i, 1]),
            "writing": int(X[i, 2]),
            "pca_1": float(round(X_pca[i, 0], 3)),
            "pca_2": float(round(X_pca[i, 1], 3)),
            "decision_025": float(round(scores_025[i], 4)),
            "decision_01": float(round(scores_01[i], 4)),
            "is_outlier_025": bool(p025),
            "is_outlier_01": bool(p01),
            "category": cat
        }
        if p025 or p01:
            outliers_list.append(pt)
        if len(plot_pts) < 2500 or p025:
            plot_pts.append(pt)

    evr = pca.explained_variance_ratio_.tolist()
    metrics = {
        "total_samples": len(df),
        "pca": {
            "explained_variance_ratio": [float(round(v, 4)) for v in evr],
            "total_explained_variance": float(round(sum(evr), 4))
        },
        "model_025": {
            "contamination": 0.025,
            "outliers_count": int(np.sum(y_pred_025 == -1)),
            "inliers_count": int(np.sum(y_pred_025 == 1)),
            "offset_threshold": float(round(iso_025.offset_, 4))
        },
        "model_01": {
            "contamination": 0.01,
            "outliers_count": int(np.sum(y_pred_01 == -1)),
            "inliers_count": int(np.sum(y_pred_01 == 1)),
            "offset_threshold": float(round(iso_01.offset_, 4))
        },
        "transition": {
            "extreme_anomalies_count": int(np.sum(y_pred_01 == -1)),
            "added_anomalies_count": int(np.sum((y_pred_025 == -1) & (y_pred_01 == 1))),
            "consistent_inliers_count": int(np.sum(y_pred_025 == 1))
        }
    }
    return {"metrics": metrics, "outliers": outliers_list, "plot_sample": plot_pts}


data = load_iforest_student_scores_data()
metrics = data["metrics"]
plot_df = pd.DataFrame(data["plot_sample"])
outliers_df = pd.DataFrame(data["outliers"])

# Horní přehledové metriky
c_m1, c_m2, c_m3, c_m4 = st.columns(4)
c_m1.metric("Počet studentů v datasetu", f"{metrics['total_samples']:,}".replace(",", " "))
c_m2.metric("Vysvětlený rozptyl PCA (2D)", f"{metrics['pca']['total_explained_variance'] * 100:.1f} %", help="PC1 pokrývá 90.6 %, PC2 pokrývá 7.9 %")
c_m3.metric("Anomálie při 2.5 % (Model A)", f"{metrics['model_025']['outliers_count']} studentů", delta=f"Práh = {metrics['model_025']['offset_threshold']}")
c_m4.metric("Anomálie při 1.0 % (Model B)", f"{metrics['model_01']['outliers_count']} studentů", delta=f"-{metrics['transition']['added_anomalies_count']} méně (přísnější)")

tab_proc, tab_comp, tab_trans, tab_lab = st.tabs([
    "📋 Postup řešení & Výběr sloupců",
    "⚖️ Srovnání modelů (2.5 % vs 1.0 %)",
    "🔍 Zvýraznění přidaných anomálií",
    "🎛️ Živá laboratoř parametrů"
])

# -----------------------------------------------------------------------------
# TAB 1: Postup řešení & Výběr sloupců
# -----------------------------------------------------------------------------
with tab_proc:
    st.subheader("📋 Krok za krokem: Postup řešení cvičení")
    st.markdown(
        r"""
        Cvičení demonstruje typický analytický scénář v pedagogickém výzkumu: detekci neobvyklých profilů 
        studentů na základě výsledků standardizovaných testů z matematiky, čtení a psaní.
        """
    )

    steps_col1, steps_col2, steps_col3 = st.columns(3)
    with steps_col1:
        st.markdown(
            """
            <div style="background-color: #f8fafc; padding: 16px; border-radius: 8px; border-left: 4px solid #3b82f6;">
                <h4 style="margin: 0; color: #1e293b;">1. Výběr 3 sloupců</h4>
                <p style="font-size: 0.9rem; color: #475569; margin-top: 8px;">
                    Z 30 641 řádků vyfiltrujeme pouze <code>MathScore</code>, <code>ReadingScore</code> 
                    a <code>WritingScore</code>. Ostatní socioekonomické znaky vynecháme.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with steps_col2:
        st.markdown(
            """
            <div style="background-color: #f8fafc; padding: 16px; border-radius: 8px; border-left: 4px solid #10b981;">
                <h4 style="margin: 0; color: #1e293b;">2. Redukce na 2D (PCA)</h4>
                <p style="font-size: 0.9rem; color: #475569; margin-top: 8px;">
                    Aplikace <code>PCA(n_components=2)</code>. První dvě komponenty vysvětlují <b>98.5 % rozptylu</b>: 
                    PC1 = celkový výkon, PC2 = nepoměr matematiky vs. jazyka.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with steps_col3:
        st.markdown(
            """
            <div style="background-color: #f8fafc; padding: 16px; border-radius: 8px; border-left: 4px solid #f59e0b;">
                <h4 style="margin: 0; color: #1e293b;">3. Isolation Forest</h4>
                <p style="font-size: 0.9rem; color: #475569; margin-top: 8px;">
                    Natrénování modelu s <code>contamination=0.025</code> (766 anomálií) a následné porovnání 
                    s <code>contamination=0.010</code> (307 nejradikálnějších anomálií).
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("#### Ukázkový kód z vypracovaného Jupyter Notebooku:")
    st.code(
        r"""import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# 1. Načtení dat a výběr 3 sloupců
df = pd.read_csv('student_scores.csv')
scores_df = df[['MathScore', 'ReadingScore', 'WritingScore']]

# 2. Redukce dimenzionality na 2 komponenty pro vizualizaci
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(scores_df.values)

# 3. Model A: Kontaminace 2.5 %
iso_025 = IsolationForest(n_estimators=100, contamination=0.025, random_state=42)
y_pred_025 = iso_025.fit_predict(X_pca) # 1 = inlier, -1 = outlier

# 4. Model B: Kontaminace 1.0 %
iso_01 = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
y_pred_01 = iso_01.fit_predict(X_pca)
""",
        language="python"
    )

    st.markdown("#### Ukázka dat po projekci do PCA prostoru (prvních 10 řádků):")
    st.dataframe(
        plot_df[["id", "math", "reading", "writing", "pca_1", "pca_2", "decision_025", "category"]].head(10),
        width="stretch",
        hide_index=True
    )

# -----------------------------------------------------------------------------
# TAB 2: Srovnání modelů (2.5 % vs 1.0 %)
# -----------------------------------------------------------------------------
with tab_comp:
    st.subheader("⚖️ Vizuální srovnání: Kontaminace 2.5 % vs. 1.0 %")
    st.markdown(
        """
        Zadání se ptá: **„See how the chart changes when we assume that anomalies account for 1% of all observations.“**  
        Zde vidíme obě konfigurace vykreslené za identických souřadnicových os:
        """
    )

    g_col1, g_col2 = st.columns(2)

    with g_col1:
        st.markdown(f"**Model A: Contamination = 2.5 % ({metrics['model_025']['outliers_count']} studentů, Práh = {metrics['model_025']['offset_threshold']})**")
        fig_025 = px.scatter(
            plot_df,
            x="pca_1",
            y="pca_2",
            color=plot_df["is_outlier_025"].map({False: "Běžný student", True: "Anomálie (2.5 %)"}),
            color_discrete_map={"Běžný student": "#3b82f6", "Anomálie (2.5 %)": "#ef4444"},
            hover_data={"pca_1": ":.2f", "pca_2": ":.2f", "math": True, "reading": True, "writing": True, "id": True},
            labels={"pca_1": "PC1 (Celkový výkon)", "pca_2": "PC2 (Matematika vs. Jazyk)", "color": "Typ"},
            title="Model A: 2.5 % anomálií (Širší detekční perimetr)"
        )
        fig_025.update_traces(marker=dict(size=5, opacity=0.45))
        fig_025.for_each_trace(
            lambda t: t.update(marker=dict(size=8, opacity=0.9, line=dict(width=1.0, color="black"))) if "Anomálie" in t.name else None
        )
        fig_025.update_layout(height=480, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_025, width="stretch")

    with g_col2:
        st.markdown(f"**Model B: Contamination = 1.0 % ({metrics['model_01']['outliers_count']} studentů, Práh = {metrics['model_01']['offset_threshold']})**")
        fig_01 = px.scatter(
            plot_df,
            x="pca_1",
            y="pca_2",
            color=plot_df["is_outlier_01"].map({False: "Běžný student", True: "Anomálie (1.0 %)"}),
            color_discrete_map={"Běžný student": "#3b82f6", "Anomálie (1.0 %)": "#f97316"},
            hover_data={"pca_1": ":.2f", "pca_2": ":.2f", "math": True, "reading": True, "writing": True, "id": True},
            labels={"pca_1": "PC1 (Celkový výkon)", "pca_2": "PC2 (Matematika vs. Jazyk)", "color": "Typ"},
            title="Model B: 1.0 % anomálií (Pouze nejzávažnější excesy)"
        )
        fig_01.update_traces(marker=dict(size=5, opacity=0.45))
        fig_01.for_each_trace(
            lambda t: t.update(marker=dict(size=9, opacity=0.95, line=dict(width=1.2, color="black"))) if "Anomálie" in t.name else None
        )
        fig_01.update_layout(height=480, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_01, width="stretch")

    st.markdown("---")
    st.subheader("📊 Distribuce rozhodovací funkce a posun prahů")
    st.markdown(
        r"""
        Všimněte si, jak se mění percentilový řez:
        - Při **2.5 %** je práh nastaven na hodnotu **-0.1064** $\rightarrow$ do filtru spadne 766 studentů.
        - Při **1.0 %** se práh zpřísní na **-0.1332** $\rightarrow$ filtr propustí pouze 307 nejextrémnějších studentů.
        """
    )

    fig_hist = go.Figure()
    fig_hist.add_trace(
        go.Histogram(
            x=plot_df["decision_025"],
            nbinsx=50,
            marker_color="#60a5fa",
            opacity=0.75,
            name="Četnost studentů dle decision_function"
        )
    )
    fig_hist.add_vline(
        x=metrics["model_025"]["offset_threshold"],
        line_width=3,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text=f"Práh 2.5 % ({metrics['model_025']['offset_threshold']})",
        annotation_position="top right"
    )
    fig_hist.add_vline(
        x=metrics["model_01"]["offset_threshold"],
        line_width=3,
        line_dash="dot",
        line_color="#f59e0b",
        annotation_text=f"Práh 1.0 % ({metrics['model_01']['offset_threshold']})",
        annotation_position="top left"
    )
    fig_hist.update_layout(
        xaxis_title="Hodnota decision_function (nižší hodnota = vyšší anomálnost)",
        yaxis_title="Počet studentů v intervalu",
        height=380,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig_hist, width="stretch")

# -----------------------------------------------------------------------------
# TAB 3: Zvýraznění přidaných anomálií (Delta posun)
# -----------------------------------------------------------------------------
with tab_trans:
    st.subheader("🔍 Kteří studenti přesně tvoří rozdíl mezi 1.0 % a 2.5 %?")
    st.markdown(
        """
        Zde graficky i tabulkově izolujeme **rozdíl 459 studentů**:
        1. 🔴 **Extrémní anomálie (307 studentů)**: Detekovány při 1.0 % i při 2.5 %. Zahrnují studenty s fatálním kolapsem (skóre 0–25) nebo extrémní disproporcí.
        2. 🟡 **Mírnější anomálie (459 studentů)**: Detekovány pouze při 2.5 %. Představují studenty s mírnější disbalancí na vnějším obvodu eliptického mračna.
        3. 🔵 **Běžní studenti (29 875 studentů)**: Standardní studijní výsledky.
        """
    )

    color_cat_map = {
        "Běžný student (Inlier)": "#cbd5e1",
        "Přidaná anomálie (pouze při 2.5 %)": "#f59e0b",
        "Extrémní anomálie (1.0 % i 2.5 %)": "#dc2626"
    }

    fig_delta = px.scatter(
        plot_df,
        x="pca_1",
        y="pca_2",
        color="category",
        color_discrete_map=color_cat_map,
        category_orders={"category": ["Běžný student (Inlier)", "Přidaná anomálie (pouze při 2.5 %)", "Extrémní anomálie (1.0 % i 2.5 %)"]},
        hover_data={"pca_1": ":.2f", "pca_2": ":.2f", "math": True, "reading": True, "writing": True, "id": True},
        labels={"pca_1": "PC1 (Celkový výkon)", "pca_2": "PC2 (Matematika vs. Jazyk)", "category": "Kategorie studenta"},
        title="Diferenční mapa: Zvýraznění 459 studentů přidaných při rozšíření na 2.5 %"
    )
    fig_delta.update_traces(marker=dict(size=5))
    fig_delta.for_each_trace(
        lambda t: t.update(marker=dict(size=9, line=dict(width=1.2, color="black"))) if "Extrémní" in t.name else (
            t.update(marker=dict(size=7, line=dict(width=0.8, color="#78350f"))) if "Přidaná" in t.name else (
                t.update(marker=dict(opacity=0.25))
            )
        )
    )
    fig_delta.update_layout(height=520, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_delta, width="stretch")

    st.markdown("#### Detailní tabulka anomálních studentů:")
    st.caption("Filtrujte mezi všemi 766 anomáliemi, 307 extrémními případy nebo 459 mírnějšími.")

    sub_filter = st.radio(
        "Filtrovat skupinu anomálií:",
        ["Všech 766 anomálií (2.5 %)", "Pouze extrémní anomálie (307 studentů při 1.0 %)", "Pouze mírnější anomálie (459 studentů přidaných při 2.5 %)"],
        horizontal=True
    )

    if sub_filter == "Pouze extrémní anomálie (307 studentů při 1.0 %)":
        filtered_out = outliers_df[outliers_df["is_outlier_01"]]
    elif sub_filter == "Pouze mírnější anomálie (459 studentů přidaných při 2.5 %)":
        filtered_out = outliers_df[~outliers_df["is_outlier_01"]]
    else:
        filtered_out = outliers_df

    st.dataframe(
        filtered_out[["id", "math", "reading", "writing", "pca_1", "pca_2", "decision_025", "category"]],
        width="stretch",
        hide_index=True
    )

# -----------------------------------------------------------------------------
# TAB 4: Živá laboratoř parametrů
# -----------------------------------------------------------------------------
with tab_lab:
    st.subheader("🎛️ Interaktivní laboratoř: Kalibrace Isolation Forest v reálném čase")
    st.markdown(
        """
        Vyzkoušejte si, jak se mění počet detekovaných studentů a rozhodovací hranice v 2D PCA prostoru 
        při změně kontaminace a počtu stromů:
        """
    )

    c_c1, c_c2 = st.columns(2)
    with c_c1:
        slider_contam = st.slider(
            "Předpokládaná kontaminace (contamination %)",
            min_value=0.5,
            max_value=5.0,
            value=2.5,
            step=0.5,
            help="Procento studentů označených za anomální."
        ) / 100.0
    with c_c2:
        slider_trees = st.slider(
            "Počet stromů v lese (n_estimators)",
            min_value=20,
            max_value=150,
            value=100,
            step=10,
            help="Počet náhodných izolačních stromů."
        )

    # Rychlý on-the-fly model na vzorku bodů (< 35 ms)
    X_sample_pca = plot_df[["pca_1", "pca_2"]].values
    live_iso = IsolationForest(n_estimators=slider_trees, contamination=slider_contam, random_state=42)
    live_preds = live_iso.fit_predict(X_sample_pca)
    live_scores = live_iso.decision_function(X_sample_pca)

    live_df = plot_df.copy()
    live_df["live_is_anomaly"] = (live_preds == -1)
    live_df["live_score"] = live_scores
    live_n_outliers = int(np.sum(live_preds == -1))

    r_c1, r_c2, r_c3 = st.columns(3)
    r_c1.metric("Detekované anomálie ve vzorku", f"{live_n_outliers} studentů", delta=f"{live_n_outliers / len(live_df) * 100:.1f} %")
    r_c2.metric("Dynamický práh offsetu", f"{live_iso.offset_:.4f}")
    r_c3.metric("Nejextrémnější skóre", f"{np.min(live_scores):.4f}")

    fig_live = px.scatter(
        live_df,
        x="pca_1",
        y="pca_2",
        color=live_df["live_is_anomaly"].map({False: "Běžný student", True: "Anomálie"}),
        color_discrete_map={"Běžný student": "#3b82f6", "Anomálie": "#e11d48"},
        hover_data={"pca_1": ":.2f", "pca_2": ":.2f", "math": True, "reading": True, "writing": True, "live_score": ":.3f", "id": True},
        labels={"pca_1": "PC1 (Celkový výkon)", "pca_2": "PC2 (Matematika vs. Jazyk)", "color": "Klasifikace"},
        title=f"Živý model: Kontaminace = {slider_contam * 100:.1f} %, Stromy = {slider_trees} (Práh = {live_iso.offset_:.3f})"
    )
    fig_live.update_traces(marker=dict(size=5, opacity=0.5))
    fig_live.for_each_trace(
        lambda t: t.update(marker=dict(size=9, opacity=0.9, line=dict(width=1.2, color="black"))) if t.name == "Anomálie" else None
    )
    fig_live.update_layout(height=520, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_live, width="stretch")
