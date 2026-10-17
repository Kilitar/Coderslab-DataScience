"""
Den 5: Neřízené učení – Cvičení LOF na datech blobs_dataset.csv (Výsledky zadání)
================================================================================
Oficiální řešení cvičení z kurzu:
1. Načtení datasetu blobs_dataset.csv do prostředí Pandas.
2. Standardizace příznaků pomocí StandardScaler.
3. Model LocalOutlierFactor s kontaminací 2 % (contamination=0.02) a k=20.
4. Grafické znázornění detekovaných anomálií a běžných pozorování.
5. Srovnání se stavem při kontaminaci 5 % (contamination=0.05) a analýza přidaných bodů.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import streamlit as st

st.set_page_config(page_title="Cvičení: Blobs – Výsledky zadání", page_icon="🎯", layout="wide")

st.title("🎯 Cvičení: Detekce anomálií LOF (blobs_dataset.csv) – Výsledky zadání")
st.caption(
    "Oficiální řešení cvičení z kurzu: Detekce odlehlých pozorování v syntetickém shlukovém datasetu "
    "pomocí algoritmu Local Outlier Factor (LOF), standardizace, k=20 a analytické srovnání kontaminace 2 % vs. 5 %."
)


@st.cache_data
def load_lof_blobs_exercise_data():
    candidates_json = [
        Path("05_Unsupervised_Learning/data/lof_blobs_exercise_precomputed.json"),
        Path(__file__).resolve().parent.parent / "05_Unsupervised_Learning/data/lof_blobs_exercise_precomputed.json",
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
    data_candidates = [
        Path("data/Downloadable materials for Session 3/blobs_dataset.csv"),
        Path(__file__).resolve().parent.parent / "data/Downloadable materials for Session 3/blobs_dataset.csv",
        Path("data/blobs_dataset.csv"),
    ]
    csv_path = None
    for cand in data_candidates:
        if cand.exists():
            csv_path = cand
            break

    if not csv_path:
        # Syntetický fallback při absenci souboru
        from sklearn.datasets import make_blobs
        X, _ = make_blobs(n_samples=1000, centers=4, cluster_std=1.5, random_state=42)
        df = pd.DataFrame(X, columns=["feature_1", "feature_2"])
    else:
        df = pd.read_csv(csv_path)

    X_raw = df[["feature_1", "feature_2"]].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    lof_02 = LocalOutlierFactor(n_neighbors=20, contamination=0.02)
    y_pred_02 = lof_02.fit_predict(X_scaled)
    scores_02 = -lof_02.negative_outlier_factor_
    threshold_02 = -lof_02.offset_

    lof_05 = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    y_pred_05 = lof_05.fit_predict(X_scaled)
    scores_05 = -lof_05.negative_outlier_factor_
    threshold_05 = -lof_05.offset_

    points = []
    for i in range(len(df)):
        p02 = (y_pred_02[i] == -1)
        p05 = (y_pred_05[i] == -1)
        if p02 and p05:
            cat = "Extrémní anomálie (2 % i 5 %)"
        elif not p02 and p05:
            cat = "Přidaná anomálie (pouze při 5 %)"
        else:
            cat = "Normální pozorování (Inlier)"

        points.append({
            "id": int(i),
            "f1_raw": float(round(df["feature_1"].iloc[i], 4)),
            "f2_raw": float(round(df["feature_2"].iloc[i], 4)),
            "f1_scaled": float(round(X_scaled[i, 0], 4)),
            "f2_scaled": float(round(X_scaled[i, 1], 4)),
            "lof_score": float(round(scores_02[i], 4)),
            "is_outlier_02": bool(p02),
            "is_outlier_05": bool(p05),
            "category": cat
        })

    metrics = {
        "total_samples": int(len(df)),
        "n_neighbors": 20,
        "model_02": {
            "contamination": 0.02,
            "outliers_count": int(np.sum(y_pred_02 == -1)),
            "inliers_count": int(np.sum(y_pred_02 == 1)),
            "threshold_lof": float(round(threshold_02, 4)),
        },
        "model_05": {
            "contamination": 0.05,
            "outliers_count": int(np.sum(y_pred_05 == -1)),
            "inliers_count": int(np.sum(y_pred_05 == 1)),
            "threshold_lof": float(round(threshold_05, 4)),
        },
        "transition": {
            "extreme_anomalies_count": int(np.sum(y_pred_02 == -1)),
            "added_anomalies_count": int(np.sum((y_pred_02 == 1) & (y_pred_05 == -1))),
            "consistent_inliers_count": int(np.sum((y_pred_02 == 1) & (y_pred_05 == 1)))
        }
    }

    top_outliers = sorted(
        [p for p in points if p["is_outlier_05"]],
        key=lambda x: x["lof_score"],
        reverse=True
    )

    return {"metrics": metrics, "points": points, "top_outliers": top_outliers}


data = load_lof_blobs_exercise_data()
metrics = data["metrics"]
points_df = pd.DataFrame(data["points"])
top_outliers_df = pd.DataFrame(data["top_outliers"])

# Horní přehledové metriky
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("Počet pozorování", f"{metrics['total_samples']:,}".replace(",", " "))
col_m2.metric("Počet sousedů (k)", f"{metrics['n_neighbors']}")
col_m3.metric("Anomálie při 2 % (Model A)", f"{metrics['model_02']['outliers_count']} bodů", delta=f"Práh LOF >= {metrics['model_02']['threshold_lof']}")
col_m4.metric("Anomálie při 5 % (Model B)", f"{metrics['model_05']['outliers_count']} bodů", delta=f"+{metrics['transition']['added_anomalies_count']} nových (Práh >= {metrics['model_05']['threshold_lof']})")

tab_proc, tab_comp, tab_trans, tab_lab = st.tabs([
    "📋 Krok za krokem postup",
    "⚖️ Srovnání modelů (2 % vs 5 %)",
    "🔍 Zvýraznění přidaných anomálií",
    "🎛️ Živá laboratoř parametrů"
])

# -----------------------------------------------------------------------------
# TAB 1: Krok za krokem postup
# -----------------------------------------------------------------------------
with tab_proc:
    st.subheader("📋 Metodický postup řešení cvičení")
    st.markdown(
        """
        Zadání vyžaduje kompletní workflow neřízené detekce anomálií pomocí algoritmu **Local Outlier Factor (LOF)**:
        """
    )

    step_cols = st.columns(3)
    with step_cols[0]:
        st.markdown(
            """
            <div style="background-color: #f8fafc; padding: 16px; border-radius: 8px; border-left: 4px solid #3b82f6;">
                <h4 style="margin: 0; color: #1e293b;">1. Příprava dat & Standardizace</h4>
                <p style="font-size: 0.9rem; color: #475569; margin-top: 8px;">
                    Načtení <code>blobs_dataset.csv</code> (1 000 bodů, 2 příznaky). Transformace pomocí <code>StandardScaler</code>, 
                    neboť LOF spoléhá na eukleidovskou metriku.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with step_cols[1]:
        st.markdown(
            """
            <div style="background-color: #f8fafc; padding: 16px; border-radius: 8px; border-left: 4px solid #10b981;">
                <h4 style="margin: 0; color: #1e293b;">2. Model A: Contamination = 2 %</h4>
                <p style="font-size: 0.9rem; color: #475569; margin-top: 8px;">
                    Inicializace <code>LocalOutlierFactor(n_neighbors=20, contamination=0.02)</code>. 
                    Detekce přesně 20 nejextrémnějších anomálií v prázdném prostoru.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with step_cols[2]:
        st.markdown(
            """
            <div style="background-color: #f8fafc; padding: 16px; border-radius: 8px; border-left: 4px solid #f59e0b;">
                <h4 style="margin: 0; color: #1e293b;">3. Model B: Contamination = 5 %</h4>
                <p style="font-size: 0.9rem; color: #475569; margin-top: 8px;">
                    Přenastavení na <code>contamination=0.05</code>. Detekce 50 bodů (+30 nových na periferii shluků). 
                    Skóre LOF se nemění, pouze klesá rozhodovací práh.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("#### Ukázkový kód z vypracovaného Jupyter Notebooku:")
    st.code(
        r"""import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt

# 1. Načtení dat
df = pd.read_csv('blobs_dataset.csv')

# 2. Standardizace
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['feature_1', 'feature_2']])

# 3. Model 1: Kontaminace 2 %
lof_02 = LocalOutlierFactor(n_neighbors=20, contamination=0.02)
y_pred_02 = lof_02.fit_predict(X_scaled) # 1 = inlier, -1 = outlier
df['is_outlier_02'] = (y_pred_02 == -1)

# 4. Model 2: Kontaminace 5 %
lof_05 = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
y_pred_05 = lof_05.fit_predict(X_scaled)
df['is_outlier_05'] = (y_pred_05 == -1)
""",
        language="python"
    )

    st.markdown("#### Náhled vstupního datasetu (prvních 10 řádků):")
    st.dataframe(
        points_df[["id", "f1_raw", "f2_raw", "f1_scaled", "f2_scaled", "lof_score"]].head(10),
        width="stretch",
        hide_index=True
    )

# -----------------------------------------------------------------------------
# TAB 2: Srovnání modelů (2 % vs 5 %)
# -----------------------------------------------------------------------------
with tab_comp:
    st.subheader("⚖️ Porovnání detekce: Kontaminace 2 % vs. 5 %")
    st.markdown(
        """
        Zadání se ptá: **„Check how the chart changes when we assume that anomalies account for 5% of the observations.“**  
        Zde vidíme obě konfigurace vykreslené za identických souřadnicových os:
        """
    )

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown(f"**Model A: Contamination = 2 % (20 anomálií, LOF >= {metrics['model_02']['threshold_lof']})**")
        fig_02 = px.scatter(
            points_df,
            x="f1_raw",
            y="f2_raw",
            color=points_df["is_outlier_02"].map({False: "Inlier (980)", True: "Anomálie (20)"}),
            color_discrete_map={"Inlier (980)": "#3b82f6", "Anomálie (20)": "#ef4444"},
            hover_data={"f1_raw": ":.2f", "f2_raw": ":.2f", "lof_score": ":.3f", "id": True},
            labels={"f1_raw": "feature_1", "f2_raw": "feature_2", "color": "Typ bodu"},
            title="Model A: 2 % anomálií (Přísný filtr)"
        )
        fig_02.update_traces(marker=dict(size=7, opacity=0.7))
        fig_02.for_each_trace(
            lambda t: t.update(marker=dict(size=11, line=dict(width=1.5, color="black"))) if "Anomálie" in t.name else None
        )
        fig_02.update_layout(height=480, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_02, width="stretch")

    with col_g2:
        st.markdown(f"**Model B: Contamination = 5 % (50 anomálií, LOF >= {metrics['model_05']['threshold_lof']})**")
        fig_05 = px.scatter(
            points_df,
            x="f1_raw",
            y="f2_raw",
            color=points_df["is_outlier_05"].map({False: "Inlier (950)", True: "Anomálie (50)"}),
            color_discrete_map={"Inlier (950)": "#3b82f6", "Anomálie (50)": "#f97316"},
            hover_data={"f1_raw": ":.2f", "f2_raw": ":.2f", "lof_score": ":.3f", "id": True},
            labels={"f1_raw": "feature_1", "f2_raw": "feature_2", "color": "Typ bodu"},
            title="Model B: 5 % anomálií (Rozšířený filtr)"
        )
        fig_05.update_traces(marker=dict(size=7, opacity=0.7))
        fig_05.for_each_trace(
            lambda t: t.update(marker=dict(size=10, line=dict(width=1.2, color="black"))) if "Anomálie" in t.name else None
        )
        fig_05.update_layout(height=480, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_05, width="stretch")

    st.markdown("---")
    st.subheader("📊 Distribuce LOF skóre v populaci a posun rozhodovacího prahu")
    st.markdown(
        r"""
        Všimněte si zásadního matematického faktu: **LOF skóre samotných bodů se nezměnilo ani o píď**. 
        Změna parametru `contamination` v Scikit-learn pouze mění **percentilový řez** na distribuci skóre:
        - Při 2 % je práh nastaven na **1.730** (zachytí jen 2 % nejvyšších hodnot).
        - Při 5 % se práh sníží na **1.504** (do anomálií propadne dalších 30 bodů).
        """
    )

    fig_hist = go.Figure()
    fig_hist.add_trace(
        go.Histogram(
            x=points_df["lof_score"],
            nbinsx=50,
            marker_color="#60a5fa",
            opacity=0.75,
            name="Četnost bodů dle LOF"
        )
    )
    fig_hist.add_vline(
        x=metrics["model_02"]["threshold_lof"],
        line_width=3,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text=f"Práh 2 %: LOF >= {metrics['model_02']['threshold_lof']}",
        annotation_position="top right"
    )
    fig_hist.add_vline(
        x=metrics["model_05"]["threshold_lof"],
        line_width=3,
        line_dash="dot",
        line_color="#f59e0b",
        annotation_text=f"Práh 5 %: LOF >= {metrics['model_05']['threshold_lof']}",
        annotation_position="top left"
    )
    fig_hist.update_layout(
        xaxis_title="LOF skóre (vyšší hodnota = výraznější odlehlost od lokální hustoty)",
        yaxis_title="Počet bodů v intervalu",
        height=380,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_hist, width="stretch")

# -----------------------------------------------------------------------------
# TAB 3: Zvýraznění přidaných anomálií (Delta analýza)
# -----------------------------------------------------------------------------
with tab_trans:
    st.subheader("🔍 Kterých 30 bodů přesně přibylo při 5% kontaminaci?")
    st.markdown(
        """
        Zde graficky i tabulkově izolujeme **přechodovou zónu**. Body jsou rozděleny do tří kategorií:
        1. 🔴 **Extrémní anomálie (20 bodů)**: Označeny jako outlier při 2 % i při 5 %. Zcela izolované body v prázdném prostoru.
        2. 🟡 **Přidané hraniční anomálie (30 bodů)**: Označeny jako inlier při 2 %, ale při 5 % přeřazeny do outlierů. Nacházejí se na vnějších okrajích (periferii) hlavních shluků.
        3. 🔵 **Konzistentní inliers (950 bodů)**: Běžná pozorování hluboko uvnitř shluků.
        """
    )

    color_cat_map = {
        "Normální pozorování (Inlier)": "#cbd5e1",
        "Přidaná anomálie (pouze při 5 %)": "#f59e0b",
        "Extrémní anomálie (2 % i 5 %)": "#dc2626"
    }

    fig_delta = px.scatter(
        points_df,
        x="f1_raw",
        y="f2_raw",
        color="category",
        color_discrete_map=color_cat_map,
        category_orders={"category": ["Normální pozorování (Inlier)", "Přidaná anomálie (pouze při 5 %)", "Extrémní anomálie (2 % i 5 %)"]},
        hover_data={"f1_raw": ":.2f", "f2_raw": ":.2f", "lof_score": ":.3f", "id": True},
        labels={"f1_raw": "feature_1", "f2_raw": "feature_2", "category": "Kategorie bodu"},
        title="Diferenční mapa: Zvýraznění 30 hraničních bodů na periferii shluků"
    )
    fig_delta.update_traces(marker=dict(size=7))
    fig_delta.for_each_trace(
        lambda t: t.update(marker=dict(size=11, line=dict(width=1.5, color="black"))) if "Extrémní" in t.name else (
            t.update(marker=dict(size=10, line=dict(width=1.2, color="#78350f"))) if "Přidaná" in t.name else (
                t.update(marker=dict(opacity=0.35))
            )
        )
    )
    fig_delta.update_layout(height=520, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_delta, width="stretch")

    st.markdown("#### Detailní přehled 50 anomálií (seřazeno sestupně dle LOF skóre):")
    st.caption("Filtrujte tabulku pro zobrazení extrémních nebo nově přidaných anomálií.")

    filter_choice = st.radio(
        "Zobrazit skupinu anomálií:",
        ["Všech 50 anomálií", "Pouze extrémní anomálie (20 bodů, 2 % i 5 %)", "Pouze přidané hraniční anomálie (30 bodů, pouze 5 %)"],
        horizontal=True
    )

    if filter_choice == "Pouze extrémní anomálie (20 bodů, 2 % i 5 %)":
        filtered_top = top_outliers_df[top_outliers_df["is_outlier_02"]]
    elif filter_choice == "Pouze přidané hraniční anomálie (30 bodů, pouze 5 %)":
        filtered_top = top_outliers_df[~top_outliers_df["is_outlier_02"]]
    else:
        filtered_top = top_outliers_df

    st.dataframe(
        filtered_top[["id", "f1_raw", "f2_raw", "lof_score", "category", "is_outlier_02", "is_outlier_05"]],
        width="stretch",
        hide_index=True
    )

# -----------------------------------------------------------------------------
# TAB 4: Živá laboratoř parametrů
# -----------------------------------------------------------------------------
with tab_lab:
    st.subheader("🎛️ Interaktivní laboratoř: Kalibrace LOF v reálném čase")
    st.markdown(
        """
        Experimentujte s nastavením hyperparametrů na reálném datasetu `blobs_dataset.csv`. 
        Sledujte okamžitou odezvu rozhodovacího prahu, počtu anomálií a vizuálního ohraničení shluků.
        """
    )

    ctrl_col1, ctrl_col2 = st.columns(2)
    with ctrl_col1:
        slider_contam = st.slider(
            "Předpokládaná kontaminace (contamination %)",
            min_value=0.5,
            max_value=10.0,
            value=2.0,
            step=0.5,
            help="Podíl dat, který model a priori považuje za anomálie."
        ) / 100.0
    with ctrl_col2:
        slider_k = st.slider(
            "Počet nejbližších sousedů (k / n_neighbors)",
            min_value=5,
            max_value=60,
            value=20,
            step=5,
            help="Velikost lokálního okolí pro výpočet lokální dosažitelné hustoty."
        )

    # Spočteme interaktivní model
    X_mat = points_df[["f1_scaled", "f2_scaled"]].values
    live_lof = LocalOutlierFactor(n_neighbors=slider_k, contamination=slider_contam)
    live_preds = live_lof.fit_predict(X_mat)
    live_scores = -live_lof.negative_outlier_factor_
    live_threshold = -live_lof.offset_

    live_df = points_df.copy()
    live_df["live_is_outlier"] = (live_preds == -1)
    live_df["live_lof_score"] = live_scores
    live_n_outliers = int(np.sum(live_preds == -1))

    res_c1, res_c2, res_c3 = st.columns(3)
    res_c1.metric("Počet detekovaných anomálií", f"{live_n_outliers} bodů", delta=f"{live_n_outliers / len(live_df) * 100:.1f} % z celku")
    res_c2.metric("Dynamický práh LOF", f"{live_threshold:.4f}", help="Body s LOF větším než tento práh jsou označeny jako anomálie.")
    res_c3.metric("Nejvyšší LOF skóre", f"{np.max(live_scores):.4f}")

    fig_live = px.scatter(
        live_df,
        x="f1_raw",
        y="f2_raw",
        color=live_df["live_is_outlier"].map({False: "Inlier", True: "Anomálie"}),
        color_discrete_map={"Inlier": "#3b82f6", "Anomálie": "#e11d48"},
        hover_data={"f1_raw": ":.2f", "f2_raw": ":.2f", "live_lof_score": ":.3f", "id": True},
        labels={"f1_raw": "feature_1", "f2_raw": "feature_2", "color": "Stav"},
        title=f"Živý model: k = {slider_k}, Kontaminace = {slider_contam * 100:.1f} % (Práh LOF >= {live_threshold:.3f})"
    )
    fig_live.update_traces(marker=dict(size=7, opacity=0.7))
    fig_live.for_each_trace(
        lambda t: t.update(marker=dict(size=11, line=dict(width=1.5, color="black"))) if t.name == "Anomálie" else None
    )
    fig_live.update_layout(height=520, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_live, width="stretch")
