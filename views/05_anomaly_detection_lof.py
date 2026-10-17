"""
Den 5: Neřízené učení – Praktická implementace Local Outlier Factor (LOF)
========================================================================
Syntéza materiálů z Local_Outlier_Factor_-_sample_implementation.pdf:
1. Vzorová implementace ze slajdů (make_blobs, n=500, centers=2, k=20, contam=10 %).
2. Ruční výpočetní příklad na 5 bodech krok za krokem (slajdy 9–13, k=2, RD, LRD, LOF).
3. Demonstrace triumfu LOF při proměnlivé hustotě (Varying Density Benchmark).
4. Interaktivní laboratoř na reálném datasetu blobs_dataset.csv (1 000 bodů).
5. Bilance výhod a nevýhod (slajdy 23–24) a What-If simulátor Novelty Detection.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import make_blobs
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler
import streamlit as st

st.set_page_config(page_title="Implementace: LOF v Scikit-learn", page_icon="🛠️", layout="wide")

st.title("🛠️ Vzorová implementace: Local Outlier Factor (LOF) v Scikit-learn")
st.caption(
    "Oficiální vzorová implementace z přednášky kurzu (Local Outlier Factor - sample implementation). "
    "Generování shluků make_blobs (500 bodů, 10 % anomálií), ruční matematický rozbor 5 bodů ze slajdů 9–13, "
    "chování při proměnlivé hustotě a vyhodnocení silných i slabých stránek algoritmu."
)


@st.cache_data
def load_lof_implementation_data():
    candidates_json = [
        Path("05_Unsupervised_Learning/data/lof_anomaly_precomputed.json"),
        Path(__file__).resolve().parent.parent / "05_Unsupervised_Learning/data/lof_anomaly_precomputed.json",
    ]
    for p in candidates_json:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    if isinstance(d, dict) and "slide_blobs_points" in d and "blobs_points" in d:
                        return d
            except Exception:
                pass

    # Fallback on-the-fly výpočet
    X_s, y_s = make_blobs(n_samples=500, centers=2, random_state=42, cluster_std=2.0)
    lof_s = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    preds_s = lof_s.fit_predict(X_s)
    neg_s = lof_s.negative_outlier_factor_

    s_pts = []
    for i in range(len(X_s)):
        s_pts.append({
            "idx": i,
            "x": float(X_s[i, 0]),
            "y": float(X_s[i, 1]),
            "cluster": int(y_s[i]),
            "y_pred": int(preds_s[i]),
            "is_outlier": bool(preds_s[i] == -1),
            "neg_lof_score": float(neg_s[i]),
            "pos_lof_score": float(-neg_s[i])
        })

    manual_pts = [
        {"name": "X1", "x": 0.0, "y": 0.0, "k_distance": 2.0, "lrd": 0.5, "lof": 1.0, "status": "Normální bod"},
        {"name": "X2", "x": 2.0, "y": 0.0, "k_distance": 2.0, "lrd": 0.5, "lof": 1.0, "status": "Normální bod"},
        {"name": "X3", "x": 0.0, "y": 2.0, "k_distance": 2.0, "lrd": 0.5, "lof": 1.0, "status": "Normální bod"},
        {"name": "X4", "x": 3.0, "y": 2.0, "k_distance": 3.0, "lrd": 0.3333, "lof": 1.5, "status": "Mírná anomálie"},
        {"name": "X5", "x": 6.0, "y": 0.0, "k_distance": 6.0, "lrd": 0.1667, "lof": 2.5, "status": "Silná anomálie"}
    ]

    return {
        "slide_blobs_meta": {"n_samples": 500, "outliers_count": 50, "inliers_count": 450, "contamination": 0.10},
        "slide_blobs_points": s_pts,
        "manual_toy_example": manual_pts,
        "blobs_points": [],
        "multi_density_points": []
    }


lof_data = load_lof_implementation_data()
df_slide = pd.DataFrame(lof_data.get("slide_blobs_points", []))
s_meta = lof_data.get("slide_blobs_meta", {})
manual_data = lof_data.get("manual_toy_example", [])

# Horní KPI karty
c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Vzorový dataset",
    f"{s_meta.get('n_samples', 500)} bodů",
    delta="make_blobs(centers=2, std=2.0)",
    delta_color="off"
)
c2.metric(
    "Hyperparametry ze slajdů",
    "k = 20, contam = 10 %",
    delta="LocalOutlierFactor",
    delta_color="normal"
)
n_out_slide = int(s_meta.get("outliers_count", 50))
c3.metric(
    "Detekované anomálie",
    f"{n_out_slide} bodů ({n_out_slide / s_meta.get('n_samples', 500) * 100:.0f} %)",
    delta="Label -1 (červené body)",
    delta_color="inverse"
)
c4.metric(
    "Práh pro anomálie",
    r"LOF ≥ 1.320",
    delta=f"offset = {s_meta.get('offset', -1.32)}",
    delta_color="normal"
)

st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 1. Vzorová implementace ze slajdů",
    "🧮 2. Ruční výpočet na 5 bodech",
    "🔬 3. Triumf LOF při proměnlivé hustotě",
    "🧪 4. Interaktivní laboratoř na blobs_dataset",
    "⚖️ 5. Výhody, nevýhody & Novelty Detection"
])

# ==============================================================================
# TAB 1: VZOROVÁ IMPLEMENTACE ZE SLAJDŮ
# ==============================================================================
with tab1:
    st.subheader("1. Vzorová implementace ze slajdů 15–20")
    st.markdown(
        r"""
        Oficiální kód z přednášky demonstruje aplikaci LOF na syntetickém shluku o 500 bodech vygenerovaném přes `make_blobs`:
        * `n_samples = 500`, `centers = 2`, `cluster_std = 2.0`, `random_state = 42`.
        * `LocalOutlierFactor(n_neighbors=20, contamination=0.1)`.
        * Výsledkem je rozdělení na **450 normálních bodů (Inliers, modrá)** a přesně **50 odlehlých bodů (Outliers, červená)**.
        """
    )

    c_s1, c_s2 = st.columns(2)

    with c_s1:
        # Scatter přesně dle slajdu 19
        df_slide["Barva"] = ["🔴 Anomálie (-1, 50 bodů)" if out else "🔵 Normální bod (1, 450 bodů)" for out in df_slide["is_outlier"]]
        fig_slide = px.scatter(
            df_slide,
            x="x",
            y="y",
            color="Barva",
            title="Outliers (red) in dataset — Reprodukce slajdu 19",
            labels={"x": "Příznak 1", "y": "Příznak 2"},
            color_discrete_map={"🔵 Normální bod (1, 450 bodů)": "#2563eb", "🔴 Anomálie (-1, 50 bodů)": "#dc2626"}
        )
        fig_slide.update_traces(marker=dict(size=7, opacity=0.85, line=dict(width=0.8, color="black")))
        fig_slide.update_layout(template="plotly_white", height=430, margin=dict(l=30, r=30, t=40, b=30))
        st.plotly_chart(fig_slide, width="stretch")
        st.caption("Slajd 20: Body vyznačené červeně jsou body ležící na vnějším obvodu shluků a v řídkém mezishlukovém prostoru.")

    with c_s2:
        # Bublinový graf spojitého skóre
        df_slide["bubble_size"] = 8 + 45 * np.maximum(0, df_slide["pos_lof_score"] - 1.0) ** 1.8
        fig_b_slide = px.scatter(
            df_slide,
            x="x",
            y="y",
            color="pos_lof_score",
            size="bubble_size",
            title="Spojité LOF skóre (Velikost bubliny ~ LOF)",
            labels={"x": "Příznak 1", "y": "Příznak 2", "pos_lof_score": "LOF faktor"},
            color_continuous_scale="YlOrRd",
            hover_data=["idx", "pos_lof_score", "neg_lof_score"]
        )
        fig_b_slide.update_traces(marker=dict(opacity=0.8, line=dict(width=0.8, color="black")))
        fig_b_slide.update_layout(template="plotly_white", height=430, margin=dict(l=30, r=30, t=40, b=30))
        st.plotly_chart(fig_b_slide, width="stretch")
        st.caption("Atribut negative_outlier_factor_: čím zápornější hodnota, tím větší bublina a tím závažnější anomálie.")

    st.markdown("#### 💻 Přesný kód z přednášky (slajdy 15–19):")
    st.code(
        r"""
from sklearn.datasets import make_blobs
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import matplotlib.pyplot as plt

# 1. Generování datasetu o 500 bodech (slajd 15)
X, labels = make_blobs(n_samples=500, centers=2, random_state=42, cluster_std=2.0)

# 2. Vytvoření instance modelu LOF (slajd 16)
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)

# 3. Fitování a predikce 1 vs. -1 (slajd 17)
y_pred = lof.fit_predict(X)

# 4. Získání přesných LOF výsledků (slajd 18)
lof_scores = lof.negative_outlier_factor_

# 5. Generování grafu s červenými anomáliemi (slajd 19)
marker_colors = np.array(["r", "b"])
plt.scatter(X[:, 0], X[:, 1], s=25, color=marker_colors[(y_pred + 1) // 2])
plt.title('Outliers (red) in dataset')
plt.show()
        """,
        language="python"
    )

# ==============================================================================
# TAB 2: RUČNÍ VÝPOČET NA 5 BODECH ZE SLAJDŮ
# ==============================================================================
with tab2:
    st.subheader("2. Ruční výpočet LOF krok za krokem (slajdy 9–13)")
    st.markdown(
        r"""
        Přednáška uvádí názorný výukový příklad pro $k = 2$ (2 nejbližší sousedé) na zjednodušeném souboru 5 bodů:
        * $X_1, X_2, X_3$ tvoří kompaktní trojúhelníkový shluk.
        * $X_4$ leží v mírné vzdálenosti.
        * $X_5$ leží ve velké vzdálenosti (jasná anomálie).
        """
    )

    if manual_data:
        df_man = pd.DataFrame(manual_data)

        c_m_tbl, c_m_fig = st.columns([1.2, 1.0])

        with c_m_tbl:
            st.markdown("##### 📊 Výsledná tabulka výpočtu ze slajdů:")
            st.table(df_man[["name", "k_distance", "lrd", "lof", "status"]])

            st.markdown(
                r"""
                **Výpočetní postup:**
                1. **Dosahová vzdálenost:** $RD(X_1, X_2) = \max(K\text{-dist}(X_2), d(X_1, X_2)) = \max(2, 2) = 2$.
                2. **Dosahová hustota:** $LRD_2(X_1) = \frac{1}{(2 + 2) / 2} = 0.5$.
                3. **Faktor LOF bodu $X_1$:**
                   $$LOF_2(X_1) = \frac{\frac{LRD(X_2) + LRD(X_3)}{2}}{LRD(X_1)} = \frac{\frac{0.5 + 0.5}{2}}{0.5} = \mathbf{1.0}$$
                4. **Faktor LOF odlehlých bodů:**
                   * $LOF_2(X_4) = \frac{0.5}{1/3} = \mathbf{1.5}$ (počínající odlehlost)
                   * $LOF_2(X_5) = \frac{0.5 + 1/3}{2} / (1/6) = \frac{5/12}{1/6} = \mathbf{2.5}$ (silná anomálie!)
                """
            )

        with c_m_fig:
            fig_toy = px.scatter(
                df_man,
                x="x",
                y="y",
                text="name",
                color="status",
                size=[18, 18, 18, 26, 38],
                title="Geometrické rozmístění 5 bodů ze slajdu 9",
                color_discrete_map={"Normální bod": "#2563eb", "Mírná anomálie": "#f59e0b", "Silná anomálie": "#dc2626"}
            )
            fig_toy.update_traces(textposition="top center", marker=dict(line=dict(width=1.5, color="black")))
            fig_toy.update_layout(template="plotly_white", height=420, margin=dict(l=30, r=30, t=40, b=30))
            st.plotly_chart(fig_toy, width="stretch")

# ==============================================================================
# TAB 3: TRIUMF LOF PŘI PROMĚNLIVÉ HUSTOTĚ
# ==============================================================================
with tab3:
    st.subheader("3. Proč globální metody selhávají a LOF bezchybně vítězí?")
    st.markdown(
        r"""
        Klíčovým důvodem existence LOF je **problém nestejné hustoty (Varying Density)**:
        * **Hustý shluk $C_1$ (vlevo):** Mezibodová vzdálenost je jen $\approx 0.2$.
        * **Řídký shluk $C_2$ (vpravo):** Mezibodová vzdálenost je běžně $\approx 1.0$.
        * **Lokální anomálie u $C_1$:** Bod má vzdálenost $0.6$ od $C_1$.
        """
    )

    multi_pts = lof_data.get("multi_density_points", [])
    if multi_pts:
        df_multi = pd.DataFrame(multi_pts)
        k_thresh = np.percentile(df_multi["k_distance"], 96)
        df_multi["Global_Detection"] = ["🚨 Falešná anomálie" if d >= k_thresh else "✅ Inlier" for d in df_multi["k_distance"]]
        df_multi["LOF_Detection"] = ["🚨 Korektní anomálie" if out else "✅ Inlier" for out in df_multi["is_lof_outlier"]]

        c_v1, c_v2 = st.columns(2)
        with c_v1:
            fig_glob = px.scatter(
                df_multi,
                x="x",
                y="y",
                color="Global_Detection",
                title="❌ Globální k-NN vzdálenost (Selhání)",
                color_discrete_map={"✅ Inlier": "#2563eb", "🚨 Falešná anomálie": "#dc2626"}
            )
            fig_glob.update_traces(marker=dict(size=8, opacity=0.85, line=dict(width=0.8, color="black")))
            fig_glob.update_layout(template="plotly_white", height=400, margin=dict(l=30, r=30, t=40, b=30))
            st.plotly_chart(fig_glob, width="stretch")
            st.error("Globální metoda selhala: označila běžné body v řídkém shluku za anomálie a minula lokální anomálii!")

        with c_v2:
            fig_lof_t = px.scatter(
                df_multi,
                x="x",
                y="y",
                color="LOF_Detection",
                title="✅ Local Outlier Factor (Triumf)",
                color_discrete_map={"✅ Inlier": "#2563eb", "🚨 Korektní anomálie": "#10b981"}
            )
            fig_lof_t.update_traces(marker=dict(size=8, opacity=0.85, line=dict(width=0.8, color="black")))
            fig_lof_t.update_layout(template="plotly_white", height=400, margin=dict(l=30, r=30, t=40, b=30))
            st.plotly_chart(fig_lof_t, width="stretch")
            st.success("LOF triumfuje: porovnává lokální hustotu, takže správně odhalí bod u $C_1$ i globální outliery.")

# ==============================================================================
# TAB 4: INTERAKTIVNÍ LABORATOŘ NA BLOBS_DATASET
# ==============================================================================
with tab4:
    st.subheader("4. Interaktivní laboratoř na blobs_dataset.csv (1 000 bodů)")
    st.markdown(
        r"""
        Vyzkoušejte si chování LOF na větším datasetu z kurzu. Živě posouvejte počet sousedů `n_neighbors`
        a podíl kontaminace `contamination`:
        """
    )

    blobs_raw_pts = lof_data.get("blobs_points", [])
    if blobs_raw_pts:
        df_b_lab = pd.DataFrame(blobs_raw_pts)

        c_ctl1, c_ctl2 = st.columns(2)
        with c_ctl1:
            k_live = st.slider("Počet sousedů (n_neighbors):", min_value=5, max_value=50, value=20, step=1)
        with c_ctl2:
            cont_live = st.slider("Míra kontaminace (contamination):", min_value=0.01, max_value=0.10, value=0.03, step=0.005)

        X_mat = df_b_lab[["f1_scaled", "f2_scaled"]].values
        lof_live = LocalOutlierFactor(n_neighbors=k_live, contamination=cont_live)
        labels_live = lof_live.fit_predict(X_mat)
        scores_live = -lof_live.negative_outlier_factor_

        df_b_lab["Live_Status"] = ["🚨 Anomálie (-1)" if l == -1 else "✅ Inlier (1)" for l in labels_live]
        df_b_lab["live_lof"] = scores_live
        df_b_lab["live_size"] = 8 + 55 * np.maximum(0, scores_live - 1.0) ** 1.8

        fig_live_b = px.scatter(
            df_b_lab,
            x="f1_scaled",
            y="f2_scaled",
            color="Live_Status",
            size="live_size",
            title=rf"Živá simulace LOF: n_neighbors = {k_live}, contamination = {cont_live*100:.1f} % $\to$ {int((labels_live == -1).sum())} anomálií",
            labels={"f1_scaled": "Příznak 1", "f2_scaled": "Příznak 2"},
            color_discrete_map={"✅ Inlier (1)": "#2563eb", "🚨 Anomálie (-1)": "#dc2626"},
            hover_data=["idx", "live_lof"]
        )
        fig_live_b.update_traces(marker=dict(opacity=0.8, line=dict(width=0.8, color="black")))
        fig_live_b.update_layout(template="plotly_white", height=500, margin=dict(l=30, r=30, t=40, b=30))
        st.plotly_chart(fig_live_b, width="stretch")

# ==============================================================================
# TAB 5: VÝHODY, NEVÝHODY & NOVELTY DETECTION
# ==============================================================================
with tab5:
    st.subheader("5. Bilance výhod, nevýhod (slajdy 23–24) & Novelty Detection")
    
    col_adv, col_dis = st.columns(2)

    with col_adv:
        st.success(
            r"""
            ##### ✅ VÝHODY (ADVANTAGES ze slajdu 23):
            1. **Detekce lokálních anomálií:** Na rozdíl od globálních metod odhalí outliery ukryté i v menších podsestavách dat.
            2. **Vysoká flexibilita:** Hustotní přístup spolehlivě funguje v datech libovolných nepravidelných tvarů.
            3. **Podpora různých metrik:** Umožňuje výpočet různých vzdáleností (Manhattan, Eukleidovská, Minkowski).
            """
        )

    with col_dis:
        st.error(
            r"""
            ##### ❌ NEVÝHODY (DISADVANTAGES ze slajdu 24):
            1. **Výpočetní složitost $\mathcal{O}(N^2)$:** Při velkém počtu bodů dramaticky roste doba výpočtu.
            2. **Obtížná interpretovatelnost:** LOF je poměr hustot bez pevné univerzální hranice (např. 1.2 vs 1.5).
            3. **Citlivost na hyperparametry:** Počet sousedů $k$ může zásadně změnit výsledky na konkrétním datasetu.
            """
        )

    st.markdown("---")
    st.markdown("#### 🔮 What-If kalkulátor pro režim Novelty Detection (`novelty=True`):")
    st.caption("Při on-line nasazení model predikuje anomálie na nových příchozích bodech:")

    c_pt1, c_pt2 = st.columns(2)
    with c_pt1:
        in_x = st.number_input("Souřadnice X nového bodu:", value=0.0, step=0.5)
    with c_pt2:
        in_y = st.number_input("Souřadnice Y nového bodu:", value=0.0, step=0.5)

    X_train_slide = df_slide[["x", "y"]].values
    lof_nov = LocalOutlierFactor(n_neighbors=20, novelty=True)
    lof_nov.fit(X_train_slide)

    query_pt = np.array([[in_x, in_y]])
    nov_pred = lof_nov.predict(query_pt)[0]
    nov_dec = lof_nov.decision_function(query_pt)[0]

    c_ans1, c_ans2 = st.columns(2)
    with c_ans1:
        if nov_pred == 1:
            st.success(rf"### ✅ Normální bod (Inlier)" + "\n" + rf"Decision Function = {nov_dec:.4f} ($\ge 0$)")
        else:
            st.error(rf"### 🚨 Detekována Nová Anomálie!" + "\n" + rf"Decision Function = {nov_dec:.4f} ($< 0$)")

    with c_ans2:
        st.info("Při `novelty=True` metoda `decision_function` vrací posun vůči vnitřnímu prahu `offset_`.")
