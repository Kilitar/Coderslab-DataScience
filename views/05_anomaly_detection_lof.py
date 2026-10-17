"""
Den 5: Neřízené učení – Praktická implementace Local Outlier Factor (LOF)
========================================================================
Praktická laboratoř a Scikit-learn implementace:
1. Vzorové natrénování LOF na datasetu blobs_dataset.csv (1 000 pozorování).
2. Experimentální demonstrace triumfu LOF při proměnlivé hustotě (Varying Density).
3. Interaktivní laboratoř: ladění parametrů n_neighbors a contamination v reálném čase.
4. Režim Novelty Detection (novelty=True) pro inferenci na nových datech.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler
import streamlit as st

st.set_page_config(page_title="Implementace: LOF v Scikit-learn", page_icon="🛠️", layout="wide")

st.title("🛠️ Praktická implementace: Local Outlier Factor (LOF) v Scikit-learn")
st.caption(
    "Aplikace algoritmu Local Outlier Factor na datech blobs_dataset.csv (1 000 bodů), vizualizace skóre anomality, "
    "demonstrace chování při proměnlivé hustotě (Varying Density) a živý What-If simulátor Novelty Detection."
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
                    if isinstance(d, dict) and "blobs_points" in d and "multi_density_points" in d:
                        return d
            except Exception:
                pass

    # Fallback on-the-fly výpočet
    candidates_csv = [
        Path("data/Downloadable materials for Session 3/blobs_dataset.csv"),
        Path(__file__).resolve().parent.parent / "data/Downloadable materials for Session 3/blobs_dataset.csv",
        Path("blobs_dataset.csv"),
    ]
    df_raw = None
    for cp in candidates_csv:
        if cp.exists():
            df_raw = pd.read_csv(cp)
            break

    if df_raw is None:
        # Syntetický fallback
        np.random.seed(42)
        df_raw = pd.DataFrame(np.random.normal(size=(1000, 2)), columns=["feature_1", "feature_2"])

    X_b = df_raw[["feature_1", "feature_2"]].values
    sc = StandardScaler()
    X_bs = sc.fit_transform(X_b)

    lof_model = LocalOutlierFactor(n_neighbors=20, contamination=0.03)
    b_labels = lof_model.fit_predict(X_bs)
    b_scores = (-lof_model.negative_outlier_factor_).tolist()

    blobs_pts = []
    for i in range(len(df_raw)):
        blobs_pts.append({
            "idx": i,
            "f1_raw": float(X_b[i, 0]),
            "f2_raw": float(X_b[i, 1]),
            "f1_scaled": float(X_bs[i, 0]),
            "f2_scaled": float(X_bs[i, 1]),
            "lof_score": round(float(b_scores[i]), 4),
            "is_lof_outlier": bool(b_labels[i] == -1),
            "is_iforest_outlier": False,
            "if_score": 0.5
        })

    return {
        "blobs_meta": {"n_samples": len(df_raw), "lof_outliers_count": int(np.sum(b_labels == -1))},
        "blobs_points": blobs_pts,
        "multi_density_points": []
    }


lof_data = load_lof_implementation_data()
df_blobs = pd.DataFrame(lof_data["blobs_points"])
b_meta = lof_data.get("blobs_meta", {})

# Horní KPI karty
c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Velikost datasetu",
    f"{len(df_blobs):,} bodů".replace(",", " "),
    delta="blobs_dataset.csv",
    delta_color="off"
)
c2.metric(
    "Hyperparametry",
    "k = 20, contam = 3 %",
    delta="LocalOutlierFactor",
    delta_color="normal"
)
n_outliers = int(df_blobs["is_lof_outlier"].sum())
c3.metric(
    "Identifikované anomálie",
    f"{n_outliers} bodů ({n_outliers / len(df_blobs) * 100:.1f} %)",
    delta="Label -1 (Outliers)",
    delta_color="inverse"
)
max_lof = float(df_blobs["lof_score"].max()) if not df_blobs.empty else 1.0
c4.metric(
    "Maximální LOF skóre",
    f"{max_lof:.3f}",
    delta="Extrémní lokální odlehlost",
    delta_color="inverse"
)

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "📋 1. Vzorová implementace na blobs_dataset",
    "🔬 2. Triumf LOF při proměnlivé hustotě",
    "🧪 3. Interaktivní laboratoř parametrů",
    "🔮 4. Outlier Detection vs. Novelty Detection"
])

# ==============================================================================
# TAB 1: IMPLEMENTACE NA BLOBS_DATASET
# ==============================================================================
with tab1:
    st.subheader("1. Vzorová implementace LOF na datech blobs_dataset.csv")
    st.markdown(
        r"""
        Aplikace algoritmu `LocalOutlierFactor` na standardizovaných 2D datech s 1 000 pozorováními.
        Pro každé pozorování je spočten faktor lokální odlehlosti ($\text{LOF} \ge 1.0$) 
        a body s nejvyšším skóre jsou označeny jako anomálie (label `-1`).
        """
    )

    c_g1, c_g2 = st.columns(2)

    with c_g1:
        # Binární scatter
        df_blobs["Status"] = ["🚨 Anomálie (-1)" if x else "✅ Normální bod (Inlier)" for x in df_blobs["is_lof_outlier"]]
        fig_bin = px.scatter(
            df_blobs,
            x="f1_scaled",
            y="f2_scaled",
            color="Status",
            title=f"LOF: Binární klasifikace anomálií ({n_outliers} z {len(df_blobs)} bodů)",
            labels={"f1_scaled": "Standardizovaný Příznak 1", "f2_scaled": "Standardizovaný Příznak 2"},
            color_discrete_map={"✅ Normální bod (Inlier)": "#2563eb", "🚨 Anomálie (-1)": "#dc2626"}
        )
        fig_bin.update_traces(marker=dict(size=7, opacity=0.8, line=dict(width=0.8, color="black")))
        fig_bin.update_layout(template="plotly_white", height=430, margin=dict(l=30, r=30, t=40, b=30))
        st.plotly_chart(fig_bin, width="stretch")

    with c_g2:
        # Bublinový graf LOF skóre
        # Velikost bubliny úměrná (LOF - 1)^1.8
        df_blobs["bubble_size"] = 8 + 55 * np.maximum(0, df_blobs["lof_score"] - 1.0) ** 1.8
        fig_bub = px.scatter(
            df_blobs,
            x="f1_scaled",
            y="f2_scaled",
            color="lof_score",
            size="bubble_size",
            title="LOF: Spojité skóre anomality (Velikost bubliny ~ LOF)",
            labels={"f1_scaled": "Příznak 1", "f2_scaled": "Příznak 2", "lof_score": "LOF faktor"},
            color_continuous_scale="YlOrRd",
            hover_data=["idx", "lof_score", "Status"]
        )
        fig_bub.update_traces(marker=dict(opacity=0.8, line=dict(width=0.8, color="black")))
        fig_bub.update_layout(template="plotly_white", height=430, margin=dict(l=30, r=30, t=40, b=30))
        st.plotly_chart(fig_bub, width="stretch")

    st.markdown("#### 💻 Scikit-learn implementační kód:")
    st.code(
        r"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor

# 1. Načtení dat a standardizace
df = pd.read_csv("blobs_dataset.csv")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[["feature_1", "feature_2"]])

# 2. Inicializace a trénování LOF
# contamination=0.03 definuje, že 3 % bodů s nejnižší hustotou označíme jako anomálie
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.03)
labels = lof.fit_predict(X_scaled)  # Vrátí 1 pro Inliery a -1 pro Outliery

# 3. Získání původního kladného LOF skóre
# negative_outlier_factor_ v Scikit-learn nabývá záporných hodnot (čím zápornější, tím větší anomálie)
lof_scores = -lof.negative_outlier_factor_

print(f"Detekováno anomálií: {(labels == -1).sum()} z {len(df)} pozorování")
print(f"Maximální LOF skóre: {lof_scores.max():.4f}")
        """,
        language="python"
    )

# ==============================================================================
# TAB 2: TRIUMF LOF PŘI PROMĚNLIVÉ HUSTOTĚ
# ==============================================================================
with tab2:
    st.subheader("2. Proč globální metody selhávají a LOF bezchybně vítězí?")
    st.markdown(
        r"""
        Zásadní myšlenkový průlom algoritmu LOF (Breunig et al., 2000) spočívá v řešení **proměnlivé hustoty (Varying Density)**:
        * **Hustý shluk $C_1$ (vlevo):** Body jsou natěsnané těsně u sebe. Vzdálenost mezi sousedy je jen $\approx 0.2$.
        * **Řídký shluk $C_2$ (vpravo):** Body jsou široce rozptýlené. Běžná vzdálenost mezi sousedy je $\approx 1.0$.
        * **Lokální anomálie:** Body ležící kousek vedle hustého shluku $C_1$ mají vzdálenost $\approx 0.6$.
        """
    )

    multi_pts = lof_data.get("multi_density_points", [])
    if multi_pts:
        df_multi = pd.DataFrame(multi_pts)

        # Spočteme globální k-NN distance (bez hustotní normalizace)
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
                title="❌ Globální vzdálenost k-NN (Selhání)",
                color_discrete_map={"✅ Inlier": "#2563eb", "🚨 Falešná anomálie": "#dc2626"},
                hover_data=["true_type", "k_distance"]
            )
            fig_glob.update_traces(marker=dict(size=8, opacity=0.85, line=dict(width=0.8, color="black")))
            fig_glob.update_layout(template="plotly_white", height=420, margin=dict(l=30, r=30, t=40, b=30))
            st.plotly_chart(fig_glob, width="stretch")
            st.error(
                "Globální vzdálenost selhává! Označí regulérní body v řídkém shluku $C_2$ za anomálie, "
                "zatímco lokální anomálie u hustého shluku $C_1$ zcela přehlédne."
            )

        with c_v2:
            fig_lof_t = px.scatter(
                df_multi,
                x="x",
                y="y",
                color="LOF_Detection",
                title="✅ Local Outlier Factor (Triumf)",
                color_discrete_map={"✅ Inlier": "#2563eb", "🚨 Korektní anomálie": "#10b981"},
                hover_data=["true_type", "lof_score"]
            )
            fig_lof_t.update_traces(marker=dict(size=8, opacity=0.85, line=dict(width=0.8, color="black")))
            fig_lof_t.update_layout(template="plotly_white", height=420, margin=dict(l=30, r=30, t=40, b=30))
            st.plotly_chart(fig_lof_t, width="stretch")
            st.success(
                "LOF triumfuje! Porovnává hustotu s okolím: bod u $C_1$ má mnohem nižší hustotu než jeho sousedé, "
                "takže je správně detekován, zatímco body v $C_2$ mají stejnou hustotu jako jejich sousedé a zůstávají Inliery."
            )

# ==============================================================================
# TAB 3: INTERAKTIVNÍ LABORATOŘ PARAMETRŮ
# ==============================================================================
with tab3:
    st.subheader(r"3. Interaktivní laboratoř: Živé ladění hyperparametrů LOF")
    st.markdown(
        r"""
        Vyzkoušejte si, jak změna počtu sousedů `n_neighbors` a očekávané míry kontaminace `contamination` 
        ovlivňuje identifikaci odlehlých bodů na datech `blobs_dataset.csv`.
        """
    )

    c_s1, c_s2 = st.columns(2)
    with c_s1:
        user_k = st.slider(
            "Počet sousedů (n_neighbors):",
            min_value=5,
            max_value=60,
            value=20,
            step=1,
            help="Definuje velikost okolí pro výpočet dosahové hustoty LRD. Doporučeno 15–35."
        )
    with c_s2:
        user_contam = st.slider(
            "Míra kontaminace (contamination):",
            min_value=0.005,
            max_value=0.10,
            value=0.03,
            step=0.005,
            format="%.3f",
            help="Očekávaný podíl anomálií v datasetu. Určuje práh pro rozdělení na Inlier / Outlier."
        )

    # Živý přepočet LOF (< 20 ms)
    X_live = df_blobs[["f1_scaled", "f2_scaled"]].values
    live_lof = LocalOutlierFactor(n_neighbors=user_k, contamination=user_contam)
    live_labels = live_lof.fit_predict(X_live)
    live_scores = -live_lof.negative_outlier_factor_

    live_outliers_count = int(np.sum(live_labels == -1))
    live_inliers_count = int(np.sum(live_labels == 1))

    c_m1, c_m2, c_m3 = st.columns(3)
    c_m1.metric("Počet detekovaných anomálií", f"{live_outliers_count} bodů", delta=f"{user_contam*100:.1f} % z datasetu")
    c_m2.metric("Počet normálních bodů (Inliers)", f"{live_inliers_count} bodů", delta=f"{live_inliers_count/len(df_blobs)*100:.1f} %")
    c_m3.metric("Práh LOF pro anomálii", f"{live_scores[live_labels == -1].min():.3f}", delta=f"Max LOF = {live_scores.max():.3f}")

    df_sim = df_blobs.copy()
    df_sim["Sim_Status"] = ["🚨 Anomálie (-1)" if l == -1 else "✅ Normální bod (1)" for l in live_labels]
    df_sim["live_score"] = live_scores
    df_sim["live_size"] = 8 + 55 * np.maximum(0, live_scores - 1.0) ** 1.8

    fig_sim = px.scatter(
        df_sim,
        x="f1_scaled",
        y="f2_scaled",
        color="Sim_Status",
        size="live_size",
        title=rf"Živá simulace LOF: n_neighbors = {user_k}, contamination = {user_contam*100:.1f} %",
        labels={"f1_scaled": "Příznak 1", "f2_scaled": "Příznak 2", "Sim_Status": "Klasifikace"},
        color_discrete_map={"✅ Normální bod (1)": "#2563eb", "🚨 Anomálie (-1)": "#dc2626"},
        hover_data=["idx", "live_score"]
    )
    fig_sim.update_traces(marker=dict(opacity=0.8, line=dict(width=0.8, color="black")))
    fig_sim.update_layout(template="plotly_white", height=500, margin=dict(l=30, r=30, t=40, b=30))
    st.plotly_chart(fig_sim, width="stretch")

# ==============================================================================
# TAB 4: OUTLIER VS NOVELTY DETECTION
# ==============================================================================
with tab4:
    st.subheader("4. Outlier Detection vs. Novelty Detection v Scikit-learn API")
    st.markdown(
        r"""
        V praxi je nutné rozlišovat dva zcela odlišné scénáře nasazení:
        
        1. **Outlier Detection (`novelty=False`, výchozí):**
           * Cíl: Nalezení anomálií v **historickém trénovacím datasetu**.
           * Použití: Čištění dat, odhalení chyb měření, audit databáze.
           * Metody: Pouze `fit_predict(X)`.
        
        2. **Novelty Detection (`novelty=True`):**
           * Cíl: Model se natrénuje na čistých datech a je nasazen do produkční pipeline k **vyhodnocování nových, dosud neviděných dat v reálném čase**.
           * Použití: On-line fraud monitoring, vyhodnocování telemetrie za běhu.
           * Metody: Podporuje `predict(X_new)` a `decision_function(X_new)`.
        """
    )

    st.markdown("#### 🧪 What-If kalkulátor pro nově příchozí pozorování (`novelty=True`):")
    st.caption("Vyzkoušejte zadat souřadnice nového bodu a sledujte produkční rozhodnutí modelu:")

    c_in1, c_in2 = st.columns(2)
    with c_in1:
        new_x = st.number_input("Nový bod: Standardizovaný Příznak 1 ($z_1$):", value=0.0, step=0.2)
    with c_in2:
        new_y = st.number_input("Nový bod: Standardizovaný Příznak 2 ($z_2$):", value=0.0, step=0.2)

    # Natrénujeme model novelty=True
    lof_nov = LocalOutlierFactor(n_neighbors=20, novelty=True)
    lof_nov.fit(X_live)

    sample_pt = np.array([[new_x, new_y]])
    pred_res = lof_nov.predict(sample_pt)[0]
    dec_score = lof_nov.decision_function(sample_pt)[0]

    c_res1, c_res2 = st.columns(2)
    with c_res1:
        if pred_res == 1:
            st.success(rf"### ✅ Výsledek: Normální pozorování (Inlier)" + "\n" + rf"Decision Function = {dec_score:.4f} ($\ge 0$)")
        else:
            st.error(rf"### 🚨 Výsledek: Detekována Nová Anomálie!" + "\n" + rf"Decision Function = {dec_score:.4f} ($< 0$)")

    with c_res2:
        st.info(
            r"""
            **Rozhodovací pravidlo Novelty Detection:**
            * $\text{decision\_function} \ge 0 \implies$ Bod leží v dostatečně husté oblasti (normální).
            * $\text{decision\_function} < 0 \implies$ Bod leží v podkriticky řídkém prostoru (anomálie).
            """
        )
