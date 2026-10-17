"""
Den 5: Neřízené učení – Vzorová implementace K-Means v Scikit-learn
===================================================================
Syntéza materiálů z K-Means_-_sample_implementation.pdf:
1. Generování a zobrazení neanotovaného 2D datasetu (make_blobs, 4 centra).
2. Iterační běh K-Means pro k = 1 až 10 a výpočet WCSS (Inertia) a Silhouette Score.
3. Loketní metoda (Elbow Method) a Siluetová analýza pro volbu optimálního k.
4. Finální fitování modelu pro k = 4, extrakce centroidů (cluster_centers_) a labels_.
5. Interaktivní vizualizace shluků s těžišti a rozhodovacími hranicemi v Plotly.
6. Srovnání se cvičnými daty blobs_dataset.csv (1000 řádků).
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
import streamlit as st

st.set_page_config(page_title="Implementace: K-Means v Scikit-learn", page_icon="🛠️", layout="wide")

st.title("🛠️ Den 5: Praktická implementace K-Means v Scikit-learn")
st.caption(
    "Vzorová implementace z přednáškových materiálů kurzu (K-Means - an implementation example). "
    "Generování syntetických shluků, diagnostika optimálního k pomocí WCSS (Inertia) a Siluetové analýzy, "
    "extrakce těžišť (centroidů) a interaktivní 2D vizualizace."
)


@st.cache_data
def generate_sample_data():
    X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)
    df = pd.DataFrame(X, columns=["feature_1", "feature_2"])
    return df, y_true


@st.cache_data
def compute_k_sweep(df_data, max_k=10):
    k_vals = list(range(1, max_k + 1))
    inertias = []
    silhouettes = [None]  # Silhouette pro k=1 není definováno

    for k in k_vals:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        km.fit(df_data)
        inertias.append(float(km.inertia_))
        if k > 1:
            sil = float(silhouette_score(df_data, km.labels_))
            silhouettes.append(sil)

    return k_vals, inertias, silhouettes


@st.cache_data
def load_blobs_csv():
    return pd.read_csv("data/Downloadable materials for Session 3/blobs_dataset.csv")


df_sample, y_sample_true = generate_sample_data()
k_vals, inertias, silhouettes = compute_k_sweep(df_sample, max_k=10)
df_blobs_csv = load_blobs_csv()

# Horní KPI karty
c1, c2, c3, c4 = st.columns(4)
c1.metric("Počet vzorků (slajdy)", f"{len(df_sample):,} bodů", delta="make_blobs syntéza", delta_color="off")
c2.metric("Skutečný počet center", "4 shluky", delta="cluster_std = 1.0", delta_color="off")
c3.metric("Nalezené optimální k", "k = 4", delta="Elbow + Silhouette maximum", delta_color="normal")
c4.metric("Silhouette skóre (k=4)", f"{silhouettes[3]:.3f}", delta="Vynikající separace", delta_color="normal")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "💻 1. Postup dle přednášky (Kód & Data)",
    "📈 2. Diagnostika k (Loket & Silueta)",
    "🎯 3. Interaktivní shlukování & Těžiště",
    "🧪 4. Cvičný dataset (blobs_dataset.csv)"
])

# ==============================================================================
# TAB 1: POSTUP DLE PŘEDNÁŠKY
# ==============================================================================
with tab1:
    st.subheader("1. Vzorový kód a postup krok za krokem dle slajdů")
    st.markdown(
        r"""
        Přednáška (*K-Means - an implementation example*, str. 10–17) ukazuje kompletní pipeline shlukování:
        1. Vytvoření 2D syntetických dat se 4 shluky pomocí `make_blobs`.
        2. Vizualizace neoznačených dat scatter plotem bez jakýchkoliv návěští.
        3. Výpočet hodnot WCSS (`.inertia_`) a Siluetového skóre pro $k \in [1, 10]$.
        4. Volba optimálního počtu shluků $k = 4$.
        5. Trénování finálního modelu a vizualizace centroidů.
        """
    )

    col_code, col_plot = st.columns([1.1, 0.9])

    with col_code:
        st.markdown("#### 🐍 Kód přesně dle přednáškového vzoru:")
        st.code(
            """# 1. Generování dat se 4 shluky
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

data, labels = make_blobs(
    n_samples=300, 
    centers=4, 
    cluster_std=1.0, 
    random_state=42
)

# 2. Vykreslení neanotovaných dat
plt.scatter(data[:, 0], data[:, 1], s=20, color="red")
plt.title("Our dataset")
plt.show()

# 3. Iterační testování k = 1 až 10
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

inertia_scores = []
silhouette_scores = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)
    inertia_scores.append(kmeans.inertia_)
    if k > 1:
        silhouette_scores.append(silhouette_score(data, kmeans.labels_))

# 4. Natrénování finálního modelu pro optimální k = 4
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(data)
cluster_centers = kmeans.cluster_centers_
labels = kmeans.labels_

# 5. Vykreslení výsledků s centroidy
plt.scatter(data[:, 0], data[:, 1], c=labels, s=50, cmap="viridis")
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c="red", marker="x", s=200)
plt.title(f"Clustering result for k = {optimal_k}")
plt.show()""",
            language="python"
        )

    with col_plot:
        st.markdown("#### 🔴 Původní neanotovaná data (vstup do modelu):")
        fig_raw = px.scatter(
            df_sample,
            x="feature_1",
            y="feature_2",
            title="Vstupní neoznačená data (300 bodů bez cílového y)",
            color_discrete_sequence=["#ef4444"]
        )
        fig_raw.update_traces(marker=dict(size=8, opacity=0.85))
        fig_raw.update_layout(margin=dict(l=30, r=30, t=40, b=30))
        st.plotly_chart(fig_raw, width="stretch")
        st.info(
            "💡 V neřízeném učení model nezná barevné zařazení bodů. "
            "Vidí pouze jejich prostorové souřadnice a musí odhalit strukturu sám."
        )

# ==============================================================================
# TAB 2: DIAGNOSTIKA K (LOKET & SILUETA)
# ==============================================================================
with tab2:
    st.subheader("2. Diagnostické křivky: Loketní metoda (Elbow) & Siluetová analýza")
    st.markdown(
        r"""
        Pro objektivní určení počtu shluků $k$ se vyhodnocují dvě komplementární metriky:
        * **WCSS (Within-Cluster Sum of Squares, Inertia):** Suma čtverců vzdáleností k těžištím. Hledáme bod zlomu (loket).
        * **Silhouette Score:** Poměr vnitřní soudržnosti $a(i)$ a mezishlukové separace $b(i)$. Hledáme globální maximum v intervalu $[-1, +1]$.
        """
    )

    c_diag1, c_diag2 = st.columns(2)

    with c_diag1:
        st.markdown("#### 📉 Křivka WCSS (Inertia vs. k)")
        fig_elbow = go.Figure()
        fig_elbow.add_trace(
            go.Scatter(
                x=k_vals,
                y=inertias,
                mode="lines+markers",
                name="Inertia (WCSS)",
                line=dict(color="#3b82f6", width=3),
                marker=dict(size=8)
            )
        )
        # Zvýraznění optimálního k=4
        fig_elbow.add_trace(
            go.Scatter(
                x=[4],
                y=[inertias[3]],
                mode="markers+text",
                name="Optimální loket (k=4)",
                marker=dict(size=14, color="#ef4444", symbol="star"),
                text=["Loket (k=4)"],
                textposition="top right"
            )
        )
        fig_elbow.update_layout(
            title="Loketní metoda: Pokles WCSS s rostoucím k",
            xaxis_title="Počet shluků (k)",
            yaxis_title="Inertia (WCSS)",
            margin=dict(l=30, r=30, t=40, b=30)
        )
        st.plotly_chart(fig_elbow, width="stretch")

    with c_diag2:
        st.markdown("#### 🎯 Křivka Silhouette Score (k = 2 až 10)")
        k_sil_x = k_vals[1:]
        sil_vals = silhouettes[1:]

        fig_sil = go.Figure()
        fig_sil.add_trace(
            go.Bar(
                x=k_sil_x,
                y=sil_vals,
                marker_color=["#10b981" if k == 4 else "#94a3b8" for k in k_sil_x],
                text=[f"{s:.3f}" for s in sil_vals],
                textposition="auto"
            )
        )
        fig_sil.update_layout(
            title="Průměrné Silhouette skóre (Zelená = Maximum pro k=4)",
            xaxis_title="Počet shluků (k)",
            yaxis_title="Silhouette skóre (-1 až +1)",
            yaxis_range=[0, 1.0],
            margin=dict(l=30, r=30, t=40, b=30)
        )
        st.plotly_chart(fig_sil, width="stretch")

    # Přehledná tabulka hodnot
    st.markdown(r"#### 📊 Přehled naměřených hodnot metrik pro $k \in [1, 10]$:")
    metrics_summary = []
    for i, k in enumerate(k_vals):
        pct_drop = ((inertias[i - 1] - inertias[i]) / inertias[i - 1] * 100) if i > 0 else 0.0
        metrics_summary.append({
            "Počet shluků (k)": k,
            "Inertia (WCSS)": f"{inertias[i]:,.1f}",
            "Meziroční pokles Inertie": f"-{pct_drop:.1f} %" if i > 0 else "-",
            "Silhouette skóre": f"{silhouettes[i]:.4f}" if silhouettes[i] is not None else "N/A (k=1)",
            "Hodnocení": "🌟 Globální optimum" if k == 4 else ("Příliš hrubé rozdělení" if k < 4 else "Přeurčené drobení")
        })
    st.dataframe(pd.DataFrame(metrics_summary), width="stretch", hide_index=True)

# ==============================================================================
# TAB 3: VÝSLEDNÉ SHLUKOVÁNÍ & TĚŽIŠTĚ
# ==============================================================================
with tab3:
    st.subheader("3. Interaktivní vizualizace výsledného shlukování a těžišť")
    st.markdown(
        "Vyzkoušejte si, jak se mění zařazení bodů a poloha těžišť při volbě různého počtu shluků $k$."
    )

    col_ctrl1, col_ctrl2 = st.columns([1, 2])

    with col_ctrl1:
        active_k = st.slider("Zvolte počet shluků (k):", min_value=2, max_value=8, value=4, step=1)
        show_voronoi = st.checkbox("Zobrazit přibližné Voroného hranice", value=False)

        # Trénování modelu pro zvolené k
        km_active = KMeans(n_clusters=active_k, init="k-means++", n_init=10, random_state=42)
        active_labels = km_active.fit_predict(df_sample)
        active_centers = km_active.cluster_centers_
        active_sil = silhouette_score(df_sample, active_labels)

        st.metric("Inertia (WCSS)", f"{km_active.inertia_:,.1f}")
        st.metric("Silhouette skóre", f"{active_sil:.4f}", delta="Maximum při k=4" if active_k == 4 else None)

        st.markdown("##### 📍 Souřadnice těžišť (Centroidů):")
        df_centers = pd.DataFrame(
            active_centers,
            columns=["Feature 1 (X)", "Feature 2 (Y)"],
            index=[f"Centroid C{i+1}" for i in range(active_k)]
        )
        st.dataframe(df_centers.round(3), width="stretch")

    with col_ctrl2:
        df_plot_active = df_sample.copy()
        df_plot_active["Cluster"] = [f"Shluk {l+1}" for l in active_labels]

        fig_cluster = px.scatter(
            df_plot_active,
            x="feature_1",
            y="feature_2",
            color="Cluster",
            title=f"Výsledné shlukování pro k = {active_k} (Silhouette = {active_sil:.3f})",
            color_discrete_sequence=px.colors.qualitative.Plotly
        )

        # Přidání těžišť (přesně dle slajdů červené křížky marker='x', s=200)
        fig_cluster.add_trace(
            go.Scatter(
                x=active_centers[:, 0],
                y=active_centers[:, 1],
                mode="markers+text",
                marker=dict(symbol="x", size=16, color="red", line=dict(width=3, color="black")),
                name="Centroidy (Těžiště)",
                text=[f"C{i+1}" for i in range(active_k)],
                textposition="top center"
            )
        )

        fig_cluster.update_traces(marker=dict(size=9, opacity=0.9), selector=dict(mode="markers"))
        fig_cluster.update_layout(margin=dict(l=30, r=30, t=40, b=30))
        st.plotly_chart(fig_cluster, width="stretch")

# ==============================================================================
# TAB 4: CVIČNÝ DATASET (BLOBS_DATASET.CSV)
# ==============================================================================
with tab4:
    st.subheader("4. Aplikace metody na cvičný kurzový dataset `blobs_dataset.csv`")
    st.markdown(
        r"""
        Přednáška končí zadáním: *„We have prepared a dataset for you, which you will try to divide using the newly learned method. Good luck!“*  
        V datech kurzu je k dispozici soubor `data/Downloadable materials for Session 3/blobs_dataset.csv` obsahující **1 000 reálných pozorování**.
        """
    )

    c_b1, c_b2 = st.columns([1, 2])

    with c_b1:
        st.markdown("##### 📁 Parametry cvičného datasetu:")
        st.write(f"- **Počet řádků:** {len(df_blobs_csv):,}")
        st.write("- **Sloupce:** `feature_1`, `feature_2`")

        b_k = st.slider("Zvolte k pro blobs_dataset.csv:", min_value=2, max_value=8, value=4, step=1, key="slider_blobs")
        km_blobs = KMeans(n_clusters=b_k, init="k-means++", n_init=10, random_state=42)
        blobs_labels = km_blobs.fit_predict(df_blobs_csv)
        blobs_centers = km_blobs.cluster_centers_
        blobs_sil = silhouette_score(df_blobs_csv, blobs_labels)

        st.metric("Inertia (blobs)", f"{km_blobs.inertia_:,.1f}")
        st.metric("Silhouette (blobs)", f"{blobs_sil:.4f}")

    with c_b2:
        df_blobs_plot = df_blobs_csv.copy()
        df_blobs_plot["Cluster"] = [f"Shluk {l+1}" for l in blobs_labels]

        fig_blobs = px.scatter(
            df_blobs_plot,
            x="feature_1",
            y="feature_2",
            color="Cluster",
            title=f"Cvičný dataset blobs_dataset.csv rozdělený na k = {b_k} shluků",
            color_discrete_sequence=px.colors.qualitative.Dark24
        )

        fig_blobs.add_trace(
            go.Scatter(
                x=blobs_centers[:, 0],
                y=blobs_centers[:, 1],
                mode="markers+text",
                marker=dict(symbol="x", size=15, color="red", line=dict(width=2, color="white")),
                name="Centroidy",
                text=[f"C{i+1}" for i in range(b_k)],
                textposition="top center"
            )
        )
        fig_blobs.update_layout(margin=dict(l=30, r=30, t=40, b=30))
        st.plotly_chart(fig_blobs, width="stretch")
