import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import streamlit as st

st.set_page_config(page_title="Teorie: Shlukování & K-Means", page_icon="📖", layout="wide")

st.title("📖 Den 5: Neřízené učení – Algoritmus K-Means & Shluková analýza")
st.caption(
    "Teoretický a praktický průvodce rozkladovým shlukováním: Minimalizace WCSS (Inertia), "
    "princip Expectation-Maximization, K-Means++, diagnostika loketní metodou a Siluetovou analýzou."
)

# Horní KPI karty
c1, c2, c3, c4 = st.columns(4)
c1.metric("1. Paradigma", "Neřízené učení", delta="Bez cílové proměnné (y)", delta_color="off")
c2.metric("2. Cíl optimalizace", "Minimalizace WCSS", delta="Suma čtverců vzdáleností k těžištím", delta_color="normal")
c3.metric("3. Inicializace", "K-Means++", delta="Pravděpodobnostní rozptyl centroidů", delta_color="normal")
c4.metric("4. Diagnostika k", "Elbow & Silhouette", delta="Hledání optimálního počtu shluků", delta_color="normal")

st.markdown("---")


@st.cache_data
def load_blobs_data():
    return pd.read_csv("data/Downloadable materials for Session 3/blobs_dataset.csv")


@st.cache_data
def compute_clustering_metrics(df_points):
    inertias = []
    silhouettes = []
    k_range = list(range(2, 11))
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        km.fit(df_points)
        inertias.append(float(km.inertia_))
        silhouettes.append(float(silhouette_score(df_points, km.labels_)))
    return k_range, inertias, silhouettes


df_blobs = load_blobs_data()
k_range, inertias, silhouettes = compute_clustering_metrics(df_blobs)

tab1, tab2, tab3, tab4 = st.tabs([
    "📐 1. Matematický princip & Lloydův algoritmus",
    "🧪 2. Interaktivní simulátor shlukování (blobs)",
    "📈 3. Volba k: Loketní metoda & Silueta",
    "⚠️ 4. Limity & Kdy K-Means selhává"
])

# ==============================================================================
# TAB 1: MATEMATICKÝ PRINCIP
# ==============================================================================
with tab1:
    st.subheader("1. Jak funguje algoritmus K-Means (Lloydův algoritmus)?")

    st.markdown(
        r"""
        Algoritmus K-Means rozděluje $n$ pozorování do $k$ disjunktních shluků $S = \{S_1, S_2, \dots, S_k\}$ 
        takovým způsobem, aby minimalizoval celkový součet čtverců vzdáleností bodů od jejich příslušných těžišť (**WCSS – Within-Cluster Sum of Squares**, v Scikit-learn nazývaný **Inertia**):
        
        $$\mathcal{J}(S, \mu) = \sum_{j=1}^{k} \sum_{x_i \in S_j} \|x_i - \mu_j\|^2$$
        
        Kde $\mu_j$ je centroid (aritmetický průměr) všech bodů náležejících do shluku $S_j$.
        """
    )

    col_em1, col_em2 = st.columns(2)

    with col_em1:
        st.markdown("#### 🔄 Iterační proces Expectation-Maximization (EM):")
        st.write(
            r"""
            1. **Inicializace (K-Means++):**  
               Vybere se $k$ počátečních těžišť $\mu_1, \dots, \mu_k$ tak, aby byla od sebe geometricky co nejdále.
            2. **Krok přiřazení (Assignment / E-step):**  
               Každý bod $x_i$ je přiřazen k nejbližšímu centroidu dle Euklidovské normy:
               $$S_j = \arg\min_l \|x_i - \mu_l\|^2$$
            3. **Krok aktualizace (Update / M-step):**  
               Každé těžiště se přepočítá jako těžiště nových členů:
               $$\mu_j = \frac{1}{|S_j|} \sum_{x_i \in S_j} x_i$$
            4. **Konvergence:**  
               Kroky 2 a 3 se opakují, dokud se poloha centroidů nezastaví (změna $<\varepsilon$).
            """
        )

    with col_em2:
        st.markdown("#### ⚡ Proč je K-Means++ tak důležitý?")
        st.write(
            r"""
            Původní náhodný výběr centroidů (Random initialization) často způsobil uvíznutí v mělkém lokálním minimu, 
            kdy dvě těžiště začínala ve stejném shluku.
            
            **K-Means++ (Arthur & Vassilvitskii, 2007)** řeší problém pravděpodobnostním vážením:
            * První centroid je náhodný bod.
            * Další centroidy jsou vybírány s pravděpodobností úměrnou **čtverci vzdálenosti** k nejbližšímu již zvolenému centroidu:
              $$P(x) = \frac{D(x)^2}{\sum D(x)^2}$$
            * Tím se zaručí maximální rozptyl těžišť po celém prostoru a výrazně rychlejší konvergence!
            """
        )

# ==============================================================================
# TAB 2: INTERAKTIVNÍ SIMULÁTOR
# ==============================================================================
with tab2:
    st.subheader("2. Interaktivní simulátor shlukování na datech `blobs_dataset.csv`")
    st.markdown(
        "Vyzkoušejte si, jak algoritmus K-Means reaguje na volbu počtu shluků $k$ a kde umisťuje centroidy."
    )

    col_sim_ctrl1, col_sim_ctrl2 = st.columns([1, 2])

    with col_sim_ctrl1:
        selected_k = st.slider("Počet shluků (k):", min_value=2, max_value=8, value=4, step=1)
        show_centroids = st.checkbox("Zobrazit centroidy (těžiště shluků)", value=True)
        
        # Fitování K-Means
        km_sim = KMeans(n_clusters=selected_k, init="k-means++", n_init=10, random_state=42)
        cluster_labels = km_sim.fit_predict(df_blobs)
        current_inertia = km_sim.inertia_
        current_sil = silhouette_score(df_blobs, cluster_labels)

        st.metric("Inertia (WCSS)", f"{current_inertia:,.1f}")
        st.metric("Silhouette skóre", f"{current_sil:.4f}")

    with col_sim_ctrl2:
        df_plot = df_blobs.copy()
        df_plot["Cluster"] = [f"Shluk {l+1}" for l in cluster_labels]

        fig_cluster = px.scatter(
            df_plot,
            x="feature_1",
            y="feature_2",
            color="Cluster",
            title=f"Výsledné shlukování pro k = {selected_k} (Silhouette = {current_sil:.3f})",
            color_discrete_sequence=px.colors.qualitative.Plotly
        )

        if show_centroids:
            centers = km_sim.cluster_centers_
            fig_cluster.add_trace(
                go.Scatter(
                    x=centers[:, 0],
                    y=centers[:, 1],
                    mode="markers+text",
                    marker=dict(symbol="x", size=14, color="black", line=dict(width=2, color="white")),
                    name="Centroidy (Těžiště)",
                    text=[f"C{i+1}" for i in range(selected_k)],
                    textposition="top center"
                )
            )

        fig_cluster.update_layout(margin=dict(l=40, r=40, t=50, b=40))
        st.plotly_chart(fig_cluster, width="stretch")

# ==============================================================================
# TAB 3: VOLBA K (ELBOW & SILHOUETTE)
# ==============================================================================
with tab3:
    st.subheader("3. Diagnostika optimálního počtu shluků $k$")

    c_chart1, c_chart2 = st.columns(2)

    with c_chart1:
        st.markdown("#### 📉 Loketní metoda (Elbow Method / WCSS)")
        fig_elbow = go.Figure()
        fig_elbow.add_trace(
            go.Scatter(
                x=k_range,
                y=inertias,
                mode="lines+markers",
                line=dict(color="#3b82f6", width=3),
                marker=dict(size=8)
            )
        )
        fig_elbow.update_layout(
            title="Křivka WCSS (Inertia) vs Počet shluků k",
            xaxis_title="Počet shluků (k)",
            yaxis_title="Inertia (WCSS)",
            margin=dict(l=40, r=40, t=50, b=40)
        )
        st.plotly_chart(fig_elbow, width="stretch")
        st.caption("Bod zlomu (loket) ukazuje, od kterého $k$ již přidávání dalších shluků nepřináší zásadní pokles chyby.")

    with c_chart2:
        st.markdown("#### 🎯 Siluetová analýza (Silhouette Score)")
        fig_sil = go.Figure()
        fig_sil.add_trace(
            go.Bar(
                x=k_range,
                y=silhouettes,
                marker_color="#10b981",
                text=[f"{s:.3f}" for s in silhouettes],
                textposition="auto"
            )
        )
        fig_sil.update_layout(
            title="Průměrné Silhouette skóre pro různé hodnoty k",
            xaxis_title="Počet shluků (k)",
            yaxis_title="Silhouette Score (-1 až +1)",
            yaxis_range=[0, 0.8],
            margin=dict(l=40, r=40, t=50, b=40)
        )
        st.plotly_chart(fig_sil, width="stretch")
        st.caption("Vyšší hodnota značí lepší vnitřní soudržnost a větší separaci od sousedních shluků.")

# ==============================================================================
# TAB 4: LIMITY K-MEANS
# ==============================================================================
with tab4:
    st.subheader("4. Kdy K-Means selhává a na co si dát pozor?")

    col_lim1, col_lim2 = st.columns(2)

    with col_lim1:
        st.error(
            r"""
            **1. Předpoklad sférického tvaru (Izotropie):**
            K-Means měří Euklidovskou vzdálenost od centroidu. Předpokládá tedy, že shluky tvoří kruhy nebo koule. 
            Pokud mají data tvar protáhlých elipsoidů, půlměsíců nebo soustředných kružnic, K-Means je nesmyslně rozseká!
            *(Řešení: DBSCAN, Spectral Clustering nebo GMM).*
            
            **2. Extrémní citlivost na měřítko:**
            Pokud je jedna proměnná v tisících (např. plat) a druhá v jednotkách (věk), K-Means bude shlukovat pouze podle platu.
            **Vždy před K-Means použijte `StandardScaler`!**
            """
        )

    with col_lim2:
        st.warning(
            r"""
            **3. Rozdílné hustoty a velikosti shluků:**
            Pokud má jeden přirozený shluk 10 000 bodů a druhý jen 50, K-Means má tendenci velký shluk rozdělit a malý pohltit.
            
            **4. Citlivost na odlehlé hodnoty (Outliers):**
            Jelikož centroid je aritmetický průměr, jediný extrémní bod dokáže těžiště posunout o stovky jednotek.
            *(Řešení: K-Medoids / PAM, DBSCAN nebo ořezání outlierů).*
            """
        )
