"""
Den 5: Neřízené učení – Vzorová implementace DBSCAN v Scikit-learn
===================================================================
Syntéza materiálů z DBSCAN_-_sample_implementation.pdf:
1. Generování nekonvexního 2D datasetu se dvěma soustřednými kružnicemi (make_circles).
2. Standardizace StandardScaler jako nutná podmínka pro algoritmy založené na vzdálenostech.
3. Demonstrace selhání K-Means (lineární dělení Voroného hranicemi) vs. úspěch DBSCAN.
4. Vliv hyperparametrů: eps (poloměr sousedství) a min_samples (minimální počet bodů jádra).
5. Objektivní stanovení parametru eps pomocí k-NN distance elbow grafu.
6. Klasifikace a vizualizace typů bodů: Core (jádrové), Border (hraniční) a Noise (šum -1).
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_circles
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import streamlit as st

st.set_page_config(page_title="Implementace: DBSCAN v Scikit-learn", page_icon="🛠️", layout="wide")

st.title("🛠️ Den 5: Praktická implementace DBSCAN v Scikit-learn")
st.caption(
    "Vzorová implementace z přednáškových materiálů kurzu (DBSCAN - an implementation example). "
    "Hustotní shlukování nekonvexních struktur (soustředné kruhy make_circles), srovnání s K-Means, "
    "objektivní určení eps pomocí k-NN grafu a detekce odlehlých hodnot (-1)."
)


@st.cache_data
def load_dbscan_circles_data():
    candidates_json = [
        Path("05_Unsupervised_Learning/data/dbscan_circles_precomputed.json"),
        Path(__file__).resolve().parent.parent / "05_Unsupervised_Learning/data/dbscan_circles_precomputed.json",
    ]
    for p in candidates_json:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    if isinstance(d, dict) and "points" in d and "configs" in d:
                        return d
            except Exception:
                pass

    # Fallback on-the-fly výpočet (< 20 ms)
    raw_data, labels_t = make_circles(n_samples=300, factor=0.5, noise=0.1, random_state=42)
    sc = StandardScaler()
    sc_data = sc.fit_transform(raw_data)

    knn = NearestNeighbors(n_neighbors=5)
    knn.fit(sc_data)
    dists, _ = knn.kneighbors(sc_data)
    sorted_d = np.sort(dists[:, -1]).tolist()

    km_f = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=42).fit(sc_data)

    return {
        "summary": {
            "n_samples": 300,
            "factor": 0.5,
            "noise": 0.1,
            "kmeans_comparison": {
                "k": 2,
                "labels": [int(l) for l in km_f.labels_],
                "centers": km_f.cluster_centers_.tolist()
            }
        },
        "points": {
            "X": sc_data[:, 0].tolist(),
            "Y": sc_data[:, 1].tolist(),
            "true_labels": [int(l) for l in labels_t]
        },
        "knn_elbow": {
            "sorted_distances": sorted_d,
            "suggested_eps": 0.30
        },
        "configs": []
    }


precomputed = load_dbscan_circles_data()

pts = precomputed["points"]
df_pts = pd.DataFrame({"X": pts["X"], "Y": pts["Y"], "True_Circle": pts["true_labels"]})
km_info = precomputed["summary"]["kmeans_comparison"]

# Horní KPI karty
c1, c2, c3, c4 = st.columns(4)
c1.metric("Dataset (Přednáška)", f"{precomputed['summary']['n_samples']} bodů", delta="make_circles (factor=0.5)", delta_color="off")
c2.metric("Geometrická struktura", "Soustředné kruhy", delta="Nekonvexní / lineárně neseparabilní", delta_color="normal")
c3.metric("Výsledek K-Means", "Selhání (k=2)", delta="Kruhy rozseknuty napůl", delta_color="inverse")
c4.metric("Výsledek DBSCAN", "2 kruhy + šum", delta="Dokonalá separace prstenců", delta_color="normal")

st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💻 1. Postup dle přednášky (Kód & Data)",
    "🥊 2. Selhání K-Means vs. Triumf DBSCAN",
    "🎛️ 3. Interaktivní laboratoř DBSCAN",
    "📈 4. Objektivní volba ε (k-NN graf)",
    "📍 5. Typologie bodů (Core vs. Border vs. Noise)"
])

# ==============================================================================
# TAB 1: POSTUP DLE PŘEDNÁŠKY
# ==============================================================================
with tab1:
    st.subheader("1. Kód a generování dat přesně dle přednáškového vzoru")
    st.markdown(
        r"""
        Přednáška (*DBSCAN - an implementation example*, str. 9–14) demonstruje hustotní shlukování 
        na datech vygenerovaných funkcí `make_circles`:
        * **300 pozorování** tvořících vnitřní a vnější kruh s poměrem poloměrů `factor=0.5` a šumem `noise=0.1`.
        * Data jsou povinně standardizována pomocí `StandardScaler`, protože metriky vzdáleností vyžadují jednotné měřítko.
        """
    )

    col_code, col_plot = st.columns([1.1, 0.9])

    with col_code:
        st.markdown("#### 💻 Spuštěný Python kód (dle slajdů):")
        st.code(
            """from sklearn.datasets import make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# 1. Generování dat se dvěma kruhy
data, labels = make_circles(
    n_samples=300, 
    factor=0.5, 
    noise=0.1, 
    random_state=42
)

# 2. Standardizace (kritický krok pro vzdálenostní algoritmy!)
data_scaled = StandardScaler().fit_transform(data)

# 3. Zobrazení vstupních dat
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], color='green', marker='o', s=30)
plt.title("Our dataset")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# 4. Fitování DBSCAN (Slajd 13)
dbscan = DBSCAN(eps=0.4, min_samples=15)
dbscan.fit(data_scaled)

# 5. Vyhodnocení návěští (body s -1 jsou outliery)
labels = dbscan.labels_
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=labels, marker='o', s=25)
plt.title("Dataset after DBSCAN clustering")
plt.show()""",
            language="python"
        )

    with col_plot:
        st.markdown("#### 🟢 Vstupní neanotovaná data (make_circles):")
        fig_raw = px.scatter(
            df_pts,
            x="X",
            y="Y",
            title="Vstupní neoznačená data (300 bodů, 2 soustředné prstence)",
            color_discrete_sequence=["#16a34a"]
        )
        fig_raw.update_traces(marker=dict(size=8, opacity=0.85, line=dict(width=0.5, color="black")))
        fig_raw.update_layout(margin=dict(l=30, r=30, t=40, b=30))
        st.plotly_chart(fig_raw, width="stretch")
        st.info(
            "💡 Lidské oko okamžitě vidí dva kruhy. "
            "Pro těžišťový algoritmus K-Means je však tento úkol neřešitelný!"
        )

# ==============================================================================
# TAB 2: SROVNÁNÍ K-MEANS VS DBSCAN
# ==============================================================================
with tab2:
    st.subheader("2. Proč K-Means selhává a DBSCAN bezchybně vítězí?")
    st.markdown(
        r"""
        Srovnání chování obou algoritmů na identických datech:
        * **K-Means ($k=2$):** Minimalizuje vzdálenosti k těžištím $\mu_1, \mu_2$. Dělící hranice je lineární nadrovina, která oba kruhy přeřízne.
        * **DBSCAN ($\varepsilon=0.30$, min\_samples$=8$):** Sleduje souvislé husté oblasti bodů. Přirozeně obkrouží vnitřní i vnější prstenec.
        """
    )

    c_comp1, c_comp2 = st.columns(2)

    with c_comp1:
        st.markdown("#### ❌ Selhání K-Means ($k=2$)")
        df_km = df_pts.copy()
        df_km["Cluster"] = [f"Shluk {l + 1}" for l in km_info["labels"]]

        fig_km = px.scatter(
            df_km,
            x="X",
            y="Y",
            color="Cluster",
            title="K-Means (k=2): Lineární hranice rozsekne soustředné kruhy",
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        centers_arr = np.array(km_info["centers"])
        fig_km.add_trace(
            go.Scatter(
                x=centers_arr[:, 0],
                y=centers_arr[:, 1],
                mode="markers+text",
                name="Centroidy K-Means",
                marker=dict(symbol="x", size=18, color="red", line=dict(width=3, color="black")),
                text=["C1", "C2"],
                textposition="top center"
            )
        )
        fig_km.update_layout(margin=dict(l=30, r=30, t=40, b=30))
        st.plotly_chart(fig_km, width="stretch")
        st.error("K-Means předpokládá kulové (izotropní) shluky. U vnořených struktur je principiálně nepoužitelný.")

    with c_comp2:
        st.markdown(r"#### ✅ Triumf DBSCAN ($\varepsilon=0.30$, min_samples$=8$)")
        db_demo = DBSCAN(eps=0.30, min_samples=8)
        db_demo_labels = db_demo.fit_predict(df_pts[["X", "Y"]].values)
        df_db = df_pts.copy()
        df_db["Cluster"] = [f"Kruh {l + 1}" if l != -1 else "⚠️ Šum / Outlier (-1)" for l in db_demo_labels]

        fig_db = px.scatter(
            df_db,
            x="X",
            y="Y",
            color="Cluster",
            title="DBSCAN: Dokonalá separace vnitřního a vnějšího kruhu",
            color_discrete_sequence=px.colors.qualitative.Dark24
        )
        fig_db.update_layout(margin=dict(l=30, r=30, t=40, b=30))
        st.plotly_chart(fig_db, width="stretch")
        st.success("DBSCAN sleduje hustotu v $\\varepsilon$-okolí. Tvar shluku je pro něj irelevantní.")

# ==============================================================================
# TAB 3: INTERAKTIVNÍ LABORATOŘ
# ==============================================================================
with tab3:
    st.subheader("3. Interaktivní laboratoř: Vliv hyperparametrů $\\varepsilon$ a min_samples")
    st.markdown(
        "Vyzkoušejte si, jak citlivý je DBSCAN na změnu parametrů. "
        "Můžete otestovat přesné konfigurace ze slajdů 13 a 15 přednášky nebo nalézt vlastní nastavení."
    )

    c_ctrl1, c_ctrl2 = st.columns([1, 2.2])

    with c_ctrl1:
        st.markdown("##### ⚙️ Nastavení hyperparametrů:")
        preset = st.selectbox(
            "Přednastavené konfigurace:",
            [
                "Vlastní nastavení",
                "Slajd 13: eps=0.40, min_samples=15",
                "Slajd 15: eps=0.25, min_samples=10",
                "Optimální: eps=0.30, min_samples=8"
            ]
        )

        if preset == "Slajd 13: eps=0.40, min_samples=15":
            init_eps, init_min = 0.40, 15
        elif preset == "Slajd 15: eps=0.25, min_samples=10":
            init_eps, init_min = 0.25, 10
        elif preset == "Optimální: eps=0.30, min_samples=8":
            init_eps, init_min = 0.30, 8
        else:
            init_eps, init_min = 0.30, 8

        user_eps = st.slider("Poloměr sousedství (eps):", min_value=0.10, max_value=0.60, value=init_eps, step=0.02)
        user_min_samples = st.slider("Minimální počet bodů (min_samples):", min_value=3, max_value=25, value=init_min, step=1)

        # Živý výpočet DBSCAN
        db_live = DBSCAN(eps=user_eps, min_samples=user_min_samples)
        live_labels = db_live.fit_predict(df_pts[["X", "Y"]].values)
        live_clusters = len(set(live_labels)) - (1 if -1 in live_labels else 0)
        live_noise = int(list(live_labels).count(-1))

        st.metric("Počet nalezených shluků", f"{live_clusters} shluků")
        st.metric("Detekováno bodů šumu (-1)", f"{live_noise} bodů ({live_noise / len(df_pts) * 100:.1f} %)")

    with c_ctrl2:
        df_live = df_pts.copy()
        df_live["Cluster"] = [f"Shluk {l + 1}" if l != -1 else "⚠️ Šum / Outlier (-1)" for l in live_labels]

        fig_live = px.scatter(
            df_live,
            x="X",
            y="Y",
            color="Cluster",
            title=rf"DBSCAN: $\varepsilon = {user_eps:.2f}$, min\_samples = {user_min_samples} $\to$ {live_clusters} shluků, {live_noise} šum",
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        fig_live.update_traces(marker=dict(size=8, opacity=0.85))
        fig_live.update_layout(margin=dict(l=30, r=30, t=40, b=30))
        st.plotly_chart(fig_live, width="stretch")

# ==============================================================================
# TAB 4: OBJEKTIVNÍ VOLBA EPS (k-NN GRAF)
# ==============================================================================
with tab4:
    st.subheader("4. Objektivní stanovení $\\varepsilon$ pomocí k-NN Distance Elbow grafu")
    st.markdown(
        r"""
        Slajd 16 přednášky uvádí: *„To optimize hyperparameters, we don't necessarily have to rely on trial and error alone. We can get support from tools such as the nearest neighbor algorithm.“*
        
        **Postup k-NN distance elbow metody:**
        1. Pro každý bod spočítáme vzdálenost k jeho $k$-tému nejbližšímu sousedovi (kde $k = \text{min\_samples}$, např. 5).
        2. Vzdálenosti seřadíme vzestupně od nejmenší po největší.
        3. Vykreslíme křivku. Bod, kde křivka začíná **prudce stoupat (loket / elbow)**, odpovídá přechodu z hustého jádra do řídkého prostoru.
        """
    )

    knn_data = precomputed["knn_elbow"]
    sorted_dists = knn_data["sorted_distances"]

    fig_knn = go.Figure()
    fig_knn.add_trace(
        go.Scatter(
            x=list(range(len(sorted_dists))),
            y=sorted_dists,
            mode="lines",
            line=dict(color="#2563eb", width=3),
            name="Seřazené k-NN vzdálenosti (k=5)"
        )
    )
    fig_knn.add_trace(
        go.Scatter(
            x=[260],
            y=[0.30],
            mode="markers+text",
            marker=dict(size=14, color="#dc2626", symbol="star"),
            text=["Bod zlomu (Eps ~ 0.30)"],
            textposition="top left",
            name="Doporučený Eps"
        )
    )
    fig_knn.add_hline(y=0.30, line_dash="dash", line_color="#dc2626", annotation_text="Eps = 0.30")
    fig_knn.update_layout(
        title="k-NN Distance Graph (5. nejbližší soused) pro nalezení optimálního Epsilonu",
        xaxis_title="Body seřazené dle vzdálenosti (0 až 300)",
        yaxis_title="Vzdálenost k 5. nejbližšímu sousedovi",
        margin=dict(l=30, r=30, t=40, b=30)
    )
    st.plotly_chart(fig_knn, width="stretch")
    st.caption("Před bodem zlomu leží body uvnitř hustých shluků. Za bodem zlomu vzdálenost prudce roste – jedná se o body v řídkém meziprostoru a šum.")

# ==============================================================================
# TAB 5: TYPOLOGIE BODŮ
# ==============================================================================
with tab5:
    st.subheader("5. Detailní typologie bodů: Core (Jádro) vs. Border (Hranice) vs. Noise (Šum)")
    st.markdown(
        r"""
        Přednáška (str. 6–7) definuje 3 stavy datových bodů:
        * **Jádrový bod (Core point):** Má ve svém $\varepsilon$-okolí alespoň `min_samples` bodů (včetně sebe).
        * **Hraniční bod (Border point):** Má méně než `min_samples` bodů, ale leží v dosahu $\varepsilon$ nějakého Core bodu.
        * **Šum / Outlier (-1):** Není jádrový ani hraniční. Zůstává nezařazen.
        """
    )

    db_class = DBSCAN(eps=0.30, min_samples=8)
    cl_labels = db_class.fit_predict(df_pts[["X", "Y"]].values)
    core_set = set(db_class.core_sample_indices_)

    p_classes = []
    for i, l in enumerate(cl_labels):
        if l == -1:
            p_classes.append("Šum / Outlier (-1)")
        elif i in core_set:
            p_classes.append(f"Core bod (Kruh {l + 1})")
        else:
            p_classes.append(f"Border bod (Kruh {l + 1})")

    df_class = df_pts.copy()
    df_class["Point_Type"] = p_classes

    col_t_plot, col_t_stats = st.columns([2, 1])

    with col_t_plot:
        fig_types = px.scatter(
            df_class,
            x="X",
            y="Y",
            color="Point_Type",
            title="Rozlišení Core, Border a Noise bodů (eps=0.30, min_samples=8)",
            color_discrete_map={
                "Core bod (Kruh 1)": "#2563eb",
                "Core bod (Kruh 2)": "#e11d48",
                "Border bod (Kruh 1)": "#93c5fd",
                "Border bod (Kruh 2)": "#fca5a5",
                "Šum / Outlier (-1)": "#0f172a"
            }
        )
        fig_types.update_traces(marker=dict(size=8, opacity=0.9))
        fig_types.update_layout(margin=dict(l=30, r=30, t=40, b=30))
        st.plotly_chart(fig_types, width="stretch")

    with col_t_stats:
        st.markdown("##### 📊 Zastoupení kategorií v datech:")
        type_counts = df_class["Point_Type"].value_counts().reset_index()
        type_counts.columns = ["Typ bodu", "Počet"]
        type_counts["Podíl"] = (type_counts["Počet"] / len(df_class) * 100).round(1).astype(str) + " %"
        st.dataframe(type_counts, width="stretch", hide_index=True)
