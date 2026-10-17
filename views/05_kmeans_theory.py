import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_circles, make_moons
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import streamlit as st

st.set_page_config(page_title="Teorie: Shlukování & K-Means", page_icon="📖", layout="wide")

st.title("📖 Den 5: Neřízené učení – Shluková analýza (Clustering), K-Means & DBSCAN")
st.caption(
    "Ucelený teoretický a praktický rozbor neřízeného shlukování na základě kurzových materiálů (Resources/Day 5 PDF). "
    "Taxonomie algoritmů, K-Means a minimalizace WCSS, K-Means++, hustotní DBSCAN, diagnostika a moderní trendy 2026."
)

# Horní KPI karty
c1, c2, c3, c4 = st.columns(4)
c1.metric("1. Paradigma", "Neřízené učení", delta="Bez cílové proměnné (y)", delta_color="off")
c2.metric("2. Těžišťové metody", "K-Means (Lloyd / K-Means++)", delta="Minimalizace WCSS (Inertia)", delta_color="normal")
c3.metric("3. Hustotní metody", "DBSCAN", delta="Nekonvexní tvary & filtrace šumu", delta_color="normal")
c4.metric("4. Diagnostika k", "Elbow & Silhouette", delta="Objektivní volba počtu shluků", delta_color="normal")

st.markdown("---")


@st.cache_data
def load_blobs_data():
    return pd.read_csv("data/Downloadable materials for Session 3/blobs_dataset.csv")


@st.cache_data
def compute_blobs_metrics(df_points):
    inertias = []
    silhouettes = []
    k_range = list(range(2, 11))
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        km.fit(df_points)
        inertias.append(float(km.inertia_))
        silhouettes.append(float(silhouette_score(df_points, km.labels_)))
    return k_range, inertias, silhouettes


@st.cache_data
def get_synthetic_datasets(n_samples=400, noise=0.06, random_state=42):
    X_moons, _ = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    X_circles, _ = make_circles(n_samples=n_samples, factor=0.5, noise=noise, random_state=random_state)
    return pd.DataFrame(X_moons, columns=["feature_1", "feature_2"]), pd.DataFrame(
        X_circles, columns=["feature_1", "feature_2"]
    )


df_blobs = load_blobs_data()
k_range, inertias, silhouettes = compute_blobs_metrics(df_blobs)
df_moons, df_circles = get_synthetic_datasets()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📚 1. Úvod & Taxonomie",
    "📐 2. K-Means: Matematika & Lloydův algoritmus",
    "🧪 3. Interaktivní laboratoř (K-Means vs. DBSCAN)",
    "📈 4. Volba k: Loketní metoda & Silueta",
    "🔬 5. DBSCAN: Hustotní shlukování & Šum",
    "⚖️ 6. Srovnání, Guardrails & Moderní trendy 2026"
])

# ==============================================================================
# TAB 1: ÚVOD & TAXONOMIE
# ==============================================================================
with tab1:
    st.subheader("1. Co je shlukování (Clustering) a jak se dělí?")
    st.markdown(
        r"""
        Shlukování je disciplína **neřízeného strojového učení (Unsupervised Learning)**. 
        Na rozdíl od regrese či klasifikace nemáme k dispozici žádné referenční návěští $y$ (tzv. ground truth). 
        Cílem je odhalit skryté geometrické či pravděpodobnostní struktury v datech $X$ a rozdělit objekty do přirozených skupin.
        """
    )

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown(
            r"""
            ##### 🎯 Dva fundamentální principy dobrého shlukování:
            1. **Vysoká vnitroshluková kompaktnost (High Intra-cluster Similarity):**  
               Objekty uvnitř stejného shluku si jsou maximálně podobné (malá vzájemná vzdálenost).
            2. **Vysoká mezishluková separabilita (High Inter-cluster Dissimilarity):**  
               Objekty z různých shluků jsou si maximálně odlišné (velká vzájemná vzdálenost).
            """
        )
        st.info(
            r"""
            **Rozlišení typů shlukování:**
            * **Deterministické (Hard / Crisp) shlukování:** Každý bod patří **striktně do jednoho** shluku (např. K-Means, DBSCAN).
            * **Fuzzy (Soft / Pravděpodobnostní) shlukování:** Bod může náležet do více shluků s určitou pravděpodobností (např. Gaussian Mixture Models, Fuzzy C-Means).
            """
        )

    with col_t2:
        st.markdown("##### 🌍 6 klíčových praktických domén (dle prezentace kurzu):")
        st.markdown(
            """
            * 🛒 **Analýza trhu (Market Analysis):** Segmentace zákazníků podle historie nákupů a preferencí pro cílený marketing.
            * 🏥 **Medicínské obrazy (Medical Images):** Detekce homogenních tkání a patologických lézí na snímcích RTG a MRI.
            * 🧬 **Genetické sekvence (Bioinformatics):** Seskupování příbuzných genů s podobnou expresí a objevování nových biologických druhů.
            * 🌐 **Sociální sítě (Social Media):** Detekce zájmových komunit, dynamika šíření informací a identifikace influencerů.
            * 📝 **Textová analýza (Text Mining):** Shlukování dokumentů, článků a recenzí do tematických okruhů.
            * ⚽ **Analýza hráčů (Player Analysis):** Třídění sportovců dle statistik pro sestavování optimálních tréninkových skupin a týmové synergie.
            """
        )

    st.markdown("---")
    st.markdown("#### 🏛️ 4 architektonické rodiny shlukovacích algoritmů:")

    tax_df = pd.DataFrame([
        {
            "Rodina algoritmů": "Centroid-based (Těžišťové)",
            "Kritérium podobnosti": "Vzdálenost bodů od těžiště (centroidu)",
            "Typičtí zástupci": "K-Means, K-Medoids (PAM), MiniBatchKMeans",
            "Tvary shluků": "Výhradně kulové / izotropní",
            "Citlivost na outliery": "Vysoká (outlier vychýlí průměr)"
        },
        {
            "Rodina algoritmů": "Distribution-based (Distribuční)",
            "Kritérium podobnosti": "Pravděpodobnost z vícerozměrného rozdělení",
            "Typičtí zástupci": "Gaussian Mixture Models (GMM / EM)",
            "Tvary shluků": "Elipsoidní dle kovarianční matice",
            "Citlivost na outliery": "Střední (modeluje pravděpodobnost)"
        },
        {
            "Rodina algoritmů": "Density-based (Hustotní)",
            "Kritérium podobnosti": "Hustota bodů v lokálním sousedství",
            "Typičtí zástupci": "DBSCAN, HDBSCAN, OPTICS",
            "Tvary shluků": "Libovolné nekonvexní tvary",
            "Citlivost na outliery": "Nízká (explicitně filtruje šum jako -1)"
        },
        {
            "Rodina algoritmů": "Hierarchical (Hierarchické)",
            "Kritérium podobnosti": "Párové vzdálenosti v dendrogramu",
            "Typičtí zástupci": "Aglomerativní (Ward, Complete), Divizivní",
            "Tvary shluků": "Závisí na vazebném kritériu (linkage)",
            "Citlivost na outliery": "Závisí na zvoleném linkage"
        }
    ])
    st.dataframe(tax_df, width="stretch", hide_index=True)

# ==============================================================================
# TAB 2: MATEMATICKÝ PRINCIP K-MEANS
# ==============================================================================
with tab2:
    st.subheader("2. Matematický princip algoritmu K-Means (Lloydův algoritmus)")
    st.markdown(
        r"""
        Algoritmus K-Means rozděluje $n$ pozorování do $k$ disjunktních shluků $S = \{S_1, S_2, \dots, S_k\}$ 
        takovým způsobem, aby minimalizoval celkový součet čtverců vzdáleností bodů od jejich příslušných těžišť 
        (**WCSS – Within-Cluster Sum of Squares**, v Scikit-learn označovaný jako **Inertia**):
        
        $$\mathcal{J}(S, \mu) = \sum_{j=1}^{k} \sum_{x_i \in S_j} \|x_i - \mu_j\|^2$$
        
        Kde $\mu_j$ je centroid (aritmetický průměr) všech bodů náležejících do shluku $S_j$.
        """
    )

    col_em1, col_em2 = st.columns(2)
    with col_em1:
        st.markdown("#### 🔄 Iterační proces Expectation-Maximization (EM):")
        st.markdown(
            r"""
            1. **Inicializace:**  
               Vybere se požadovaný počet shluků $k$ a zvolí se $k$ počátečních těžišť $\mu_1, \dots, \mu_k$.
            2. **Krok přiřazení (Assignment step / E-step):**  
               Každý bod $x_i$ je přiřazen k nejbližšímu centroidu dle Euklidovské vzdálenosti:
               $$S_j^{(t)} = \left\{ x_i : \|x_i - \mu_j^{(t)}\|^2 \le \|x_i - \mu_{l}^{(t)}\|^2 \quad \forall l = 1, \dots, k \right\}$$
            3. **Krok aktualizace (Update step / M-step):**  
               Pro každý shluk se přepočítá nová poloha těžiště jako průměr přiřazených bodů:
               $$\mu_j^{(t+1)} = \frac{1}{|S_j^{(t)}|} \sum_{x_i \in S_j^{(t)}} x_i$$
            4. **Konvergence:**  
               Kroky 2 a 3 se cyklicky opakují, dokud se poloha centroidů nemění ($\Delta \mu < \varepsilon$) nebo není dosažen limit `max_iter=300`.
            """
        )

    with col_em2:
        st.markdown("#### ⚡ Proč je K-Means++ standardem?")
        st.markdown(
            r"""
            Původní Lloydův algoritmus vybíral $k$ centroidů zcela náhodně (**Random Initialization**). 
            To vedlo k častému uvíznutí v mělkých lokálních minimech, zejména pokud dva počáteční centroidy 
            spadly do stejného přirozeného shluku.
            
            **K-Means++ (Arthur & Vassilvitskii, 2007)** řeší problém pravděpodobnostním rozptylem:
            1. První centroid $\mu_1$ je vybrán náhodně z dat.
            2. Pro každý další bod $x_i$ se spočítá vzdálenost $D(x_i)$ k **nejbližšímu již zvolenému centroidu**.
            3. Další centroid je vybrán s pravděpodobností úměrnou **čtverci vzdálenosti**:
               $$P(x_i) = \frac{D(x_i)^2}{\sum_{l=1}^n D(x_l)^2}$$
            4. Krok se opakuje pro všech $k$ těžišť.
            
            > **Výsledek:** Centroidy jsou od počátku maximálně geometricky separované, konvergence je 2× rychlejší a riziko suboptimálního výsledku dramaticky klesá.
            """
        )

# ==============================================================================
# TAB 3: INTERAKTIVNÍ SIMULÁTOR (K-MEANS VS DBSCAN)
# ==============================================================================
with tab3:
    st.subheader("3. Interaktivní laboratoř: Srovnání K-Means vs. DBSCAN")
    st.markdown(
        "Vyzkoušejte si chování obou algoritmů na různých typech geometrie dat. "
        "Sledujte, kdy K-Means exceluje a na jakých tvarech totálně selhává ve srovnání s hustotním DBSCANem."
    )

    c_sel1, c_sel2, c_sel3 = st.columns(3)
    with c_sel1:
        chosen_dataset = st.selectbox(
            "Vyberte geometrii datasetu:",
            ["Kulovité shluky (blobs_dataset.csv)", "Dva půlměsíce (make_moons)", "Soustředné kruhy (make_circles)"]
        )
    with c_sel2:
        sim_k = st.slider("K-Means: Počet shluků (k):", min_value=2, max_value=8, value=4 if "blobs" in chosen_dataset else 2)
    with c_sel3:
        if "blobs" in chosen_dataset:
            sim_eps = st.slider("DBSCAN: Epsilon (ε):", min_value=0.5, max_value=3.5, value=1.5, step=0.1)
            sim_min_samples = st.slider("DBSCAN: Min samples:", min_value=3, max_value=20, value=8, step=1)
        elif "moons" in chosen_dataset:
            sim_eps = st.slider("DBSCAN: Epsilon (ε):", min_value=0.1, max_value=0.5, value=0.25, step=0.02)
            sim_min_samples = st.slider("DBSCAN: Min samples:", min_value=3, max_value=15, value=5, step=1)
        else:
            sim_eps = st.slider("DBSCAN: Epsilon (ε):", min_value=0.1, max_value=0.4, value=0.18, step=0.02)
            sim_min_samples = st.slider("DBSCAN: Min samples:", min_value=3, max_value=15, value=5, step=1)

    # Příprava dat
    if "blobs" in chosen_dataset:
        df_active = df_blobs.copy()
    elif "moons" in chosen_dataset:
        df_active = df_moons.copy()
    else:
        df_active = df_circles.copy()

    # Modelování K-Means
    km_model = KMeans(n_clusters=sim_k, init="k-means++", n_init=10, random_state=42)
    km_labels = km_model.fit_predict(df_active)
    km_inertia = km_model.inertia_
    km_sil = silhouette_score(df_active, km_labels)

    # Modelování DBSCAN
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_active) if "blobs" not in chosen_dataset else df_active.values
    db_model = DBSCAN(eps=sim_eps, min_samples=sim_min_samples)
    db_labels = db_model.fit_predict(X_scaled)
    n_db_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    n_db_noise = list(db_labels).count(-1)

    col_graph1, col_graph2 = st.columns(2)

    with col_graph1:
        st.markdown(f"#### 🔵 K-Means ($k={sim_k}$)")
        st.caption(f"Inertia (WCSS): **{km_inertia:,.1f}** | Silhouette Score: **{km_sil:.3f}**")
        df_km = df_active.copy()
        df_km["Cluster"] = [f"Shluk {l+1}" for l in km_labels]

        fig_km = px.scatter(
            df_km,
            x="feature_1",
            y="feature_2",
            color="Cluster",
            title=f"K-Means: k = {sim_k}",
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        centers = km_model.cluster_centers_
        fig_km.add_trace(
            go.Scatter(
                x=centers[:, 0],
                y=centers[:, 1],
                mode="markers+text",
                marker=dict(symbol="x", size=14, color="black", line=dict(width=2, color="white")),
                name="Centroidy",
                text=[f"C{i+1}" for i in range(sim_k)],
                textposition="top center"
            )
        )
        fig_km.update_layout(margin=dict(l=30, r=30, t=40, b=30))
        st.plotly_chart(fig_km, width="stretch")

    with col_graph2:
        st.markdown(rf"#### 🟣 DBSCAN ($\epsilon={sim_eps}$, min\_samples$={sim_min_samples}$)")
        st.caption(f"Nalezeno shluků: **{n_db_clusters}** | Detekováno bodů šumu (-1): **{n_db_noise}**")
        df_db = df_active.copy()
        df_db["Cluster"] = [f"Shluk {l+1}" if l != -1 else "⚠️ Šum / Outlier (-1)" for l in db_labels]

        fig_db = px.scatter(
            df_db,
            x="feature_1",
            y="feature_2",
            color="Cluster",
            title=f"DBSCAN: {n_db_clusters} shluků, {n_db_noise} šum",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_db.update_layout(margin=dict(l=30, r=30, t=40, b=30))
        st.plotly_chart(fig_db, width="stretch")

    if "moons" in chosen_dataset or "circles" in chosen_dataset:
        st.warning(
            "💡 **Klíčové pozorování:** Na nekonvexních strukturách (půlměsíce, kruhy) K-Means selhává, "
            "protože prostor rozděluje lineárními hranicemi (Voroného buňky) mezi centroidy. "
            "DBSCAN naopak bez problémů následuje souvislé husté oblasti a správně odhalí přirozenou strukturu!"
        )

# ==============================================================================
# TAB 4: VOLBA K (LOKET & SILUETA)
# ==============================================================================
with tab4:
    st.subheader("4. Diagnostika optimálního počtu shluků $k$")
    st.markdown(
        r"""
        V K-Means je počet shluků $k$ externí hyperparametr. Pro jeho objektivní volbu se kombinují dvě klíčové metody:
        """
    )

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
        st.caption("Bod zlomu (loket) ukazuje hodnotu $k$, od které přidávání dalších centroidů přináší jen marginální pokles chyby.")

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
        st.caption("Vyšší hodnota značí lepší vnitřní soudržnost a výraznější separaci od sousedních shluků.")

    st.markdown(
        r"""
        ##### 📐 Matematický vzorec Siluetového koeficientu:
        Pro datový bod $i$ se spočítá:
        * $a(i)$: Průměrná vzdálenost bodu $i$ od všech ostatních bodů v **jeho vlastním shluku** (soudržnost).
        * $b(i)$: Průměrná vzdálenost bodu $i$ od bodů v **nejbližším sousedním shluku** (separace).
        
        $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}, \quad s(i) \in [-1, +1]$$
        
        * **$s(i) \approx +1$:** Bod je hluboko uvnitř svého shluku a daleko od sousedů (ideální stav).
        * **$s(i) \approx 0$:** Bod leží přímo na rozhodovací hranici mezi dvěma shluky.
        * **$s(i) < 0$:** Bod leží blíže k sousednímu shluku než ke svému vlastnímu (chybné přiřazení).
        """
    )

# ==============================================================================
# TAB 5: DBSCAN POD LUPOU
# ==============================================================================
with tab5:
    st.subheader("5. Algoritmus DBSCAN: Hustotní shlukování & Detekce šumu")
    st.markdown(
        r"""
        **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** 
        je fundamentální hustotní algoritmus, který rozděluje data na základě koncentrace bodů v prostoru.
        """
    )

    c_db_info1, c_db_info2 = st.columns(2)
    with c_db_info1:
        st.markdown(
            r"""
            ##### ⚙️ Dva klíčové parametry:
            1. **$\varepsilon$ (Epsilon):**  
               Maximální poloměr sousedství okolo daného bodu.
            2. **`min_samples`:**  
               Minimální počet bodů v $\varepsilon$-okolí, aby byl bod prohlášen za jádrový (Core point).
            """
        )
        st.markdown(
            r"""
            ##### 📍 3 typy datových bodů v DBSCAN:
            * **Jádrový bod (Core point):** Má ve svém $\varepsilon$-okolí alespoň `min_samples` bodů.
            * **Hraniční bod (Border point):** Má méně než `min_samples` sousedů, ale spadá do $\varepsilon$-okolí nějakého Core bodu.
            * **Šum / Odlehlý bod (Noise point):** Není jádrový ani hraniční. Zůstane nezařazen a označen návěštím `-1`.
            """
        )

    with c_db_info2:
        st.success(
            r"""
            ##### 🌟 Proč je DBSCAN tak ceněný v praxi?
            * **Nemusíte předem hádat počet shluků $k$:**  
              Algoritmus sám přirozeně identifikuje, kolik shluků data obsahují.
            * **Detekce anomálií zdarma:**  
              Odlehlé hodnoty a šum nejsou násilně vtahovány do shluků, ale jsou explicitně označeny jako `-1`.
            * **Robustnost vůči libovolné geometrii:**  
              Zvládá prstence, křivky, spirály a protáhlé shluky.
            """
        )
        st.error(
            r"""
            ##### ⚠️ Kdy naopak DBSCAN naráží?
            * **Různé hustoty v různých částech prostoru:**  
              Protože parametr $\varepsilon$ je globální a fixní, nelze současně dobře zachytit velmi hustý i řidší shluk *(řešením je HDBSCAN)*.
            * **Vysoká dimenze (Kletba dimenzionality):**  
              Ve vícerozměrném prostoru je hustota bodů extrémně nízká a Euklidovská vzdálenost ztrácí kontrast.
            """
        )

# ==============================================================================
# TAB 6: SROVNÁNÍ, GUARDRAILS & MODERNÍ TRENDY 2026
# ==============================================================================
with tab6:
    st.subheader("6. Expertní srovnávací matice, Guardrails & Moderní trendy 2026")

    st.markdown("#### ⚖️ Srovnání K-Means vs. DBSCAN vs. GMM")
    comp_df = pd.DataFrame([
        {
            "Parametr": "Určení počtu shluků",
            "K-Means": "Povinné předem (k)",
            "DBSCAN": "Automatické (dle ε a min_samples)",
            "Gaussian Mixture (GMM)": "Povinné předem (k)"
        },
        {
            "Parametr": "Typ shlukování",
            "K-Means": "Deterministické (Hard)",
            "DBSCAN": "Deterministické (Hard + Šum)",
            "Gaussian Mixture (GMM)": "Fuzzy / Pravděpodobnostní (Soft)"
        },
        {
            "Parametr": "Předpokládaný tvar",
            "K-Means": "Kulový (sférický)",
            "DBSCAN": "Libovolný nekonvexní tvar",
            "Gaussian Mixture (GMM)": "Elipsoidní (kovarianční matice)"
        },
        {
            "Parametr": "Ošetření outlierů",
            "K-Means": "Vychýlí těžiště",
            "DBSCAN": "Označí jako šum (-1)",
            "Gaussian Mixture (GMM)": "Přiřadí nízké pravděpodobnosti"
        },
        {
            "Parametr": "Časová složitost",
            "K-Means": "O(n · k · d · iter) – velmi rychlý",
            "DBSCAN": "O(n log n) s k-d tree, O(n²) bez",
            "Gaussian Mixture (GMM)": "O(n · k · d² · iter) – pomalejší"
        }
    ])
    st.dataframe(comp_df, width="stretch", hide_index=True)

    st.markdown("---")
    st.markdown("#### 🛡️ Závazné metodické Guardrails pro shlukování:")

    g1, g2 = st.columns(2)
    with g1:
        st.error(
            r"""
            **1. Zákaz shlukování bez standardizace (`StandardScaler`):**
            Euklidovská vzdálenost $\sqrt{\sum (x_i - y_i)^2}$ sčítá čtverce rozdílů. 
            Pokud je příjem v desetitisících USD a věk v desítkách let, vzdálenost je z 99.9 % dána příjmem.
            **VŽDY nejprve škálujte data!**
            
            **2. Kletba dimenzionality (Curse of Dimensionality):**
            Při dimenzi $d > 50$ se vzdálenosti mezi všemi body vyrovnávají. 
            Před shlukováním vždy aplikujte redukci dimenzionality (PCA) nebo kosinové metriky.
            """
        )
    with g2:
        st.warning(
            r"""
            **3. K-Means shlukuje i uniformní náhodný šum:**
            K-Means vám vždy vrátí $k$ shluků, i když data žádnou vnitřní strukturu nemají. 
            Vždy ověřujte statistickou významnost a Siluetové skóre.
            
            **4. Vícečetné běhy (`n_init=10`):**
            K-Means může uvíznout v lokálním minimu. V Scikit-learn ponechte `n_init=10` pro výběr nejlepšího běhu.
            """
        )

    st.markdown("---")
    st.markdown("#### 🚀 Moderní ML & AI trendy ve shlukování (Stav k 2026):")
    st.markdown(
        """
        * 🌲 **HDBSCAN (Hierarchical DBSCAN):**  
          Moderní standard v Scikit-learn (`sklearn.cluster.HDBSCAN`). Překonává fixní parametr $\varepsilon$ pomocí hierarchického stromu hustoty a automaticky extrahuje shluky s proměnlivou hustotou.
        * ⚡ **MiniBatchKMeans & GPU akcelerace:**  
          Pro miliony záznamů (clickstream, telemetrie) se používá stochastický gradientní sestup na mini-dávkách nebo knihovna cuML s GPU akcelerací (100× zrychlení).
        * 🌐 **Spherical K-Means pro Transformer Embeddingy:**  
          Textové embeddingy z LLM (OpenAI, BERT) se shlukují na jednotkové sféře pomocí kosinové vzdálenosti namísto Euklidovské normy.
        * 🔍 **XAI & Explainable Clustering:**  
          Použití zástupných stromů (Surrogate Decision Trees) a SHAP hodnot k objasnění byznysových charakteristik jednotlivých zákaznických segmentů.
        * 🤖 **LLM-assisted Cluster Profiling:**  
          Automatické generování byznysových person a marketingových profilů: statistické profily centroidů jsou předány LLM, které jim přiřadí srozumitelná pojmenování a byznysová doporučení.
        """
    )
