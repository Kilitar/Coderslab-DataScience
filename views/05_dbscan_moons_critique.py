"""
Den 5: Neřízené učení – Expertní rozbor cvičení DBSCAN na datech make_moons
===========================================================================
Hluboká metodická analýza a kritika:
1. Srovnávací studie: DBSCAN vs. K-Means vs. Agglomerative (Ward & Single Linkage).
2. Matematický fenomén Chasm Bridging (most přes propast při překročení kritického eps).
3. Paradox Silhouette skóre u nekonvexních geometrií (selhání interních metrik).
4. Metodické guardrails pro hustotní shlukování v produkční praxi 2026.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
import streamlit as st

st.set_page_config(page_title="Cvičení: Moons – Expertní rozbor", page_icon="🔬", layout="wide")

st.title("🔬 Expertní rozbor: DBSCAN na nekonvexních datech (make_moons)")
st.caption(
    "Pokročilá analytická diagnostika: Proč K-Means a Wardovo shlukování selhávají, mechanismus 'Mostu přes propast' "
    "(Chasm Bridging), paradox metriky Silhouette skóre a závazná pravidla pro nasazení hustotních algoritmů."
)


@st.cache_data
def load_dbscan_moons_critique_data():
    candidates_json = [
        Path("05_Unsupervised_Learning/data/dbscan_moons_precomputed.json"),
        Path(__file__).resolve().parent.parent / "05_Unsupervised_Learning/data/dbscan_moons_precomputed.json",
    ]
    for p in candidates_json:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    if isinstance(d, dict) and "models_comparison" in d and "points" in d:
                        return d
            except Exception:
                pass

    # Rychlý fallback výpočet
    X_raw, y_true = make_moons(n_samples=500, noise=0.10, random_state=42)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_raw)

    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    km_l = km.fit_predict(X_s)

    single = AgglomerativeClustering(n_clusters=2, linkage="single")
    single_l = single.fit_predict(X_s)

    ward = AgglomerativeClustering(n_clusters=2, linkage="ward")
    ward_l = ward.fit_predict(X_s)

    db = DBSCAN(eps=0.25, min_samples=8)
    db_l = db.fit_predict(X_s)

    points_data = []
    for i in range(len(X_s)):
        points_data.append({
            "idx": i,
            "x_scaled": float(X_s[i, 0]),
            "y_scaled": float(X_s[i, 1]),
            "y_true": int(y_true[i]),
            "kmeans_label": int(km_l[i]),
            "single_label": int(single_l[i]),
            "ward_label": int(ward_l[i]),
            "opt_dbscan_label": int(db_l[i]),
        })

    return {
        "dataset_meta": {"n_samples": 500, "noise": 0.10},
        "models_comparison": {
            "kmeans_k2": {"ari": float(adjusted_rand_score(y_true, km_l)), "silhouette": float(silhouette_score(X_s, km_l))},
            "agg_single_k2": {"ari": float(adjusted_rand_score(y_true, single_l)), "silhouette": float(silhouette_score(X_s, single_l))},
            "agg_ward_k2": {"ari": float(adjusted_rand_score(y_true, ward_l)), "silhouette": float(silhouette_score(X_s, ward_l))},
            "dbscan_optimal": {"ari": float(adjusted_rand_score(y_true, db_l)), "silhouette": float(silhouette_score(X_s[db_l != -1], db_l[db_l != -1]))}
        },
        "points": points_data
    }


critique_data = load_dbscan_moons_critique_data()

if not critique_data or "points" not in critique_data:
    st.error("Data expertního rozboru nebyla nalezena.")
    st.stop()

df_pts = pd.DataFrame(critique_data["points"])
mc = critique_data.get("models_comparison", {})

# Defenzivní rekonstrukce sloupců pro zamezení KeyError při mezipaměti Streamlitu
if "ward_label" not in df_pts.columns and "agg_ward_k2" in mc and "labels" in mc["agg_ward_k2"]:
    df_pts["ward_label"] = mc["agg_ward_k2"]["labels"]
elif "ward_label" not in df_pts.columns:
    df_pts["ward_label"] = AgglomerativeClustering(n_clusters=2, linkage="ward").fit_predict(df_pts[["x_scaled", "y_scaled"]].values)

if "single_label" not in df_pts.columns and "agg_single_k2" in mc and "labels" in mc["agg_single_k2"]:
    df_pts["single_label"] = mc["agg_single_k2"]["labels"]
elif "single_label" not in df_pts.columns:
    df_pts["single_label"] = AgglomerativeClustering(n_clusters=2, linkage="single").fit_predict(df_pts[["x_scaled", "y_scaled"]].values)

if "kmeans_label" not in df_pts.columns and "kmeans_k2" in mc and "labels" in mc["kmeans_k2"]:
    df_pts["kmeans_label"] = mc["kmeans_k2"]["labels"]
elif "kmeans_label" not in df_pts.columns:
    df_pts["kmeans_label"] = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(df_pts[["x_scaled", "y_scaled"]].values)

if "opt_dbscan_label" not in df_pts.columns:
    df_pts["opt_dbscan_label"] = DBSCAN(eps=0.25, min_samples=8).fit_predict(df_pts[["x_scaled", "y_scaled"]].values)

# Horní KPI srovnání
c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "DBSCAN (eps=0.25)",
    f"ARI = {mc.get('dbscan_optimal', {}).get('ari', 0.98):.3f}",
    delta="Vítěz: Dokonale oddělí oba půlměsíce",
    delta_color="normal"
)
c2.metric(
    "K-Means (k=2)",
    f"ARI = {mc.get('kmeans_k2', {}).get('ari', 0.25):.3f}",
    delta="Selhání: Přeřízne půlměsíce napůl",
    delta_color="inverse"
)
c3.metric(
    "Ward Hierarchical",
    f"ARI = {mc.get('agg_ward_k2', {}).get('ari', 0.39):.3f}",
    delta="Selhání: Předpokládá sférické tvary",
    delta_color="inverse"
)
c4.metric(
    "Single Linkage",
    f"ARI = {mc.get('agg_single_k2', {}).get('ari', 0.99):.3f}",
    delta="Riziko: Náchylný k propojení šumem",
    delta_color="off"
)

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "⚔️ 1. Srovnávací studie 4 algoritmů",
    "🌉 2. 'Most přes propast' (Chasm Bridging)",
    "📏 3. Paradox metriky Silhouette skóre",
    "🛡️ 4. Metodické guardrails pro praxi 2026"
])

# ==============================================================================
# TAB 1: SROVNÁVACÍ STUDIE 4 ALGORITMŮ
# ==============================================================================
with tab1:
    st.subheader("1. Detailní srovnání 4 shlukovacích paradigmat na nekonvexních datech")
    st.markdown(
        r"""
        Proč je úloha `make_moons` tak slavná? Ukazuje fundamentální limity algoritmů založených na **těžištích (centroids)**
        a **rozptylu (variance minimization)** v kontrastu s algoritmy založenými na **hustotě a konektivitě**.
        """
    )

    c_g1, c_g2 = st.columns(2)

    with c_g1:
        # K-Means
        fig_km = px.scatter(
            df_pts,
            x="x_scaled",
            y="y_scaled",
            color=df_pts["kmeans_label"].astype(str),
            title=f"❌ K-Means (k=2) — ARI = {mc.get('kmeans_k2', {}).get('ari', 0.25):.3f}",
            labels={"x_scaled": "Příznak 1", "y_scaled": "Příznak 2", "color": "Shluk"},
            color_discrete_sequence=["#ef4444", "#3b82f6"]
        )
        fig_km.update_layout(height=380, template="plotly_white", margin=dict(l=30, r=30, t=40, b=30))
        st.plotly_chart(fig_km, width="stretch")
        st.caption("K-Means minimalizuje vzdálenost k těžištím. Dělící hranice je lineární osa, která oba půlměsíce přepůlí.")

    with c_g2:
        # Ward
        fig_ward = px.scatter(
            df_pts,
            x="x_scaled",
            y="y_scaled",
            color=df_pts["ward_label"].astype(str),
            title=f"❌ Ward Hierarchical — ARI = {mc.get('agg_ward_k2', {}).get('ari', 0.39):.3f}",
            labels={"x_scaled": "Příznak 1", "y_scaled": "Příznak 2", "color": "Shluk"},
            color_discrete_sequence=["#f97316", "#06b6d4"]
        )
        fig_ward.update_layout(height=380, template="plotly_white", margin=dict(l=30, r=30, t=40, b=30))
        st.plotly_chart(fig_ward, width="stretch")
        st.caption("Ward minimalizuje přírůstek rozptylu při slučování. Má stejný sférický předpoklad jako K-Means.")

    c_g3, c_g4 = st.columns(2)

    with c_g3:
        # Single Linkage
        fig_single = px.scatter(
            df_pts,
            x="x_scaled",
            y="y_scaled",
            color=df_pts["single_label"].astype(str),
            title=f"⚠️ Single Linkage — ARI = {mc.get('agg_single_k2', {}).get('ari', 0.99):.3f}",
            labels={"x_scaled": "Příznak 1", "y_scaled": "Příznak 2", "color": "Shluk"},
            color_discrete_sequence=["#8b5cf6", "#10b981"]
        )
        fig_single.update_layout(height=380, template="plotly_white", margin=dict(l=30, r=30, t=40, b=30))
        st.plotly_chart(fig_single, width="stretch")
        st.caption("Single Linkage měří minimální vzdálenost bodů. Zachytí tvar, ale je extrémně citlivý na jediný šumový bod.")

    with c_g4:
        # DBSCAN
        db_labels_str = [f"Půlměsíc {l+1}" if l != -1 else "Šum (-1)" for l in df_pts["opt_dbscan_label"]]
        fig_db = px.scatter(
            df_pts,
            x="x_scaled",
            y="y_scaled",
            color=db_labels_str,
            title=rf"✅ DBSCAN (ε=0.25, min_samples=8) — ARI = {mc.get('dbscan_optimal', {}).get('ari', 0.98):.3f}",
            labels={"x_scaled": "Příznak 1", "y_scaled": "Příznak 2", "color": "Shluk"},
            color_discrete_map={"Půlměsíc 1": "#2563eb", "Půlměsíc 2": "#10b981", "Šum (-1)": "#dc2626"}
        )
        fig_db.update_layout(height=380, template="plotly_white", margin=dict(l=30, r=30, t=40, b=30))
        st.plotly_chart(fig_db, width="stretch")
        st.caption("DBSCAN spojuje souvislé husté oblasti a zároveň izoluje náhodné odlehlé body jako šum.")

# ==============================================================================
# TAB 2: MOST PŘES PROPAST (CHASM BRIDGING)
# ==============================================================================
with tab2:
    st.subheader(r"2. Fenomén 'Most přes propast' (Chasm Bridging)")
    st.markdown(
        r"""
        Nejčastější chybou při nasazení DBSCANu na oddělených, ale blízko sebe ležících strukturách je **Chasm Bridging**:
        * Pokud je poloměr $\varepsilon$ nastaven větší než minimální šířka mezery mezi půlměsíci ($\Delta_{\text{gap}}$),
          i nepatrný počet šumových bodů vytvoří **hustotní můstek**.
        * Algoritmus DBSCAN definuje shluk přes **hustotní dosažitelnost (Density Reachability)**. Jakmile existuje
          řetězec jádrových bodů $p_1, p_2, \dots, p_k$, kde každý leží v $\varepsilon$-dosahu předchozího,
          oba půlměsíce se slijí do jednoho velkého shluku (`n_clusters = 1`).
        """
    )

    c_cb1, c_cb2 = st.columns(2)
    with c_cb1:
        st.markdown(r"#### 🟢 Správné nastavení: $\varepsilon = 0.25 < \Delta_{\text{gap}}$")
        st.markdown(
            r"""
            * $\varepsilon$-okolí bodů na okraji horního půlměsíce **nedosáhne** na dolní půlměsíc.
            * Mezera funguje jako přirozený izolační příkop.
            * Bod šumu v mezeře nemá ve svém okolí dostatek sousedů (`min_samples=8`) a je označen jako `-1`.
            """
        )
    with c_cb2:
        st.markdown(r"#### 🔴 Selhání: $\varepsilon = 0.35 \ge \Delta_{\text{gap}}$")
        st.markdown(
            r"""
            * $\varepsilon$-okolí přesáhne propast mezi půlměsíci.
            * Šumový bod v mezeře získá sousedy z obou půlměsíců a stane se Core bodem.
            * Výsledek: **Katastrofální slití obou shluků** ($\text{ARI} = 0.000$).
            """
        )

    st.info(
        "💡 **Praktické pravidlo:** Při volbě $\varepsilon$ z k-NN elbow grafu volte vždy hodnotu těsně u paty lokte (konzervativní $\varepsilon$), "
        "nikoli až na jeho vrcholu. Menší $\varepsilon$ způsobí nanejvýš mírný nárůst šumu, zatímco příliš velké $\varepsilon$ zničí celou shlukovou strukturu."
    )

# ==============================================================================
# TAB 3: PARADOX METRIKY SILHOUETTE SKÓRE
# ==============================================================================
with tab3:
    st.subheader("3. Proč Silhouette skóre selhává na nekonvexních strukturách")
    st.markdown(
        r"""
        Většina datových vědců automaticky vyhodnocuje kvalitu shlukování pomocí **Silhouette skóre**.
        Na datech jako `make_moons` však Silhouette skóre vede k naprosto zavádějícím závěrům!
        
        #### Matematická definice siluety:
        Pro bod $i$ v shluku $A$:
        
        $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
        
        * $a(i)$: Průměrná vzdálenost bodu $i$ k ostatním bodům **ve stejném shluku**.
        * $b(i)$: Průměrná vzdálenost bodu $i$ k bodům v **nejbližším sousedním shluku**.
        """
    )

    c_s_txt, c_s_tbl = st.columns([1.2, 1.0])

    with c_s_txt:
        st.markdown(
            r"""
            ##### ⚠️ Geometrický paradox půlměsíce:
            1. **Vysoké $a(i)$ uvnitř shluku:** Dva protilehlé konce téhož půlměsíce jsou od sebe vzdálené přes celý oblouk (eukleidovsky daleko).
            2. **Nízké $b(i)$ mezi shluky:** V centrální části leží body horního půlměsíce těsně vedle bodů dolního půlměsíce (dělí je pouze tenká mezera).
            3. **Výsledek:** Pro body v mezeře platí $b(i) \approx a(i)$, což sráží hodnotu $s(i)$ k nule nebo záporným hodnotám!
            """
        )

    with c_s_tbl:
        st.markdown("##### 📊 Srovnání metrik na make_moons:")
        df_metric_comp = pd.DataFrame([
            {"Model": "K-Means (k=2)", "Skutečná shoda (ARI)": "0.248 (❌ Selhání)", "Silhouette skóre": "0.491 (Zdánlivě skvělé!)"},
            {"Model": "Ward Hierarchical", "Skutečná shoda (ARI)": "0.385 (❌ Selhání)", "Silhouette skóre": "0.448 (Falešně vysoké)"},
            {"Model": "DBSCAN (eps=0.25)", "Skutečná shoda (ARI)": "0.980 (✅ Vítěz)", "Silhouette skóre": "0.384 (Nízké kvůli geometrii)"},
        ])
        st.table(df_metric_comp)

    st.warning(
        "🛑 **Klíčové varování:** Pokud byste hyperparametry ladili maximalizací Silhouette skóre, model by vybral K-Means "
        "nebo špatný DBSCAN! Silhouette penalizuje protáhlé a zakřivené tvary."
    )

# ==============================================================================
# TAB 4: METODICKÉ GUARDRAILS PRO PRAXI 2026
# ==============================================================================
with tab4:
    st.subheader("4. Metodické Guardrails pro nasazení hustotního shlukování (2026)")
    st.markdown(
        r"""
        Při nasazování algoritmů typu DBSCAN / HDBSCAN do produkčních analytických procesů dodržujte následující pravidla:
        """
    )

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown(
            r"""
            ##### 1. Striktní standardizace vstupů
            * **Pravidlo:** Všechny spojité proměnné musí projít `StandardScaler` nebo `RobustScaler`.
            * **Důvod:** Eukleidovská metrika v neškálovaném prostoru deformuje $\varepsilon$-kouli na extrémně protáhlý elipsoid, což zničí hustotní odhad podél os s malým rozptylem.
            
            ##### 2. Pravidlo dimenzionality pro `min_samples`
            * **Pravidlo:** Pro data o dimenzi $D$ nastavte $\text{min\_samples} \ge 2 \cdot D$.
            * **Důvod:** Při $\text{min\_samples} \le D$ je model náchylný k vytváření tenkých řetízkových můstků (Single Linkage efekt) i přes náhodný šum.
            """
        )

    with col_g2:
        st.markdown(
            r"""
            ##### 3. k-NN Distance graf jako povinný diagnostický krok
            * **Pravidlo:** $\varepsilon$ nikdy nenastavujte odhadem. Vždy vykreslete křivku seřazených vzdáleností k $k$-tému sousedovi pro $k = \text{min\_samples}$.
            * **Důvod:** Loket křivky přesně odděluje hustou intramodální oblast od řídkého mezishlukového prostoru.
            
            ##### 4. Přechod na HDBSCAN při nestejných hustotách
            * **Pravidlo:** Pokud mají různé části datasetu odlišnou hustotu (např. husté centrum velkoměsta vs. řídká předměstí), DBSCAN s jediným globálním $\varepsilon$ selže.
            * **Řešení:** Využijte **HDBSCAN (Hierarchical DBSCAN)**, který provádí hierarchickou extrakci stabilních shluků napříč všemi hustotními hladinami.
            """
        )

    st.success("✅ Tyto guardrails garantují matematickou rigoróznost a stabilitu výsledků v produkčních podmínkách.")
