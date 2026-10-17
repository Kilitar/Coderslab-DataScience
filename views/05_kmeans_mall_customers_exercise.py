import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="K-Means: Zákazníci OC – Výsledky zadání", page_icon="🎯", layout="wide")

st.title("🎯 Cvičení: Segmentace zákazníků OC pomocí K-Means – Výsledky zadání")
st.caption(
    "Oficiální řešení cvičení z kurzu: Segmentace zákazníků nákupního centra na datech **mall_customers.csv** "
    "pomocí algoritmu K-Means, standardizace proměnných, určení optimálního k a vizualizace těžišť."
)

@st.cache_data
def load_mall_customers_exercise_data():
    candidates_json = [
        Path("05_Unsupervised_Learning/data/mall_customers_kmeans_precomputed.json"),
        Path(__file__).resolve().parent.parent / "05_Unsupervised_Learning/data/mall_customers_kmeans_precomputed.json",
    ]
    for p in candidates_json:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    if isinstance(d, dict) and "pairs" in d and "age_vs_spending" in d["pairs"]:
                        return d
            except Exception:
                pass

    # Fallback on-the-fly výpočet, pokud JSON chybí nebo je v mezipaměti neplatný
    candidates_csv = [
        Path("data/Downloadable materials for Session 3/mall_customers.csv"),
        Path(__file__).resolve().parent.parent / "data/Downloadable materials for Session 3/mall_customers.csv",
        Path("mall_customers.csv"),
    ]
    df_found = None
    for cp in candidates_csv:
        if cp.exists():
            df_found = pd.read_csv(cp)
            break

    if df_found is None:
        return None

    # Rychlý fallback výpočet (< 25 ms)
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    from sklearn.preprocessing import StandardScaler

    fallback_pairs = {
        "age_vs_spending": {"features": ["age", "spending_score"], "default_k": 4},
        "income_vs_spending": {"features": ["annual_income", "spending_score"], "default_k": 5},
    }
    fallback_res = {
        "summary": {
            "n_samples": len(df_found)
        },
        "pairs": {}
    }

    for p_key, p_cfg in fallback_pairs.items():
        feats = p_cfg["features"]
        scaler = StandardScaler()
        X_s = scaler.fit_transform(df_found[feats])
        inertias = []
        silhouettes = [None]
        davies_bouldin = [None]
        models = {}

        for k in range(1, 11):
            km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
            km.fit(X_s)
            inertias.append(float(km.inertia_))
            if k > 1:
                sil = float(silhouette_score(X_s, km.labels_))
                db = float(davies_bouldin_score(X_s, km.labels_))
                silhouettes.append(sil)
                davies_bouldin.append(db)
                c_orig = scaler.inverse_transform(km.cluster_centers_)
                profs = []
                for cid in range(k):
                    c_mask = km.labels_ == cid
                    profs.append({
                        "cluster_id": cid,
                        "count": int(c_mask.sum()),
                        "percentage": float(c_mask.sum() / len(df_found) * 100),
                        "centers_orig": {f: float(c_orig[cid, idx]) for idx, f in enumerate(feats)},
                        "centers_scaled": {f: float(km.cluster_centers_[cid, idx]) for idx, f in enumerate(feats)}
                    })
                models[str(k)] = {
                    "inertia": float(km.inertia_),
                    "silhouette": sil,
                    "davies_bouldin": db,
                    "labels": [int(l) for l in km.labels_],
                    "centers_scaled": km.cluster_centers_.tolist(),
                    "centers_orig": c_orig.tolist(),
                    "profiles": profs
                }

        fallback_res["pairs"][p_key] = {
            "name": p_key,
            "features": feats,
            "default_k": p_cfg["default_k"],
            "k_range": list(range(1, 11)),
            "inertias": inertias,
            "silhouettes": silhouettes,
            "davies_bouldin": davies_bouldin,
            "models": models
        }

    return fallback_res


@st.cache_data
def load_mall_customers_raw_df():
    candidates = [
        Path("data/Downloadable materials for Session 3/mall_customers.csv"),
        Path(__file__).resolve().parent.parent / "data/Downloadable materials for Session 3/mall_customers.csv",
        Path("mall_customers.csv")
    ]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p)
    return None


data = load_mall_customers_exercise_data()
df_raw = load_mall_customers_raw_df()

if not data or "pairs" not in data or df_raw is None:
    st.error("Data datasetu mall_customers.csv nebyla nalezena.")
    st.stop()

pair_data = data["pairs"]["age_vs_spending"]
k_opt = pair_data["default_k"]
model_opt = pair_data["models"][str(k_opt)]

# Horní KPI karty
c1, c2, c3, c4 = st.columns(4)
c1.metric("Počet zákazníků", f"{data['summary']['n_samples']} klientů", delta="Bez chybějících hodnot", delta_color="off")
c2.metric("Vybrané proměnné", "Věk & Skóre utrácení", delta="Standardizováno (StandardScaler)", delta_color="normal")
c3.metric("Optimální počet shluků", f"k = {k_opt}", delta="Elbow bod zlomu & Silueta", delta_color="normal")
c4.metric("Silhouette skóre (k=4)", f"{model_opt['silhouette']:.4f}", delta="4 kompaktní klientské segmenty", delta_color="normal")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "📋 1. Postup cvičení krok za krokem",
    "📈 2. Diagnostika k (Loketní křivka & Silueta)",
    "🎯 3. Výsledné shluky & Zvýrazněná těžiště",
    "👥 4. Byznysová interpretace (Klientské persony)"
])

# ==============================================================================
# TAB 1: POSTUP CVIČENÍ
# ==============================================================================
with tab1:
    st.subheader("1. Systematické řešení kroků zadání")

    col_code, col_info = st.columns([1.1, 0.9])

    with col_code:
        st.markdown("#### 💻 Spuštěný Python kód (přesně dle zadání):")
        st.code(
            r"""import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# 1. Načtení dat
df = pd.read_csv('mall_customers.csv')

# 2. Výběr dvou spojitých veličin (včetně věku)
features = ['age', 'spending_score']
X = df[features]

# 3. Povinná standardizace proměnných
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Loketní metoda a Silhouette skóre pro určení optimálního k
inertias = []
silhouettes = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    if k > 1:
        silhouettes.append(silhouette_score(X_scaled, km.labels_))

# 5. Fitování modelu pro vybraný optimální počet shluků (k = 4)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, random_state=42)
kmeans.fit(X_scaled)

# 6. Přiřazení atributů do požadovaných proměnných
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Centroidy v původních jednotkách (roky, body skóre)
centroids_orig = scaler.inverse_transform(centroids)

# 7. Bodový graf s obarvenými shluky a zřetelnými těžišti
plt.figure(figsize=(10, 6))
plt.scatter(df['age'], df['spending_score'], c=labels, cmap='tab10', s=60, label='Zákazníci')
plt.scatter(centroids_orig[:, 0], centroids_orig[:, 1], c='red', marker='X', s=250, label='Centroidy')
plt.title(f'Segmentace zákazníků K-Means (k = {optimal_k})')
plt.xlabel('Věk (Age)')
plt.ylabel('Skóre utrácení (Spending Score)')
plt.legend()
plt.show()""",
            language="python"
        )

    with col_info:
        st.markdown("#### ✅ Kontrolní seznam splnění požadavků zadání:")
        st.success(
            """
            * [x] **Načtení datasetu `mall_customers.csv`** do DataFrame
            * [x] **Import potřebných knihoven** (`pandas`, `scikit-learn`, `matplotlib`)
            * [x] **Výběr dvou spojitých veličin včetně věku**: `age` a `spending_score`
            * [x] **Standardizace proměnných** pomocí `StandardScaler`
            * [x] **Loketní metoda (Elbow) a metriky** pro stanovení optimálního $k = 4$
            * [x] **Fitování modelu `KMeans`** pro vybraný počet shluků
            * [x] **Přiřazení atributů** do proměnných `labels` a `centroids`
            * [x] **Scatter plot** s barevným odlišením jednotlivých shluků
            * [x] **Zvýraznění těžišť** červenými zřetelnými značkami vůči bodům
            """
        )

        st.markdown("##### 🔍 Základní statistika datasetu:")
        stat_summary = pd.DataFrame([
            {"Proměnná": "Věk (age)", "Minimum": "18 let", "Průměr": "38.9 let", "Maximum": "70 let", "Směrodatná odchylka": "14.0"},
            {"Proměnná": "Příjem (annual_income)", "Minimum": "15 tis. USD", "Průměr": "60.6 tis. USD", "Maximum": "137 tis. USD", "Směrodatná odchylka": "26.3"},
            {"Proměnná": "Skóre utrácení (spending_score)", "Minimum": "1 bod", "Průměr": "50.2 bodů", "Maximum": "99 bodů", "Směrodatná odchylka": "25.8"}
        ])
        st.dataframe(stat_summary, width="stretch", hide_index=True)

# ==============================================================================
# TAB 2: DIAGNOSTIKA K
# ==============================================================================
with tab2:
    st.subheader("2. Diagnostické křivky pro volbu počtu shluků $k$")
    st.markdown(
        r"""
        V souladu se zadáním jsme aplikovali **loketní metodu (Elbow method)** a metriku **Silhouette Score** 
        na standardizovaných proměnných `age` a `spending_score`.
        """
    )

    c_g1, c_g2 = st.columns(2)

    with c_g1:
        st.markdown("#### 📉 Loketní křivka (Inertia / WCSS)")
        k_list = pair_data["k_range"]
        inertias = pair_data["inertias"]

        fig_elb = go.Figure()
        fig_elb.add_trace(
            go.Scatter(
                x=k_list,
                y=inertias,
                mode="lines+markers",
                name="Inertia (WCSS)",
                line=dict(color="#2563eb", width=3),
                marker=dict(size=8)
            )
        )
        fig_elb.add_trace(
            go.Scatter(
                x=[4],
                y=[inertias[3]],
                mode="markers+text",
                name="Zvolený loket (k=4)",
                marker=dict(size=14, color="#dc2626", symbol="star"),
                text=["Bod zlomu (k=4)"],
                textposition="top right"
            )
        )
        fig_elb.update_layout(
            title="Loketní metoda: Pokles WCSS vs. Počet shluků k",
            xaxis_title="Počet shluků (k)",
            yaxis_title="Inertia (WCSS)",
            margin=dict(l=30, r=30, t=40, b=30)
        )
        st.plotly_chart(fig_elb, width="stretch")

    with c_g2:
        st.markdown("#### 🎯 Silhouette Score vs. k")
        sil_vals = pair_data["silhouettes"][1:]
        k_sil = k_list[1:]

        fig_sil = go.Figure()
        fig_sil.add_trace(
            go.Bar(
                x=k_sil,
                y=sil_vals,
                marker_color=["#10b981" if k == 4 else "#94a3b8" for k in k_sil],
                text=[f"{s:.3f}" for s in sil_vals],
                textposition="auto"
            )
        )
        fig_sil.update_layout(
            title="Průměrné Silhouette skóre (Zelená = k=4)",
            xaxis_title="Počet shluků (k)",
            yaxis_title="Silhouette skóre (-1 až +1)",
            yaxis_range=[0, 0.6],
            margin=dict(l=30, r=30, t=40, b=30)
        )
        st.plotly_chart(fig_sil, width="stretch")

    st.markdown("#### 📊 Přehled metrik pro jednotlivá $k$:")
    tab_rows = []
    for idx, k in enumerate(k_list):
        i_val = inertias[idx]
        drop = ((inertias[idx - 1] - i_val) / inertias[idx - 1] * 100) if idx > 0 else 0.0
        s_val = pair_data["silhouettes"][idx]
        db_val = pair_data["davies_bouldin"][idx]
        tab_rows.append({
            "Počet shluků (k)": k,
            "Inertia (WCSS)": f"{i_val:,.1f}",
            "Pokles WCSS": f"-{drop:.1f} %" if idx > 0 else "-",
            "Silhouette Score": f"{s_val:.4f}" if s_val is not None else "N/A",
            "Davies-Bouldin Index": f"{db_val:.3f}" if db_val is not None else "N/A",
            "Hodnocení": "🌟 Zvolené optimum (vyvážené segmenty)" if k == 4 else ("Příliš hrubé" if k < 4 else "Přemíra fragmentace")
        })
    st.dataframe(pd.DataFrame(tab_rows), width="stretch", hide_index=True)

# ==============================================================================
# TAB 3: VÝSLEDNÉ SHLUKY & CENTROIDY
# ==============================================================================
with tab3:
    st.subheader("3. Interaktivní vizualizace shluků a těžišť (Scatter Plot)")
    st.markdown(
        "Graf níže zobrazuje zákazníky rozdělené do 4 barevných shluků s výrazně zvýrazněnými centroidy."
    )

    c_opt1, c_opt2 = st.columns([1, 2.2])

    with c_opt1:
        st.markdown("##### ⚙️ Možnosti zobrazení:")
        scale_mode = st.radio(
            "Jednotky na osách:",
            ["Původní fyzické jednotky (roky a body skóre)", "Standardizované jednotky (z-skóre)"]
        )

        st.markdown("##### 📍 Souřadnice těžišť (Centroidů):")
        profiles = model_opt["profiles"]
        center_summary = []
        for p in profiles:
            cid = p["cluster_id"]
            if "Původní" in scale_mode:
                center_summary.append({
                    "Shluk": f"Shluk {cid + 1}",
                    "Počet klientů": f"{p['count']} ({p['percentage']:.1f} %)",
                    "Věk těžiště": f"{p['centers_orig']['age']:.1f} let",
                    "Skóre utrácení": f"{p['centers_orig']['spending_score']:.1f} b."
                })
            else:
                center_summary.append({
                    "Shluk": f"Shluk {cid + 1}",
                    "Počet klientů": f"{p['count']} ({p['percentage']:.1f} %)",
                    "Věk (z-skóre)": f"{p['centers_scaled']['age']:.2f}",
                    "Skóre (z-skóre)": f"{p['centers_scaled']['spending_score']:.2f}"
                })
        st.dataframe(pd.DataFrame(center_summary), width="stretch", hide_index=True)

    with c_opt2:
        labels_list = model_opt["labels"]
        df_display = df_raw.copy()
        df_display["Cluster"] = [f"Shluk {l + 1}" for l in labels_list]

        if "Původní" in scale_mode:
            fig_scat = px.scatter(
                df_display,
                x="age",
                y="spending_score",
                color="Cluster",
                hover_data=["customer_id", "gender", "annual_income"],
                title="Segmentace zákazníků: Věk vs. Skóre utrácení (k = 4)",
                labels={"age": "Věk zákazníka (roky)", "spending_score": "Skóre utrácení (1–100)"},
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            # Centroidy v původních jednotkách
            centers_arr = np.array(model_opt["centers_orig"])
            fig_scat.add_trace(
                go.Scatter(
                    x=centers_arr[:, 0],
                    y=centers_arr[:, 1],
                    mode="markers+text",
                    name="Centroidy (Těžiště)",
                    marker=dict(symbol="x", size=18, color="red", line=dict(width=3, color="black")),
                    text=[f"C{i+1}" for i in range(k_opt)],
                    textposition="top center"
                )
            )
        else:
            # Standardizované souřadnice
            scaler_temp = StandardScaler()
            X_temp = scaler_temp.fit_transform(df_display[["age", "spending_score"]])
            df_display["age_z"] = X_temp[:, 0]
            df_display["spending_z"] = X_temp[:, 1]

            fig_scat = px.scatter(
                df_display,
                x="age_z",
                y="spending_z",
                color="Cluster",
                hover_data=["customer_id", "gender", "annual_income"],
                title="Segmentace zákazníků ve standardizovaném prostoru z-skóre (k = 4)",
                labels={"age_z": "Věk (standardizovaný)", "spending_z": "Skóre utrácení (standardizované)"},
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            centers_arr = np.array(model_opt["centers_scaled"])
            fig_scat.add_trace(
                go.Scatter(
                    x=centers_arr[:, 0],
                    y=centers_arr[:, 1],
                    mode="markers+text",
                    name="Centroidy (Těžiště)",
                    marker=dict(symbol="x", size=18, color="red", line=dict(width=3, color="black")),
                    text=[f"C{i+1}" for i in range(k_opt)],
                    textposition="top center"
                )
            )

        fig_scat.update_traces(marker=dict(size=9, opacity=0.85), selector=dict(mode="markers"))
        fig_scat.update_layout(margin=dict(l=30, r=30, t=40, b=30))
        st.plotly_chart(fig_scat, width="stretch")

# ==============================================================================
# TAB 4: BYZNYSOVÁ INTERPRETACE
# ==============================================================================
with tab4:
    st.subheader("4. Byznysové klientské persony a doporučení pro marketing OC")
    st.markdown(
        "K-Means rozdělil zákaznickou bázi do 4 jasně profilovaných segmentů:"
    )

    col_p1, col_p2 = st.columns(2)

    with col_p1:
        st.info(
            r"""
            ##### 🛍️ Persona 1: Mladí prémioví nadšenci (Young High Spenders)
            * **Průměrný věk:** ~25–30 let
            * **Skóre utrácení:** 75–90 bodů (extrémně vysoké)
            * **Charakteristika:** Mladá generace s vysokou ochotou utrácet za zážitky, módu, prémiovou elektroniku a gastro.
            * **Doporučená strategie:** Influencer marketing, trendy značky, sociální sítě (Instagram, TikTok), limitované edice.
            """
        )
        st.warning(
            r"""
            ##### 🎒 Persona 2: Mladí racionální zákazníci (Young Low Spenders)
            * **Průměrný věk:** ~25–30 let
            * **Skóre utrácení:** 20–40 bodů (nízké)
            * **Charakteristika:** Studenti a lidé na začátku kariéry, kteří nákupní centrum navštěvují, ale utrácejí obezřetně.
            * **Doporučená strategie:** Studentské slevy, sezónní výprodeje, cenově dostupné fast-fashion a cenově výhodné gastro nabídky.
            """
        )

    with col_p2:
        st.success(
            r"""
            ##### 🏆 Persona 3: Zralí loajální klienti (Older Moderate Spenders)
            * **Průměrný věk:** ~50–55 let
            * **Skóre utrácení:** 45–60 bodů (stabilní průměr)
            * **Charakteristika:** Nejpočetnější a nejstabilnější segment zákazníků v produktivním věku. Pravidelné nákupy rodinného charakteru.
            * **Doporučená strategie:** Věrnostní kluby, prémiový komfort (parkování, salonky), kvalita a spolehlivost služeb.
            """
        )
        st.error(
            r"""
            ##### 👓 Persona 4: Konzervativní senioři (Older Low Spenders)
            * **Průměrný věk:** ~55–65 let
            * **Skóre utrácení:** 10–30 bodů (velmi nízké)
            * **Charakteristika:** Zákazníci vyššího věku, kteří nakupují výhradně plánované základní potřeby (lékárny, potraviny).
            * **Doporučená strategie:** Seniorské slevové dny, přehledné značení v centru, dostupnost lékárenských a zdravotnických služeb.
            """
        )
