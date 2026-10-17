import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import streamlit as st

st.set_page_config(page_title="K-Means: Zákazníci OC – Expertní rozbor", page_icon="🔬", layout="wide")

st.title("🔬 Cvičení: Segmentace zákazníků OC – Expertní analýza & 3D Diagnostika")
st.caption(
    "Hloubkový metodický rozbor nad rámec základního zadání kurzu: Srovnání různých kombinací příznaků, "
    "klasický 5-cluster Kaggle benchmark (Příjem vs. Utrácení), plně interaktivní 3D shlukování "
    "a vliv standardizace na geometrii Voroného buněk."
)

@st.cache_data
def load_mall_customers_critique_data():
    candidates_json = [
        Path("05_Unsupervised_Learning/data/mall_customers_kmeans_precomputed.json"),
        Path(__file__).resolve().parent.parent / "05_Unsupervised_Learning/data/mall_customers_kmeans_precomputed.json",
    ]
    for p in candidates_json:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    if isinstance(d, dict) and "pairs" in d and "income_vs_spending" in d["pairs"]:
                        return d
            except Exception:
                pass

    # Fallback on-the-fly výpočet pro 4 konfigurace
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

    from sklearn.cluster import KMeans
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
    from sklearn.preprocessing import StandardScaler

    fallback_pairs = {
        "age_vs_spending": {"name": "Věk & Skóre utrácení (Zadání kurzu)", "features": ["age", "spending_score"], "default_k": 4},
        "age_vs_income": {"name": "Věk & Roční příjem", "features": ["age", "annual_income"], "default_k": 3},
        "income_vs_spending": {"name": "Roční příjem & Skóre utrácení (Kaggle Benchmark)", "features": ["annual_income", "spending_score"], "default_k": 5},
        "all_3d": {"name": "3D Segmentace: Věk, Příjem & Skóre utrácení", "features": ["age", "annual_income", "spending_score"], "default_k": 6}
    }
    fallback_res = {"summary": {"n_samples": len(df_found)}, "pairs": {}}

    for p_key, p_cfg in fallback_pairs.items():
        feats = p_cfg["features"]
        scaler = StandardScaler()
        X_s = scaler.fit_transform(df_found[feats])
        inertias = []
        silhouettes = [None]
        davies_bouldin = [None]
        calinski = [None]
        models = {}

        for k in range(1, 11):
            km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
            km.fit(X_s)
            inertias.append(float(km.inertia_))
            if k > 1:
                sil = float(silhouette_score(X_s, km.labels_))
                db = float(davies_bouldin_score(X_s, km.labels_))
                ch = float(calinski_harabasz_score(X_s, km.labels_))
                silhouettes.append(sil)
                davies_bouldin.append(db)
                calinski.append(ch)
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
                    "calinski_harabasz": ch,
                    "labels": [int(l) for l in km.labels_],
                    "centers_scaled": km.cluster_centers_.tolist(),
                    "centers_orig": c_orig.tolist(),
                    "profiles": profs
                }

        fallback_res["pairs"][p_key] = {
            "name": p_cfg["name"],
            "features": feats,
            "default_k": p_cfg["default_k"],
            "k_range": list(range(1, 11)),
            "inertias": inertias,
            "silhouettes": silhouettes,
            "davies_bouldin": davies_bouldin,
            "calinski_harabasz": calinski,
            "models": models
        }

    return fallback_res


@st.cache_data
def load_mall_customers_critique_raw_df():
    candidates = [
        Path("data/Downloadable materials for Session 3/mall_customers.csv"),
        Path(__file__).resolve().parent.parent / "data/Downloadable materials for Session 3/mall_customers.csv",
        Path("mall_customers.csv")
    ]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p)
    return None


data = load_mall_customers_critique_data()
df_raw = load_mall_customers_critique_raw_df()

if not data or "pairs" not in data or df_raw is None:
    st.error("Data datasetu mall_customers.csv nebyla nalezena.")
    st.stop()

# Horní KPI karty
c1, c2, c3, c4 = st.columns(4)
c1.metric("Kaggle Benchmark", "5 shluků", delta="Příjem vs. Skóre utrácení", delta_color="normal")
c2.metric("Kaggle Silhouette", f"{data['pairs']['income_vs_spending']['models']['5']['silhouette']:.4f}", delta="Vynikající separace (>0.55)", delta_color="normal")
c3.metric("3D Model (3 příznaky)", "6 shluků", delta="Věk + Příjem + Skóre", delta_color="normal")
c4.metric("Doporučení pro retail", "RFM + Demografie", delta="Moderní standard 2026", delta_color="off")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 1. Srovnání kombinací příznaků (A / B / C)",
    "🏆 2. Klasický Kaggle benchmark (5 segmentů)",
    "🧊 3. Plnohodnotná 3D segmentace (Plotly)",
    "🛡️ 4. Metodické guardrails & Retail praxe 2026"
])

# ==============================================================================
# TAB 1: SROVNÁNÍ KOMBINACÍ PŘÍZNAKŮ
# ==============================================================================
with tab1:
    st.subheader("1. Proč je volba dvojice příznaků zásadní pro byznys?")
    st.markdown(
        r"""
        Školní zadání požadovalo shlukování se zahrnutím věku (`age`). Podívejme se na srovnání, jak se mění kvalita 
        a interpretovatelnost shluků v závislosti na zvolené dvojici proměnných:
        """
    )

    chosen_pair_key = st.selectbox(
        "Vyberte zkoumanou dvojici příznaků:",
        list(data["pairs"].keys()),
        format_func=lambda k: data["pairs"][k]["name"]
    )

    p_info = data["pairs"][chosen_pair_key]
    col_p_left, col_p_right = st.columns([1, 2])

    with col_p_left:
        st.markdown(f"##### 📊 Metriky pro: {p_info['name']}")
        st.write(f"- **Příznaky:** `{', '.join(p_info['features'])}`")

        sel_k = st.slider(
            "Počet shluků (k):",
            min_value=2,
            max_value=8,
            value=p_info["default_k"],
            key=f"slider_{chosen_pair_key}"
        )

        m_active = p_info["models"][str(sel_k)]
        st.metric("Inertia (WCSS)", f"{m_active['inertia']:,.1f}")
        st.metric("Silhouette skóre", f"{m_active['silhouette']:.4f}")
        st.metric("Davies-Bouldin Index", f"{m_active['davies_bouldin']:.3f}")
        st.metric("Calinski-Harabasz Index", f"{m_active['calinski_harabasz']:,.1f}")

    with col_p_right:
        if len(p_info["features"]) == 2:
            feat_x, feat_y = p_info["features"][0], p_info["features"][1]
            df_plot = df_raw.copy()
            df_plot["Cluster"] = [f"Shluk {l + 1}" for l in m_active["labels"]]

            fig_2d = px.scatter(
                df_plot,
                x=feat_x,
                y=feat_y,
                color="Cluster",
                hover_data=["customer_id", "gender", "age", "annual_income", "spending_score"],
                title=f"{p_info['name']} – Výsledek pro k = {sel_k} (Silhouette = {m_active['silhouette']:.3f})",
                color_discrete_sequence=px.colors.qualitative.Plotly
            )

            centers_2d = np.array(m_active["centers_orig"])
            fig_2d.add_trace(
                go.Scatter(
                    x=centers_2d[:, 0],
                    y=centers_2d[:, 1],
                    mode="markers+text",
                    name="Centroidy",
                    marker=dict(symbol="x", size=18, color="red", line=dict(width=3, color="black")),
                    text=[f"C{i+1}" for i in range(sel_k)],
                    textposition="top center"
                )
            )
            fig_2d.update_layout(margin=dict(l=30, r=30, t=40, b=30))
            st.plotly_chart(fig_2d, width="stretch")
        else:
            st.info("Pro 3D vizualizaci všech 3 příznaků přejděte na záložku '3. Plnohodnotná 3D segmentace'.")

# ==============================================================================
# TAB 2: KLASICKÝ KAGGLE BENCHMARK
# ==============================================================================
with tab2:
    st.subheader("2. Klasický retail benchmark: Roční příjem vs. Skóre utrácení (5 shluků)")
    st.markdown(
        r"""
        V celosvětové data science komunitě je dataset `mall_customers.csv` nejznámější díky segmentaci 
        na proměnných **Roční příjem (`annual_income`)** a **Skóre utrácení (`spending_score`)**.
        Tato kombinace dosahuje nejvyššího Siluetového skóre (**$s = 0.5547$**) a vytváří **5 legendárních retailových person**:
        """
    )

    kagg_model = data["pairs"]["income_vs_spending"]["models"]["5"]
    df_kagg = df_raw.copy()

    kaggle_personas = {
        0: "🛍️ Rozhazovači (Careless / High Spend, Low Income)",
        1: "🎯 Cílová skupina / VIP (Target / High Spend, High Income)",
        2: "⚖️ Standardní střed (Standard / Moderate Spend, Moderate Income)",
        3: "🔒 Opatrní / Šetřiví (Careful / Low Spend, High Income)",
        4: "🪙 Nízkopříjmoví skromní (Sensible / Low Spend, Low Income)"
    }

    df_kagg["Cluster"] = [f"Shluk {l + 1}" for l in kagg_model["labels"]]

    col_kagg_plot, col_kagg_desc = st.columns([1.6, 1.4])

    with col_kagg_plot:
        fig_kagg = px.scatter(
            df_kagg,
            x="annual_income",
            y="spending_score",
            color="Cluster",
            hover_data=["customer_id", "gender", "age"],
            title="Kaggle Gold Standard: Příjem vs. Skóre utrácení (k = 5, Sil = 0.555)",
            labels={"annual_income": "Roční příjem (tis. USD)", "spending_score": "Skóre utrácení (1–100)"},
            color_discrete_sequence=px.colors.qualitative.Dark24
        )

        c_kagg_arr = np.array(kagg_model["centers_orig"])
        fig_kagg.add_trace(
            go.Scatter(
                x=c_kagg_arr[:, 0],
                y=c_kagg_arr[:, 1],
                mode="markers+text",
                name="Těžiště",
                marker=dict(symbol="x", size=18, color="red", line=dict(width=3, color="black")),
                text=[f"C{i+1}" for i in range(5)],
                textposition="top center"
            )
        )
        fig_kagg.update_layout(margin=dict(l=30, r=30, t=40, b=30))
        st.plotly_chart(fig_kagg, width="stretch")

    with col_kagg_desc:
        st.markdown("##### 💎 5 strategických marketingových segmentů:")
        st.markdown(
            """
            1. **Cílová skupina (VIP - High Income, High Spend):**  
               Zlatý důl nákupního centra. Vysoké příjmy a vysoká ochota utrácet. Vyžadují prémiový servis a luxusní butiky.
            2. **Opatrní (Careful - High Income, Low Spend):**  
               Vysoké příjmy, ale nízká spotřeba. Racionální klienti, kteří potřebují přesvědčit kvalitou a exkluzivitou, nikoliv slevami.
            3. **Rozhazovači (Careless - Low Income, High Spend):**  
               Mladší zákazníci žijící nad poměry. Vysoká afinita k módním výstřelkům, kávárnám a impulzivním nákupům.
            4. **Skromní (Sensible - Low Income, Low Spend):**  
               Citliví na cenu, nakupují pouze nezbytnosti v diskontních prodejnách.
            5. **Standardní střed (Standard - Average Income, Average Spend):**  
               Největší masa běžných rodinných nákupů.
            """
        )

# ==============================================================================
# TAB 3: PLNA 3D SEGMENTACE
# ==============================================================================
with tab3:
    st.subheader("3. 3D Interaktivní prostor: Věk + Příjem + Skóre utrácení")
    st.markdown(
        r"""
        V reálném marketingu nemusíme dělat kompromis mezi věkem a příjmem. 
        K-Means v 3D prostoru rozděluje data podle všech 3 spojitých veličin současně.
        Rotujte graf myší pro zkoumání prostorové separace shluků:
        """
    )

    k_3d = 6
    m_3d = data["pairs"]["all_3d"]["models"][str(k_3d)]

    df_3d = df_raw.copy()
    df_3d["Cluster"] = [f"Shluk {l + 1}" for l in m_3d["labels"]]

    fig_3d = px.scatter_3d(
        df_3d,
        x="age",
        y="annual_income",
        z="spending_score",
        color="Cluster",
        hover_data=["customer_id", "gender"],
        title=f"3D Segmentace zákazníků OC (k = {k_3d}, Silhouette = {m_3d['silhouette']:.3f})",
        labels={
            "age": "Věk (roky)",
            "annual_income": "Příjem (tis. USD)",
            "spending_score": "Skóre utrácení (1–100)"
        },
        color_discrete_sequence=px.colors.qualitative.Plotly
    )

    c_3d_arr = np.array(m_3d["centers_orig"])
    fig_3d.add_trace(
        go.Scatter3d(
            x=c_3d_arr[:, 0],
            y=c_3d_arr[:, 1],
            z=c_3d_arr[:, 2],
            mode="markers+text",
            name="Centroidy",
            marker=dict(symbol="diamond", size=10, color="red"),
            text=[f"C{i+1}" for i in range(k_3d)],
            textposition="top center"
        )
    )

    fig_3d.update_layout(
        scene=dict(
            xaxis_title="Věk (roky)",
            yaxis_title="Roční příjem (tis. USD)",
            zaxis_title="Skóre utrácení"
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        height=650
    )
    st.plotly_chart(fig_3d, width="stretch")

# ==============================================================================
# TAB 4: METODICKÉ GUARDRAILS & RETAIL PRAXE 2026
# ==============================================================================
with tab4:
    st.subheader("4. Metodická úskalí a přechod na moderní retailové standardy (2026)")

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.error(
            r"""
            ##### ⚠️ Důsledky vynechání `StandardScaler`:
            * **Příjem vs. Věk:**  
              Rozptyl ročního příjmu je $\sigma^2 \approx 690$, zatímco rozptyl věku je $\sigma^2 \approx 195$ (3.5× menší!).
            * Bez standardizace by Euklidovská vzdálenost preferovala rozdíly v příjmech a věk by byl de facto potlačen.
            * **Závazné pravidlo:** Při kombinaci veličin s různými fyzikálními jednotkami je `StandardScaler` naprostou nutností!
            """
        )

    with col_g2:
        st.success(
            r"""
            ##### 🚀 Moderní retailová segmentace (Stav k 2026):
            1. **Přechod od statických demografií k RFM:**  
               Moderní e-shopy a OC nehodnotí zákazníka jen podle věku a příjmu, ale podle transakční dynamiky:
               * **R (Recency):** Kdy naposledy nakoupil?
               * **F (Frequency):** Jak často se vrací?
               * **M (Monetary):** Kolik celkem utratil za rok?
            2. **Sekvenční modely a embeddings:**  
               Využití klientských vektorů z nákupních košíků (tzv. Customer2Vec).
            3. **Automatizace marketingových kampaní (LLM):**  
               Centroidy jsou předány jazykovému modelu, který generuje personalizované newslettery pro každý shluk zvlášť.
            """
        )
