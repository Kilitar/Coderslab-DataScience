import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="PCA Diamanty – Výsledky zadání", page_icon="🎯", layout="wide")

st.title("🎯 Cvičení: Analýza hlavních komponent (PCA) – Diamanty")
st.caption(
    "Oficiální řešení cvičení z kurzu: Redukce dimenzionality datasetu **diamonds_preprocessed.csv** "
    "pomocí metody PCA, vyhodnocení podílu vysvětleného rozptylu a sestavení nového DataFrame `diamonds_pca_df`."
)

DATA_PATH = Path("05_Unsupervised_Learning/data/pca_diamonds_precomputed.json")


@st.cache_data
def load_precomputed():
    if DATA_PATH.exists():
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


data = load_precomputed()

if not data:
    st.error("Předpočítaná data nebyla nalezena. Spusťte prosím generátor precomputed dat.")
    st.stop()

# Horní KPI karty
c1, c2, c3, c4 = st.columns(4)
c1.metric("Původní příznaky (X)", f"{len(data['features'])} proměnných", delta="carat, rozměry x,y,z, kvalita")
c2.metric("Vybrané komponenty", f"{data['chosen_n']} komponenty (PC1–PC4)", delta="Kritérium 80 % – 90 % variance")
c3.metric("Vysvětlený rozptyl (Standardized)", f"{data['chosen_scaled_variance'] * 100:.2f} %", delta="PC1 až PC4 dohromady")
c4.metric("Počet diamantů", f"{data['total_rows']:,} řádků", delta="Cílová proměnná: price (USD)")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "📋 1. Postup cvičení krok za krokem",
    "📈 2. Scree Plot & Kritérium výběru (80 % – 90 %)",
    "💎 3. Výsledný DataFrame diamonds_pca_df",
    "🧭 4. Interaktivní vizualizace komponent"
])

# ==============================================================================
# TAB 1: POSTUP CVIČENÍ
# ==============================================================================
with tab1:
    st.subheader("1. Systematické řešení kroků zadání")

    col_code, col_info = st.columns([1.1, 0.9])

    with col_code:
        st.markdown("#### 💻 Spuštěný Python kód (dle zadání):")
        st.code(
            r'''import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Krok 1 & 3: Načtení dat
df = pd.read_csv('diamonds_preprocessed.csv')

# Krok 4: Oddělení nezávislých proměnných X a závislé y
X = df.drop(columns=['price'])
y = df['price']

# Standardizace před PCA (přednáškový standard)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Krok 5: Výpočet skóre rozptylu ve for cyklu pro 1 až N komponent
variance_scores = {}
for n in range(1, X.shape[1] + 1):
    pca = PCA(n_components=n)
    pca.fit(X_scaled)
    # Součet poměru vysvětleného rozptylu
    variance_scores[n] = float(np.sum(pca.explained_variance_ratio_))

# Krok 6: Výběr počtu komponent (80 % <= score <= 90 %)
selected_n = [n for n, score in variance_scores.items() if 0.80 <= score <= 0.90][0]

# Krok 7: Vytvoření diamonds_pca_df
pca_final = PCA(n_components=selected_n)
X_pca = pca_final.fit_transform(X_scaled)
diamonds_pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(selected_n)])
diamonds_pca_df['price'] = y.values
''',
            language="python"
        )

    with col_info:
        st.markdown("#### 📊 Výsledný slovník skóre rozptylu (`variance_scores`):")
        
        dict_data = []
        for k, v in data["scaled_dict"].items():
            k_int = int(k)
            in_range = 0.80 <= v <= 0.90
            status = "✅ VYBRÁNO (80 % – 90 %)" if in_range else ("❌ Pod 80 %" if v < 0.80 else "Nad 90 %")
            dict_data.append({
                "Počet komponent (k)": f"k = {k_int}",
                "Vysvětlený rozptyl": f"{v * 100:.2f} %",
                "Podmínka zadání": status
            })

        st.dataframe(pd.DataFrame(dict_data), width="stretch", hide_index=True)

        st.info(
            r"""
            **Klíčové zjištění cvičení:**
            * Při $k = 3$ vysvětlují komponenty pouze **73.26 %** rozptylu (pod hranicí 80 %).
            * Při $k = 4$ vysvětlují **82.87 %** rozptylu (přesně v intervalu 80 % až 90 %).
            * Při $k = 5$ již rozptyl stoupá na **92.32 %** (nad hranicí 90 %).
            
            Jediným správným řešením splňujícím zadání je tedy volba **4 hlavních komponent**.
            """
        )

# ==============================================================================
# TAB 2: SCREE PLOT & VÝBĚR
# ==============================================================================
with tab2:
    st.subheader("2. Analýza vysvětleného rozptylu (Scree Plot & Kritérium)")

    col_scree1, col_scree2 = st.columns([1.2, 0.8])

    with col_scree1:
        comp_labels = [f"PC{i+1}" for i in range(len(data["features"]))]
        indiv_pct = [v * 100 for v in data["scaled_indiv"]]
        cum_pct = [v * 100 for v in data["scaled_cum"]]

        fig_scree = go.Figure()

        # Sloupcový graf jednotlivých komponent
        fig_scree.add_trace(
            go.Bar(
                x=comp_labels,
                y=indiv_pct,
                name="Jednotlivá komponenta",
                marker_color="#3b82f6",
                text=[f"{v:.1f} %" for v in indiv_pct],
                textposition="auto"
            )
        )

        # Čárový graf kumulativního rozptylu
        fig_scree.add_trace(
            go.Scatter(
                x=comp_labels,
                y=cum_pct,
                name="Kumulativní rozptyl",
                mode="lines+markers+text",
                line=dict(color="#f59e0b", width=3),
                marker=dict(size=8),
                text=[f"{v:.1f} %" for v in cum_pct],
                textposition="top center"
            )
        )

        # Hranice 80 % a 90 %
        fig_scree.add_hline(y=80, line_dash="dash", line_color="#ef4444", annotation_text="Minimum: 80 %")
        fig_scree.add_hline(y=90, line_dash="dash", line_color="#10b981", annotation_text="Maximum: 90 %")

        # Zvýraznění vybrané 4. komponenty
        fig_scree.add_vrect(
            x0="PC3.5" if "PC3.5" in comp_labels else 2.5,
            x1=3.5,
            fillcolor="#10b981",
            opacity=0.15,
            line_width=0,
            annotation_text="Zvoleno k = 4",
            annotation_position="bottom right"
        )

        fig_scree.update_layout(
            title="Scree Plot & Kumulativní vysvětlený rozptyl (Standardized PCA)",
            xaxis_title="Hlavní komponenty",
            yaxis_title="Podíl rozptylu (%)",
            yaxis_range=[0, 105],
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=50, b=40)
        )

        st.plotly_chart(fig_scree, width="stretch")

    with col_scree2:
        st.markdown("#### 🎯 Rozpad rozptylu prvních 4 komponent:")

        t_breakdown = pd.DataFrame([
            {"Komponenta": "PC1", "Individuální rozptyl": f"{indiv_pct[0]:.2f} %", "Kumulativně": f"{cum_pct[0]:.2f} %", "Dominantní význam": "Fyzická velikost & hmotnost (carat, x, y, z)"},
            {"Komponenta": "PC2", "Individuální rozptyl": f"{indiv_pct[1]:.2f} %", "Kumulativně": f"{cum_pct[1]:.2f} %", "Dominantní význam": "Optické proporce (depth vs table)"},
            {"Komponenta": "PC3", "Individuální rozptyl": f"{indiv_pct[2]:.2f} %", "Kumulativně": f"{cum_pct[2]:.2f} %", "Dominantní význam": "Kvalitativní kódování (color, clarity)"},
            {"Komponenta": "PC4", "Individuální rozptyl": f"{indiv_pct[3]:.2f} %", "Kumulativně": f"{cum_pct[3]:.2f} %", "Dominantní význam": "Kvalita brusu (cut) vs rozptyl stolu"},
        ])
        st.dataframe(t_breakdown, width="stretch", hide_index=True)

        st.success(
            r"""
            **Matematický souhrn:**
            Redukcí z původních 9 dimenzí na 4 komponenty jsme:
            1. **Snížili dimenzionalitu o více než 55 %** (z 9 na 4).
            2. **Zachovali 82.87 % celkové variability** původních dat.
            3. **Odstranili multikolinearitu:** Korelace mezi jednotlivými komponentami $\text{corr}(\text{PC}_i, \text{PC}_j) = 0$ pro $i \neq j$.
            """
        )

# ==============================================================================
# TAB 3: VÝSLEDNÝ DATAFRAME
# ==============================================================================
with tab3:
    st.subheader("3. Výsledný DataFrame: `diamonds_pca_df`")
    st.markdown(
        "Níže je ukázka nově vytvořeného DataFrame sestaveného podle bodu 7 zadání: "
        "sloupce `PC1`, `PC2`, `PC3`, `PC4` spojené s cílovou proměnnou `price` (v USD)."
    )

    df_preview = pd.DataFrame(data["head_rows"])
    st.dataframe(df_preview, width="stretch", hide_index=True)

    c_dl1, c_dl2 = st.columns([1, 1])
    with c_dl1:
        st.download_button(
            "📥 Stáhnout náhled diamonds_pca_df (CSV)",
            data=df_preview.to_csv(index=False).encode("utf-8"),
            file_name="diamonds_pca_sample.csv",
            mime="text/csv",
            width="stretch"
        )
    with c_dl2:
        st.caption(
            "Poznámka: Celý dataset čítá 53 899 řádků. Každý řádek reprezentuje transformovaný diamant "
            "v novém 4-dimenzionálním ortogonálním souřadném systému."
        )

# ==============================================================================
# TAB 4: INTERAKTIVNÍ PROJEKCE
# ==============================================================================
with tab4:
    st.subheader("4. Interaktivní geometrie diamantů v prostoru hlavních komponent")

    sample_df = pd.DataFrame(data["sample_data"])

    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        color_by = st.selectbox(
            "Barevné kódování bodů (Color by):",
            ["price", "carat", "cut", "clarity", "depth"],
            index=0
        )
    with col_ctrl2:
        comp_x = st.selectbox("Osa X:", ["PC1", "PC2", "PC3", "PC4"], index=0)
        comp_y = st.selectbox("Osa Y:", ["PC1", "PC2", "PC3", "PC4"], index=1)

    fig_scatter = px.scatter(
        sample_df,
        x=comp_x,
        y=comp_y,
        color=color_by,
        color_continuous_scale="Plasma" if color_by in ["price", "carat"] else "Viridis",
        hover_data=["price", "carat", "cut", "clarity"],
        title=f"Projekce diamantů: {comp_x} vs {comp_y} (barevně dle {color_by})"
    )
    fig_scatter.update_layout(
        margin=dict(l=40, r=40, t=50, b=40),
        coloraxis_colorbar=dict(title=f"{color_by} (USD)" if color_by == "price" else color_by)
    )
    st.plotly_chart(fig_scatter, width="stretch")

    st.info(
        "💡 **Pozorování z projekce:** Podél osy **PC1** vidíme jasný plynulý gradient ceny a karátů. "
        "PC1 totiž zachycuje společný rozptyl fyzických rozměrů diamantu ($x, y, z$) a jeho hmotnosti."
    )
