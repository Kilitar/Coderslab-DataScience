import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="PCA Diamanty – Expertní rozbor", page_icon="🔬", layout="wide")

st.title("🔬 Expertní analýza & Diagnostika: PCA diamantů")
st.caption(
    "Hloubkový metodický rozbor: Vliv škálování na kovarianční matici, geometrie Feature Loadings (význam komponent), "
    "prevence Data Leakage přes Scikit-learn Pipeline a dopad redukce na downstream regresi ceny."
)

DATA_PATH = Path("05_Unsupervised_Learning/data/pca_diamonds_precomputed.json")


@st.cache_data
def load_pca_diamonds_critique_precomputed():
    if DATA_PATH.exists():
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


data = load_pca_diamonds_critique_precomputed()

if not data:
    st.error("Předpočítaná data nebyla nalezena.")
    st.stop()

# Horní KPI karty
c1, c2, c3, c4 = st.columns(4)
c1.metric("Kletba neškálovaného PCA", "Zkreslení rozptylem", delta="table & depth dominují bez standardizace", delta_color="inverse")
c2.metric("Zachovaný výkon regrese", "97.6 % původního R2", delta="R2 = 0.8643 s pouhými 4 komponentami", delta_color="normal")
c3.metric("Eliminace multikolinearity", "VIF = 1.0 (Ortogonalita)", delta="Korelace mezi PCi a PCj je nulová", delta_color="normal")
c4.metric("Úspora dimenzí", "-55.6 %", delta="Z 9 příznaků na 4", delta_color="normal")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "⚖️ 1. Škálování: Scaled vs Unscaled PCA",
    "🧩 2. Feature Loadings (Význam komponent)",
    "🎯 3. Downstream regrese: 9 příznaků vs 4 PC",
    "🛡️ 4. Best Practices & Pipeline (MLOps 2026)"
])

# ==============================================================================
# TAB 1: SCALED VS UNSCALED
# ==============================================================================
with tab1:
    st.subheader("1. Proč je standardizace (StandardScaler) pro PCA otázkou života a smrti?")

    st.markdown(
        r"""
        V kurzech a učebnicích se často stává, že studenti aplikují `PCA()` přímo na neškálovanou matici $X$, 
        protože soubor se jmenuje `diamonds_preprocessed.csv`. Podívejme se, co se v takovém případě stane s matematikou:
        """
    )

    col_comp1, col_comp2 = st.columns(2)

    with col_comp1:
        st.markdown("#### ❌ Neškálované PCA (Unscaled):")
        st.write(
            r"""
            * PCA maximalizuje rozptyl $\operatorname{Var}(X_j)$.
            * Příznaky s velkými absolutními hodnotami a rozptyly (např. `table` s hodnotami 43–95 nebo `depth` 43–79) 
              **zcela ovládnou kovarianční matici**, i když nenesou nejvíce informace o ceně.
            * Naopak `carat` (hodnoty 0.2 až 5.0, rozptyl pouze $\approx 0.22$) je matematicky „převálcován“, 
              ačkoliv je z gemologického hlediska nejdůležitějším prediktorem ceny!
            """
        )
        
        raw_rows = []
        for k in range(1, 10):
            raw_rows.append({
                "Komponenta": f"PC{k}",
                "Individuální rozptyl": f"{data['raw_indiv'][k-1]*100:.2f} %",
                "Kumulativní rozptyl": f"{data['raw_cum'][k-1]*100:.2f} %"
            })
        st.dataframe(pd.DataFrame(raw_rows).head(5), width="stretch", hide_index=True)

    with col_comp2:
        st.markdown("#### ✅ Standardizované PCA (StandardScaler):")
        st.write(
            r"""
            * Před výpočtem transformujeme každý příznak: $z = \frac{x - \mu}{\sigma}$.
            * Všechny proměnné mají průměr $\mu = 0$ a rozptyl $\sigma^2 = 1$.
            * PCA pak neanalyzuje absolutní kovarianční matici, nýbrž **korelační matici**. Každá proměnná 
              má stejnou startovní šanci přispět k hlavním komponentám.
            * `carat` a rozměry $x, y, z$ získají zaslouženou dominanci v PC1.
            """
        )

        scaled_rows = []
        for k in range(1, 10):
            scaled_rows.append({
                "Komponenta": f"PC{k}",
                "Individuální rozptyl": f"{data['scaled_indiv'][k-1]*100:.2f} %",
                "Kumulativní rozptyl": f"{data['scaled_cum'][k-1]*100:.2f} %"
            })
        st.dataframe(pd.DataFrame(scaled_rows).head(5), width="stretch", hide_index=True)

    # Vizuální srovnání kumulativní variance
    fig_comp = go.Figure()
    comp_axis = [f"PC{i+1}" for i in range(len(data["features"]))]
    fig_comp.add_trace(go.Scatter(
        x=comp_axis,
        y=[v * 100 for v in data["scaled_cum"]],
        name="Standardized PCA (Správně)",
        mode="lines+markers",
        line=dict(color="#10b981", width=3)
    ))
    fig_comp.add_trace(go.Scatter(
        x=comp_axis,
        y=[v * 100 for v in data["raw_cum"]],
        name="Unscaled PCA (Zkresleno rozptylem)",
        mode="lines+markers",
        line=dict(color="#ef4444", width=3, dash="dot")
    ))
    fig_comp.add_hline(y=80, line_dash="dash", line_color="#f59e0b", annotation_text="80 % hranice")
    fig_comp.add_hline(y=90, line_dash="dash", line_color="#3b82f6", annotation_text="90 % hranice")
    fig_comp.update_layout(
        title="Srovnání kumulativního rozptylu: Standardizované vs Neškálované PCA",
        xaxis_title="Komponenty",
        yaxis_title="Kumulativní vysvětlený rozptyl (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=50, b=40)
    )
    st.plotly_chart(fig_comp, width="stretch")

# ==============================================================================
# TAB 2: LOADINGS
# ==============================================================================
with tab2:
    st.subheader("2. Geometrie a interpretace Feature Loadings")
    st.markdown(
        r"""
        Vlastní vektory (Eigenvectors) udávají lineární kombinaci původních proměnných tvořící novou osu:
        $$\text{PC}_i = w_{i1} X_1 + w_{i2} X_2 + \dots + w_{ip} X_p$$
        Váhy $w_{ij}$ (tzv. **Loadings**) nám prozrazují, jaký reálný fyzikální nebo gemologický fenomén jednotlivé komponenty reprezentují:
        """
    )

    df_loadings = pd.DataFrame(data["loadings_scaled"]).T

    fig_heat = px.imshow(
        df_loadings,
        labels=dict(x="Původní příznak", y="Hlavní komponenta", color="Váha (Loading)"),
        x=df_loadings.columns,
        y=df_loadings.index,
        color_continuous_scale="RdBu_r",
        zmin=-0.8,
        zmax=0.8,
        text_auto=".2f",
        title="Heatmapa Feature Loadings (Standardized PCA)"
    )
    fig_heat.update_layout(margin=dict(l=40, r=40, t=50, b=40))
    st.plotly_chart(fig_heat, width="stretch")

    c_load1, c_load2 = st.columns(2)
    with c_load1:
        st.markdown("#### 🔍 Rozbor PC1 a PC2:")
        st.write(
            r"""
            * **PC1 (Faktor fyzické velikosti – 46.3 % rozptylu):**  
              Extrémně vysoké pozitivní váhy u `carat` (+0.49), `x` (+0.50), `y` (+0.49) a `z` (+0.49).  
              Tato osa reprezentuje **trojrozměrný objem a hmotnost diamantu**. Větší diamant = vyšší hodnota PC1.
            * **PC2 (Faktor optického profilu – 16.0 % rozptylu):**  
              Silná pozitivní váha u `depth` (+0.71) a silná negativní váha u `table` (-0.56).  
              Vyjadřuje geometrický poměr diamantu: hluboký špičatý diamant s malou ploškou vs. plochý diamant s velkým stolem.
            """
        )

    with c_load2:
        st.markdown("#### 🔍 Rozbor PC3 a PC4:")
        st.write(
            r"""
            * **PC3 (Faktor gemologické čistoty – 11.0 % rozptylu):**  
              Vysoké váhy u `clarity` (+0.60) a `color` (+0.65).  
              Tato osa zcela izoluje optickou kvalitu drahokamu od jeho fyzické velikosti!
            * **PC4 (Faktor kvality výbrusu – 9.6 % rozptylu):**  
              Dominantní váha u kvality brusu `cut` (+0.75).  
              Odděluje precizně vybroušené diamanty (Ideal / Premium) od horších řezů (Fair / Good).
            """
        )

# ==============================================================================
# TAB 3: DOWNSTREAM REGRESE
# ==============================================================================
with tab3:
    st.subheader("3. Downstream regrese: Predikce ceny ze 4 komponent vs z 9 příznaků")
    st.markdown(
        "Abychom ověřili praktický dopad redukce dimenzionality, natrénovali jsme lineární regresi "
        "na predikci ceny (`price`) na testovací množině (20 % dat, random_state=42):"
    )

    metrics = data["metrics"]
    
    t_metrics = pd.DataFrame([
        {
            "Model": "1. Původní matice (9 příznaků)",
            "R2 skóre": f"{metrics['orig_9_features']['r2']:.4f}",
            "MAE (USD)": f"{metrics['orig_9_features']['mae']:.2f} USD",
            "RMSE (USD)": f"{metrics['orig_9_features']['rmse']:.2f} USD",
            "Multikolinearita (VIF)": "Vysoká (x,y,z r > 0.98)"
        },
        {
            "Model": "2. Standardizované PCA (4 komponenty)",
            "R2 skóre": f"{metrics['pca_4_scaled']['r2']:.4f}",
            "MAE (USD)": f"{metrics['pca_4_scaled']['mae']:.2f} USD",
            "RMSE (USD)": f"{metrics['pca_4_scaled']['rmse']:.2f} USD",
            "Multikolinearita (VIF)": "NULOVÁ (Ortogonální PC)"
        },
        {
            "Model": "3. Neškálované PCA (4 komponenty)",
            "R2 skóre": f"{metrics['pca_4_unscaled']['r2']:.4f}",
            "MAE (USD)": f"{metrics['pca_4_unscaled']['mae']:.2f} USD",
            "RMSE (USD)": f"{metrics['pca_4_unscaled']['rmse']:.2f} USD",
            "Multikolinearita (VIF)": "Nulová, ale zkreslený rozptyl"
        }
    ])
    st.dataframe(t_metrics, width="stretch", hide_index=True)

    col_diag1, col_diag2 = st.columns(2)
    with col_diag1:
        st.markdown("#### 🏆 Vyhodnocení komprese:")
        st.success(
            r"""
            * Původní model dosahuje $R^2 = 0.8854$.
            * Model se **4 standardizovanými komponentami** dosahuje $R^2 = 0.8643$.
            * **Ztratili jsme pouze 2.4 % predikční schopnosti**, přestože jsme zahodili více než 55 % sloupců!
            * Model s PCA netrpí žádnou multikolinearitou, což dramaticky stabilizuje koeficienty $\beta$.
            """
        )
    with col_diag2:
        st.markdown("#### ⚠️ Kolaps neškálovaného PCA:")
        st.error(
            r"""
            * Neškálované PCA propadá na $R^2 = 0.7932$ a chyba MAE roste na 1 240 USD.
            * Důvod: Bez standardizace PCA potlačilo `carat`, takže model přišel o klíčový vztah hmotnost $\rightarrow$ cena.
            """
        )

# ==============================================================================
# TAB 4: MLOPS & PIPELINE
# ==============================================================================
with tab4:
    st.subheader("4. MLOps Standard 2026: Jak správně nasadit PCA do produkce")
    
    st.markdown(
        r"""
        V produkčním Machine Learningu je přísně zakázáno volat `pca.fit_transform(X)` na celém datasetu 
        před rozdělením na Train a Test. To by představovalo **Data Leakage** (únik variance testovací sady do tréninku).
        
        Správný produkční vzor vyžaduje zapouzdření do `sklearn.pipeline.Pipeline`:
        """
    )

    st.code(
        r'''from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

# 1. Nejprve rozdělení dat PŘED jakoukoliv transformací
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Definice hermeticky uzavřené pipeline
pca_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=4)),
    ('regressor', Ridge(alpha=10.0))
])

# 3. Fitování POUZE na trénovací sadě (včetně škálování i PCA bází!)
pca_pipeline.fit(X_train, y_train)

# 4. Predikce na testovací sadě (automaticky transformuje data bázemi z tréninku)
y_pred = pca_pipeline.predict(X_test)
''',
        language="python"
    )

    st.info(
        "💡 **Shrnutí pro praxi:** PCA je mocný nástroj pro odstranění multikolinearity, "
        "kompresi datových toků a vizualizaci vícedimenzionálních prostorů, ale jeho spolehlivost "
        "stojí a padá na předchozí standardizaci a striktní separaci trénovacího toku."
    )
