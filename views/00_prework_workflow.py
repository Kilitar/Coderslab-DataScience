from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

base_dir = Path(__file__).resolve().parent.parent
theory_path = base_dir / "00_Prework" / "theory" / "03_data_prep_scikit_workflow.md"

st.title("🛠️ Prework 03: Příprava dat, Scikit-learn & Životní cyklus modelu")
st.caption("Preprocessing dat, strategie škálování, prevence Data Leakage, Scikit-learn Pipelines, MLOps úskalí a doporučené zdroje (09/2026).")

tab1, tab2, tab3 = st.tabs([
    "📖 Teoretický rozbor",
    "⚙️ Laboratoř škálování (Scaler Compare)",
    "📚 Doporučené zdroje a literatura (2026)"
])

# -----------------------------------------------------------------------------
# TAB 1: TEORETICKÝ TEXT
# -----------------------------------------------------------------------------
with tab1:
    if theory_path.exists():
        with open(theory_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.warning("Soubor 03_data_prep_scikit_workflow.md nebyl nalezen.")

# -----------------------------------------------------------------------------
# TAB 2: INTERAKTIVNÍ DEMO ŠKÁLOVÁNÍ
# -----------------------------------------------------------------------------
with tab2:
    st.markdown("### ⚙️ Porovnání metod škálování v Scikit-learn")
    st.markdown(r"""
    Prozkoumejte, jak různé škálovače reagují na data s **extrémními odlehlými hodnotami (outliery)**.
    - **Původní data:** Běžné hodnoty kolem 10–50 s několika extrémy až do 500.
    - **StandardScaler (Z-score):** Středí kolem 0 a dělí směrodatnou odchylkou. Outliery však směrodatnou odchylku nafouknou a stlačí normální body k sobě.
    - **MinMaxScaler:** Stlačí vše striktně do intervalu $[0, 1]$. Outlier na 500 způsobí, že 99 % běžných dat skončí stlačeno mezi 0.0 a 0.08!
    - **RobustScaler:** Používá medián a IQR ($Q_3 - Q_1$). Outliery ho neovlivní, normální data zůstávají krásně rozprostřená.
    """)

    np.random.seed(42)
    normal_data = np.random.normal(loc=30, scale=8, size=150)
    outliers = np.array([250, 380, 490])
    raw_vals = np.concatenate([normal_data, outliers]).reshape(-1, 1)

    std_scaler = StandardScaler().fit_transform(raw_vals).flatten()
    minmax_scaler = MinMaxScaler().fit_transform(raw_vals).flatten()
    robust_scaler = RobustScaler().fit_transform(raw_vals).flatten()

    scale_df = pd.DataFrame({
        "Původní surová data": raw_vals.flatten(),
        "StandardScaler (Z-score)": std_scaler,
        "MinMaxScaler ([0, 1])": minmax_scaler,
        "RobustScaler (IQR)": robust_scaler
    })

    chosen_method = st.selectbox(
        "Vyberte metodu pro detailní náhled:",
        options=[
            "StandardScaler (Z-score)",
            "MinMaxScaler ([0, 1])",
            "RobustScaler (IQR)",
            "Všechny dohromady (Boxplot porovnání)"
        ],
        index=3
    )

    if chosen_method == "Všechny dohromady (Boxplot porovnání)":
        melted_df = scale_df[["StandardScaler (Z-score)", "MinMaxScaler ([0, 1])", "RobustScaler (IQR)"]].melt(
            var_name="Metoda škálování", value_name="Transformovaná hodnota"
        )
        fig_box = px.box(
            melted_df,
            x="Metoda škálování",
            y="Transformovaná hodnota",
            color="Metoda škálování",
            points="outliers",
            title="Srovnání distribucí po transformaci různými škálovači"
        )
        fig_box.update_layout(height=450, showlegend=False)
        st.plotly_chart(fig_box, width="stretch")
    else:
        fig_hist = px.histogram(
            scale_df,
            x=chosen_method,
            nbins=35,
            title=f"Distribuce dat po aplikaci: {chosen_method}",
            labels={"x": chosen_method}
        )
        fig_hist.update_traces(marker_color="#2A9D8F")
        fig_hist.update_layout(height=380)
        st.plotly_chart(fig_hist, width="stretch")

    st.success(
        "🎯 **Doporučení pro praxi:** Pokud váš dataset obsahuje výrazné odlehlé hodnoty nebo šikmá rozdělení "
        "(např. ceny nemovitostí, velikosti transakcí), je **RobustScaler** bezpečnější volbou než MinMaxScaler nebo StandardScaler."
    )

# -----------------------------------------------------------------------------
# TAB 3: ZDROJE A LITERATURA
# -----------------------------------------------------------------------------
with tab3:
    st.markdown("### 📚 Odkazy na oficiální zdroje a doporučenou literaturu (2026)")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🌐 Oficiální dokumentace klíčových knihoven:")
        st.markdown("""
        - [📘 **Scikit-learn User Guide**](https://scikit-learn.org/stable/user_guide.html)  
          *Neocenitelná studnice algoritmů, matematických formulací a best practices.*
        - [🐼 **Pandas User Guide & API**](https://pandas.pydata.org/docs/user_guide/index.html)  
          *Kompletní návody na čištění dat, agregace, spojování tabulek a optimalizaci paměti.*
        - [⚡ **NumPy User Guide**](https://numpy.org/doc/stable/user/)  
          *Základy vektorizace, broadcastingu a manipulace s n-rozměrnými poli.*
        - [📊 **Plotly Python Documentation**](https://plotly.com/python/)  
          *Dokumentace pro tvorbu moderních, responsivních webových grafů.*
        - [👑 **Streamlit Docs**](https://docs.streamlit.io/)  
          *Oficiální manuál pro tvorbu datových aplikací a dashboardů.*
        """)

    with col2:
        st.markdown("#### 📖 Knihy považované za bibli oboru:")
        st.markdown("""
        - **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow**  
          *Autor: Aurélien Géron (O'Reilly)*  
          Nejlepší praktická učebnice od základů lineární regrese až po Deep Learning.
        - **An Introduction to Statistical Learning (ISLR with Applications in Python)**  
          *Autoři: James, Witten, Hastie, Tibshirani (Springer)*  
          Kultovní kniha z Princetonu/Stanfordu s vynikajícím matematickým fundamentem, dostupná **zcela zdarma online**.
        - **Designing Machine Learning Systems**  
          *Autorka: Chip Huyen (O'Reilly)*  
          Kompletní příručka pro přechod z Jupyter Notebooku do reálné produkce (MLOps, monitoring driftu, pipeline).
        """)
