"""
Domácí úkol (Session 2): Expertní analýza & Kritika modelu Neuronových sítí (Auto MPG)
=======================================================================================
Dataset: data/auto_mpg.csv (398 záznamů)
Model: Keras Sequential MLP
Precomputed: 04_Homework/data/auto_mpg_nn_precomputed.json
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.title("🔬 Expertní analýza & Diagnostika: Neuronové sítě (Auto MPG)")
st.caption(
    "Kritické zhodnocení: Problém skrytého textu '?' v numerických datech, fyzikální 'MPG iluze' nelinearity paliva, "
    "proč stromy (Random Forest) překonávají neuronové sítě na malých tabulárních datech a metodika prevence data leakage."
)

base_dir = Path(__file__).resolve().parent.parent
json_path = base_dir / "04_Homework" / "data" / "auto_mpg_nn_precomputed.json"


@st.cache_data
def load_mpg_stats():
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


stats = load_mpg_stats()
metrics = stats["test_metrics"] if stats else {}
models_cmp = stats["model_comparison"] if stats else {}

# Horní KPI karty
c1, c2, c3, c4 = st.columns(4)
c1.metric("Keras MLP R²", f"{metrics.get('r2', 0.7726):.4f}", delta="Gradientní regrese na 8 příznacích")
c2.metric("Random Forest R²", f"{models_cmp.get('random_forest', {}).get('r2', 0.8874):.4f}", delta="+11.5 % vyšší přesnost stromů")
c3.metric("Skrytý text '?'", "6 neplatných řádků", delta="Kritické riziko pro float konverzi")
c4.metric("Fyzika spotřeby", "Nelineární MPG", delta="Hyperbolický vztah k energii")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "⚠️ 1. Skrytý znak '?' a nástrahy Pandas dtypes",
    "🥊 2. Benchmark: Proč stromy porážejí sítě na N < 400?",
    "⛽ 3. Fyzika spalování: 'MPG iluze' vs. L/100 km",
    "🛡️ 4. Data Leakage při škálování & One-Hot kódování"
])

# ==============================================================================
# TAB 1: ZNAK '?' V NUMERICKÝCH DATECH
# ==============================================================================
with tab1:
    st.subheader("Nebezpečí tichého přetypování na 'object' v Pandas")

    st.markdown(
        """
        V souboru `auto_mpg.csv` je proměnná `horsepower` na první pohled číselná. Obsahuje však **6 záznamů s otazníkem (`'?'`)**, 
        který v reálných databázích slouží jako zástupný znak pro neznámou hodnotu.
        
        ### Co se stane, pokud nepoužijeme profilování (`ydata_profiling`)?
        1. **Tiché přetypování:** Pandas při načítání CSV narazí na `'?'` a celý sloupec automaticky interpretuje jako `object` (string).
        2. **Selhání modelování:** Pokus předat matici s textovými poli do `StandardScaler` nebo Kerasu okamžitě havaruje na `ValueError: could not convert string to float: '?'`.
        3. **Nezbytnost profilování:** Knihovna `ydata_profiling` nebo explicitní kontrola `df.info()` a `df['horsepower'].value_counts()` 
           tyto anomálie okamžitě odhalí.
        """
    )

    st.code(
        """# Správný postup ošetření:
# 1. Nahrazení otazníku hodnotou NaN
df['horsepower'] = pd.to_numeric(df['horsepower'].replace('?', np.nan))

# 2. Odstranění řádků (nebo mediánová imputace dle počtu válců)
df = df.dropna().reset_index(drop=True)
""",
        language="python"
    )

    st.info(
        "💡 **Produkční doporučení:** U malých datasetů ($N \\approx 398$) je odstranění 6 řádků (1.5 % dat) přijatelné. "
        "V robustní produkční pipeline je však lepší chybějící výkon imputovat např. mediánem v rámci stejného počtu válců a modelového roku."
    )

# ==============================================================================
# TAB 2: PROČ STROMY PORÁŽEJÍ SÍTĚ
# ==============================================================================
with tab2:
    st.subheader("Tabulární data malé velikosti: Srovnání Random Forest vs. Keras MLP")

    st.markdown(
        """
        Při srovnání na testovací sadě vidíme markantní rozdíl ve výkonu:
        - **Random Forest:** $\\text{MAE} = 1.71$ MPG, $\\text{RMSE} = 2.40$ MPG, $R^2 = 0.887$
        - **Keras MLP:** $\\text{MAE} = 2.55$ MPG, $\\text{RMSE} = 3.41$ MPG, $R^2 = 0.773$
        - **Lineární regrese:** $\\text{MAE} = 2.46$ MPG, $\\text{RMSE} = 3.26$ MPG, $R^2 = 0.792$
        """
    )

    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.markdown("#### 🌳 Proč dominují stromy?")
        st.markdown(
            """
            1. **Neinvariantnost vůči škálování:** Rozhodovací stromy pracují s jednorozměrnými prahovými děleními ($X_j > c$), 
               takže nepotřebují normalizaci a jsou odolné vůči odlehlým hodnotám.
            2. **Okamžité zachycení interakcí:** Interakce typu *„pokud má auto 8 válců a rok výroby < 1974, spotřeba prudce roste“* 
               jsou v hloubce stromu zachyceny přímo a přirozeně.
            3. **Malý vzorek ($N < 400$):** Random forest netrpí lokálními minimy gradientního sestupu a nepotřebuje tisíce gradientních kroků.
            """
        )

    with col_t2:
        st.markdown("#### 🧠 Omezení neuronových sítí na malých tabulkách")
        st.markdown(
            """
            1. **Induktivní bias:** Hluboké sítě excelují na datech s prostorovou či sekvenční strukturou (obrazy, audio, text). 
               Na tabulárních sloupcích bez přirozené prostorové topologie musí veškeré vztahy objevovat z nuly.
            2. **Riziko přeučení vs. podučení:** S $2\\,689$ parametry na pouhých $313$ trénovacích vzorcích je síť náchylná 
               k memorování šumu nebo uváznutí v suboptimálním lokálním minimu.
            3. **Hyperparametrická citlivost:** Výsledek sítě drasticky závisí na volbě learning rate, počtu neuronů, dávce i inicializaci vah.
            """
        )

# ==============================================================================
# TAB 3: FYZIKA SPOTŘEBY A MPG ILUZE
# ==============================================================================
with tab3:
    st.subheader("Fyzikální nelinearita: Proč je MPG zrádná metrika?")

    st.markdown(
        r"""
        V americkém automobilovém průmyslu je standardem **MPG (Miles Per Gallon)**. Z fyzikálního hlediska 
        však práce potřebná k překonání valivého a aerodynamického odporu roste lineárně se vzdáleností a hmotností:
        $$W = F \cdot d \implies \text{Spotřebované palivo } V \propto \text{Hmotnost} \cdot d$$
        
        Pokud počítáme spotřebu jako $\frac{V}{d}$ (**L / 100 km** nebo Gallons per 100 Miles), vztah k výkonu a hmotnosti je **lineární**.  
        Pokud však počítáme $\frac{d}{V}$ (**MPG**), zavádíme do modelu **hyperbolickou nelinearitu**:
        $$\text{MPG} = \frac{1}{\text{Spotřeba na kilometr}}$$
        """
    )

    # Vizualizace MPG iluze
    x_l100 = np.linspace(4, 25, 100)
    y_mpg = 235.215 / x_l100

    fig_mpg = go.Figure()
    fig_mpg.add_trace(go.Scatter(
        x=x_l100, y=y_mpg, mode="lines",
        line=dict(color="#dc2626", width=3),
        name="Hyperbola MPG vs. L/100km"
    ))
    fig_mpg.update_layout(
        title="Nelineární průběh převodu: Spotřeba v L/100km vs. MPG",
        xaxis_title="Reálná spotřeba paliva [L / 100 km] (lineární vůči nákladům)",
        yaxis_title="Metrika MPG [Miles Per Gallon]",
        template="plotly_white",
        height=350,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig_mpg, width="stretch")

    st.markdown(
        """
        > **Závěr pro Data Science:** Pokud by cílovou proměnnou byla přímo spotřeba paliva v L/100km (či GPM), 
        > lineární modely i neuronové sítě by dosahovaly vyšší přesnosti, protože by se nemusely učit obrácenou hyperbolickou funkci $1/x$.
        """
    )

# ==============================================================================
# TAB 4: DATA LEAKAGE A ONE-HOT ENCODING
# ==============================================================================
with tab4:
    st.subheader("Metodika: Prevence Data Leakage a správné kódování kategorií")

    st.markdown(
        """
        V zadání cvičení je sekvence kroků formulována takto:
        > *„Proveďte normalizaci dat pomocí třídy StandardScaler. Transformujte kategorické proměnné pomocí metody .get_dummies(). Rozdělte data na trénovací sadu a testovací sadu.“*
        
        ### ⚠️ Pozor na Data Leakage (Únik informací z testovací sady):
        Pokud bychom aplikovali `StandardScaler().fit_transform()` na celý dataset **před rozdělením**:
        1. Průměr $\\mu$ a směrodatná odchylka $\\sigma$ testovací sady by unikly do normalizace trénovacích dat.
        2. Testovací sada by přestala být nezávislým měřítkem generalizace.
        
        **Správné řešení:**
        ```python
        # 1. Nejprve rozdělení na train a test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

        # 2. Scaler FITUJEME POUZE NA TRÉNOVACÍCH DATECH!
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # 3. Na testovací data aplikujeme POUZE TRANSFORM:
        X_test_scaled = scaler.transform(X_test)
        ```
        """
    )

    st.markdown("---")
    st.markdown("### 🏷️ Dummy Variable Trap (Past fiktivních proměnných)")
    st.markdown(
        """
        Při použití `pd.get_dummies()` na sloupec `origin` (obsahující 3 hodnoty: USA, Europe, Japan) 
        je v regresních a lineárních modelech klíčové nastavit parametr `drop_first=True`.
        
        - Pokud ponecháme všechny 3 sloupce: $\\text{USA} + \\text{Europe} + \\text{Japan} = 1$. Sloupce jsou perfektně lineárně závislé, 
          což způsobuje singularitu matice kovariancí.
        - S `drop_first=True` vzniknou pouze 2 sloupce (`origin_Japan`, `origin_USA`), a Evropa funguje jako přirozený referenční bod.
        """
    )
