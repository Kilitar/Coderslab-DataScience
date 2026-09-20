import streamlit as st
import pandas as pd

st.title("🧭 Expertní analýza: Interaktivní průvodce výběrem modelu")
st.caption("Jak přemýšlet jako seniorní Machine Learning inženýr: Interaktivní rozhodovací strom (Decision Flowchart) pro volbu správného regresního algoritmu.")

st.markdown(r"""
Častá otázka studentů: *„Mám nová data. Podle čeho se mám rozhodnout, zda použít OLS, Ridge, Lasso, Polynom nebo Rozhodovací strom?“*  
Odpovězte na 3 klíčové otázky o vašem projektu a sledujte, jak se **v níže zobrazeném diagramu stromu automaticky rozsvítí vítězná cesta a doporučený model**!
""")

# =============================================================================
# 1. INTERAKTIVNÍ DOTAZNÍK
# =============================================================================
with st.container(border=True):
    q_data_type = st.radio(
        "1. Jaký charakter mají vaše prediktory (nezávislé proměnné)?",
        [
            "Převážně spojité fyzikální veličiny s plynulými závislostmi (např. teplota, koncentrace, rozměry).",
            "Směs kategorií, skokových tarifů, administrativních zón a tabulkových čísel.",
            "Obrovské množství sloupců (desítky až stovky), z nichž mnohé jsou silně korelované nebo zbytečné."
        ]
    )

    q_goal = st.radio(
        "2. Co je hlavním byznys cílem vašeho modelu?",
        [
            "100% čistá interpretovatelnost koeficientů pro management nebo regulátora (např. bankovní scoring, ekonometrie).",
            "Maximální možná predikční přesnost bez ohledu na složitost vnitřních pravidel.",
            "Automatický výběr nejdůležitějších faktorů (Feature Selection) z obřího množství proměnných."
        ]
    )

    q_extrapol = st.radio(
        "3. Budete předpovídat data ležící mimo rozsah dosud naměřených hodnot (extrapolace)?",
        [
            "Ano (např. předpověď budoucího trendu růstu, inflace nebo extrémních hodnot).",
            "Ne (nová data budou v podobném rozsahu jako dosavadní trénovací data)."
        ]
    )

# Logika vyhodnocení
is_extrapol = "extrapolace" in q_extrapol and "Ano" in q_extrapol
is_lasso = "Obrovské množství sloupců" in q_data_type or "Feature Selection" in q_goal
is_tree = "Směs kategorií" in q_data_type or "Maximální možná" in q_goal

if is_extrapol:
    winner_key = "RIDGE"
elif is_tree:
    winner_key = "TREE"
elif is_lasso:
    winner_key = "LASSO"
else:
    winner_key = "OLS"

# =============================================================================
# 2. VÝSLEDEK DOPORUČENÍ
# =============================================================================
st.markdown("### 🎯 Doporučená strategie pro váš projekt:")

if winner_key == "RIDGE":
    st.error(
        "🚫 **STOP: Zákaz použití Rozhodovacích stromů!**  \n"
        "Rozhodovací stromy (CART) ani lesy (Random Forest) **neumí extrapolovat**. Mimo rozsah trénovacích dat narazí na konstantní strop.  \n"
        "👉 **Doporučený model: Lineární regrese s L2 regularizací (Ridge)** nebo polynomiální trend nízkého stupně."
    )
elif winner_key == "LASSO":
    st.success(
        "🏆 **Doporučený model: Lasso regrese (L1) nebo Elastic Net**  \n"
        "Díky L1 penalizaci model automaticky vynuluje šumové a redundantní příznaky a zanechá pouze klíčové tahouny.  \n"
        "💡 **Pozor:** Nezapomeňte před trénováním data normalizovat pomocí `StandardScaler`!"
    )
elif winner_key == "TREE":
    st.success(
        "🏆 **Doporučený model: Rozhodovací strom (CART) s prořezáním (nebo Random Forest v Dni 2)**  \n"
        "Stromy excelují na tabulkových datech, nevyžadují škálování a perfektně zvládají pravoúhlé ohraničení zón a skokové tarify.  \n"
        "💡 **Pozor:** Vždy ladit `max_depth` a `min_samples_leaf`, aby nedošlo k gigantickému přeučení!"
    )
else:
    st.success(
        "🏆 **Doporučený model: OLS Lineární regrese (s lehkou Ridge penalizací)**  \n"
        "Každý koeficient beta má přímou ekonomickou interpretaci (*'při zvýšení plochy o 1 m² vzroste cena o X Kč'*).  \n"
        "Pokud jsou prediktory korelované, přidejte lehkou L2 penalizaci (Ridge)."
    )

st.markdown("---")

# =============================================================================
# 3. VELKÝ VIZUÁLNÍ DIAGRAM ROZHODOVACÍHO STROMU (MERMAID TREE)
# =============================================================================
st.markdown("### 🌲 Vizuální rozhodovací strom pro výběr algoritmu:")
st.markdown("Přehledný grafický strom logických rozhodnutí. Podle vaší volby nahoře je vítězná větev a model **zvýrazněna zeleně (ACTIVE)**:")

# Dynamické Mermaid styly
c_default = "fill:#1E293B,stroke:#475569,stroke-width:2px,color:#FFFFFF"
c_active = "fill:#064E3B,stroke:#10B981,stroke-width:4px,color:#34D399,font-weight:bold"
c_root = "fill:#0F172A,stroke:#3B82F6,stroke-width:3px,color:#60A5FA,font-weight:bold"

style_ridge = c_active if winner_key == "RIDGE" else c_default
style_tree = c_active if winner_key == "TREE" else c_default
style_lasso = c_active if winner_key == "LASSO" else c_default
style_ols = c_active if winner_key == "OLS" else c_default

mermaid_code = f"""
graph TD
    classDef defaultStyle {c_default};
    classDef activeStyle {c_active};
    classDef rootStyle {c_root};

    START["📂 MÁME TABULKOVÁ DATA?"]:::rootStyle
    
    START -->|ANO| Q_EXTRAPOL["🎯 POTŘEBUJEME EXTRAPOLOVAT DO BUDOUCNA?"]
    START -->|NE| DEEP["🖼️ Nestrukturovaná data (Text, Foto)<br>👉 Deep Learning / Neuronové sítě"]
    
    Q_EXTRAPOL -->|ANO: Mimo rozsah dat| RIDGE["🏆 RIDGE REGRESE (L2)<br>Lineární trend drží směr<br>⚠️ Stromy přísně zakázány!"]:::styleRidge
    Q_EXTRAPOL -->|NE: V rámci známých dat| Q_NONLIN["📐 OBSAHUJÍ DATA SKOKY NEBO ZÓNY?"]
    
    Q_NONLIN -->|ANO: Tarify, zóny, skoky| TREE["🏆 ROZHODOVACÍ STROM (CART)<br>nebo Random Forest<br>✔️ Nevyžaduje škálování"]:::styleTree
    Q_NONLIN -->|NE: Plynulé vztahy| Q_COLS["📊 MÁME DESÍTKY AŽ STOVKY PŘÍZNAKŮ?"]
    
    Q_COLS -->|ANO: Mnoho šumu / korelací| LASSO["🏆 LASSO (L1) / ELASTIC NET<br>Automatický výběr příznaků<br>⚠️ Nutný StandardScaler"]:::styleLasso
    Q_COLS -->|NE: Málo čistých sloupců| OLS["🏆 OLS LINEÁRNÍ REGRESE<br>100% interpretovatelné koeficienty<br>✔️ Standard pro reporting"]:::styleOls

    class RIDGE styleRidge;
    class TREE styleTree;
    class LASSO styleLasso;
    class OLS styleOls;
"""

# Zástupný kód stylů pro Mermaid
mermaid_rendered = f"""
```mermaid
graph TD
    START["📂 1. MÁME TABULKOVÁ DATA?"] -->|ANO| Q_EXTRAPOL["🎯 2. POTŘEBUJEME EXTRAPOLOVAT DO BUDOUCNA?"]
    START -->|NE| DEEP["🖼️ Nestrukturovaná data (Text, Foto)<br>👉 Deep Learning"]
    
    Q_EXTRAPOL -->|ANO: Předpověď mimo data| RIDGE["🏆 RIDGE REGRESE (L2) / LINEÁRNÍ TREND<br>⚠️ Stromy přísně zakázány (konstantní strop)!"]
    Q_EXTRAPOL -->|NE: Předpověď v mezích dat| Q_NONLIN["📐 3. JSOU V DATECH SKOKY ČI ZÓNY?"]
    
    Q_NONLIN -->|ANO: Zóny, skokové tarify| TREE["🏆 ROZHODOVACÍ STROM (CART) / RANDOM FOREST<br>✔️ Nevyžaduje škálování, zachytí skoky"]
    Q_NONLIN -->|NE: Plynulé vztahy| Q_COLS["📊 4. MÁME VELKÉ MNOŽSTVÍ ŠUMU A SLOUPCŮ?"]
    
    Q_COLS -->|ANO: Mnoho irelevantních proměnných| LASSO["🏆 LASSO REGRESE (L1) / ELASTIC NET<br>✔️ Automatický výběr nejdůležitějších příznaků"]
    Q_COLS -->|NE: Málo kvalitních prediktorů| OLS["🏆 ZÁKLADNÍ LINEÁRNÍ REGRESE (OLS)<br>✔️ Maximální byznys interpretovatelnost vah"]
```
"""
st.markdown(mermaid_rendered)

st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

# =============================================================================
# 4. VELKÝ REFERENČNÍ CHEAT-SHEET
# =============================================================================
st.markdown("#### 📋 Přehledná srovnávací tabulka rozhodování:")
summary_chooser = [
    {
        "Kritérium / Situace": "Potřebuji extrapolovat do budoucna",
        "Doporučený model": "Lineární regrese / Ridge (L2)",
        "Zakázaný model": "Rozhodovací stromy (CART/RF)",
        "Důvod": "Stromy narazí na konstantní strop trénovacích dat."
    },
    {
        "Kritérium / Situace": "Mám stovky sloupců a mnoho šumu",
        "Doporučený model": "Lasso regrese (L1) / Elastic Net",
        "Zakázaný model": "Základní OLS bez penalizace",
        "Důvod": "OLS se přeučí; Lasso vynuluje nepotřebné váhy."
    },
    {
        "Kritérium / Situace": "Skokové tarify, zóny, kategorie",
        "Doporučený model": "Rozhodovací strom (CART) / Random Forest",
        "Zakázaný model": "Hladká přímka OLS",
        "Důvod": "Přímka nedokáže ohraničit pravoúhlé enklávy a skoky."
    },
    {
        "Kritérium / Situace": "Bankovní scoring, audit, regulace",
        "Doporučený model": "OLS Lineární regrese",
        "Zakázaný model": "Polynomy vysokých stupňů, hluboké stromy",
        "Důvod": "Nutnost vysvětlit každý koeficient regulátorovi."
    }
]
st.dataframe(pd.DataFrame(summary_chooser), width="stretch")
