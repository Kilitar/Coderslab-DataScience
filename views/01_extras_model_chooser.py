import streamlit as st

st.title("🧭 Expertní analýza: Interaktivní průvodce výběrem modelu")
st.caption("Jak přemýšlet jako seniorní Machine Learning inženýr: Rozhodovací strom pro volbu správného regresního algoritmu podle povahy vašich dat a byznys cílů.")

st.markdown(r"""
Častá otázka studentů: *„Mám nová data. Podle čeho se mám rozhodnout, zda použít OLS, Ridge, Lasso, Polynom nebo Rozhodovací strom?“*  
Odpovězte na několik otázek o vašem projektu a algoritmus vám doporučí optimální postup:
""")

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

st.markdown("### 🎯 Doporučená strategie pro váš projekt:")

# Vyhodnocení
if "extrapolace" in q_extrapol and "Ano" in q_extrapol:
    st.error(
        "🚫 **STOP: Zákaz použití Rozhodovacích stromů!**  \n"
        "Rozhodovací stromy (CART) ani lesy (Random Forest) **neumí extrapolovat**. Mimo rozsah trénovacích dat narazí na konstantní strop.  \n"
        "👉 **Doporučení:** Použijte **Lineární regresi s L2 regularizací (Ridge)** nebo polynomiální trend nízkého stupně."
    )
elif "Obrovské množství sloupců" in q_data_type or "Feature Selection" in q_goal:
    st.success(
        "🏆 **Doporučený model: Lasso regrese (L1) nebo Elastic Net**  \n"
        "Díky L1 penalizaci model automaticky vynuluje šumové a redundantní příznaky a zanechá pouze klíčové tahouny.  \n"
        "💡 **Pozor:** Nezapomeňte před trénováním data normalizovat pomocí `StandardScaler`!"
    )
elif "Směs kategorií" in q_data_type or "Maximální možná" in q_goal:
    st.success(
        "🏆 **Doporučený model: Rozhodovací strom (CART) s prořezáním (nebo Random Forest v Dni 2)**  \n"
        "Stromy excelují na tabulkových datech, nevyžadují škálování a perfektně zvládají pravoúhlé ohraničení zón a skokové tarify.  \n"
        "💡 **Pozor:** Vždy ladit `max_depth` a `min_samples_leaf`, aby nedošlo k gigantickému přeučení!"
    )
elif "interpretovatelnost" in q_goal:
    st.success(
        "🏆 **Doporučený model: OLS Lineární regrese nebo Ridge regrese**  \n"
        "Každý koeficient beta má přímou ekonomickou interpretaci (*'při zvýšení plochy o 1 m² vzroste cena o X Kč'*).  \n"
        "Pokud jsou prediktory korelované, přidejte lehkou L2 penalizaci (Ridge)."
    )
else:
    st.info(
        "🏆 **Doporučený postup: Začněte OLS přímkou jako baselinou a porovnejte s prořezaným stromem.**"
    )

st.markdown("---")
st.markdown("#### 🗺️ Vizuální rozhodovací strom pro výběr algoritmu:")
st.markdown("""
```
                   Máte k dispozici tabulková data?
                             /          \\
                         ANO /            \\ NE (text, obraz)
                            v                v
                 Potřebujete extrapolaci?     -> Deep Learning
                       /          \\
                   ANO /            \\ NE
                      v                v
            -> Lineární / Ridge       Jsou data nelineární / se skoky?
                                            /          \\
                                        ANO /            \\ NE
                                           v                v
                                   -> Decision Tree     Máte mnoho sloupců?
                                  (nebo Random Forest)       /          \\
                                                         ANO /            \\ NE
                                                            v                v
                                                        -> Lasso          -> OLS
```
""")
