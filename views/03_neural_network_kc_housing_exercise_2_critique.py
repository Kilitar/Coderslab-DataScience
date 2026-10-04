"""
Den 3: Neuronové sítě – Cvičení 2 (Predikce cen nemovitostí KC Housing) – Expertní analýza & SOTA 2026
==================================================================================================
Expertní rozbor cvičení:
1. Tabulkové dilema (Tabular Data Dilemma): Kdy použít stromy (XGBoost/RF) a kdy neuronové sítě.
2. Problém prostorové korelace (lat/long): Proč plně propojená vrstva obtížně modeluje lokální geografické bubliny.
3. Heteroskedasticita luxusních nemovitostí: Výhoda log-transformace cíle.
4. Velká celokurzovní evoluce regresních modelů na King County Housing (Den 1 až Den 3).
"""

import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


@st.cache_data
def load_kc_house_mlp_data():
    base_dir = Path(__file__).resolve().parent.parent
    json_path = base_dir / "03_Advanced_ML_Neural_Networks" / "data" / "kc_house_mlp_exercise_2_precomputed.json"
    if not json_path.exists():
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def render_neural_network_kc_housing_exercise_2_critique_view():
    st.title("🔬 Expertní analýza: Tabulkové sítě & Geografické nelinearity (SOTA 2026)")
    st.markdown(
        r"""
        V této sekci zkoumáme fundamentální otázku moderního strojového učení: 
        **Kdy se na tabulkových datech vyplatí použít neuronovou síť (MLP)** a kdy stále dominují 
        stromové ansámbly (**Random Forest** a **XGBoost**)?
        """
    )

    data = load_kc_house_mlp_data()
    if not data:
        st.error("Předpočtená data nebyla nalezena. Spusťte skript `03_Advanced_ML_Neural_Networks/09_neural_network_kc_housing_exercise_2.py`.")
        return

    exps = data["experiments"]
    baselines = data["baselines"]
    best_exp = exps["exp3_3layers"]

    # Horní KPI karty
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Keras MLP MAE", f"{best_exp['test_mae']:,.0f} USD", delta="78 157 USD při 655 vahách")
    with k2:
        st.metric("Random Forest MAE", f"{baselines['random_forest']['mae']:,.0f} USD", delta="Vítěz tabulkových dat (70 148 USD)")
    with k3:
        st.metric("Rozdíl proti OLS", f"-{baselines['linear_regression']['mae'] - best_exp['test_mae']:,.0f} USD", delta="Zlepšení o 38 %")
    with k4:
        st.metric("Klíčový limit sítí", "Geografická nelinearita", delta="lat/long tvoří složité shluky")

    st.markdown("---")

    t1, t2, t3, t4 = st.tabs([
        "⚔️ Sítě vs. Stromy na tabulkách",
        "🗺️ Problém lat/long & Geografie",
        "🏆 Celokurzovní žebříček King County",
        "💡 Produkční doporučení 2026"
    ])

    # =========================================================================
    # TAB 1: SÍTĚ VS STROMY NA TABULKÁCH
    # =========================================================================
    with t1:
        st.subheader("1. Tabulkové dilema (Tabular Data Dilemma): Proč stromy stále drží krok?")
        st.markdown(
            r"""
            Zatímco v počítačovém vidění (obrázky) a NLP (text) neuronové sítě naprosto převálcovaly klasické algoritmy, 
            na **strukturovaných tabulkových datech** (jako jsou nemovitosti v King County) je souboj mnohem vyrovnanější:
            """
        )

        c_t1, c_t2 = st.columns(2)
        with c_t1:
            st.info(
                r"""
                #### 🌲 Stromové modely (Random Forest, XGBoost):
                - **Invariance vůči měřítku:** Netřeba škálovat ani normalizovat data (`StandardScaler` není nutný).
                - **Robustnost na odlehlé hodnoty:** Extrémní rozlohy pozemků (`sqft_lot`) nezkreslí váhy.
                - **Přirozené zachycení skokových hranic:** Stromy štěpí prostor po osách rovnoběžných se souřadnicemi.
                - **Výsledek na KC Housing:** Random Forest dosahuje MAE **~70 000 USD** bez nutnosti ladění rychlosti učení.
                """
            )
        with c_t2:
            st.success(
                r"""
                #### 🧠 Neuronové sítě (Keras MLP):
                - **Hladké aproximační křivky:** Výstupy sítě jsou spojité a derivovatelné (žádné skoky).
                - **Extrémní citlivost na škálování:** Bez škálování $X$ a $y$ gradientní sestup okamžitě diverguje (exploze gradientu).
                - **Kompaktnost v produkci:** Vítězný model má pouhých **655 čísel (vah)** v paměti $\to$ velikost modelu je pouze pár kilobajtů!
                - **Výsledek na KC Housing:** Velmi slušná MAE **~78 000 USD** při bleskové odezvě.
                """
            )

    # =========================================================================
    # TAB 2: GEOGRAFIE
    # =========================================================================
    with t2:
        st.subheader("2. Proč je souřadnice lat/long pro plně propojenou vrstvu oříšek?")
        st.markdown(
            r"""
            V datasetu King County mají klíčový vliv na cenu zeměpisná šířka (`lat`) a délka (`long`). 
            Bohaté čtvrti jako **Mercer Island** nebo **Medina** tvoří izolované ostrůvky obklopené levnějšími oblastmi:
            """
        )

        st.warning(
            r"""
            **Matematický limit plně propojené vrstvy:**  
            Vrstvy `Dense` počítají vnitřní potenciál jako $z = w_{\text{lat}} \cdot \text{lat} + w_{\text{long}} \cdot \text{long} + b$.  
            To v rovině odpovídá **přímým oddělujícím liniím**.  
            Aby síť vymezila uzavřený kruhový ostrov vysokých cen kolem jezera, musí složit několik skrytých neuronů 
            s nelineární aktivací ReLU dohromady jako polygon!
            
            **Moderní SOTA řešení k 10/2026:**
            - **Spatial Embeddings:** Převod souřadnic na prostorové frekvence přes Fourierovy příznaky ($\sin(\omega x), \cos(\omega x)$).
            - **Target Encoding poštovních směrovacích čísel (`zipcode`):** Vyhlazený průměr cen v dané čtvrti.
            """
        )

    # =========================================================================
    # TAB 3: VELKÝ ŽEBŘÍČEK KING COUNTY
    # =========================================================================
    with t3:
        st.subheader("3. Celokurzovní srovnávací žebříček na datasetu King County Housing")
        st.markdown(
            r"""
            Přehled vývoje přesnosti predikce cen nemovitostí od Dne 1 až po současné neuronové sítě:
            """
        )

        all_kc_models = pd.DataFrame([
            {"Fáze kurzu": "Den 1: Lineární regrese", "Model": "Jednoduchá regrese (sqft_living)", "MAE (USD)": "~ 170 000 USD", "R²": "0.49", "Poznámka": "Pouze 1 příznak – hrubý odhad"},
            {"Fáze kurzu": "Den 1: Vícerozměrná OLS", "Model": "Vícerozměrná OLS (18 příznaků)", "MAE (USD)": f"{baselines['linear_regression']['mae']:,.0f} USD", "R²": f"{baselines['linear_regression']['r2']*100:.1f} %", "Poznámka": "Základní lineární benchmark"},
            {"Fáze kurzu": "Den 1 / Den 2: Stromy", "Model": "Rozhodovací strom (DecisionTree)", "MAE (USD)": f"{baselines['decision_tree']['mae']:,.0f} USD", "R²": f"{baselines['decision_tree']['r2']*100:.1f} %", "Poznámka": "Přeučuje se na listech"},
            {"Fáze kurzu": "Den 3: Neuronové sítě (MLP)", "Model": "Keras MLP (3 vrstvy: 18->12->6)", "MAE (USD)": f"{best_exp['test_mae']:,.0f} USD", "R²": f"{best_exp['test_r2']*100:.1f} %", "Poznámka": "Kompaktní parametrický model (655 vah)"},
            {"Fáze kurzu": "Den 3: Bagging", "Model": "Náhodný les (Random Forest 100)", "MAE (USD)": f"{baselines['random_forest']['mae']:,.0f} USD", "R²": f"{baselines['random_forest']['r2']*100:.1f} %", "Poznámka": "🏆 Vítěz na tomto konkrétním datasetu"}
        ])
        st.table(all_kc_models)

    # =========================================================================
    # TAB 4: PRODUKČNÍ DOPORUČENÍ
    # =========================================================================
    with t4:
        st.subheader("4. Best Practices pro nasazení regresních neuronových sítí (10/2026)")
        st.markdown(
            r"""
            1. **Vždy normalizujte vstupy i výstup:** Bez `StandardScaler` gradientní sestup u cen nemovitostí selže kvůli obřím hodnotám ztráty ($10^{11}$).
            2. **Logaritmická transformace cíle:** Trénování na $\log(\text{price})$ zabraňuje tomu, aby luxusní vily za miliony dolarů dominovaly ztrátové funkci na úkor běžných rodinných domků.
            3. **Early Stopping:** Sledujte validační MSE a zastavte trénování, jakmile se křivka zploští.
            4. **Kompaktnost v IoT / Mobile:** Model s 655 parametry zabírá necelých **3 KB paměti** a predikuje za zlomek milisekundy – ideální pro mobilní kalkulačky cen nemovitostí bez závislosti na cloudu!
            """
        )


if __name__ == "__main__":
    render_neural_network_kc_housing_exercise_2_critique_view()
