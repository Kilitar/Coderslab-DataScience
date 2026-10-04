"""
Den 3: Ucelené shrnutí pokročilých modelů strojového učení & Závěrečný kvíz
===========================================================================
Závěrečná syntéza 3 stěžejních pilířů Dne 3 (Advanced Machine Learning Models):
1. Bagging & Náhodný les (Random Forest) – redukce rozptylu průměrováním.
2. Boosting & XGBoost – sekvenční redukce vychýlení učením z reziduí.
3. Neuronové sítě (Deep Learning) & Keras – Feedforward, nelinearity a Backpropagation.
+ Velká celokurzovní srovnávací matice a interaktivní závěrečný vědomostní kvíz.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def render_day_3_summary_view():
    st.title("🎓 Den 3: Velká syntéza Advanced ML & Závěrečný kvíz")
    st.markdown(
        r"""
        Třetí den kurzu Data Science & Machine Learning završil přechod od klasických modelů 
        k moderním pokročilým přístupům: **ansámblovému učení (Bagging, Boosting)** a **hlubokému učení (Neuronové sítě)**.
        Níže naleznete ucelenou syntézu, srovnávací matici, rozhodovacího průvodce a závěrečný test znalostí.
        """
    )

    # Horní KPI karty
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("1. Pilíř: Bagging", "Random Forest", delta="Snížení rozptylu (Variance)")
    with k2:
        st.metric("2. Pilíř: Boosting", "XGBoost", delta="Snížení vychýlení (Bias)")
    with k3:
        st.metric("3. Pilíř: Deep Learning", "Keras / TensorFlow", delta="Feedforward + Backprop")
    with k4:
        st.metric("Celokurzovní šampión", "XGBoost (Diamanty)", delta="Rekordní MAE 264.79 USD")

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🏛️ Tři pilíře Advanced ML",
        "📊 Srovnávací matice & Průvodce",
        "🏆 Celokurzovní žebříček přesnosti",
        "🧠 Závěrečný znalostní kvíz (Quiz Time)"
    ])

    # =========================================================================
    # TAB 1: TŘI PILÍŘE
    # =========================================================================
    with tab1:
        st.subheader("Architektura pokročilých modelů Dne 3")
        st.markdown(
            r"""
            Každý ze tří přístupů řeší kompromis mezi vychýlením a rozptylem (*Bias-Variance Trade-off*) odlišnou filozofií:
            """
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            with st.container(border=True):
                st.markdown("### 🌲 1. Bagging (Random Forest)")
                st.markdown(
                    r"""
                    **Filozofie:** *„Moudrost davu nezávislých expertů.“*
                    - **Bázové modely:** Hluboké, plně vyrostlé stromy (nízký bias, vysoký rozptyl).
                    - **Princip:** Trénování mnoha nezávislých stromů paralelně na náhodných bootstrapových vzorcích dat i příznaků (*subspace*).
                    - **Výsledná predikce:** Průměr (regrese) nebo většinové hlasování (klasifikace).
                    - **Klíčový přínos:** **Radikální redukce rozptylu (Variance)** bez zvýšení biasu.
                    - **Odolnost:** Imunní vůči lokálnímu šumu a outlierům.
                    """
                )
        with c2:
            with st.container(border=True):
                st.markdown("### 🚀 2. Boosting (XGBoost)")
                st.markdown(
                    r"""
                    **Filozofie:** *„Učení se z vlastních minulých chyb.“*
                    - **Bázové modely:** Velmi mělké stromy (*Weak Learners*, vysoký bias, nízký rozptyl).
                    - **Princip:** Modely se trénují sekvenčně za sebou přímo na **reziduích (chybách)** předchozího stavu.
                    - **Výsledná predikce:** Aditivní součet vážený rychlostí učení (*Learning Rate* $\eta$).
                    - **Klíčový přínos:** **Radikální redukce vychýlení (Bias)**.
                    - **XGBoost inovace:** Vestavěná $L_1/L_2$ regularizace, podpora chybějících hodnot (`NaN`) a cache-aware paralelizace.
                    """
                )
        with c3:
            with st.container(border=True):
                st.markdown("### 🧠 3. Neuronové sítě (Keras)")
                st.markdown(
                    r"""
                    **Filozofie:** *„Biologicky inspirovaná hierarchie reprezentací.“*
                    - **Stavební blok:** Neuron počítající vážený součet $z = Wx + b$ s nelineární aktivací.
                    - **Princip:** Dopředný průchod (**Feedforward**) $\to$ výpočet ztráty $\to$ zpětné šíření chyb (**Backpropagation**) přes řetízkové pravidlo.
                    - **Klíčový přínos:** **Univerzální aproximátor** libovolně složitých nelineárních funkcí.
                    - **Doména:** Obraz (CNN), text/NLP (Transformers), zvuk a kompaktní parametrické regrese.
                    """
                )

    # =========================================================================
    # TAB 2: SROVNÁVACÍ MATICE & PRŮVODCE
    # =========================================================================
    with tab2:
        st.subheader("Srovnávací matice pokročilých přístupů")
        
        matrix_df = pd.DataFrame([
            {"Kritérium": "Typ bázových modelů", "Random Forest": "Hluboké stromy (Low Bias, High Var)", "XGBoost": "Mělké stromy (High Bias, Low Var)", "Neuronové sítě (MLP)": "Vrstvy neuronů s nelinearitou"},
            {"Kritérium": "Způsob trénování", "Random Forest": "Paralelní (nezávislé)", "XGBoost": "Sekvenční (na reziduích)", "Neuronové sítě (MLP)": "Dopředný průchod + Backprop"},
            {"Kritérium": "Primární cíl učení", "Random Forest": "Snížení rozptylu (Variance)", "XGBoost": "Snížení vychýlení (Bias)", "Neuronové sítě (MLP)": "Aproximace složitých vazeb"},
            {"Kritérium": "Nutnost škálování vstupů", "Random Forest": "❌ Není nutné", "XGBoost": "❌ Není nutné", "Neuronové sítě (MLP)": "⚠️ Kriticky nutné (StandardScaler)"},
            {"Kritérium": "Citlivost na odlehlé hodnoty", "Random Forest": "Velmi nízká (průměr zahladí šum)", "XGBoost": "Střední (může se zacyklit na outlieru)", "Neuronové sítě (MLP)": "Vysoká (outlier způsobí obří gradient)"},
            {"Kritérium": "Velikost modelu v paměti", "Random Forest": "Velká (desítky MB)", "XGBoost": "Střední (stovky KB)", "Neuronové sítě (MLP)": "Extrémně malá (jednotky KB)"},
            {"Kritérium": "Rychlost inference (predikce)", "Random Forest": "Střední", "XGBoost": "Rychlá", "Neuronové sítě (MLP)": "Blesková (násobení matic)"},
            {"Kritérium": "Typická doména", "Random Forest": "Zašuměná tabulková data", "XGBoost": "Soutěžní tabulková data (Kaggle)", "Neuronové sítě (MLP)": "Nestrukturovaná data (obraz, text)"}
        ])
        st.table(matrix_df)

        st.markdown("---")
        st.subheader("Diagnostický průvodce: Kdy jaký model nasadit k 10/2026?")
        
        c_p1, c_p2 = st.columns(2)
        with c_p1:
            st.info(
                r"""
                #### 📋 Tabulková data (CSV, Tabulky, Relační databáze):
                1. **Malý dataset ($N < 2\,000$) s vysokým šumem:**  
                   $\implies$ **Random Forest**. Bagging je stabilní a nepřeučí se na náhodném šumu (potvrzeno v Cvičení 1 na pacientech se srdcem).
                2. **Střední až obří dataset ($N > 5\,000$):**  
                   $\implies$ **XGBoost / LightGBM**. Gradientní boosting systematicky dosahuje nejvyšší přesnosti (potvrzeno rekordním MAE 264.79 USD na diamantech).
                """
            )
        with c_p2:
            st.success(
                r"""
                #### 🖼️ Nestrukturovaná & Speciální data:
                1. **Obrázky, Computer Vision, Audio:**  
                   $\implies$ **Konvoluční neuronové sítě (CNN)**. Prostorová invariance a filtry $3 \times 3$ drtí jakékoliv stromové řešení.
                2. **Text, NLP, Sekvence:**  
                   $\implies$ **Transformers / Rekurentní sítě**.
                3. **Kompaktní nasazení do mikrokontrolérů / Mobile (IoT):**  
                   $\implies$ **Neuronové sítě (MLP)**. Model má pouhých pár stovek parametrů (velikost pod 3 KB).
                """
            )

    # =========================================================================
    # TAB 3: CELOKURZOVNÍ ŽEBŘÍČEK PŘESNOSTI
    # =========================================================================
    with tab3:
        st.subheader("Evoluce přesnosti napříč celým kurzem (Den 1 až Den 3)")
        st.markdown(
            r"""
            Podívejte se, jak se vyvíjela přesnost předpovědí na dvou stěžejních datasetech kurzu:
            """
        )

        st.markdown("#### 💎 Regrese cen diamantů (diamonds.csv)")
        diam_df = pd.DataFrame([
            {"Fáze kurzu": "Den 1: Lineární regrese", "Algoritmus": "Vícerozměrná OLS regrese", "Test MAE (USD)": "~ 740 USD", "R² skóre": "0.919", "Hodnocení": "Hrubý lineární odhad"},
            {"Fáze kurzu": "Den 1: Polynomiální regrese", "Algoritmus": "Polynom 2. stupně", "Test MAE (USD)": "~ 520 USD", "R² skóre": "0.948", "Hodnocení": "Zlepšení, ale exploze příznaků"},
            {"Fáze kurzu": "Den 1 / Den 2: Stromy", "Algoritmus": "Rozhodovací strom (DecisionTree)", "Test MAE (USD)": "358.34 USD", "R² skóre": "0.965", "Hodnocení": "Schodovitá aproximace"},
            {"Fáze kurzu": "Den 3: Bagging", "Algoritmus": "Random Forest (100 stromů)", "Test MAE (USD)": "268.21 USD", "R² skóre": "0.981", "Hodnocení": "Paralelní průměrování"},
            {"Fáze kurzu": "Den 3: Boosting (Zadání)", "Algoritmus": "XGBoost (Školní parametry)", "Test MAE (USD)": "273.23 USD", "R² skóre": "0.980", "Hodnocení": "Ignorován min_samples_leaf"},
            {"Fáze kurzu": "Den 3: Boosting (SOTA 2026)", "Algoritmus": "XGBoost (Nativní ladění)", "Test MAE (USD)": "264.79 USD", "R² skóre": "0.981", "Hodnocení": "🏆 Absolutní rekord celého kurzu!"}
        ])
        st.table(diam_df)

        st.markdown("#### 🏡 Regrese cen nemovitostí (kc_house_data_preprocessed.csv)")
        kc_df = pd.DataFrame([
            {"Fáze kurzu": "Den 1: Lineární regrese", "Algoritmus": "Vícerozměrná OLS (18 příznaků)", "Test MAE (USD)": "125 744 USD", "R² skóre": "70.1 %", "Hodnocení": "Nezachytí geografické bubliny"},
            {"Fáze kurzu": "Den 1 / Den 2: Stromy", "Algoritmus": "Rozhodovací strom (DecisionTree)", "Test MAE (USD)": "105 168 USD", "R² skóre": "76.1 %", "Hodnocení": "Vysoký rozptyl"},
            {"Fáze kurzu": "Den 3: Neuronové sítě", "Algoritmus": "Keras MLP (3 vrstvy: 18->12->6)", "Test MAE (USD)": "78 157 USD", "R² skóre": "86.2 %", "Hodnocení": "Kompaktní parametrický model (3 KB)"},
            {"Fáze kurzu": "Den 3: Bagging", "Algoritmus": "Random Forest (100 stromů)", "Test MAE (USD)": "70 148 USD", "R² skóre": "88.0 %", "Hodnocení": "🏆 Vítěz na tomto tabulkovém datasetu"}
        ])
        st.table(kc_df)

    # =========================================================================
    # TAB 4: ZÁVĚREČNÝ KVÍZ
    # =========================================================================
    with tab4:
        st.subheader("4. Velký závěrečný vědomostní kvíz Dne 3 (Quiz Time!)")
        st.markdown(
            r"""
            Ověřte své znalosti ze všech tří pilířů dnešního dne:
            """
        )

        q1 = st.radio(
            "1. Jaký je hlavní rozdíl v cíli mezi technikami Bagging a Boosting?",
            [
                "Bagging snižuje vychýlení (Bias), zatímco Boosting snižuje rozptyl (Variance).",
                "Bagging primárně snižuje rozptyl (Variance) průměrováním nezávislých hlubokých modelů, zatímco Boosting sekvenčně snižuje vychýlení (Bias) učením z reziduí.",
                "Mezi Baggingem a Boostingem není žádný matematický rozdíl, liší se pouze implementací v knihovnách."
            ],
            key="sum_q1"
        )
        if q1 == "Bagging primárně snižuje rozptyl (Variance) průměrováním nezávislých hlubokých modelů, zatímco Boosting sekvenčně snižuje vychýlení (Bias) učením z reziduí.":
            st.success("Přesně tak! Bagging = paralelní průměrování (Var reduction), Boosting = sekvenční řetězení na chybách (Bias reduction).")

        st.markdown("---")

        q2 = st.radio(
            "2. Proč algoritmus Random Forest v každém uzlu stromu vybírá split pouze z náhodné podmnožiny příznaků (typicky odmocnina z počtu sloupců)?",
            [
                "Aby se ušetřila paměť na grafické kartě.",
                "Aby se de-korelovaly jednotlivé stromy a dominantní příznak neovládl kořeny všech stromů v lese.",
                "Protože stromy v Scikit-learn neumí pracovat s více než 10 příznaky současně."
            ],
            key="sum_q2"
        )
        if q2 == "Aby se de-korelovaly jednotlivé stromy a dominantní příznak neovládl kořeny všech stromů v lese.":
            st.success("Správně! De-korelace stromů zajišťuje, že průměrování jejich predikcí přinese maximální redukci rozptylu.")

        st.markdown("---")

        q3 = st.radio(
            "3. Co představuje hodnota rezidua v Gradient Boostingu?",
            [
                "Zbytek po celočíselném dělení počtu stromů.",
                "Rozdíl mezi skutečnou hodnotou cílové proměnné a predikcí dosavadního modelu: r = y - F(x).",
                "Náhodný šum přidaný do dat za účelem regularizace."
            ],
            key="sum_q3"
        )
        if q3 == "Rozdíl mezi skutečnou hodnotou cílové proměnné a predikcí dosavadního modelu: r = y - F(x).":
            st.success("Výborně! Každý další slabý strom se učí přímo předpovídat tento rozdíl.")

        st.markdown("---")

        q4 = st.radio(
            "4. Co je to Backpropagation v neuronových sítích a jaký matematický princip využívá?",
            [
                "Je to náhodné přehazování vah za účelem nalezení nejnižší chyby.",
                "Je to proces zpětného šíření chyby z výstupu na vstup pomocí řetízkového pravidla diferenciálního počtu (Chain Rule) pro výpočet gradientů vah.",
                "Je to technika zmenšování obrázků na rozlišení 28x28 pixelů."
            ],
            key="sum_q4"
        )
        if q4 == "Je to proces zpětného šíření chyby z výstupu na vstup pomocí řetízkového pravidla diferenciálního počtu (Chain Rule) pro výpočet gradientů vah.":
            st.success("Přesně! Řetízkové pravidlo umožňuje efektivně spočítat derivaci celkové chyby podle libovolné váhy v síti.")

        st.markdown("---")

        q5 = st.radio(
            "5. Proč se v moderních hlubokých neuronových sítích ve skrytých vrstvách nepoužívá aktivační funkce Sigmoid?",
            [
                "Protože Sigmoid neumí pracovat se zápornými čísly na vstupu.",
                "Kvůli problému mizejícího gradientu (Vanishing Gradient): derivace Sigmoidu je maximálně 0.25, takže gradient v hluboké síti exponenciálně vyprchá k nule.",
                "Protože Keras funkci Sigmoid ve verzi 3 již nepodporuje."
            ],
            key="sum_q5"
        )
        if q5 == "Kvůli problému mizejícího gradientu (Vanishing Gradient): derivace Sigmoidu je maximálně 0.25, takže gradient v hluboké síti exponenciálně vyprchá k nule.":
            st.success("Správně! (0.25)^L způsobí vymizení gradientu. Proto ve skrytých vrstvách vládne ReLU a GELU.")

        st.markdown("---")

        q6 = st.radio(
            "6. Kdy je vhodné na tabulkových datech dát přednost Random Forestu či XGBoostu před neuronovou sítí?",
            [
                "Vždy, protože neuronové sítě neumí počítat s čísly.",
                "Téměř vždy u malých až středních tabulkových datasetů bez prostorové/sekvenční struktury – stromy nevyžadují škálování, jsou imunní vůči outlierům a dosahují špičkových výsledků s minimem tuningu.",
                "Pouze tehdy, když nemáme k dispozici grafickou kartu (GPU)."
            ],
            key="sum_q6"
        )
        if q6 == "Téměř vždy u malých až středních tabulkových datasetů bez prostorové/sekvenční struktury – stromy nevyžadují škálování, jsou imunní vůči outlierům a dosahují špičkových výsledků s minimem tuningu.":
            st.success("Přesně tak! Na tabulkových datech zůstávají stromové ansámbly zlatým standardem průmyslu.")


if __name__ == "__main__":
    render_day_3_summary_view()
