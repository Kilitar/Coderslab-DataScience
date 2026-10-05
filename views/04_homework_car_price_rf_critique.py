"""
Domácí úkol (Session 2): Expertní analýza & Kritika modelu Random Forest (Ceny aut)
===================================================================================
Dataset: data/car_data.csv (19 237 vozidel)
Model: RandomForestRegressor
Precomputed: 04_Homework/data/car_price_rf_precomputed.json
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.title("🔬 Expertní analýza & Diagnostika: Random Forest (Ceny aut)")
st.caption(
    "Kritické zhodnocení skrytých defektů v datech (Excel Date Bug, extrémní odlehlé hodnoty za 26 milionů USD), "
    "analýza důležitosti příznaků a doporučení pro produkční nasazení v automotive e-commerce."
)

base_dir = Path(__file__).resolve().parent.parent
json_path = base_dir / "04_Homework" / "data" / "car_price_rf_precomputed.json"


@st.cache_data
def load_car_stats():
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


stats = load_car_stats()
metrics = stats["test_metrics"] if stats else {}
feat_imp = stats["feature_importances"] if stats else []

# Horní KPI karty
c1, c2, c3, c4 = st.columns(4)
c1.metric("Klíčový driver ceny", "Rok výroby (prod_year)", delta="Důležitost 20.5 %")
c2.metric("Motorizace & Prestiž", "Objem motoru + Značka", delta="Důležitost > 27 %")
c3.metric("Ošetření anomálie", "Odstranění 26M USD chyby", delta="R² zvýšeno z -19.1 na +0.65")
c4.metric("Kardinalita modelu", "1 580 kategorií vyřazeno", delta="Prevence 1 689 sloupců")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "🌲 1. Důležitost příznaků (Feature Importance)",
    "💸 2. Anomálie za 26M USD & Excel Date Bug",
    "🧩 3. Past vysoké kardinality & Overfitting",
    "🏭 4. Produkční doporučení & Best Practices"
])

# ==============================================================================
# TAB 1: FEATURE IMPORTANCE
# ==============================================================================
with tab1:
    st.subheader("Které technické a tržní parametry nejvíce hýbou cenou vozidla?")

    st.markdown(
        """
        Náhodný les měří významnost každého příznaku pomocí **MDI (Mean Decrease in Impurity)** – 
        tedy o kolik každý příznak v průměru snížil rozptyl čtvercové chyby (MSE) napříč všemi 100 stromy.
        """
    )

    if feat_imp:
        f_df = pd.DataFrame(feat_imp).head(15)
        # České názvy pro srozumitelnost
        clean_labels = {
            "prod_year": "Rok výroby (prod_year)",
            "engine_volume": "Objem motoru (engine_volume)",
            "manufacturer_LAMBORGHINI": "Značka: LAMBORGHINI (Supersport)",
            "airbags": "Počet airbagů (airbags)",
            "mileage": "Stav tachometru (mileage)",
            "cylinders": "Počet válců motoru (cylinders)",
            "gear_box_type_Tiptronic": "Převodovka: Tiptronic",
            "gear_box_type_Manual": "Převodovka: Manuální",
            "drive_wheels_Front": "Pohon: Přední náprava",
            "manufacturer_MERCEDES-BENZ": "Značka: MERCEDES-BENZ",
            "fuel_type_Hybrid": "Palivo: Hybridní pohon",
            "category_Jeep": "Kategorie: SUV / Jeep",
            "manufacturer_LAND ROVER": "Značka: LAND ROVER",
            "manufacturer_PORSCHE": "Značka: PORSCHE",
            "wheel_Right-hand drive": "Řízení vpravo"
        }
        f_df["clean_name"] = f_df["feature"].apply(lambda x: clean_labels.get(x, x))
        f_df = f_df.sort_values(by="importance", ascending=True)

        fig_imp = px.bar(
            f_df,
            x="importance",
            y="clean_name",
            orientation="h",
            labels={"importance": "Gini Importance (MDI)", "clean_name": "Příznak"},
            color="importance",
            color_continuous_scale="Viridis",
            title="Top 15 nejdůležitějších příznaků pro odhad ceny automobilu"
        )
        fig_imp.update_layout(height=480, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_imp, width="stretch")

    st.markdown("##### 💡 Analytické postřehy:")
    st.markdown(
        """
        1. **Stáří vozu je dominantní (`prod_year` = 20.5 %):** Pokles tržní hodnoty (amortizace) 
           není lineární, ale vykazuje exponenciální pokles v prvních 3–5 letech, který rozhodovací stromy 
           dokáží modelovat mnohem přesněji než klasická OLS lineární regrese.
        2. **Objem motoru (`engine_volume` = 14.4 %) a počet válců:** Silné motory (V6, V8, 3.0 l+) 
           korelují s prémiovým a luxusním segmentem trhu.
        3. **Prémiové značky (Lamborghini, Porsche, Mercedes-Benz, Land Rover):** Samotná binární indikace 
           supersportovní značky jako Lamborghini má enormní vliv na rozdělení stromů do uzlů s cenami 200 000+ USD.
        4. **Bezpečnost (`airbags` = 9.2 %):** Počet airbagů často slouží stromům jako nepřímý ukazatel 
           moderní platformy a vyšší výbavové linie vozidla.
        """
    )


# ==============================================================================
# TAB 2: ANOMÁLIE A EXCEL DATE BUG
# ==============================================================================
with tab2:
    st.subheader("Skryté nástrahy reálných dat: Proč nestačí jen spustit model")

    st.markdown(
        """
        Při přípravě dat jsme narazili na dva zásadní fenomény, se kterými se datoví vědci v praxi 
        setkávají velmi často: **tichá korupce dat Excelem** a **zničující vliv jediného odlehlého pozorování na MSE**.
        """
    )

    col_bug1, col_bug2 = st.columns([1, 1])

    with col_bug1:
        st.markdown("##### 📅 1. The Excel Date Auto-Format Bug:")
        st.markdown(
            """
            Ve sloupci `doors` se v původním souboru nacházely hodnoty:
            - `'04-May'` (18 323 řádků)
            - `'02-Mar'` (776 řádků)
            - `'>5'` (128 řádků)

            **Příčina vzniku:**  
            Při exportu z databáze obsahovala data textové intervaly počtu dveří: `"4-5"` a `"2-3"`. 
            Pokud někdo v řetězci otevřel CSV soubor v programu **Microsoft Excel**, Excel automaticky a bez varování 
            přetypoval text `"4-5"` na datum **4. května** (`04-May`) a `"2-3"` na **2. března** (`02-Mar`).

            **Dopad na model bez nápravy:**  
            Pokud bychom tyto sloupce neošetřili, One-Hot Encoder by vygeneroval náhodné sloupce `doors_04-May`, 
            čímž by se ztratila veškerá sémantická vazba na geometrii karoserie a rodinnou praktičnost. 
            Opravili jsme mapováním: `{'04-May': '4-5', '02-Mar': '2-3'}`.
            """
        )

    with col_bug2:
        st.markdown("##### 💸 2. Anomálie za 26 milionů USD (Opel Combo):")
        st.markdown(
            """
            V řádku indexu `16983` se v datasetu nacházel běžný užitkový vůz:  
            **OPEL Combo z roku 1999 s motorem 1.7 D a cenou 26 307 500 USD!**  
            Jedná se o evidentní překlep uživatele nebo chybu scraperu inzertního portálu.

            **Matematická katastrofa regresního stromu:**  
            Regresní strom minimalizuje rozptyl v listech a predikuje aritmetický průměr listu:
            """
        )
        st.latex(r"\hat{y}_{\text{leaf}} = \frac{1}{N} \sum_{i=1}^N y_i")
        st.markdown(
            """
            Pokud se v listu ocitne tato hodnota, průměrná predikovaná cena listu vystřelí do milionů USD. 
            Při testování pak model pro několik běžných aut předpověděl cenu **3,3 milionu USD**!
            """
        )

    st.markdown("---")
    st.markdown("##### ⚖️ Srovnání výkonu modelu: S anomálií vs. Očištěný model:")

    comp_df = pd.DataFrame([
        {
            "Stav datasetu": "⚠️ Původní data (s 26M USD chybou)",
            "Testovací R²": "-19.1369 (Záporný!)",
            "Testovací RMSE": "90 035 USD",
            "Testovací MAE": "8 592 USD",
            "Diagnóza": "Model je horší než prostý průměr; stromy predikují miliony."
        },
        {
            "Stav datasetu": "✅ Očištěná data (anomálie vyřazena)",
            "Testovací R²": "+0.6492 (64.9 % vysvětleno)",
            "Testovací RMSE": "11 867 USD",
            "Testovací MAE": "4 979 USD",
            "Diagnóza": "Realistické odhady; stabilní listové průměry napříč ansámblem."
        }
    ])
    st.dataframe(comp_df, hide_index=True, width="stretch")


# ==============================================================================
# TAB 3: VYSOKÁ KARDINALITA
# ==============================================================================
with tab3:
    st.subheader("Past vysoké kardinality: Proč byl vyřazen sloupec `model`?")

    st.markdown(
        """
        Zadání obsahovalo instrukci: *„Remove unnecessary columns from the data and correct data in the columns storing categorical data (only where necessary).“*  
        Kromě technických sloupců `Unnamed: 0` a `id` je klíčovým rozhodnutím vyřazení sloupce **`model`**.
        """
    )

    kard_c1, kard_c2 = st.columns([1, 1])

    with kard_c1:
        st.markdown("##### 📊 Porovnání kardinality kategorických sloupců:")
        kard_data = pd.DataFrame([
            {"Sloupec": "model (Model vozu)", "Počet unikátních hodnot": 1580, "Dopad při One-Hot": "1 580 řídkých sloupců", "Rozhodnutí": "❌ Vyřazeno jako zbytečné / škodlivé"},
            {"Sloupec": "manufacturer (Značka)", "Počet unikátních hodnot": 65, "Dopad při One-Hot": "65 binárních příznaků", "Rozhodnutí": "✅ Ponecháno (robustní agregace)"},
            {"Sloupec": "color (Barva)", "Počet unikátních hodnot": 16, "Dopad při One-Hot": "16 binárních příznaků", "Rozhodnutí": "✅ Ponecháno"},
            {"Sloupec": "category (Typ karoserie)", "Počet unikátních hodnot": 11, "Dopad při One-Hot": "11 binárních příznaků", "Rozhodnutí": "✅ Ponecháno"},
            {"Sloupec": "fuel_type (Palivo)", "Počet unikátních hodnot": 7, "Dopad při One-Hot": "7 binárních příznaků", "Rozhodnutí": "✅ Ponecháno"},
            {"Sloupec": "gear_box_type (Převodovka)", "Počet unikátních hodnot": 4, "Dopad při One-Hot": "4 binární příznaky", "Rozhodnutí": "✅ Ponecháno"},
            {"Sloupec": "drive_wheels (Pohon)", "Počet unikátních hodnot": 3, "Dopad při One-Hot": "3 binární příznaky", "Rozhodnutí": "✅ Ponecháno"},
            {"Sloupec": "doors (Dveře)", "Počet unikátních hodnot": 3, "Dopad při One-Hot": "3 binární příznaky", "Rozhodnutí": "✅ Ponecháno (po opravě)"},
            {"Sloupec": "wheel (Pozice volantu)", "Počet unikátních hodnot": 2, "Dopad při One-Hot": "1 binární příznak", "Rozhodnutí": "✅ Ponecháno"}
        ])
        st.dataframe(kard_data, hide_index=True, width="stretch")

    with kard_c2:
        st.markdown("##### ⚠️ Proč je One-Hot s 1 580 sloupci pro náhodný les problém?")
        st.markdown(
            """
            1. **Prokletí dimenzionality (Curse of Dimensionality):**  
               Většina z 1 580 modelů aut se v datasetu vyskytuje pouze 1× nebo 2×. Tyto sloupce obsahují 99.99 % nul.
            2. **Náhodný výběr příznaků ve stromech (`max_features`):**  
               Při každém dělení uzlu náhodný les náhodně vybere $\\sqrt{p}$ příznaků. Pokud je v matici 1 580 sloupců modelů 
               a pouze 10 skutečně informativních technických proměnných, algoritmus při většině splitů vybírá pouze 
               šumové binární indikátory vzácných modelů!
            3. **Paměť a rychlost:**  
               Trénovací matice bez modelu má 109 sloupců a trénování trvá **0.4 s**, s modelem má 1 689 sloupců 
               a vyžaduje 15× více paměti bez měřitelného zlepšení na testovací sadě.
            """
        )


# ==============================================================================
# TAB 4: PRODUKČNÍ DOPORUČENÍ
# ==============================================================================
with tab4:
    st.subheader("💡 Doporučení pro produkční nasazení cenových modelů v automotive")

    st.markdown(
        """
        Pokud by tento model měl být nasazen do reálné produkce (např. pro automatický výkup vozidel 
        v AAA Auto, Carvago nebo AutoESA), doporučujeme následující vylepšení:
        """
    )

    rec1, rec2 = st.columns([1, 1])

    with rec1:
        st.markdown("##### 1. Logaritmická transformace cíle (Target Log-Transform):")
        st.markdown(
            """
            Ceny automobilů mají silně pravostranně zešikmenou distribuci (log-normální charakter).  
            Místo trénování na absolutní hodnotě $y$ je standardem trénovat na:
            """
        )
        st.latex(r"z = \ln(\text{price})")
        st.markdown(
            """
            - **Výhoda:** Minimalizace MSE na logaritmovaných cenách odpovídá minimalizaci **relativní procentuální chyby (MAPE)**. 
              Chyba 2 000 USD u auta za 10 000 USD (20 %) má stejnou váhu jako chyba 20 000 USD u auta za 100 000 USD (20 %).
            """
        )

        st.markdown("##### 2. Target Encoding s regularizací pro `model`:")
        st.markdown(
            """
            Místo úplného vyřazení sloupce `model` lze použít **Target Encoding** s aditivním vyhlazováním (M-estimate smoothing), 
            který nahradí název modelu průměrnou historickou cenou s vážením k celkovému průměru značky.
            """
        )

    with rec2:
        st.markdown("##### 3. Kvantilový náhodný les (Quantile Regression Forests):")
        st.markdown(
            """
            Běžný náhodný les vrací pouze bodový odhad (střední hodnotu). Pro výkupní ocenění je kritické znát:
            - **10. percentil:** Bezpečná konzervativní výkupní cena (garance rychlého prodeje).
            - **50. percentil (Medián):** Férová tržní cena.
            - **90. percentil:** Maximální optimistická nabídková cena.
            """
        )

        st.markdown("##### 4. Validace datových kontraktů (Data Quality Guardrails):")
        st.markdown(
            """
            - Nasazení validačních schémat (např. balíček `pandera` nebo `Great Expectations`), 
              která automaticky detekují Excel date-patterny (`04-May`) a zastaví pipeline dříve, 
              než poškozená data dorazí do inference.
            - Automatický clipping cen pod 500 USD a nad 500 000 USD.
            """
        )
