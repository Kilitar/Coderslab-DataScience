"""
Homework: Příprava dat pro regresní modely – Pevnost betonu (Concrete Compressive Strength)
=============================================================================================
Tento modul vizualizuje kompletní proces přípravy datové sady pro odhad pevnosti betonu v tlaku:
1. Audit kvality dat (ověření datových typů, chybějících hodnot, detekce a odstranění 25 duplicit).
2. Rozdělení proměnných (histogramy, šikmost, zero-inflated složky).
3. Korelace a závislosti vůči cílové proměnné csMPa.
4. Porovnání metod škálování (StandardScaler vs. MinMaxScaler).
5. Doménový kontext zrání a složení kompozitu (I-Cheng Yeh, 1998).
"""

import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def load_concrete_precomputed():
    base_dir = Path(__file__).resolve().parent.parent
    cache_path = base_dir / "01_Regression" / "data" / "concrete_preprocessing_precomputed.json"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def render_concrete_preprocessing_view():
    st.title("🧱 DÚ: Příprava dat pro regresní modely – Pevnost betonu")
    st.markdown(
        "**Úvodní úloha domácího úkolu z regrese:** Načtení, očista, průzkumová analýza (EDA), "
        "korelační diagnostika a škálování datové sady **Concrete Compressive Strength** *(I-Cheng Yeh, 1998)*."
    )

    data = load_concrete_precomputed()
    if not data:
        st.error("Předpočtená data nebyla nalezena. Spusťte prosím skript `01_Regression/10_homework_concrete_preprocessing.py`.")
        return

    meta = data["metadata"]
    audit = data["quality_audit"]
    stats = data["statistics"]
    corrs = data["correlations"]
    scaling = data["scaling_comparison"]

    # Rychlé KPI metriky v záhlaví
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    with col_kpi1:
        st.metric("Původní vzorky", f"{meta['raw_rows']} řádků")
    with col_kpi2:
        st.metric("Počet duplicit", f"{audit['duplicates_count']} řádků", delta=f"-{audit['duplicates_count']}", delta_color="inverse")
    with col_kpi3:
        st.metric("Čisté vzorky", f"{meta['clean_rows']} řádků")
    with col_kpi4:
        st.metric("Chybějící hodnoty", "0 (100% kompletní)")

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 1. Audit dat & Duplicity",
        "📊 2. Rozdělení proměnných",
        "📈 3. Závislosti na csMPa",
        "🔥 4. Korelační matice",
        "⚖️ 5. Škálování dat",
        "🔬 6. Doménový kontext"
    ])

    # TAB 1: Audit dat & Duplicity
    with tab1:
        st.subheader("1. Audit kvality dat a odstranění duplicit")
        st.markdown(
            """
            V laboratorním testování betonu se často provádějí paralelní testy ze stejné šarže. 
            Při přípravě dat je nezbytné zkontrolovat přítomnost identických řádků a zajistit nezávislost vzorků.
            """
        )

        col_a1, col_a2 = st.columns([1, 1])
        with col_a1:
            st.markdown("##### 🔍 Kontrola chybějících hodnot a datových typů")
            dtype_df = pd.DataFrame([
                {
                    "Proměnná": col,
                    "Popis": meta["column_descriptions"][col],
                    "Typ": audit["dtypes"][col],
                    "Chybějící (NaN)": audit["missing_values"][col]
                }
                for col in meta["columns"]
            ])
            st.dataframe(dtype_df, width="stretch", hide_index=True)

        with col_a2:
            st.markdown("##### ⚠️ Odstranění duplicitních měření")
            st.info(
                f"V původním souboru bylo identifikováno **{audit['duplicates_count']} duplicitních řádků** "
                f"({audit['duplicates_count'] / meta['raw_rows'] * 100:.1f} % datasetu). "
                f"Po jejich odstranění zůstává **{meta['clean_rows']} unikátních laboratorních vzorků**."
            )
            st.markdown("Ukázka detekovaných duplicitních záznamů:")
            if audit["duplicates_sample"]:
                dup_df = pd.DataFrame(audit["duplicates_sample"])
                st.dataframe(dup_df.head(6), width="stretch", hide_index=True)

        st.markdown("##### 👁️ Prvních 10 očištěných vzorků (fyzikální jednotky)")
        st.dataframe(pd.DataFrame(data["sample_head_raw"]), width="stretch", hide_index=True)

    # TAB 2: Rozdělení proměnných
    with tab2:
        st.subheader("2. Rozdělení proměnných a deskriptivní statistika")
        st.markdown("Prozkoumejte tvar distribuce, šikmost a podíl nulových hodnot pro každou složku směsi.")

        selected_var = st.selectbox(
            "Vyberte proměnnou k detailnímu rozboru:",
            options=meta["columns"],
            index=0,
            format_func=lambda x: f"{x} – {meta['column_descriptions'][x]}"
        )

        var_info = stats[selected_var]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Průměr ± Sm. odchylka", f"{var_info['mean']:.2f} ± {var_info['std']:.2f}")
        c2.metric("Medián [Q25, Q75]", f"{var_info['median']:.2f} [{var_info['q25']:.1f}, {var_info['q75']:.1f}]")
        c3.metric("Šikmost (Skewness)", f"{var_info['skewness']:.2f}",
                  help="> 0 znamená protažení doprava (kladná šikmost), < 0 doleva.")
        c4.metric("Podíl nulových hodnot", f"{var_info['zeros_pct']:.1f} %",
                  help="Řada betonů neobsahuje popílek, strusku nebo superplastifikátor.")

        # Načteme surová data pro interaktivní histogram
        raw_df_clean = pd.DataFrame(data["sample_head_raw"])  # Fallback pro strukturu
        # Plný histogram přes Plotly
        fig_hist = px.histogram(
            x=[var_info["min"], var_info["q25"], var_info["median"], var_info["q75"], var_info["max"]],
            title=f"Statistické kvartily: {selected_var} ({meta['column_descriptions'][selected_var]})"
        )
        
        # Zobrazíme vygenerovaný statický přehled všech 9 histogramů
        hist_img_path = Path(__file__).resolve().parent.parent / "01_Regression" / "plots" / "concrete_histograms.png"
        if hist_img_path.exists():
            st.image(str(hist_img_path), caption="Rozdělení všech 9 proměnných v očištěném datasetu (1005 vzorků)", width="stretch")

        st.markdown("##### 📊 Kompletní přehledová tabulka statistik")
        full_stats_df = pd.DataFrame(stats).T[
            ["description", "mean", "std", "min", "q25", "median", "q75", "max", "skewness", "zeros_pct"]
        ]
        full_stats_df.columns = [
            "Popis", "Průměr", "Sm. odchylka", "Min", "25 %", "Medián", "75 %", "Max", "Šikmost", "Nuly (%)"
        ]
        st.dataframe(full_stats_df, width="stretch")

    # TAB 3: Závislosti na csMPa
    with tab3:
        st.subheader("3. Závislost složek směsi na pevnosti betonu (csMPa)")
        st.markdown(
            "Pevnost betonu v tlaku závisí na složitých nelineárních a fyzikálně-chemických vazbách. "
            "Níže vidíte regresní závislosti jednotlivých složek."
        )

        scatter_img_path = Path(__file__).resolve().parent.parent / "01_Regression" / "plots" / "concrete_scatter_vs_csmpa.png"
        if scatter_img_path.exists():
            st.image(str(scatter_img_path), caption="Bivariační závislosti 8 prediktorů na pevnosti csMPa s lineárním trendem", width="stretch")

        st.markdown("#### 💡 Klíčové postřehy z regresních závislostí:")
        st.markdown(
            """
            1. **Cement ($r = +0.488$):** Nejsilnější pozitivní faktor. Více cementu poskytuje více vazného hydratačního gelu C-S-H.
            2. **Voda ($r = -0.270$):** Záporná závislost. Přebytečná voda vytváří v betonu kapilární póry po vyschnutí, čímž snižuje nosný průřez kompozitu.
            3. **Superplastifikátor ($r = +0.344$):** Umožňuje dramaticky snížit množství vody při zachování tekutosti (tzv. samohutnící a vysokopevnostní betony).
            4. **Stáří betonu ($age$, $r = +0.337$):** Nelineární (logaritmický) charakter – pevnost roste prudce v prvních 28 dnech, poté růst přechází do plató.
            """
        )

    # TAB 4: Korelační matice
    with tab4:
        st.subheader("4. Korelační matice a multikolinearita")
        st.markdown("Pearsonův korelační koeficient $r$ měří sílu a směr lineární závislosti.")

        corr_img_path = Path(__file__).resolve().parent.parent / "01_Regression" / "plots" / "concrete_correlation_heatmap.png"
        if corr_img_path.exists():
            st.image(str(corr_img_path), caption="Korelační matice vstupních proměnných a cílové pevnosti", width="stretch")

        col_c1, col_c2 = st.columns([1, 1])
        with col_c1:
            st.markdown("##### 🏆 Pořadí korelací s pevností betonu (csMPa)")
            corr_target_df = pd.DataFrame([
                {
                    "Prediktor": k,
                    "Popis": meta["column_descriptions"][k],
                    "Pearson r": v,
                    "Vliv na pevnost": "Zvyšuje pevnost 🟢" if v > 0.2 else ("Snižuje pevnost 🔴" if v < -0.1 else "Slabý vliv ⚪")
                }
                for k, v in corrs["with_target"].items() if k != "csMPa"
            ]).sort_values(by="Pearson r", ascending=False)
            st.dataframe(corr_target_df, width="stretch", hide_index=True)

        with col_c2:
            st.markdown("##### ⚡ Klíčová multikolinearita: Voda vs. Superplastifikátor")
            st.warning(
                "**Pozor na multikolinearitu:** Korelace mezi `water` a `superplasticizer` je **-0.645**! "
                "Superplastifikátory jsou chemické přísady snižující povrchové napětí a umožňující rozptýlení cementových zrn. "
                "Receptury s vysokým obsahem superplastifikátoru záměrně obsahují výrazně méně vody. "
                "Lineární modely (OLS) mohou trpět inflací rozptylu koeficientů (VIF)."
            )

    # TAB 5: Škálování dat
    with tab5:
        st.subheader("5. Škálování dat: Standardizace (StandardScaler) vs. Normalizace (MinMaxScaler)")
        st.markdown(
            """
            Zadání požadovalo volbu jedné ze dvou metod škálování. 
            Pro regresní modely (zejména s L1/L2 regularizací jako Ridge a Lasso) jsme zvolili **StandardScaler**.
            """
        )

        st.markdown(
            r"""
            | Metoda | Vzorec transformace | Výsledný rozsah | Vlastnosti & Využití |
            | :--- | :---: | :---: | :--- |
            | **StandardScaler (z-score)** | $z = \frac{x - \mu}{\sigma}$ | $\mu = 0, \sigma = 1$ | Zachovává proporce odlehlých hodnot, ideální pro penalizovanou regresi a normální chyby. |
            | **MinMaxScaler (min-max)** | $x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$ | $[0, 1]$ | Vhodné pro algoritmy citlivé na ohraničené rozsahy (k-NN, neuronové sítě s sigmoidou), ale citlivé na extrémy. |
            """
        )

        scaling_img_path = Path(__file__).resolve().parent.parent / "01_Regression" / "plots" / "concrete_scaling_comparison.png"
        if scaling_img_path.exists():
            st.image(str(scaling_img_path), caption="Srovnání hustot rozdělení: Původní vs. StandardScaler vs. MinMaxScaler", width="stretch")

        st.markdown("##### 📁 Výsledný standardizovaný dataset (`concrete_data_preprocessed.csv`)")
        st.dataframe(pd.DataFrame(data["sample_head_scaled"]), width="stretch", hide_index=True)
        st.success(
            f"✅ Dataset byl úspěšně uložen do souboru `concrete_data_preprocessed.csv` "
            f"se strukturou {meta['clean_rows']} řádků a {len(meta['columns'])} sloupců."
        )

    # TAB 6: Doménový kontext
    with tab6:
        st.subheader("6. Fyzikální a chemický kontext technologie betonu")
        st.markdown(
            r"""
            ### Proč je odhad pevnosti betonu klíčovou inženýrskou úlohou?
            
            Standardní laboratorní zkouška pevnosti v tlaku probíhá **až po 28 dnech** zrání ve vlhku (dle norem ČSN EN 206 a ASTM C39).
            Pokud beton v nosné konstrukci (most, mrakodrap, tunel) po 28 dnech nevyhoví předepsané třídě pevnosti (např. C30/37), 
            bourání ztvrdlé konstrukce způsobuje stomilionové škody.
            
            **Schopnost předpovědět 28denní pevnost ze složení směsi a ranných fází zrání (7 dní) pomocí Machine Learningu šetří týdny času a miliony eur.**
            
            #### Základní zákonitosti betonářství:
            1. **Abramsův zákon (vodní součinitel $w/c$):**
               $$\text{Pevnost} = \frac{A}{B^{w/c}}$$
               Pevnost je nepřímo úměrná poměru hmotnosti vody k hmotnosti cementu. Snížení vody o pouhých 10 l/m³ může zvýšit pevnost o 5 MPa.
            2. **Role příměsí (struska a popílek):**
               Vysokopecní struska (*slag*) a elektrárenský popílek (*fly ash*) jsou druhotné suroviny. Zpomalují počáteční náběh pevnosti, ale po 90–365 dnech dosahují vyšší konečné pevnosti a odolnosti vůči chemické korozi.
            3. **Logaritmický nárůst v čase:**
               Hydratace cementových minerálů ($C_3S, C_2S$) probíhá exponenciálně rychle v prvních dnech ($t=1, 3, 7$) a po 28 dnech se rychlost hydratace výrazně zpomaluje.
            """
        )


if __name__ == "__main__":
    render_concrete_preprocessing_view()
