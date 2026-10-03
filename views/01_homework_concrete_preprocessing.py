"""
Homework: Příprava dat pro regresní modely – Pevnost betonu (Concrete Compressive Strength)
=============================================================================================
Tento modul vizualizuje kompletní proces přípravy datové sady pro odhad pevnosti betonu v tlaku:
1. Audit kvality dat (ověření datových typů, chybějících hodnot, detekce a odstranění 25 duplicit).
2. Interaktivní rozdělení proměnných (Plotly histogramy, boxploty, šikmost).
3. Bivariační analýza a závislosti vůči cílové proměnné csMPa s OLS trendlinemi.
4. Interaktivní korelační matice s tooltipy.
5. Porovnání metod škálování (StandardScaler vs. MinMaxScaler).
6. Doménový kontext zrání a složení kompozitu (I-Cheng Yeh, 1998).
"""

import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


@st.cache_data
def load_concrete_preprocessed_df():
    base_dir = Path(__file__).resolve().parent.parent
    csv_path = base_dir / "01_Regression" / "data" / "concrete_data_preprocessed.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


@st.cache_data
def load_concrete_raw_clean_df():
    base_dir = Path(__file__).resolve().parent.parent
    raw_path = base_dir / "01_Regression" / "data" / "concrete_data.csv"
    if raw_path.exists():
        try:
            df = pd.read_csv(raw_path, encoding="latin1")
        except Exception:
            df = pd.read_csv(raw_path, encoding="utf-8")
        df.columns = [
            "cement", "slag", "flyash", "water",
            "superplasticizer", "coarseaggregate", "fineaggregate", "age", "csMPa"
        ]
        return df.drop_duplicates().reset_index(drop=True)
    return None


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
    df_raw = load_concrete_raw_clean_df()
    df_prep = load_concrete_preprocessed_df()

    if not data or df_raw is None:
        st.error("Předpočtená data nebyla nalezena. Spusťte prosím skript `01_Regression/10_homework_concrete_preprocessing.py`.")
        return

    meta = data["metadata"]
    audit = data["quality_audit"]
    stats = data["statistics"]
    corrs = data["correlations"]["with_target"]

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
        "🔍 Audit kvality dat",
        "📊 Rozdělení proměnných (Plotly)",
        "📈 Závislosti na csMPa (Plotly)",
        "🔥 Korelační matice (Plotly)",
        "⚖️ Škálování dat",
        "🏗️ Doménový kontext"
    ])

    # TAB 1: Audit kvality dat
    with tab1:
        st.subheader("1. Audit kvality a integrity laboratorních dat")
        st.markdown(
            "Před trénováním jakéhokoliv modelu je nezbytné prověřit konzistenci dat. "
            "Datová sada obsahuje 8 prediktorů (hmotnosti složek betonové směsi v $\\text{kg/m}^3$ a věk ve dnech) "
            "a 1 cílovou proměnnou (pevnost v tlaku v MPa)."
        )

        col_a1, col_a2 = st.columns([1, 1])
        with col_a1:
            st.markdown("##### 📋 Přehled proměnných a datových typů")
            cols_info = []
            for col in meta["columns"]:
                cols_info.append({
                    "Název": col,
                    "Popis veličiny": meta["column_descriptions"][col],
                    "Datový typ": audit["dtypes"].get(col, "float64"),
                    "Chybějící (NaN)": audit["missing_values"].get(col, 0)
                })
            st.dataframe(pd.DataFrame(cols_info), width="stretch", hide_index=True)

        with col_a2:
            st.markdown("##### ⚠️ Nález a odstranění duplicitních měření")
            st.warning(
                f"Při auditu bylo detekováno **{audit['duplicates_count']} duplicitních záznamů** "
                f"({audit['duplicates_count'] / meta['raw_rows'] * 100:.2f} % celého datasetu). "
                "V laboratorní praxi se často provádí více zkoušek téže šarže betonu současně. "
                "V ML však ponechání duplicit vede k úniku dat mezi trénovací a testovací sadou "
                "(tzv. data leakage) a nadhodnocení přesnosti."
            )
            st.markdown("Ukázka detekovaných duplicitních řádků:")
            st.dataframe(pd.DataFrame(audit["duplicates_sample"]), width="stretch", hide_index=True)

        st.markdown("##### 🔬 Náhled očištěného datasetu (prvních 5 řádků)")
        st.dataframe(pd.DataFrame(data["sample_head_raw"]), width="stretch", hide_index=True)

    # TAB 2: Rozdělení proměnných (Plně interaktivní Plotly)
    with tab2:
        st.subheader("2. Interaktivní rozdělení proměnných a detekce odlehlých hodnot")
        st.markdown(
            "Vyberte proměnnou a prozkoumejte její histogram, hustotu pravděpodobnosti a boxplot s přesnými kvartily."
        )

        selected_var = st.selectbox(
            "Vyberte proměnnou pro detailní analýzu:",
            options=meta["columns"],
            format_func=lambda x: f"{x} – {meta['column_descriptions'][x]}",
            key="eda_var_select"
        )

        var_info = stats[selected_var]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Průměr ± Sm. odchylka", f"{var_info['mean']:.2f} ± {var_info['std']:.2f}")
        c2.metric("Medián [Q25, Q75]", f"{var_info['median']:.2f} [{var_info['q25']:.1f}, {var_info['q75']:.1f}]")
        c3.metric("Šikmost (Skewness)", f"{var_info['skewness']:.2f}",
                  help="> 0 znamená protažení doprava (kladná šikmost), < 0 doleva.")
        c4.metric("Podíl nulových hodnot", f"{var_info['zeros_pct']:.1f} %",
                  help="Řada betonů neobsahuje popílek, strusku nebo superplastifikátor.")

        col_opts1, col_opts2 = st.columns([1, 1])
        with col_opts1:
            nbins = st.slider("Počet intervalů histogramu (bins):", min_value=10, max_value=60, value=30, key="bins_slider")
        with col_opts2:
            show_box = st.checkbox("Zobrazit Boxplot nad histogramem", value=True)

        fig_hist = px.histogram(
            df_raw,
            x=selected_var,
            nbins=nbins,
            marginal="box" if show_box else None,
            title=f"Interaktivní rozdělení: {selected_var} ({meta['column_descriptions'][selected_var]})",
            color_discrete_sequence=["#1f77b4"],
            opacity=0.85
        )
        fig_hist.update_layout(
            bargap=0.05,
            xaxis_title=f"{selected_var} ({meta['column_descriptions'][selected_var]})",
            yaxis_title="Četnost vzorků",
            height=450,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_hist, width="stretch")

        st.markdown("##### 📊 Kompletní přehledová tabulka statistik všech proměnných")
        full_stats_df = pd.DataFrame(stats).T[
            ["description", "mean", "std", "min", "q25", "median", "q75", "max", "skewness", "zeros_pct"]
        ]
        full_stats_df.columns = [
            "Popis", "Průměr", "Sm. odchylka", "Min", "25 %", "Medián", "75 %", "Max", "Šikmost", "Nuly (%)"
        ]
        st.dataframe(full_stats_df, width="stretch")

    # TAB 3: Závislosti na csMPa (Plně interaktivní Plotly scatter)
    with tab3:
        st.subheader("3. Interaktivní bivariační závislosti složek směsi na pevnosti betonu (csMPa)")
        st.markdown(
            "Zkoumejte vztah mezi libovolným prediktorem a výslednou pevností $csMPa$. "
            "Pohybem myši zobrazíte přesné složení každého testovacího vzorku."
        )

        pred_col1, pred_col2 = st.columns([1, 1])
        with pred_col1:
            x_var = st.selectbox(
                "Vyberte prediktor na ose X:",
                options=meta["columns"][:-1],
                format_func=lambda x: f"{x} – {meta['column_descriptions'][x]}",
                key="scatter_x_select"
            )
        with pred_col2:
            color_var = st.selectbox(
                "Barevné kódování bodů podle:",
                options=["age", "cement", "water", "superplasticizer", "slag"],
                format_func=lambda x: f"{x} – {meta['column_descriptions'][x]}",
                key="scatter_color_select"
            )

        # Interaktivní Plotly scatter bez externí závislosti na statsmodels
        fig_scatter = px.scatter(
            df_raw,
            x=x_var,
            y="csMPa",
            color=color_var,
            color_continuous_scale="Plasma",
            title=f"Závislost pevnosti betonu na {x_var} (Barevně: {color_var})",
            labels={
                x_var: f"{x_var} ({meta['column_descriptions'][x_var]})",
                "csMPa": "Pevnost v tlaku (MPa)",
                color_var: color_var
            },
            hover_data={
                "cement": True,
                "water": True,
                "age": True,
                "csMPa": ":.2f"
            }
        )
        # Manuální přidání lineární trendline přes NumPy
        slope, intercept_val = np.polyfit(df_raw[x_var], df_raw["csMPa"], 1)
        x_trend = np.linspace(df_raw[x_var].min(), df_raw[x_var].max(), 100)
        fig_scatter.add_trace(go.Scatter(
            x=x_trend,
            y=slope * x_trend + intercept_val,
            mode="lines",
            name=f"Lineární trend (sklon {slope:+.2f})",
            line=dict(color="red", width=2)
        ))
        fig_scatter.update_layout(height=500, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_scatter, width="stretch")

        r_val = corrs.get(x_var, 0.0)
        st.info(f"**Pearsonův korelační koeficient pro `{x_var}` vs `csMPa`:** **$r = {r_val:+.4f}$**")

        st.markdown("#### 💡 Klíčové postřehy z materiálové technologie:")
        st.markdown(
            """
            1. **Cement ($r = +0.488$):** Nejsilnější pozitivní faktor. Více cementu poskytuje více vazného hydratačního gelu C-S-H.
            2. **Voda ($r = -0.270$):** Záporná závislost. Přebytečná voda vytváří v betonu kapilární póry po vyschnutí, čímž snižuje nosný průřez kompozitu.
            3. **Superplastifikátor ($r = +0.344$):** Umožňuje dramaticky snížit množství vody při zachování tekutosti (tzv. samohutnící a vysokopevnostní betony).
            4. **Stáří betonu ($age$, $r = +0.337$):** Nelineární (logaritmický) charakter – pevnost roste prudce v prvních 28 dnech, poté růst přechází do plató.
            """
        )

    # TAB 4: Korelační matice (Plně interaktivní Plotly Heatmap)
    with tab4:
        st.subheader("4. Interaktivní korelační matice a multikolinearita")
        st.markdown("Pearsonův korelační koeficient $r$ měří sílu a směr lineární závislosti.")

        corr_matrix = df_raw.corr()

        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Korelační matice vstupních proměnných a cílové pevnosti betonu (Pearson r)",
            aspect="auto"
        )
        fig_corr.update_layout(height=520, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_corr, width="stretch")

        col_c1, col_c2 = st.columns([1, 1])
        with col_c1:
            st.markdown("##### 🏆 Pořadí korelací s pevností betonu (csMPa)")
            corr_target_df = pd.DataFrame([
                {
                    "Prediktor": k,
                    "Popis": meta["column_descriptions"].get(k, k),
                    "Pearson r": f"{v:+.4f}",
                    "Vazba": "Kladná 🟢" if v > 0 else "Záporná 🔴"
                }
                for k, v in sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)
                if k != "csMPa"
            ])
            st.dataframe(corr_target_df, width="stretch", hide_index=True)

        with col_c2:
            st.markdown("##### ⚠️ Klíčová multikolinearita v praxi")
            st.warning(
                "**Pozor na multikolinearitu:** Korelace mezi `water` a `superplasticizer` je **-0.645**! "
                "Superplastifikátory jsou chemické přísady snižující povrchové napětí a umožňující rozptýlení cementových zrn. "
                "Receptury s vysokým obsahem superplastifikátoru záměrně obsahují výrazně méně vody. "
                "Lineární modely (OLS) mohou trpět inflací rozptylu koeficientů (VIF)."
            )

    # TAB 5: Škálování dat (Plně interaktivní Plotly)
    with tab5:
        st.subheader("5. Škálování dat: Standardizace (StandardScaler) vs. Normalizace (MinMaxScaler)")
        st.markdown(
            "Prozkoumejte, jak se změní distribuce libovolné proměnné před a po aplikaci standardizace / normalizace."
        )

        st.markdown(
            r"""
            | Metoda | Vzorec transformace | Výsledný rozsah | Vlastnosti & Využití |
            | :--- | :---: | :---: | :--- |
            | **StandardScaler (z-score)** | $z = \frac{x - \mu}{\sigma}$ | $\mu = 0, \sigma = 1$ | Zachovává proporce odlehlých hodnot, ideální pro penalizovanou regresi a normální chyby. |
            | **MinMaxScaler (min-max)** | $x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$ | $[0, 1]$ | Vhodné pro algoritmy citlivé na ohraničené rozsahy (k-NN, neuronové sítě s sigmoidou), ale citlivé na extrémy. |
            """
        )

        scale_feat = st.selectbox(
            "Vyberte proměnnou pro srovnání škálování:",
            options=meta["columns"][:-1],
            format_func=lambda x: f"{x} – {meta['column_descriptions'][x]}",
            key="scale_feat_select"
        )

        raw_vals = df_raw[scale_feat]
        scaled_z = (raw_vals - raw_vals.mean()) / raw_vals.std()
        scaled_mm = (raw_vals - raw_vals.min()) / (raw_vals.max() - raw_vals.min())

        fig_scale = go.Figure()
        fig_scale.add_trace(go.Histogram(x=scaled_z, name="StandardScaler (Z-Score, μ=0, σ=1)", opacity=0.6, marker_color="#2ca02c"))
        fig_scale.add_trace(go.Histogram(x=scaled_mm, name="MinMaxScaler [0, 1]", opacity=0.6, marker_color="#ff7f0e"))
        fig_scale.update_layout(
            barmode="overlay",
            title=f"Porovnání rozdělení po škálování pro: {scale_feat}",
            xaxis_title="Škálovaná hodnota",
            yaxis_title="Četnost",
            height=400,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_scale, width="stretch")

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
