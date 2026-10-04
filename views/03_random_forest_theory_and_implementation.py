"""
Den 3: Náhodný les (Random Forest) – Teorie a implementace v Scikit-learn
========================================================================
Interaktivní výukový modul pro 2. a 3. část teorie Dne 3:
1. Teoretický rozbor algoritmu Random Forest: Dvojitá náhodnost (Feature Bagging), redukce rozptylu, OOB.
2. Kompletní Scikit-learn API: Konstruktory, parametry, atributy a metody (RandomForestClassifier & Regressor).
3. Benchmark ze slajdů: Realitní data z Melbourne (DecisionTreeRegressor vs. RandomForestRegressor).
4. Expertní kritické zhodnocení: MDI bias, mýtus chybějících dat v scikit-learn, extrapolace a prostorový únik.
5. Moderní ML & AI rozšíření (Stav k 10/2026): TreeSHAP, cuML GPU akcelerace, ExtraTrees, HistGradientBoosting.
6. Interaktivní vědomostní kvíz.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def render_random_forest_theory_view():
    st.title("🌲 Den 3: Náhodný les (Random Forest) – Teorie & Implementace")
    st.markdown(
        r"""
        **Aplikace Baggingu v praxi:** Jak propojit desítky či stovky rozhodovacích stromů do jednoho 
        mimořádně stabilního ansámblu pomocí **dvojité náhodnosti** (*Bootstrapping* + *Feature Subspaces*). 
        Kompletní rozbor Scikit-learn rozhraní, benchmark na realitních datech z Melbourne a expertní diagnostika.
        """
    )

    # Horní KPI karty
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Dvojitá náhodnost", "Řádky + Sloupce", delta="Dekorelace stromů (ρ → min)")
    with k2:
        st.metric("Melbourne Housing pokles MAE", "-104 879 AUD", delta="-27.8 % proti stromu")
    with k3:
        st.metric("Out-Of-Bag (OOB)", "≈ 36.8 % dat", delta="Validace bez train_test_split")
    with k4:
        st.metric("Scikit-learn paralelizace", "n_jobs = -1", delta="100 % vytížení CPU jader")

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🌲 Teorie & Dvojitá náhodnost",
        "🛠️ Scikit-learn API & Parametry",
        "🏡 Melbourne Housing Benchmark",
        "🔬 Expertní kritika & Úskalí",
        "🚀 SOTA Rozšíření 10/2026",
        "📝 Vědomostní kvíz"
    ])

    # =========================================================================
    # TAB 1: TEORIE & DVOJITÁ NÁHODNOST
    # =========================================================================
    with tab1:
        st.subheader("1. Proč Random Forest? Princip dvojité náhodnosti")
        st.markdown(
            r"""
            Samotný Bagging aplikovaný na stromy náhodně sampluje pouze trénovací pozorování (řádky) s vracením. 
            Pokud je však v datech jeden dominantní příznak, **všechny stromy by ho vybraly jako kořenový uzel**. 
            Tím by vznikly vzájemně silně korelované stromy ($\rho > 0$) a síla ansámblu by byla limitována.
            
            **Leo Breiman (2001)** proto navrhl **Random Forest**, který zavádí **dvojitou náhodnost**:
            """
        )

        col_rand1, col_rand2 = st.columns(2)
        with col_rand1:
            st.info(
                r"""
                #### 1. Náhodný výběr vzorků (Bootstrapping)
                - Každý strom trénuje na náhodném výběru $N$ pozorování s vracením (*with replacement*).
                - Cca **63.2 % unikátních řádků** je v tréninku (In-Bag).
                - Cca **36.8 % řádků** zůstává nevyužito jako **Out-Of-Bag (OOB)** validační sada.
                """
            )
        with col_rand2:
            st.success(
                r"""
                #### 2. Náhodný výběr příznaků (Feature Bagging)
                - Při **každém štěpení uzlu** se neprohledávají všechny příznaky $p$.
                - Náhodně se vylosuje pouze malá podmnožina $m$ příznaků ($m < p$).
                - Typicky $m = \sqrt{p}$ pro klasifikaci a $m = p/3$ pro regresi.
                - **Výsledek:** Stromy jsou od sebe maximálně odlišné (nízká korelace $\rho$).
                """
            )

        st.markdown("---")
        st.subheader("📊 Příklady fungování ze slajdů kurzu")

        col_ex_c, col_ex_r = st.columns(2)
        with col_ex_c:
            st.markdown("##### 🏖️ A. Klasifikace: Výběr destinace dovolené (Slajd 9)")
            st.caption("5 rozhodovacích stromů trénovaných na různých vzorcích i příznacích:")
            
            df_ex_c = pd.DataFrame({
                "Model": [f"Strom {i}" for i in range(1, 6)],
                "Predikce": ["Španělsko", "Portugalsko", "Španělsko", "Španělsko", "Portugalsko"],
                "Hlas": ["Červená (Vítěz)", "Zelená", "Červená (Vítěz)", "Červená (Vítěz)", "Zelená"]
            })
            st.dataframe(df_ex_c, width="stretch", hide_index=True)
            st.success("✅ **Výsledek většinového hlasování (Majority Vote):** 3 : 2 pro **Španělsko**.")

        with col_ex_r:
            st.markdown("##### ✈️ B. Regrese: Odhad ceny zájezdu (Slajd 11)")
            st.caption("3 rozhodovací stromy predikují cenu zájezdu v australských dolarech:")
            
            p1, p2, p3 = 4000.0, 4600.0, 4950.0
            avg_price = (p1 + p2 + p3) / 3.0
            
            st.latex(r"\hat{y} = \frac{4000 + 4600 + 4950}{3} = \frac{13550}{3} = 4516.67\text{ AUD}")
            st.metric("Agregovaná predikce lesa", f"{avg_price:.2f} AUD", delta="Aritmetický průměr 3 stromů")

        st.markdown("---")
        st.subheader("📉 Monitorování Out-Of-Bag (OOB) chyby a volba n_estimators")
        st.markdown(
            r"""
            Kolik stromů (`n_estimators`) je optimální? Pokud je stromů málo, některé vzorky a příznaky se do modelu 
            ani nedostanou. Sledujeme křivku OOB chyby – jakmile se ustálí, další stromy již nepřinášejí zpřesnění:
            """
        )

        n_trees_range = np.arange(5, 155, 5)
        # Teoretická křivka konvergence OOB chyby
        simulated_oob_mae = 270000 + 105000 * np.exp(-n_trees_range / 25) + np.random.RandomState(42).normal(0, 1500, len(n_trees_range))
        
        fig_oob = go.Figure()
        fig_oob.add_trace(go.Scatter(
            x=n_trees_range,
            y=simulated_oob_mae,
            mode="lines+markers",
            name="OOB MAE chyba",
            line=dict(color="#10b981", width=3)
        ))
        fig_oob.add_vline(
            x=100,
            line_dash="dash",
            line_color="#ef4444",
            annotation_text="Doporučený default (n_estimators=100)"
        )
        fig_oob.update_layout(
            title="Konvergence Out-Of-Bag chyby s rostoucím počtem stromů",
            xaxis_title="Počet rozhodovacích stromů (n_estimators)",
            yaxis_title="OOB MAE (AUD)",
            height=320,
            margin=dict(l=20, r=20, t=35, b=20)
        )
        st.plotly_chart(fig_oob, width="stretch")

    # =========================================================================
    # TAB 2: SCIKIT-LEARN API & PARAMETRY
    # =========================================================================
    with tab2:
        st.subheader("2. Kompletní Scikit-learn API: Třídy a hyperparametry")
        st.markdown(
            r"""
            V knihovně `scikit-learn` máme v balíčku `sklearn.ensemble` dvě dvojčata:
            - `RandomForestClassifier`: pro klasifikační úlohy.
            - `RandomForestRegressor`: pro regresní úlohy.
            """
        )

        st.markdown("#### ⚙️ Přehled klíčových hyperparametrů konstruktoru")
        params_data = [
            ("n_estimators", "100", "Počet stromů v lese. Vyšší hodnota zvyšuje stabilitu, ale prodlužuje trénink."),
            ("criterion", "'gini' / 'squared_error'", "Metrika štěpení uzlu. Klasifikace: 'gini', 'entropy'. Regrese: 'squared_error', 'absolute_error'."),
            ("max_depth", "None", "Maximální hloubka stromů. None = stromy rostou plně až do čistých listů."),
            ("min_samples_split", "2", "Minimální počet vzorků nutný pro rozdělení vnitřního uzlu stromu."),
            ("min_samples_leaf", "1", "Minimální počet vzorků požadovaný v koncovém listu (klíčový regulátor přeučení)."),
            ("max_features", "'sqrt' / 1.0", "Počet náhodně vybíraných příznaků při každém dělení. 'sqrt' = odmocnina z celkového počtu."),
            ("max_leaf_nodes", "None", "Omezení celkového počtu koncových listů pro kontrolu složitosti."),
            ("bootstrap", "True", "Zda vytvářet bootstrap vzorky s vracením (základ Baggingu)."),
            ("oob_score", "False", "Zda počítat Out-of-Bag validační skóre (doporučujeme nastavit na True)."),
            ("n_jobs", "None", "Počet paralelních CPU vláken. Nastavení -1 využije všechna dostupná jádra procesoru."),
            ("random_state", "None", "Seed pro pseudonáhodný generátor zajišťující přesnou reprodukovatelnost."),
        ]
        df_params = pd.DataFrame(params_data, columns=["Parametr", "Default v Scikit-learn", "Význam & Doporučení"])
        st.dataframe(df_params, width="stretch", hide_index=True)

        st.markdown("---")
        st.markdown("#### 📌 Přehled klíčových atributů a metod po natrénování")
        c_att1, c_att2 = st.columns(2)
        with c_att1:
            st.info(
                r"""
                ##### Klíčové atributy (po volání `.fit()`):
                - `rf.estimators_`: Seznam všech natrénovaných stromů (`DecisionTreeRegressor` / `Classifier`).
                - `rf.feature_importances_`: Pole důležitostí jednotlivých příznaků na základě poklesu nečistoty (MDI).
                - `rf.feature_names_in_`: Názvy sloupců z trénovacího `DataFrame`.
                - `rf.oob_score_`: Skóre na OOB datech ($R^2$ pro regresi, Accuracy pro klasifikaci).
                """
            )
        with c_att2:
            st.success(
                r"""
                ##### Klíčové metody:
                - `.fit(X, y)`: Natrénování celého lesa nezávislých stromů.
                - `.predict(X)`: Generování predikcí (průměr nebo většina hlasů).
                - `.score(X, y)`: Výpočet koeficientu determinace $R^2$ nebo Accuracy.
                - `.predict_proba(X)`: Odhad pravděpodobností tříd (pouze `RandomForestClassifier`).
                """
            )

    # =========================================================================
    # TAB 3: MELBOURNE HOUSING BENCHMARK
    # =========================================================================
    with tab3:
        st.subheader("3. Benchmark ze slajdů: Reality v Melbourne (`melb_house_data.csv`)")
        st.markdown(
            r"""
            V prezentaci je provedeno přímé srovnání samostatného rozhodovacího stromu a náhodného lesa 
            při predikci tržní ceny nemovitostí v Melbourne (Austrálie).
            """
        )

        col_code, col_kpi = st.columns([1.1, 0.9])
        with col_code:
            st.code(
                """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# 1. Příprava dat a rozdělení
melbourne_data = pd.read_csv("melb_house_data.csv")
X = melbourne_data.drop("price", axis=1)
y = melbourne_data["price"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.3
)

# 2. Samostatný rozhodovací strom (Benchmark)
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print(mean_absolute_error(y_test, y_pred_dt))
# --> 376 988 AUD

# 3. Náhodný les (Random Forest)
rf = RandomForestRegressor(random_state=42, n_estimators=100, oob_score=True)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print(mean_absolute_error(y_test, y_pred_rf))
# --> 272 109 AUD
                """,
                language="python"
            )

        with col_kpi:
            st.markdown("#### 🎯 Srovnání výsledků (Dle slajdů 14–15)")
            mae_dt = 376988
            mae_rf = 272109
            diff_mae = mae_dt - mae_rf
            pct_improvement = (diff_mae / mae_dt) * 100

            st.metric("Chyba samostatného stromu (MAE)", f"{mae_dt:,} AUD".replace(",", " "))
            st.metric(
                "Chyba Random Forestu (MAE)", 
                f"{mae_rf:,} AUD".replace(",", " "), 
                delta=f"-{diff_mae:,} AUD ({pct_improvement:.1f} %)".replace(",", " "),
                delta_color="normal"
            )
            st.info(
                f"💡 **Výsledek:** Pouhým nasazením Random Forestu s výchozími parametry došlo ke snížení "
                f"průměrné absolutní chyby o **{diff_mae:,} AUD** (zlepšení o **{pct_improvement:.1f} %**)!".replace(",", " ")
            )

        # Plotly bar chart srovnání
        fig_melb = go.Figure()
        fig_melb.add_trace(go.Bar(
            x=["DecisionTreeRegressor (1 strom)", "RandomForestRegressor (100 stromů)"],
            y=[mae_dt, mae_rf],
            text=[f"{mae_dt:,} AUD".replace(",", " "), f"{mae_rf:,} AUD".replace(",", " ")],
            textposition="auto",
            marker_color=["#ef4444", "#10b981"]
        ))
        fig_melb.update_layout(
            title="Průměrná absolutní chyba (MAE) na testovací sadě v Melbourne",
            yaxis_title="Střední absolutní chyba (MAE v AUD)",
            height=320,
            margin=dict(l=20, r=20, t=35, b=20)
        )
        st.plotly_chart(fig_melb, width="stretch")

    # =========================================================================
    # TAB 4: EXPERTNÍ KRITIKA & ÚSKALÍ
    # =========================================================================
    with tab4:
        st.subheader("4. Expertní kritické zhodnocení: Co ve slajdech chybí?")
        st.markdown(
            r"""
            Výklad ve školních materiálech obsahuje některá nebezpečná zjednodušení, 
            která v reálné produkční praxi vedou k pádům skriptů nebo klamavým výsledkům:
            """
        )

        c_crit1, c_crit2 = st.columns(2)
        with c_crit1:
            st.error(
                r"""
                #### ⚠️ 1. Mýtus o chybějících hodnotách v Scikit-learn
                - **Tvrzení ze slajdu 16:** *„Due to the mechanism... the random forest is relatively immune to missing values. It is not necessary to remove or fill them.“*
                - **Tvrdá realita:** V Scikit-learn `RandomForestClassifier/Regressor` **nepodporuje `NaN` hodnoty**!
                - Spuštění `.fit()` na datech s chybějícími hodnotami vyvolá fatální `ValueError: Input contains NaN`.
                - **Řešení:** Vždy je nutné použít `Pipeline` s `SimpleImputer` nebo `KNNImputer`.
                """
            )
            st.warning(
                r"""
                #### 📉 2. Neschopnost extrapolace v regresi
                - Predikce stromu je průměrem hodnot v listu.
                - Random Forest **nikdy nedokáže předpovědět hodnotu vyšší než maximum v trénovacích datech** (ani nižší než minimum).
                - Při silném inflačním růstu cen nemovitostí nebo časových trendech Random Forest selhává, pokud není kombinován s lineárním trendem.
                """
            )

        with c_crit2:
            st.warning(
                r"""
                #### 📊 3. Zkreslení MDI (*Mean Decrease in Impurity*)
                - Výchozí `rf.feature_importances_` měří pokles nečistoty.
                - **Kardinální vada:** MDI systematicky **nadhodnocuje spojité proměnné a kategorické proměnné s vysokou kardinalitou** (mnoho kategorií, PSČ, ID).
                - Dokonce i čistě náhodný sloupec s mnoha unikátními čísly získá vysokou důležitost!
                - **Řešení:** Vždy ověřovat pomocí `permutation_importance` na testovací sadě.
                """
            )
            st.error(
                r"""
                #### 🗺️ 4. Prostorový a časový únik (Data Leakage) při OOB
                - Out-Of-Bag evaluace předpokládá nezávislost vzorků (*i.i.d.*).
                - U realitních dat v Melbourne jsou domy v jedné ulici silně prostorově závislé.
                - Pokud se jeden dům dostane do tréninku a sousední dům do OOB, OOB skóre je **klamně optimistické** kvůli prostorovému úniku informací.
                - **Řešení:** Bloková / prostorová křížová validace (*Spatial K-Fold*).
                """
            )

    # =========================================================================
    # TAB 5: SOTA ROZŠÍŘENÍ 10/2026
    # =========================================================================
    with tab5:
        st.subheader("5. Moderní ML & AI rozšíření: Stav k říjnu 2026")
        st.markdown(
            r"""
            Jak se s náhodnými lesy pracuje v moderním produkčním machine learningu a MLOps k 10/2026?
            """
        )

        c_sota1, c_sota2 = st.columns(2)
        with c_sota1:
            st.markdown("#### ⚡ Hardwarová akcelerace: NVIDIA cuML & HistGradientBoosting")
            st.markdown(
                r"""
                - **`cuml.ensemble.RandomForestRegressor` (NVIDIA RAPIDS):** 
                  Umožňuje trénovat les s 1 000 stromy na milionech řádků přímo v paměti GPU. Nabízí **až 50x zrychlení** proti vícejádrovému CPU.
                - **Histogramové stromy (`HistGradientBoostingRegressor`):** 
                  Spojité proměnné jsou binovány do 256 košů (integers), což zrychluje trénink o řád a nativně podporuje chybějící hodnoty (`NaN`).
                - **`ExtraTreesClassifier / Regressor`:**
                  Dělicí prahy v uzlech se netestují deterministicky, ale losují se zcela náhodně $\to$ ještě nižší rozptyl a bleskový trénink.
                """
            )

        with c_sota2:
            st.markdown("#### 🔍 Moderní interpretovatelnost: TreeSHAP (XAI)")
            st.markdown(
                r"""
                - Místo nespolehlivého MDI se dnes standardně nasazuje **TreeSHAP** (`shap.TreeExplainer`).
                - Vypočítává exaktní Shapleyho hodnoty v polynomiálním čase $O(T L D^2)$, kde $T$ je počet stromů a $D$ jejich hloubka.
                - Umožňuje přesně vysvětlit každou jednotlivou cenovou nabídku nemovitosti i globální závislosti trhu bez zkreslení kardinalitou.
                """
            )

        st.markdown("---")
        st.markdown("#### 🏆 Srovnání algoritmů na tabulkových datech (Benchmark 2026)")
        df_bench = pd.DataFrame({
            "Algoritmus": ["Decision Tree", "Random Forest", "ExtraTrees", "HistGradientBoosting / LightGBM", "cuML Random Forest (GPU)"],
            "Rychlost tréninku": ["Blesková", "Střední (CPU limit)", "Rychlá", "Velmi rychlá", "Extrémní (GPU)"],
            "Odolnost proti přeučení": ["Velmi nízká", "Vysoká", "Velmi vysoká", "Vysoká (s early stoppingem)", "Vysoká"],
            "Nativní podpora NaN": ["Ne (v sklearn)", "Ne (v sklearn)", "Ne (v sklearn)", "Ano (nativně)", "Ano"],
            "Schopnost extrapolace": ["Ne", "Ne", "Ne", "Částečně", "Ne"]
        })
        st.dataframe(df_bench, width="stretch", hide_index=True)

    # =========================================================================
    # TAB 6: VĚDOMOSTNÍ KVÍZ
    # =========================================================================
    with tab6:
        st.subheader("📝 Rychlý znalostní kvíz: Random Forest & Scikit-learn")

        q1 = st.radio(
            "1. V čem spočívá klíčový rozdíl mezi čistým Baggingem na stromech a algoritmem Random Forest?",
            [
                "Random Forest používá neuronové sítě místo stromů.",
                "Random Forest kromě náhodného vzorkování řádků (bootstrapping) náhodně vybírá i podmnožinu příznaků (sloupců) při každém dělení uzlu.",
                "Random Forest trénuje stromy sekvenčně za sebou, zatímco Bagging paralelně.",
                "Mezi Baggingem a Random Forestem není žádný rozdíl."
            ],
            key="rf_q1"
        )

        q2 = st.radio(
            "2. Proč je v Random Forestu výhodné použít plně vyrostlé (hluboké) stromy bez ořezání?",
            [
                "Hluboké stromy mají nízké vychýlení (low bias) a jejich vysoký rozptyl (variance) ansámbl efektivně vyruší průměrováním.",
                "Hluboké stromy se trénují rychleji než mělké.",
                "Hluboké stromy zabírají méně paměti RAM.",
                "Scikit-learn neumožňuje omezit hloubku stromu."
            ],
            key="rf_q2"
        )

        q3 = st.radio(
            "3. Jak se chová standardní `RandomForestRegressor` v Scikit-learn, pokud vstupní data obsahují hodnoty NaN?",
            [
                "Automaticky doplní chybějící hodnoty mediánem.",
                "Ignoruje řádky s chybějícími hodnotami bez varování.",
                "Zhavaruje s chybou `ValueError: Input contains NaN` – je nutné použít SimpleImputer v Pipeline.",
                "Převede NaN na nulu a pokračuje v tréninku."
            ],
            key="rf_q3"
        )

        q4 = st.radio(
            "4. Jaký je hlavní nedostatek výchozí metriky `feature_importances_` (MDI) u Random Forestu?",
            [
                "Trvá déle než samotný trénink modelu.",
                "Systematicky nadhodnocuje proměnné s vysokou kardinalitou (mnoho unikátních hodnot) a náhodný šum.",
                "Vrací záporné hodnoty pro kategorické příznaky.",
                "Funguje pouze pro binární klasifikaci."
            ],
            key="rf_q4"
        )

        if st.button("Vyhodnotit kvíz Random Forest", width="stretch"):
            score = 0
            if "náhodně vybírá i podmnožinu příznaků" in q1:
                score += 1
            if "Hluboké stromy mají nízké vychýlení" in q1 or "Hluboké stromy mají nízké vychýlení" in q2:
                score += 1
            if "Zhavaruje s chybou `ValueError: Input contains NaN`" in q3:
                score += 1
            if "nadhodnocuje proměnné s vysokou kardinalitou" in q4:
                score += 1

            if score == 4:
                st.balloons()
                st.success("🎉 Excelentně! 4 ze 4 bodů! Dokonale ovládáte teorii, Scikit-learn implementaci i úskalí Random Forestu.")
            else:
                st.warning(f"Získali jste {score} ze 4 bodů. Projděte si záložky s teorií a expertní kritikou.")


render_random_forest_theory_view()
