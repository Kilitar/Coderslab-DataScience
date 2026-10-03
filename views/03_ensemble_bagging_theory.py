"""
Den 3: Ansámblové metody a Bagging (Ensemble Methods & Bagging)
==============================================================
Interaktivní výukový modul pro první část teorie Dne 3:
1. Teoretický rozbor kompromisu Bias-Variance a matematické zdůvodnění ansámblů.
2. 4 rodiny ansámblů: Voting, Stacking, Bagging, Boosting (včetně interaktivních kalkulátorů z přednášky).
3. Hloubkový simulátor Bootstrappingu a Out-Of-Bag (OOB) evaluace s matematickým důkazem 1/e (36.8 %).
4. Živý experiment: Samotný strom vs. BaggingClassifier na reálných kardiologických datech Dne 3 (heart_data).
5. Interaktivní vědomostní kvíz k ověření pochopení.
"""

from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score


@st.cache_data
def load_heart_dataset():
    base_dir = Path(__file__).resolve().parent.parent
    csv_path = base_dir / "data" / "MAL_downloadable materials_session 2" / "Day 3" / "heart_data_normalized.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return df
    return None


def render_ensemble_bagging_view():
    st.title("🌲 Den 3: Ansámblové metody & Bagging")
    st.markdown(
        "**Úvod do 3. dne (Session 2):** Přechod od základních samostatných modelů k pokročilým **ansámblovým metodám** "
        "(Ensemble Learning) – princip „moudrosti davu“, redukce rozptylu (Variance), paralelní trénování na bootstrapových vzorcích "
        "a srovnání 4 základních architektur: **Voting**, **Stacking**, **Bagging** a **Boosting**."
    )

    # Horní KPI karty
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Klíčový cíl Baggingu", "Redukce rozptylu (Variance)", delta="Prevence přeučení")
    with c2:
        st.metric("4 rodiny ansámblů", "Voting, Stacking, Bagging, Boosting", delta="Paralelní vs. Sekvenční")
    with c3:
        st.metric("Bootstrapping", "Výběr s vracením", delta="≈ 63.2 % in-bag, 36.8 % OOB")
    with c4:
        st.metric("Škálovatelnost", "Plná paralelizace", delta="n_jobs = -1 (Nezávislé modely)")

    st.markdown("---")

    # Záložky modulu
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "⚖️ Proč ansámbly? (Bias-Variance)",
        "🗺️ 4 Rodiny ansámblů & Kalkulátor",
        "🎲 Bootstrapping & OOB simulátor",
        "🫀 Živý experiment: Strom vs. Bagging",
        "📝 Vědomostní kvíz"
    ])

    # =========================================================================
    # TAB 1: PROČ ANSÁMBLY? (BIAS-VARIANCE)
    # =========================================================================
    with tab1:
        st.subheader("1. Proč používáme ansámblové metody? Kompromis Bias-Variance")
        st.markdown(
            "Při trénování samostatných modelů (lineární modely, rozhodovací stromy, k-NN) neustále balancujeme "
            "mezi dvěma protichůdnými zdroji chyb:"
        )

        col_bv1, col_bv2 = st.columns([1, 1])
        with col_bv1:
            st.error(
                """
                #### 🎯 Vychýlení (Bias) $\\to$ Podtrénování (Underfitting)
                - Chyba způsobená přílišným zjednodušením modelu (např. snaha proložit nelineární data přímkou).
                - Model má vysokou chybu jak na trénovací, tak na testovací sadě.
                - Nereaguje dostatečně na vztahy v datech.
                """
            )
        with col_bv2:
            st.warning(
                """
                #### 🌊 Rozptyl (Variance) $\\to$ Přeučení (Overfitting)
                - Citlivost modelu na náhodný šum a drobné fluktuace v trénovací sadě.
                - Typické pro **hluboké rozhodovací stromy**: na trénovacích datech dosahují 100% přesnosti, ale na nových datech selhávají.
                - Malá změna v trénovacích datech vede k dramatické změně celého modelu.
                """
            )

        st.markdown(
            r"""
            $$\text{Celková očekávaná chyba} = \underbrace{\text{Bias}^2}_{\text{Vychýlení}} + \underbrace{\text{Variance}}_{\text{Rozptyl}} + \underbrace{\sigma_{\epsilon}^2}_{\text{Neodstranitelný šum}}$$
            """
        )

        st.markdown("---")
        st.subheader("📉 Interaktivní simulátor: Jak roste stabilita a klesá rozptyl s počtem modelů")
        st.markdown(
            r"Pokud zprůměrujeme $M$ modelů s rozptylem $\sigma^2$ a vzájemnou korelací $\rho$, celkový rozptyl ansámblu je: "
            r"$$\text{Var}(\text{Ansámbl}) = \rho \sigma^2 + \frac{1 - \rho}{M} \sigma^2$$"
        )

        col_sim1, col_sim2 = st.columns([1, 2])
        with col_sim1:
            m_trees = st.slider("Počet modelů v ansámblu (M):", min_value=1, max_value=50, value=15, step=1)
            rho_val = st.slider("Vzájemná korelace modelů (ρ):", min_value=0.0, max_value=0.9, value=0.3, step=0.05)
            st.caption("Čím nižší je korelace mezi modely (vyšší diverzita), tím efektivněji ansámbl redukuje rozptyl!")

        with col_sim2:
            m_range = np.arange(1, 51)
            var_curve = rho_val * 1.0 + ((1.0 - rho_val) / m_range) * 1.0
            
            fig_var = go.Figure()
            fig_var.add_trace(go.Scatter(
                x=m_range,
                y=var_curve,
                mode="lines+markers",
                name="Rozptyl ansámblu",
                line=dict(color="#3b82f6", width=3)
            ))
            fig_var.add_vline(
                x=m_trees,
                line_dash="dash",
                line_color="#ef4444",
                annotation_text=f"M={m_trees} (Rozptyl: {var_curve[m_trees-1]:.3f})"
            )
            fig_var.update_layout(
                title="Pokles celkového rozptylu s rostoucím počtem stromů M",
                xaxis_title="Počet modelů (M)",
                yaxis_title="Relativní rozptyl (Variance)",
                height=320,
                margin=dict(l=20, r=20, t=35, b=20)
            )
            st.plotly_chart(fig_var, width="stretch")

    # =========================================================================
    # TAB 2: 4 RODINY ANSÁMBLŮ & KALKULÁTOR
    # =========================================================================
    with tab2:
        st.subheader("2. Čtyři základní typy ansámblů (Ensemble Methods - Types)")
        st.markdown(
            "Podle způsobu přípravy trénovacích dat a mechanismu agregace dělíme ansámbly do čtyř hlavních kategorií:"
        )

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.info(
                """
                #### 1. Hlasování (Voting)
                - **Data:** Stejná trénovací množina pro všechny modely.
                - **Algoritmy:** Různé (např. SVM + Logistická regrese + k-NN).
                - **Regrese:** Aritmetický nebo vážený průměr.
                - **Klasifikace:** Většinové (Majority / Hard) nebo pravděpodobnostní (Weighted / Soft) hlasování.
                """
            )
            st.warning(
                """
                #### 3. Bagging (Bootstrap Aggregating)
                - **Data:** Různé náhodné vzorky vytvořené **výběrem s vracením** (Bootstrapping).
                - **Algoritmy:** Stejný typ algoritmu (typicky rozhodovací strom).
                - **Trénování:** Nezávislé, plně paralelizovatelné.
                - **Cíl:** Drastické snížení rozptylu u nestabilních modelů.
                """
            )
        with col_t2:
            st.success(
                """
                #### 2. Stacking (Stacked Generalization)
                - **Data:** Stejná data pro bázové modely.
                - **Mechanismus:** Predikce bázových modelů slouží jako **vstupy pro meta-model** (např. Logistickou regresi).
                - **Meta-model:** Učí se optimálně kombinovat síly jednotlivých modelů.
                """
            )
            st.error(
                """
                #### 4. Boosting
                - **Data:** Postupně převážená data nebo rezidua předchozích kroků.
                - **Mechanismus:** Sekvenční trénování slabých modelů (*Weak Learners*).
                - **Trénování:** Každý model se učí z chyb předchozího.
                - **Cíl:** Snížení jak rozptylu, tak především **vychýlení (Bias)**.
                """
            )

        st.markdown("---")
        st.subheader("🧮 Interaktivní kalkulátory ze slajdů přednášky")

        c_calc1, c_calc2 = st.columns(2)

        with c_calc1:
            st.markdown("##### 🗳️ Příklad klasifikace: Většinové hlasování (Majority Voting)")
            st.caption("Podle slajdu 8: 5 modelů hlasuje pro třídu 0 nebo 1.")

            votes = []
            v_cols = st.columns(5)
            for idx, c in enumerate(v_cols, 1):
                with c:
                    v = st.selectbox(f"M{idx}", [0, 1], index=0 if idx <= 3 else 1, key=f"vote_{idx}")
                    votes.append(v)

            count_0 = votes.count(0)
            count_1 = votes.count(1)
            final_class = 0 if count_0 > count_1 else 1

            st.metric(
                "Výsledek hlasování",
                f"Třída {final_class}",
                delta=f"Poměr {count_0} : {count_1} (Třída 0 : Třída 1)"
            )
            if votes == [0, 0, 0, 1, 1]:
                st.success("Přesně odpovídá příkladu ze slajdu 8: Tři modely hlásí 0, dva hlásí 1 $\\to$ Vítězí třída 0 (poměr 3:2).")

        with c_calc2:
            st.markdown("##### 📈 Příklad regrese: Průměrování predikcí (Bagging Aggregation)")
            st.caption("Podle slajdu 17: Model 1 vrátí 100, Model 2 vrátí 150, Model 3 vrátí 200.")

            rc1, rc2, rc3 = st.columns(3)
            with rc1:
                val1 = st.number_input("Model #1", value=100.0, step=10.0)
            with rc2:
                val2 = st.number_input("Model #2", value=150.0, step=10.0)
            with rc3:
                val3 = st.number_input("Model #3", value=200.0, step=10.0)

            avg_res = (val1 + val2 + val3) / 3.0
            st.latex(r"\hat{y} = \frac{" + f"{val1:.0f} + {val2:.0f} + {val3:.0f}" + r"}{3} = " + f"{avg_res:.1f}")
            st.metric("Agregovaná finální predikce", f"{avg_res:.1f}")

    # =========================================================================
    # TAB 3: BOOTSTRAPPING & OOB SIMULÁTOR
    # =========================================================================
    with tab3:
        st.subheader("3. Bootstrapping: Výběr s vracením a Out-Of-Bag (OOB) data")
        st.markdown(
            "Základním kamenem techniky Bagging je **Bootstrapping**: Z původní trénovací sady o velikosti $N$ "
            "vytváříme nové vzorky náhodným tahem s opakováním (výběr s vracením – *with replacement*)."
        )

        col_boot1, col_boot2 = st.columns([1, 1])
        with col_boot1:
            st.markdown("##### 📐 Matematický důkaz: Proč zůstává 36.8 % dat stranou?")
            st.write(
                r"""
                Uvažujme dataset s $N$ pozorováními:
                1. Pravděpodobnost, že konkrétní řádek **nebude** vybrán v prvním tahu:
                   $$P = 1 - \frac{1}{N}$$
                2. Vytváříme bootstrap vzorek o stejné velikosti $N$ tahů. Pravděpodobnost, že řádek **nebude vybrán ani jednou**:
                   $$P(\text{nevybráno v } N \text{ tazích}) = \left(1 - \frac{1}{N}\right)^N$$
                3. Pro rostoucí velikost datasetu $N \to \infty$ tato hodnota konverguje k Eulerovu číslu:
                   $$\lim_{N \to \infty} \left(1 - \frac{1}{N}\right)^N = \frac{1}{e} \approx 0.367879 \approx \mathbf{36.8\,\%}$$
                
                **Závěr:** Přibližně **63.2 %** vzorků je v trénovacím vzorku (některé vícekrát) a **36.8 %** tvoří tzv. **Out-Of-Bag (OOB)** sadu, 
                která slouží k bezplatnému testování generalizace modelu!
                """
            )

        with col_boot2:
            st.markdown("##### 🎲 Živá simulace losování s vracením:")
            n_sim = st.slider("Velikost datasetu (N):", min_value=10, max_value=500, value=100, step=10)
            
            # Provedení náhodného bootstrap losování
            original_indices = np.arange(n_sim)
            sampled_indices = np.random.choice(original_indices, size=n_sim, replace=True)
            unique_sampled = np.unique(sampled_indices)
            
            in_bag_count = len(unique_sampled)
            oob_count = n_sim - in_bag_count
            in_bag_pct = (in_bag_count / n_sim) * 100
            oob_pct = (oob_count / n_sim) * 100

            fig_pie = go.Figure(data=[go.Pie(
                labels=["In-Bag (Trénovací unikáty)", "Out-Of-Bag (OOB Testovací)"],
                values=[in_bag_count, oob_count],
                hole=0.45,
                marker_colors=["#3b82f6", "#f59e0b"]
            )])
            fig_pie.update_layout(
                title=f"Výsledek simulace pro N = {n_sim} (OOB: {oob_pct:.1f} %)",
                height=280,
                margin=dict(l=20, r=20, t=35, b=20)
            )
            st.plotly_chart(fig_pie, width="stretch")

            st.caption(f"Unikátně vylosováno: **{in_bag_count} z {n_sim}** ({in_bag_pct:.1f} %). Teoretické očekávání: 63.2 % vs. 36.8 %.")

    # =========================================================================
    # TAB 4: ŽIVÝ EXPERIMENT: STROM VS. BAGGING (HEART DATA)
    # =========================================================================
    with tab4:
        st.subheader("4. Živý experiment: Samotný rozhodovací strom vs. BaggingClassifier")
        st.markdown(
            "Vyzkoušejme si sílu Baggingu na reálném kardiologickém datasetu pro **Den 3** (`heart_data_normalized.csv`, 303 pacientů). "
            "Cílem je predikovat přítomnost onemocnění srdce (`ahd_yes`)."
        )

        df_heart = load_heart_dataset()
        if df_heart is None:
            st.error("Dataset `heart_data_normalized.csv` nebyl nalezen.")
        else:
            feature_cols = [c for c in df_heart.columns if c != "ahd_yes"]
            X = df_heart[feature_cols]
            y = df_heart["ahd_yes"]

            col_exp_opt1, col_exp_opt2 = st.columns([1, 1])
            with col_exp_opt1:
                test_sz = st.slider("Velikost testovací sady:", min_value=0.15, max_value=0.40, value=0.25, step=0.05)
                n_estimators = st.slider("Počet stromů v Baggingu (n_estimators):", min_value=2, max_value=60, value=20, step=2)
            with col_exp_opt2:
                seed_val = st.number_input("Random Seed:", value=42, step=1)
                st.info("Trénujeme neprořezaný rozhodovací strom (vysoký rozptyl) proti ansámblu mnoha neprořezaných stromů.")

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_sz, random_state=int(seed_val), stratify=y)

            # 1. Samostatný strom
            single_tree = DecisionTreeClassifier(random_state=int(seed_val))
            single_tree.fit(X_train, y_train)
            tree_pred = single_tree.predict(X_test)
            tree_train_acc = single_tree.score(X_train, y_train)
            tree_test_acc = accuracy_score(y_test, tree_pred)
            tree_f1 = f1_score(y_test, tree_pred)
            tree_rec = recall_score(y_test, tree_pred)

            # 2. Bagging ansámbl
            bagging = BaggingClassifier(
                estimator=DecisionTreeClassifier(),
                n_estimators=int(n_estimators),
                oob_score=True,
                random_state=int(seed_val),
                n_jobs=-1
            )
            bagging.fit(X_train, y_train)
            bag_pred = bagging.predict(X_test)
            bag_train_acc = bagging.score(X_train, y_train)
            bag_test_acc = accuracy_score(y_test, bag_pred)
            bag_f1 = f1_score(y_test, bag_pred)
            bag_rec = recall_score(y_test, bag_pred)
            bag_oob = bagging.oob_score_

            st.markdown("---")
            st.markdown("#### 📊 Výsledky srovnání na testovací sadě:")

            res_cols = st.columns(4)
            with res_cols[0]:
                st.metric(
                    "Testovací Accuracy",
                    f"{bag_test_acc * 100:.1f} %",
                    delta=f"{(bag_test_acc - tree_test_acc) * 100:+.1f} % oproti stromu"
                )
            with res_cols[1]:
                st.metric(
                    "F1-Score",
                    f"{bag_f1:.4f}",
                    delta=f"{bag_f1 - tree_f1:+.4f} oproti stromu"
                )
            with res_cols[2]:
                st.metric(
                    "Recall (Záchyt onemocnění)",
                    f"{bag_rec * 100:.1f} %",
                    delta=f"{(bag_rec - tree_rec) * 100:+.1f} %"
                )
            with res_cols[3]:
                st.metric(
                    "OOB Score (Bezplatná validace)",
                    f"{bag_oob * 100:.1f} %",
                    delta="Vestavěná validace bez test setu"
                )

            # Srovnávací Plotly Bar chart
            metrics_df = pd.DataFrame({
                "Metrika": ["Trénovací přesnost", "Testovací přesnost", "Testovací F1-Score", "Recall (Záchyt)"],
                "Jediný strom (Overfitted)": [tree_train_acc * 100, tree_test_acc * 100, tree_f1 * 100, tree_rec * 100],
                f"Bagging ({n_estimators} stromů)": [bag_train_acc * 100, bag_test_acc * 100, bag_f1 * 100, bag_rec * 100]
            })

            fig_comp_m = go.Figure()
            fig_comp_m.add_trace(go.Bar(
                x=metrics_df["Metrika"],
                y=metrics_df["Jediný strom (Overfitted)"],
                name="Samostatný strom",
                marker_color="#ef4444"
            ))
            fig_comp_m.add_trace(go.Bar(
                x=metrics_df["Metrika"],
                y=metrics_df[f"Bagging ({n_estimators} stromů)"],
                name=f"Bagging ({n_estimators} stromů)",
                marker_color="#10b981"
            ))
            fig_comp_m.update_layout(
                barmode="group",
                height=340,
                margin=dict(l=20, r=20, t=30, b=20),
                yaxis_title="Hodnota metriky (%)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_comp_m, width="stretch")
            st.caption(
                "💡 **Klíčové pozorování:** Samostatný strom má 100 % na trénovací sadě (typické přeučení / vysoký rozptyl). "
                "Bagging díky agregaci nezávislých stromů stabilizuje rozhodování a výrazně zvyšuje přesnost i F1 na nových testovacích pacientech!"
            )

    # =========================================================================
    # TAB 5: VĚDOMOSTNÍ KVÍZ
    # =========================================================================
    with tab5:
        st.subheader("📝 Rychlý vědomostní kvíz k ověření pochopení Baggingu")

        q1 = st.radio(
            "1. Co je primárním cílem techniky Bagging při práci s rozhodovacími stromy?",
            [
                "Zvýšit složitost a hloubku stromu.",
                "Snížit rozptyl (Variance) a zvýšit stabilitu predikcí zprůměrováním nezávislých modelů.",
                "Zpomalit výpočetní čas.",
                "Odstranit všechny chybějící hodnoty."
            ]
        )

        q2 = st.radio(
            "2. Pokud v klasifikačním ansámblu 5 modelů hlasují tři modely pro třídu 0 a dva modely pro třídu 1, jaký je výsledek většinového hlasování (Majority Voting)?",
            [
                "Výsledkem je třída 1, protože má vyšší číslo.",
                "Výsledkem je třída 0 na základě většinového poměru 3:2.",
                "Výsledkem je náhodné číslo.",
                "Modely musí hlasovat znovu."
            ]
        )

        q3 = st.radio(
            "3. Jaký podíl trénovacích dat v průměru NENÍ vybrán do konkrétního bootstrap vzorku (tzv. Out-Of-Bag – OOB data)?",
            [
                "Přesně 50.0 % dat.",
                "Přibližně 36.8 % dat (odpovídá limitě 1/e).",
                "Přesně 0 %, protože se vybírají všechna data.",
                "Přibližně 10.0 % dat."
            ]
        )

        q4 = st.radio(
            "4. Jaký je zásadní rozdíl mezi architekturami Bagging a Boosting?",
            [
                "V Baggingu se modely trénují nezávisle a paralelně, zatímco v Boostingu se trénují sekvenčně za sebou a každý se učí z chyb předchozího.",
                "Bagging funguje pouze na obrázcích, Boosting pouze na textech.",
                "Bagging nelze použít pro regresi.",
                "Mezi Baggingem a Boostingem není žádný rozdíl."
            ]
        )

        if st.button("Vyhodnotit kvíz Bagging", width="stretch"):
            score = 0
            if "Snížit rozptyl (Variance) a zvýšit stabilitu" in q1:
                score += 1
            if "Výsledkem je třída 0 na základě většinového poměru 3:2" in q2:
                score += 1
            if "Přibližně 36.8 % dat (odpovídá limitě 1/e)" in q3:
                score += 1
            if "V Baggingu se modely trénují nezávisle a paralelně" in q4:
                score += 1

            if score == 4:
                st.balloons()
                st.success("🎉 Skvěle! 4 ze 4 správně! Máte dokonalý základ pro pokročilé ansámbly a Random Forest.")
            else:
                st.warning(f"Získali jste {score} ze 4 bodů. Projděte si záložky s teorií a kalkulátory.")


render_ensemble_bagging_view()
