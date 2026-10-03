"""
Den 2: Ucelené shrnutí klasifikačních modelů a interaktivní znalostní kvíz
========================================================================
Syntéza 4 stěžejních klasifikačních algoritmů probíraných v rámci 2. dne kurzu:
- k-Nearest Neighbors (k-NN)
- Logistická regrese (Logistic Regression)
- Rozhodovací stromy (Decision Trees)
- Support Vector Machines (SVM)
+ Evaluační metriky a interaktivní závěrečný test.
"""

from typing import Dict, Any
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def render_day_2_summary_view():
    st.title("🎓 Den 2: Ucelené shrnutí klasifikace & Znalostní kvíz")
    st.markdown(
        """
        Druhý den kurzu Data Science & Machine Learning byl věnován **klasifikačním úlohám** 
        – od instančního učení přes pravděpodobnostní a stromové modely až po geometrickou maximalizaci marže.
        Níže naleznete ucelenou syntézu, srovnávací analýzu, diagnostického průvodce výběrem modelu a interaktivní kvíz.
        """
    )

    tab_overview, tab_comparison, tab_metrics, tab_quiz = st.tabs(
        [
            "🏛️ 4 klasifikační rodiny",
            "📊 Srovnávací matice & Průvodce",
            "🎯 Evaluační metriky",
            "🧠 Znalostní kvíz (Quiz Time)",
        ]
    )

    # ---------------------------------------------------------
    # TAB 1: 4 Klasifikační rodiny
    # ---------------------------------------------------------
    with tab_overview:
        st.subheader("Architektura klasifikačních paradigmat")
        st.markdown(
            """
            Každý algoritmus nahlíží na problém separace tříd z jiné perspektivy:
            """
        )

        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown("### 1. k-Nearest Neighbors (k-NN)")
                st.markdown(
                    """
                    **Filosofie:** *„Řekni mi, kdo jsou tvoji sousedé, a já ti řeknu, kdo jsi.“*
                    - **Typ učení:** *Lazy learning* (žádné parametry k trénování, pamatuje si celá trénovací data).
                    - **Princip:** Hlasování $k$ nejbližších vzorků ve stavovém prostoru podle eukleidovské či jiné metriky.
                    - **Vliv parametru $k$:**
                      - Malé $k$ (např. $k=1$): Členitá hranice, náchylnost k šumu a přeučení.
                      - Velké $k$: Hladká hranice, riziko podučení (přehlasování majoritní třídou).
                    - **Podmínka fungování:** Všechny numerické příznaky **musí být škálovány**!
                    """
                )

            with st.container(border=True):
                st.markdown("### 2. Logistická regrese")
                st.markdown(
                    r"""
                    **Filosofie:** *„Lineární kombinace vstupů modelující logaritmus šance (logit).“*
                    - **Typ učení:** Parametrický pravděpodobnostní model.
                    - **Princip:** Mapování lineárního prediktoru $z = \mathbf{w}^T \mathbf{x} + b$ přes sigmoidální funkci $\sigma(z) \in [0, 1]$.
                    - **Výhoda:** Poskytuje kalibrované pravděpodobnosti příslušnosti ke třídě.
                    - **Interpretovatelnost:** Exponenciála váhy $e^{w_j}$ vyjadřuje poměr šancí (*Odds Ratio*).
                    - **Regularizace:** Parametr $C$ (inverzní síla L1/L2 regularizace).
                    """
                )

        with col2:
            with st.container(border=True):
                st.markdown("### 3. Rozhodovací strom (Decision Tree)")
                st.markdown(
                    r"""
                    **Filosofie:** *„Hierarchická posloupnost binárních otázek štěpících prostor.“*
                    - **Typ učení:** Neparametrický pravidlový model (bílý model – white-box).
                    - **Princip:** Rekurzivní dělení dat podle prahu $x_j \le \theta$, které maximalizuje pokles nečistoty (Gini či Entropie).
                    - **Výhoda:** Nevyžaduje škálování dat, přirozeně zvládá interakce a je plně vizualizovatelný.
                    - **Riziko:** Rychlé přeučení bez omezení hloubky (`max_depth`, `min_samples_leaf`).
                    - **Hranice:** Ortogonální (po osách orientované schodovité hranice).
                    """
                )

            with st.container(border=True):
                st.markdown("### 4. Support Vector Machine (SVM)")
                st.markdown(
                    r"""
                    **Filosofie:** *„Hledání nadroviny s maximální geometrickou marží mezi třídami.“*
                    - **Typ učení:** Geometrický optimalizační model (konvexní kvadratické programování).
                    - **Princip:** Rozhodovací hranici určují pouze kritické hraniční vzorky – **podpůrné vektory**.
                    - **Jádrový trik (Kernel Trick):** Mapování do vícerozměrného prostoru pro nelineární oddělení (RBF, polynom).
                    - **Parametry:** $C$ (tolerance chyb v marži) a $\gamma$ (lokalita vlivu RBF jádra).
                    - **Výhoda:** Vynikající generalizace a stabilita ve vysoké dimenzi.
                    """
                )

    # ---------------------------------------------------------
    # TAB 2: Srovnávací matice & Rozhodovací průvodce
    # ---------------------------------------------------------
    with tab_comparison:
        st.subheader("Komparativní analýza klasifikátorů")

        comparison_data = [
            {
                "Algoritmus": "k-NN",
                "Tréninková rychlost": "Blesková O(1)",
                "Predikční rychlost": "Pomalá O(N·D)",
                "Interpretovatelnost": "Střední (lokální)",
                "Škálování nutné?": "Kriticky ANO",
                "Odolnost k dimenzionalitě": "Nízká (Curse of Dim.)",
                "Nelineární hranice": "Přirozeně ANO",
                "Výstup": "Hlasování / lokální poměr",
            },
            {
                "Algoritmus": "Logistická regrese",
                "Tréninková rychlost": "Velmi rychlá",
                "Predikční rychlost": "Blesková O(D)",
                "Interpretovatelnost": "Vysoká (koeficienty)",
                "Škálování nutné?": "Doporučeno (pro regularizaci)",
                "Odolnost k dimenzionalitě": "Vysoká (s L1/L2)",
                "Nelineární hranice": "NE (vyžaduje polynomy)",
                "Výstup": "Skutečná pravděpodobnost",
            },
            {
                "Algoritmus": "Rozhodovací strom",
                "Tréninková rychlost": "Střední",
                "Predikční rychlost": "Velmi rychlá O(depth)",
                "Interpretovatelnost": "Vynikající (pravidla if-then)",
                "Škálování nutné?": "NE (zcela invariantní)",
                "Odolnost k dimenzionalitě": "Střední (vybírá příznaky)",
                "Nelineární hranice": "ANO (schodovitá)",
                "Výstup": "Čistota listu / podíl tříd",
            },
            {
                "Algoritmus": "SVM (SVC)",
                "Tréninková rychlost": "Pomalá O(N² až N³)",
                "Predikční rychlost": "Rychlá O(n_sv · D)",
                "Interpretovatelnost": "Nízká (černá skříňka u RBF)",
                "Škálování nutné?": "Kriticky ANO",
                "Odolnost k dimenzionalitě": "Vynikající (i pro D > N)",
                "Nelineární hranice": "Vynikající (s RBF jádrem)",
                "Výstup": "Vzdálenost od marže (rozhodnutí)",
            },
        ]
        df_comp = pd.DataFrame(comparison_data)
        st.dataframe(df_comp, width="stretch", hide_index=True)

        st.markdown("---")
        st.subheader("🧭 Průvodce volbou modelu v praxi")

        diag_col1, diag_col2 = st.columns([1.2, 1])
        with diag_col1:
            st.markdown(
                r"""
                | Požadavek úlohy | Doporučený model | Zdůvodnění |
                | :--- | :--- | :--- |
                | **Regulovaný sektor (bankovnictví, medicína, audit)** | **Logistická regrese** nebo **Mělký strom** | Nutnost přesně vysvětlit, proč byl úvěr zamítnut nebo stanovena diagnóza. |
                | **Rychlý prototyp bez přípravy dat** | **Rozhodovací strom** | Nevadí mu chybějící škálování, různé jednotky ani nelineární vazby. |
                | **Extrémně vysoká dimenze (texty, genetika, bio)** | **SVM (Lineární jádro)** | Maximální marže zabraňuje přeučení i při $D \gg N$. |
                | **Složité prostorové klastry a nelinearity** | **SVM (RBF)** nebo **k-NN** | RBF jádro projektuje data do nekonečně-rozměrného prostoru Hilbertových prostorů. |
                | **Potřeba kalibrovaných pravděpodobností** | **Logistická regrese** | Výstupem je reálná aposteriorní pravděpodobnost $P(Y=1|X)$. |
                """
            )
        with diag_col2:
            st.info(
                """
                **Pravidlo zlatého standardu:**
                1. Začněte s **Logistickou regresí** jako spolehlivým lineárním benchmarkem.
                2. Vyzkoušejte **Rozhodovací strom** pro odhalení prahů a důležitosti příznaků.
                3. Nasaďte **SVM** pro maximalizaci přesnosti na náročných hranicích.
                4. V pokročilé fázi zkombinujte stromy do **ansámblů** (Random Forest / Gradient Boosting).
                """
            )

    # ---------------------------------------------------------
    # TAB 3: Evaluační metriky
    # ---------------------------------------------------------
    with tab_metrics:
        st.subheader("Přehled evaluačních metrik klasifikace")
        st.markdown(
            r"""
            Proč nestačí pouhá **Accuracy**? Při nevyvážených datech (např. 95 % zdravých, 5 % nemocných) 
            triviální model predikující vždy „zdravý“ dosáhne $95\,\%$ Accuracy, ale má nulovou lékařskou hodnotu!
            """
        )

        m_col1, m_col2 = st.columns(2)
        with m_col1:
            with st.container(border=True):
                st.markdown("#### Precision (Přesnost / Kladná prediktivní hodnota)")
                st.latex(r"\text{Precision} = \frac{TP}{TP + FP}")
                st.markdown(
                    """
                    - **Význam:** Pokud model označí vzorek jako pozitivní, s jakou jistotou tomu můžeme věřit?
                    - **Priorita:** Minimalizovat **False Positives (FP)**.
                    - **Typické příklady:**
                      - *Spam filtr:* Nechceme smazat důležitý pracovní email jako spam.
                      - *Investiční doporučení:* Nechceme koupit akcii, která zkrachuje.
                    """
                )

            with st.container(border=True):
                st.markdown("#### Recall / Senzitivita (Úplnost)")
                st.latex(r"\text{Recall} = \frac{TP}{TP + FN}")
                st.markdown(
                    """
                    - **Význam:** Kolik procent skutečně nemocných / podvodných případů dokázal model zachytit?
                    - **Priorita:** Minimalizovat **False Negatives (FN)**.
                    - **Typické příklady:**
                      - *Medicínská diagnostika páteře:* Nechceme poslat domů nemocného pacienta jako „zdravého“.
                      - *Detekce podvodů u platebních karet:* Nechceme minout probíhající zneužití karty.
                    """
                )

        with m_col2:
            with st.container(border=True):
                st.markdown("#### F1-Skóre")
                st.latex(r"\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}")
                st.markdown(
                    """
                    - **Význam:** Harmonický průměr Precision a Recall.
                    - **Vlastnost:** Je silně penalizován, pokud jedna ze složek drasticky propadne (na rozdíl od aritmetického průměru).
                    - **Ideální využití:** Univerzální metrika při nevyváženém rozdělení tříd.
                    """
                )

            with st.container(border=True):
                st.markdown("#### ROC-AUC & Log Loss")
                st.markdown(
                    """
                    - **ROC-AUC (Area Under ROC Curve):**
                      - Hodnotí rozlišovací schopnost modelu napříč všemi možnými prahy (od 0 do 1).
                      - Hodnota $0.5$ = náhodné hádání, hodnota $1.0$ = perfektní oddělení tříd.
                    - **Log Loss (Křížová entropie):**
                      - Penalizuje nejistotu a příliš sebevědomé chybné predikce pravděpodobností.
                    """
                )

    # ---------------------------------------------------------
    # TAB 4: Interaktivní kvíz (Quiz Time)
    # ---------------------------------------------------------
    with tab_quiz:
        st.subheader("🧠 Znalostní kvíz: Ověřte si své porozumění 2. dni!")
        st.markdown(
            """
            Odpovězte na následující otázky vycházející z probírané teorie a cvičení 2. dne. 
            Po stisknutí tlačítka získáte okamžité vyhodnocení s detailním vysvětlením.
            """
        )

        quiz_form = st.form(key="day2_quiz_form")
        with quiz_form:
            q1 = st.radio(
                "**1. Který z uvedených algoritmů je tzv. 'lazy learner' a v trénovací fázi pouze ukládá data do paměti bez optimalizace vah?**",
                [
                    "A) Logistická regrese",
                    "B) k-Nearest Neighbors (k-NN)",
                    "C) Support Vector Machine (SVM)",
                    "D) Rozhodovací strom",
                ],
                index=None,
            )

            q2 = st.radio(
                "**2. Co se stane s rozhodovací hranicí modelu k-NN, pokud nastavíme hyperparametr k na extrémně nízkou hodnotu (např. k = 1)?**",
                [
                    "A) Hranice se stane dokonale hladkou a lineární.",
                    "B) Model bude trpět vysokým biasem (podučením).",
                    "C) Hranice bude extrémně členitá a model bude náchylný k přeučení (overfittingu) na šum.",
                    "D) Model přestane fungovat, protože k musí být alespoň 3.",
                ],
                index=None,
            )

            q3 = st.radio(
                "**3. V nemocnici nasazujeme model pro včasné odhalení rakoviny. Minout nemocného pacienta (False Negative) je fatální. Kterou metriku musíme primárně maximalizovat?**",
                [
                    "A) Accuracy",
                    "B) Precision",
                    "C) Recall (Senzitivitu)",
                    "D) Specificitu",
                ],
                index=None,
            )

            q4 = st.radio(
                "**4. Jakou funkci využívá logistická regrese pro transformaci lineární kombinace vstupů do intervalu pravděpodobností (0, 1)?**",
                [
                    "A) Gaussovské RBF jádro",
                    "B) Sigmoidální (logistickou) funkci",
                    "C) ReLU (Rectified Linear Unit)",
                    "D) Kvadratickou ztrátovou funkci OLS",
                ],
                index=None,
            )

            q5 = st.radio(
                "**5. Proč rozhodovací strom v základní podobě nevyžaduje normalizaci ani standardizaci vstupních numerických příznaků?**",
                [
                    "A) Protože strom interně automaticky aplikuje StandardScaler.",
                    "B) Protože rozhodovací podmínka v uzlu závisí pouze na monotónním pořadí hodnot jednoho příznaku, nikoliv na jeho absolutním měřítku.",
                    "C) Protože stromy pracují výhradně s kategorickými proměnnými.",
                    "D) Není to pravda, bez normalizace se strom nenatrénuje.",
                ],
                index=None,
            )

            q6 = st.radio(
                "**6. Co jsou to 'podpůrné vektory' (Support Vectors) v algoritmu SVM?**",
                [
                    "A) Všechny datové body, které leží daleko od rozhodovací hranice.",
                    "B) Váhy násobené gradientem v logistické regresi.",
                    "C) Kritické hraniční trénovací vzorky ležící na okrajích marže, které jako jediné definují polohu oddělující nadroviny.",
                    "D) Vektory reprezentující průměrné středy shluků.",
                ],
                index=None,
            )

            submitted = quiz_form.form_submit_button("Vyhodnotit kvíz", type="primary")

        if submitted:
            score = 0
            total = 6

            st.markdown("### 📋 Výsledky a vysvětlení:")

            # Q1
            if q1 and q1.startswith("B"):
                score += 1
                st.success("✅ **Otázka 1: Správně!** k-NN je typickým příkladem instančního / líného učení (lazy learning).")
            else:
                st.error("❌ **Otázka 1: Nesprávně.** Správná odpověď je **B) k-Nearest Neighbors**. k-NN nepočítá žádné parametry ani vnitřní rovnice, pouze prohledává paměť.")

            # Q2
            if q2 and q2.startswith("C"):
                score += 1
                st.success("✅ **Otázka 2: Správně!** Malé $k=1$ sleduje každý jednotlivý bod v datech včetně šumu a odlehlých hodnot, což vede k přeučení.")
            else:
                st.error("❌ **Otázka 2: Nesprávně.** Správná odpověď je **C) Hranice bude extrémně členitá a náchylná k přeučení**.")

            # Q3
            if q3 and q3.startswith("C"):
                score += 1
                st.success("✅ **Otázka 3: Správně!** Recall ($TP / (TP + FN)$) měří schopnost zachytit pozitivní případy a penalizuje False Negatives.")
            else:
                st.error("❌ **Otázka 3: Nesprávně.** Správná odpověď je **C) Recall (Senzitivitu)**. Zmeškání pacienta je $FN$, proto maximalizujeme Recall.")

            # Q4
            if q4 and q4.startswith("B"):
                score += 1
                st.success(r"✅ **Otázka 4: Správně!** Logistická regrese používá sigmoid $\sigma(z) = 1 / (1 + e^{-z})$.")
            else:
                st.error("❌ **Otázka 4: Nesprávně.** Správná odpověď je **B) Sigmoidální (logistická) funkce**.")

            # Q5
            if q5 and q5.startswith("B"):
                score += 1
                st.success(r"✅ **Otázka 5: Správně!** Podmínka $x_j \le \theta$ dělí data podle seřazeného pořadí hodnot, takže libovolná monotónní transformace měřítka nemá na výsledek vliv.")
            else:
                st.error("❌ **Otázka 5: Nesprávně.** Správná odpověď je **B) Rozhodovací podmínka závisí pouze na monotónním uspořádání hodnot**.")

            # Q6
            if q6 and q6.startswith("C"):
                score += 1
                st.success("✅ **Otázka 6: Správně!** Podpůrné vektory jsou jediné body, které přímo určují marži a orientaci nadroviny v SVM.")
            else:
                st.error("❌ **Otázka 6: Nesprávně.** Správná odpověď je **C) Kritické hraniční vzorky ležící na okrajích marže**.")

            st.markdown("---")
            pct = (score / total) * 100
            if score == total:
                st.balloons()
                st.success(f"🏆 **Gratulujeme! Dosáhli jste plného počtu bodů: {score} z {total} ({pct:.0f} %)!** Máte perfektní přehled o klasifikačních metodách.")
            elif score >= 4:
                st.info(f"👍 **Velmi dobrý výsledek: {score} z {total} ({pct:.0f} %).** Základní principy máte zvládnuté.")
            else:
                st.warning(f"⚠️ **Získali jste {score} z {total} ({pct:.0f} %).** Doporučujeme projít si záložky s přehledem algoritmů a evaluačních metrik.")


if __name__ == "__main__":
    render_day_2_summary_view()
