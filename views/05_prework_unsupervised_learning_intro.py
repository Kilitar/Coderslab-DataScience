"""
Prework Session 3: Úvod do neřízeného učení (Introduction to Unsupervised Learning)
==================================================================================
Interaktivní výukový modul pro přípravu na Den 5 (Session 3):
1. Supervised vs. Unsupervised Learning: Fundamentální rozdíly v trénování a datech.
2. Interaktivní simulátor shlukování aut (Cena vs. Počet prodaných kusů ze zadání).
3. 6 klíčových aplikací neřízeného učení v reálné praxi.
4. Interaktivní Clustering & Anomaly Playground (K-Means, PCA, Isolation Forest).
5. Vědomostní kvíz k ověření pochopení.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import streamlit as st


def render_unsupervised_learning_intro_view():
    st.title("🌐 Prework Session 3: Úvod do neřízeného učení (Unsupervised Learning)")
    st.caption(
        "Příprava na Session 3: Jak algoritmy odhalují skryté zákonitosti, vnitřní strukturu a anomálie "
        "v datech **bez přítomnosti cílové proměnné (bez anotovaných štítků)**. Od shlukování přes redukci dimenzionality až po detekci podvodů."
    )

    # Horní KPI karty
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Vstupní data", "Pouze matice X", delta="Absence cílových štítků y")
    with c2:
        st.metric("Hlavní cíl", "Skrytá struktura", delta="Shluky, latentní prostor, anomálie")
    with c3:
        st.metric("Typické úlohy", "Clustering & PCA", delta="Segmentace & Redukce dimenzí")
    with c4:
        st.metric("Hodnocení kvality", "Vnitřní metriky", delta="Silhouette / Davies-Bouldin")

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "⚖️ 1. Učení s učitelem vs. Bez učitele",
        "🚗 2. Příklad: Shlukování automobilů",
        "🛠️ 3. Šest pilířů využití v praxi",
        "🧪 4. Interaktivní Playground (Clustering & Anomálie)",
        "🎓 5. Vědomostní kvíz"
    ])

    # ==============================================================================
    # TAB 1: SUPERVISED VS UNSUPERVISED
    # ==============================================================================
    with tab1:
        st.subheader("1. Fundamentální rozdíl: Učení s učitelem vs. Učení bez učitele")

        st.markdown(
            """
            Během prvních čtyř dnů kurzu jsme se věnovali **učení s učitelem (Supervised Learning)**, 
            kde algoritmus dostává dvojice $(X, y)$ a učí se mapovací funkci $f(X) \approx y$.
            
            V **neřízeném učení (Unsupervised Learning)** máme k dispozici **pouze nezávislé proměnné $X$**. 
            Algoritmus nemá k dispozici žádnou „správnou odpověď“ ani chybový signál od učitele. 
            Jeho úkolem je autonomně extrahovat vnitřní geometrii, shluky a statistické hustoty v datech.
            """
        )

        col_flow1, col_flow2 = st.columns(2)
        with col_flow1:
            st.markdown("#### 🎯 Supervised Learning (S učitelem)")
            st.info(
                r"""
                - **Vstup:** Příznaky $X$ a známý cíl $y$ (kategorie či spojitá hodnota).
                - **Cíl:** Minimalizovat predikční chybu: $\mathcal{L}(y, \hat{y}) \to \min$.
                - **Zpětná vazba:** Explicitní (model okamžitě ví, o kolik se spletl).
                - **Výstup:** Predikce štítku (Klasifikace) nebo čísla (Regrese).
                """
            )
        with col_flow2:
            st.markdown("#### 🔍 Unsupervised Learning (Bez učitele)")
            st.success(
                """
                - **Vstup:** Pouze příznaky $X$ bez cílové proměnné.
                - **Cíl:** Nalézt přirozené shluky, zredukovat dimenze nebo detekovat odlehlé body.
                - **Zpětná vazba:** Žádná (algoritmus optimalizuje vnitřní metriku – např. rozptyl či vzdálenost k centroidu).
                - **Výstup:** Přiřazení do shluku, nový nízkorozměrný prostor nebo index anomálie.
                """
            )

        st.markdown("#### 📊 Přehledné srovnání vlastností (z materiálů kurzu)")
        cmp_df = pd.DataFrame([
            {
                "Kritérium": "Definice",
                "Unsupervised learning": "Probíhá bez dozoru. Hledá skryté vzorce a geometrii v datech.",
                "Supervised learning": "Probíhá s dozorem. Učí se vztah mezi vstupem X a zadaným výstupem Y."
            },
            {
                "Kritérium": "Vstupní data",
                "Unsupervised learning": "Neanotovaná data (pouze X, chybí štítky či cílové hodnoty).",
                "Supervised learning": "Anotovaná data (každému pozorování odpovídá známý štítek či hodnota)."
            },
            {
                "Kritérium": "Zpracování dat",
                "Unsupervised learning": "Model přijímá pouze X a hledá vzájemné vzdálenosti a hustoty.",
                "Supervised learning": "Model přijímá X i Y a optimalizuje parametry pro predikci Y z X."
            },
            {
                "Kritérium": "Typy problémů",
                "Unsupervised learning": "Shlukování (Clustering), redukce dimenzionality, asociační pravidla.",
                "Supervised learning": "Klasifikace a regrese."
            },
            {
                "Kritérium": "Metriky úspěšnosti",
                "Unsupervised learning": "Subjektivní / Heuristické (Silhouette skóre, vysvětlená variance).",
                "Supervised learning": "Objektivní a přesné (Accuracy, F1, MSE, RMSE, R²)."
            },
            {
                "Kritérium": "Příklady algoritmů",
                "Unsupervised learning": "K-Means, DBSCAN, Hierarchical Clustering, PCA, Isolation Forest.",
                "Supervised learning": "Lineární/Logistická regrese, Decision Tree, Random Forest, XGBoost, SVM."
            },
            {
                "Kritérium": "Příklady využití",
                "Unsupervised learning": "Segmentace zákazníků, doporučovací systémy, detekce podvodů (fraud).",
                "Supervised learning": "Predikce cen, klasifikace spamu, rozpoznávání obrazu, medicínská diagnostika."
            }
        ])
        st.dataframe(cmp_df, hide_index=True, width="stretch")

    # ==============================================================================
    # TAB 2: PŘÍKLAD SHLUKOVÁNÍ AUTOMOBILŮ
    # ==============================================================================
    with tab2:
        st.subheader("2. Příklad z kurzu: Shlukování automobilů dle ceny a prodejů")

        st.markdown(
            """
            Představme si situaci popsanou v materiálech: Máme data o automobilech, kde známe pouze 
            **průměrnou cenu modelu** a **počet prodaných kusů**. Neznáme však žádné detaily o tržních segmentech.
            
            Pomocí shlukovacího algoritmu (např. **K-Means**) můžeme data rozdělit do homogenních skupin bez jakéhokoliv předchozího štítkování.
            """
        )

        # Generování realistických dat ze zadání
        np.random.seed(42)
        # Cluster 1: Levná masová auta (nízká cena, vysoké prodeje)
        c1_price = np.random.normal(18000, 3500, 60)
        c1_sales = np.random.normal(12000, 2500, 60)
        # Cluster 2: Střední třída / Rodinné vozy (střední cena, střední prodeje)
        c2_price = np.random.normal(42000, 6000, 50)
        c2_sales = np.random.normal(5500, 1500, 50)
        # Cluster 3: Luxusní / Prémiové vozy (vysoká cena, nízké prodeje)
        c3_price = np.random.normal(85000, 12000, 35)
        c3_sales = np.random.normal(1200, 400, 35)

        car_df = pd.DataFrame({
            "price": np.concatenate([c1_price, c2_price, c3_price]),
            "sales": np.concatenate([c1_sales, c2_sales, c3_sales])
        })
        car_df["price"] = np.clip(car_df["price"], 8000, 140000)
        car_df["sales"] = np.clip(car_df["sales"], 100, 20000)

        col_cars_ctrl, col_cars_plot = st.columns([0.8, 1.2])

        with col_cars_ctrl:
            st.markdown("#### 🎛️ Nastavení shlukování")
            k_clusters = st.slider("Počet shluků K (požadované segmenty)", min_value=2, max_value=5, value=3)
            show_unlabeled = st.checkbox("Zobrazit původní neoznačená data", value=False)

            # Fit K-Means
            km = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
            car_df["cluster"] = km.fit_predict(car_df[["price", "sales"]])
            cluster_names = {
                0: "Segment A", 1: "Segment B", 2: "Segment C", 3: "Segment D", 4: "Segment E"
            }
            car_df["segment"] = car_df["cluster"].map(cluster_names)

            st.markdown("#### 💡 Co nám shlukování odhalilo?")
            st.write(
                f"""
                - Algoritmus autonomně rozdělil **{len(car_df)} modelů aut** do **{k_clusters} homogenních segmentů**.
                - **Využití v Feature Engineeringu:** Výsledný štítek `segment` můžeme nyní přidat jako **nový kategorický příznak** 
                  do následného klasifikačního modelu (např. předpověď značky či marže).
                """
            )

        with col_cars_plot:
            if show_unlabeled:
                fig_car = px.scatter(
                    car_df, x="price", y="sales",
                    title="Původní neanotovaná data (Pouze cena vs. prodeje)",
                    labels={"price": "Průměrná cena modelu [USD]", "sales": "Počet prodaných kusů za rok"},
                    color_discrete_sequence=["#64748b"]
                )
            else:
                fig_car = px.scatter(
                    car_df, x="price", y="sales", color="segment",
                    title=f"Výsledek K-Means shlukování (K = {k_clusters})",
                    labels={"price": "Průměrná cena modelu [USD]", "sales": "Počet prodaných kusů za rok"},
                    color_discrete_sequence=["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6"]
                )
                # Vykreslení centroidů
                centroids = km.cluster_centers_
                fig_car.add_trace(go.Scatter(
                    x=centroids[:, 0], y=centroids[:, 1],
                    mode="markers",
                    marker=dict(size=16, color="black", symbol="x", line=dict(width=2)),
                    name="Centroidy shluků"
                ))

            fig_car.update_layout(template="plotly_white", height=420)
            st.plotly_chart(fig_car, width="stretch")

    # ==============================================================================
    # TAB 3: ŠEST PILÍŘŮ VYUŽITÍ V PRAXI
    # ==============================================================================
    with tab3:
        st.subheader("3. Šest klíčových oblastí využití neřízeného učení v reálném byznysu")

        p_col1, p_col2 = st.columns(2)

        with p_col1:
            st.markdown("#### 1️⃣ Data Mining & Explorativní analýza (EDA)")
            st.write(
                "Před trénováním prediktivních modelů pomáhá neřízené učení odhalit přirozenou strukturu dat, "
                "korelační vazby a skryté podskupiny. Zlepšuje kvalitu datové přípravy před nasazením řízených modelů."
            )

            st.markdown("#### 2️⃣ Redukce dimenzionality (PCA, t-SNE, UMAP)")
            st.write(
                "Transformace vysokodimenzionálních dat (např. 10 000 genů či 512D textových embeddingů) "
                "do několika málo hlavních komponent při **zachování maximální variance**. "
                "Zrychluje trénování modelů a eliminuje tzv. *prokletí dimenzionality (Curse of Dimensionality)*."
            )

            st.markdown("#### 3️⃣ Detekce anomálií & Podvodů (Fraud Detection)")
            st.write(
                "Algoritmy jako **Isolation Forest** nebo **Local Outlier Factor (LOF)** detekují vzorky, "
                "které se výrazně liší od běžného chování. Využívá se ve financích (zablokování podezřelé karetní transakce), "
                "v průmyslu (detekce vadného ložiska) i v IT bezpečnosti (odhalení kybernetického útoku)."
            )

        with p_col2:
            st.markdown("#### 4️⃣ Segmentace zákazníků (Customer Segmentation)")
            st.write(
                "E-shopy a banky seskupují zákazníky dle nákupního chování, obratu a frekvence návštěv (RFM analýza). "
                "Marketingové týmy pak mohou cílit personalizované kampaně na míru každému segmentu "
                "(např. 'lovci slev' vs. 'bonitní prémioví klienti')."
            )

            st.markdown("#### 5️⃣ Doporučovací systémy (Recommender Systems)")
            st.write(
                "Analýza podobnosti uživatelů a položek (Collaborative Filtering). Pokud uživatelé A a B v minulosti "
                "sledovali stejné filmy a uživatel A si právě pustil novinku, systém ji automaticky doporučí i uživateli B."
            )

            st.markdown("#### 6️⃣ Generování nových dat (Generative AI & GANs)")
            st.write(
                "Generativní modely (Generative Adversarial Networks – GAN, VAE, difúzní modely) "
                "se učí skrytou distribuci reálných dat a dokážou generovat syntetické trénovací vzorky, "
                "fotorealistické obrazy, hlas či textové formulace."
            )

    # ==============================================================================
    # TAB 4: INTERAKTIVNÍ PLAYGROUND
    # ==============================================================================
    with tab4:
        st.subheader("4. Interaktivní laboratoř: Clustering vs. Detekce anomálií")

        st.markdown(
            "Vyzkoušejte si, jak různé algoritmy neřízeného učení reagují na rozložení dat. "
            "Můžete přepínat mezi **K-Means shlukováním** a **Isolation Forest detekcí anomálií**."
        )

        pg_c1, pg_c2 = st.columns([0.8, 1.2])

        with pg_c1:
            st.markdown("#### ⚙️ Generátor dat")
            data_type = st.selectbox("Typ geometrie dat:", ["Syntetické shluky (Blobs)", "Data s anomáliemi (Outliers)"])
            n_samples = st.slider("Počet vzorků N:", min_value=100, max_value=600, value=300, step=50)
            noise_level = st.slider("Úroveň šumu:", min_value=0.5, max_value=3.0, value=1.2, step=0.1)

            np.random.seed(42)
            if data_type == "Syntetické shluky (Blobs)":
                n_per_cluster = n_samples // 3
                b1 = np.random.normal(loc=[-4, -3], scale=noise_level, size=(n_per_cluster, 2))
                b2 = np.random.normal(loc=[3, 4], scale=noise_level, size=(n_per_cluster, 2))
                b3 = np.random.normal(loc=[5, -4], scale=noise_level, size=(n_samples - 2 * n_per_cluster, 2))
                X_synth = np.vstack([b1, b2, b3])
            else:
                # Regular dense center + sparse outliers
                n_normal = int(n_samples * 0.9)
                n_out = n_samples - n_normal
                normal_pts = np.random.normal(loc=[0, 0], scale=noise_level, size=(n_normal, 2))
                outlier_pts = np.random.uniform(low=-10, high=10, size=(n_out, 2))
                X_synth = np.vstack([normal_pts, outlier_pts])

            mode = st.radio("Zvolte algoritmus:", ["K-Means Clustering", "Isolation Forest (Anomálie)"])

        with pg_c2:
            if mode == "K-Means Clustering":
                k_val = st.slider("Počet shluků K:", 2, 6, 3)
                km_pg = KMeans(n_clusters=k_val, random_state=42, n_init=10)
                labels_pg = km_pg.fit_predict(X_synth)

                plot_df = pd.DataFrame(X_synth, columns=["X1", "X2"])
                plot_df["Shluk"] = [f"Shluk {l+1}" for l in labels_pg]

                fig_pg = px.scatter(
                    plot_df, x="X1", y="X2", color="Shluk",
                    title=f"K-Means Clustering (K = {k_val})",
                    color_discrete_sequence=px.colors.qualitative.Plotly
                )
                centers = km_pg.cluster_centers_
                fig_pg.add_trace(go.Scatter(
                    x=centers[:, 0], y=centers[:, 1],
                    mode="markers",
                    marker=dict(size=14, color="black", symbol="x", line=dict(width=2)),
                    name="Centroidy"
                ))
            else:
                contamination = st.slider("Předpokládaný podíl anomálií (Contamination):", 0.01, 0.20, 0.08, step=0.01)
                iso = IsolationForest(contamination=contamination, random_state=42)
                preds_iso = iso.fit_predict(X_synth)

                plot_df = pd.DataFrame(X_synth, columns=["X1", "X2"])
                plot_df["Status"] = np.where(preds_iso == -1, "🔴 Anomálie (Outlier)", "🟢 Běžný vzorek (Inlier)")

                fig_pg = px.scatter(
                    plot_df, x="X1", y="X2", color="Status",
                    title=f"Isolation Forest – Detekce anomálií (Detekováno {(preds_iso == -1).sum()} anomálií)",
                    color_discrete_map={"🟢 Běžný vzorek (Inlier)": "#3b82f6", "🔴 Anomálie (Outlier)": "#ef4444"}
                )

            fig_pg.update_layout(template="plotly_white", height=420)
            st.plotly_chart(fig_pg, width="stretch")

    # ==============================================================================
    # TAB 5: VĚDOMOSTNÍ KVÍZ
    # ==============================================================================
    with tab5:
        st.subheader("5. Vědomostní kvíz: Ověřte si pochopení konceptů")

        q1 = st.radio(
            "1. Jaký je hlavní rozdíl ve vstupních datech mezi učením s učitelem a bez učitele?",
            [
                "A) Učení bez učitele vyžaduje textová data, zatímco učení s učitelem tabulková.",
                "B) Učení bez učitele nemá k dispozici cílové štítky (výstupy y), model pracuje pouze s maticí X.",
                "C) Učení bez učitele nepoužívá matice čísel, ale pouze relační databáze."
            ],
            key="q1"
        )
        if st.button("Ověřit otázku 1"):
            if q1.startswith("B)"):
                st.success("Správně! V neřízeném učení chybí jakýkoliv cílový vektor y.")
            else:
                st.error("Nesprávně. Klíčovým rozdílem je nepřítomnost cílových štítků y.")

        st.markdown("---")

        q2 = st.radio(
            "2. Jak může výsledek shlukování (clustering) pomoci následnému modelu řízeného učení?",
            [
                "A) Nijak, výstupy neřízeného učení se nesmí kombinovat s řízeným.",
                "B) Shluky lze převést na nový sloupec (nový příznak / segment), který zvýší predikční sílu klasifikátoru.",
                "C) Shlukování automaticky smaže všechny chybějící hodnoty v datech."
            ],
            key="q2"
        )
        if st.button("Ověřit otázku 2"):
            if q2.startswith("B)"):
                st.success("Přesně tak! Číslo shluku slouží jako nový inženýrský příznak (Feature Engineering).")
            else:
                st.error("Nesprávně. Číslo přiřazeného shluku je vynikající doplňující příznak.")

        st.markdown("---")

        q3 = st.radio(
            "3. Který z následujících algoritmů patří mezi metody neřízeného učení?",
            [
                "A) Linear Regression",
                "B) Random Forest Classifier",
                "C) Isolation Forest a Principal Component Analysis (PCA)"
            ],
            key="q3"
        )
        if st.button("Ověřit otázku 3"):
            if q3.startswith("C)"):
                st.success("Výborně! PCA i Isolation Forest pracují bez cílových anotací.")
            else:
                st.error("Chyba. Lineární regrese i Random Forest vyžadují cílovou proměnnou y.")


if __name__ == "__main__":
    render_unsupervised_learning_intro_view()
