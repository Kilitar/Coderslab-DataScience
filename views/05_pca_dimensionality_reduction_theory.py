"""
Den 5: Neřízené učení – Redukce dimenzionality (PCA) – Teorie & Matematika
==========================================================================
Syntéza materiálů z Dimensionality_reduction_-_introduction.pdf:
1. Prokletí dimenzionality (Curse of Dimensionality) a proč redukovat dimenze.
2. Matematický algoritmus PCA krok za krokem: Centrování, kovarianční matice,
   vlastní čísla (Eigenvalues), vlastní vektory (Eigenvectors) a projekce.
3. Kompletní ruční výpočet na příkladu ze slidů (4 studenti x 3 zkoušky).
4. Interaktivní vizuální geometrie PCA: Hledání směru maximálního rozptylu.
5. Vědomostní kontrola klíčových pojmů.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.title("📖 Den 5: Redukce dimenzionality & Matematika PCA")
st.caption(
    "Teoretický průvodce metodou analýzy hlavních komponent (**Principal Component Analysis – PCA**): "
    "Odstranění šumu, řešení prokletí dimenzionality, kovarianční matice, spektrální rozklad a projekce dat do ortogonální báze."
)

# Horní KPI karty
c1, c2, c3, c4 = st.columns(4)
c1.metric("1. Motivace", "Curse of Dimensionality", delta="Řídkost prostoru při D > 3", delta_color="inverse")
c2.metric("2. Cíl PCA", "Maximalizace rozptylu", delta="Zachování klíčové variance", delta_color="normal")
c3.metric("3. Matematické jádro", "Kovarianční matice", delta="Eigenvalues & Eigenvectors", delta_color="normal")
c4.metric("4. Vlastnost bází", "Ortogonalita", delta="Nekorelované složky (r = 0)", delta_color="off")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "🌌 1. Prokletí dimenzionality & Proč redukovat",
    "📐 2. Matematický postup PCA krok za krokem",
    "🎓 3. Příklad z přednášky: Výsledky studentů",
    "🧭 4. Geometrická intuice: Rotace a rozptyl"
])

# ==============================================================================
# TAB 1: PROKLETÍ DIMENZIONALITY
# ==============================================================================
with tab1:
    st.subheader("1. Proč redukujeme dimenzionalitu a co je to 'Prokletí dimenzionality'?")

    st.markdown(
        """
        V moderní datové vědě běžně pracujeme s datasety majícími desítky, stovky až tisíce příznaků 
        (např. finanční indikátory, medicínské biomarkery nebo textové embeddingy). 
        S rostoucím počtem dimenzí $D$ však narážíme na fundamentální matematické a výpočetní překážky:
        """
    )

    col_curse1, col_curse2 = st.columns(2)

    with col_curse1:
        st.markdown("#### ⚠️ Problémy vysoké dimenzionality:")
        st.write(
            r"""
            1. **Exponenciální nárůst objemu prostoru:**  
               S každou přidanou dimenzí objem prostoru roste exponenciálně ($V \propto r^D$). Data se stávají extrémně řídká (*sparse*).
            2. **Ztráta rozlišovací schopnosti vzdáleností:**  
               V Eukleidovském prostoru se vzdálenosti mezi nejbližším a nejvzdálenějším bodem relativně vyrovnávají, což zhoršuje přesnost modelů jako k-NN, SVM či K-Means.
            3. **Riziko přeučení (Overfitting):**  
               Vysoký počet proměnných umožňuje modelu naučit se náhodný šum místo skutečného signálu.
            4. **Nemožnost vizualizace:**  
               Lidský mozek dokáže vnímat nanejvýš 3 prostorové dimenze. Data o 30 či 100 sloupcích nelze přímo zobrazit.
            """
        )

    with col_curse2:
        st.markdown("#### 🎯 Co přináší redukce dimenzionality?")
        st.success(
            """
            - **Odstranění šumu a redundantních proměnných:** Pokud dvě proměnné silně korelují, PCA je sloučí do jedné dominantní komponenty.
            - **Extrémní zrychlení výpočtů:** Zmenšení matice příznaků zkracuje čas trénování algoritmů o řády.
            - **Zvýšení generalizace:** Snížením stupňů volnosti nutíme model soustředit se na podstatný signál.
            - **Vizuální explorace:** Redukce na 2 nebo 3 komponenty umožňuje tvorbu 2D/3D scatter plotů s jasnou interpretací.
            """
        )

    st.markdown("---")
    st.markdown("### 📊 Demonstrace řídnutí prostoru se vzrůstající dimenzí")
    
    dims = [1, 2, 3, 5, 10, 20, 50, 100]
    # Poměr objemu jednotkové koule k jednotkové krychli
    from scipy.special import gamma
    ratios = [(np.pi**(d/2) / gamma(d/2 + 1)) / (2**d) for d in dims]
    
    fig_curse = go.Figure()
    fig_curse.add_trace(go.Scatter(
        x=dims, y=ratios, mode="lines+markers",
        line=dict(color="#ef4444", width=3),
        marker=dict(size=8, color="#b91c1c")
    ))
    fig_curse.update_layout(
        title="Propad relativního objemu vepsané sféry (ilustrace prázdnoty vysokodimenzionálního prostoru)",
        xaxis_title="Počet dimenzí (D)",
        yaxis_title="Podíl objemu sféry ku krychli",
        yaxis_type="log",
        template="plotly_white",
        height=320,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig_curse, width="stretch")

# ==============================================================================
# TAB 2: MATEMATICKÝ POSTUP KROK ZA KROKEM
# ==============================================================================
with tab2:
    st.subheader("2. Algoritmus PCA: Matematický aparát krok za krokem")

    st.markdown(
        r"""
        Metoda **PCA (Principal Component Analysis)** transformuje původní sadu proměnných $X \in \mathbb{R}^{N \times D}$ 
        do nového souřadného systému $Z \in \mathbb{R}^{N \times k}$ (kde $k \le D$), přičemž nové osy 
        (hlavní komponenty) jsou lineárními kombinacemi původních proměnných, jsou vzájemně **ortogonální (nekorelované)** 
        a jsou seřazeny sestupně podle velikosti vysvětleného rozptylu.
        """
    )

    s1, s2 = st.columns(2)

    with s1:
        st.markdown("#### Krok 1: Výpočet průměru a centrování dat")
        st.markdown(
            r"""
            Pro každý sloupec $j \in \{1, \dots, D\}$ spočítáme průměr $\bar{X}_j$:
            $$\bar{X}_j = \frac{1}{N} \sum_{i=1}^{N} X_{i,j}$$
            Od každé hodnoty odečteme průměr (centrování do počátku souřadnic):
            $$X_{\text{centered}} = X - \bar{X}$$
            *(Poznámka: Pokud mají příznaky různé fyzikální jednotky, je nutné provést i dělení směrodatnou odchylkou $\sigma_j$ – tzv. standardizaci).*
            """
        )

        st.markdown(r"#### Krok 2: Výpočet kovarianční matice $\mathbf{\Sigma}$")
        st.markdown(
            r"""
            Kovariance mezi dvěma proměnnými $X$ a $Y$ měří jejich lineární provázanost:
            $$\text{cov}(X, Y) = \frac{1}{N} \sum_{i=1}^{N} (X_i - \bar{X})(Y_i - \bar{Y})$$
            V maticovém vyjádření je čtvercová kovarianční matice rozměru $D \times D$:
            $$\mathbf{\Sigma} = \frac{1}{N} X_{\text{centered}}^T X_{\text{centered}}$$
            Na diagonále leží rozptyly jednotlivých proměnných, mimo diagonálu kovariance mezi dvojicemi.
            """
        )

    with s2:
        st.markdown("#### Krok 3: Spektrální rozklad (Vlastní čísla & Vektory)")
        st.markdown(
            r"""
            Hledáme vektory $\mathbf{v}$ a skaláry $\lambda$, které splňují charakteristickou rovnici:
            $$\mathbf{\Sigma} \mathbf{v} = \lambda \mathbf{v} \iff (\mathbf{\Sigma} - \lambda \mathbf{I}) \mathbf{v} = \mathbf{0}$$
            - **Vlastní vektor ($\mathbf{v}_j$):** Udává **směr** $j$-té hlavní komponenty v prostoru. Vektory jsou jednotkové ($\|\mathbf{v}_j\| = 1$) a navzájem kolmé ($\mathbf{v}_i \cdot \mathbf{v}_j = 0$).
            - **Vlastní číslo ($\lambda_j$):** Udává **množství rozptylu** (varianci), které tato komponenta zachycuje.
            """
        )

        st.markdown("#### Krok 4: Seřazení a projekce do podprostoru")
        st.markdown(
            r"""
            1. Seřadíme vlastní čísla sestupně: $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_D$.
            2. Vybereme prvních $k$ vlastních vektorů a sestavíme projekční matici $\mathbf{W} \in \mathbb{R}^{D \times k}$:
               $$\mathbf{W} = [\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_k]$$
            3. Původní data promítneme do nového prostoru:
               $$\mathbf{Z} = X_{\text{centered}} \cdot \mathbf{W}$$
               Výsledná matice $\mathbf{Z} \in \mathbb{R}^{N \times k}$ má požadovanou nižší dimenzi!
            """
        )

# ==============================================================================
# TAB 3: PŘÍKLAD ZE SLIDŮ (VÝSLEDKY STUDENTŮ)
# ==============================================================================
with tab3:
    st.subheader("3. Kompletní ruční výpočet na příkladu ze slidů (4 studenti x 3 zkoušky)")

    st.markdown(
        """
        V prezentačních materiálech (`Dimensionality_reduction_-_introduction.pdf`, str. 15–17) je uveden 
        názorný studijní příklad: Výsledky **4 studentů** ve **3 zkouškách** kurzu Data Analyst:
        1. *Introduction to Data Analysis*
        2. *SQL*
        3. *Data Visualization*
        """
    )

    # Přesná data z PDF
    student_data = pd.DataFrame({
        "Student": [1, 2, 3, 4],
        "Intro_Data_Analysis": [60.0, 70.0, 80.0, 70.0],
        "SQL": [100.0, 60.0, 80.0, 90.0],
        "Data_Visualization": [90.0, 70.0, 80.0, 60.0]
    })

    st.markdown("#### 📋 Původní matice bodů studentů")
    st.dataframe(student_data, hide_index=True, width="stretch")

    c_calc1, c_calc2 = st.columns(2)

    with c_calc1:
        st.markdown(r"##### 1. Průměrné hodnoty zkoušek $\bar{X}$ (ze slidu 15):")
        means = student_data[["Intro_Data_Analysis", "SQL", "Data_Visualization"]].mean()
        m_intro = means['Intro_Data_Analysis']
        m_sql = means['SQL']
        m_vis = means['Data_Visualization']
        st.markdown(
            rf"""
            - $\bar{{X}}_{{\text{{Intro}}}} = \frac{{60 + 70 + 80 + 70}}{{4}} = \mathbf{{{m_intro:.1f}}}$
            - $\bar{{X}}_{{\text{{SQL}}}} = \frac{{100 + 60 + 80 + 90}}{{4}} = \mathbf{{{m_sql:.1f}}}$
            - $\bar{{X}}_{{\text{{Vis}}}} = \frac{{90 + 70 + 80 + 60}}{{4}} = \mathbf{{{m_vis:.1f}}}$
            
            Vektor průměrů: $\bar{{\mathbf{{X}}}} = [70.0, \; 82.5, \; 75.0]$
            """
        )

        st.markdown(r"##### 2. Kovarianční matice $\mathbf{\Sigma}$ (ze slidu 16):")
        # Výpočet kovariance s N (populační, jak je v PDF formuli 1/N)
        X_vals = student_data[["Intro_Data_Analysis", "SQL", "Data_Visualization"]].values
        X_cent = X_vals - means.values
        cov_matrix = np.dot(X_cent.T, X_cent) / len(student_data)

        st.markdown(
            rf"""
            $$\mathbf{{\Sigma}} = \begin{{bmatrix}} 
            {cov_matrix[0,0]:.2f} & {cov_matrix[0,1]:.2f} & {cov_matrix[0,2]:.2f} \\
            {cov_matrix[1,0]:.2f} & {cov_matrix[1,1]:.2f} & {cov_matrix[1,2]:.2f} \\
            {cov_matrix[2,0]:.2f} & {cov_matrix[2,1]:.2f} & {cov_matrix[2,2]:.2f}
            \end{bmatrix}$$
            *(Na diagonále jsou rozptyly zkoušek: 66.67, 291.67, 166.67).*
            """
        )

    with c_calc2:
        st.markdown("##### 3. Vlastní čísla a vektory (ze slidu 17):")
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        # Seřadit sestupně
        idx_sort = np.argsort(eigenvalues)[::-1]
        eig_vals_sorted = eigenvalues[idx_sort]
        eig_vecs_sorted = eigenvectors[:, idx_sort]

        ev0 = eig_vals_sorted[0]
        ev1 = eig_vals_sorted[1]
        ev2 = eig_vals_sorted[2]
        ev_sum = eig_vals_sorted.sum()

        st.markdown(
            rf"""
            Vlastní čísla odpovídající rozptylu v jednotlivých směrech:
            - $\lambda_1 = \mathbf{{{ev0:.3f}}}$ (cca 67.6 % celkové variance)
            - $\lambda_2 = \mathbf{{{ev1:.3f}}}$ (cca 23.8 % celkové variance)
            - $\lambda_3 = \mathbf{{{ev2:.3f}}}$ (cca 8.6 % celkové variance)
            
            Celkový rozptyl: $\sum \lambda_i = {ev_sum:.2f}$
            """
        )

        st.markdown("##### 4. Výsledná redukce do 2D (PC1 a PC2):")
        W_2 = eig_vecs_sorted[:, :2]
        Z_projected = np.dot(X_cent, W_2)

        proj_df = pd.DataFrame({
            "Student": student_data["Student"],
            "PC1": np.round(Z_projected[:, 0], 3),
            "PC2": np.round(Z_projected[:, 1], 3)
        })
        st.dataframe(proj_df, hide_index=True, width="stretch")
        st.caption("Původní 3D prostor zkoušek byl beze ztráty podstatné informace transformován do 2D roviny!")

# ==============================================================================
# TAB 4: GEOMETRICKÁ INTUICE
# ==============================================================================
with tab4:
    st.subheader("4. Geometrická intuice: Jak PCA nachází směr maximálního rozptylu")

    st.markdown(
        """
        Představte si mračno bodů ve 2D. Algoritmus PCA hledá přímku (vektor), na kterou když body kolmo promítneme, 
        bude **rozptyl promítnutých bodů co největší** (a součet čtverců vzdáleností bodů od přímky co nejmenší).
        """
    )

    # Interaktivní generátor rotace
    np.random.seed(42)
    x_base = np.random.normal(0, 3, 150)
    y_base = 0.75 * x_base + np.random.normal(0, 1, 150)
    demo_df = pd.DataFrame({"X": x_base, "Y": y_base})

    # Výpočet PCA směrů
    cov_demo = np.cov(demo_df.values.T)
    evals_demo, evecs_demo = np.linalg.eigh(cov_demo)
    idx_d = np.argsort(evals_demo)[::-1]
    v1 = evecs_demo[:, idx_d[0]] * np.sqrt(evals_demo[idx_d[0]]) * 2
    v2 = evecs_demo[:, idx_d[1]] * np.sqrt(evals_demo[idx_d[1]]) * 2

    fig_geo = px.scatter(
        demo_df, x="X", y="Y",
        title="Hlavní komponenty jako nové osy souřadného systému",
        color_discrete_sequence=["#94a3b8"]
    )
    # Vykreslení PC1 šipky
    fig_geo.add_annotation(
        x=v1[0], y=v1[1], ax=0, ay=0,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=3, arrowsize=1.5, arrowwidth=3, arrowcolor="#ef4444",
        text="PC1 (Max rozptyl)"
    )
    # Vykreslení PC2 šipky
    fig_geo.add_annotation(
        x=v2[0], y=v2[1], ax=0, ay=0,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=3, arrowsize=1.5, arrowwidth=3, arrowcolor="#3b82f6",
        text="PC2 (Ortogonální směr)"
    )
    fig_geo.update_layout(template="plotly_white", height=420)
    st.plotly_chart(fig_geo, width="stretch")

    st.info(
        "💡 **Klíčový závěr:** První hlavní komponenta (**červená PC1**) míří ve směru největšího roztažení dat. "
        "Druhá komponenta (**modrá PC2**) je na ni striktně kolmá (ortogonální) a zachycuje zbývající rozptyl."
    )
