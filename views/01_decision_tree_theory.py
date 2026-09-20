from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

base_dir = Path(__file__).resolve().parent.parent
theory_md_path = base_dir / "01_Regression" / "theory" / "06_decision_tree_regression_theory.md"

st.title("🌳 Teorie 6: Rozhodovací strom v regresi (Decision Tree)")
st.caption("Interaktivní průvodce neparametrickým modelováním: Po částech konstantní schodovitá regrese, kletba přeučení (max_depth), pravoúhlé řezy v 2D prostoru a limity extrapolace.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📖 Teoretický rozbor",
    "💡 Příklad ze života: Jak strom přemýšlí",
    "🧪 Simulátor: Schodovitá regrese & Hloubka stromu",
    "📐 2D Rozřezání prostoru (Hyperboxy)",
    "🚨 Past extrapolace (Proč strom neumí trendy)"
])

# =============================================================================
# TAB 1: TEORETICKÝ TEXT Z MARKDOWNU
# =============================================================================
with tab1:
    if theory_md_path.exists():
        with open(theory_md_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.warning("Dokument 06_decision_tree_regression_theory.md nebyl nalezen.")

# =============================================================================
# TAB 2: PŘÍKLAD ZE ŽIVOTA PRO NOVÁČKY (ODHAD PLATU V IT)
# =============================================================================
with tab2:
    st.markdown("### 💡 Jak strom doopravdy přemýšlí? Příklad ze života bez složité matematiky")
    st.markdown(r"""
    Představte si, že jste **personalista nebo vedoucí týmu** a máte určit férový nástupní plat pro nového uchazeče v Data Science / IT.
    
    Jak postupuje lidský mozek? **Nepočítá žádné derivace ani složité funkce.**  
    Místo toho si v hlavě pokládá sérii logických otázek typu **ANO / NE**:
    1. *„Má uchazeč alespoň 2.5 roku praxe?“*  
       $\to$ Pokud **NE**, spadá do kategorie **Junior** se základní sazbou.  
       $\to$ Pokud **ANO**, jdeme na další otázku.
    2. *„Ovládá uchazeč produkční Cloud, Docker nebo MLOps?“*  
       $\to$ Pokud **NE**, spadá do pásma **Standardní analytik / Medior**.  
       $\to$ Pokud **ANO**, posouvá se do pásma **Seniorní Cloud / ML specialista** s prémiovým ohodnocením.
    
    Přesně toto je podstata **Rozhodovacího stromu pro regresi (DecisionTreeRegressor)**:
    - **Otázky v uzlech:** Strom automaticky najde nejlepší prahové hodnoty (např. `Praxe <= 2.5 roku`).
    - **Škatulky (Listy):** Každý uchazeč propadne podle svých vlastností do jedné konkrétní škatulky.
    - **Predikce (Číslo):** Odhadem je **průměrný plat všech lidí**, kteří do této škatulky v historických datech spadli!
    """)

    with st.container(border=True):
        st.markdown("#### 🎯 Vyzkoušejte si to: Interaktivní kalkulačka platu v IT")
        col_u1, col_u2, col_u3 = st.columns(3)
        with col_u1:
            user_exp = st.slider("Vaše roky praxe v oboru:", min_value=0.0, max_value=12.0, value=3.5, step=0.5)
        with col_u2:
            user_cloud = st.radio("Zkušenost s Cloudem / Dockerem / MLOps:", ["Ne (pouze lokální skripty)", "Ano (produkční nasazení)"], horizontal=True)
        with col_u3:
            user_role = st.selectbox("Zaměření role:", ["Data Analyst", "Data Scientist / ML Engineer"])

    # Generování realistické syntetické populace (180 IT profesionálů)
    np.random.seed(42)
    n_staff = 180
    exp_data = np.random.uniform(0.0, 11.5, n_staff)
    cloud_data = np.random.choice([0, 1], size=n_staff, p=[0.55, 0.45])
    role_data = np.random.choice([0, 1], size=n_staff, p=[0.50, 0.50])

    base_salary = 42000
    salary_data = (
        base_salary 
        + np.where(exp_data < 2.5, exp_data * 5000, 12500 + (exp_data - 2.5) * 8000)
        + cloud_data * (18000 + exp_data * 3500)
        + role_data * 8000
        + np.random.normal(0, 4500, n_staff)
    )

    df_salary = pd.DataFrame({
        "praxe": exp_data,
        "cloud": cloud_data,
        "role": role_data,
        "plat": np.round(salary_data, -2)
    })

    # Natrénujeme jednoduchý strom s max_depth=3 pro maximální interpretovatelnost
    tree_salary = DecisionTreeRegressor(max_depth=3, random_state=42)
    tree_salary.fit(df_salary[["praxe", "cloud", "role"]], df_salary["plat"])

    # Lineární regrese pro okamžité srovnání
    ols_salary = LinearRegression()
    ols_salary.fit(df_salary[["praxe", "cloud", "role"]], df_salary["plat"])

    # Příprava vstupu uživatele
    u_cloud_num = 1 if "Ano" in user_cloud else 0
    u_role_num = 1 if "ML" in user_role else 0
    u_df = pd.DataFrame([[user_exp, u_cloud_num, u_role_num]], columns=["praxe", "cloud", "role"])

    user_pred_tree = float(tree_salary.predict(u_df)[0])
    user_pred_ols = float(ols_salary.predict(u_df)[0])

    # Zobrazení výsledků kalkulátoru
    res1, res2, res3 = st.columns(3)
    with res1:
        st.metric("🌳 Odhad stromu (Vaše škatulka)", f"{user_pred_tree:,.0f} Kč".replace(",", " "))
    with res2:
        diff_val = user_pred_ols - user_pred_tree
        st.metric("📈 Odhad lineární regrese (Vzorec)", f"{user_pred_ols:,.0f} Kč".replace(",", " "), delta=f"{diff_val:+,.0f} Kč".replace(",", " "), delta_color="off")
    with res3:
        st.metric("👥 Vzorek trhu pro model", f"{n_staff} specialistů")

    # Jak strom dospěl k výsledku
    st.markdown("#### 🧭 Kudy váš profil prošel stromem?")
    step1 = "✅ **Praxe > 2.5 roku** (větvení doprava ➔ Medior/Senior)" if user_exp >= 2.5 else "👶 **Praxe ≤ 2.5 roku** (větvení doleva ➔ Junior)"
    step2 = "☁️ **Má Cloud/MLOps** (prémiové pásmo)" if u_cloud_num == 1 else "💻 **Bez Cloudu** (standardní tarif)"
    st.success(f"**Rozhodovací cesta vašeho profilu:** {step1} ➔ {step2} ➔ **Průměrný plat ve vaší škatulce: {user_pred_tree:,.0f} Kč**")

    # Vizuální srovnání: Skuteční kolegové vs. Schodovitá křivka
    mask_view = (df_salary["cloud"] == u_cloud_num) & (df_salary["role"] == u_role_num)
    df_subset = df_salary[mask_view].sort_values("praxe")

    eval_exp = np.linspace(0, 12, 200)
    eval_df = pd.DataFrame({
        "praxe": eval_exp,
        "cloud": np.full(200, u_cloud_num),
        "role": np.full(200, u_role_num)
    })
    curve_tree = tree_salary.predict(eval_df)
    curve_ols = ols_salary.predict(eval_df)

    fig_life = go.Figure()
    fig_life.add_trace(go.Scatter(
        x=df_subset["praxe"], y=df_subset["plat"],
        mode="markers", name=f"Data z trhu ({'S cloudem' if u_cloud_num else 'Bez cloudu'})",
        marker=dict(size=8, color="#38BDF8", opacity=0.75, line=dict(color="#0284C7", width=1))
    ))
    fig_life.add_trace(go.Scatter(
        x=eval_exp, y=curve_tree,
        mode="lines", name="Rozhodovací strom (Mzdové platové schody)",
        line=dict(color="#EF4444", width=3)
    ))
    fig_life.add_trace(go.Scatter(
        x=eval_exp, y=curve_ols,
        mode="lines", name="Lineární regrese (Přísná přímka)",
        line=dict(color="#94A3B8", width=2, dash="dash")
    ))
    fig_life.add_trace(go.Scatter(
        x=[user_exp], y=[user_pred_tree],
        mode="markers+text", name="Váš zadaný profil",
        text=[f"  Vy: {user_pred_tree:,.0f} Kč".replace(",", " ")],
        textposition="top center",
        marker=dict(size=14, color="#10B981", symbol="star", line=dict(color="white", width=2))
    ))

    fig_life.update_layout(
        title="Platové tarify v oboru: Skutečná data vs. Schodovitá tabulka stromu vs. Přímka OLS",
        xaxis_title="Roky praxe",
        yaxis_title="Měsíční hrubá mzda (Kč)",
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_life, width="stretch")

    st.info(
        "🧠 **Aha-moment pro nováčka:** "
        "Všimněte si, jak strom nevytváří hladkou přímku, ale **mzdové tarify (schody)**. "
        "Lidé s praxí 3.5 roku a 4.8 roku spadnou do stejné škatulky a dostanou identický odhad mzdy. "
        "Teprve když překročíte další zlomový práh seniority (např. 6 let), posune vás strom na vyšší platový schod!"
    )

# =============================================================================
# TAB 3: INTERAKTIVNÍ SIMULÁTOR HLOUBKY STROMU (1D SCHODOVITÁ REGRESE)
# =============================================================================
with tab3:
    st.markdown("### 🧪 Laboratoř: Jak hloubka stromu (`max_depth`) tvaruje predikční schody")
    st.markdown(r"""
    Rozhodovací strom na rozdíl od lineární regrese **nepředpokládá přímku ani křivku**. 
    Dělí osu $X$ na jednotlivé intervaly a v každém predikuje **konstantní průměr hodnot**. 
    Vyzkoušejte si, jak se s rostoucí hloubkou stromu zjemňují schody a kdy dochází k přetrénování (**overfitting**).
    """)

    dataset_choice = st.radio(
        "Vyberte typ experimentálních dat:",
        options=[
            "1. Parabola ze zadání kurzu (make_regression, y = y²)",
            "2. Nelineární sinusovka s rostoucím trendem"
        ],
        horizontal=True
    )

    with st.container(border=True):
        col_ctrl1, col_ctrl2 = st.columns(2)

        with col_ctrl1:
            depth_option = st.select_slider(
                "Maximální hloubka stromu (max_depth):",
                options=["1 (Pařez / Stump)", "2", "3", "4", "5", "7", "Neomezeno (None)"],
                value="3"
            )
            if "Neomezeno" in depth_option:
                selected_depth = None
            else:
                selected_depth = int(depth_option.split()[0])

        with col_ctrl2:
            min_leaf = st.slider(
                "Minimální počet vzorků v listu (min_samples_leaf):",
                min_value=1,
                max_value=25,
                value=1,
                help="Minimální počet trénovacích vzorků vyžadovaný v koncovém listu. Vyšší hodnota tlumí přetrénování na šumu."
            )

        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            show_ols = st.checkbox("📈 Zobrazit referenční lineární OLS regresi pro srovnání", value=True)
        with col_opt2:
            if "sinusovka" in dataset_choice.lower():
                noise_level = st.slider("Úroveň šumu (pro variantu 2):", min_value=0.1, max_value=1.5, value=0.45, step=0.05)
            else:
                noise_level = 0.45
                st.caption("ℹ️ *Generátor paraboly ze zadání kurzu používá fixní náhodný šum (noise=30).*")

    # Generování dat
    if "Parabola" in dataset_choice:
        X_mat, y_reg = make_regression(n_samples=150, n_features=1, noise=30, random_state=42)
        y_noisy = y_reg ** 2
        X_raw = X_mat.flatten()
        sort_idx = np.argsort(X_raw)
        X_raw = X_raw[sort_idx]
        y_noisy = y_noisy[sort_idx]
        y_true_clean = None
    else:
        np.random.seed(42)
        n_pts = 160
        X_raw = np.sort(np.random.uniform(0.5, 10.0, n_pts))
        y_true_clean = 2.2 * np.sin(X_raw) + 0.6 * X_raw
        y_noisy = y_true_clean + np.random.normal(0, noise_level, n_pts)

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw.reshape(-1, 1), y_noisy, test_size=0.25, random_state=42
    )

    # Trénování rozhodovacího stromu
    tree_reg = DecisionTreeRegressor(
        max_depth=selected_depth,
        min_samples_leaf=min_leaf,
        random_state=42
    )
    tree_reg.fit(X_train, y_train)

    train_pred = tree_reg.predict(X_train)
    test_pred = tree_reg.predict(X_test)

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    n_leaves = tree_reg.get_n_leaves()

    # Zobrazení metrik
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.metric("Počet listů", f"{n_leaves}")
    with kpi2:
        st.metric("Trénovací R²", f"{train_r2:.4f}")
    with kpi3:
        st.metric("Testovací R²", f"{test_r2:.4f}", delta=f"{test_r2 - train_r2:+.4f}")
    with kpi4:
        st.metric("Trénovací MSE", f"{train_mse:,.1f}" if train_mse > 1000 else f"{train_mse:.3f}")
    with kpi5:
        delta_mse = test_mse - train_mse
        st.metric(
            "Testovací MSE",
            f"{test_mse:,.1f}" if test_mse > 1000 else f"{test_mse:.3f}",
            delta=f"{delta_mse:+,.1f}" if abs(delta_mse) > 1000 else f"{delta_mse:+.3f}",
            delta_color="inverse"
        )

    # Vytvoření jemné osy pro vizualizaci hladké křivky predikce
    x_min, x_max = float(np.min(X_raw)), float(np.max(X_raw))
    X_plot = np.linspace(x_min, x_max, 600).reshape(-1, 1)
    y_tree_plot = tree_reg.predict(X_plot)

    fig_tree = go.Figure()

    # 1. Trénovací data
    fig_tree.add_trace(go.Scatter(
        x=X_train.flatten(), y=y_train,
        mode="markers", name="Trénovací vzorky",
        marker=dict(size=8, color="#38BDF8", opacity=0.75, line=dict(color="#0284C7", width=1))
    ))

    # 2. Testovací data
    fig_tree.add_trace(go.Scatter(
        x=X_test.flatten(), y=y_test,
        mode="markers", name="Testovací vzorky (Neznámá data)",
        marker=dict(size=9, color="#F59E0B", symbol="diamond", opacity=0.85)
    ))

    # 3. Skutečná generující funkce (pokud je k dispozici)
    if y_true_clean is not None:
        y_true_plot = 2.2 * np.sin(X_plot.flatten()) + 0.6 * X_plot.flatten()
        fig_tree.add_trace(go.Scatter(
            x=X_plot.flatten(), y=y_true_plot,
            mode="lines", name="Skutečný deterministický signál f(x)",
            line=dict(color="#10B981", width=2, dash="dash")
        ))

    # 4. Schodovitá predikce stromu
    fig_tree.add_trace(go.Scatter(
        x=X_plot.flatten(), y=y_tree_plot,
        mode="lines", name=f"Rozhodovací strom (Hloubka {selected_depth or 'None'})",
        line=dict(color="#EF4444", width=3)
    ))

    # 5. Volitelná lineární OLS regrese
    if show_ols:
        ols = LinearRegression().fit(X_train, y_train)
        y_ols_plot = ols.predict(X_plot)
        fig_tree.add_trace(go.Scatter(
            x=X_plot.flatten(), y=y_ols_plot,
            mode="lines", name="Lineární OLS regrese",
            line=dict(color="#94A3B8", width=2, dash="dot")
        ))

    fig_tree.update_layout(
        title=f"Schodovitá regrese rozhodovacího stromu (Hloubka: {depth_option}, Listů: {n_leaves})",
        xaxis_title="Vstupní prediktor X",
        yaxis_title="Cílová veličina Y",
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_tree, width="stretch")

    # Vizuální stromový diagram z kurzu
    with st.expander("🌳 Zobrazit vizuální diagram rozhodovacího stromu (`sklearn.tree.plot_tree`)", expanded=(selected_depth is not None and selected_depth <= 3)):
        if selected_depth is None or selected_depth > 4:
            st.info("ℹ️ Pro stromy s hloubkou větší než 4 je diagram příliš rozsáhlý pro zobrazení na obrazovce. Nastavte hloubku na 1 až 4 pro přehledný náhled.")
        else:
            # Dynamické dimenzování podle hloubky, aby nedocházelo k překrývání uzlů v listech
            if selected_depth == 1:
                fig_size = (10, 4)
                f_size = 10
            elif selected_depth == 2:
                fig_size = (14, 5)
                f_size = 9
            elif selected_depth == 3:
                fig_size = (20, 6.5)
                f_size = 8
            else:  # hloubka 4
                fig_size = (28, 8.5)
                f_size = 6

            fig_tree_arch, ax_tree = plt.subplots(figsize=fig_size)
            plot_tree(
                tree_reg,
                feature_names=["X"],
                filled=True,
                rounded=True,
                precision=2,
                ax=ax_tree,
                fontsize=int(f_size)
            )
            fig_tree_arch.tight_layout()
            st.pyplot(fig_tree_arch, width="stretch")
            plt.close(fig_tree_arch)

    # Diagnostické hlášky podle zvolené hloubky
    if selected_depth == 1:
        st.info("ℹ️ **Podtrénování (Underfitting):** Strom s hloubkou 1 (tzv. Decision Stump) rozdělil data pouze jedním řezem na dvě konstantní hladiny. Model je příliš primitivní a nedokáže zachytit dynamiku.")
    elif selected_depth in [2, 3, 4]:
        st.success("✅ **Vyvážený model:** Strom vytváří smysluplnou po částech konstantní aproximaci. Testovací $R^2$ je vysoké a model skvěle generalizuje bez kopírování náhodného šumu.")
    else:
        st.warning("⚠️ **Extrémní přeučení (Overfitting):** Strom si zapamatoval konkrétní šumové fluktuace trénovacích dat (Trénovací $R^2 \\approx 1.0$). Kolem jednotlivých bodů vytváří úzké svislé věžičky, což dramaticky zhoršuje testovací chybu!")

# =============================================================================
# TAB 4: 2D ROZŘEZÁNÍ PROSTORU NA HYPERBOXY
# =============================================================================
with tab4:
    st.markdown("### 📐 2D Rozřezání prostoru: Proč strom vytváří pravoúhlé bloky?")
    st.markdown(r"""
    Vícerozměrné modely (jako neuronové sítě nebo SVM) dokáží vytvářet diagonální či hladce zakřivené hranice. 
    **Rozhodovací strom se však v každém uzlu ptá pouze na jednu jedinou proměnnou ($X_j \le s$).**  
    Proto jsou všechny řezy v prostoru striktně **rovnoběžné s osami souřadnic** a dělí prostor na soustavu pravoúhlých oblastí (hyperboxů).
    """)

    with st.container(border=True):
        col_2d_a, col_2d_b = st.columns([1, 1])
        with col_2d_a:
            depth_2d = st.slider("Hloubka stromu pro 2D data (max_depth):", min_value=1, max_value=5, value=3)
        with col_2d_b:
            st.info("💡 **Scénář 2D prostoru:** Odhad ceny nemovitosti z **Podlahové plochy ($m^2$)** a **Vzdálenosti od centra ($km$)**.")

    np.random.seed(101)
    n_2d = 200
    sqft = np.random.uniform(30, 160, n_2d)
    dist_center = np.random.uniform(1, 25, n_2d)

    # Cena: roste s plochou, klesá se vzdáleností od centra + interakce
    price_true = (
        30000 * sqft 
        - 45000 * dist_center 
        + 0.15 * (sqft * (25 - dist_center) * 1000)
        + 1200000 
        + np.random.normal(0, 250000, n_2d)
    )

    df_2d = pd.DataFrame({
        "sqft": sqft,
        "dist": dist_center,
        "price": price_true
    })

    # Fit 2D stromu
    tree_2d = DecisionTreeRegressor(max_depth=depth_2d, random_state=42)
    tree_2d.fit(df_2d[["sqft", "dist"]], df_2d["price"])

    # Vytvoření husté mřížky pro vykreslení pravoúhlých ploch
    grid_sqft = np.linspace(30, 160, 100)
    grid_dist = np.linspace(1, 25, 100)
    mesh_x, mesh_y = np.meshgrid(grid_sqft, grid_dist)
    grid_points = np.c_[mesh_x.ravel(), mesh_y.ravel()]
    grid_df = pd.DataFrame(grid_points, columns=["sqft", "dist"])
    grid_preds = tree_2d.predict(grid_df).reshape(mesh_x.shape)

    fig_2d = go.Figure()

    # Kontura predikčních bloků stromu
    fig_2d.add_trace(go.Contour(
        x=grid_sqft,
        y=grid_dist,
        z=grid_preds,
        colorscale="Viridis",
        opacity=0.65,
        colorbar=dict(title="Predikce ceny (Kč)"),
        contours=dict(showlines=True)
    ))

    # Skutečné trénovací body
    fig_2d.add_trace(go.Scatter(
        x=df_2d["sqft"],
        y=df_2d["dist"],
        mode="markers",
        name="Nemovitosti v datech",
        marker=dict(
            size=8,
            color=df_2d["price"],
            colorscale="Viridis",
            showscale=False,
            line=dict(color="white", width=1)
        ),
        hovertemplate="Plocha: %{x:.0f} m²<br>Vzdálenost: %{y:.1f} km<br>Cena: %{marker.color:,.0f} Kč<extra></extra>"
    ))

    fig_2d.update_layout(
        title=f"Pravoúhlé řezy rozhodovacího stromu v 2D prostoru (Hloubka = {depth_2d})",
        xaxis_title="Podlahová plocha (m²)",
        yaxis_title="Vzdálenost od centra města (km)",
        height=520
    )
    st.plotly_chart(fig_2d, width="stretch")

    st.info(
        "💡 **Klíčové vizuální zjištění:** "
        "Všimněte si, jak hranice mezi barvami tvoří výhradně svislé a vodorovné linie. "
        "Pokud by skutečná závislost měla tvar úhlopříčky (např. poměr cena/plocha), "
        "musí strom vybudovat stovky drobných pravoúhlých schodů (tzv. schodovité schodiště), "
        "což vede k neefektivitě ve srovnání s lineárním modelem."
    )

# =============================================================================
# TAB 5: PAST EXTRAPOLACE (LIMIT ROZHODOVACÍHO STROMU)
# =============================================================================
with tab5:
    st.markdown("### 🚨 Zásadní úskalí: Proč rozhodovací strom nedokáže extrapolovat trendy?")
    st.markdown(r"""
    Jednou z nejnebezpečnějších vlastností stromových modelů v praxi je jejich **neschopnost extrapolace**.
    - **Lineární regrese:** Má globální směrnici $\beta$, takže pokud data rostou, predikuje růst i mimo historický interval ($x > x_{\max}$).
    - **Rozhodovací strom:** Mimo rozsah trénovacích dat **nemá žádné další řezy**. Jakékoliv nové pozorování propadne do nejkrajnějšího listu a dostane **přesně stejnou konstantní predikci**!
    """)

    np.random.seed(99)
    # Trénovací data v intervalu [1, 10]
    x_tr = np.linspace(1, 10, 80)
    y_tr = 3.5 * x_tr + np.random.normal(0, 2.0, 80)

    # Testovací data v zóně extrapolace [10, 16]
    x_extrap = np.linspace(10, 16, 50)
    y_extrap_true = 3.5 * x_extrap + np.random.normal(0, 2.0, 50)

    # Fit obou modelů výhradně na trénovacím intervalu [1, 10]
    tree_extrap = DecisionTreeRegressor(max_depth=4, random_state=42).fit(x_tr.reshape(-1, 1), y_tr)
    ols_extrap = LinearRegression().fit(x_tr.reshape(-1, 1), y_tr)

    # Predikce napříč celým rozsahem [1, 16]
    x_full_eval = np.linspace(1, 16, 300).reshape(-1, 1)
    tree_full_preds = tree_extrap.predict(x_full_eval)
    ols_full_preds = ols_extrap.predict(x_full_eval)

    fig_extrap = go.Figure()

    # 1. Trénovací body
    fig_extrap.add_trace(go.Scatter(
        x=x_tr, y=y_tr,
        mode="markers", name="Trénovací data (Historie: 1 až 10)",
        marker=dict(size=8, color="#38BDF8")
    ))

    # 2. Reálná budoucí data (Extrapolace)
    fig_extrap.add_trace(go.Scatter(
        x=x_extrap, y=y_extrap_true,
        mode="markers", name="Reálná budoucnost / Extrapolace (> 10)",
        marker=dict(size=8, color="#F59E0B", symbol="triangle-up")
    ))

    # 3. Predikce Lineární regrese
    fig_extrap.add_trace(go.Scatter(
        x=x_full_eval.flatten(), y=ols_full_preds,
        mode="lines", name="Lineární OLS regrese (Pokračuje v trendu)",
        line=dict(color="#10B981", width=3, dash="dash")
    ))

    # 4. Predikce Rozhodovacího stromu
    fig_extrap.add_trace(go.Scatter(
        x=x_full_eval.flatten(), y=tree_full_preds,
        mode="lines", name="Rozhodovací strom (Narazil na horizontální strop!)",
        line=dict(color="#EF4444", width=3)
    ))

    # Zvýraznění oblasti extrapolace
    fig_extrap.add_vrect(
        x0=10.0, x1=16.0, fillcolor="red", opacity=0.08,
        annotation_text="Zóna extrapolace (> 10)", annotation_position="top left"
    )

    fig_extrap.update_layout(
        title="Demonstrace selhání extrapolace: Strom vs. Lineární regrese",
        xaxis_title="Čas / Velikost X",
        yaxis_title="Hodnota odezvy Y",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_extrap, width="stretch")

    st.error(
        r"""🚨 **Důsledek pro praxi (Např. predikce inflace, tržeb nebo cen akcií):**  
Pokud trénujete model na datech, kde maximální cena byla 10 milionů Kč, a v budoucnu přijde inflační vlna s cenami 15 milionů Kč, 
**samostatný rozhodovací strom ani Random Forest NIKDY nepředpoví hodnotu vyšší než 10 milionů!**  
Pro data s dlouhodobým lineárním trendem se proto vždy musí trend nejprve odseparovat (detrending) nebo použít lineární model."""
    )
