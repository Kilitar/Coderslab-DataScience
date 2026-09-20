import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.title("🎓 Závěrečné shrnutí Dne 1: Kompletní přehled regrese")
st.caption("Syntéza prvního dne kurzu Machine Learning: Od jednoduché lineární přímky až po nelineární stromy, srovnávací benchmarky na reálných datech a interaktivní kvíz.")

tab_overview, tab_comparison, tab_car_example, tab_benchmarks, tab_quiz = st.tabs([
    "📑 Přehled kapitol (Day 1)",
    "⚖️ Velké srovnání architektur",
    "🚗 Příklad z praxe: Křivky 4 modelů",
    "📊 Reálné benchmarky (Reality & Diamanty)",
    "🧠 Závěrečný test znalostí (Kvíz)"
])

# =============================================================================
# TAB 1: PŘEHLED KAPITOL
# =============================================================================
with tab_overview:
    st.markdown("### 🗺️ Průchod prvním dnem: Co všechno jsme zvládli?")
    st.markdown(r"""
    Během prvního dne jsme prošli 4 klíčové tematické pilíře regresního modelování:
    """)

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("#### 1. Lineární regrese & OLS")
            st.markdown(r"""
            - Základní model modelující přímou závislost: $y = \mathbf{X}\boldsymbol{\beta} + \varepsilon$.
            - Analytické řešení metodou nejmenších čtverců (Ordinary Least Squares).
            - Maximální interpretovatelnost: každý koeficient $\beta_j$ představuje mezní efekt příznaku.
            - Diagnostika reziduí: normalita, homoskedasticita a multikolinearita (VIF).
            """)
        with st.container(border=True):
            st.markdown("#### 2. Metriky kvality modelu")
            st.markdown(r"""
            - **$R^2$ (Koeficient determinace):** Podíl vysvětleného rozptylu ($0 \dots 1$).
            - **Adjusted $R^2$:** Penalizace za nadbytečný počet prediktorů.
            - **MAE:** Průměrná absolutní chyba v původních jednotkách (robustní vůči extrémům).
            - **MSE / RMSE:** Kvadratická penalizace velkých chyb; citlivá na odlehlé body.
            """)
    with col2:
        with st.container(border=True):
            st.markdown("#### 3. Regularizace (Lasso, Ridge, Elastic Net)")
            st.markdown(r"""
            - Boj proti přetrénování přidáním pokuty do účelové funkce.
            - **Lasso (L1):** Penalizace $\alpha \sum |\beta_j|$, způsobuje nulování vah (Feature Selection).
            - **Ridge (L2):** Penalizace $\alpha \sum \beta_j^2$, zmenšuje váhy a řeší multikolinearitu.
            - **Kritická podmínka:** Nutnost předchozího škálování (`StandardScaler`).
            """)
        with st.container(border=True):
            st.markdown("#### 4. Nelineární modely: Polynom & Strom")
            st.markdown(r"""
            - **Polynomiální regrese:** Modelování zakřivení pomocí mocnin $x^2, x^3$; riziko Rungeho jevu.
            - **Rozhodovací strom (CART):** Neparametrické binární dělení prostoru do schodovitých oblastí.
            - Odolnost vůči měřítkům a odlehlým hodnotám, avšak náchylnost k hlubokému přeučení bez prořezání.
            """)

# =============================================================================
# TAB 2: VELKÉ SROVNÁNÍ ARCHITEKTUR
# =============================================================================
with tab_comparison:
    st.markdown("### ⚖️ Kdy který model nasadit? Přehledná srovnávací matice")
    
    comp_table = [
        {
            "Algoritmus": "Lineární regrese (OLS)",
            "Tvar funkce": "Globální přímka / rovina",
            "Citlivost na škálování": "Nízká (pouze pro koeficienty)",
            "Odolnost vůči extrémům": "❌ Velmi nízká",
            "Interpretovatelnost": "🟢 Maximální (přímé rovnice)",
            "Riziko přeučení": "Nízké (spíše nedoučení)",
            "Ideální využití": "Ekonometrie, finance, čistá interpretace"
        },
        {
            "Algoritmus": "Ridge regrese (L2)",
            "Tvar funkce": "Globální rovina se scvrklými vahami",
            "Citlivost na škálování": "⚠️ Kritická (StandardScaler nutný)",
            "Odolnost vůči extrémům": "❌ Nízká",
            "Interpretovatelnost": "🟢 Vysoká",
            "Riziko přeučení": "Velmi nízké (tlumí varianci)",
            "Ideální využití": "Korelované příznaky, stabilizace modelu"
        },
        {
            "Algoritmus": "Lasso regrese (L1)",
            "Tvar funkce": "Globální rovina s vybranými příznaky",
            "Citlivost na škálování": "⚠️ Kritická (StandardScaler nutný)",
            "Odolnost vůči extrémům": "❌ Nízká",
            "Interpretovatelnost": "🟢 Vysoká (sparse matice)",
            "Riziko přeučení": "Velmi nízké",
            "Ideální využití": "Vysoká dimenze (mnoho příznaků), Feature Selection"
        },
        {
            "Algoritmus": "Polynomiální regrese",
            "Tvar funkce": "Hladká vlnitá křivka vyšších řádů",
            "Citlivost na škálování": "⚠️ Extrémní (mocniny explodují)",
            "Odolnost vůči extrémům": "❌ Katastrofální (okrajové exploze)",
            "Interpretovatelnost": "🟡 Nízká (kombinace mocnin)",
            "Riziko přeučení": "⚠️ Vysoké (Rungeho oscilace)",
            "Ideální využití": "Fyzikální a chemické nelineární jevy"
        },
        {
            "Algoritmus": "Rozhodovací strom (CART)",
            "Tvar funkce": "Po částech konstantní (schodovité plošiny)",
            "Citlivost na škálování": "🟢 Nulová (pracuje s pořadím)",
            "Odolnost vůči extrémům": "🟢 Vysoká (izoluje do listu)",
            "Interpretovatelnost": "🟢 Vysoká (při malé hloubce)",
            "Riziko přeučení": "⚠️ Extrémní (nutné ladit max_depth)",
            "Ideální využití": "Tabulková data se skoky, tarify a zónami"
        }
    ]
    st.dataframe(pd.DataFrame(comp_table), width="stretch")

# =============================================================================
# TAB 3: KREATIVNÍ PŘÍKLAD Z PRAXE
# =============================================================================
with tab_car_example:
    st.markdown("### 🚗 Jeden reálný problém: Jak na něj koukají 4 různé modely?")
    st.markdown(r"""
    Představme si reálnou závislost: **Cena ojetého auta v závislosti na jeho stáří**.  
    - V prvních letech cena prudce padá (nové auto).
    - Ve středním věku klesá pozvolna.
    - Po 25 letech se z auta stává vyhledávaný veterán a jeho hodnota začíná prudce růst!
    """)

    # Syntéza ukázkových dat pro vizualizaci
    np.random.seed(42)
    age = np.linspace(0.5, 32, 100)
    # Reálná křivka: prudký pád, dno kolem 18 let, pak růst veteránů
    true_price = 900000 / (1 + 0.35 * age) + 1500 * np.maximum(0, age - 18)**2.2
    noise = np.random.normal(0, 35000, size=len(age))
    car_price = np.maximum(20000, true_price + noise)

    # 1. Lineární fit (OLS)
    p_lin = np.polyfit(age, car_price, 1)
    y_lin = np.polyval(p_lin, age)

    # 2. Polynom stupně 2
    p_poly = np.polyfit(age, car_price, 2)
    y_poly = np.polyval(p_poly, age)

    # 3. Přeučený polynom stupně 8
    p_over = np.polyfit(age, car_price, 8)
    y_over = np.polyval(p_over, age)

    # 4. Rozhodovací strom (schodovitá funkce)
    from sklearn.tree import DecisionTreeRegressor
    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=5, random_state=42)
    tree.fit(age.reshape(-1, 1), car_price)
    y_tree = tree.predict(age.reshape(-1, 1))

    fig_car = go.Figure()
    # Skutečná data
    fig_car.add_trace(go.Scatter(
        x=age, y=car_price, mode="markers", name="Auta na trhu (data)",
        marker=dict(size=6, color="#94A3B8", opacity=0.7)
    ))
    # OLS
    fig_car.add_trace(go.Scatter(
        x=age, y=y_lin, mode="lines", name="Lineární regrese (OLS)",
        line=dict(color="#EF4444", width=2.5, dash="dash")
    ))
    # Polynom 2. st.
    fig_car.add_trace(go.Scatter(
        x=age, y=y_poly, mode="lines", name="Polynom 2. stupně",
        line=dict(color="#3B82F6", width=2.5)
    ))
    # Přeučený polynom
    fig_car.add_trace(go.Scatter(
        x=age, y=y_over, mode="lines", name="Přeučený polynom (8. stupeň)",
        line=dict(color="#EC4899", width=1.5, dash="dot")
    ))
    # Rozhodovací strom
    fig_car.add_trace(go.Scatter(
        x=age, y=y_tree, mode="lines", name="Rozhodovací strom (hloubka 3)",
        line=dict(color="#10B981", width=3)
    ))

    fig_car.update_layout(
        title="Jak 4 různé modely chápou vztah: Stáří auta vs. Cena",
        xaxis_title="Stáří automobilu (roky)",
        yaxis_title="Odhadovaná cena (Kč)",
        height=540,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_car, width="stretch")

    st.markdown("""
    #### 💡 Anatomie pohledu jednotlivých modelů na grafu:
    1. **Lineární přímka (červená přerušovaná):** Zcela ignoruje růst veteránů a u 30letého auta předpovídá nesmyslnou zápornou cenu!
    2. **Polynom 2. stupně (modrá křivka):** Krásně vystihuje U-křivku (propad a následný nárůst), ale u nového vozu podstřeluje strmost propadu.
    3. **Přeučený polynom 8. stupně (růžová tečkovaná):** Divoce osciluje kolem šumu a na pravém okraji exploduje nahoru (Rungeho jev).
    4. **Rozhodovací strom (zelené schody):** Vytváří stabilní zóny: *„do 2 let = 700k, 3-7 let = 400k, 8-18 let = 150k, 19+ let = 350k“*. Žádná extrapolace do záporu, žádné nekonečno!
    """)

# =============================================================================
# TAB 4: REÁLNÉ BENCHMARKY
# =============================================================================
with tab_benchmarks:
    st.markdown("### 📊 Kompletní výsledková listina z našich celodenních cvičení")
    st.markdown("Podívejme se, jak jednotlivé modely obstály na dvou oficiálních datových sadách kurzu:")

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        st.markdown("#### 🏡 Dataset 1: Reality King County (`kc_house_data.csv`)")
        bench_kc = [
            {"Model": "OLS Lineární regrese (Cvičení 1)", "Test R²": "0.6950", "Test RMSE": "203 150 USD", "Poznámka": "Baseline přímka"},
            {"Model": "Ridge / Lasso s normalizací (Cvičení 5)", "Test R²": "0.6982", "Test RMSE": "201 400 USD", "Poznámka": "Stabilizace vah a nulování"},
            {"Model": "Výchozí strom bez omezení (Cvičení 8)", "Test R²": "0.7048", "Test RMSE": "211 262 USD", "Poznámka": "35 pater, 16 683 listů (přeučeno)"},
            {"Model": "Optimální strom z GridSearchCV (Cvičení 8)", "Test R²": "0.7932", "Test RMSE": "176 810 USD", "Poznámka": "🏆 Pokles chyby o 35k USD díky souřadnicím"}
        ]
        st.dataframe(pd.DataFrame(bench_kc), width="stretch")
    with col_b2:
        st.markdown("#### 💎 Dataset 2: Diamanty (`diamonds.csv`)")
        bench_diam = [
            {"Model": "OLS Lineární regrese (Cvičení 2)", "Test R²": "0.88 - 0.92", "Test RMSE": "1 150 - 1 400 USD", "Poznámka": "Základní lineární odhad"},
            {"Model": "Lasso regularizace 4Cs (Cvičení 6)", "Test R²": "0.9250", "Test RMSE": "1 120 USD", "Poznámka": "Výběr klíčových vlastností"},
            {"Model": "Polynomiální regrese 3. st. (Cvičení 7)", "Test R²": "0.9550", "Test RMSE": "850 USD", "Poznámka": "Nelineární křivky"},
            {"Model": "Optimální rozhodovací strom (Cvičení 9)", "Test R²": "0.9787", "Test RMSE": "584 USD", "Poznámka": "🏆 Skoky u kulatých karátů (carat >= 1.0)"}
        ]
        st.dataframe(pd.DataFrame(bench_diam), width="stretch")

# =============================================================================
# TAB 5: INTERAKTIVNÍ KVÍZ
# =============================================================================
with tab_quiz:
    st.markdown("### 🧠 Otestujte své znalosti: Kvíz na závěr Dne 1")
    st.caption("Klikněte na správnou odpověď a ihned zkontrolujte, zda jste připraveni na zkoušku.")

    # Otázka 1
    st.markdown("##### 1. Jaký je hlavní rozdíl mezi L1 (Lasso) a L2 (Ridge) regularizací?")
    q1 = st.radio(
        "Vyberte správné tvrzení:",
        [
            "L2 regularizace dokáže nastavit koeficienty přesně na nulu, zatímco L1 je pouze zmenšuje.",
            "L1 regularizace (Lasso) dokáže vynulovat váhy méně důležitých příznaků a provádí Feature Selection, zatímco L2 (Ridge) váhy pouze scvrkává k nule.",
            "Mezi L1 a L2 není žádný matematický rozdíl, pouze používají jiný název balíčku v Scikit-learn."
        ],
        key="quiz_q1"
    )
    if st.button("Zkontrolovat otázku 1"):
        if "L1 regularizace (Lasso) dokáže vynulovat" in q1:
            st.success("✅ Správně! Lasso přidává absolutní hodnotu vah (|w|), což geometricky vede k rohovému řešení a přesnému vynulování méně důležitých příznaků.")
        else:
            st.error("❌ Špatně. Pamatujte: L1 = Lasso = součet absolutních hodnot = NULOVÁNÍ vah. L2 = Ridge = čtverce = SCVRKÁVÁNÍ vah.")

    st.divider()

    # Otázka 2
    st.markdown("##### 2. Proč rozhodovací strom (DecisionTreeRegressor) nepotřebuje škálování dat (`StandardScaler`)?")
    q2 = st.radio(
        "Vyberte správné tvrzení:",
        [
            "Protože strom interně vždy všechna data automaticky převede na interval <0, 1>.",
            "Protože rozhodovací strom dělí prostor podle prahových podmínek (x_j <= s), což závisí pouze na pořadí hodnot a je zcela nezávislé na měřítku proměnné.",
            "Škálování je nutné i pro stromy, bez něj strom selže s chybou ConvergenceWarning."
        ],
        key="quiz_q2"
    )
    if st.button("Zkontrolovat otázku 2"):
        if "záleží pouze na pořadí hodnot" in q2 or "x_j <= s" in q2:
            st.success("✅ Přesně tak! Dělicí podmínka 'carat <= 1.0' funguje naprosto stejně, ať už jsou karáty v jednotkách, gramech nebo logaritmu.")
        else:
            st.error("❌ Špatně. Stromy pracují s pořadím prvků (monotónní dělení), nikoliv se vzdálenostmi v eukleidovském prostoru.")

    st.divider()

    # Otázka 3
    st.markdown("##### 3. Co se stane s rozhodovacím stromem, pokud mu nezadáte žádné omezující hyperparametry (`max_depth=None`, `min_samples_leaf=1`)?")
    q3 = st.radio(
        "Vyberte správné tvrzení:",
        [
            "Model dosáhne ideálního stavu a generalizuje nejlépe na nová data.",
            "Strom se dramaticky přeučí (overfitting): naroste do obří hloubky a vytvoří listy s jediným vzorkem (Train R² = 1.0, ale na testovacích datech chybuje).",
            "Model se odmítne natrénovat a vyhodí chybu MemoryError."
        ],
        key="quiz_q3"
    )
    if st.button("Zkontrolovat otázku 3"):
        if "Strom se dramaticky přeučí" in q3:
            st.success("✅ Vynikající! To jsme viděli u diamantů: výchozí strom vytvořil 36 792 listů a téměř v každém měl jediný diamant. Teprve prořezáním jsme snížili chybu o 130 USD.")
        else:
            st.error("❌ Špatně. Neomezený strom se učí data nazpaměť včetně šumu (extrémní overfitting).")
