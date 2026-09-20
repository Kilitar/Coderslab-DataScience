import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.title("🎓 Závěrečné shrnutí Dne 1: Kompletní přehled regrese")
st.caption("Syntéza prvního dne kurzu Machine Learning: Od jednoduché lineární přímky až po nelineární stromy, srovnávací benchmarky na reálných datech a komplexní česko-anglický glosář pojmů.")

tab_overview, tab_comparison, tab_car_example, tab_benchmarks, tab_dict = st.tabs([
    "📑 Přehled kapitol (Day 1)",
    "⚖️ Velké srovnání architektur",
    "🚗 Příklad z praxe: Křivky 4 modelů",
    "📊 Reálné benchmarky (Reality & Diamanty)",
    "📚 Glosář klíčových pojmů (CZ / EN Dictionary)"
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
            - **$R^2$ (Koeficient determinace):** Podíl vysvětleného rozptylu. Na trénovací sadě s interceptem nabývá hodnot $0 \dots 1$. Na testovací sadě může nabývat i záporných hodnot ($R^2 \in (-\infty, 1]$), pokud je model horší než pouhé tipování průměru $\bar{y}$.
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
    1. **Lineární přímka (červená přerušovaná):** Zcela ignoruje U-křivku. Snaží se najít kompromis mezi novými auty a veterány, takže je téměř vodorovná: u nového vozu těžce podhodnotí cenu (odhadne jen ~300 tis. místo 750 tis. Kč) a u veterána ji rovněž podstřelí (~225 tis. místo 450 tis. Kč)!
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
            {"Model": "OLS Lineární regrese (Cvičení 2)", "Test R²": "0.9095", "Test RMSE": "1 172.5 USD", "Poznámka": "Základní lineární odhad"},
            {"Model": "Lasso / Ridge regularizace 4Cs (Cvičení 6)", "Test R²": "0.9095", "Test RMSE": "1 172.1 USD", "Poznámka": "Stabilizace vah a multikolinearity"},
            {"Model": "Polynom 2. stupně (Kvadratický OLS, Cvičení 7)", "Test R²": "0.9655", "Test RMSE": "723.7 USD", "Poznámka": "Optimální stupeň, 54 příznaků"},
            {"Model": "Polynom 3. stupně (Surový kubický OLS, Cvičení 7)", "Test R²": "0.8591", "Test RMSE": "1 463.0 USD", "Poznámka": "⚠️ Kolaps variance / multikolinearita (219 příznaků)"},
            {"Model": "Polynom 3. stupně + Ridge α=100 (Cvičení 7)", "Test R²": "0.9744", "Test RMSE": "623.2 USD", "Poznámka": "🏆 Regularizace L2 zkrotila 219 příznaků"},
            {"Model": "Optimální rozhodovací strom (Cvičení 9)", "Test R²": "0.9785", "Test RMSE": "587.0 USD", "Poznámka": "🏆 Zachycení skoků u kulatých karátů (carat >= 1.0)"}
        ]
        st.dataframe(pd.DataFrame(bench_diam), width="stretch")

# =============================================================================
# TAB 5: KOMPLEXNÍ GLOSÁŘ POJMŮ (CZ / EN DICTIONARY)
# =============================================================================
with tab_dict:
    st.markdown("### 📚 Komplexní glosář klíčových pojmů Machine Learningu (CZ / EN)")
    st.markdown(r"""
    Slovník nejdůležitějších pojmů, se kterými se v praxi setkáte. Každý pojem obsahuje **český i anglický název**, 
    přesnou definici, souvislosti s regresními modely a praktický příklad.
    """)

    # Databáze pojmů
    glossary_items = [
        {
            "kategorie": "Chování a chyby modelů",
            "cz": "Přeučení (Přetrénování)",
            "en": "Overfitting",
            "definice": "Stav, kdy se model naučil trénovací data 'nazpaměť' včetně náhodného šumu a lokálních výkyvů. Na trénovacích datech má perfektní metriky (např. R² = 1.0), ale na nových testovacích datech zcela selhává.",
            "kontext": "Typický pro neomezené stromy (35 pater u King County s 16k listy) nebo polynomy vysokých stupňů.",
            "reseni": "Prořezání stromu (max_depth, min_samples_leaf), regularizace (L1/L2), křížová validace (K-Fold CV), přidání více dat."
        },
        {
            "kategorie": "Chování a chyby modelů",
            "cz": "Nedoučení",
            "en": "Underfitting",
            "definice": "Stav, kdy je model příliš jednoduchý na to, aby zachytil skutečné zákonitosti v datech. Má vysokou chybu jak na trénovací, tak na testovací sadě.",
            "kontext": "Aplikace základní OLS přímky na parabolickou nebo schodovitou závislost (např. ceny veteránů nebo skokové karáty).",
            "reseni": "Zvýšení kapacity modelu, přidání nelineárních členů (polynom), přechod na rozhodovací stromy, tvorba lepších příznaků (Feature Engineering)."
        },
        {
            "kategorie": "Chování a chyby modelů",
            "cz": "Kompromis mezi vychýlením a rozptylem",
            "en": "Bias-Variance Tradeoff",
            "definice": "Fundamentální dilema v ML. Celková chyba se skládá z: (1) Bias (chyba z příliš zjednodušujících předpokladů) a (2) Variance (citlivost modelu na malé změny v trénovacích datech).",
            "kontext": "Lineární regrese má vysoký bias a nízkou variance. Hluboký strom má nulový bias, ale obrovskou variance.",
            "reseni": "Nalezení sladkého bodu (Sweet Spot) pomocí hyperparametrů nebo ansámblových metod (Random Forest, Gradient Boosting)."
        },
        {
            "kategorie": "Matematika a regularizace",
            "cz": "Regularizace",
            "en": "Regularization",
            "definice": "Metoda omezování složitosti modelu přidáním penalizačního členu (pokuty) k chybové účelové funkci: Ztráta = MSE + Pokuta(váhy).",
            "kontext": "Zabraňuje vahám beta narůst do obřích hodnot u korelovaných nebo vysokodimenzionálních dat.",
            "reseni": "L1 (Lasso), L2 (Ridge) nebo jejich kombinace (Elastic Net)."
        },
        {
            "kategorie": "Matematika a regularizace",
            "cz": "L1 regularizace (Lasso)",
            "en": "L1 Regularization (Least Absolute Shrinkage and Selection Operator)",
            "definice": "Penalizace úměrná součtu absolutních hodnot koeficientů: α · Σ|β_j|. Díky tvaru diamantového omezení v geometrii dokáže nastavit nevýznamné váhy přesně na 0.",
            "kontext": "Funguje jako automatický výběr nejdůležitějších příznaků (Feature Selection) u datasetů s mnoha proměnnými.",
            "reseni": "Využijte, pokud máte podezření, že většina sloupců je irelevantní šum."
        },
        {
            "kategorie": "Matematika a regularizace",
            "cz": "L2 regularizace (Ridge / Tichonov)",
            "en": "L2 Regularization (Ridge Regression)",
            "definice": "Penalizace úměrná součtu čtverců koeficientů: α · Σ(β_j)². Stlačuje (scvrkává) váhy směrem k nule, ale žádnou nevynuluje úplně.",
            "kontext": "Klíčový nástroj pro stabilizaci regrese při silné multikolinearitě (když spolu vlastnosti domu či diamantu silně korelují).",
            "reseni": "Využijte, pokud chcete zachovat všechny příznaky, ale zabránit divokým výkyvům koeficientů."
        },
        {
            "kategorie": "Příprava dat & Data Leakage",
            "cz": "Únik informací z dat",
            "en": "Data Leakage",
            "definice": "Závažná metodologická chyba, kdy se informace z testovací (nebo validační) sady neúmyslně dostanou do procesu trénování modelu.",
            "kontext": "Spuštění StandardScaleru nebo PolynomialFeatures na celém datasetu PŘED rozdělením na train/test.",
            "reseni": "Vždy nejprve rozdělit data (train_test_split) a scaler fitovat POUZE na train sadě (scaler.fit_transform(X_train), na test jen scaler.transform(X_test))."
        },
        {
            "kategorie": "Příprava dat & Data Leakage",
            "cz": "Multikolinearita",
            "en": "Multicollinearity",
            "definice": "Stav, kdy jsou dva nebo více prediktorů v lineární regresi vzájemně silně korelované (jeden lze téměř přesně předpovědět z druhého).",
            "kontext": "Např. rozměry diamantu x, y, z a váha carat, nebo sqft_living a sqft_above u domů.",
            "reseni": "Měření přes VIF (Variance Inflation Factor), odstranění redundantních sloupců nebo nasazení Ridge/Lasso regularizace."
        },
        {
            "kategorie": "Příprava dat & Data Leakage",
            "cz": "Kletba dimenzionality",
            "en": "Curse of Dimensionality",
            "definice": "Fenomén, kdy s rostoucím počtem příznaků (dimenzí) roste objem prostoru exponenciálně, data se stávají extrémně řídká a vzdálenosti mezi body ztrácejí rozlišovací schopnost.",
            "kontext": "Vytvoření polynomu 3. stupně na 9 vybraných prediktorech diamantů vygenerovalo 219 členů a způsobilo kolaps variance OLS (na 26 sloupcích by bez biasu vzniklo dokonce 3 653 členů).",
            "reseni": "Redukce dimenzionality (PCA), selekce příznaků (Lasso), doménový výběr pouze fyzikálních proměnných."
        },
        {
            "kategorie": "Architektura rozhodovacích stromů",
            "cz": "Prořezávání stromu",
            "en": "Tree Pruning",
            "definice": "Technika redukce velikosti a hloubky rozhodovacího stromu odstraněním větví, které nepřinášejí statisticky významné zlepšení generalizace.",
            "kontext": "Pre-pruning (zastavení růstu pomocí max_depth, min_samples_leaf) a Post-pruning (Cost-Complexity Pruning přes parametr ccp_alpha).",
            "reseni": "U diamantů jsme snížili počet listů z 36 792 na 1 840, čímž chyba RMSE klesla o 131 USD."
        },
        {
            "kategorie": "Architektura rozhodovacích stromů",
            "cz": "Důležitost příznaků (MDI)",
            "en": "Feature Importance (Mean Decrease in Impurity)",
            "definice": "Metrika vyjadřující relativní přínos každého příznaku pro celkovou přesnost stromu. Počítá se jako celkový pokles nečistoty (MSE) napříč všemi uzly, kde byl daný příznak vybrán k dělení.",
            "kontext": "U diamantů tvořily carat a y přes 75 % veškerého rozhodování; u domů dominovaly sqft_living a grade.",
            "reseni": "Rychlá identifikace klíčových byznys driverů bez nutnosti složité statistické analýzy."
        },
        {
            "kategorie": "Architektura rozhodovacích stromů",
            "cz": "Neschopnost extrapolace",
            "en": "Inability to Extrapolate",
            "definice": "Zásadní vlastnost rozhodovacích stromů: model nedokáže předpovědět hodnotu ležící mimo rozsah trénovacích dat. Jeho predikce je v listu vždy konstantním průměrem.",
            "kontext": "Pokud má nejdražší dům v trénovacích datech 7 mil. USD, strom pro zámek za 50 mil. USD nikdy nepředpoví více než 7 mil. USD (narazí na konstantní strop).",
            "reseni": "Pokud je nutná extrapolace rostoucího trendu do budoucna, volit lineární/polynomiální modely nebo hybridní přístupy."
        },
        {
            "kategorie": "Metriky a vyhodnocování",
            "cz": "Koeficient determinace (R²)",
            "en": "Coefficient of Determination (R-squared)",
            "definice": "Míra vyjadřující podíl celkového rozptylu cílové proměnné vysvětlený modelem: R² = 1 - (SS_res / SS_tot). Na trénovacích datech s absolutním členem leží v intervalu [0, 1]. Na nových testovacích datech může nabývat i záporných hodnot (-∞, 1], pokud model predikuje hůře než triviální průměr cílové proměnné ȳ.",
            "kontext": "Hodnota 0.79 znamená, že model vysvětlil 79 % cenových rozdílů mezi domy; zbylých 21 % je nepozorovaný šum nebo chybějící proměnné.",
            "reseni": "Pozor na porovnávání na trénovací sadě (R² tam vždy roste s přidáním jakéhokoliv sloupce – proto existuje Adjusted R²)."
        },
        {
            "kategorie": "Metriky a vyhodnocování",
            "cz": "Průměrná absolutní chyba",
            "en": "Mean Absolute Error (MAE)",
            "definice": "Průměr absolutních hodnot odchylek: (1/n) · Σ|y_i - ŷ_i|. Měří se přímo v původních jednotkách (USD, Kč, kg).",
            "kontext": "Pokud je MAE u diamantů 308 USD, v průměru se model plete o 308 USD na jednom diamantu.",
            "reseni": "Ideální metrika pro prezentaci managementu a byznys zadavatelům, protože je intuitivní a nedeformují ji extrémy."
        },
        {
            "kategorie": "Metriky a vyhodnocování",
            "cz": "Odmocnina z průměrné čtvercové chyby",
            "en": "Root Mean Squared Error (RMSE)",
            "definice": "Odmocnina z průměru čtverců chyb: √[(1/n) · Σ(y_i - ŷ_i)²]. Rovněž v původních jednotkách, ale penalizuje velké chyby mnohem přísněji než MAE.",
            "kontext": "Pokud je RMSE výrazně vyšší než MAE (např. RMSE 584 USD vs. MAE 308 USD), model má několik závažných selhání na odlehlých hodnotách.",
            "reseni": "Standardní metrika v soutěžích (Kaggle) a v úlohách, kde je velká chyba kriticky nebezpečná."
        }
    ]

    # Interaktivní filtrování a vyhledávání
    col_f1, col_f2 = st.columns([2, 3])
    with col_f1:
        categories = ["Všechny kategorie"] + sorted(list(set(item["kategorie"] for item in glossary_items)))
        sel_cat = st.selectbox("Filtrovat dle kategorie:", categories)
    with col_f2:
        search_query = st.text_input("🔍 Vyhledat pojem (česky nebo anglicky):", placeholder="např. overfitting, regularizace, leakage...")

    # Filtrování dat
    filtered_items = glossary_items
    if sel_cat != "Všechny kategorie":
        filtered_items = [i for i in filtered_items if i["kategorie"] == sel_cat]
    if search_query:
        q = search_query.lower()
        filtered_items = [
            i for i in filtered_items
            if q in i["cz"].lower() or q in i["en"].lower() or q in i["definice"].lower()
        ]

    st.caption(f"Zobrazeno {len(filtered_items)} z {len(glossary_items)} klíčových pojmů")

    # Vykreslení karet pojmů
    for item in filtered_items:
        with st.container(border=True):
            head_col1, head_col2 = st.columns([3, 2])
            with head_col1:
                st.markdown(f"#### 📖 {item['cz']}")
                st.markdown(f"🇬🇧 **Anglický termín:** `{item['en']}`")
            with head_col2:
                st.markdown(f"<div style='text-align: right;'><span style='background-color: #334155; padding: 4px 10px; border-radius: 8px; font-size: 0.85em;'>📁 {item['kategorie']}</span></div>", unsafe_allow_html=True)
            
            st.markdown(f"**Definice:** {item['definice']}")
            
            c_box1, c_box2 = st.columns(2)
            with c_box1:
                st.info(f"🔎 **V kontextu regrese:**\n{item['kontext']}")
            with c_box2:
                st.success(f"🛠️ **Jak řešit v praxi:**\n{item['reseni']}")

