import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.title("🔬 Expertní kritika: Úskalí a limity polynomiální regrese")
st.caption("Hluboká analýza Rungeova jevu, kletby dimenzionality, multikolinearity mocnin a moderních alternativ pro nelineární modelování.")

# =============================================================================
# RYCHLÉ NAČTENÍ PŘEDPOČÍTANÝCH DAT (OKAMŽITÝ RENDER < 0.01s)
# =============================================================================
@st.cache_data
def load_critique_precomputed():
    base_dir = Path(__file__).resolve().parent.parent
    json_path = base_dir / "01_Regression" / "data" / "diamonds_poly_precomputed.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    curve_data = data["extrapolation_curve"]
    curve_df = pd.DataFrame({
        "Carat": curve_data["carat"],
        "Stupeň 1 (Lineární OLS)": curve_data["pred_deg1"],
        "Stupeň 2 (Kvadratický OLS)": curve_data["pred_deg2"],
        "Stupeň 3 (Kubický OLS - Bez regularizace)": curve_data["pred_deg3_ols"],
        "Stupeň 3 + Ridge (α=100)": curve_data["pred_deg3_ridge"]
    })
    return curve_df

# =============================================================================
# KAPITOLA 1: KLETBA DIMENZIONALITY
# =============================================================================
st.markdown("### 1. Kletba dimenzionality a kombinatorická exploze")

st.markdown(r"""
Když v Scikit-learn zavoláme `PolynomialFeatures(degree=d)`, vygenerují se všechny možné kombinace součinů proměnných až do stupně $d$.
Počet výsledných parametrů $P$ (bez absolutního členu) pro $n$ výchozích proměnných se řídí kombinačním číslem:

$$P = \binom{n + d}{d} - 1 = \frac{(n + d)!}{n! \cdot d!} - 1$$

Pro náš dataset diamantů s $n = 9$ parametry:
""")

col1, col2 = st.columns([1, 1])

with col1:
    dim_df = pd.DataFrame([
        {"Stupeň (d)": 1, "Počet příznaků (P)": 9, "Vzorec": "9 lineárních členů", "Poměr k OLS": "1×", "Riziko přeučení": "Minimální"},
        {"Stupeň (d)": 2, "Počet příznaků (P)": 54, "Vzorec": "9 lineárních + 9 kvadratických + 36 interakcí", "Poměr k OLS": "6×", "Riziko přeučení": "Nízké"},
        {"Stupeň (d)": 3, "Počet příznaků (P)": 219, "Vzorec": "Kombinace do 3. řádu", "Poměr k OLS": "24.3×", "Riziko přeučení": "Vysoké (Kolaps)"},
        {"Stupeň (d)": 4, "Počet příznaků (P)": 714, "Vzorec": "Kombinace do 4. řádu", "Poměr k OLS": "79.3×", "Riziko přeučení": "Extrémní"},
        {"Stupeň (d)": 5, "Počet příznaků (P)": 2001, "Vzorec": "Kombinace do 5. řádu", "Poměr k OLS": "222.3×", "Riziko přeučení": "Kritické"}
    ])
    st.dataframe(dim_df, width="stretch", hide_index=True)

with col2:
    fig_dim = px.line(
        dim_df,
        x="Stupeň (d)",
        y="Počet příznaků (P)",
        markers=True,
        title="Kombinatorický nárůst počtu příznaků (pro 9 proměnných)",
        labels={"Počet příznaků (P)": "Počet proměnných v matici Z"}
    )
    fig_dim.update_traces(line_color="#E63946", line_width=3)
    fig_dim.update_layout(height=320)
    st.plotly_chart(fig_dim, width="stretch")

st.info(
    "💡 **Ponaučení pro datového vědce:** "
    "Přidání každého dalšího stupně exponenciálně zahušťuje příznakový prostor. "
    "Ačkoliv máme 53 908 řádků, při 219 příznacích (stupeň 3) již běžná OLS regrese selhává, "
    "protože většina těchto 219 příznaků je vzájemně silně multikolineární!"
)

st.markdown("---")

# =============================================================================
# KAPITOLA 2: RUNGEŮV JEV A EXTRAPOLACE
# =============================================================================
st.markdown("### 2. Rungeův jev a divoká oscilace na okrajích (Extrapolace)")

st.markdown(r"""
V numerické matematice popsal **Carl Runge (1901)** jev, kdy interpolace polynomem vysokého stupně s ekvidistantními body vykazuje 
**divoké oscilace u okrajů intervalu**. V polynomiální regresi se tento jev projevuje tak, že model na okrajích trénovacích dat 
nebo při extrapolaci vyprodukuje zcela nerealistické hodnoty (např. záporné ceny nebo explozi do nekonečna).

Podívejte se na chování našich modelů diamantů pro diamanty s hmotností od 0.2 do 4.0 karátů (nad 3 karáty jsou v datech jen jednotky vzorků):
""")

curve_df = load_critique_precomputed()

fig_curve = go.Figure()

fig_curve.add_trace(go.Scatter(
    x=curve_df["Carat"], y=curve_df["Stupeň 1 (Lineární OLS)"],
    mode="lines", name="Stupeň 1 (Lineární OLS)", line=dict(color="#457B9D", dash="dash")
))
fig_curve.add_trace(go.Scatter(
    x=curve_df["Carat"], y=curve_df["Stupeň 2 (Kvadratický OLS)"],
    mode="lines", name="Stupeň 2 (Kvadratický OLS)", line=dict(color="#2A9D8F", width=3)
))
fig_curve.add_trace(go.Scatter(
    x=curve_df["Carat"], y=curve_df["Stupeň 3 (Kubický OLS - Bez regularizace)"],
    mode="lines", name="Stupeň 3 (Kubický OLS)", line=dict(color="#E63946", width=2)
))
fig_curve.add_trace(go.Scatter(
    x=curve_df["Carat"], y=curve_df["Stupeň 3 + Ridge (α=100)"],
    mode="lines", name="Stupeň 3 + Ridge (α=100)", line=dict(color="#1D3557", width=3)
))

# Vyznačení hranice extrapolace
fig_curve.add_vrect(
    x0=3.0, x1=4.0, fillcolor="red", opacity=0.08,
    annotation_text="Zóna extrapolace (> 3 ct)", annotation_position="top left"
)

fig_curve.update_layout(
    title="Chování modelů při extrapolaci karátu (0.2 až 4.0 karátů)",
    xaxis_title="Hmotnost diamantu (Carat)",
    yaxis_title="Odhadovaná cena diamantu (USD)",
    height=480,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig_curve, width="stretch")

st.warning(
    "⚠️ **Co z grafu vidíme:**\n\n"
    "- **Stupeň 1 (Lineární OLS):** Roste konstantně. Pro malé diamanty přeceňuje, pro velké diamanty fatálně podhodnocuje (nerespektuje kvadratickou povahu ceny).\n"
    "- **Stupeň 2 (Kvadratický OLS):** Perfektně kopíruje fyzikální a tržní realitu – parabolický růst ceny s rostoucí velikostí diamantu.\n"
    "- **Stupeň 3 (Kubický OLS):** V interpolaci (kolem 1 ct) je přesný, ale v extrapolaci nebo na okrajích jeho kubické členy ($x^3$) způsobují nelineární deformaci a model ztrácí stabilitu.\n"
    "- **Stupeň 3 + Ridge:** Regularizace $L_2$ zkrotila divoké koeficienty kubických členů a vrátila křivce hladký, realistický průběh!"
)

st.markdown("---")

# =============================================================================
# KAPITOLA 3: MULTIKOLINEARITA MOCNIN A ZÁCHRANA POMOCÍ RIDGE
# =============================================================================
st.markdown("### 3. Vnitřní multikolinearita mocnin a záchrana pomocí Ridge ($L_2$)")

st.markdown(r"""
Jedním z největších skrytých problémů polynomiální regrese je **vnitřní strukturální multikolinearita**.
Když máme proměnnou $x$ s kladnými hodnotami (např. `carat` od 0.2 do 3.0), pak $x$ a $x^2$ korelují s koeficientem $r > 0.95$ a $x^2$ s $x^3$ s $r > 0.98$!

OLS regrese hledá váhy řešením normální rovnice:
$$\hat{\beta}_{\text{OLS}} = (Z^T Z)^{-1} Z^T y$$

Když jsou sloupce v matici $Z$ téměř lineárně závislé, matice $Z^T Z$ je **špatně podmíněná** (má determinant blízký nule a obrovské číslo podmíněnosti $\kappa(Z) > 10^7$).
Inverze $(Z^T Z)^{-1}$ dramaticky zesiluje i nepatrný šum v datech, což způsobí, že koeficienty $\beta$ vystřelí do obrovských kladných a záporných čísel, která se vzájemně odečítají.

#### Jak Ridge regularizace zachraňuje situaci:
Ridge regrese přidává k matici $Z^T Z$ diagonální člen $\alpha I$:

$$\hat{\beta}_{\text{Ridge}} = (Z^T Z + \alpha I)^{-1} Z^T y$$

Tento zdánlivě drobný přídavek $\alpha I$ odtlačí všechna vlastní čísla matice od nuly. Matice se stane ostře regulární a invertibilní. 
Proto u stupně 3 Ridge s $\alpha = 100$:
- Snížil testovací RMSE z **1 463 USD na 623 USD**!
- Zvýšil testovací $R^2$ z **0.8591 na 0.9744**!
""")

st.markdown("---")

# =============================================================================
# KAPITOLA 4: CHYBA PŘI UMOCŇOVÁNÍ KATEGORICKÝCH / DUMMY PROMĚNNÝCH
# =============================================================================
st.markdown("### 4. Metodická past: Kvadrování ordinálních a dummy proměnných")

st.markdown(r"""
V praxi mnoho začínajících datových vědců udělá to, že předá celou matici $X$ do `PolynomialFeatures(degree=2)`.
V našem datasetu máme proměnné:
- Fyzikální spojité veličiny: `carat`, `x`, `y`, `z`, `depth`, `table`
- Ordinální/kategoriální stupnice: `cut` (1..5), `color` (1..7), `clarity` (1..8)

Co se stane, když umocníme ordinální veličinu?
1. **Ztráta interpretační škály:** U `clarity` (čistota) je rozdíl mezi stupněm 1 a 2 vnímán jako jeden krok na stupnici. Pokud model vytvoří `clarity^2`, krok mezi stupněm 7 a 8 má váhu $8^2 - 7^2 = 15$, zatímco mezi 1 a 2 jen $2^2 - 1^2 = 3$. Tím do modelu vnášíme umělé nelineární zakřivení, které neodpovídá gemologické realitě.
2. **U One-Hot Dummy proměnných je to fatální:** Pokud by byl `cut` zakódován přes One-Hot Encoding (hodnoty 0 a 1), pak $0^2 = 0$ a $1^2 = 1$. Dummy sloupec umocněný na druhou je **naprosto identický** s původním sloupcem! `PolynomialFeatures` by tak vytvořil přesně duplicitní sloupce, což vede k dokonalé multikolinearitě.
""")

st.success(
    "🎯 **Správný postup v praxi:** "
    "Použijte `ColumnTransformer` a aplikujte `PolynomialFeatures` **pouze na spojité numerické sloupce** (např. `carat`, `x`, `y`, `z`). "
    "Kategoriální proměnné nechte lineární, případně povolte pouze jejich vzájemné interakce se spojitými proměnnými (`interaction_only=True`)."
)

st.markdown("---")

# =============================================================================
# KAPITOLA 5: MODERNÍ ALTERNATIVY V ROCE 2026
# =============================================================================
st.markdown("### 5. Co používá moderní Machine Learning místo vysokých polynomů?")

st.markdown(r"""
V moderní datové vědě se globální polynomy stupně 3 a vyšší pro reálná data již téměř nepoužívají. Místo nich se uplatňují tři elegantnější architektury:
""")

alt_col1, alt_col2, alt_col3 = st.columns(3)

with alt_col1:
    st.markdown("#### 1. Po částech kubické spliny (Splines)")
    st.markdown(r"""
    - Knihovna: `sklearn.preprocessing.SplineTransformer`
    - Místo jednoho globálního polynomu rozdělí rozsah veličiny pomocí uzlů (knots) a pro každý úsek fituje lokální kubický polynom.
    - **Obrovská výhoda:** Na okrajích se chovají lineárně, takže **netrpí Rungeovým jevem** ani extrapolací do nekonečna!
    """)

with alt_col2:
    st.markdown("#### 2. GAM modely (Generalized Additive Models)")
    st.markdown(r"""
    - Knihovny: `pyGAM`, `InterpretML (Explainable Boosting Machines - EBM)`
    - Modeluje závislost ve tvaru:
      $$y = \beta_0 + f_1(x_1) + f_2(x_2) + f_{12}(x_1, x_2) + \dots$$
    - Zachovává 100% interpretovatelnost a vizualizaci parciálních závislostí pro business partnery.
    """)

with alt_col3:
    st.markdown("#### 3. Gradient Boosted Trees")
    st.markdown(r"""
    - Knihovny: `HistGradientBoostingRegressor`, `LightGBM`, `XGBoost`
    - Stromy přirozeně a automaticky zachycují nelinearity i libovolně komplexní interakce bez nutnosti generovat jedinou polynomiální proměnnou.
    - Na našem datasetu dosáhl testovacího **$R^2 = 0.9820$ a MAE pouhých 277 USD**!
    """)
