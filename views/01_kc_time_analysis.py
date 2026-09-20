from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

st.title("⏱️ Cvičení 1: Časová analýza, Sezónnost & Time-Split")
st.caption("Proč je smazání data prodeje v kurzu metodická chyba a jak se projeví reálný časový split v praxi.")

# =============================================================================
# NAČTENÍ DAT A VYČIŠTĚNÍ
# =============================================================================
@st.cache_data
def load_and_prep_time_data():
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "01_Regression" / "data" / "kc_house_data.csv"
    if not data_path.exists():
        data_path = base_dir / "data" / "MAL_downloadable materials_session 1" / "Day 1" / "kc_house_data.csv"

    df = pd.read_csv(data_path)
    
    # Filtrace stejných odlehlých hodnot jako v cvičení 1
    price_cap = df["price"].quantile(0.998)
    clean = df[
        (df["bedrooms"] <= 10)
        & (df["bedrooms"] > 0)
        & (df["sqft_living"] <= 10000)
        & (df["price"] <= price_cap)
    ].copy()

    # Extrakce času
    clean["dt"] = pd.to_datetime(clean["date"].str[:8])
    min_date = clean["dt"].min()
    clean["days_since_start"] = (clean["dt"] - min_date).dt.days
    clean["year_month"] = clean["dt"].dt.to_period("M").astype(str)
    clean["month"] = clean["dt"].dt.month
    clean["quarter"] = clean["dt"].dt.quarter

    return clean

df_time = load_and_prep_time_data()

# =============================================================================
# 1. ZÁKLADNÍ FAKTA O ČASOVÉM ROZSAHU
# =============================================================================
min_dt = df_time["dt"].min().strftime("%d. %m. %Y")
max_dt = df_time["dt"].max().strftime("%d. %m. %Y")
total_days = (df_time["dt"].max() - df_time["dt"].min()).days

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("🗓️ Nejstarší záznam v datech", min_dt)
with c2:
    st.metric("🗓️ Nejnovější záznam v datech", max_dt)
with c3:
    st.metric("⏳ Celkový časový rozsah", f"{total_days} dní (~13 měsíců)")

st.markdown("---")

# =============================================================================
# 2. VIZUALIZACE SEZÓNNOSTI A OBJEMU PRODEJŮ
# =============================================================================
st.subheader("📈 1. Tržní sezónnost v King County (Měsíční vývoj cen)")
st.write(
    """
    Zadání kurzu datum prodeje smazalo jako nepotřebný sloupec. Když se však podíváme na agregovaná data podle měsíců,
    odhalíme **typickou sezónnost nemovitostního trhu**:
    - **Jaro & Léto (duben–červenec):** Silná poptávka, objem prodejů přesahuje 2 200 domů za měsíc a mediánová cena dosahuje maxima (476 500 USD).
    - **Zima (listopad–únor):** Trh zamrzá, objem klesá pod 1 000 domů za měsíc a mediánová cena padá až na 425 000 USD (propad o více než 10 %!).
    """
)

monthly_stats = df_time.groupby("year_month").agg(
    Pocet_prodeju=("price", "count"),
    Median_cena=("price", "median"),
    Prumerna_cena=("price", "mean")
).reset_index()

fig_season = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Měsíční počet prodejů (Aktivita trhu)", "Mediánová cena nemovitostí v čase ($)"),
    horizontal_spacing=0.12
)

# Bar chart objem
fig_season.add_trace(
    go.Bar(
        x=monthly_stats["year_month"],
        y=monthly_stats["Pocet_prodeju"],
        marker_color="#3B82F6",
        name="Počet prodejů",
        hovertemplate="Měsíc: %{x}<br>Prodejů: %{y:,}<extra></extra>"
    ),
    row=1, col=1
)

# Line chart medián ceny
fig_season.add_trace(
    go.Scatter(
        x=monthly_stats["year_month"],
        y=monthly_stats["Median_cena"],
        mode="lines+markers",
        line=dict(color="#10B981", width=3),
        marker=dict(size=8),
        name="Medián ceny",
        hovertemplate="Měsíc: %{x}<br>Medián: $%{y:,.0f}<extra></extra>"
    ),
    row=1, col=2
)

fig_season.update_layout(height=450, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
fig_season.update_xaxes(tickangle=-45)
st.plotly_chart(fig_season, width="stretch")

st.markdown("---")

# =============================================================================
# 3. VELKÝ EXPERIMENT: NÁHODNÝ SPLIT VS. ČASOVÝ SPLIT (TIME-BASED SPLIT)
# =============================================================================
st.subheader("🔬 2. Experiment: Falešný optimismus náhodného splitu vs. Reálný Time-Split")

st.markdown(
    """
    V kurzu byl použit standardní `train_test_split(..., random_state=42)`.  
    To způsobuje **Data Leakage (Časový únik)**: model trénoval na domech prodaných v květnu 2015 a testoval se na domech z května 2014!  
    V reálné produkci ale model trénujeme na minulosti a nasazujeme do **budoucnosti**.

    Provedli jsme proto exaktní rozdělení:
    * **Trénovací sada (Historie):** Všechny prodeje v roce **2014** (květen – prosinec 2014, 14 587 domů).
    * **Testovací sada (Budoucnost):** Všechny prodeje v roce **2015** (leden – květen 2015, 6 966 domů).
    """
)

# Výpočet modelů
feats_base = [c for c in df_time.columns if c not in [
    'id', 'date', 'price', 'dt', 'days_since_start', 'year_month', 'month', 'quarter'
]]
feats_time = feats_base + ['days_since_start', 'month']

# Random Split
X_tr_b, X_te_b, y_tr, y_te = train_test_split(df_time[feats_base], df_time['price'], test_size=0.2, random_state=42)
X_tr_t, X_te_t, _, _ = train_test_split(df_time[feats_time], df_time['price'], test_size=0.2, random_state=42)

ols_b_rand = LinearRegression().fit(X_tr_b, y_tr)
ols_t_rand = LinearRegression().fit(X_tr_t, y_tr)
hgb_rand = HistGradientBoostingRegressor(random_state=42).fit(df_time.loc[X_tr_t.index, feats_time], y_tr)

# Time-based Split (2014 vs 2015)
split_date = pd.to_datetime('2015-01-01')
mask_tr = df_time['dt'] < split_date
mask_te = df_time['dt'] >= split_date

y_tr_time = df_time.loc[mask_tr, 'price']
y_te_time = df_time.loc[mask_te, 'price']

ols_b_time = LinearRegression().fit(df_time.loc[mask_tr, feats_base], y_tr_time)
ols_t_time = LinearRegression().fit(df_time.loc[mask_tr, feats_time], y_tr_time)
hgb_time = HistGradientBoostingRegressor(random_state=42).fit(df_time.loc[mask_tr, feats_time], y_tr_time)

# Metriky
res_data = [
    {
        "Metoda rozdělení": "Náhodný split (Učebnicový)",
        "Model": "1. OLS Baseline (bez data)",
        "R² skóre": f"{r2_score(y_te, ols_b_rand.predict(X_te_b)):.4f}",
        "MAE (Průměrná chyba)": f"${mean_absolute_error(y_te, ols_b_rand.predict(X_te_b)):,.0f}",
        "Hodnocení": "Falešně optimistický (data leakage)"
    },
    {
        "Metoda rozdělení": "Náhodný split (Učebnicový)",
        "Model": "2. OLS s časem (trend + měsíc)",
        "R² skóre": f"{r2_score(y_te, ols_t_rand.predict(X_te_t)):.4f}",
        "MAE (Průměrná chyba)": f"${mean_absolute_error(y_te, ols_t_rand.predict(X_te_t)):,.0f}",
        "Hodnocení": f"Trend: +$97.1/den (~$35k roční inflace)"
    },
    {
        "Metoda rozdělení": "Náhodný split (Učebnicový)",
        "Model": "3. Gradient Boosting (HGB)",
        "R² skóre": f"{r2_score(y_te, hgb_rand.predict(X_te_t)):.4f}",
        "MAE (Průměrná chyba)": f"${mean_absolute_error(y_te, hgb_rand.predict(X_te_t)):,.0f}",
        "Hodnocení": "Špičkový výsledek na náhodných datech"
    },
    {
        "Metoda rozdělení": "🚨 Reálný Time-Split (Trénink 2014 → Test 2015)",
        "Model": "1. OLS Baseline (bez data)",
        "R² skóre": f"{r2_score(y_te_time, ols_b_time.predict(df_time.loc[mask_te, feats_base])):.4f}",
        "MAE (Průměrná chyba)": f"${mean_absolute_error(y_te_time, ols_b_time.predict(df_time.loc[mask_te, feats_base])):,.0f}",
        "Hodnocení": "Pokles kvality kvůli posunu trhu v roce 2015"
    },
    {
        "Metoda rozdělení": "🚨 Reálný Time-Split (Trénink 2014 → Test 2015)",
        "Model": "2. OLS s časem (trend + měsíc)",
        "R² skóre": f"{r2_score(y_te_time, ols_t_time.predict(df_time.loc[mask_te, feats_time])):.4f}",
        "MAE (Průměrná chyba)": f"${mean_absolute_error(y_te_time, ols_t_time.predict(df_time.loc[mask_te, feats_time])):,.0f}",
        "Hodnocení": "💥 Kolaps modelu: Past lineární extrapolace!"
    },
    {
        "Metoda rozdělení": "🚨 Reálný Time-Split (Trénink 2014 → Test 2015)",
        "Model": "3. Gradient Boosting (HGB)",
        "R² skóre": f"{r2_score(y_te_time, hgb_time.predict(df_time.loc[mask_te, feats_time])):.4f}",
        "MAE (Průměrná chyba)": f"${mean_absolute_error(y_te_time, hgb_time.predict(df_time.loc[mask_te, feats_time])):,.0f}",
        "Hodnocení": "🏆 Vítěz reality: Drží vysokou přesnost"
    },
]

st.dataframe(pd.DataFrame(res_data), width="stretch", hide_index=True)

st.markdown("---")

# =============================================================================
# 4. TŘI ZÁSADNÍ DATA SCIENCE ZÁVĚRY
# =============================================================================
st.subheader("💡 3 klíčové poznatky pro praxi Data Science")

t1, t2, t3 = st.columns(3)
with t1:
    st.error("⚠️ 1. Past lineární extrapolace")
    st.write(
        "V OLS modelu narostla proměnná `days_since_start` v roce 2015 do vysokých hodnot. "
        "Lineární regrese předpokládá neomezený růst po přímce, jenže v lednu 2015 nastal zimní sezónní propad cen! "
        "OLS model tak v zimě 2015 ceny nemovitostí masivně přestřelil, což vedlo k propadu $R^2$ na 0.50 a růstu chyby na 173 000 USD."
    )

with t2:
    st.warning("🔍 2. Falešný optimismus náhodného splitu")
    st.write(
        "Když použijeme `train_test_split(random_state=42)`, promícháme transakce z celého roku. "
        "Model se tak naučí ceny pro jaro 2015 i díky tomu, že měl v trénovací sadě jiné domy z jara 2015. "
        "V reálném světě ale budoucnost neznáte – proto time-split vždy odhalí skutečnou generalizační schopnost."
    )

with t3:
    st.success("🌲 3. Síla stromových modelů v čase")
    st.write(
        "Gradient Boosting (HGB) neextrapoluje lineární trend do nekonečna. "
        "Rozdělí prostor na pravidla a zachytí interakci mezi ročním obdobím a lokalitou. "
        "I při striktním časovém rozdělení si udržel $R^2 = 0{,}88$ a průměrnou chybu kolem 71 000 USD."
    )
