import streamlit as st

st.title("🧭 Expertní analýza: Interaktivní průvodce výběrem modelu")
st.caption("Jak přemýšlet jako seniorní Machine Learning inženýr: Rozhodovací strom pro volbu správného regresního algoritmu podle povahy vašich dat a byznys cílů.")

st.markdown(r"""
Častá otázka studentů: *„Mám nová data. Podle čeho se mám rozhodnout, zda použít OLS, Ridge, Lasso, Polynom nebo Rozhodovací strom?“*  
Odpovězte na několik otázek o vašem projektu a algoritmus vám doporučí optimální postup:
""")

with st.container(border=True):
    q_data_type = st.radio(
        "1. Jaký charakter mají vaše prediktory (nezávislé proměnné)?",
        [
            "Převážně spojité fyzikální veličiny s plynulými závislostmi (např. teplota, koncentrace, rozměry).",
            "Směs kategorií, skokových tarifů, administrativních zón a tabulkových čísel.",
            "Obrovské množství sloupců (desítky až stovky), z nichž mnohé jsou silně korelované nebo zbytečné."
        ]
    )

    q_goal = st.radio(
        "2. Co je hlavním byznys cílem vašeho modelu?",
        [
            "100% čistá interpretovatelnost koeficientů pro management nebo regulátora (např. bankovní scoring, ekonometrie).",
            "Maximální možná predikční přesnost bez ohledu na složitost vnitřních pravidel.",
            "Automatický výběr nejdůležitějších faktorů (Feature Selection) z obřího množství proměnných."
        ]
    )

    q_extrapol = st.radio(
        "3. Budete předpovídat data ležící mimo rozsah dosud naměřených hodnot (extrapolace)?",
        [
            "Ano (např. předpověď budoucího trendu růstu, inflace nebo extrémních hodnot).",
            "Ne (nová data budou v podobném rozsahu jako dosavadní trénovací data)."
        ]
    )

st.markdown("### 🎯 Doporučená strategie pro váš projekt:")

# Vyhodnocení
if "extrapolace" in q_extrapol and "Ano" in q_extrapol:
    st.error(
        "🚫 **STOP: Zákaz použití Rozhodovacích stromů!**  \n"
        "Rozhodovací stromy (CART) ani lesy (Random Forest) **neumí extrapolovat**. Mimo rozsah trénovacích dat narazí na konstantní strop.  \n"
        "👉 **Doporučení:** Použijte **Lineární regresi s L2 regularizací (Ridge)** nebo polynomiální trend nízkého stupně."
    )
elif "Obrovské množství sloupců" in q_data_type or "Feature Selection" in q_goal:
    st.success(
        "🏆 **Doporučený model: Lasso regrese (L1) nebo Elastic Net**  \n"
        "Díky L1 penalizaci model automaticky vynuluje šumové a redundantní příznaky a zanechá pouze klíčové tahouny.  \n"
        "💡 **Pozor:** Nezapomeňte před trénováním data normalizovat pomocí `StandardScaler`!"
    )
elif "Směs kategorií" in q_data_type or "Maximální možná" in q_goal:
    st.success(
        "🏆 **Doporučený model: Rozhodovací strom (CART) s prořezáním (nebo Random Forest v Dni 2)**  \n"
        "Stromy excelují na tabulkových datech, nevyžadují škálování a perfektně zvládají pravoúhlé ohraničení zón a skokové tarify.  \n"
        "💡 **Pozor:** Vždy ladit `max_depth` a `min_samples_leaf`, aby nedošlo k gigantickému přeučení!"
    )
elif "interpretovatelnost" in q_goal:
    st.success(
        "🏆 **Doporučený model: OLS Lineární regrese nebo Ridge regrese**  \n"
        "Každý koeficient beta má přímou ekonomickou interpretaci (*'při zvýšení plochy o 1 m² vzroste cena o X Kč'*).  \n"
        "Pokud jsou prediktory korelované, přidejte lehkou L2 penalizaci (Ridge)."
    )
else:
    st.info(
        "🏆 **Doporučený postup: Začněte OLS přímkou jako baselinou a porovnejte s prořezaným stromem.**"
    )

st.markdown("---")
st.markdown("### 🗺️ Interaktivní vizuální průvodce: Váš rozhodovací strom výběru")
st.markdown("Přehledná vizuální navigace krok za krokem. Všechny karty mají **velké, čisté písmo** a dynamicky se přizpůsobují vašim volbám:")

# Kroky rozhodování podle voleb nahoře
krok1_nazev = "Tabulková data"
krok1_popis = "Máme k dispozici strukturovaná data s čísly a kategoriemi."

is_extrapol = "extrapolace" in q_extrapol and "Ano" in q_extrapol
is_lasso = "Obrovské množství sloupců" in q_data_type or "Feature Selection" in q_goal
is_tree = "Směs kategorií" in q_data_type or "Maximální možná" in q_goal

# 1. ŘADA: Vstupní bod a první rozcestí
c_step1, c_arrow1, c_step2 = st.columns([4, 1, 4])
with c_step1:
    st.markdown("""
    <div style="background-color: #1E293B; border: 2px solid #3B82F6; border-radius: 12px; padding: 18px; text-align: center;">
        <div style="font-size: 1.25em; font-weight: bold; color: #60A5FA; margin-bottom: 6px;">📂 Krok 1: Typ vstupních dat</div>
        <div style="font-size: 1.05em; color: #E2E8F0;">Máme tabulková data (čísla, kategorie, sloupce)</div>
    </div>
    """, unsafe_allow_html=True)

with c_arrow1:
    st.markdown("""
    <div style="text-align: center; padding-top: 25px; font-size: 2em; color: #94A3B8;">➡️</div>
    """, unsafe_allow_html=True)

with c_step2:
    border_col2 = "#10B981" if is_extrapol else "#3B82F6"
    st.markdown(f"""
    <div style="background-color: #1E293B; border: 2px solid {border_col2}; border-radius: 12px; padding: 18px; text-align: center;">
        <div style="font-size: 1.25em; font-weight: bold; color: #60A5FA; margin-bottom: 6px;">🎯 Krok 2: Potřeba extrapolace?</div>
        <div style="font-size: 1.05em; color: #E2E8F0;">{"⚠️ ANO (Předpovídáme do budoucna / mimo data)" if is_extrapol else "✔️ NE (Predikce v rámci známého rozsahu)"}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

# 2. ŘADA: Větvění na základě kroků
if is_extrapol:
    st.markdown("""
    <div style="background-color: #064E3B; border: 3px solid #10B981; border-radius: 16px; padding: 24px; margin-top: 10px;">
        <div style="font-size: 1.5em; font-weight: bold; color: #34D399; margin-bottom: 8px;">
            🏆 Vítězný model: Ridge regrese (L2) nebo Lineární trend
        </div>
        <div style="font-size: 1.15em; color: #F1F5F9; line-height: 1.6;">
            <b>Proč právě tento model?</b><br>
            Při potřebě předpovídat <b>mimo dosud naměřené hodnoty</b> (např. inflace, růst firmy za 5 let) 
            je <span style="color: #F87171; font-weight: bold;">přísně zakázáno používat rozhodovací stromy i lesy</span>! 
            Stromy by narazily na konstantní strop trénovacích dat. Lineární model s L2 regularizací naopak bezpečně drží směr trendu a tlumí výkyvy.
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    # Krok 3: Nelinearity vs. Počet sloupců
    c_sub1, c_arrow2, c_sub2 = st.columns([4, 1, 4])
    with c_sub1:
        border_col3 = "#10B981" if is_tree else "#475569"
        st.markdown(f"""
        <div style="background-color: #1E293B; border: 2px solid {border_col3}; border-radius: 12px; padding: 18px; text-align: center;">
            <div style="font-size: 1.25em; font-weight: bold; color: #38BDF8; margin-bottom: 6px;">🌲 Větev A: Nelinearity & Skoky</div>
            <div style="font-size: 1.05em; color: #E2E8F0;">Zóny, skokové tarify, pravoúhlé hranice</div>
        </div>
        """, unsafe_allow_html=True)

    with c_arrow2:
        st.markdown("""
        <div style="text-align: center; padding-top: 25px; font-size: 2em; color: #94A3B8;">nebo</div>
        """, unsafe_allow_html=True)

    with c_sub2:
        border_col4 = "#10B981" if (is_lasso or not is_tree) else "#475569"
        st.markdown(f"""
        <div style="background-color: #1E293B; border: 2px solid {border_col4}; border-radius: 12px; padding: 18px; text-align: center;">
            <div style="font-size: 1.25em; font-weight: bold; color: #F59E0B; margin-bottom: 6px;">📐 Větev B: Plynulé vztahy</div>
            <div style="font-size: 1.05em; color: #E2E8F0;">Mnoho sloupců, šum, požadavek na koeficienty</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # Finální doporučení
    if is_tree:
        st.markdown("""
        <div style="background-color: #064E3B; border: 3px solid #10B981; border-radius: 16px; padding: 24px;">
            <div style="font-size: 1.5em; font-weight: bold; color: #34D399; margin-bottom: 8px;">
                🏆 Vítězný model: Rozhodovací strom (CART) s prořezáním (nebo Random Forest)
            </div>
            <div style="font-size: 1.15em; color: #F1F5F9; line-height: 1.6;">
                <b>Proč právě tento model?</b><br>
                Vaše data obsahují kategorie, geografické zóny nebo skoky (jako u diamantů na 1.00 karátu). 
                Rozhodovací strom nevyžaduje škálování, nezajímá ho multikolinearita a vytvoří přesná if-else pravidla.
                Nezapomeňte však omezit <code>max_depth</code> a <code>min_samples_leaf</code>, abyste se vyhnuli přeučení.
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif is_lasso:
        st.markdown("""
        <div style="background-color: #064E3B; border: 3px solid #10B981; border-radius: 16px; padding: 24px;">
            <div style="font-size: 1.5em; font-weight: bold; color: #34D399; margin-bottom: 8px;">
                🏆 Vítězný model: Lasso regrese (L1) nebo Elastic Net
            </div>
            <div style="font-size: 1.15em; color: #F1F5F9; line-height: 1.6;">
                <b>Proč právě tento model?</b><br>
                Máte obrovské množství prediktorů a potřebujete automatický výběr (Feature Selection). 
                L1 regularizace vynuluje váhy nepodstatných proměnných. Před trénováním nezapomeňte data normalizovat pomocí <code>StandardScaler</code>!
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background-color: #064E3B; border: 3px solid #10B981; border-radius: 16px; padding: 24px;">
            <div style="font-size: 1.5em; font-weight: bold; color: #34D399; margin-bottom: 8px;">
                🏆 Vítězný model: OLS Lineární regrese (s lehkou Ridge penalizací)
            </div>
            <div style="font-size: 1.15em; color: #F1F5F9; line-height: 1.6;">
                <b>Proč právě tento model?</b><br>
                Požadujete maximální možnou interpretovatelnost pro management či regulátory. 
                Každý koeficient beta má přímý finanční význam (*„zvětšení plochy o 1 m² zvýší cenu o X Kč“*).
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)

# Přehledná referenční srovnávací tabulka pro rychlou orientaci (velká a čitelná)
st.markdown("#### 📋 Rychlý tahák pro volbu modelu (Cheat-Sheet):")
summary_chooser = [
    {"Kritérium / Situace": "Potřebuji extrapolovat do budoucna", "Doporučený model": "Lineární regrese / Ridge (L2)", "Zakázaný model": "Rozhodovací stromy (strop!)"},
    {"Kritérium / Situace": "Mám stovky sloupců a šum", "Doporučený model": "Lasso regrese (L1) / Elastic Net", "Zakázaný model": "Základní OLS (přeučení)"},
    {"Kritérium / Situace": "Skokové tarify, zóny, kategorie", "Doporučený model": "Rozhodovací strom (CART) / Random Forest", "Zakázaný model": "Čistá přímka OLS"},
    {"Kritérium / Situace": "Bankovní scoring, regulace, audit", "Doporučený model": "OLS Lineární regrese", "Zakázaný model": "Složité polynomy a hluboké stromy"}
]
st.dataframe(pd.DataFrame(summary_chooser), width="stretch")

