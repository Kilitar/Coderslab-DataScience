"""
Den 5: Neřízené učení – Velká syntéza & Závěrečný znalostní kvíz
================================================================
Komplexní syntéza 5. dne kurzu Data Science & Machine Learning:
1. Tři stěžejní pilíře neřízeného učení (Redukce dimenzionality, Shlukování, Detekce anomálií).
2. Velká srovnávací matice 5 hlavních algoritmů (PCA, K-Means, DBSCAN, LOF, Isolation Forest).
3. Symbióza neřízeného a řízeného učení (Feature Engineering, čištění dat, eliminace kolinearity).
4. Interaktivní závěrečný test (Quiz Time z přednášky Day_5_summary.pdf).
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Závěr Dne 5: Velká syntéza", page_icon="🏆", layout="wide")

st.title("🏆 Den 5: Velká syntéza neřízeného učení & Znalostní kvíz")
st.caption(
    "Oficiální zakončení 5. dne kurzu (Day 5 summary): Ucelený přehled metod bez učitele, "
    "srovnávací analýza algoritmů PCA, K-Means, DBSCAN, LOF a Isolation Forest, "
    "metodické desatero pro praxi 2026 a interaktivní závěrečný kvíz."
)

tab_pillars, tab_matrix, tab_symbiosis, tab_quiz = st.tabs([
    "🏛️ Tři pilíře neřízeného učení",
    "📊 Velká srovnávací matice",
    "🔄 Symbióza s řízeným učením",
    "🧠 Znalostní kvíz (Quiz Time)"
])

# -----------------------------------------------------------------------------
# TAB 1: Tři pilíře neřízeného učení
# -----------------------------------------------------------------------------
with tab_pillars:
    st.subheader("🏛️ Architektura neřízeného učení (Unsupervised Learning)")
    st.markdown(
        """
        Zatímco v předchozích dnech kurzu jsme se zabývali **řízeným učením** (Supervised Learning), 
        kde algoritmus hledal mapování mezi vstupem $X$ a známým cílem $y$, neřízené učení pracuje 
        **bez jakýchkoliv cílových štítků**. Jeho úkolem je samostatně odhalit skrytou vnitřní strukturu dat.
        """
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("### 1. Redukce dimenzionality")
            st.markdown(
                """
                **Cíl:** Komprese proměnných z $\\mathbb{R}^D$ do $\\mathbb{R}^d$ ($d \\ll D$) s minimální ztrátou rozptylu.

                - **PCA:** Lineární maximalizace rozptylu podél ortogonálních os.
                - **t-SNE:** Nelineární zachování lokálního sousedství pro 2D grafy.
                - **LDA:** Hledání os maximalizujících separaci mezi skupinami.
                """
            )

    with c2:
        with st.container(border=True):
            st.markdown("### 2. Shluková analýza")
            st.markdown(
                """
                **Cíl:** Automatické seskupení podobných pozorování do přirozených klastrů.

                - **K-Means:** Minimalizace WCSS kolem k centroidů (sférické shluky).
                - **DBSCAN:** Hustotní shlukování (nekonvexní tvary, filtrace šumu).
                - **Využití:** Segmentace zákazníků, text clustering, medicína.
                """
            )

    with c3:
        with st.container(border=True):
            st.markdown("### 3. Detekce anomálií")
            st.markdown(
                """
                **Cíl:** Identifikace vzácných, neobvyklých či podezřelých vzorů v datech.

                - **Isolation Forest:** Přímá stromová izolace (škáluje na miliony řádků).
                - **Local Outlier Factor:** Lokální hustota vůči k-sousedům.
                - **Využití:** Detekce bankovních podvodů, kybernetika, čištění dat.
                """
            )

    st.markdown("---")
    st.subheader("⚖️ Klíčové srovnání: Supervised vs. Unsupervised učení")
    comp_df = pd.DataFrame([
        {"Dimenze": "Vstupní data", "Supervised Learning": "Příznaky X + Labely y", "Unsupervised Learning": "Pouze příznaky X (bez labelů)"},
        {"Dimenze": "Hlavní cíl", "Supervised Learning": "Predikce cílové proměnné", "Unsupervised Learning": "Hledání vnitřní struktury dat"},
        {"Dimenze": "Měření úspěšnosti", "Supervised Learning": "Objektivní testovací metriky (RMSE, Accuracy)", "Unsupervised Learning": "Interní metriky (Silhouette, Inertia, EVR)"},
        {"Dimenze": "Validace v produkci", "Supervised Learning": "Snadná (porovnání ground truth s predikcí)", "Unsupervised Learning": "Vyžaduje doménovou expertízu a byznys evaluaci"},
    ])
    st.dataframe(comp_df, width="stretch", hide_index=True)

# -----------------------------------------------------------------------------
# TAB 2: Velká srovnávací matice
# -----------------------------------------------------------------------------
with tab_matrix:
    st.subheader("📊 Velká srovnávací matice algoritmů Dne 5")
    st.markdown(
        """
        Komplexní přehled pěti klíčových algoritmů probraných během Dne 5. 
        Tato tabulka slouží jako referenční manuál pro volbu správného modelu v produkční praxi:
        """
    )

    full_matrix = pd.DataFrame([
        {
            "Algoritmus": "PCA",
            "Kategorie": "Redukce dimenzionality",
            "Základní princip": "Ortogonální rotace souřadnic maximalizující rozptyl",
            "Složitost": "O(min(N³, D³))",
            "Škálování": "✅ Kriticky nutné",
            "Klíčový hyperparametr": "n_components",
            "Ideální pro": "Odstranění multikolinearity, komprese, 2D vizualizace"
        },
        {
            "Algoritmus": "K-Means",
            "Kategorie": "Centroidní shlukování",
            "Základní princip": "Minimalizace kvadratické vzdálenosti k těžištím (WCSS)",
            "Složitost": "O(N · k · I · D)",
            "Škálování": "✅ Kriticky nutné",
            "Klíčový hyperparametr": "n_clusters, n_init",
            "Ideální pro": "Rychlá segmentace sférických konvexních shluků"
        },
        {
            "Algoritmus": "DBSCAN",
            "Kategorie": "Hustotní shlukování",
            "Základní princip": "Hledání souvislých oblastí s hustotou min_samples v poloměru eps",
            "Složitost": "O(N log N) až O(N²)",
            "Škálování": "✅ Kriticky nutné",
            "Klíčový hyperparametr": "eps, min_samples",
            "Ideální pro": "Nekonvexní shluky (půlměsíce, kruhy), robustnost k šumu"
        },
        {
            "Algoritmus": "LOF",
            "Kategorie": "Detekce anomálií",
            "Základní princip": "Poměr lokální dosažitelné hustoty bodu k jeho k-sousedům",
            "Složitost": "O(N² · D)",
            "Škálování": "✅ Kriticky nutné",
            "Klíčový hyperparametr": "n_neighbors, contamination",
            "Ideální pro": "Shluková data s proměnlivou hustotou, N < 50 000"
        },
        {
            "Algoritmus": "Isolation Forest",
            "Kategorie": "Detekce anomálií",
            "Základní princip": "Přímá stromová izolace vzácných bodů mělkými řezy",
            "Složitost": "O(t · ψ log ψ) — Lineární",
            "Škálování": "❌ Není nutné",
            "Klíčový hyperparametr": "n_estimators, contamination",
            "Ideální pro": "Masivní vysokodimenzionální data, N > 100 000"
        }
    ])
    st.dataframe(full_matrix, width="stretch", hide_index=True)

    st.markdown("---")
    st.subheader("🧭 Průvodce rozhodováním: Jaký algoritmus zvolit?")
    
    dec_col1, dec_col2 = st.columns(2)
    with dec_col1:
        st.markdown(
            r"""
            **Otázka 1: Jaký je primární byznysový cíl?**
            - **Komprese nebo vizualizace dat:** Použijte **PCA** (lineární) nebo **t-SNE** (nelineární).
            - **Rozdělení zákazníků / entit do skupin:** Použijte **K-Means** (známé $k$, sférické) nebo **DBSCAN** (neznámý počet, libovolný tvar).
            - **Vyhledání podezřelých transakcí či selhání:** Použijte **Isolation Forest** (velká data) nebo **LOF** (proměnlivá hustota shluků).
            """
        )
    with dec_col2:
        st.markdown(
            r"""
            **Otázka 2: Jaká je velikost a povaha dat?**
            - **$N > 100\,000$ řádků:** Vyhněte se LOF a plnému DBSCAN (riziko přetečení RAM). Zvolte **Isolation Forest** a **MiniBatchKMeans**.
            - **Nelineární vazby v prostoru:** K-Means i PCA selžou; nasaďte **DBSCAN** a **Kernel PCA / t-SNE**.
            - **Neškálovaná data v různých jednotkách:** Zvolte **Isolation Forest** (stromy jsou invariantní), nebo nezapomeňte na `StandardScaler`.
            """
        )

# -----------------------------------------------------------------------------
# TAB 3: Symbióza s řízeným učením
# -----------------------------------------------------------------------------
with tab_symbiosis:
    st.subheader("🔄 Symbióza: Jak neřízené učení akceleruje Supervised ML modely")
    st.markdown(
        """
        V reálných datových pipelines moderních firem nefunguje neřízené učení izolovaně. 
        Představuje **nejvýkonnější nástroj pro Feature Engineering a čištění dat** před natrénováním 
        regresních a klasifikačních modelů (XGBoost, Random Forest, CatBoost, neuronové sítě):
        """
    )

    s_c1, s_c2 = st.columns(2)
    with s_c1:
        with st.container(border=True):
            st.markdown("### 1. Čištění trénovacích dat od toxických anomálií")
            st.markdown(
                """
                Odlehlá pozorování (např. chyby zápisu z čidel) brutálně vychylují regresní přímku OLS 
                nebo nutí stromy štěpit zbytečné hluboké větve. 
                
                **Workflow:** Natrénujte `IsolationForest`, vyfiltrujte nejextrémnější 1–2 % anomálií 
                a finální model trénujte na čistých datech.
                """
            )
        with st.container(border=True):
            st.markdown("### 2. Eliminace multikolinearity v lineárních modelech")
            st.markdown(
                """
                Pokud máme stovky silně korelovaných znaků, OLS má nestabilní koeficienty s obří variancí (vysoké VIF).
                
                **Workflow:** Aplikujte `PCA`, ponechte prvních $k$ ortogonálních komponent 
                vysvětlujících 95 % rozptylu (Principal Component Regression - PCR).
                """
            )

    with s_c2:
        with st.container(border=True):
            st.markdown("### 3. Klastrové vzdálenosti jako nové prediktory")
            st.markdown(
                """
                Natrénujte `KMeans(n_clusters=k)` na datech a pomocí `.transform(X)` 
                vygenerujte vzdálenosti ke všem $k$ těžištím. 
                
                **Workflow:** Tyto vzdálenosti přidejte jako nové nelineární příznaky pro gradient boosting. 
                Strom pak dokáže snadno oddělit lokální chování trhu.
                """
            )
        with st.container(border=True):
            st.markdown("### 4. Semi-Supervised Learning & Pseudo-Labeling")
            st.markdown(
                """
                Pokud máte 100 000 neoznačených řádků a pouze 1 000 označených, shlukování seskupí 
                neoznačená data kolem známých vzorků.
                
                **Workflow:** Přenesení labelů na těsně přiléhající sousedy uvnitř hustých shluků 
                dramaticky rozšiřuje trénovací množinu.
                """
            )

# -----------------------------------------------------------------------------
# TAB 4: Znalostní kvíz (Quiz Time)
# -----------------------------------------------------------------------------
with tab_quiz:
    st.subheader("🧠 Znalostní kvíz: Prověřte své znalosti z Dne 5 (Quiz Time)")
    st.markdown(
        """
        Oficiální kvíz z přednášky (*Day 5 summary, slajd 15*). 
        Odpovězte na 5 klíčových koncepčních otázek pokrývajících redukci dimenzionality, shlukování a detekci anomálií:
        """
    )

    score = 0
    total_q = 5

    # Otázka 1
    st.markdown("#### 1. Proč algoritmus DBSCAN lépe separuje nekonvexní shluky (např. dva do sebe vnořené půlměsíce) než K-Means?")
    ans1 = st.radio(
        "Vyberte správnou odpověď na otázku 1:",
        [
            "Protože DBSCAN používá náhodné stromy, které dělí prostor rovnoběžně s osami.",
            "Protože DBSCAN hledá hustotně souvislé oblasti libovolného tvaru, zatímco K-Means předpokládá sférické shluky kolem centroidů.",
            "Protože K-Means neumí pracovat s více než dvěma shluky.",
            "Protože DBSCAN automaticky normalizuje data pomocí StandardScaler."
        ],
        index=None,
        key="q1"
    )

    # Otázka 2
    st.markdown("#### 2. Jaká je základní myšlenka algoritmu Isolation Forest při detekci odlehlých pozorování?")
    ans2 = st.radio(
        "Vyberte správnou odpověď na otázku 2:",
        [
            "Anomálie tvoří nejhustší shluk v datech a vyžadují hluboké stromy.",
            "Anomálie jsou vzácné a odlišné, proto je náhodné řezy stromů izolují v malé hloubce (krátká cesta h(x)).",
            "Anomálie se určují výhradně pomocí Z-skóre z kovarianční matice.",
            "Model počítá eukleidovskou vzdálenost ke všem ostatním bodům v celém datasetu."
        ],
        index=None,
        key="q2"
    )

    # Otázka 3
    st.markdown("#### 3. Co hrozí u algoritmu Local Outlier Factor (LOF), pokud zvolíme příliš malý počet sousedů (např. k = 2)?")
    ans3 = st.radio(
        "Vyberte správnou odpověď na otázku 3:",
        [
            "Výpočet se stane kvadraticky pomalejším.",
            "Dojde k fenoménu Outlier Masking – dvojice blízkých anomálií se navzájem označí za husté okolí a model je přehlédne.",
            "Model odmítne predikovat záporné hodnoty offsetu.",
            "Všechny body budou označeny za běžná pozorování s LOF = 0."
        ],
        index=None,
        key="q3"
    )

    # Otázka 4
    st.markdown("#### 4. Které tvrzení o metodě hlavních komponent (PCA) je PRAVDIVÉ?")
    ans4 = st.radio(
        "Vyberte správnou odpověď na otázku 4:",
        [
            "První hlavní komponenta (PC1) vždy vysvětluje největší podíl celkového rozptylu dat.",
            "PCA vyžaduje, aby data nebyla předem škálována, jinak ztrácí smysl.",
            "Jednotlivé hlavní komponenty jsou mezi sebou silně korelované.",
            "PCA je nelineární algoritmus ideální pro soustředné kruhy."
        ],
        index=None,
        key="q4"
    )

    # Otázka 5
    st.markdown("#### 5. Jak funguje parametr contamination v Scikit-learn implementacích LOF a Isolation Forest?")
    ans5 = st.radio(
        "Vyberte správnou odpověď na otázku 5:",
        [
            "Přímo mění skóre anomálií jednotlivých bodů v datech.",
            "Určuje počet stromů v lese.",
            "Nastavuje percentilový rozhodovací práh (offset), kterým se spojité skóre dělí na inliers (+1) a outliers (-1).",
            "Automaticky maže podezřelá data z pevného disku."
        ],
        index=None,
        key="q5"
    )

    st.markdown("---")
    if st.button("Vyhodnotit kvíz", type="primary"):
        # Evaluace
        q1_ok = (ans1 == "Protože DBSCAN hledá hustotně souvislé oblasti libovolného tvaru, zatímco K-Means předpokládá sférické shluky kolem centroidů.")
        q2_ok = (ans2 == "Anomálie jsou vzácné a odlišné, proto je náhodné řezy stromů izolují v malé hloubce (krátká cesta h(x)).")
        q3_ok = (ans3 == "Dojde k fenoménu Outlier Masking – dvojice blízkých anomálií se navzájem označí za husté okolí a model je přehlédne.")
        q4_ok = (ans4 == "První hlavní komponenta (PC1) vždy vysvětluje největší podíl celkového rozptylu dat.")
        q5_ok = (ans5 == "Nastavuje percentilový rozhodovací práh (offset), kterým se spojité skóre dělí na inliers (+1) a outliers (-1).")

        score = sum([q1_ok, q2_ok, q3_ok, q4_ok, q5_ok])

        if score == 5:
            st.balloons()
            st.success(f"🎉 **Fantastický výsledek! Získali jste plný počet: {score} / {total_q} bodů (100 %).** Dokonale ovládáte neřízené učení Dne 5!")
        elif score >= 3:
            st.info(f"👍 **Dobrý výsledek: {score} / {total_q} bodů ({score/total_q*100:.0f} %).** Projděte si níže vysvětlení k otázkám, kde jste váhali.")
        else:
            st.warning(f"⚠️ **Výsledek: {score} / {total_q} bodů.** Doporučujeme projít si teoretické průvodce a vzorové implementace v sekcích Dne 5.")

        # Zpětná vazba po otázkách
        with st.expander("Rozbalit detailní vysvětlení odpovědí", expanded=True):
            st.markdown(f"1. **DBSCAN vs K-Means:** {'✅ Správně' if q1_ok else '❌ Chyba'} – DBSCAN se nespoléhá na centroidy, nýbrž propojuje body v hustých oblastech libovolné geometrie.")
            st.markdown(f"2. **Princip Isolation Forest:** {'✅ Správně' if q2_ok else '❌ Chyba'} – Izolace v mělkých hloubkách je základním kamenem algoritmu (Liu et al., 2008).")
            st.markdown(f"3. **Problém malého k u LOF:** {'✅ Správně' if q3_ok else '❌ Chyba'} – Dvojice blízkých anomálií se navzájem zamaskuje, proto se doporučuje k v rozmezí 10 až 20.")
            st.markdown(f"4. **Vlastnost PC1:** {'✅ Správně' if q4_ok else '❌ Chyba'} – PC1 je definována jako směr s maximálním rozptylem a komponenty jsou vzájemně ortogonální (nekorelované).")
            st.markdown(f"5. **Parametr contamination:** {'✅ Správně' if q5_ok else '❌ Chyba'} – Skóre bodů je invariantní; kontaminace slouží výhradně k určení hraničního kvantilu.")
