"""
Den 5: Neřízené učení – Teorie detekce anomálií & Local Outlier Factor (LOF)
============================================================================
Syntéza materiálů z Anomaly_detection_-_introduction.pdf:
1. Úvod do detekce anomálií a odlehlých hodnot (Outlier Mining).
2. Taxonomie anomálií: Bodové (Point), Kontextové (Contextual) a Kolektivní (Collective).
3. Široké průmyslové uplatnění: Fraud, Health, Predictive Maintenance, Quality Control, Seismology.
4. Přehled metodik: Statistické testy (3-sigma), DBSCAN, Isolation Forest a Local Outlier Factor.
5. Matematické základy LOF: k-distance, Reachability Distance, LRD a výsledný poměr LOF.
6. Metodické guardrails pro detekci anomálií v produkci 2026.
"""

import streamlit as st

st.set_page_config(page_title="Teorie: Detekce anomálií & LOF", page_icon="📖", layout="wide")

st.title("📖 Teorie: Detekce anomálií & Local Outlier Factor (LOF)")
st.caption(
    "Teoretický základ z přednáškových materiálů kurzu (Anomaly detection - an introduction). "
    "Taxonomie odlehlých hodnot, průmyslové use-cases, princip stromových ansámblů (Isolation Forest) "
    "a rigorózní matematická konstrukce hustotního algoritmu Local Outlier Factor (LOF)."
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌐 1. Úvod & Typologie anomálií",
    "🏭 2. Průmyslové & Byznysové aplikace",
    "📐 3. Matematická konstrukce LOF",
    "⚖️ 4. Srovnání detekčních paradigmat",
    "🛡️ 5. Metodické guardrails pro rok 2026"
])

# ==============================================================================
# TAB 1: ÚVOD & TYPOLOGIE ANOMÁLIÍ
# ==============================================================================
with tab1:
    st.subheader("1. Co je detekce anomálií a jak se odchylky klasifikují?")
    st.markdown(
        r"""
        V organizacích po celém světě roste podíl rozhodnutí řízených daty (**Data-Driven Decisions**). 
        Proces trénování modelů strojového učení se znečištěnými daty vede k výstavbě chybných predikčních modelů. 
        **Detekce anomálií (Anomaly Detection)** je proces identifikace neobvyklých, vzácných nebo odlehlých pozorování, 
        která se výrazně odlišují od většinového vzorce v datasetu.
        
        Přednáška kurzu (slajdy 5–6) člení anomálie do tří fundamentálních typů:
        """
    )

    c_typ1, c_typ2, c_typ3 = st.columns(3)

    with c_typ1:
        st.error(
            r"""
            ##### 🔴 1. Bodové anomálie (Point Anomalies)
            * **Definice:** Jednotlivé pozorování, jehož hodnota se drasticky odchyluje od zbytku celého datasetu.
            * **Příklad:** Jednorázový převod částky $5\,000\,000\text{ Kč}$ z účtu, kde historicky probíhaly transakce v řádu stokorun.
            * **Řešení:** Globální testy odlehlosti, Z-skóre, Isolation Forest, LOF.
            """
        )

    with c_typ2:
        st.warning(
            r"""
            ##### 🟡 2. Kontextové anomálie (Contextual)
            * **Definice:** Pozorování, které je v obecném kontextu normální, ale v konkrétním rámci (čas, lokalita, sezóna) představuje anomálii.
            * **Příklad:** Venkovní teplota $32\,^\circ\text{C}$ je v červenci běžná, ale v lednu ve střední Evropě představuje extrémní anomálii.
            * **Řešení:** Segmentace dle kontextových atributů, časové řady (SARIMA, Prophet).
            """
        )

    with c_typ3:
        st.info(
            r"""
            ##### 🔵 3. Kolektivní anomálie (Collective)
            * **Definice:** Skupina datových bodů, které jednotlivě nevybočují, ale jako celek tvoří anomální sekvenci či vzorec.
            * **Příklad:** 50 plateb po sobě ve výši $1\text{ USD}$ během 30 sekund (tzv. carding test ukradené karty).
            * **Řešení:** Markovovy modely, rekurentní sítě (LSTM), sekvenční analýza.
            """
        )

    st.markdown("---")
    st.markdown("#### 🎯 Proč je detekce anomálií samostatnou disciplínou (slajd 7)?")
    st.markdown(
        r"""
        * **Extrémní vzácnost:** Anomálie tvoří typicky $<1\,\%$ vzorku. Klasické klasifikátory selhávají na masivní nevyváženosti tříd (Class Imbalance).
        * **Neexistence popisků (Unsupervised povaha):** V reálném nasazení nemáme k dispozici označená trénovací data s labelem *„podvodník“* či *„havárie turbíny“*. Model se musí naučit definovat hranice normality sám.
        * **Proměnlivost normálu (Concept Drift):** Co je normální chování uživatele v pátek večer, je anomálií v pondělí ráno. Algoritmus musí být adaptabilní na evoluci vzorců.
        """
    )

# ==============================================================================
# TAB 2: PRŮMYSLOVÉ & BYZNYSOVÉ APLIKACE
# ==============================================================================
with tab2:
    st.subheader("2. Široké průmyslové a komerční uplatnění (slajdy 9–10)")
    st.markdown(
        r"""
        Detekce anomálií představuje jednu z komerčně nejcennějších aplikací strojového učení. 
        Přednáška uvádí pět klíčových aplikačních domén:
        """
    )

    c_app1, c_app2 = st.columns(2)

    with c_app1:
        st.markdown(
            r"""
            ##### 💳 1. Finanční transakce a boj proti podvodům (Fraud Detection)
            * **Sledovaná data:** Částka, čas, frekvence plateb, geografická lokace terminálu, IP adresa.
            * **Mechanismus:** Okamžitá blokace podezřelého převodu, který se vymyká profilu uživatele.
            
            ##### 🩺 2. Zdravotnictví & Nositelná elektronika (Healthcare IoT)
            * **Sledovaná data:** EKG křivka, srdeční tep, krevní tlak, saturace kyslíkem (SpO2).
            * **Mechanismus:** Chytré hodinky v reálném čase detekují fibrilaci síní nebo náhlý kolaps vitálních funkcí a přivolají pomoc.
            
            ##### ⚙️ 3. Monitorování výkonu strojů (Equipment Performance Monitoring)
            * **Sledovaná data:** Akustické emise, vibrační senzory, teplota ložisek elektromotorů.
            * **Mechanismus:** Prediktivní údržba (Predictive Maintenance) odhalí mikrotrhlinu měsíce před katastrofálním zadřením motoru.
            """
        )

    with c_app2:
        st.markdown(
            r"""
            ##### 🏭 4. Montážní a výrobní linky (Automated Quality Inspection)
            * **Sledovaná data:** Obrazová data z průmyslových kamer, 3D laserové profily svárů.
            * **Mechanismus:** Vizuální defektoskopie výrobků při vysokorychlostním posuvu linky (odhalení škrábanců, vad laku).
            
            ##### ⛏️ 5. Těžební průmysl a seismologie (Seismic Safety)
            * **Sledovaná data:** Frekvenční spektra ze seismografů rozmístěných v hlubinných dolech.
            * **Mechanismus:** Včasná detekce otřesů půdy a pnutí hornin indikující bezprostřední riziko závalu a nutnost evakuace.
            
            ##### 🧹 6. Čištění trénovacích dat pro ML (Data Preprocessing)
            * **Dopad:** Extrémní odlehlé hodnoty zásadně zkreslují ztrátové funkce ($MSE$) a koeficienty OLS regrese. Očištění dat před trénováním zvyšuje stabilitu a přesnost modelů.
            """
        )

# ==============================================================================
# TAB 3: MATEMATICKÁ KONSTRUKCE LOF
# ==============================================================================
with tab3:
    st.subheader("3. Rigorózní matematická konstrukce Local Outlier Factor (LOF)")
    st.markdown(
        r"""
        Algoritmus **Local Outlier Factor (Breunig et al., 2000)** měří stupeň odlehlosti bodu vztažený k hustotě jeho bezprostředního okolí. 
        Skládá se ze čtyř přesných matematických definic:
        """
    )

    st.markdown(
        r"""
        #### 🔹 Krok 1: $k$-vzdálenost bodu $p$ ($k\text{-distance}(p)$)
        Pro celé číslo $k \ge 1$ je $k$-vzdálenost bodu $p$, značeno $d_k(p)$, eukleidovská vzdálenost $d(p, o)$ mezi $p$ a objektem $o \in D$ taková, že:
        1. Nejméně pro $k$ objektů $o' \in D \setminus \{p\}$ platí $d(p, o') \le d(p, o)$.
        2. Nanejvýš pro $k-1$ objektů platí $d(p, o') < d(p, o)$.
        
        Množina $k$ nejbližších sousedů bodu $p$ se značí $N_k(p)$.
        
        ---
        
        #### 🔹 Krok 2: Dosahová vzdálenost (Reachability Distance)
        Dosahová vzdálenost bodu $p$ vzhledem k referenčnímu sousedovi $o$ je definována jako:
        
        $$\text{reach-dist}_k(p, o) = \max\left\{ d_k(o),\, d(p, o) \right\}$$
        
        > **Pedagogický význam:** Pokud bod $p$ leží velmi blízko bodu $o$ (uvnitř jeho $k$-sféry), vzdálenost se uměle „zarazí“ na hodnotě $d_k(o)$. Tím se dramaticky potlačují náhodné fluktuace vzdáleností uvnitř hustých shluků!
        
        ---
        
        #### 🔹 Krok 3: Lokální dosahová hustota (Local Reachability Density, LRD)
        Lokální dosahová hustota $\text{lrd}_k(p)$ je převrácenou hodnotou průměrné dosahové vzdálenosti bodu $p$ od jeho $k$ nejbližších sousedů:
        
        $$\text{lrd}_k(p) = \frac{|N_k(p)|}{\sum_{o \in N_k(p)} \text{reach-dist}_k(p, o)}$$
        
        * Má-li bod kolem sebe těsně natěsnané sousedy $\to$ dosahová vzdálenost je malá $\to$ $\text{lrd}_k(p)$ je **vysoká**.
        * Leží-li bod v osamocení $\to$ dosahová vzdálenost je velká $\to$ $\text{lrd}_k(p)$ je **nízká**.
        
        ---
        
        #### 🔹 Krok 4: Výsledný Local Outlier Factor (LOF)
        Faktor LOF bodu $p$ je průměrný poměr lokálních dosahových hustot jeho sousedů ku hustotě samotného bodu $p$:
        
        $$\text{LOF}_k(p) = \frac{\sum_{o \in N_k(p)} \frac{\text{lrd}_k(o)}{\text{lrd}_k(p)}}{|N_k(p)|} = \frac{\frac{1}{|N_k(p)|} \sum_{o \in N_k(p)} \text{lrd}_k(o)}{\text{lrd}_k(p)}$$
        """
    )

    st.markdown("#### 🎯 Interpretace hodnot LOF skóre:")
    c_i1, c_i2, c_i3 = st.columns(3)
    with c_i1:
        st.success(
            r"""
            **$\text{LOF} \approx 1.0$ (Inlier)**
            Hustota bodu je srovnatelná s hustotou jeho sousedů. Bod leží uvnitř homogenního shluku.
            """
        )
    with c_i2:
        st.info(
            r"""
            **$\text{LOF} < 1.0$ (Dense Core)**
            Bod se nachází v ještě hustším jádru shluku než jeho okolní sousedé.
            """
        )
    with c_i3:
        st.error(
            r"""
            **$\text{LOF} > 1.5$ (Signifikantní anomálie)**
            Hustota bodu je výrazně nižší než hustota jeho sousedů. Bod je lokální outlier!
            """
        )

# ==============================================================================
# TAB 4: SROVNÁNÍ METODIK
# ==============================================================================
with tab4:
    st.subheader("4. Srovnání detekčních paradigmat v moderní datové vědě")
    st.markdown(
        r"""
        V machine learningu neexistuje univerzální algoritmus pro všechny typy dat. 
        Přednáška uvádí srovnání základních přístupů:
        """
    )

    st.markdown(
        r"""
        | Algoritmus | Typ učení | Výpočetní složitost | Schopnost řešit proměnlivou hustotu | Doporučené použití |
        | :--- | :---: | :---: | :---: | :--- |
        | **Statistické Z-skóre** | Parametrické | $\mathcal{O}(N)$ | ❌ Zcela selhává | 1D unimodální Gaussovská data |
        | **Tukeyho IQR Boxplot** | Neparametrické | $\mathcal{O}(N \log N)$ | ❌ Zcela selhává | Rychlý screening 1D tabulkových dat |
        | **DBSCAN (label -1)** | Neřízené shlukování | $\mathcal{O}(N \log N)$ | ❌ Globální $\varepsilon$ selže | Nekonvexní data se stejnou hustotou |
        | **Isolation Forest (iForest)** | Stromový ansámbl | $\mathcal{O}(T \cdot \psi \log \psi)$ | 🟡 Průměrná | Vysokodimenzionální data ($D > 20$), rychlý streaming |
        | **Local Outlier Factor (LOF)** | Hustotní k-NN | $\mathcal{O}(N^2)$ (nebo $\mathcal{O}(N \log N)$ k-d tree) | ✅ **Exceluje (Unikátní přednost)** | Prostorová data, lokální shluky s různou hustotou ($D \le 15$) |
        """
    )

    st.markdown("#### 🌲 Princip Isolation Forest (slajd 14):")
    st.markdown(
        r"""
        * **Myšlenka:** Anomálie jsou vzácné a strukturálně odlišné. Pokud prostor náhodně řežeme kolmými nadrovinami (řezy podél náhodně vybraného příznaku a náhodného prahu), anomálie je izolována do samostatného listu stromu mnohem rychleji (menší hloubka $h(x)$) než běžné body v hustých oblastech.
        * **Výhoda:** Extrémní rychlost, lineární škálovatelnost $\mathcal{O}(N)$, skvělé výsledky ve vysokých dimenzích.
        """
    )

# ==============================================================================
# TAB 5: METODICKÉ GUARDRAILS
# ==============================================================================
with tab5:
    st.subheader("5. Závazné metodické Guardrails pro detekci anomálií (2026)")
    st.markdown(
        r"""
        Při nasazování algoritmů typu LOF a Isolation Forest do produkčních analytických procesů dodržujte následující pravidla:
        """
    )

    c_g1, c_g2 = st.columns(2)

    with c_g1:
        st.markdown(
            r"""
            ##### 1. Striktní standardizace před LOF
            * **Pravidlo:** Všechny spojité proměnné musí projít `StandardScaler` nebo `RobustScaler`.
            * **Důvod:** Eukleidovská metrika v neškálovaném prostoru způsobí, že příznak s velkým rozptylem zcela zdominuje výpočet $k$-vzdáleností.
            
            ##### 2. Pravidlo pro volbu `n_neighbors` v LOF
            * **Pravidlo:** Nastavujte $15 \le \text{n\_neighbors} \le 35$.
            * **Důvod:** Při $\text{n\_neighbors} < 10$ je model náchylný k falešným poplachům kvůli náhodným mikroseskupením šumu. Při $\text{n\_neighbors} > N_{\text{min\_cluster}}$ ztratí model schopnost detekovat lokální anomálie u malých shluků.
            """
        )

    with c_g2:
        st.markdown(
            r"""
            ##### 3. Kalibrace parametru `contamination`
            * **Pravidlo:** Parametr `contamination` nikdy nenechávejte na náhodě. V byznysové praxi musí odpovídat kapacitě vyšetřovacího týmu (např. fraud analytiků).
            * **Dopad:** Pokud nastavíte kontaminaci na $10\,\%$, model označí každou desátou transakci za podvod, což povede k masivnímu zahlcení falešnými poplachy (False Positives).
            
            ##### 4. Outlier Detection vs. Novelty Detection
            * **Pravidlo:** Pro čištění historického datasetu použijte výchozí `novelty=False`. Pro skórování nových příchozích transakcí v reálném čase **musíte zapnout `novelty=True`**, abyste mohli volat metody `predict()` a `decision_function()`.
            """
        )

    st.success("✅ Dodržení těchto guardrails zajišťuje matematickou správnost a stabilitu detekce v produkčních podmínkách.")
