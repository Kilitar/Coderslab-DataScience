"""
Závěr kurzu: Celkové shrnutí Machine Learning (Prework až Den 5)
===============================================================
Interaktivní syntéza celého kurzu:
1. Časová osa & mapa dovedností (Prework, Regrese, Klasifikace, Ensembly & Sítě, NLP, Neřízené učení)
2. Interaktivní selektor optimálního modelu podle byznysového zadání
3. Srovnávací přehled metrik, evaluace a pasti v praxi
4. Velký závěrečný test znalostí (Grand Quiz z celého kurzu)
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Celkové shrnutí ML kurzu", page_icon="🎓", layout="wide")

st.title("🎓 Celkové shrnutí kurzu Machine Learning")
st.caption(
    "Kompletní retrospektiva všech 5 dnů a Preworku: Od explorace dat, přes klasické modely "
    "a hluboké neuronové sítě, až po zpracování přirozeného jazyka a neřízené učení."
)

tab_map, tab_selector, tab_metrics, tab_quiz = st.tabs([
    "🗺️ Cesta kurzem (0. až 5. den)",
    "🧭 Interaktivní selektor modelu",
    "⚖️ Metriky & Pasti v praxi",
    "🏆 Závěrečný Grand Quiz"
])

# -----------------------------------------------------------------------------
# TAB 1: Cesta kurzem
# -----------------------------------------------------------------------------
with tab_map:
    st.subheader("🗺️ Mapa zvládnutých modulů a kompetencí")
    st.markdown(
        """
        Během kurzu jsme vybudovali ucelený stack dovedností moderního datového vědce.
        Každý modul logicky navazoval na předchozí a rozšiřoval schopnost řešit komplexnější problémy.
        """
    )

    modules_data = [
        {
            "Den": "0. Prework",
            "Téma": "Základy Pythonu & Příprava dat",
            "Klíčové algoritmy & nástroje": "NumPy, Pandas, Matplotlib, StandardScaler, train_test_split",
            "Výstup": "Vyčištěná a předzpracovaná data bez úniku (Data Leakage)"
        },
        {
            "Den": "1. Regrese",
            "Téma": "Predikce spojitých veličin",
            "Klíčové algoritmy & nástroje": "Lineární regrese, Polynomiální regrese, Ridge (L2), Lasso (L1), ElasticNet",
            "Výstup": "Predikce cen, tržeb; evaluace přes MSE, RMSE, MAE a R²"
        },
        {
            "Den": "2. Klasifikace",
            "Téma": "Kategoriální rozhodování & Pravděpodobnosti",
            "Klíčové algoritmy & nástroje": "Logistická regrese, Decision Trees, k-NN, Support Vector Machines (SVM)",
            "Výstup": "Detekce churnu, diagnózy; evaluace přes Precision, Recall, F1 a ROC-AUC"
        },
        {
            "Den": "3. Pokročilý ML",
            "Téma": "Ansámbly, Hluboké sítě & Tuning",
            "Klíčové algoritmy & nástroje": "Random Forest (Bagging), XGBoost (Boosting), MLP neuronové sítě, GridSearchCV",
            "Výstup": "Vítězné modely pro tabulární data, modelování nelineárních závislostí"
        },
        {
            "Den": "4. NLP",
            "Téma": "Zpracování přirozeného jazyka",
            "Klíčové algoritmy & nástroje": "Tokenizace, Lemmatizace, BoW, TF-IDF, Word2Vec (CBOW/Skip-gram), BERT Transformers",
            "Výstup": "Klasifikace textu, sentiment analýza, sémantické embeddingy slov"
        },
        {
            "Den": "5. Neřízené učení",
            "Téma": "Struktura dat bez cílové proměnné",
            "Klíčové algoritmy & nástroje": "PCA, t-SNE, K-Means, DBSCAN, Isolation Forest, Local Outlier Factor (LOF)",
            "Výstup": "Redukce dimenzionality, zákaznická segmentace, detekce anomálií"
        }
    ]
    df_modules = pd.DataFrame(modules_data)
    st.dataframe(df_modules, hide_index=True)

    st.markdown("---")
    st.subheader("📊 Rozložení dovedností v typickém ML projektu")
    c1, c2 = st.columns([1, 1])
    with c1:
        skills = {
            "Fáze projektu": [
                "1. Porozumění byznysu & Definice problému",
                "2. Sběr, čištění & Explorace dat (EDA)",
                "3. Feature Engineering & Preprocessing",
                "4. Výběr architektury & Trénování modelu",
                "5. Evaluace, Diagnostika chyb & Tuning",
                "6. Deployment, MLOps & Monitoring"
            ],
            "Časová náročnost (%)": [15, 30, 25, 10, 10, 10]
        }
        fig_pie = px.pie(
            skills,
            names="Fáze projektu",
            values="Časová náročnost (%)",
            title="Kde datový vědec tráví reálný čas v praxi?",
            color_discrete_sequence=px.colors.qualitative.Prism,
            hole=0.4
        )
        st.plotly_chart(fig_pie)
    with c2:
        st.markdown(
            """
            > **Zlaté pravidlo praxe:**
            > 
            > Většina času (až 70–80 %) nespočívá v ladění neuronových sítí nebo volbě exotického algoritmu,
            > nýbrž v **důkladné přípravě dat, pochopení domény a konstrukci kvalitních příznaků (Feature Engineering)**.
            > 
            > Špatná data nenapraví ani nejhlubší model (*Garbage In, Garbage Out*), zatímco čistá data
            > a smysluplné příznaky umožní zazářit i jednoduché lineární či logistické regresi.
            """
        )

# -----------------------------------------------------------------------------
# TAB 2: Interaktivní selektor modelu
# -----------------------------------------------------------------------------
with tab_selector:
    st.subheader("🧭 Průvodce volbou algoritmu pro nový projekt")
    st.markdown("Zadejte charakteristiku vaší úlohy a získejte okamžité doporučení optimálního modelu:")

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        task_type = st.selectbox(
            "1. Jaký je cíl projektu?",
            [
                "Predikce spojitého čísla (Regrese)",
                "Predikce diskrétní třídy (Klasifikace)",
                "Zpracování textu / NLP",
                "Nalezení přirozených skupin (Segmentace)",
                "Hledání podezřelých vzorců (Detekce anomálií)",
                "Vizuální redukce dimenzí (2D/3D zobrazení)"
            ]
        )
    with col_s2:
        data_size = st.selectbox(
            "2. Velikost datasetu (počet řádků):",
            [
                "Malý (< 1 000 řádků)",
                "Střední (1 000 až 100 000 řádků)",
                "Velký (> 100 000 řádků)"
            ]
        )
    with col_s3:
        interpretability = st.selectbox(
            "3. Požadavek na interpretovatelnost (White-box vs. Black-box):",
            [
                "Kritická (Musíme klientovi/auditu vysvětlit přesný vliv každého koeficientu)",
                "Střední (Stačí nám Feature Importance a SHAP hodnoty)",
                "Nízká (Záleží výhradně na maximální predikční přesnosti)"
            ]
        )

    # Logika doporučení
    st.markdown("---")
    st.subheader("🎯 Doporučené řešení a architektura")

    if "Regrese" in task_type:
        if "Kritická" in interpretability:
            rec_model = "Lineární regrese s Ridge / Lasso regularizací"
            desc = "Umožňuje přímou interpretaci koeficientů beta. Lasso navíc automaticky vynuluje irelevantní příznaky."
            alt = "Rozhodovací strom (omezený na max_depth <= 4) pro grafické rozhodovací podmínky."
        elif "Velký" in data_size or "Nízká" in interpretability:
            rec_model = "XGBoost Regressor / LightGBM nebo Hluboká neuronová síť (MLP)"
            desc = "Extrémní predikční schopnost pro komplexní nelineární vztahy a interakce vyšších řádů."
            alt = "Random Forest Regressor jako robustní baseline bez nutnosti ladit learning rate."
        else:
            rec_model = "Random Forest Regressor nebo Gradient Boosting"
            desc = "Vyvážený poměr mezi přesností, stabilitou vůči odlehlým hodnotám a dostupností Feature Importance."
            alt = "Polynomiální regrese s Ridge regularizací."

    elif "Klasifikace" in task_type:
        if "Kritická" in interpretability:
            rec_model = "Logistická regrese nebo Rozhodovací strom"
            desc = "Logistická regrese přímo vrací odds ratio (poměry šancí). Strom poskytuje srozumitelná pravidla typu IF-THEN."
            alt = "Lineární Support Vector Machine (LinearSVC)."
        elif "Velký" in data_size or "Nízká" in interpretability:
            rec_model = "XGBoost Classifier nebo MLP neuronová síť"
            desc = "De-facto standard pro tabulární soutěže na Kaggle. Skvěle zvládá chybějící data i nelineární hranice."
            alt = "Random Forest Classifier."
        else:
            rec_model = "Random Forest Classifier nebo SVM s RBF jádrem"
            desc = "Random Forest je vysoce odolný vůči přetrénování a nevyžaduje škálování. SVM s RBF jádrem exceluje u menších dat se složitou geometrií."
            alt = "k-NN (s povinným StandardScalerem)."

    elif "NLP" in task_type:
        if "Kritická" in interpretability or "Malý" in data_size:
            rec_model = "TF-IDF + Logistická regrese / Naive Bayes"
            desc = "Rychlé, lehké a překvapivě silné řešení pro klasifikaci témat či spamu s minimálními výpočetními nároky."
            alt = "Bag of Words (CountVectorizer) + LinearSVC."
        else:
            rec_model = "Předtrénovaný Transformer (BERT / RoBERTa / Small LLM)"
            desc = "Stav současné techniky (SOTA). Hluboké sémantické porozumění kontextu a jemným nuancím jazyka."
            alt = "Word2Vec embeddingy + BiLSTM / CNN pro text."

    elif "Segmentace" in task_type:
        rec_model = "K-Means (pro sférické shluky) nebo DBSCAN (pro libovolné tvary a šum)"
        desc = "Pokud máte představu o počtu segmentů a data tvoří kompaktní shluky, použijte K-Means (s Elbow metodou). Pokud jsou shluky protáhlé a data obsahují šum, použijte DBSCAN."
        alt = "Hierarchické shlukování (Wardova metoda) pro tvorbu dendrogramu."

    elif "Detekce anomálií" in task_type:
        rec_model = "Isolation Forest (globální anomálie) nebo LOF (lokální hustotní anomálie)"
        desc = "Isolation Forest je extrémně rychlý a lineárně škálovatelný algoritmus pro tabulární data. LOF použijte, pokud se hustota dat v různých regionech liší."
        alt = "One-Class SVM nebo Autoencoder (pro obrazová data)."

    else:
        rec_model = "PCA (lineární redukce) nebo t-SNE / UMAP (nelineární vizualizace)"
        desc = "PCA použijte pro zachování globálního rozptylu a rychlou kompresi. t-SNE zvolte pro poutavé 2D mapy s oddělenými shluky."
        alt = "LDA (pokud máme známé třídy a chceme maximalizovat jejich separaci)."

    res_c1, res_c2 = st.columns([2, 1])
    with res_c1:
        st.success(f"🏆 **Hlavní doporučení:** {rec_model}")
        st.write(f"**Proč právě tento model:** {desc}")
        st.info(f"💡 **Alternativní varianta:** {alt}")
    with res_c2:
        st.markdown(
            """
            <div style="background-color: #f1f5f9; padding: 16px; border-radius: 8px; border-left: 4px solid #6366f1;">
                <b>Doporučený postup ověření:</b>
                <ol style="margin: 0; padding-left: 18px; font-size: 0.9rem;">
                    <li>Vytvořit jednoduchý baseline model.</li>
                    <li>Ověřit metriky na K-Fold křížové validaci.</li>
                    <li>Zkusit komplexnější ansámbl (XGBoost/RF).</li>
                    <li>Porovnat přínos vs. latenci v produkci.</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True
        )

# -----------------------------------------------------------------------------
# TAB 3: Metriky & Pasti v praxi
# -----------------------------------------------------------------------------
with tab_metrics:
    st.subheader("⚖️ Výběr správné metriky & Nejčastější pasti praxe")

    st.markdown("#### 1. Volba metriky podle byznysového cíle")
    metrics_summary = [
        {
            "Typ problému": "Regrese",
            "Metrika": "RMSE (Root Mean Squared Error)",
            "Kdy zvolit": "Když je velká chyba neúměrně nebezpečnější než malá chyba (kvadratická penalizace).",
            "Příklad z praxe": "Predikce zpoždění letů, predikce spotřeby elektrické sítě."
        },
        {
            "Typ problému": "Regrese",
            "Metrika": "MAE (Mean Absolute Error)",
            "Kdy zvolit": "Když data obsahují odlehlé hodnoty a chceme lineární, robustní interpretaci průměrné odchylky.",
            "Příklad z praxe": "Ocenění nemovitostí, předpověď doby doručení zásilky."
        },
        {
            "Typ problému": "Klasifikace",
            "Metrika": "Precision (Přesnost)",
            "Kdy zvolit": "Když je Falešně pozitivní výsledek (FP) velmi drahý nebo poškozující.",
            "Příklad z praxe": "Klasifikace spamu (nechceme smazat důležitou fakturu), doporučování investic."
        },
        {
            "Typ problému": "Klasifikace",
            "Metrika": "Recall (Senzitivita)",
            "Kdy zvolit": "Když je Falešně negativní výsledek (FN) fatální a musíme zachytit maximum případů.",
            "Příklad z praxe": "Diagnostika závažných nemocí, detekce podvodných plateb (Fraud Detection)."
        },
        {
            "Typ problému": "Klasifikace",
            "Metrika": "ROC-AUC / PR-AUC",
            "Kdy zvolit": "Hodnocení schopnosti modelu řadit pravděpodobnosti nezávisle na zvoleném prahu. U nevyvážených dat PR-AUC.",
            "Příklad z praxe": "Scoring v bankovnictví (credit scoring), konverzní modely v marketingu."
        }
    ]
    st.dataframe(pd.DataFrame(metrics_summary), hide_index=True)

    st.markdown("---")
    st.markdown("#### 2. Tři nejčastější chyby (Gotchas) začínajících data scientistů")

    g1, g2, g3 = st.columns(3)
    with g1:
        st.error("🚨 1. Data Leakage (Únik dat)")
        st.markdown(
            """
            - **Chyba:** Škálování (`fit_transform`) nebo výběr příznaků na celém datasetu před rozdělením na train/test.
            - **Důsledek:** Nerealisticky optimistické metriky při validaci, okamžitý kolaps přesnosti v produkci.
            - **Řešení:** Vždy používat `sklearn.pipeline.Pipeline`.
            """
        )
    with g2:
        st.warning("⚠️ 2. Slepá důvěra v Accuracy")
        st.markdown(
            """
            - **Chyba:** Měření úspěšnosti pomocí Accuracy u datasetu, kde je 99 % vzorků negativních.
            - **Důsledek:** Triviální model, který vždy predikuje 0, má 99% accuracy, ale nezachytí ani jediný podvod.
            - **Řešení:** Používat Confusion Matrix, Precision, Recall a PR-AUC.
            """
        )
    with g3:
        st.info("💡 3. Chybějící škálování")
        st.markdown(
            """
            - **Chyba:** Ponechání veličin v původních jednotkách (např. plat v desítkách tisíc vs. věk v desítkách let) pro k-NN, SVM, PCA či K-Means.
            - **Důsledek:** Veličina s velkým rozsahem zcela dominuje výpočtům eukleidovské vzdálenosti.
            - **Řešení:** Aplikovat `StandardScaler`.
            """
        )

# -----------------------------------------------------------------------------
# TAB 4: Závěrečný Grand Quiz
# -----------------------------------------------------------------------------
with tab_quiz:
    st.subheader("🏆 Velký závěrečný test znalostí z celého kurzu")
    st.markdown(
        "Ověřte si své porozumění stěžejním konceptům z celého kurzu. "
        "Test obsahuje 7 komplexních otázek napříč všemi 5 moduly."
    )

    questions = [
        {
            "id": "q1",
            "question": "1. Jaký je hlavní rozdíl mezi L1 (Lasso) a L2 (Ridge) regularizací v lineární regresi?",
            "options": [
                "L1 penalizuje čtverce vah a zachovává všechny proměnné, zatímco L2 je vynuluje.",
                "L1 penalizuje absolutní hodnotu vah a může některé koeficienty nastavit přesně na nulu (Feature Selection), zatímco L2 váhy pouze zmenšuje.",
                "L1 se používá pouze pro klasifikaci, zatímco L2 se smí použít jen u regrese.",
                "Mezi L1 a L2 není v matematickém výsledku žádný funkční rozdíl."
            ],
            "correct": 1,
            "explanation": "Lasso (L1) penalizace vede díky diamantovému tvaru omezující podmínky k nulování koeficientů, čímž provádí automatickou selekci příznaků. Ridge (L2) koeficienty přibližuje nule, ale žádný úplně nevynuluje."
        },
        {
            "id": "q2",
            "question": "2. Proč je u nevyváženého datasetu (např. 98 % zdravých pacientů a 2 % nemocných) nevhodné hodnotit klasifikátor pomocí Accuracy?",
            "options": [
                "Protože hodnota Accuracy nemůže u nevyvážených dat překročit 50 %.",
                "Protože triviální model předpovídající všem pacientům 'zdravý' dosáhne 98% Accuracy, přestože neodhalí ani jediného nemocného.",
                "Protože Scikit-learn zakazuje počítat Accuracy u dat s méně než 50 % minoritní třídy.",
                "Protože Accuracy je definována pouze pro regresní úlohy."
            ],
            "correct": 1,
            "explanation": "V nevyvážených datech většina snadno maskuje neschopnost modelu predikovat minoritní třídu. V medicíně je klíčový Recall (Senzitivita), abychom nepřehlédli skutečně nemocné (minimalizace False Negatives)."
        },
        {
            "id": "q3",
            "question": "3. Jaký je principiální rozdíl mezi Baggingem (např. Random Forest) a Boostingem (např. XGBoost)?",
            "options": [
                "Bagging trénuje modely nezávisle a paralelně pro snížení rozptylu (Variance), zatímco Boosting trénuje modely sekvenčně, kde každý další opravuje chyby předchozích, čímž snižuje vychýlení (Bias).",
                "Bagging používá pouze lineární modely, zatímco Boosting funguje pouze s neuronovými sítěmi.",
                "Boosting je určen pouze pro neřízené učení, kdežto Bagging pro řízené učení.",
                "Random Forest vyžaduje normalizaci dat přes StandardScaler, zatímco XGBoost nikoliv."
            ],
            "correct": 0,
            "explanation": "Bagging (Bootstrap Aggregating) paralelně zprůměruje mnoho plně vyrostlých stromů a redukuje varianci. Boosting buduje sekvenci mělkých stromů, z nichž každý cílí na reziduální chyby předešlých."
        },
        {
            "id": "q4",
            "question": "4. Co reprezentuje hodnota TF-IDF slova v textovém dokumentu?",
            "options": [
                "Gramatickou správnost a délku daného slova.",
                "Důležitost slova: je vysoká, pokud se slovo často vyskytuje v tomto dokumentu (vysoké TF), ale vzácně v celém korpusu dokumentů (vysoké IDF).",
                "Celkový počet znaků ve slově dělený počtem odstavců.",
                "Pravděpodobnost, že dané slovo je stop-slovem."
            ],
            "correct": 1,
            "explanation": "TF-IDF kombinuje četnost v konkrétním textu s jeho informační vzácností napříč všemi dokumenty. Běžná slova jako 'a', 'v', 'je' mají velmi nízké IDF a jsou potlačena."
        },
        {
            "id": "q5",
            "question": "5. Které z následujících tvrzení o algoritmu K-Means je PRAVDIVÉ?",
            "options": [
                "K-Means dokáže spolehlivě odhalit shluky libovolných nepravidelných tvarů (např. soustředné kružnice nebo půlměsíce).",
                "K-Means nevyžaduje zadání počtu shluků K a určí si ho sám automaticky.",
                "K-Means minimalizuje vnitroshlukovou sumu čtverců vzdáleností (Inercii) a předpokládá sférické shluky podobné velikosti.",
                "K-Means automaticky detekuje odlehlé hodnoty a přiřadí jim shluk -1."
            ],
            "correct": 2,
            "explanation": "K-Means předpokládá konvexní (kulovité) shluky a vyžaduje předem definované K. Nepravidelné tvary a automatickou detekci šumu (-1) řeší DBSCAN."
        },
        {
            "id": "q6",
            "question": "6. Proč je v algoritmu PCA (Principal Component Analysis) povinné předem provést standardizaci dat (StandardScaler)?",
            "options": [
                "Protože PCA funguje pouze s celými čísly od 0 do 255.",
                "Protože PCA hledá směry maximálního rozptylu; veličina s velkým rozsahem (např. příjem v milionech) by rozptylem zcela zastínila ostatní smysluplné proměnné.",
                "Standardizace není potřeba, PCA je vůči měřítku proměnných zcela invariantní.",
                "Protože Scikit-learn bez standardizace vyvolá běhovou chybu NotImplementedError."
            ],
            "correct": 1,
            "explanation": "PCA maximalizuje rozptyl $Var(X)$. Proměnná s velkými čísly má uměle obrovský rozptyl, takže by se první hlavní komponenta stala téměř identickou s touto jedinou proměnnou bez ohledu na reálnou strukturu dat."
        },
        {
            "id": "q7",
            "question": "7. Jak funguje algoritmus Isolation Forest při detekci anomálií?",
            "options": [
                "Počítá hustotu pravděpodobnosti pomocí Gaussovských křivek.",
                "Trénuje klasifikátor s učitelem na označených datech z minulosti.",
                "Náhodně dělí prostor pomocí stromů; anomálie jsou osamocené body, které se izolují na velmi malém počtu náhodných řezů (mají krátkou hloubku ve stromu).",
                "Počítá vzdálenost každého bodu k těžišti všech dat."
            ],
            "correct": 2,
            "explanation": "Isolation Forest vychází z principu, že anomálie jsou 'málo početné a odlišné'. Proto je náhodné binární dělení oddělí od zbytku dat mnohem rychleji (kratší průměrná hloubka $h(x)$)."
        }
    ]

    total_score = 0
    all_answered = True

    for item in questions:
        q_id = item["id"]
        user_choice = st.radio(
            item["question"],
            options=item["options"],
            index=None,
            key=f"grand_{q_id}"
        )
        if user_choice is None:
            all_answered = False
        else:
            chosen_idx = item["options"].index(user_choice)
            if chosen_idx == item["correct"]:
                st.success(f"✅ Správně! {item['explanation']}")
                total_score += 1
            else:
                st.error(f"❌ Nesprávně. Správná odpověď byla: **{item['options'][item['correct']]}**")
                st.info(f"Vysvětlení: {item['explanation']}")
        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("---")
    if all_answered:
        score_percent = (total_score / len(questions)) * 100
        st.metric("Celkové skóre z Grand Quizu", f"{total_score} z {len(questions)}", f"{score_percent:.1f} %")
        if total_score == len(questions):
            st.balloons()
            st.success("🎉 Excelentní výkon! Zvládli jste celý závěrečný test bez jediné chyby. Jste plně připraveni na reálné projekty!")
        elif total_score >= 5:
            st.success("👏 Skvělý výsledek! Máte pevné základy ve všech klíčových disciplínách machine learningu.")
        else:
            st.warning("Doporučujeme projít si sekce Cesta kurzem a Metriky v praxi pro upevnění konceptů.")
    else:
        st.info("Zodpovězte prosím všech 7 otázek pro vyhodnocení celkového výsledku.")
