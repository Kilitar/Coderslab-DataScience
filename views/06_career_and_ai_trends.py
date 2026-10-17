"""
Budoucí vývoj: Trendy v AI, Generativní AI & Zdroje
===================================================
Interaktivní průvodce dalším rozvojem data scientisty:
1. Generativní AI & LLM ekosystém (ChatGPT, Gemini, LLaMA, Difúzní modely)
2. Machine Learning v kyberbezpečnosti (Detekce anomálií v provozu, phishing, malware)
3. Knihovna doporučených zdrojů (Knihy, Kaggle, YouTube, odborná fóra)
4. Kariérní roadmapa & Dovednostní strom data scientisty
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Trendy AI & Kam dál", page_icon="🔮", layout="wide")

st.title("🔮 Budoucí vývoj: Trendy v AI, Generativní AI & Zdroje")
st.caption(
    "Rozcestník pro další profesní růst: Kde čerpat znalosti po kurzu, jak se orientovat v explozi "
    "Generativní AI a LLMs, a jak machine learning proměňuje klíčová odvětví jako kyberbezpečnost."
)

tab_genai, tab_sec, tab_resources, tab_career = st.tabs([
    "🤖 Generativní AI & LLMs",
    "🛡️ ML v kyberbezpečnosti",
    "📚 Doporučené knihy & Zdroje",
    "🗺️ Kariérní roadmapa Data Scientisty"
])

# -----------------------------------------------------------------------------
# TAB 1: Generativní AI & LLMs
# -----------------------------------------------------------------------------
with tab_genai:
    st.subheader("🤖 Revoluce Generativní AI (GenAI) a velkých jazykových modelů")
    st.markdown(
        """
        Zatímco tradiční machine learning (který jsme zkoumali v kurzu) se zaměřuje na **diskriminativní úlohy** 
        (klasifikace existujícího stavu, regrese čísla, hledání anomálií), **Generativní AI** se učí distribuci 
        dat tak, aby dokázala generovat **zcela nový obsah** (text, kód, obrázky, hudbu).
        """
    )

    c_g1, c_g2, c_g3 = st.columns(3)
    with c_g1:
        with st.container(border=True):
            st.markdown("### 💬 Velké jazykové modely (LLMs)")
            st.markdown(
                """
                Modely s desítkami až stovkami miliard parametrů trénované na předpověď dalšího tokenu.

                - **OpenAI (GPT-4o, ChatGPT):** Univerzální asistent s podporou kódu a multimodality.
                - **Google (Gemini 1.5 Pro):** Extrémní kontextové okno (až 2M tokenů) a multimodální chápání.
                - **Meta (LLaMA 3):** Špičkové open-weight modely pro privátní on-premise nasazení.
                """
            )
    with c_g2:
        with st.container(border=True):
            st.markdown("### 🎨 Difúzní obrazové modely")
            st.markdown(
                """
                Generování fotorealistických děl a grafiky z textového popisu (Text-to-Image).

                - **Midjourney:** Špičková estetika, grafický design, filmový vzhled.
                - **DALL-E 3:** Hluboká integrace s ChatGPT a precizní dodržování složitých zadání.
                - **Stable Diffusion:** Open-source standard s plnou kontrolou nad váhami a LoRA adaptéry.
                """
            )
    with c_g3:
        with st.container(border=True):
            st.markdown("### 💻 Asistované programování")
            st.markdown(
                """
                Párové programování, automatizace testů a generování datových pipeline.

                - **Cursor / GitHub Copilot:** Kontextové doplňování kódu napříč celým repozitářem.
                - **Antigravity / Gemini CLI:** Agentické řešení komplexních úloh přímo v IDE a terminálu.
                - **Claude 3.5 Sonnet:** Bezkonkurenční schopnost v refaktoringu a logickém ladění.
                """
            )

    st.markdown("---")
    st.subheader("⚠️ Zásadní limity a výzvy LLM pro datové vědce")
    l1, l2, l3, l4 = st.columns(4)
    with l1:
        st.warning("**1. Halucinace**\n\nModel dokáže s absolutní gramatickou jistotou tvrdit naprostý faktický nesmysl.")
    with l2:
        st.warning("**2. Nedeterminismus**\n\nPři teplotě > 0 vrátí model na totožný prompt pokaždé jinou odpověď.")
    with l3:
        st.warning("**3. Časový cutoff**\n\nModel nezná události po datu trénování; vyžaduje RAG architekturu pro přístup k čerstvým datům.")
    with l4:
        st.warning("**4. Bezpečnost & Injekce**\n\nRiziko útoků Prompt Injection, úniku citlivých dat a generování škodlivého obsahu.")

# -----------------------------------------------------------------------------
# TAB 2: ML v kyberbezpečnosti
# -----------------------------------------------------------------------------
with tab_sec:
    st.subheader("🛡️ Machine Learning v kybernetické bezpečnosti")
    st.markdown(
        """
        Dle mezinárodních analýz roste trh AI v kyberbezpečnosti více než čtyřnásobně.
        Tradiční signatury (hash známého viru) selhávají proti polymorfnímu malwaru a zero-day útokům.
        ML přináší schopnost detekovat **abnormální vzorce v reálném čase**.
        """
    )

    sec_cases = [
        {
            "Oblast ochrany": "Síťová detekce anomálií (NDR)",
            "Jak ML pomáhá": "Algoritmy (Isolation Forest, DBSCAN) monitorují síťový provoz a odhalí komunikaci s neznámými IP adresami, neobvyklé porty nebo noční úniky dat (Exfiltration).",
            "Použitý ML přístup": "Neřízené učení (Unsupervised) & detekce odlehlých hodnot."
        },
        {
            "Oblast ochrany": "Detekce neznámého malwaru",
            "Jak ML pomáhá": "Klasifikace podezřelých souborů ne podle hashe, ale podle dynamického chování (volání API, zápisy do registrů, entropie souboru).",
            "Použitý ML přístup": "XGBoost, Random Forest na extrahovaných příznacích spustitelných souborů."
        },
        {
            "Oblast ochrany": "Antiphishing & Sociální inženýrství",
            "Jak ML pomáhá": "NLP modely analyzují text emailu, nátlak na emoce (urgentní výzvy), stylizaci a podvržené URL domény.",
            "Použitý ML přístup": "TF-IDF + Logistická regrese nebo jemně laděný BERT pro analýzu sentimentu a nátlaku."
        },
        {
            "Oblast ochrany": "Analýza auditních logů (SIEM/SOAR)",
            "Jak ML pomáhá": "Zpracování miliard řádků bezpečnostních záznamů ze serverů, korelace událostí a odhalení postupu útočníka v síti (Lateral Movement).",
            "Použitý ML přístup": "Sekvenční modely, grafové neuronové sítě a k-NN nad logy."
        }
    ]
    st.dataframe(pd.DataFrame(sec_cases), hide_index=True)

    st.markdown("---")
    st.subheader("⚖️ Klíčové dilema v SOC: Falešné poplachy vs. Prohlédnutý útok")
    c_f1, c_f2 = st.columns(2)
    with c_f1:
        st.error(
            "🔴 **False Negative (FN) = Přehlédnutý útok**\n\n"
            "Útočník pronikl do infrastruktury, ale model incident označil za běžný provoz. "
            "Následek: Šifrování ransomwarem, krádež databáze zákazníků, obrovské škody."
        )
    with c_f2:
        st.warning(
            "🟡 **False Positive (FP) = Falešný poplach**\n\n"
            "Běžný zaměstnanec si stáhl velký soubor a model spustil alarm. "
            "Následek: **Alert Fatigue (únava operátorů)**. Bezpečnostní analytici denně prověřují stovky falešných poplachů a mohou přehlédnout skutečný incident."
        )

# -----------------------------------------------------------------------------
# TAB 3: Doporučené knihy & Zdroje
# -----------------------------------------------------------------------------
with tab_resources:
    st.subheader("📚 Osvědčené zdroje pro další studium")

    col_b, col_y = st.columns([1, 1])

    with col_b:
        st.markdown("#### 📖 Doporučená odborná literatura")
        books = [
            {
                "Název & Autor": "Hands-On Machine Learning (Aurélien Géron)",
                "Pro koho": "Mírně pokročilí v Pythonu",
                "Obsah": "Naprostá klasika. Skvělý přechod od Scikit-learn ke Kerasu a TensorFlow."
            },
            {
                "Název & Autor": "AI and Machine Learning for Coders (Laurence Moroney)",
                "Pro koho": "Programátoři a softwaroví inženýři",
                "Obsah": "Praktická aplikace ML bez zbytečného akademického formalismu."
            },
            {
                "Název & Autor": "Advanced Machine Learning with Python (John Hearty)",
                "Pro koho": "Zájemci o specializovaná data",
                "Obsah": "Zpracování obrazu, zvuku, textu a složitějších tabulárních úloh."
            },
            {
                "Název & Autor": "Deep Learning (Goodfellow, Bengio, Courville)",
                "Pro koho": "Hluboké teoretické porozumění",
                "Obsah": "Matematické základy lineární algebry, optimalizace a hlubokých architektur."
            }
        ]
        st.dataframe(pd.DataFrame(books), hide_index=True)

        st.markdown("#### 🌐 Významné komunity & Fóra")
        st.markdown(
            """
            - **[Kaggle.com](https://www.kaggle.com):** Soutěže, veřejné datasety, inspirace z vítězných notebooků.
            - **[Cross Validated](https://stats.stackexchange.com):** Specializované fórum pro statistiku a teorii ML.
            - **[Reddit r/MachineLearning](https://www.reddit.com/r/MachineLearning):** Aktuální dění, nové publikace a diskuse.
            - **[Hugging Face](https://huggingface.co):** Ústřední repozitář pro předtrénované modely a datasety.
            """
        )

    with col_y:
        st.markdown("#### 📺 Špičkové YouTube kanály")
        channels = [
            {
                "Kanál": "StatQuest with Josh Starmer",
                "Zaměření": "Vizuální a hudební vysvětlení matematiky algoritmů (BAM!).",
                "Proč sledovat": "Nejlepší intuice pro PCA, SVM, Decision Trees i Transformers."
            },
            {
                "Kanál": "Andrej Karpathy",
                "Zaměření": "Hluboké neuronové sítě a LLM od základů (Zero to Hero).",
                "Proč sledovat": "Naprogramování backpropagation a GPT v čistém Pythonu bez knihoven."
            },
            {
                "Kanál": "Sentdex (Harrison Kinsley)",
                "Zaměření": "Praktické projekty v Pythonu (finance, robotika, ML).",
                "Proč sledovat": "Obrovská knihovna tutoriálů s důrazem na reálný kód."
            },
            {
                "Kanál": "3Blue1Brown (Grant Sanderson)",
                "Zaměření": "Geniální animace lineární algebry a neuronových sítí.",
                "Proč sledovat": "Pochopení maticových transformací a gradientního sestupu."
            }
        ]
        st.dataframe(pd.DataFrame(channels), hide_index=True)

# -----------------------------------------------------------------------------
# TAB 4: Kariérní roadmapa
# -----------------------------------------------------------------------------
with tab_career:
    st.subheader("🗺️ Kariérní roadmapa a rozvoj dovedností Data Scientisty")
    st.markdown(
        """
        Kam můžete po absolvování kurzu směřovat? Data Science nabízí několik specializací 
        podle toho, zda preferujete byznys, matematiku, nebo softwarové inženýrství:
        """
    )

    c_r1, c_r2, c_r3 = st.columns(3)
    with c_r1:
        with st.container(border=True):
            st.markdown("### 📊 Data Scientist / Analytik")
            st.markdown(
                """
                *Most mezi byznysem a daty.*

                - A/B testování a statistické hypotézy
                - Feature Engineering a doménové znalosti
                - Prediktivní modelování (Churn, LTV, cenotvorba)
                - Prezentace a vizualizace pro management
                """
            )
    with c_r2:
        with st.container(border=True):
            st.markdown("### ⚙️ Machine Learning Engineer")
            st.markdown(
                """
                *Přenášení modelů do škálovatelné produkce.*

                - MLOps, Docker, Kubernetes, CI/CD pro ML
                - Optimalizace latence a kvantizace modelů (ONNX)
                - REST API, mikroservisy (FastAPI, Triton)
                - Monitoring driftu a automatizovaný retraining
                """
            )
    with c_r3:
        with st.container(border=True):
            st.markdown("### 🤖 AI / LLM Engineer")
            st.markdown(
                """
                *Tvorba moderních aplikací nad velkými modely.*

                - RAG architektury a vektorové databáze
                - Jemné doladění (Fine-Tuning, LoRA, QLoRA)
                - Autonomní AI agenti a orchestrace (LangChain, LlamaIndex)
                - Evaluace a bezpečnost LLM systémů
                """
            )

    st.markdown("---")
    st.success(
        "🎉 **Gratulujeme k úspěšnému završení teoretického i praktického bloku Machine Learning!** "
        "Máte v rukou robustní základ metod, modelů i inženýrských návyků. "
        "Ať se vám daří v závěrečném projektu i v reálných výzvách světa umělé inteligence!"
    )
