"""
Den 4: Ucelené shrnutí NLP & Závěrečný vědomostní kvíz
======================================================
Podklad: Resources/Day 4 PDF/Day_4_summary.pdf
Syntéza 4 klíčových etap NLP:
1. Předzpracování textových dat (Tokenizace, Normalizace, Stopwords, Lemmatizace).
2. Klasické frekvenční reprezentace (Bag of Words, TF-IDF).
3. Distribuovaná sémantická vnoření (Word2Vec, CBOW, Skip-gram).
4. Moderní Deep Learning (Transformer, Attention Is All You Need, BERT).
+ Celodenní srovnávací matice a interaktivní závěrečný vědomostní kvíz ("Time for a quiz").
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.title("🎓 Den 4: Ucelené shrnutí NLP & Závěrečný kvíz")
st.caption(
    "Závěrečná syntéza čtvrtého dne kurzu: Od čištění a tokenizace přes klasické Bag of Words a TF-IDF "
    "až po sémantická vnoření Word2Vec a revoluční obousměrné transformery BERT."
)

# Horní KPI karty
c1, c2, c3, c4 = st.columns(4)
c1.metric("1. Etapa: Čištění", "Předzpracování", delta="Tokenizace & Lemmatizace")
c2.metric("2. Etapa: Frekvence", "BoW & TF-IDF", delta="CountVectorizer & Tfidf")
c3.metric("3. Etapa: Sémantika", "Word2Vec", delta="Husté vektory (CBOW / Skip-gram)")
c4.metric("4. Etapa: Kontext", "Transformers & BERT", delta="Obousměrná Self-Attention")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "🏛️ 1. Čtyři etapy vývoje NLP",
    "📊 2. Srovnávací matice & Rozhodovací strom",
    "🌐 3. Ekosystém LLM a průmysl (Google, Meta, OpenAI)",
    "🧠 4. Závěrečný vědomostní kvíz (Quiz Time)"
])

# ==============================================================================
# TAB 1: ČTYŘI ETAPY VÝVOJE NLP
# ==============================================================================
with tab1:
    st.subheader("Evoluční cesta zpracování přirozeného jazyka")
    st.markdown(
        """
        Během Dne 4 jsme prošli kompletní technologickou evoluci reprezentace textu 
        od základního čištění až po moderní jazykové modely:
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(59, 130, 246, 0.4); border-radius: 10px; padding: 18px; margin-bottom: 15px;">
                <h4 style="color: #60a5fa; margin-top: 0;">1. Předzpracování textu (Preprocessing)</h4>
                <ul>
                    <li><strong>Tokenizace:</strong> Rozklad na slova a interpunkci (stavební jednotky).</li>
                    <li><strong>Normalizace:</strong> Převod na malá písmena, odstranění HTML značek a diakritiky.</li>
                    <li><strong>Stopwords:</strong> Odstranění častých, ale nevýznamových slov (<em>the, is, a, v, na, že</em>).</li>
                    <li><strong>Lemmatizace:</strong> Převod na slovníkový tvar (lemma) s ohledem na slovní druhy.</li>
                </ul>
            </div>
            <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(16, 185, 129, 0.4); border-radius: 10px; padding: 18px; margin-bottom: 15px;">
                <h4 style="color: #34d399; margin-top: 0;">2. Frekvenční vektorizace (BoW & TF-IDF)</h4>
                <ul>
                    <li><strong>Bag of Words:</strong> Četnosti slov v dokumentu bez ohledu na pořadí (<code>CountVectorizer</code>).</li>
                    <li><strong>TF-IDF:</strong> Zvýhodnění termínů specifických pro daný dokument vůči celému korpusu (<code>TfidfVectorizer</code>).</li>
                    <li><em>Limit:</em> Obrovské řídké matice (Sparse) a nulové chápání synonymie.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(245, 158, 11, 0.4); border-radius: 10px; padding: 18px; margin-bottom: 15px;">
                <h4 style="color: #fbbf24; margin-top: 0;">3. Distribuovaná vnoření (Word2Vec)</h4>
                <ul>
                    <li><strong>Nízká dimenze:</strong> Přechod na husté vektory (100–300 dimenzí).</li>
                    <li><strong>Vektorová algebra:</strong> Podobná slova leží blízko sebe (kosinová podobnost).</li>
                    <li><strong>CBOW vs Skip-gram:</strong> Předpovídání slova z kontextu vs kontextu ze slova.</li>
                    <li><em>Nástroj:</em> Knihovna <code>gensim</code>.</li>
                </ul>
            </div>
            <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(168, 85, 247, 0.4); border-radius: 10px; padding: 18px; margin-bottom: 15px;">
                <h4 style="color: #c084fc; margin-top: 0;">4. Transformery & BERT (Deep Learning)</h4>
                <ul>
                    <li><strong>Attention Is All You Need:</strong> 100% paralelizace na GPU, odstranění rekurencí.</li>
                    <li><strong>Obousměrný BERT:</strong> Slovo vnímá levý i pravý kontext současně ve všech vrstvách.</li>
                    <li><strong>Samo-učící se úlohy:</strong> Masked Language Model (MLM) a Next Sentence Prediction (NSP).</li>
                    <li><em>Nástroj:</em> Knihovna <code>transformers</code> (Hugging Face).</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("##### 📈 Přehled dosažených přesností v našich praktických cvičeních (IMDb dataset):")
    perf_data = pd.DataFrame([
        {"Metoda": "Bag of Words (10 000 feat.) + Logistická regrese", "Přesnost (Accuracy)": 85.75, "Typ vektoru": "Řídký (Sparse 10k)"},
        {"Metoda": "TF-IDF (10 000 feat.) + Logistická regrese", "Přesnost (Accuracy)": 87.30, "Typ vektoru": "Řídký vážený (10k)"},
        {"Metoda": "TF-IDF (10 000 feat.) + LinearSVC", "Přesnost (Accuracy)": 86.60, "Typ vektoru": "Řídký vážený (10k)"},
        {"Metoda": "Word2Vec (100D průměr) + Logistická regrese", "Přesnost (Accuracy)": 84.65, "Typ vektoru": "Hustý sémantický (100D)"},
    ])
    fig_perf = px.bar(
        perf_data,
        x="Přesnost (Accuracy)",
        y="Metoda",
        orientation="h",
        color="Přesnost (Accuracy)",
        color_continuous_scale="Tealgrn",
        text=perf_data["Přesnost (Accuracy)"].apply(lambda x: f"{x:.2f} %"),
        title="Srovnání přesnosti klasifikátorů na testovací sadě IMDb (2 000 recenzí)"
    )
    fig_perf.update_layout(template="plotly_dark", height=320, margin=dict(l=40, r=40, t=50, b=30), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_perf, width="stretch")


# ==============================================================================
# TAB 2: SROVNÁVACÍ MATICE & PRŮVODCE
# ==============================================================================
with tab2:
    st.subheader("Velká srovnávací matice metod NLP")

    matrix_df = pd.DataFrame([
        {
            "Metoda": "Bag of Words (BoW)",
            "Dimenze": "10 000+ (Řídký)",
            "Pořadí slov": "❌ Ignoruje",
            "Kontext": "❌ Žádný",
            "Sémantická afinita": "❌ Ne",
            "Výpočetní nároky": "⚡ Minimální (CPU)",
            "Kdy nasadit v praxi": "Základní baseline pro klasifikaci textů a detekci klíčových slov"
        },
        {
            "Metoda": "TF-IDF",
            "Dimenze": "10 000+ (Řídký)",
            "Pořadí slov": "❌ Ignoruje",
            "Kontext": "❌ Žádný",
            "Sémantická afinita": "❌ Ne",
            "Výpočetní nároky": "⚡ Minimální (CPU)",
            "Kdy nasadit v praxi": "Informační vyhledávání (Search), rychlá klasifikace zpráv a spamu"
        },
        {
            "Metoda": "Word2Vec",
            "Dimenze": "100 až 300 (Hustý)",
            "Pořadí slov": "⚠️ Lokální okno",
            "Kontext": "❌ Statický vektor",
            "Sémantická afinita": "✅ Výborná (Kosinus)",
            "Výpočetní nároky": "⚖️ Nízké / Střední (CPU)",
            "Kdy nasadit v praxi": "Doporučovací systémy, hledání synonym, expanze dotazů v e-shopech"
        },
        {
            "Metoda": "BERT & Transformery",
            "Dimenze": "768 až 1024 (Hustý)",
            "Pořadí slov": "✅ Poziční kódování",
            "Kontext": "⭐ Plně obousměrný",
            "Sémantická afinita": "⭐ Špičková",
            "Výpočetní nároky": "🔥 Vysoké (GPU / TPU)",
            "Kdy nasadit v praxi": "Question Answering, přesná analýza sentimentu, NER, sémantické vyhledávání"
        }
    ])
    st.dataframe(matrix_df, hide_index=True, width="stretch")

    st.markdown("---")
    st.subheader("🧭 Rozhodovací strom: Kterou metodu zvolit?")

    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        st.markdown(
            """
            ##### 1. Máte limitované výpočetní zdroje?
            * **Řešení:** Zvolte **TF-IDF + LinearSVC** nebo **Logistickou regresi**.
            * Natrénuje se za pár sekund na jakémkoliv CPU.
            * Pro klasifikaci témat a sentimentu dosahuje překvapivě vysoké přesnosti (až 87+ %).
            """
        )
    with col_d2:
        st.markdown(
            """
            ##### 2. Potřebujete hledat synonyma a podobné produkty?
            * **Řešení:** Zvolte **Word2Vec** nebo **FastText**.
            * Vektory mají jen 100–300 rozměrů, skvěle se indexují v paměti (Faiss, Annoy).
            * Umožňuje okamžitý výpočet kosinové afinity mezi slovy a frázemi.
            """
        )
    with col_d3:
        st.markdown(
            """
            ##### 3. Záleží na hlubokém porozumění a kontextu?
            * **Řešení:** Zvolte **BERT / RoBERTa / DeBERTa**.
            * Zvládá ironii, zápor (*"není to špatné"*), dvojí význam slov i vztahy mezi celými větami.
            * Vyžaduje GPU akceleraci pro odvozování a jemné doladění (Fine-Tuning).
            """
        )


# ==============================================================================
# TAB 3: EKOSYSTÉM LLM A PRŮMYSL (STAV K 10/2026)
# ==============================================================================
with tab3:
    st.subheader("Globální technologičtí lídři v NLP & Velké jazykové modely (Stav k 10/2026)")
    st.markdown(
        """
        Závěr Dne 4 (prezentace `Day_4_summary.pdf`, slidy 12–13) zdůrazňuje, 
        jak inovace představené v tomto kurzu položily základy současné revoluci v AI. 
        Od publikace Transformeru (2017) a BERTu (2018) se svět posunul od čistého generování textu 
        k **multimodálním reasoning modelům** a **autonomním softwarovým agentům**:
        """
    )

    # FRONTIER SPOTLIGHT BOX
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, rgba(30, 27, 75, 0.8), rgba(49, 46, 129, 0.8)); border: 1px solid rgba(129, 140, 248, 0.6); border-radius: 12px; padding: 20px; margin-bottom: 25px;">
            <h4 style="color: #a5b4fc; margin-top: 0;">🚀 Velká trojka Frontier AI (Aktuální stav k 10/2026): Argon, Astra & Fable</h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px;">
                <div style="background: rgba(15, 23, 42, 0.6); border-left: 4px solid #60a5fa; padding: 12px; border-radius: 6px;">
                    <strong style="color: #60a5fa;">⚡ Gemini 4 Argon (Google)</strong><br>
                    <small><em>Oznámeno: 30. září 2026</em></small><br>
                    První vlajkový model generace Gemini 4. Průlomový <strong>výstupní limit 1 000 000 tokenů</strong> pro dlouhodobé autonomní úlohy, kompletní migrace gigantických C/C++ systémů do Rustu (Fuchsia Zircon s 800k+ řádky) a kyberbezpečnost v programu Fairwind.
                </div>
                <div style="background: rgba(15, 23, 42, 0.6); border-left: 4px solid #34d399; padding: 12px; border-radius: 6px;">
                    <strong style="color: #34d399;">🌌 GPT-6 Astra (OpenAI)</strong><br>
                    <small><em>Vydáno: 4. září 2026</em></small><br>
                    Vlajková loď nové generace OpenAI navržená jako plnohodnotný <strong>Computer Operator</strong> (autonomní inspekce obrazovky, Blender 3D, QA, web). Rekordy: <strong>99.9 % na ARC-AGI-3</strong> a 97.6 % na FrontierMath. Doplněn o modely <em>Sol</em>, <em>Luna</em> a verzi pro NVIDIA Blackwell.
                </div>
                <div style="background: rgba(15, 23, 42, 0.6); border-left: 4px solid #fbbf24; padding: 12px; border-radius: 6px;">
                    <strong style="color: #fbbf24;">🎭 Claude Fable 5 & 5.1 (Anthropic)</strong><br>
                    <small><em>Vydáno: červen & září 2026</em></small><br>
                    Nejvyšší reasoning tier u Anthropicu (stojící nad Opus/Sonnet) pro komplexní vícedenní agentní běhy, vědecké modelování a matematiku. Doplněn o privátní výzkumnou třídu <em>Mythos</em> pro kyberbezpečnost a biomedicínu (Project Glasswing).
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    col_l1, col_l2 = st.columns(2)

    with col_l1:
        st.markdown(
            """
            <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(59, 130, 246, 0.4); border-radius: 10px; padding: 18px; margin-bottom: 15px;">
                <h4 style="color: #60a5fa; margin-top: 0;">🌐 Google (Google DeepMind)</h4>
                <p><strong>Od architektury Transformer k multimodálnímu rozumu:</strong></p>
                <ul>
                    <li><strong>2013–2018:</strong> Vynález <em>Word2Vec</em> (Mikolov), architektury <em>Transformer</em> (Vaswani et al.) a modelu <em>BERT</em>.</li>
                    <li><strong>Project Astra:</strong> Univerzální asistent s kontinuálním zrakem a pamětí v reálném čase (integrován v Gemini Live).</li>
                    <li><strong>Gemini 1.5 & 2.0:</strong> Kontextové okno <strong>2 000 000+ tokenů</strong> a nativní multimodalita (text, zvuk, video).</li>
                    <li><strong>10/2026 – Gemini 4 Argon:</strong> Nová generace pro dlouhodobý autonomní softwarový vývoj a masivní refaktoring.</li>
                    <li><strong>Otevřené modely Gemma 2 & 3:</strong> Špičkové otevřené váhy pro vývojáře.</li>
                </ul>
            </div>
            <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(16, 185, 129, 0.4); border-radius: 10px; padding: 18px; margin-bottom: 15px;">
                <h4 style="color: #34d399; margin-top: 0;">🤖 OpenAI & Microsoft</h4>
                <p><strong>Průkopník generativní AI a reasoning modelů:</strong></p>
                <ul>
                    <li><strong>2018–2022:</strong> Série <em>GPT-1</em> až <em>GPT-3</em>, spuštění fenoménu <em>ChatGPT</em> a komerční nasazení RLHF.</li>
                    <li><strong>GPT-4o (Omni):</strong> Sjednocená neuronová síť pro text, zrak a obousměrný realtime hlas s latencí pod 300 ms.</li>
                    <li><strong>OpenAI o1 ("Strawberry"), o3 & o3-mini:</strong> Posun k <strong>internímu řetězci uvažování (Chain-of-Thought)</strong>; řešení úloh na úrovni PhD v matematice (IMO) a autonomní programování.</li>
                    <li><strong>09/2026 – GPT-6 Astra, Sol & Luna:</strong> Skok do éry autonomního ovládání počítače (Computer Operator), ARC-AGI-3 (99.9 %) a optimalizace pro architekturu Blackwell.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_l2:
        st.markdown(
            """
            <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(168, 85, 247, 0.4); border-radius: 10px; padding: 18px; margin-bottom: 15px;">
                <h4 style="color: #c084fc; margin-top: 0;">🦙 Meta & Otevřený ekosystém (Open Weights)</h4>
                <p><strong>Demokratizace AI a podpora otevřené vědy:</strong></p>
                <ul>
                    <li><strong>2016–2023:</strong> <em>FastText</em>, <em>LLaMA</em> a <em>LLaMA-2</em>, které rozpoutaly globální open-source hnutí.</li>
                    <li><strong>Llama 3.1 (405B) & 3.2:</strong> První gigantický otevřený model vyrovnávající se uzavřeným špičkám se 128k kontextem a vizí.</li>
                    <li><strong>Llama 3.3 & Llama 4 (MoE):</strong> Otevřené reasoning modely provozovatelné lokálně v soukromí přes <em>Ollama</em> a <em>vLLM</em>.</li>
                    <li><strong>Hugging Face:</strong> Globální domov pro více než 1 000 000 otevřených modelů a knihovnu <code>transformers</code>.</li>
                </ul>
            </div>
            <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(245, 158, 11, 0.4); border-radius: 10px; padding: 18px; margin-bottom: 15px;">
                <h4 style="color: #fbbf24; margin-top: 0;">🧠 Anthropic & Agentní revoluce</h4>
                <p><strong>Bezpečnostní výzkum, kódování a uvažování:</strong></p>
                <ul>
                    <li><strong>Constitutional AI:</strong> Výcvik podle etických ústavních pravidel.</li>
                    <li><strong>Claude 3.5 Sonnet / Haiku / Opus:</strong> Průmyslový standard pro softwarové inženýrství a logiku.</li>
                    <li><strong>Claude Fable 5 & 5.1 (2026):</strong> Špičkový reasoning model pro dlouhotrvající agentní úkoly.</li>
                    <li><strong>Computer Use:</strong> Schopnost modelu přímo ovládat desktopové GUI (myš, klávesnice, kód v reálném OS).</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("##### 🚀 Evoluce klíčových parametrů NLP (2018 vs. 2026):")

    evo_df = pd.DataFrame([
        {
            "Éra / Model": "BERT-Base (2018)",
            "Kontextové okno": "512 tokenů (~350 slov)",
            "Počet parametrů": "110 milionů (0.11 B)",
            "Modalita": "Pouze čistý text",
            "Uvažování (Reasoning)": "Statické doplnění tokenu [MASK]",
            "Hardware pro trénink": "16 TPU jader (4 dny)"
        },
        {
            "Éra / Model": "GPT-3 (2020)",
            "Kontextové okno": "2 048 tokenů (~1 500 slov)",
            "Počet parametrů": "175 miliard (175 B)",
            "Modalita": "Pouze text",
            "Uvažování (Reasoning)": "Předpovídání dalšího slova (Autoregresivní)",
            "Hardware pro trénink": "Tisíce V100 GPU (měsíce)"
        },
        {
            "Éra / Model": "GPT-4 / Claude 3.5 (2023–2024)",
            "Kontextové okno": "128 000 – 200 000 tokenů",
            "Počet parametrů": "MoE (~1,8 bilionu / 1.8 T)",
            "Modalita": "Multimodální (Text, Obrázky, Kód)",
            "Uvažování (Reasoning)": "Pokročilý instruction-following, nástroje (Tools)",
            "Hardware pro trénink": "Desítky tisíc H100 GPU"
        },
        {
            "Éra / Model": "Současnost (Stav 10/2026)",
            "Kontextové okno": "2 000 000+ tokenů (Gemini 1.5/2.0)",
            "Počet parametrů": "Adaptivní MoE + Test-Time Compute (o1/o3)",
            "Modalita": "Nativní Omnimodalita (Text, Obraz, Zvuk, Video)",
            "Uvažování (Reasoning)": "Interní Chain-of-Thought, Computer Use & Autonomní agenti",
            "Hardware pro trénink": "Klastry B200 / H200 / TPU v5p & Inference Scaling"
        }
    ])
    st.dataframe(evo_df, hide_index=True, width="stretch")


# ==============================================================================
# TAB 4: ZÁVĚREČNÝ VĚDOMOSTNÍ KVÍZ
# ==============================================================================
with tab4:
    st.subheader("🧠 Závěrečný test znalostí Dne 4 (Time for a quiz!)")
    st.caption("Otestujte si své znalosti ze všech témat Dne 4: Od čištění textu až po BERT!")

    quiz_questions = [
        {
            "question": "1. Jaký je zásadní rozdíl mezi Stemmingem a Lemmatizací?",
            "options": [
                "Stemming je pomalejší, protože vyžaduje slovník spisovného jazyka.",
                "Stemming ořezává koncovky podle pevných pravidel (může vytvořit neexistující kmen), zatímco lemmatizace převádí slovo na gramaticky platný základní tvar (lemma).",
                "Lemmatizace ignoruje slovní druhy (POS tags), zatímco stemming je zohledňuje.",
                "Mezi stemmingem a lemmatizací není žádný praktický rozdíl."
            ],
            "correct": 1,
            "explanation": "Stemming (např. Porter) je hrubý heuristický algoritmus, který často vytvoří neexistující kmen (např. 'univers'). Lemmatizace využívá morfologický slovník a slovní druhy pro nalezení správného základního tvaru (např. 'better' -> 'good')."
        },
        {
            "question": "2. Co znamená zkratka TF-IDF a co vyjadřuje komponenta IDF?",
            "options": [
                "Total Frequency - Index Data File; vyjadřuje velikost souboru.",
                "Term Frequency - Inverse Document Frequency; IDF snižuje váhu slov, která se vyskytují v téměř všech dokumentech celého korpusu.",
                "Text Filtering - Internal Dictionary File; IDF počítá počet překlepů.",
                "Token Frequency - Instant Data Frame; IDF měří délku věty."
            ],
            "correct": 1,
            "explanation": "IDF = log(N / DF). Pokud se slovo vyskytuje v každém dokumentu (např. 'film' v korpusu filmových recenzí), hodnota IDF se blíží nule a jeho vliv na rozlišení dokumentů je potlačen."
        },
        {
            "question": "3. Které dvě trénovací metody nabízí model Word2Vec?",
            "options": [
                "Bag of Words a TF-IDF.",
                "Lasso a Ridge.",
                "CBOW (Continuous Bag of Words) a Skip-gram.",
                "Encoder a Decoder."
            ],
            "correct": 2,
            "explanation": "Word2Vec lze trénovat jako CBOW (předpověď cílového slova z okolí) nebo Skip-gram (předpověď okolního kontextu z cílového slova)."
        },
        {
            "question": "4. Proč byl vědecký článek z roku 2017 nazván 'Attention Is All You Need'?",
            "options": [
                "Protože doporučil používat pouze rekurentní sítě LSTM.",
                "Protože dokázal, že architektura složená výhradně z mechanismu Self-Attention (bez RNN smyček a konvolucí) překonává předchozí sekvenční modely a umožňuje plnou paralelizaci na GPU.",
                "Protože upozornil na problém mizejícího gradientu v lineární regresi.",
                "Protože představil nový způsob měření přesnosti pomocí F1-skóre."
            ],
            "correct": 1,
            "explanation": "Transformer zcela opustil rekurentní smyčky (RNN) a ukázal, že samotný mechanismus pozornosti (Self-Attention) stačí pro modelování jazyka a přináší obrovskou výpočetní rychlost díky paralelizaci."
        },
        {
            "question": "5. Jaké dvě samo-učící se úlohy (pre-training) využívá model BERT?",
            "options": [
                "Lineární regresi a K-Means clustering.",
                "Masked Language Model (MLM, pravidlo 15 % tokenů) a Next Sentence Prediction (NSP).",
                "Překlad z angličtiny do francouzštiny a generování obrázků.",
                "Detekci stop-slov a kosinovou podobnost."
            ],
            "correct": 1,
            "explanation": "BERT se učil na 3,3 miliardách slov bez lidských anotací pomocí: 1) MLM (doplňování maskovaných slov [MASK] z obousměrného kontextu) a 2) NSP (binární klasifikace, zda věta B skutečně navazuje na větu A)."
        },
        {
            "question": "6. K čemu v modelu BERT slouží speciální token [CLS]?",
            "options": [
                "K vymazání vyrovnávací paměti (Clear Screen).",
                "K oddělení první věty od druhé věty.",
                "Stojí na začátku sekvence a jeho výstupní kontextový vektor slouží jako souhrnná reprezentace celé věty pro klasifikační úlohy.",
                "Označuje konec dokumentu."
            ],
            "correct": 2,
            "explanation": "Token [CLS] (Classification) je vždy na nulté pozici. V poslední vrstvě Transformeru jeho 768-dimenzionální vektor integruje informace ze všech ostatních slov věty a vstupuje do klasifikátoru (např. pro určení pozitivního/negativního sentimentu)."
        }
    ]

    score = 0
    total = len(quiz_questions)

    for i, q in enumerate(quiz_questions):
        st.markdown(f"#### {q['question']}")
        user_ans = st.radio(
            f"Vyberte odpověď pro otázku {i+1}:",
            q["options"],
            key=f"quiz_d4_{i}",
            index=None
        )

        if user_ans is not None:
            ans_idx = q["options"].index(user_ans)
            if ans_idx == q["correct"]:
                st.success("✅ **Správně!** " + q["explanation"])
                score += 1
            else:
                st.error("❌ **Špatně.** " + q["explanation"])
        st.markdown("---")

    if st.button("Vyhodnotit výsledky kvízu"):
        st.markdown(f"### 🏆 Vaše celkové skóre: **{score} z {total} bodů** ({score/total*100:.0f} %)")
        if score == total:
            st.balloons()
            st.success("🎉 **Vynikající výkon!** Ovládli jste veškeré koncepty Dne 4 NLP na expertní úrovni.")
        elif score >= total * 0.7:
            st.info("👏 **Velmi dobrý výsledek!** Máte solidní základy v moderním NLP.")
        else:
            st.warning("💡 Doporučujeme projít si znovu teoretické přehledy a interaktivní laboratorní stránky.")
