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
# TAB 3: EKOSYSTÉM LLM A PRŮMYSL
# ==============================================================================
with tab3:
    st.subheader("Globální technologičtí lídři v NLP & Velké jazykové modely")
    st.markdown(
        """
        Závěr Dne 4 (prezentace `Day_4_summary.pdf`, slidy 12–13) zdůrazňuje, 
        jak inovace představené v tomto kurzu formují globální trh s AI:
        """
    )

    col_l1, col_l2, col_l3 = st.columns(3)

    with col_l1:
        st.markdown(
            """
            <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(59, 130, 246, 0.4); border-radius: 10px; padding: 18px; height: 100%;">
                <h4 style="color: #60a5fa; margin-top: 0;">🌐 Google</h4>
                <p><strong>Základní přínos pro lidstvo:</strong></p>
                <ul>
                    <li>2013: <strong>Word2Vec</strong> (Mikolov et al.)</li>
                    <li>2017: <strong>Transformer</strong> ("Attention Is All You Need")</li>
                    <li>2018: <strong>BERT</strong> (Obousměrný enkodér)</li>
                    <li>2020: <strong>T5</strong> (Text-to-Text Transfer Transformer)</li>
                    <li>Současnost: <strong>Gemini</strong> (Multimodální LLM)</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_l2:
        st.markdown(
            """
            <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(16, 185, 129, 0.4); border-radius: 10px; padding: 18px; height: 100%;">
                <h4 style="color: #34d399; margin-top: 0;">🤖 OpenAI & Microsoft</h4>
                <p><strong>Lídr v generativním AI:</strong></p>
                <ul>
                    <li>2018: <strong>GPT-1</strong> (Generative Pre-trained Transformer)</li>
                    <li>2019: <strong>GPT-2</strong> (Škálování jazykových modelů)</li>
                    <li>2020: <strong>GPT-3</strong> (175 miliard parametrů)</li>
                    <li>2022+: <strong>ChatGPT & GPT-4</strong> (RLHF a instruované modely)</li>
                    <li>Současnost: <strong>GPT-4o & Copilot</strong></li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_l3:
        st.markdown(
            """
            <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(168, 85, 247, 0.4); border-radius: 10px; padding: 18px; height: 100%;">
                <h4 style="color: #c084fc; margin-top: 0;">🦙 Meta & Open-Source</h4>
                <p><strong>Šampion otevřeného výzkumu:</strong></p>
                <ul>
                    <li>2016: <strong>FastText</strong> (Subword n-gram embeddingy)</li>
                    <li>2023: <strong>LLaMA & LLaMA-2</strong> (Vznik open-source LLM revoluce)</li>
                    <li>2024: <strong>Llama-3</strong> (Špičkové 8B a 70B otevřené modely)</li>
                    <li><strong>Hugging Face:</strong> Globální domov pro sdílení modelů a datasetů.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )


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
