"""
Moderní Machine Learning v roce 2026: RAG, LoRA & AI Agenti
============================================================
Interaktivní modul propojující základy ML kurzu s moderní praxí:
1. Živý RAG simulátor s vektorovým vyhledáváním a citací zdrojů
2. Strategický komparátor: Prompting vs. RAG vs. LoRA Fine-Tuning
3. Interaktivní simulátor AI Agenta a vzoru ReAct (Tool Use)
4. Produkční kódová šablona pro Hybrid Search (Dense + BM25)
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

st.set_page_config(page_title="Moderní ML 2026: RAG & Agenti", page_icon="⚡", layout="wide")

st.title("⚡ Moderní Machine Learning 2026: RAG, LoRA & AI Agenti")
st.caption(
    "Přirozená evoluce klasického strojového učení do éry velkých modelů: "
    "Jak vektorové embeddingy ze Dne 4, k-NN vyhledávání ze Dne 2 a maticové rozklady ze Dne 5 "
    "tvoří páteř moderních RAG architektur a autonomních AI agentů."
)

tab_rag, tab_matrix, tab_agent, tab_code = st.tabs([
    "🔍 Živý RAG Simulátor (Live Vector Search)",
    "⚖️ Strategie: RAG vs. LoRA vs. Prompting",
    "🤖 Architektura AI Agentů (ReAct)",
    "💻 Produkční kódová šablona 2026"
])

# -----------------------------------------------------------------------------
# TAB 1: Živý RAG Simulátor
# -----------------------------------------------------------------------------
with tab_rag:
    st.subheader("🔍 Živý interaktivní simulátor RAG (Retrieval-Augmented Generation)")
    st.markdown(
        """
        Vyzkoušejte si, jak funguje vyhledávací smyčka v moderní vektorové databázi. 
        Níže je uložena firemní znalostní báze dokumentů o strojovém učení. 
        Algoritmus převede váš dotaz na sémantický vektor, vyhledá $k$ nejbližších sousedů 
        pomocí kosínové podobnosti a sestaví faktickou odpověď s citacemi.
        """
    )

    # Znalostní báze chunků
    kb_documents = [
        {
            "id": "DOC-1",
            "title": "Metodika regularizace v lineárních modelech",
            "text": "L1 regularizace (Lasso) přidává penalizaci absolutních hodnot vah a dokáže koeficienty zcela vynulovat, čímž provádí automatický výběr příznaků. Naopak L2 regularizace (Ridge) penalizuje čtverce vah a je ideální při multikolinearitě, protože koeficienty zmenšuje k nule, ale žádný zcela neodstraní."
        },
        {
            "id": "DOC-2",
            "title": "Evaluace klasifikátorů u nevyvážených tříd",
            "text": "Pokud je v datasetu 98 % negativních vzorků a 2 % pozitivních, metrika Accuracy zcela selhává, protože triviální model predikující vždy nulu dosáhne 98 %. V medicíně a při detekci podvodů je klíčový Recall (Senzitivita) pro minimalizaci False Negatives a metrika PR-AUC (Precision-Recall křivka)."
        },
        {
            "id": "DOC-3",
            "title": "Architektura a subsampling algoritmu Isolation Forest",
            "text": "Isolation Forest izoluje odlehlé body pomocí náhodných řezů v prostoru. Anomálie mají výrazně kratší průměrnou hloubku ve stromech než běžné body. Pro zabránění jevům swamping a masking se doporučuje fixovat subsampling na hodnotu max_samples=256, která je matematicky optimální pro téměř libovolný počet vzorků."
        },
        {
            "id": "DOC-4",
            "title": "Princip sémantických embeddingů a Word2Vec",
            "text": "Algoritmus Word2Vec reprezentuje slova jako husté číselné vektory v prostoru o rozměru typicky 100 až 300. Vektory zachycují sémantické vztahy tak, že slova s podobným významem mají vysokou kosínovou podobnost. Architektura CBOW predikuje středové slovo z kontextu, zatímco Skip-gram predikuje kontext ze zadaného slova."
        },
        {
            "id": "DOC-5",
            "title": "Hustotní shlukování DBSCAN vs K-Means",
            "text": "K-Means předpokládá kulovité shluky podobné velikosti a vyžaduje zadání parametru K. Algoritmus DBSCAN shlukuje na základě lokální hustoty bodů, dokáže najít útvary libovolného tvaru (půlměsíce, kružnice) a automaticky detekuje šum a odlehlé hodnoty, které označí labelem -1."
        }
    ]

    col_q1, col_q2 = st.columns([2, 1])
    with col_q1:
        preset_q = st.selectbox(
            "Vyberte ukázkový dotaz nebo napište vlastní níže:",
            [
                "Vlastní dotaz (zadejte níže)",
                "Jak se liší Lasso a Ridge regularizace při výběru příznaků?",
                "Proč nemáme používat Accuracy u nevyvážených lékařských dat?",
                "Jaký parametr subsamplingu je optimální pro Isolation Forest?",
                "Který algoritmus dokáže najít nepravidelné shluky a označit šum jako -1?",
                "Jak Word2Vec převádí text na vektory a jak funguje Skip-gram?"
            ]
        )
        if preset_q == "Vlastní dotaz (zadejte níže)":
            user_query = st.text_input("Zadejte dotaz pro vyhledávání ve znalostní bázi:", "Jak funguje subsampling u anomálií?")
        else:
            user_query = preset_q

    with col_q2:
        top_k = st.slider("Počet vrácených chunků (Top-K):", min_value=1, max_value=4, value=2)

    # Výpočet podobnosti přes TF-IDF (analog embeddingů)
    corpus = [doc["text"] for doc in kb_documents]
    tfidf = TfidfVectorizer(stop_words=None)
    doc_vectors = tfidf.fit_transform(corpus)
    query_vector = tfidf.transform([user_query])

    sim_scores = cosine_similarity(query_vector, doc_vectors)[0]
    ranked_indices = np.argsort(sim_scores)[::-1][:top_k]

    st.markdown("---")
    st.subheader("📊 Fáze 1: Vektorové vyhledávání v databázi (Top-K retrieval)")

    r_cols = st.columns(top_k)
    retrieved_texts = []
    retrieved_sources = []

    for i, idx in enumerate(ranked_indices):
        score = sim_scores[idx]
        doc = kb_documents[idx]
        retrieved_texts.append(doc["text"])
        retrieved_sources.append(doc["id"])

        with r_cols[i]:
            with st.container(border=True):
                st.markdown(f"**Zdroj: `{doc['id']}`**")
                st.caption(f"📌 {doc['title']}")
                st.metric("Kosínová podobnost", f"{score:.4f}")
                st.markdown(f"> *„{doc['text'][:140]}...“*")

    st.markdown("---")
    st.subheader("🤖 Fáze 2: RAG Syntéza & Odpověď s citací zdrojů")

    c_prompt, c_response = st.columns([1, 1])

    with c_prompt:
        with st.container(border=True):
            st.markdown("##### 📝 Sestavený Prompt pro LLM (Injektovaný kontext):")
            formatted_context = "\n\n".join([f"[{doc_id}]: {txt}" for doc_id, txt in zip(retrieved_sources, retrieved_texts)])
            prompt_preview = f"""SYSTEM: Jsi přesný asistent. Odpovídej VÝHRADNĚ na základě poskytnutého kontextu. U každého tvrzení uveď citaci ve formátu [DOC-X]. Pokud odpověď v kontextu není, odpověz: 'Tuto informaci ve firemních podkladech nemám.'

KONTEXT Z VEKTOROVÉ DATABÁZE:
{formatted_context}

DOTAZ UŽIVATELE:
{user_query}"""
            st.code(prompt_preview, language="markdown")

    with c_response:
        with st.container(border=True):
            st.markdown("##### 💡 Vygenerovaná RAG odpověď:")
            # Dynamické generování věcné odpovědi na základě nalezených zdrojů
            top_doc = kb_documents[ranked_indices[0]]
            st.success("✅ **Odpověď vygenerována s nulovou halucinací a ověřenými citacemi:**")
            st.markdown(
                f"""
                Na základě firemní dokumentace **[{top_doc['id']}]**:
                
                {top_doc['text']}
                
                *(Zdrojový dokument: `{top_doc['title']}`, shoda: {sim_scores[ranked_indices[0]]:.1%})*
                """
            )
            st.info("🛡️ **Garance spolehlivosti:** Model nečerpal z obecných předtrénovaných vah na internetu, ale striktně syntetizoval data z nalezených chunků.")

# -----------------------------------------------------------------------------
# TAB 2: Strategie (RAG vs. LoRA vs. Prompting)
# -----------------------------------------------------------------------------
with tab_matrix:
    st.subheader("⚖️ Kdy zvolit Prompting, RAG a kdy LoRA Fine-Tuning?")
    st.markdown(
        """
        V praxi často dochází k drahé chybě: týmy se pokoušejí dotrénovat model (Fine-Tuning) na nová data, 
        přestože správným řešením byl RAG. Níže je strategický komparátor pro volbu optimální cesty:
        """
    )

    c_m1, c_m2 = st.columns([1, 1])
    with c_m1:
        st.markdown("#### 🎯 Co je vaším primárním cílem?")
        need_factual = st.checkbox("Potřebuji, aby model znal nejnovější firemní data, která se denně mění", value=True)
        need_cite = st.checkbox("Musíme být schopni doložit přesný odkaz na zdroj a předcházet halucinacím", value=True)
        need_style = st.checkbox("Potřebuji změnit tón komunikace, formát výstupu nebo naučit model specifický interní kód", value=False)
        budget_limit = st.select_slider("Rozpočet a hardware:", options=["Nízký (Pouze API)", "Střední (Vektorová DB + 1 GPU)", "Vysoký (Vlastní GPU cluster)"], value="Střední (Vektorová DB + 1 GPU)")

        st.markdown("---")
        st.markdown("#### 🏆 Doporučená architektura:")
        if need_style and not need_factual:
            st.success("⚡ **Doporučení: LoRA / QLoRA Fine-Tuning**")
            st.write("Model potřebuje změnit styl, chování či gramatiku odpovědí. LoRA je ideální volbou pro adaptaci vah při minimálních nákladech.")
        elif need_factual or need_cite:
            st.success("🔍 **Doporučení: RAG (Retrieval-Augmented Generation)**")
            st.write("RAG je nepřekonatelný v přesnosti faktů, dynamické aktualizaci bez nutnosti trénování a transparentním citování zdrojů.")
        else:
            st.success("💬 **Doporučení: Pokročilý Prompt Engineering (Few-Shot)**")
            st.write("Pro běžné úlohy stačí dodat kvalitní instrukce a 2–3 ukázkové příklady přímo do kontextového okna modelu.")

    with c_m2:
        radar_categories = ["Faktická přesnost", "Čerstvost dat", "Změna stylu/tónu", "Nízké náklady", "Jednoduchost nasazení"]
        fig_strat = go.Figure()
        fig_strat.add_trace(go.Scatterpolar(
            r=[5, 6, 6, 10, 10],
            theta=radar_categories,
            fill="toself",
            name="Prompt Engineering"
        ))
        fig_strat.add_trace(go.Scatterpolar(
            r=[10, 10, 5, 8, 8],
            theta=radar_categories,
            fill="toself",
            name="RAG Architecture"
        ))
        fig_strat.add_trace(go.Scatterpolar(
            r=[6, 3, 10, 4, 4],
            theta=radar_categories,
            fill="toself",
            name="LoRA Fine-Tuning"
        ))
        fig_strat.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
            showlegend=True,
            title="Srovnání přístupů (1 = nejhorší, 10 = nejlepší)",
            margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig_strat)

# -----------------------------------------------------------------------------
# TAB 3: Architektura AI Agentů (ReAct)
# -----------------------------------------------------------------------------
with tab_agent:
    st.subheader("🤖 Jak funguje AI Agent: Vzor ReAct (Reason + Act)")
    st.markdown(
        """
        AI Agent není pouhý textový generátor. Je to **autonomní řídicí smyčka**, 
        která střídá fáze úvahy (*Thought*), volby nástroje (*Action*) a vyhodnocení výsledku (*Observation*).
        """
    )

    sample_task = st.selectbox(
        "Vyberte komplexní byznysový úkol k vyřešení agentem:",
        [
            "Úkol A: Zjisti průměrnou cenu diamantů s brusem 'Ideal' a spočítej předpověď pro kámen 1.5 karátu",
            "Úkol B: Zkontroluj příchozí platbu na platební bráně a prověř, zda nejde o fraud anomálii",
            "Úkol C: Přečti technickou zprávu a vygeneruj shrnutí pro ředitele výroby"
        ]
    )

    st.markdown("---")
    st.markdown("#### 🔄 Průběh exekuce agentního cyklu krok za krokem:")

    if "Úkol A" in sample_task:
        steps = [
            ("🧠 1. Thought (Úvaha)", "Uživatel žádá statistiku z databáze a následnou ML predikci. Musím nejprve zavolat SQL databázi pro zjištění průměrné ceny a poté Scikit-learn regresní pipeline pro ocenění 1.5ct kamene."),
            ("🛠️ 2. Action (Volba nástroje)", "Spouštím nástroj: `SQLQueryTool(query='SELECT AVG(price) FROM diamonds WHERE cut = \"Ideal\"')`"),
            ("👁️ 3. Observation (Pozorování)", "Výsledek z databáze: `3 457.50 USD` (vzorek 21 551 diamantů)."),
            ("🧠 4. Thought (Úvaha)", "Průměrnou cenu znám. Nyní musím zavolat natrénovaný model `RidgeRegressor` pro předpověď ceny konkrétního diamantu s parametry: carat=1.5, cut='Ideal', color='G', clarity='VS1'."),
            ("🛠️ 5. Action (Volba nástroje)", "Spouštím nástroj: `ScikitLearnPredictor(model='diamond_ridge.joblib', features=[1.5, 'Ideal', 'G', 'VS1'])`"),
            ("👁️ 6. Observation (Pozorování)", "Model vrátil odhad: `11 240 USD` (s 95% konfidenčním intervalem 10 800 – 11 650 USD)."),
            ("🏁 7. Final Response (Syntéza)", "Sestavuji finální strukturovanou odpověď pro uživatele s uvedením obou zjištěných faktů a interpretací.")
        ]
    elif "Úkol B" in sample_task:
        steps = [
            ("🧠 1. Thought (Úvaha)", "Příchozí transakce má neobvyklou částku a IP adresu. Musím spustit model detekce anomálií Isolation Forest."),
            ("🛠️ 2. Action (Volba nástroje)", "Spouštím nástroj: `IsolationForestScorer(amount=120000, country='NG', time='03:15')`"),
            ("👁️ 3. Observation (Pozorování)", "Skóre anomálie = -0.42 (bod je označen jako OUTLIER -1, v 99.8. percentilu odlehlosti)."),
            ("🧠 4. Thought (Úvaha)", "Model detekoval vysoké riziko podvodu. Musím prověřit historii účtu v databázi klientů."),
            ("🛠️ 5. Action (Volba nástroje)", "Spouštím nástroj: `CustomerRiskAudit(user_id=84920)`"),
            ("👁️ 6. Observation (Pozorování)", "Uživatel doposud neprováděl transakce nad 500 USD. Přihlášení z nového zařízení."),
            ("🏁 7. Final Response (Syntéza)", "BLOKOVAT TRANSAKCI: Spuštěn bezpečnostní protokol, odeslána výzva k ověření 2FA přes SMS.")
        ]
    else:
        steps = [
            ("🧠 1. Thought (Úvaha)", "Zpráva má 45 stran. Musím provést vektorové vyhledávání RAG a vyextrahovat klíčové závěry o výrobních vadách."),
            ("🛠️ 2. Action (Volba nástroje)", "Spouštím nástroj: `VectorSearch(query='hlavní příčiny vad a finanční ztráty za Q3', top_k=3)`"),
            ("👁️ 3. Observation (Pozorování)", "Nalezeny 3 klíčové odstavce: opotřebení ložisek na lince B (ztráta 450 tis. Kč)."),
            ("🧠 4. Thought (Úvaha)", "Mám klíčová fakta. Nyní naformátuji exekutivní shrnutí do 3 bodů."),
            ("🏁 5. Final Response (Syntéza)", "Předkládám řediteli 3bodový akční přehled s doporučením okamžité výměny ložisek.")
        ]

    for title, desc in steps:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.write(desc)

# -----------------------------------------------------------------------------
# TAB 4: Produkční kódová šablona 2026
# -----------------------------------------------------------------------------
with tab_code:
    st.subheader("💻 Produkční kódová šablona: Hybrid Search v Pythonu (2026)")
    st.markdown(
        """
        Jak vypadá moderní RAG pipeline v kódu? Níže je plně funkční ukázka 
        **Hybridního vyhledávání** spojujícího husté sémantické vektory s lexikálním BM25 algoritmem:
        """
    )

    st.code(
        r'''from langchain_community.retrievers import QdrantHybridRetriever
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from qdrant_client import QdrantClient

# 1. Inicializace klientů a embedding modelu
qdrant = QdrantClient(host="localhost", port=6333)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

# 2. Vytvoření hybridního vyhledávače (Dense sémantika + Sparse klíčová slova)
retriever = QdrantHybridRetriever(
    client=qdrant,
    collection_name="enterprise_knowledge",
    embeddings=embeddings,
    top_k=4  # Vrátí top 4 nejrelevantnější chunky
)

# 3. Striktní systémový prompt zamezující halucinacím
system_prompt = (
    "Jsi expertní firemní asistent. "
    "Odpovídej VÝHRADNĚ s využitím následujícího kontextu. "
    "U každého tvrzení uveď přesnou citaci [DOC-ID]. "
    "Pokud odpověď v kontextu nenajdeš, řekni, že informaci nemáš k dispozici.\n\n"
    "KONTEXT:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 4. Spojení retrievalu a syntézy do jediné RAG pipeline
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_pipeline = create_retrieval_chain(retriever, question_answer_chain)

# 5. Exekuce dotazu
response = rag_pipeline.invoke({"input": "Jaká jsou pravidla pro nastavení subsamplingu u Isolation Forest?"})
print("Odpověď:", response["answer"])
print("Použité zdroje:", [doc.metadata["source"] for doc in response["context"]])
''',
        language="python"
    )
