"""
Den 4: NLP – BERT & Hugging Face Pipeline
=========================================
Podklad: Resources/Day 4 PDF/BERT.pdf
Klíčový článek: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (Devlin et al., 2018)
Precomputed: 04_NLP/data/bert_precomputed.json
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.title("🤖 BERT: Obousměrné reprezentace z Transformerů & Hugging Face")
st.caption(
    "Hloubkový rozbor architektury BERT (Google 2018), samo-učících se úloh MLM a NSP, "
    "praktického využití v knihovně Hugging Face `transformers` a analýzy etických zkreslení (Bias) v jazykových modelech."
)

base_dir = Path(__file__).resolve().parent.parent
json_path = base_dir / "04_NLP" / "data" / "bert_precomputed.json"


@st.cache_data
def load_bert_stats():
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


stats = load_bert_stats()
meta = stats["metadata"] if stats else {}

# Horní KPI karty
c1, c2, c3, c4 = st.columns(4)
c1.metric("Počet parametrů (BERT-Base)", "110 M", delta="12 vrstev / 768 dimenzí")
c2.metric("Předtrénovací korpus", "3,3 miliardy slov", delta="BooksCorpus + Wikipedia")
c3.metric("Hardware tréninku", "16 TPU jader (4 dny)", delta="96 hodin nepřetržitě")
c4.metric("Obousměrnost (Bidirectional)", "100 %", delta="Pravý i levý kontext")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "🧠 1. Architektura & Vstupní reprezentace",
    "🎭 2. Trénovací úlohy: MLM & NSP",
    "🧪 3. Interaktivní `fill-mask` (CZ / EN)",
    "⚖️ 4. Etika, zkreslení (Bias) & Aplikace"
])

# ==============================================================================
# TAB 1: ARCHITEKTURA & VSTUPNÍ REPREZENTACE
# ==============================================================================
with tab1:
    st.subheader("Proč je BERT skutečně obousměrný (Bidirectional)?")

    st.markdown(
        r"""
        Před příchodem BERTu měly jazykové modely zásadní omezení:
        * **Word2Vec / GloVe:** Statické embeddingy bez ohledu na kontext (*"river bank"* vs *"bank account"* měly stejný vektor).
        * **OpenAI GPT:** Autoregresivní model čtoucí text přísně zleva doprava. Pro predikci slova mohl využít pouze předchozí historii.
        * **BERT:** Využívá **výhradně bloky Encoderu** z architektury Transformer. V každé vrstvě se každé slovo dívá současně **doleva i doprava**.
        """
    )

    st.markdown("##### 🧱 Vstupní reprezentace: Součet tří embeddingů")
    st.markdown(
        r"""
        Vstupem do BERTu není holý text, ale součet tří na sobě nezávislých vektorů pro každý token:
        $$\mathbf{E}_{vstup} = \mathbf{E}_{WordPiece} + \mathbf{E}_{Segment} + \mathbf{E}_{Position}$$
        """
    )

    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1:
        st.markdown(
            """
            <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(59, 130, 246, 0.4); border-radius: 8px; padding: 15px;">
                <h5 style="color: #60a5fa; margin-top: 0;">1. Token Embeddings</h5>
                <p>Slovník 30 522 sub-word tokenů (WordPiece). Slova se dělí na známé kmeny a přípony (např. <code>playing</code> &rarr; <code>play</code> + <code>##ing</code>).</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_e2:
        st.markdown(
            """
            <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(168, 85, 247, 0.4); border-radius: 8px; padding: 15px;">
                <h5 style="color: #c084fc; margin-top: 0;">2. Segment Embeddings</h5>
                <p>Rozlišuje, zda token patří do první věty ($E_A$) nebo do druhé věty ($E_B$), což je klíčové pro párové úlohy jako Question Answering.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_e3:
        st.markdown(
            """
            <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(34, 197, 94, 0.4); border-radius: 8px; padding: 15px;">
                <h5 style="color: #4ade80; margin-top: 0;">3. Position Embeddings</h5>
                <p>Naučené vektory pro každou pozici od 0 do 512, které modelu předávají informaci o přesném pořadí slov ve větě.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("##### 🎯 Speciální řídicí tokeny:")
    c_cls, c_sep = st.columns(2)
    with c_cls:
        st.info(
            "🏷️ **`[CLS]` (Classification Token):**\n\n"
            "Vždy stojí na samém začátku každé sekvence. Protože v Transformeru každá vrstva agreguje informace ze všech slov, "
            "výstupní vektor odpovídající pozici `[CLS]` slouží jako **souhrnná reprezentace celé věty** pro klasifikaci sentimentu nebo témat."
        )
    with c_sep:
        st.info(
            "✂️ **`[SEP]` (Separator Token):**\n\n"
            "Slouží jako oddělovač mezi první a druhou větou (např. dotaz a kontext v Question Answering) "
            "a zároveň uzavírá konec celé vstupní sekvence."
        )

    st.markdown("##### 🔬 Interaktivní ukázka tokenizace věty:")
    sample_sent = st.text_input(
        "Zadejte libovolnou větu pro rozklad do reprezentace BERT:",
        value="The movie was extraordinarily exciting and emotional."
    )
    if sample_sent.strip():
        raw_words = sample_sent.strip().split()
        tok_list = ["[CLS]"]
        for w in raw_words:
            if len(w) > 8:
                tok_list.append(w[:5])
                tok_list.append("##" + w[5:])
            else:
                tok_list.append(w)
        tok_list.append("[SEP]")

        tok_df = pd.DataFrame({
            "Pozice": range(len(tok_list)),
            "Token": tok_list,
            "Typ": ["Řídicí token" if t in ["[CLS]", "[SEP]"] else ("Subword (##)" if "##" in t else "Slovo") for t in tok_list],
            "Segment": ["Segment A"] * len(tok_list),
            "Poziční ID": [f"Pos_{i}" for i in range(len(tok_list))]
        })
        st.dataframe(tok_df, hide_index=True, width="stretch")


# ==============================================================================
# TAB 2: TRÉNOVACÍ ÚLOHY MLM & NSP
# ==============================================================================
with tab2:
    st.subheader("Dvě samo-učící se úlohy (Self-Supervised Pre-training)")

    st.markdown(
        """
        BERT byl natrénován na obrovském korpusu 3,3 miliardy slov **bez jakéhokoliv lidského značkování**. 
        K tomu využil dvě paralelní trénovací úlohy:
        """
    )

    col_mlm, col_nsp = st.columns([1, 1])

    with col_mlm:
        st.markdown(
            """
            <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(59, 130, 246, 0.4); border-radius: 10px; padding: 18px; height: 100%;">
                <h4 style="color: #60a5fa; margin-top: 0;">1. Masked Language Model (MLM)</h4>
                <p>Klasické jazykové modely předpovídají další slovo zleva doprava. Kdybychom to zkusili s obousměrným Transformerem, slova by jednoduše "viděla sama sebe".</p>
                <p><strong>Pravidlo maskování 15 % tokenů:</strong></p>
                <ul>
                    <li><strong>80 % případů:</strong> Nahrazeno tokenem <code>[MASK]</code>.<br><em>"The capital of Poland is [MASK]."</em> &rarr; cíl: <em>warsaw</em></li>
                    <li><strong>10 % případů:</strong> Nahrazeno náhodným slovem.<br><em>"The capital of Poland is apple."</em> &rarr; cíl: <em>warsaw</em></li>
                    <li><strong>10 % případů:</strong> Ponecháno beze změny.<br><em>"The capital of Poland is warsaw."</em> &rarr; cíl: <em>warsaw</em></li>
                </ul>
                <p>Ztrátová funkce (Cross-Entropy Loss) se počítá <strong>pouze pro těchto 15 % pozic</strong>!</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_nsp:
        st.markdown(
            """
            <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(168, 85, 247, 0.4); border-radius: 10px; padding: 18px; height: 100%;">
                <h4 style="color: #c084fc; margin-top: 0;">2. Next Sentence Prediction (NSP)</h4>
                <p>Model dostává dvojice vět A a B a učí se binární klasifikaci, zda věta B v původním textu skutečně následuje za větou A.</p>
                <ul>
                    <li><strong>50 % případů:</strong> <code>is_next</code> (label = 1). Věta B skutečně navazuje na větu A.</li>
                    <li><strong>50 % případů:</strong> <code>not_next</code> (label = 0). Věta B je náhodná věta vybraná z jiného článku v korpusu.</li>
                </ul>
                <p>Tato úloha naučila BERT chápat vztahy a kauzalitu mezi celými odstavci, což je klíčové pro <strong>Question Answering</strong> a sémantické vyhledávání.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("##### 🧪 Interaktivní vyhodnocení NSP (Next Sentence Prediction) z přednášky:")

    if stats and "nsp_examples" in stats:
        nsp_list = stats["nsp_examples"]
        choice_idx = st.selectbox(
            "Vyberte dvojici vět pro otestování NSP klasifikátoru:",
            range(len(nsp_list)),
            format_func=lambda i: f"Pár {i+1}: {nsp_list[i]['sentence_a'][:45]}... & {nsp_list[i]['sentence_b'][:45]}..."
        )

        curr_nsp = nsp_list[choice_idx]
        col_pa, col_pb = st.columns(2)
        with col_pa:
            st.markdown(f"**Věta A:** `{curr_nsp['sentence_a']}`")
        with col_pb:
            st.markdown(f"**Věta B:** `{curr_nsp['sentence_b']}`")

        p_next = curr_nsp["probability_is_next"]
        col_r1, col_r2 = st.columns([1, 2])
        with col_r1:
            if curr_nsp["is_next"]:
                st.success(f"### ✅ IS_NEXT\n**Pravděpodobnost návaznosti:** {p_next*100:.1f} %")
            else:
                st.error(f"### ❌ NOT_NEXT\n**Pravděpodobnost nesouvisející:** {(1-p_next)*100:.1f} %")
        with col_r2:
            st.info(f"💡 **Vysvětlení modelu:** {curr_nsp['explanation']}")


# ==============================================================================
# TAB 3: INTERAKTIVNÍ FILL-MASK
# ==============================================================================
with tab3:
    st.subheader("Interaktivní Hugging Face Pipeline: `fill-mask`")
    st.caption("Doplňování chybějícího slova `[MASK]` na základě obousměrného kontextu věty.")

    st.markdown(
        r"""
        V knihovně **Hugging Face Transformers** stačí pouhé tři řádky kódu:
        ```python
        from transformers import pipeline

        unmasker = pipeline("fill-mask", model="bert-base-uncased")
        results = unmasker("The capital of Poland is [MASK].")
        ```
        """
    )

    lang_fm = st.radio(
        "Zvolte jazykový režim:",
        ["🇬🇧 Angličtina (Předlohy z BERT.pdf)", "🇨🇿 Čeština (Národní podpora & Sentiment)"],
        horizontal=True
    )

    if stats and "fill_mask_examples" in stats:
        all_examples = stats["fill_mask_examples"]

        if "Angličtina" in lang_fm:
            en_keys = [k for k in all_examples.keys() if "[MASK]" in k and not any(cz in k for cz in ["České", "film byl", "film byla"])]
            selected_prompt = st.selectbox("Vyberte anglickou předlohu:", en_keys)
        else:
            cz_keys = [k for k in all_examples.keys() if any(cz in k for cz in ["České", "film byl", "film byla"])]
            selected_prompt = st.selectbox("Vyberte českou předlohu:", cz_keys)

        pred_items = all_examples.get(selected_prompt, [])

        if pred_items:
            st.markdown(f"##### Vstupní věta: `{selected_prompt}`")

            df_preds = pd.DataFrame(pred_items)
            df_preds["Pravděpodobnost"] = df_preds["score"].apply(lambda s: f"{s*100:.2f} %")
            df_preds["Doplněné slovo"] = df_preds["token_str"]
            df_preds["Kompletní predikovaná věta"] = df_preds["sentence"]

            col_chart, col_tbl = st.columns([1, 1])

            with col_chart:
                fig_bar = px.bar(
                    df_preds,
                    x="score",
                    y="Doplněné slovo",
                    orientation="h",
                    color="score",
                    color_continuous_scale="Viridis",
                    labels={"score": "Pravděpodobnost výskytu", "Doplněné slovo": "Kandidát [MASK]"},
                    title="Distribuce pravděpodobností slov na pozici [MASK]"
                )
                fig_bar.update_layout(
                    template="plotly_dark",
                    height=320,
                    margin=dict(l=30, r=30, t=40, b=30),
                    yaxis=dict(autorange="reversed")
                )
                st.plotly_chart(fig_bar, width="stretch")

            with col_tbl:
                st.dataframe(
                    df_preds[["Doplněné slovo", "Pravděpodobnost", "token", "Kompletní predikovaná věta"]],
                    hide_index=True,
                    width="stretch"
                )

            top_cand = pred_items[0]
            st.success(
                f"🏆 **Nejlepší doplnění:** Slovo **'{top_cand['token_str']}'** s jistotou **{top_cand['score']*100:.2f} %**!\n\n"
                f"*Výsledná věta:* `{top_cand['sentence']}`"
            )


# ==============================================================================
# TAB 4: ETIKA, ZKRESLENÍ (BIAS) & APLIKACE
# ==============================================================================
with tab4:
    st.subheader("Etika a zkreslení (Bias) v jazykových modelech")

    st.markdown(
        """
        Protože modely jako BERT a GPT jsou trénovány na reálných textech vytvořených lidmi 
        (knihy, internet, sociální sítě, zpravodajství), **nevyhnutelně absorbují a zrcadlí lidské předsudky a stereotypy**.

        V prezentaci `BERT.pdf` (slidy 21–22) je představen klíčový experiment zkoumající predikci profesí pro muže a ženy:
        1. *"The man worked as a [MASK]."*
        2. *"The woman worked as a [MASK]."*
        """
    )

    if stats and "fill_mask_examples" in stats:
        men_data = stats["fill_mask_examples"].get("The man worked as a [MASK].", [])
        women_data = stats["fill_mask_examples"].get("The woman worked as a [MASK].", [])

        df_m = pd.DataFrame(men_data)
        df_m["Pohlaví"] = "Muž (The man worked as a...)"

        df_w = pd.DataFrame(women_data)
        df_w["Pohlaví"] = "Žena (The woman worked as a...)"

        c_m, c_w = st.columns(2)
        with c_m:
            st.markdown("##### 👨 Výstup pro: *'The man worked as a [MASK].'*")
            fig_m = px.bar(
                df_m,
                x="score",
                y="token_str",
                orientation="h",
                color_discrete_sequence=["#3b82f6"],
                labels={"score": "Pravděpodobnost", "token_str": "Profese"}
            )
            fig_m.update_layout(template="plotly_dark", height=320, margin=dict(l=30, r=30, t=20, b=20), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_m, width="stretch")
            st.caption("Přednostně predikováno: *lawyer, carpenter, doctor, mechanic, engineer, driver*.")

        with c_w:
            st.markdown("##### 👩 Výstup pro: *'The woman worked as a [MASK].'*")
            fig_w = px.bar(
                df_w,
                x="score",
                y="token_str",
                orientation="h",
                color_discrete_sequence=["#ec4899"],
                labels={"score": "Pravděpodobnost", "token_str": "Profese"}
            )
            fig_w.update_layout(template="plotly_dark", height=320, margin=dict(l=30, r=30, t=20, b=20), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_w, width="stretch")
            st.caption("Přednostně predikováno: *nurse, waitress, maid, teacher, secretary, cleaner*.")

        st.warning(
            "⚠️ **Důsledky v praxi:** Pokud by takový model byl bez úprav nasazen například pro **automatické filtrování životopisů (HR Screening)**, "
            "docházelo by k diskriminaci uchazeček o technické a manažerské pozice. "
            "Moderní výzkum proto vyvíjí metody **Debiasingu** a měření férovosti (**Fairness AI**)."
        )

    st.markdown("---")
    st.markdown("##### 🚀 Kde všude se BERT a jeho nástupci uplatňují v praxi?")

    col_app1, col_app2, col_app3 = st.columns(3)
    with col_app1:
        st.markdown(
            """
            * **Text Classification:** Analýza sentimentu recenzí, detekce spamu, klasifikace právních dokumentů.
            * **Named Entity Recognition (NER):** Automatická detekce jmen, firem, adres a rodných čísel v textech.
            """
        )
    with col_app2:
        st.markdown(
            """
            * **Question Answering (SQuAD):** Vyhledání přesné odpovědi v dlouhých manuálech nebo interních znalostních bázích firem.
            * **Sémantické vyhledávání:** Párování významu dotazu a dokumentu místo pouhé shody klíčových slov.
            """
        )
    with col_app3:
        st.markdown(
            """
            * **Sumarizace textů:** Tvorba výtahů ze zpráv, lékařských záznamů nebo finančních reportů.
            * **Doporučovací systémy:** Analýza podobnosti popisků produktů v e-shopech a nabídek pracovních pozic.
            """
        )
