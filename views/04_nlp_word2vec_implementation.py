"""
Den 4: NLP – Teorie: Word2Vec – Implementace v Gensim & Vektorové operace
========================================================================
Syntéza materiálů z Word2Vec_-_sample_implementation.pdf
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.title("📖 Den 4: Word2Vec – Implementace v Gensim & Vektorové operace")
st.caption(
    "Praktické využití Word2Vec v Pythonu pomocí knihovny `gensim`: "
    "Objekt `KeyedVectors`, operace vektorové algebry, odhalení vetřelce (`doesnt_match`), "
    "průměrování vět (`get_mean_vector`) a trénování vlastního modelu na korpusu."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("1. Knihovna", "Gensim 4.4+", delta="Průmyslový standard", delta_color="normal")
c2.metric("2. Hotové vektory", "KeyedVectors", delta="Rychlé načtení z disku", delta_color="normal")
c3.metric("3. Odhalení vetřelce", "doesnt_match()", delta="Sémantická anomálie", delta_color="off")
c4.metric("4. Sentence Vector", "get_mean_vector()", delta="Vektorová reprezentace", delta_color="off")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "🛠️ 1. Gensim & KeyedVectors",
    "🏋️ 2. Trénování vlastního modelu",
    "🧪 3. Interaktivní Gensim Laboratoř (CZ / EN)",
    "🔬 4. Expertní analýza & Best Practices"
])

with tab1:
    st.subheader("1. Knihovna Gensim a práce s předtrénovanými vektory (KeyedVectors)")
    st.markdown(
        """
        Zatímco knihovna `scikit-learn` je králem tradičních tabulkových algoritmů, 
        knihovna **`gensim`** je celosvětovým standardem pro práci s vektorovými modely textu (**Word2Vec**, **FastText**, **Doc2Vec**, **LDA**).
        
        Když pracujeme s hotovými embeddingy (např. z Wikipedie, Google News nebo Twitteru), 
        využíváme třídu **`KeyedVectors`**. Ta ukládá pouze slovník a matici vektorů bez zátěže neuronové sítě.
        """
    )

    st.markdown("#### 7 základních metod objektu `KeyedVectors` ze slidů:")
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.markdown(
            r"""
            1. **`.most_similar(positive, negative, topn)`**  
               * Hledá $n$ nejbližších slov v prostoru podle kosinové podobnosti.
               * Podporuje sémantickou aritmetiku: `positive=['woman', 'king'], negative=['man']` $\rightarrow$ `queen`.
            2. **`.distance(w1, w2)`**  
               * Vrátí kosinovou vzdálenost: $1 - \cos(\mathbf{w}_1, \mathbf{w}_2)$.
            3. **`.distances(word, other_words)`**  
               * Vypočítá vektor vzdáleností zadaného slova ke skupině slov.
            4. **`.get_vector(key)`**  
               * Vrátí numpy pole samotného embeddingu (např. 100 čísel).
            """
        )

    with m_col2:
        st.markdown(
            r"""
            5. **`.doesnt_match(words)`**  
               * Sémantický test: najde slovo, které se do dané skupiny nejméně hodí.
               * Příklad: `['car', 'truck', 'boat', 'bike']` $\rightarrow$ `'boat'` (loď není pozemní vozidlo).
            6. **`.get_mean_vector(keys, ignore_missing=True)`**  
               * Spočítá aritmetický průměr vektorů všech slov dokumentu:
                 $$\mathbf{d} = \frac{1}{|d|} \sum_{w \in d} \mathbf{v}_w$$
               * Slouží pro reprezentaci celých recenzí v klasifikátorech!
            7. **`.add_vector(key, vector)`**  
               * Dynamické přidání nového vlastního slova a vektoru do slovníku.
            """
        )

    st.markdown("---")
    st.markdown("#### Ukázka kódu pro načtení hotových vektorů:")
    st.code(
        r"""
from gensim.models import KeyedVectors
import gensim.downloader as api

# 1. Načtení lokálního souboru
word_vectors = KeyedVectors.load_word2vec_format("word2vec_en.txt", binary=False)

# 2. Stažení populárních Twitter vektorů z repozitáře Gensim
word_vectors_gensim = api.load("glove-twitter-25")

# 3. Výpočet vzdálenosti
dist = word_vectors.distance("city", "nyc")
print("Vzdálenost city - nyc:", dist)
        """,
        language="python"
    )

with tab2:
    st.subheader("2. Jak natrénovat vlastní model Word2Vec na korpusu")
    st.markdown(
        """
        V prezentaci byl model trénován na recenzích aplikace **Threads** (konkurent sítě X od společnosti Meta).
        Při trénování na vlastních datech nepíšeme architekturu sítě ručně – třída **`Word2Vec`** v Gensim má vše zabudované.
        """
    )

    st.markdown("#### Klíčové hyperparametry konstruktoru `Word2Vec()`:")
    param_data = [
        {"Parametr": "sentences", "Typ": "iterable of iterables", "Výchozí": "povinný", "Význam": "Tokenizovaný korpus jako seznam seznamů slov (list of lists) nebo LineSentence."},
        {"Parametr": "vector_size", "Typ": "int", "Výchozí": "100", "Význam": "Dimenze výsledného embeddingu (typicky 100 až 300). Větší data snesou vyšší dimenzi."},
        {"Parametr": "window", "Typ": "int", "Výchozí": "5", "Význam": "Velikost kontextového okna – počet slov vlevo a vpravo od centrálního slova."},
        {"Parametr": "min_count", "Typ": "int", "Výchozí": "5", "Význam": "Minimální počet výskytů slova. Vzácnější slova pod tímto prahem jsou zahozena jako šum."},
        {"Parametr": "sg", "Typ": "int (0 nebo 1)", "Výchozí": "0", "Význam": "0 = CBOW (rychlejší trénování), 1 = Skip-gram (lepší reprezentace vzácných slov)."},
        {"Parametr": "workers", "Typ": "int", "Výchozí": "3", "Význam": "Počet paralelních CPU vláken pro trénování (vyžaduje knihovnu Cython)."},
        {"Parametr": "epochs", "Typ": "int", "Výchozí": "5", "Význam": "Počet epoch (průchodů celým textovým korpusem). Pro menší data se volí 20 až 30."},
        {"Parametr": "alpha", "Typ": "float", "Výchozí": "0.025", "Význam": "Počáteční rychlost učení (learning rate), která během epoch postupně klesá."}
    ]
    st.dataframe(pd.DataFrame(param_data), hide_index=True, width="stretch")

    st.markdown("---")
    st.markdown("#### Ukázka trénování a uložení modelu ze slidů:")
    st.code(
        r"""
from gensim.models import Word2Vec

# 1. Trénování modelu na recenzích
model = Word2Vec(
    reviews,
    vector_size=200,
    window=2,
    min_count=3,
    workers=3,
    sg=0,
    epochs=20
)

# 2. Hledání nejpodobnějšího slova k Facebook
print(model.wv.most_similar("facebook"))

# 3. Měření sémantické podobnosti
similarity = model.wv.similarity("instagram", "threads")
print("Podobnost Instagram a Threads:", similarity)  # cca 0.627

# 4. Uložení a načtení celého modelu
model.save("word2vec.model")
loaded_model = Word2Vec.load("word2vec.model")

# 5. Online dotrénování (Fine-tuning)
loaded_model.train([["threads", "update", "is", "great"]], total_examples=1, epochs=1)
        """,
        language="python"
    )

with tab3:
    st.subheader("3. Interaktivní Gensim Laboratoř (CZ / EN standard)")
    st.markdown(
        "Vyzkoušejte si všechny klíčové metody knihovny `gensim` na předem připraveném sémantickém prostoru. "
        "Podporujeme jak **český jazyk**, tak **anglický jazyk**."
    )

    lang_choice = st.radio("Zvolte jazykový režim laboratoře:", ["🇨🇿 Čeština", "🇬🇧 Angličtina"], horizontal=True)

    # Definice sémantické báze pro simulaci Gensim metod
    if "Čeština" in lang_choice:
        demo_dict = {
            # Dopravní prostředky
            "auto": np.array([0.90, 0.10, 0.15, 0.20]),
            "kamion": np.array([0.88, 0.12, 0.25, 0.30]),
            "motorka": np.array([0.85, 0.08, 0.10, 0.15]),
            "vlak": np.array([0.80, 0.35, 0.20, 0.40]),
            "loď": np.array([0.30, 0.90, 0.50, 0.40]),
            # Královská rodina
            "král": np.array([0.15, 0.85, 0.90, 0.20]),
            "královna": np.array([0.14, 0.84, 0.92, 0.88]),
            "muž": np.array([0.12, 0.40, 0.15, 0.20]),
            "žena": np.array([0.11, 0.38, 0.16, 0.85]),
            # Technologie
            "telefon": np.array([-0.70, 0.20, 0.30, 0.10]),
            "počítač": np.array([-0.75, 0.25, 0.35, 0.15]),
            "aplikace": np.array([-0.80, 0.30, 0.40, 0.12]),
            "internet": np.array([-0.85, 0.35, 0.45, 0.18]),
            # Jídlo
            "jablko": np.array([0.10, -0.80, 0.10, 0.15]),
            "banán": np.array([0.12, -0.82, 0.12, 0.18]),
            "pizza": np.array([0.15, -0.75, 0.20, 0.25])
        }
    else:
        demo_dict = {
            # Vehicles
            "car": np.array([0.90, 0.10, 0.15, 0.20]),
            "truck": np.array([0.88, 0.12, 0.25, 0.30]),
            "bike": np.array([0.85, 0.08, 0.10, 0.15]),
            "train": np.array([0.80, 0.35, 0.20, 0.40]),
            "boat": np.array([0.30, 0.90, 0.50, 0.40]),
            # Royalty
            "king": np.array([0.15, 0.85, 0.90, 0.20]),
            "queen": np.array([0.14, 0.84, 0.92, 0.88]),
            "man": np.array([0.12, 0.40, 0.15, 0.20]),
            "woman": np.array([0.11, 0.38, 0.16, 0.85]),
            # Technology
            "phone": np.array([-0.70, 0.20, 0.30, 0.10]),
            "computer": np.array([-0.75, 0.25, 0.35, 0.15]),
            "application": np.array([-0.80, 0.30, 0.40, 0.12]),
            "internet": np.array([-0.85, 0.35, 0.45, 0.18]),
            # Food
            "apple": np.array([0.10, -0.80, 0.10, 0.15]),
            "banana": np.array([0.12, -0.82, 0.12, 0.18]),
            "pizza": np.array([0.15, -0.75, 0.20, 0.25])
        }

    def calc_cos_sim(v1, v2):
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    # Výběr operace
    op_choice = st.selectbox(
        "Vyberte metodu Gensim k otestování:",
        [
            "1. .doesnt_match() – Odhalení vetřelce ve skupině slov",
            "2. .distance() – Kosinová vzdálenost mezi dvěma slovy",
            "3. .most_similar() – Hledání nejpodobnějších slov & Analogie",
            "4. .get_mean_vector() – Výpočet průměrného vektoru věty (Sentence Embedding)"
        ]
    )

    if "doesnt_match" in op_choice:
        st.markdown("##### 🕵️ Odhalení vetřelce (`.doesnt_match`):")
        st.caption("Metoda spočítá průměrný vektor skupiny slov a vrátí slovo, které má k tomuto průměru největší kosinovou vzdálenost.")

        if "Čeština" in lang_choice:
            sample_groups = {
                "Dopravní prostředky (auto, kamion, motorka, loď)": ["auto", "kamion", "motorka", "loď"],
                "Království a jídlo (král, královna, pizza)": ["král", "královna", "pizza"],
                "Technologie a ovoce (počítač, telefon, internet, banán)": ["počítač", "telefon", "internet", "banán"]
            }
        else:
            sample_groups = {
                "Vehicles (car, truck, bike, boat) [ze slidu 12]": ["car", "truck", "bike", "boat"],
                "Royalty and food (king, queen, pizza)": ["king", "queen", "pizza"],
                "Tech and fruit (computer, phone, internet, banana)": ["computer", "phone", "internet", "banana"]
            }

        sel_group_name = st.selectbox("Vyberte testovací skupinu slov:", list(sample_groups.keys()))
        words_in_group = sample_groups[sel_group_name]

        # Výpočet centroidu a odhalení vetřelce
        vecs = [demo_dict[w] for w in words_in_group]
        mean_v = np.mean(vecs, axis=0)
        dists = {w: 1.0 - calc_cos_sim(demo_dict[w], mean_v) for w in words_in_group}
        outlier = max(dists.items(), key=lambda x: x[1])[0]

        col_out1, col_out2 = st.columns([1, 2])
        with col_out1:
            st.error(f"### 🚨 Vetřelec: `{outlier}`")
            st.caption(f"Vzdálenost od středu skupiny: {dists[outlier]:.4f}")
        with col_out2:
            st.markdown("Vzdálenost jednotlivých slov k průměrnému vektoru skupiny:")
            df_dists = pd.DataFrame(list(dists.items()), columns=["Slovo", "Kosinová vzdálenost"]).sort_values("Kosinová vzdálenost", ascending=False)
            st.dataframe(df_dists, hide_index=True, width="stretch")

    elif "distance" in op_choice:
        st.markdown("##### 📏 Kosinová vzdálenost (`.distance`):")
        w_keys = list(demo_dict.keys())
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            word_a = st.selectbox("Slovo A:", w_keys, index=0)
        with col_w2:
            word_b = st.selectbox("Slovo B:", w_keys, index=1)

        sim = calc_cos_sim(demo_dict[word_a], demo_dict[word_b])
        dist = 1.0 - sim

        st.metric(f"Vzdálenost mezi '{word_a}' a '{word_b}'", f"{dist:.4f}", delta=f"Podobnost: {sim*100:.1f} %")
        st.progress(float(max(0.0, min(1.0, sim))), text=f"Sémantické překrytí: {sim:.3f}")

    elif "most_similar" in op_choice:
        st.markdown("##### 🔍 Hledání nejpodobnějších slov (`.most_similar`):")
        w_keys = list(demo_dict.keys())
        target_w = st.selectbox("Zvolte cílové slovo:", w_keys, index=0)

        sims = {w: calc_cos_sim(demo_dict[target_w], demo_dict[w]) for w in w_keys if w != target_w}
        sorted_sims = sorted(sims.items(), key=lambda x: x[1], reverse=True)[:5]

        df_sims = pd.DataFrame(sorted_sims, columns=["Podobné slovo", "Kosinová podobnost"])
        st.dataframe(df_sims, hide_index=True, width="stretch")

    elif "get_mean_vector" in op_choice:
        st.markdown("##### 📄 Výpočet průměrného vektoru věty (`.get_mean_vector`):")
        st.caption("Aritmetický průměr vektorů všech slov reprezentuje celý dokument v jednotném formátu pro klasifikátor.")

        sample_sent = "král a královna" if "Čeština" in lang_choice else "king and queen"
        user_sent = st.text_input("Zadejte slova oddělená mezerou:", value=sample_sent)

        tokens = [w.strip().lower() for w in user_sent.split() if w.strip().lower() in demo_dict]
        if tokens:
            vec_list = [demo_dict[t] for t in tokens]
            mean_vec = np.mean(vec_list, axis=0)

            st.success(f"Zpracováno **{len(tokens)}** platných slov: `{tokens}`")
            st.markdown(f"**Výsledný vektor dokumentu (dimenze {len(mean_vec)}):**")
            st.code(str(mean_vec.round(4)), language="text")
        else:
            st.warning("Žádné ze zadaných slov se nenachází v testovacím slovníku.")

with tab4:
    st.subheader("4. Expertní analýza & Best Practices v produkci")
    st.markdown(
        """
        Kdy v praxi trénovat vlastní Word2Vec a kdy raději sáhnout po předtrénovaném modelu nebo moderním transformeru?
        """
    )

    c_bp1, c_bp2 = st.columns(2)
    with c_bp1:
        st.markdown("##### 🏆 Kdy se Word2Vec stále vyplatí (i v roce 2026):")
        st.markdown(
            """
            * **Specifický doménový slang:** Lékařské zprávy, právní smlouvy, interní logy serverů nebo kódy produktů v e-shopu. Zde obecné modely selhávají a lehký Word2Vec natrénovaný za pár sekund odvede skvělou práci.
            * **Extrémní rychlost a nízká latence:** Potřebujete vyhledávat podobné produkty nebo články za méně než **1 milisekundu**.
            * **Minimální nároky na hardware:** Word2Vec nevyžaduje GPU a běží spolehlivě i na malém jednojádrovém CPU.
            """
        )

    with c_bp2:
        st.markdown("##### ⚠️ Klíčová úskalí a limity v produkci:")
        st.markdown(
            """
            * **Jednoduché průměrování (`get_mean_vector`):** Pokud uděláte průměr slov dlouhé recenze (300 slov), vektor se rozmělní a ztratí specifický význam (tzv. * смысловая энтропия / semantic drowning*).
            * **Řešení:** Vážený průměr pomocí TF-IDF vah (slova s vysokým IDF mají při průměrování větší váhu než běžná slova).
            * **Paměťová optimalizace:** Vždy ukládat hotové vektory přes `model.wv.save()` a načítat přes `KeyedVectors.load(..., mmap='r')` – to umožní sdílet vektory mezi více procesy bez duplikace v RAM.
            """
        )
