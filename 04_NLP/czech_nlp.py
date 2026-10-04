"""
Czech NLP Engine (Český modul pro zpracování přirozeného jazyka)
================================================================
Podpora českého jazyka pro Den 4:
1. České stop-slova (spaCy blank('cs') + rozšířený lingvistický seznam).
2. Ochrana české negace (slova jako 'ne', 'ani', 'nikdy', 'žádný' a předpony 'ne-').
3. Český kmenovač (Savoy / Dolamic stemmer).
4. Český lemmatizátor (morfologický slovník nepravidelných tvarů + pravidlová deklinace).
5. Analýza diakritiky (Unicode vs. unidecode).
"""

import re
from typing import Set, Tuple, List, Dict

# 1. ČESKÁ STOP-SLOVA
CZECH_STOPWORDS: Set[str] = {
    "a", "aby", "aj", "ale", "ani", "anibo", "anebo", "ano", "asi", "aspoň", "až",
    "bez", "beze", "beztoho", "bude", "budem", "budeme", "budeš", "budete", "budou", "budu",
    "by", "byl", "byla", "byli", "bylo", "byly", "bys", "byste", "být",
    "co", "což", "či", "článek", "další", "dále", "děkovat", "děkujeme", "dnes", "do", "dva",
    "ho", "i", "já", "jak", "jako", "jaký", "je", "jeden", "jedna", "jedno", "jednou",
    "jeho", "její", "jejich", "jemu", "jen", "jenom", "ještě", "jestli", "jestliže",
    "ji", "jí", "jich", "jím", "jimi", "jinak", "jiné", "jiný", "jsem", "jsi", "jsme", "jsou", "jste",
    "k", "kam", "každý", "kde", "kdo", "kdy", "když", "ke", "kolik", "kromě", "která", "které", "kterou", "který", "kteří", "ku", "kvůli",
    "má", "mají", "máme", "mám", "máš", "máte", "mezi", "mě", "mne", "mi", "mně", "mnou",
    "moc", "mohl", "mohla", "mohli", "mohlo", "mohou", "může", "můžeme", "můžete", "můžu", "můj", "moje", "mou",
    "na", "nad", "nade", "nám", "námi", "nás", "náš", "naše", "naši", "našich", "našim",
    "nebo", "něco", "nějak", "někdo", "někde", "někdy", "než", "nic", "ní", "ním", "nimi",
    "nová", "nové", "nový", "nyní",
    "o", "od", "ode", "on", "ona", "oni", "ono", "ony", "opět",
    "pak", "po", "pod", "pode", "podle", "pokud", "pouze", "pozdě", "před", "přede", "přes", "při", "pro", "proč", "proto", "protože", "první",
    "s", "se", "sem", "si", "sice", "skoro", "své", "svůj", "svá", "sví", "svou", "svých", "svým", "svými",
    "ta", "tady", "tak", "také", "takhle", "taky", "tam", "tamhle", "tamto", "tato", "tebe", "tebou", "teď", "tedy",
    "ten", "tento", "tě", "ti", "tím", "tímto", "to", "tobě", "tohle", "toho", "tohoto", "tom", "tomto", "tomu", "tomuto", "toto", "tu", "tuto", "tvůj", "tvá", "tvé", "ty", "tyto",
    "u", "už",
    "v", "ve", "vám", "vámi", "vás", "váš", "vaše", "vaši", "vašich", "ve", "vedle", "více", "vlastně", "však", "všechno", "všichni", "vůbec", "vy",
    "z", "za", "zda", "zde", "ze", "znovu", "zpět", "zase", "že"
}

# České negace (slova a tvary, které vyjadřují zápor)
CZECH_NEGATIONS: Set[str] = {
    "ne", "ani", "nikdy", "žádný", "žádná", "žádné", "nikdo", "nikam", "nijak", "vůbec",
    "není", "nejsou", "nebyl", "nebyla", "nebylo", "nebyli", "nebyly", "nebude", "nebudou",
    "nemám", "nemáš", "nemá", "nemáme", "nemáte", "nemají", "neměl", "neměla", "neměli",
    "nemohu", "nemůže", "nemůžeme", "nemůžete", "nemohou", "nechci", "nechce", "nedoporučuji", "nechutná"
}

# 2. ČESKÝ KMENOVAČ (Dolamic / Savoy Stemmer)
def czech_stem(word: str, aggressive: bool = False) -> str:
    """
    Dolamic / Savoy rule-based stemmer pro češtinu.
    Odstraňuje pádové koncovky, plurály a deklinační přípony.
    """
    w = word.lower()
    if len(w) <= 3:
        return w

    # Palatalizační korekce (změna souhlásek před e/i)
    def fix_palatalization(s: str) -> str:
        if len(s) >= 2:
            if s.endswith("ci") or s.endswith("ce"):
                return s[:-2] + "k"
            elif s.endswith("zi") or s.endswith("ze"):
                return s[:-2] + "h"
            elif s.endswith("si") or s.endswith("se"):
                return s[:-2] + "ch"
            elif s.endswith("či") or s.endswith("če"):
                return s[:-2] + "k"
            elif s.endswith("ži") or s.endswith("že"):
                return s[:-2] + "h"
            elif s.endswith("ši") or s.endswith("še"):
                return s[:-2] + "ch"
            elif s.endswith("tě"):
                return s[:-2] + "t"
            elif s.endswith("dě"):
                return s[:-2] + "d"
            elif s.endswith("ně"):
                return s[:-2] + "n"
        return s

    # Pádové koncovky
    suffixes_case = [
        "atech", "atům", "atou", "atami", "atama", "ata", "atem", "atu", "aty", "at",
        "oušcích", "ouškům", "ouškami", "ouškem", "oušků", "oušky", "oušek", "ouška", "ouško",
        "iště", "išti", "ištích", "ištím", "ištěm", "išt",
        "ovi", "ové", "ovy", "ech", "ích", "ých", "ými", "emi", "ami", "ama",
        "ovi", "em", "ím", "ám", "ům", "ou",
        "om", "em", "ům", "am", "ím",
        "e", "i", "í", "y", "u", "ú", "o", "a", "á", "ě"
    ]
    for suf in suffixes_case:
        if len(w) - len(suf) >= 3 and w.endswith(suf):
            w = w[:-len(suf)]
            w = fix_palatalization(w)
            break

    # Přípony deminutiv
    if aggressive:
        diminutives = ["oušek", "eček", "iček", "ička", "enka", "inka", "átko", "isko"]
        for suf in diminutives:
            if len(w) - len(suf) >= 3 and w.endswith(suf):
                w = w[:-len(suf)]
                w = fix_palatalization(w)
                break

    return w


# 3. ČESKÝ LEMMATIZÁTOR
# Nepravidelná a vysokofrekvenční lemmata
CZECH_LEMMAS: Dict[str, str] = {
    # Sloveso být
    "jsem": "být", "jsi": "být", "je": "být", "jsme": "být", "jste": "být", "jsou": "být",
    "byl": "být", "byla": "být", "bylo": "být", "byli": "být", "byly": "být",
    "budu": "být", "budeš": "být", "bude": "být", "budeme": "být", "budete": "být", "budou": "být",
    "není": "nebýt", "nejsou": "nebýt", "nebyl": "nebýt", "nebyla": "nebýt", "nebylo": "nebýt", "nebyli": "nebýt", "nebude": "nebýt",
    # Mít
    "mám": "mít", "máš": "mít", "má": "mít", "máme": "mít", "máte": "mít", "mají": "mít",
    "měl": "mít", "měla": "mít", "mělo": "mít", "měli": "mít", "měly": "mít",
    "nemám": "nemít", "nemáš": "nemít", "nemá": "nemít", "neměl": "nemít",
    # Jít / jet
    "jdu": "jít", "jdeš": "jít", "jde": "jít", "jdeme": "jít", "jdete": "jít", "jdou": "jít",
    "šel": "jít", "šla": "jít", "šlo": "jít", "šli": "jít", "šly": "jít",
    "jedu": "jet", "jedeš": "jet", "jede": "jet", "jedeme": "jet", "jedou": "jet", "jel": "jet", "jela": "jet", "jeli": "jet",
    # Člověk / lidé
    "lidé": "člověk", "lidí": "člověk", "lidem": "člověk", "lidi": "člověk", "lidmi": "člověk",
    "člověka": "člověk", "člověku": "člověk", "člověkem": "člověk", "člověče": "člověk",
    # Dítě
    "děti": "dítě", "dětí": "dítě", "dětem": "dítě", "dětmi": "dítě", "dítěte": "dítě", "dítěti": "dítě",
    # Stupňování přídavných jmen
    "dobrý": "dobrý", "lepší": "dobrý", "nejlepší": "dobrý", "dobrého": "dobrý", "dobrém": "dobrý", "dobrému": "dobrý", "dobří": "dobrý", "dobré": "dobrý",
    "špatný": "špatný", "horší": "špatný", "nejhorší": "špatný", "špatného": "špatný", "špatnému": "špatný",
    "velký": "velký", "větší": "velký", "největší": "velký", "velkého": "velký", "velkému": "velký",
    "malý": "malý", "menší": "malý", "nejmenší": "malý", "malého": "malý", "malému": "malý",
    # Vybraná substantiva (vzory kůň, pes, město, hrad, kniha, kočka)
    "koně": "kůň", "koní": "kůň", "koni": "kůň", "koněm": "kůň", "koních": "kůň", "koňmi": "kůň",
    "psi": "pes", "psa": "pes", "psu": "pes", "psem": "pes", "psů": "pes", "psům": "pes", "psech": "pes",
    "kočky": "kočka", "kočce": "kočka", "kočku": "kočka", "kočkou": "kočka", "koček": "kočka", "kočkám": "kočka", "kočkách": "kočka", "kočkami": "kočka",
    "města": "město", "městu": "město", "městem": "město", "městě": "město", "měst": "město", "městům": "město", "městech": "město", "městy": "město",
    "hrady": "hrad", "hradu": "hrad", "hradem": "hrad", "hradě": "hrad", "hradů": "hrad", "hradům": "hrad", "hradech": "hrad",
    "knihy": "kniha", "knize": "kniha", "knihu": "kniha", "knihou": "kniha", "knih": "kniha", "knihám": "kniha", "knihách": "kniha", "knihami": "kniha",
    "ženy": "žena", "ženě": "žena", "ženu": "žena", "ženou": "žena", "žen": "žena", "ženám": "žena", "ženách": "žena", "ženami": "žena",
    "muži": "muž", "muže": "muž", "mužem": "muž", "mužů": "muž", "mužům": "muž", "mužích": "muž", "muži": "muž",
    "písně": "píseň", "písni": "píseň", "písní": "píseň", "písním": "píseň", "písněmi": "píseň",
    "kosti": "kost", "kostem": "kost", "kostí": "kost", "kostmi": "kost",
    # Časové pojmy
    "večera": "večer", "večeru": "večer", "večerem": "večer", "večery": "večer", "večerů": "večer",
    "rána": "ráno", "ránu": "ráno", "ránem": "ráno", "ránech": "ráno",
    "dny": "den", "dne": "den", "dnu": "den", "dni": "den", "dnem": "den", "dnů": "den", "dnech": "den", "dny": "den",
    "roky": "rok", "roku": "rok", "roce": "rok", "rokem": "rok", "let": "rok", "letům": "rok", "letech": "rok", "lety": "rok"
}

def czech_lemmatize(word: str) -> str:
    """
    Převede české slovo na slovníkový základ (lemma):
    1. Přímý lookup v nepravidelném morfologickém slovníku.
    2. Pravidlová normalizace sloves minulého času (dělal/dělali -> dělat).
    3. Normalizace adjektivních koncovek (-ého, -ému, -ém, -ých -> -ý).
    4. Substantivní deklinace.
    """
    w = word.lower()
    if w in CZECH_LEMMAS:
        return CZECH_LEMMAS[w]

    # Slovesné tvary l-ového participia (minulý čas)
    if len(w) >= 5:
        if w.endswith("ali") or w.endswith("ala") or w.endswith("alo") or w.endswith("aly"):
            return w[:-3] + "at"
        elif w.endswith("al"):
            return w[:-2] + "at"
        elif w.endswith("ěli") or w.endswith("ěla") or w.endswith("ělo") or w.endswith("ěly"):
            return w[:-3] + "et"
        elif w.endswith("ěl"):
            return w[:-2] + "et"
        elif w.endswith("ili") or w.endswith("ila") or w.endswith("ilo") or w.endswith("ily"):
            return w[:-3] + "it"
        elif w.endswith("il"):
            return w[:-2] + "it"
        elif w.endswith("ovali") or w.endswith("ovala") or w.endswith("ovalo") or w.endswith("ovaly"):
            return w[:-4] + "ovat"
        elif w.endswith("oval"):
            return w[:-3] + "ovat"

    # Přídavná jména (skloňování složeného adjektiva)
    adj_endings = ["ějšího", "ějšímu", "ějším", "ější", "ého", "ému", "ém", "ých", "ým", "ými", "ou"]
    for ae in adj_endings:
        if len(w) - len(ae) >= 3 and w.endswith(ae):
            base = w[:-len(ae)]
            return base + "ý"

    # Běžné koncovky podstatných jmen
    if w.endswith("ech") and len(w) >= 5:
        return w[:-3]
    if w.endswith("ům") and len(w) >= 4:
        return w[:-2]
    if w.endswith("ích") and len(w) >= 5:
        return w[:-3]

    return w


def is_czech_negation(word: str) -> bool:
    """Detekuje, zda slovo nese zápor v češtině."""
    w = word.lower()
    if w in CZECH_NEGATIONS:
        return True
    # České slovesné a adjektivní negace s předponou ne-
    if w.startswith("ne") and len(w) >= 5 and w not in {"nebe", "nebo", "něco", "někdo", "někdy", "někde", "nehet", "nerv", "nervy"}:
        return True
    return False


# 4. ČESKÝ MODEL SENTIMENTU (Trénovaný BoW + Bigramy + Logistická regrese)
CZECH_SENTIMENT_CORPUS = [
    # Pozitivní recenze (třída 1)
    ("Tento film byl naprosto skvělý a herci hráli výborně.", 1),
    ("Úžasný zážitek, skvělá režie a vynikající hudba.", 1),
    ("Perfektní filmové dílo, moc se mi to líbilo a vřele doporučuji všem.", 1),
    ("Geniální scénář a úchvatné herecké výkony, bavil jsem se celou dobu.", 1),
    ("Skvělý film, nádherná atmosféra, silné emoce a dojemný příběh.", 1),
    ("Jedno z nejlepších děl, které jsem viděl, bezchybné a bravurní.", 1),
    ("Vynikající kamera, skvělé dialogy a velmi poutavý děj.", 1),
    ("Tento snímek je naprostá pecka, vtipné, chytré a energické.", 1),
    ("Velmi kvalitní zpracování, režisér odvedl špičkovou práci a herci zazářili.", 1),
    ("Krásný a hluboký film, který rozhodně stojí za zhlédnutí, doporučuji.", 1),
    ("Excelentní podívaná s výborným inteligentním humorem a skvělým obsazením.", 1),
    ("Srdcovka, moc se mi to líbilo a skvěle jsem si odpočinul.", 1),
    ("Opravdu dojemný a krásný příběh s fantastickou atmosférou.", 1),
    ("Naprostá paráda, originální nápad a perfektní provedení od začátku do konce.", 1),
    ("Byl to vynikající film, herci byli skvělí a tempo bylo strhující.", 1),
    ("Líbilo se mi to moc, doporučuji do kina, stojí za to.", 1),
    ("Nádherná podívaná, film mě nadchnul a bavil.", 1),
    ("Vtipná komedie, smál jsem se nahlas, super herci.", 1),
    ("Povedený film, vizuálně nádherné a hudba byla kouzelná.", 1),
    ("Skvěle zahrané postavy a promyšlený konec, který mě potěšil.", 1),
    ("Nebyla to žádná nuda, naopak skvělá jízda plná akce a vtipu.", 1),
    ("Rozhodně nelituji peněz za lístek, byl to parádní zážitek.", 1),
    ("Nebylo to vůbec špatné, příjemně mě to překvapilo a herci byli fajn.", 1),
    ("Velmi dobrý film, který mohu s klidným svědomím doporučit.", 1),
    ("Kvalitní dílo plné skvělých myšlenek a skvělých scén.", 1),

    # Negativní recenze (třída 0)
    ("Tento film byl naprostý odpad a hrozná nuda.", 0),
    ("Příšerný scénář, trapné dialogy a nekoukatelné herecké výkony.", 0),
    ("Ztráta času, vůbec se mi to nelíbilo a nedoporučuji to nikomu.", 0),
    ("Katastrofální režie, nudný děj a celkově obrovské zklamání.", 0),
    ("Bídné zpracování, amatérské výkony a neuvěřitelně špatný konec.", 0),
    ("Naprostý propadák, klišé na každém kroku a otřesná hudba.", 0),
    ("Tento film je neskutečná hloupost a nuda, málem jsem v kině usnul.", 0),
    ("Hrozné zklamání, čekal jsem kvalitu a dostal jsem nefunkční chaos.", 0),
    ("Trapný humor, nesympatičtí herci a hloupý příběh bez kapky smyslu.", 0),
    ("Vůbec to nefunguje, slabý scénář, příšerný střih a zmatek.", 0),
    ("Tohle se opravdu nepovedlo, bída a nesnesitelné utrpení sledovat.", 0),
    ("Rozhodně nedoporučuji, zbytečný film a vyhozené peníze za vstupné.", 0),
    ("Nudné, zdlouhavé a nezajímavé od začátku až do úplného konce.", 0),
    ("Příšerná slátanina bez špetky originality, škoda času.", 0),
    ("Špatný film, herci nepředvedli vůbec nic a režie zklamala na celé čáře.", 0),
    ("Tento film nebyl vůbec dobrý a herci nepředvedli žádný výkon.", 0),
    ("Film neměl žádný děj a vůbec mě nebavil, je to hrozné.", 0),
    ("Bohužel velké zklamání, trapné a nudné scény bez jakéhokoliv nápadu.", 0),
    ("Ubohé dílo, nesmyslný scénář a příšerná kamera.", 0),
    ("Vůbec se mi to nelíbilo, film mě nudil a znechutil.", 0),
    ("Totální propadák, herci hráli hrozně a hudba rvala uši.", 0),
    ("Otřesný film, vyhozené peníze a ztracený večer, nedoporučuji.", 0),
    ("Nekoukatelné, trapné a nesmírně nudné od první minuty.", 0),
    ("Žádná zábava, jenom prázdný a zbytečný film bez duše.", 0),
    ("Zklamalo mě to na plné čáře, horší film jsem dlouho neviděl.", 0)
]


_CZECH_SENTIMENT_MODEL = None
_CZECH_VECTORIZER = None


def get_czech_sentiment_model():
    """Vrátí natrénovaný CountVectorizer a LogisticRegression model pro český sentiment."""
    global _CZECH_SENTIMENT_MODEL, _CZECH_VECTORIZER
    if _CZECH_SENTIMENT_MODEL is None:
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.linear_model import LogisticRegression

        texts = [item[0] for item in CZECH_SENTIMENT_CORPUS]
        labels = [item[1] for item in CZECH_SENTIMENT_CORPUS]

        # Využijeme unigramy i bigramy a Unicode pattern pro češtinu
        _CZECH_VECTORIZER = CountVectorizer(
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b[A-Za-zÁ-ž0-9_]+\b",
            lowercase=True
        )
        X = _CZECH_VECTORIZER.fit_transform(texts)
        _CZECH_SENTIMENT_MODEL = LogisticRegression(C=2.0, max_iter=1000, random_state=42)
        _CZECH_SENTIMENT_MODEL.fit(X, labels)

    return _CZECH_VECTORIZER, _CZECH_SENTIMENT_MODEL


def predict_czech_sentiment(text: str) -> Dict:
    """
    Predikuje sentiment českého textu a vrátí detailní rozbor slov a vah:
    - class: 1 (Pozitivní) nebo 0 (Negativní)
    - prob_positive: float
    - prob_negative: float
    - contributions: list slov/bigramů a jejich koeficientů
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "predicted_class": 0,
            "sentiment_label": "Neurčeno",
            "prob_positive": 0.5,
            "prob_negative": 0.5,
            "contributions": []
        }

    vec, model = get_czech_sentiment_model()
    x_vec = vec.transform([text])
    pred = int(model.predict(x_vec)[0])
    probs = model.predict_proba(x_vec)[0]

    feature_names = vec.get_feature_names_out()
    coefs = model.coef_[0]
    vocab_map = {term: idx for idx, term in enumerate(feature_names)}

    # Analýza unigramů i bigramů obsažených v textu
    words_raw = re.findall(r"\b[A-Za-zÁ-ž0-9_]+\b", text.lower())
    contributions = []

    # Bigramy
    for i in range(len(words_raw) - 1):
        bg = f"{words_raw[i]} {words_raw[i+1]}"
        if bg in vocab_map:
            c = coefs[vocab_map[bg]]
            contributions.append({
                "Termín": bg,
                "Typ": "Bigram (Spojení)",
                "Váha (Koeficient)": round(float(c), 4),
                "Směr": "🟢 Pozitivní" if c > 0 else "🔴 Negativní"
            })

    # Unigramy
    for w in words_raw:
        if w in vocab_map:
            c = coefs[vocab_map[w]]
            contributions.append({
                "Termín": w,
                "Typ": "Unigram (Slovo)",
                "Váha (Koeficient)": round(float(c), 4),
                "Směr": "🟢 Pozitivní" if c > 0 else "🔴 Negativní"
            })

    # Deduplikace
    seen = set()
    dedup_contrib = []
    for item in contributions:
        if item["Termín"] not in seen:
            seen.add(item["Termín"])
            dedup_contrib.append(item)

    dedup_contrib.sort(key=lambda x: abs(x["Váha (Koeficient)"]), reverse=True)

    return {
        "predicted_class": pred,
        "sentiment_label": "Pozitivní" if pred == 1 else "Negativní",
        "prob_positive": float(probs[1]),
        "prob_negative": float(probs[0]),
        "contributions": dedup_contrib
    }

