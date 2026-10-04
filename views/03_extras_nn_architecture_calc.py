"""
Den 3 Extras: 📐 Architekt & kalkulačka parametrů neuronové sítě (Shape & Param Counter)
========================================================================================
- Interaktivní návrhář architektury neuronové sítě (Keras style).
- Podpora vrstev: Dense, Conv2D, MaxPooling2D, Flatten, Dropout, GlobalAveragePooling2D.
- Výpočet výstupních rozměrů tensoru (Shape Propagation) pro každou vrstvu.
- Exaktní výpočet počtu trénovatelných parametrů (vah + biasů) a paměťové náročnosti.
- Detekce architektonických chyb (např. chybějící Flatten před Dense, rozpad dimenze pod 1x1).
- Předpřipravené šablony: MNIST LeNet CNN, KC Housing Deep MLP, Mini-VGG.
"""

from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Architekt & kalkulačka parametrů NN", page_icon="📐", layout="wide")

st.markdown("""
# 📐 Architekt & kalkulačka parametrů neuronové sítě
### Interaktivní návrh vrstev, propagace rozměrů (tensor shapes) a počítadlo vah
""")

st.info("""
Při stavbě neuronových sítí v Kerasu je nejčastějším zdrojem chyb **nesoulad dimenzí (Shape Mismatch)** 
a **exploze počtu parametrů** při přechodu z konvolučních vrstev do plně propojených (Dense). 
Tento nástroj ti umožní bezpečně navrhnout síť, spočítat každý parametr a zkontrolovat paměťovou náročnost.
""")

# --- ŠABLONY ARCHITEKTUR ---
TEMPLATES = {
    "Vlastní prázdná síť": {
        "input_shape": (28, 28, 1),
        "layers": [
            {"type": "Flatten"},
            {"type": "Dense", "units": 128, "activation": "relu", "use_bias": True},
            {"type": "Dense", "units": 10, "activation": "softmax", "use_bias": True},
        ],
    },
    "Cvičení 1: MNIST Jednoduchý MLP (784 -> 128 -> 10)": {
        "input_shape": (28, 28, 1),
        "layers": [
            {"type": "Flatten"},
            {"type": "Dense", "units": 128, "activation": "relu", "use_bias": True},
            {"type": "Dense", "units": 10, "activation": "softmax", "use_bias": True},
        ],
    },
    "Cvičení 2: KC Housing Regresní MLP (18 -> 128 -> 64 -> 1)": {
        "input_shape": (18,),
        "layers": [
            {"type": "Dense", "units": 128, "activation": "relu", "use_bias": True},
            {"type": "Dropout", "rate": 0.2},
            {"type": "Dense", "units": 64, "activation": "relu", "use_bias": True},
            {"type": "Dense", "units": 1, "activation": "linear", "use_bias": True},
        ],
    },
    "Klasická konvoluční síť: LeNet-5 na MNIST": {
        "input_shape": (28, 28, 1),
        "layers": [
            {"type": "Conv2D", "filters": 6, "kernel_size": 5, "padding": "same", "activation": "relu", "use_bias": True},
            {"type": "MaxPooling2D", "pool_size": 2},
            {"type": "Conv2D", "filters": 16, "kernel_size": 5, "padding": "valid", "activation": "relu", "use_bias": True},
            {"type": "MaxPooling2D", "pool_size": 2},
            {"type": "Flatten"},
            {"type": "Dense", "units": 120, "activation": "relu", "use_bias": True},
            {"type": "Dense", "units": 84, "activation": "relu", "use_bias": True},
            {"type": "Dense", "units": 10, "activation": "softmax", "use_bias": True},
        ],
    },
}

# --- Inicializace stavu ---
if "nn_tpl_choice" not in st.session_state:
    st.session_state.nn_tpl_choice = "Cvičení 1: MNIST Jednoduchý MLP (784 -> 128 -> 10)"
if "nn_layers" not in st.session_state:
    st.session_state.nn_layers = [dict(l) for l in TEMPLATES[st.session_state.nn_tpl_choice]["layers"]]
if "nn_input_shape" not in st.session_state:
    st.session_state.nn_input_shape = TEMPLATES[st.session_state.nn_tpl_choice]["input_shape"]

# Ovládací prvky v hlavní ploše (nezasahuje do navigace v levém docku)
with st.expander("⚙️ Šablony architektur & Vstupní rozměry (Input Shape)", expanded=True):
    col_t1, col_t2 = st.columns([1, 1])
    with col_t1:
        st.markdown("#### 1. Šablony architektur")
        tpl_keys = list(TEMPLATES.keys())
        current_tpl_idx = tpl_keys.index(st.session_state.nn_tpl_choice) if st.session_state.nn_tpl_choice in tpl_keys else 1
        selected_tpl = st.selectbox(
            "Vyber přednastavenou šablonu (načte se automaticky):",
            tpl_keys,
            index=current_tpl_idx,
            help="Při změně výběru se nová šablona okamžitě načte do tabulky níže.",
        )
        # Automatické načtení při změně v selectboxu
        if selected_tpl != st.session_state.nn_tpl_choice:
            st.session_state.nn_tpl_choice = selected_tpl
            st.session_state.nn_layers = [dict(l) for l in TEMPLATES[selected_tpl]["layers"]]
            st.session_state.nn_input_shape = TEMPLATES[selected_tpl]["input_shape"]
            st.rerun()

        if st.button("🔄 Resetovat na výchozí stav šablony"):
            st.session_state.nn_layers = [dict(l) for l in TEMPLATES[selected_tpl]["layers"]]
            st.session_state.nn_input_shape = TEMPLATES[selected_tpl]["input_shape"]
            st.rerun()

    with col_t2:
        st.markdown("#### 2. Rozměr vstupu (Input Shape)")
        is_1d_curr = len(st.session_state.nn_input_shape) == 1
        input_mode = st.radio(
            "Typ vstupních dat:",
            ["1D Tabulková data (Vektor)", "2D Obrázek (Výška, Šířka, Kanály)"],
            index=0 if is_1d_curr else 1,
            horizontal=True,
        )
        if input_mode == "1D Tabulková data (Vektor)":
            default_f = st.session_state.nn_input_shape[0] if is_1d_curr else 18
            n_features = st.number_input("Počet vstupních příznaků (Features):", min_value=1, max_value=5000, value=int(default_f))
            current_input_shape = (int(n_features),)
        else:
            default_h = st.session_state.nn_input_shape[0] if not is_1d_curr else 28
            default_w = st.session_state.nn_input_shape[1] if not is_1d_curr else 28
            default_c = st.session_state.nn_input_shape[2] if not is_1d_curr else 1
            c_h, c_w, c_c = st.columns(3)
            with c_h:
                h_in = st.number_input("Výška (H):", min_value=4, max_value=1024, value=int(default_h))
            with c_w:
                w_in = st.number_input("Šířka (W):", min_value=4, max_value=1024, value=int(default_w))
            with c_c:
                ch_in = st.number_input("Kanály (C):", min_value=1, max_value=512, value=int(default_c))
            current_input_shape = (int(h_in), int(w_in), int(ch_in))

# Správa vrstev v hlavní ploše
st.subheader("🛠️ Seznam vrstev v síti")

# Funkce pro propagaci tvarů a výpočet parametrů
def compute_architecture(input_shape: Tuple[int, ...], layers: List[Dict[str, Any]]):
    curr_shape = input_shape
    results = []
    has_error = False
    error_msg = ""
    error_layer_idx = None
    
    for i, l in enumerate(layers, 1):
        l_type = l["type"]
        params = 0
        out_shape = None
        note = ""
        
        if l_type == "Dense":
            units = l.get("units", 64)
            use_bias = l.get("use_bias", True)
            if len(curr_shape) != 1:
                has_error = True
                error_layer_idx = i
                error_msg = f"Vstup má tvar {curr_shape} (3D prostorový tensor). Dense vrstva vyžaduje 1D vektor! Vlož před ni vrstvu 'Flatten'."
                results.append({
                    "Vrstva #": i,
                    "Typ vrstvy": l_type,
                    "Vstupní rozměr": str(curr_shape),
                    "Výstupní rozměr": "❌ Neplatný tvar",
                    "Počet parametrů": 0,
                    "Konfigurace & Poznámka": f"❌ CHYBA: {error_msg}",
                })
                break
            in_dim = curr_shape[0]
            params = in_dim * units + (units if use_bias else 0)
            out_shape = (units,)
            note = f"W: ({in_dim}, {units})" + (f" + b: ({units},)" if use_bias else "")
            
        elif l_type == "Conv2D":
            filters = l.get("filters", 32)
            k = l.get("kernel_size", 3)
            pad = l.get("padding", "same")
            use_bias = l.get("use_bias", True)
            if len(curr_shape) != 3:
                has_error = True
                error_layer_idx = i
                error_msg = f"Vstup má tvar {curr_shape} (1D vektor). Conv2D vyžaduje prostorový 3D tensor (Výška, Šířka, Kanály). Konvoluci nelze použít po Dense nebo Flatten!"
                results.append({
                    "Vrstva #": i,
                    "Typ vrstvy": l_type,
                    "Vstupní rozměr": str(curr_shape),
                    "Výstupní rozměr": "❌ Neplatný tvar",
                    "Počet parametrů": 0,
                    "Konfigurace & Poznámka": f"❌ CHYBA: {error_msg}",
                })
                break
            h_in, w_in, c_in = curr_shape
            if pad == "same":
                h_out, w_out = h_in, w_in
            else: # valid
                h_out, w_out = h_in - k + 1, w_in - k + 1
            if h_out <= 0 or w_out <= 0:
                has_error = True
                error_layer_idx = i
                error_msg = f"Jádro filtru {k}x{k} je větší než prostorový rozměr vstupu {h_in}x{w_in}!"
                results.append({
                    "Vrstva #": i,
                    "Typ vrstvy": l_type,
                    "Vstupní rozměr": str(curr_shape),
                    "Výstupní rozměr": "❌ Neplatný tvar",
                    "Počet parametrů": 0,
                    "Konfigurace & Poznámka": f"❌ CHYBA: {error_msg}",
                })
                break
            params = (k * k * c_in * filters) + (filters if use_bias else 0)
            out_shape = (h_out, w_out, filters)
            note = f"Filtry: {filters}x ({k}x{k}x{c_in})" + (f" + b: ({filters},)" if use_bias else "")
            
        elif l_type == "MaxPooling2D":
            p = l.get("pool_size", 2)
            if len(curr_shape) != 3:
                has_error = True
                error_layer_idx = i
                error_msg = f"Vstup má tvar {curr_shape} (1D vektor). MaxPooling2D vyžaduje prostorový 3D tensor (Výška, Šířka, Kanály). Pooling nelze použít po Dense nebo Flatten vrstvě!"
                results.append({
                    "Vrstva #": i,
                    "Typ vrstvy": l_type,
                    "Vstupní rozměr": str(curr_shape),
                    "Výstupní rozměr": "❌ Neplatný tvar",
                    "Počet parametrů": 0,
                    "Konfigurace & Poznámka": f"❌ CHYBA: {error_msg}",
                })
                break
            h_in, w_in, c_in = curr_shape
            h_out, w_out = h_in // p, w_in // p
            if h_out <= 0 or w_out <= 0:
                has_error = True
                error_layer_idx = i
                error_msg = "Rozměry se zmenšily pod 1x1!"
                results.append({
                    "Vrstva #": i,
                    "Typ vrstvy": l_type,
                    "Vstupní rozměr": str(curr_shape),
                    "Výstupní rozměr": "❌ Neplatný tvar",
                    "Počet parametrů": 0,
                    "Konfigurace & Poznámka": f"❌ CHYBA: {error_msg}",
                })
                break
            params = 0
            out_shape = (h_out, w_out, c_in)
            note = f"Zmenšení {p}x bez parametrů"
            
        elif l_type == "Flatten":
            if len(curr_shape) == 1:
                out_shape = curr_shape
                note = "Vstup byl již 1D vektor (no-op)"
            else:
                flat_dim = 1
                for dim in curr_shape:
                    flat_dim *= dim
                out_shape = (flat_dim,)
                note = f"Rozbalení {curr_shape} -> {flat_dim}"
            params = 0
            
        elif l_type == "Dropout":
            rate = l.get("rate", 0.2)
            params = 0
            out_shape = curr_shape
            note = f"Vypíná {rate*100:.0f} % neuronů při tréninku"
            
        elif l_type == "GlobalAveragePooling2D":
            if len(curr_shape) != 3:
                has_error = True
                error_layer_idx = i
                error_msg = f"Vstup má tvar {curr_shape} (1D vektor). GlobalAveragePooling2D vyžaduje 3D tensor (H, W, C)!"
                results.append({
                    "Vrstva #": i,
                    "Typ vrstvy": l_type,
                    "Vstupní rozměr": str(curr_shape),
                    "Výstupní rozměr": "❌ Neplatný tvar",
                    "Počet parametrů": 0,
                    "Konfigurace & Poznámka": f"❌ CHYBA: {error_msg}",
                })
                break
            c_in = curr_shape[2]
            params = 0
            out_shape = (c_in,)
            note = f"Zprůměruje HxW prostor na {c_in} hodnot"
            
        results.append({
            "Vrstva #": i,
            "Typ vrstvy": l_type,
            "Vstupní rozměr": str(curr_shape),
            "Výstupní rozměr": str(out_shape),
            "Počet parametrů": params,
            "Konfigurace & Poznámka": note,
        })
        curr_shape = out_shape
        
    return results, has_error, error_msg, error_layer_idx, curr_shape

# Výpočet aktuální sítě
rows, has_err, err_msg, err_layer_idx, last_shape = compute_architecture(current_input_shape, st.session_state.nn_layers)

if len(rows) > 0:
    df_res = pd.DataFrame(rows)
    st.dataframe(df_res, hide_index=True, width="stretch")

if has_err:
    st.error(f"❌ **Architektonická neshoda rozměrů (Shape Mismatch) ve vrstvě {err_layer_idx}:** {err_msg}")
    c_fix1, c_fix2 = st.columns([1.5, 3.5])
    with c_fix1:
        if st.button(f"🗑️ Odstranit chybnou vrstvu #{err_layer_idx}"):
            if len(st.session_state.nn_layers) >= err_layer_idx:
                st.session_state.nn_layers.pop(err_layer_idx - 1)
                st.rerun()
    with c_fix2:
        st.info("💡 **Pravidlo hlubokého učení:** Konvoluce (`Conv2D`) a pooling (`MaxPooling2D`) operují v 2D ploše obrazu se šířkou, výškou a kanály `(H, W, C)`. Jakmile síť projde vrstvou `Dense` nebo `Flatten`, stane se z ní plochý 1D vektor čísel. Po vrstvě `Dense` už nelze provádět prostorové konvoluce ani pooling.")
else:
    if len(rows) > 0:
        total_params = df_res["Počet parametrů"].sum()
        memory_mb = (total_params * 4) / (1024 * 1024)  # 4 bajty na float32
        
        c_p1, c_p2, c_p3 = st.columns(3)
        c_p1.metric("Celkem trénovatelných parametrů", f"{total_params:,} vah")
        c_p2.metric("Paměť pro váhy (Float32)", f"{memory_mb:.2f} MB")
        c_p3.metric("Konečný tvar výstupu", df_res.iloc[-1]["Výstupní rozměr"])
    else:
        st.warning("Síť je zatím prázdná. Přidej vrstvy níže nebo vyber přednastavenou šablonu.")

# Ovládání přidávání vrstev
st.divider()
st.subheader("➕ Přidat nebo upravit vrstvu")

# Informace o aktuálním výstupu sítě pro uživatele
is_current_1d = (last_shape is None) or (len(last_shape) == 1)
shape_label = f"{last_shape} (1D vektor)" if is_current_1d else f"{last_shape} (3D prostorový tensor)"
st.caption(f"📌 **Aktuální výstupní tvar po poslední platné vrstvě:** `{shape_label}`")

col_a1, col_a2, col_a3, col_a4 = st.columns([1.5, 1.2, 1.2, 1.0])
with col_a1:
    new_layer_type = st.selectbox("Typ vrstvy k přidání:", ["Dense", "Conv2D", "MaxPooling2D", "Flatten", "Dropout", "GlobalAveragePooling2D"])

# Kontrola kompatibility před přidáním
if is_current_1d and new_layer_type in ["Conv2D", "MaxPooling2D", "GlobalAveragePooling2D"]:
    st.warning(f"⚠️ **Pozor na rozměry:** Výstup sítě je 1D vektor {last_shape}. Vrstva **{new_layer_type}** vyžaduje prostorový 3D tensor (H, W, C). Přidáním vznikne rozměrová chyba (Shape Mismatch). Pro 1D data použij **Dense** nebo **Dropout**.")

if new_layer_type == "Dense":
    with col_a2:
        new_units = st.number_input("Počet neuronů (units):", min_value=1, max_value=4096, value=64, step=16)
    with col_a3:
        new_act = st.selectbox("Aktivace:", ["relu", "sigmoid", "softmax", "tanh", "linear"])
    new_layer_dict = {"type": "Dense", "units": int(new_units), "activation": new_act, "use_bias": True}

elif new_layer_type == "Conv2D":
    with col_a2:
        new_filt = st.number_input("Filtry:", min_value=1, max_value=512, value=32, step=8)
        new_k = st.selectbox("Kernel size:", [1, 3, 5, 7], index=1)
    with col_a3:
        new_pad = st.selectbox("Padding:", ["same", "valid"])
        new_act = st.selectbox("Aktivace:", ["relu", "linear"])
    new_layer_dict = {"type": "Conv2D", "filters": int(new_filt), "kernel_size": int(new_k), "padding": new_pad, "activation": new_act, "use_bias": True}

elif new_layer_type == "MaxPooling2D":
    with col_a2:
        new_pool = st.selectbox("Pool size:", [2, 3, 4], index=0)
    new_layer_dict = {"type": "MaxPooling2D", "pool_size": int(new_pool)}

elif new_layer_type == "Dropout":
    with col_a2:
        new_rate = st.slider("Dropout rate:", min_value=0.05, max_value=0.8, value=0.25, step=0.05)
    new_layer_dict = {"type": "Dropout", "rate": float(new_rate)}

else:
    new_layer_dict = {"type": new_layer_type}

with col_a4:
    st.write("")
    st.write("")
    if st.button("Přidat vrstvu ⬇️"):
        st.session_state.nn_layers.append(new_layer_dict)
        st.rerun()

col_btn1, col_btn2 = st.columns([1, 4])
with col_btn1:
    if st.button("🗑️ Smazat poslední vrstvu") and len(st.session_state.nn_layers) > 0:
        st.session_state.nn_layers.pop()
        st.rerun()
with col_btn2:
    if st.button("🧹 Vyčistit všechny vrstvy"):
        st.session_state.nn_layers = []
        st.rerun()
