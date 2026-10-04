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
if "nn_layers" not in st.session_state:
    st.session_state.nn_layers = list(TEMPLATES["Cvičení 1: MNIST Jednoduchý MLP (784 -> 128 -> 10)"]["layers"])
if "nn_input_mode" not in st.session_state:
    st.session_state.nn_input_mode = "Obrázek 2D (28, 28, 1)"

# Postranní panel
with st.sidebar:
    st.header("⚙️ Šablony a vstupní data")
    selected_tpl = st.selectbox("Vyber přednastavenou šablonu:", list(TEMPLATES.keys()))
    if st.button("Načíst vybranou šablonu"):
        st.session_state.nn_layers = [dict(l) for l in TEMPLATES[selected_tpl]["layers"]]
        st.rerun()

    st.divider()
    st.header("📥 Rozměr vstupu (Input Shape)")
    input_mode = st.radio(
        "Typ vstupních dat:",
        ["1D Tabulková data (Vektor)", "2D Obrázek (Výška, Šířka, Kanály)"],
        index=0 if len(TEMPLATES[selected_tpl]["input_shape"]) == 1 else 1,
    )
    if input_mode == "1D Tabulková data (Vektor)":
        n_features = st.number_input("Počet vstupních příznaků (Features):", min_value=1, max_value=5000, value=18)
        current_input_shape = (int(n_features),)
    else:
        c_h, c_w = st.columns(2)
        with c_h:
            h_in = st.number_input("Výška (H):", min_value=4, max_value=1024, value=28)
            ch_in = st.number_input("Kanály (C):", min_value=1, max_value=512, value=1)
        with c_w:
            w_in = st.number_input("Šířka (W):", min_value=4, max_value=1024, value=28)
        current_input_shape = (int(h_in), int(w_in), int(ch_in))

# Správa vrstev v hlavní ploše
st.subheader("🛠️ Seznam vrstev v síti")

# Funkce pro propagaci tvarů a výpočet parametrů
def compute_architecture(input_shape: Tuple[int, ...], layers: List[Dict[str, Any]]):
    curr_shape = input_shape
    results = []
    has_error = False
    error_msg = ""
    
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
                error_msg = f"Chyba ve vrstvě {i} (Dense): Vstup má tvar {curr_shape}. Dense vrstva vyžaduje 1D vektor! Vlož před ni vrstvu 'Flatten'."
                break
            in_dim = curr_shape[0]
            # W: in_dim * units, b: units
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
                error_msg = f"Chyba ve vrstvě {i} (Conv2D): Vstup má tvar {curr_shape}. Conv2D vyžaduje 3D tensor (H, W, C)!"
                break
            h_in, w_in, c_in = curr_shape
            if pad == "same":
                h_out, w_out = h_in, w_in
            else: # valid
                h_out, w_out = h_in - k + 1, w_in - k + 1
            if h_out <= 0 or w_out <= 0:
                has_error = True
                error_msg = f"Chyba ve vrstvě {i} (Conv2D): Jádro filtru {k}x{k} je větší než vstupní rozměr {h_in}x{w_in}!"
                break
            # W: k * k * c_in * filters, b: filters
            params = (k * k * c_in * filters) + (filters if use_bias else 0)
            out_shape = (h_out, w_out, filters)
            note = f"Filtry: {filters}x ({k}x{k}x{c_in})" + (f" + b: ({filters},)" if use_bias else "")
            
        elif l_type == "MaxPooling2D":
            p = l.get("pool_size", 2)
            if len(curr_shape) != 3:
                has_error = True
                error_msg = f"Chyba ve vrstvě {i} (MaxPooling2D): Vstup musí být 3D tensor, ale je {curr_shape}."
                break
            h_in, w_in, c_in = curr_shape
            h_out, w_out = h_in // p, w_in // p
            if h_out <= 0 or w_out <= 0:
                has_error = True
                error_msg = f"Chyba ve vrstvě {i} (MaxPooling2D): Rozměry se zmenšily pod 1x1!"
                break
            params = 0
            out_shape = (h_out, w_out, c_in)
            note = f"Zmenšení {p}x bez parametrů"
            
        elif l_type == "Flatten":
            if len(curr_shape) == 1:
                out_shape = curr_shape
                note = "Vstup byl již 1D (no-op)"
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
                error_msg = f"Chyba ve vrstvě {i} (GlobalAveragePooling2D): Vyžaduje 3D tensor, obdržel {curr_shape}."
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
        
    return results, has_error, error_msg

# Výpočet aktuální sítě
rows, has_err, err_msg = compute_architecture(current_input_shape, st.session_state.nn_layers)

if has_err:
    st.error(f"❌ **Architektonická chyba:** {err_msg}")
else:
    df_res = pd.DataFrame(rows)
    total_params = df_res["Počet parametrů"].sum()
    memory_mb = (total_params * 4) / (1024 * 1024)  # 4 bajty na float32
    
    c_p1, c_p2, c_p3 = st.columns(3)
    c_p1.metric("Celkem trénovatelných parametrů", f"{total_params:,} vah")
    c_p2.metric("Paměť pro váhy (Float32)", f"{memory_mb:.2f} MB")
    c_p3.metric("Konečný tvar výstupu", df_res.iloc[-1]["Výstupní rozměr"] if len(df_res) > 0 else "N/A")
    
    st.dataframe(df_res, hide_index=True, use_container_width=True)

# Ovládání přidávání vrstev
st.divider()
st.subheader("➕ Přidat nebo upravit vrstvu")

col_a1, col_a2, col_a3, col_a4 = st.columns([1.5, 1.2, 1.2, 1.0])
with col_a1:
    new_layer_type = st.selectbox("Typ vrstvy k přidání:", ["Dense", "Conv2D", "MaxPooling2D", "Flatten", "Dropout", "GlobalAveragePooling2D"])

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
