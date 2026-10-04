from pathlib import Path
import streamlit.components.v1 as components

_CANVAS_DIR = Path(__file__).resolve().parent / "digit_canvas"

# declare_component must be called in an imported module so inspect.getmodule() is not None
digit_canvas = components.declare_component("digit_canvas", path=str(_CANVAS_DIR))
