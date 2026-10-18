"""Integrated evidence search; model initialization is scoped to this page."""
import streamlit as st
try:
    from components.ml_librarian.embedded import render
    render('search')
except ImportError:
    st.error('Modul ML Librarian nemá dostupné závislosti. Provozovatel musí aktualizovat instalaci podle requirements.txt.')
