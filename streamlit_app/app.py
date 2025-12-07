import sys
from pathlib import Path
import streamlit as st
from core.navigation import load_sidebar

# Add project root to Python path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))



st.set_page_config(
    page_title="FER2013 Dashboard",
    layout="wide",
    page_icon=""
)

st.title("FER2013 – Dashboard Complet")
st.markdown("""
Interface pour consulter **Exploration, Prétraitement et  ML/DL/Clustering**.
""")

# Sidebar + navigation
load_sidebar()
