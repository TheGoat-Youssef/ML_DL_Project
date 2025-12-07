import streamlit as st
from pathlib import Path
import importlib.util

MODULES_DIR = Path(__file__).parent.parent / "modules"

def load_sidebar():
    st.sidebar.title("Modules du Projet")
    
    # détecter tous les modules
    modules = [d.name for d in MODULES_DIR.iterdir() if d.is_dir()]
    selected_module = st.sidebar.selectbox("Choisir un module :", modules)

    # pages du module
    module_path = MODULES_DIR / selected_module
    pages = sorted(list(module_path.glob("*.py")))

    page_names = [p.stem.replace("_", " ").title() for p in pages]
    selected_page = st.sidebar.radio("Pages :", page_names)

    # index
    page_index = page_names.index(selected_page)
    page_file = pages[page_index]

    # --- Import file dynamically from path ---
    spec = importlib.util.spec_from_file_location(
        f"{selected_module}.{page_file.stem}", 
        page_file
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Call main()
    module.main()
