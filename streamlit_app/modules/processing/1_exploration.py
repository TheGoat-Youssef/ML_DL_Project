import sys
from pathlib import Path
import streamlit as st
from src.processing import config
from src.processing import utils


import matplotlib.pyplot as plt

def main():
    st.title("Exploration des images brutes")
    
    cls = None
    if config.CLASS_NAMES:
        cls = st.selectbox("Classe :", config.CLASS_NAMES)
    else:
        st.warning("Aucune classe trouvée dans data/raw/train !")
        return

    folder = config.RAW_DIR / "train" / cls

    images = utils.list_images(folder)

    if not images:
        st.warning("Aucune image trouvée")
        return

    # Slider to select image index
    idx = st.slider("Index image :", 0, len(images)-1, 0)
    img = utils.load_image(images[idx])

    st.image(img, caption=f"{cls} — {images[idx].name}", width=300)
    st.markdown(f"Nombre d'images : **{len(images)}**")
