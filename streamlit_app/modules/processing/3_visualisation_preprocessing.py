import streamlit as st
from pathlib import Path
import cv2
from src.processing import config
from src.processing.preprocess import Preprocessor
from src.processing.utils import load_image, list_images

def main():
    st.title("🛠 Prétraitement (face crop + resize)")

    cls = st.selectbox("Classe :", config.CLASS_NAMES)
    folder = config.RAW_DIR / "train" / cls
    images = list_images(folder)

    if not images:
        st.warning("Aucune image trouvée")
        return

    idx = st.slider("Index image :", 0, len(images)-1, 0)
    raw_img = load_image(images[idx])
    st.subheader("Image brute")
    st.image(raw_img, width=250)

    pre = Preprocessor()
    processed, status = pre.process_image(images[idx])
    
    st.subheader("Image après preprocessing")
    if processed is None:
        st.error(f"Image rejetée : {status}")
    else:
        st.image(processed, width=250)
        st.success(f"Statut : {status}")
