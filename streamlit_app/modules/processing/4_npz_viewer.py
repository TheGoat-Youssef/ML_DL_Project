import streamlit as st
import numpy as np
from src.processing import config

def main():
    st.title("🗂 Fichier processed_fer2013.npz")

    if not config.PROCESSED_NPZ.exists():
        st.warning("Aucun fichier processed_fer2013.npz trouvé !")
        return

    data = np.load(config.PROCESSED_NPZ, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    classes = data["classes"]

    st.write(f"Shape X : {X.shape}")
    st.write(f"Shape y : {y.shape}")
    st.write(f"Classes : {classes}")

    idx = st.slider("Index image :", 0, len(X)-1, 0)
    st.image(X[idx], caption=f"Label : {y[idx]}", width=300)
