import streamlit as st
from src.processing.preprocess import Preprocessor
from src.processing import config

def main():
    st.title("🛠 Preprocessing automatique")
    st.markdown("""
    Cliquez sur le bouton ci-dessous pour **lancer le preprocessing** de toutes les images.
    Cela va :
    - Détecter et recadrer les visages
    - Redimensionner les images
    - Supprimer ou marquer les images problématiques
    - Sauvegarder le fichier `processed_fer2013.npz`
    - Sauvegarder `processing_errors.json`
    """)

    if st.button("▶ Lancer le Preprocessing"):
        pre = Preprocessor()
        with st.spinner("Prétraitement en cours..."):
            X, y, errors = pre.run()
        
        st.success(f"Préprocessing terminé ! {len(X)} images traitées, {len(errors)} erreurs.")
        st.write("✅ Fichier sauvegardé :", config.PROCESSED_NPZ.name)
        st.write("⚠ Erreurs sauvegardées :", "processing_errors.json")
