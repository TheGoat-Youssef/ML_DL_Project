import streamlit as st
from pathlib import Path
import json
from src.processing import config
from src.processing.utils import load_image
import matplotlib.pyplot as plt

def main():
    st.title("!!Images rejetées!!")

    errors_file = config.PROCESSED_DIR / "processing_errors.json"
    if not errors_file.exists():
        st.warning("Aucun fichier errors trouvé. Lance le preprocessing d'abord !")
        return

    errors = json.load(open(errors_file))

    counts = {"no_face": 0, "blurry": 0, "corrupt": 0}
    for e in errors:
        counts[e["error"]] += 1

    st.write("Nombre d'erreurs par type :", counts)
    st.dataframe(errors)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(3, 2))  # very small figure
    ax.bar(counts.keys(), counts.values(), color=['red', 'orange', 'blue'])
    ax.set_title("Distribution des erreurs", fontsize=8)
    ax.set_xlabel("Type", fontsize=7)
    ax.set_ylabel("Nb images", fontsize=7)
    ax.tick_params(axis='both', which='major', labelsize=6)
    st.pyplot(fig, bbox_inches='tight')


    # --- Slider to view rejected images ---
    idx = st.slider("Voir une image rejetée :", 0, len(errors)-1, 0)
    img_path = Path(errors[idx]["image"])
    img = load_image(img_path)
    if img is not None:
        st.image(img, caption=f"{errors[idx]['error']} — {img_path.name}", width=300)
    else:
        st.error("Impossible de lire l'image")
