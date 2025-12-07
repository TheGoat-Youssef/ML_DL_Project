import streamlit as st
import os
import numpy as np
from PIL import Image
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from src.dl.data_loader import load_processed_npz
import matplotlib.pyplot as plt

def main():
    st.title("DL — Image Prediction")

    # --- 1. Select model ---
    checkpoint_dir = st.text_input("Checkpoint directory", "experiments/dl_checkpoints")
    available_models = []
    if os.path.exists(checkpoint_dir):
        available_models = [f for f in os.listdir(checkpoint_dir) if f.endswith(".keras")]

    if available_models:
        model_name = st.selectbox("Select a model", available_models)
        model_path = os.path.join(checkpoint_dir, model_name)
        model = load_model(model_path)
    else:
        st.warning("No models found in the checkpoint directory!")
        return

    # --- 2. Upload image ---
    uploaded_file = st.file_uploader("Upload an image (jpg/png/jpeg)", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")

        # Layout: two columns
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(img, caption="Uploaded Image", use_container_width=True)
        with col2:
            st.subheader("Prediction Results")

            # --- 3. Preprocess image ---
            img_resized = img.resize((48, 48))  # FER2013 size
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)  # add batch dim

            # --- 4. Predict ---
            preds = model.predict(img_array)[0]
            classes = load_processed_npz("data/processed/processed_fer2013.npz")[4]  # classes list

            pred_idx = np.argmax(preds)
            pred_class = classes[pred_idx]
            confidence = preds[pred_idx]

            st.markdown(f"**Predicted Class:** `{pred_class}`")
            st.markdown(f"**Confidence:** `{confidence*100:.2f}%`")

            # --- 5. Show all probabilities ---
            st.subheader("All Class Probabilities")
            prob_dict = {cls: float(preds[i]) for i, cls in enumerate(classes)}
            
            # Table
            prob_df = np.array(list(prob_dict.items()))
            st.table(pd.DataFrame(prob_df, columns=["Class", "Probability"]).assign(Probability=lambda x: x["Probability"].astype(float).round(3)))

            # Bar chart
            fig, ax = plt.subplots(figsize=(8,4))
            ax.bar(classes, preds, color='skyblue')
            ax.set_ylim(0, 1)
            ax.set_ylabel("Probability")
            ax.set_title("Class Probabilities")
            plt.xticks(rotation=45)
            st.pyplot(fig)

        st.success("Prediction completed ")
