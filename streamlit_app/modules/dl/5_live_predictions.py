import streamlit as st
import cv2
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from src.dl.data_loader import load_processed_npz
import time

def main():
    st.title("Live DL Facial Expression Recognition")

    # --- 1. Load model ---
    checkpoint_dir = "experiments/dl_checkpoints"
    model_name = st.selectbox("Select a model", [f for f in os.listdir(checkpoint_dir) if f.endswith(".keras")])
    model = load_model(f"{checkpoint_dir}/{model_name}")
    classes = load_processed_npz("data/processed/processed_fer2013.npz")[4]  # classes list

    # --- 2. Start live camera ---
    run = st.checkbox("Start Camera")
    FRAME_WINDOW = st.image([])  # placeholder for camera frames
    prediction_text = st.empty()  # placeholder for predicted label

    if run:
        cap = cv2.VideoCapture(0)  # open default webcam

        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame")
                break

            # Convert frame to RGB
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)

            # Preprocess for model
            img_resized = img_pil.resize((48, 48))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            preds = model.predict(img_array, verbose=0)[0]
            pred_idx = np.argmax(preds)
            pred_class = classes[pred_idx]
            confidence = preds[pred_idx]

            # Update Streamlit
            FRAME_WINDOW.image(img, channels="RGB")
            prediction_text.markdown(f"**Prediction:** {pred_class} ({confidence*100:.2f}%)")

            # Small delay to avoid overloading CPU
            time.sleep(0.1)

        cap.release()
