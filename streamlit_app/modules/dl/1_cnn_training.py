import streamlit as st
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import seaborn as sns
from src.dl.data_loader import load_processed_npz
from src.dl.cnn import build_simple_cnn, build_deeper_cnn
from src.dl.utils import make_tf_dataset
from src.dl.training import compile_model, fit_model
from src.dl.evaluation import compute_metrics

def main():
    st.title("DL — CNN Training")

    # --- Inputs ---
    path = st.text_input("NPZ path", "data/processed/processed_fer2013.npz")
    mode = st.selectbox("Architecture", ["simple", "deeper"])
    lr = st.number_input("Learning rate", 1e-5, 1e-2, value=1e-3, format="%.5f")
    batch_size = st.number_input("Batch size", 8, 256, value=32)
    epochs = st.number_input("Epochs", 1, 200, value=10)

    save_path_default = f"experiments/dl_checkpoints/cnn_{mode}.keras"
    save_path = st.text_input("Save model path", save_path_default)

    # --- Train Button ---
    if st.button("Load & Train CNN"):

        # Load data
        X_train, X_test, y_train, y_test, classes = load_processed_npz(path)

        # Ensure shape (N,H,W,3)
        if X_train.ndim == 3:  # add channel dim
            X_train = X_train[..., None]
            X_test = X_test[..., None]
        if X_train.shape[-1] == 1:  # grayscale -> RGB
            X_train = tf.image.grayscale_to_rgb(tf.convert_to_tensor(X_train)).numpy()
            X_test = tf.image.grayscale_to_rgb(tf.convert_to_tensor(X_test)).numpy()

        input_shape = X_train.shape[1:]
        num_classes = len(classes)

        # Build model
        if mode == "simple":
            model = build_simple_cnn(input_shape, num_classes)
        else:
            model = build_deeper_cnn(input_shape, num_classes)

        compile_model(model, lr=lr)

        # Prepare datasets
        train_ds = make_tf_dataset(X_train, y_train, batch_size=batch_size, shuffle=True, one_hot=True)
        val_ds = make_tf_dataset(X_test, y_test, batch_size=batch_size, shuffle=False, one_hot=True)

        # Train
        hist, fig_acc, fig_loss = fit_model(model, train_ds, val_ds, epochs=epochs, name=f"cnn_{mode}")
        st.success("Training finished")

        # --- Plot training metrics ---
        if fig_acc is not None and fig_loss is not None:
            st.subheader("Training Accuracy & Loss")
            st.pyplot(fig_acc)
            st.pyplot(fig_loss)

        # --- Evaluate ---
        preds = model.predict(X_test)
        report, cm = compute_metrics(y_test, preds, labels=classes)
        
        st.subheader("Classification Report (JSON)")
        st.json(report)

        st.subheader("Classification Report (Table)")
        import pandas as pd
        df_report = pd.DataFrame(report).transpose()
        st.table(df_report)

        # Confusion matrix
        fig_cm, ax_cm = plt.subplots(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax_cm)
        st.pyplot(fig_cm)

        # Save model
        if st.button("Save Model"):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            model.save(save_path)
            st.success(f"Model saved to {save_path}")
