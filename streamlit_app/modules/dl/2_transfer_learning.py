import streamlit as st
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.dl.data_loader import load_processed_npz
from src.dl.transfer import build_transfer_model
from src.dl.utils import make_tf_dataset
from src.dl.training import compile_model, fit_model
from src.dl.evaluation import compute_metrics

def main():
    st.title("DL — Transfer Learning")

    # --- Inputs ---
    path = st.text_input("NPZ path", "data/processed/processed_fer2013.npz")
    backbone = st.selectbox("Backbone", ["VGG16", "ResNet50", "EfficientNetB0"])
    trainable_layers = st.number_input("Unfreeze last N layers", min_value=0, max_value=200, value=0)
    lr = st.number_input("Learning rate", 1e-5, 1e-4, value=1e-4, format="%.6f")
    batch_size = st.number_input("Batch size", 8, 128, value=32)
    epochs = st.number_input("Epochs", 1, 50, value=5)

    save_path_default = f"experiments/dl_checkpoints/transfer_{backbone.lower()}.keras"
    save_path = st.text_input("Save model path", save_path_default)

    # --- Train Button ---
    if st.button("Load & Train Transfer Model"):

        # --- Load dataset ---
        if not os.path.exists(path):
            st.warning("NPZ file not found!")
            return

        X_train, X_test, y_train, y_test, classes = load_processed_npz(path)

        # Convert grayscale -> RGB
        if X_train.ndim == 3:
            X_train = X_train[..., None]
            X_test = X_test[..., None]
        if X_train.shape[-1] == 1:
            X_train = tf.image.grayscale_to_rgb(tf.convert_to_tensor(X_train)).numpy()
            X_test = tf.image.grayscale_to_rgb(tf.convert_to_tensor(X_test)).numpy()

        input_shape = X_train.shape[1:]
        num_classes = len(classes)

        # --- Build model ---
        model = build_transfer_model(
            backbone,
            input_shape,
            num_classes,
            head_units=256,
            head_dropout=0.5,
            trainable_layers=int(trainable_layers)
        )
        compile_model(model, lr=lr)

        # --- Prepare datasets ---
        train_ds = make_tf_dataset(X_train, y_train, batch_size=batch_size, shuffle=True, one_hot=True)
        val_ds = make_tf_dataset(X_test, y_test, batch_size=batch_size, shuffle=False, one_hot=True)

        # --- Train model ---
        hist, fig_acc, fig_loss = fit_model(model, train_ds, val_ds, epochs=epochs, name=f"transfer_{backbone.lower()}")
        st.success("Training finished ✅")

        # --- Plot training metrics ---
        if fig_acc is not None and fig_loss is not None:
            st.subheader("Training Accuracy & Loss")
            st.pyplot(fig_acc)
            st.pyplot(fig_loss)
        else:
            # fallback if figures not returned
            fig, ax = plt.subplots(1, 2, figsize=(14,5))
            ax[0].plot(hist.history['loss'], label='train_loss')
            ax[0].plot(hist.history['val_loss'], label='val_loss')
            ax[0].set_title("Loss over Epochs")
            ax[0].legend()
            ax[1].plot(hist.history.get('accuracy', hist.history.get('acc')), label='train_acc')
            ax[1].plot(hist.history.get('val_accuracy', hist.history.get('val_acc')), label='val_acc')
            ax[1].set_title("Accuracy over Epochs")
            ax[1].legend()
            st.subheader("Training Accuracy & Loss")
            st.pyplot(fig)

        # --- Predict & Evaluate ---
        preds = model.predict(X_test)
        report, cm = compute_metrics(y_test, preds, labels=classes)

        st.subheader("Classification Report (JSON)")
        st.json(report)

        st.subheader("Classification Report (Table)")
        df_report = pd.DataFrame(report).transpose().fillna(0).round(3)
        st.table(df_report)

        # --- Confusion matrix ---
        fig_cm, ax_cm = plt.subplots(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax_cm)
        st.pyplot(fig_cm)

        # --- Save model ---
        if st.button("Save Model"):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            model.save(save_path)
            st.success(f"Model saved to {save_path}")
