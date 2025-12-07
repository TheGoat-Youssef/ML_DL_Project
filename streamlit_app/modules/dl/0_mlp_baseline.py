import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from src.dl.data_loader import load_processed_npz
from src.dl.mlp import build_mlp
from src.dl.utils import make_tf_dataset
from src.dl.training import compile_model, fit_model
from src.dl.evaluation import compute_metrics, plot_confusion

def main():
    st.title("DL — MLP Baseline")

    path = st.text_input("NPZ path", "data/processed/processed_fer2013.npz")

    # Epochs & hyperparameters inputs BEFORE the train button
    lr = st.number_input("Learning rate", 1e-5, 1e-2, value=1e-3, format="%.5f")
    batch_size = st.number_input("Batch size", 8, 256, value=32)
    epochs = st.number_input("Epochs", 1, 200, value=10)

    if st.button("Load & Train MLP"):
        X_train, X_test, y_train, y_test, classes = load_processed_npz(path)
        input_shape = X_train.shape[1:]
        num_classes = len(classes)

        model = build_mlp(input_shape, num_classes, hidden_units=(512, 256), dropout=0.4)
        compile_model(model, lr=lr)

        train_ds = make_tf_dataset(X_train, y_train, batch_size=batch_size, shuffle=True, one_hot=True)
        val_ds = make_tf_dataset(X_test, y_test, batch_size=batch_size, shuffle=False, one_hot=True)

        hist = fit_model(model, train_ds, val_ds, epochs=epochs, name="mlp_baseline")
        st.success("Training finished")

        # --- Plot training metrics ---
        fig, ax = plt.subplots(1, 2, figsize=(14,5))
        ax[0].plot(hist.history['loss'], label='train_loss')
        ax[0].plot(hist.history['val_loss'], label='val_loss')
        ax[0].set_title("Loss over Epochs")
        ax[0].legend()
        ax[1].plot(hist.history.get('accuracy', hist.history.get('acc')), label='train_acc')
        ax[1].plot(hist.history.get('val_accuracy', hist.history.get('val_acc')), label='val_acc')
        ax[1].set_title("Accuracy over Epochs")
        ax[1].legend()
        st.pyplot(fig)

        # Predict & metrics
        preds = model.predict(X_test)
        report, cm = compute_metrics(y_test, preds, labels=classes)

        # JSON
        st.subheader("Classification Report (JSON)")
        st.json(report)

        # Table
        st.subheader("Classification Report (Table)")
        df_report = pd.DataFrame(report).transpose()
        st.table(df_report)

        # Confusion matrix
        fig_cm, ax_cm = plt.subplots(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax_cm)
        st.pyplot(fig_cm)

        # Save model
        save_path = st.text_input("Save model path", "experiments/dl_checkpoints/mlp_baseline.keras")
        if st.button("Save Model"):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            model.save(save_path)
            st.success(f"Model saved to {save_path}")


