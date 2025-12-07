import streamlit as st
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from src.dl.data_loader import load_processed_npz
from src.dl.evaluation import compute_metrics

def main():
    st.set_page_config(page_title="DL Model Comparison", layout="wide")
    st.title("DL — Compare Multiple Models")

    # --- 1. Load Dataset ---
    st.header("1 - Dataset")
    npz_path = st.text_input("NPZ file path", "data/processed/processed_fer2013.npz")
    
    if not os.path.exists(npz_path):
        st.warning("NPZ file not found!")
        return

    X_train, X_test, y_train, y_test, classes = load_processed_npz(npz_path)
    
    # Convert to RGB if grayscale
    if X_test.ndim == 3:
        X_test = X_test[..., None]
    if X_test.shape[-1] == 1:
        X_test = tf.image.grayscale_to_rgb(tf.convert_to_tensor(X_test)).numpy()

    st.success(f"Dataset loaded: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    st.write(f"Classes: {classes}")

    # --- 2. Select Models ---
    st.header("2 - Select Models")
    checkpoint_dir = st.text_input("Checkpoint directory", "experiments/dl_checkpoints")

    if not os.path.exists(checkpoint_dir):
        st.warning("Checkpoint directory not found!")
        return

    available_models = [f for f in os.listdir(checkpoint_dir) if f.endswith(".keras")]
    if not available_models:
        st.warning("No .keras models found in the directory")
        return

    selected_models = st.multiselect(
        "Select models to compare",
        options=available_models,
        default=available_models[:2]
    )

    if not selected_models:
        st.info("Please select at least one model")
        return

    # --- 3. Evaluate Models ---
    st.header("3 - Evaluate Models")
    results = []

    if st.button("Evaluate Selected Models"):
        for model_name in selected_models:
            st.subheader(f"->{model_name}")
            model_path = os.path.join(checkpoint_dir, model_name)
            model = load_model(model_path)
            preds = model.predict(X_test)
            report, cm = compute_metrics(y_test, preds, labels=classes)

            # Confusion matrix collapsible section
            with st.expander("Show Confusion Matrix"):
                fig, ax = plt.subplots(figsize=(8,6))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=classes, yticklabels=classes, ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                st.pyplot(fig)

            # Display JSON report
            st.write("Classification Report (JSON):")
            st.json(report)

            # Save results for summary table
            summary_metrics = {}
            for idx, cls_name in enumerate(classes):
                key = str(idx)  # report keys are "0", "1", ...
                summary_metrics[f"{cls_name}_precision"] = report[key]["precision"]
                summary_metrics[f"{cls_name}_recall"] = report[key]["recall"]
                summary_metrics[f"{cls_name}_f1"] = report[key]["f1-score"]
            summary_metrics["accuracy"] = report.get("accuracy", 0)
            results.append((model_name, summary_metrics))

        # --- 4. Summary Table ---
        if results:
            st.header("4 - Summary Table")
            summary_df = pd.DataFrame([r[1] for r in results], index=[r[0] for r in results])
            summary_df = summary_df.fillna(0).round(3)

            # Style table
            def highlight_best(s):
                is_max = s == s.max()
                return ['background-color: #90ee90' if v else '' for v in is_max]

            styled_df = summary_df.style \
                .apply(highlight_best, subset=summary_df.columns.drop("accuracy")) \
                .background_gradient(cmap='Blues', subset=["accuracy"]) \
                .set_properties(**{'text-align': 'center'}) \
                .set_caption("Green = Best per metric, Accuracy shown in blue gradient") \
                .format("{:.3f}")

            st.dataframe(styled_df, height=500)

            # --- 5. Download CSV ---
            csv = summary_df.to_csv().encode('utf-8')
            st.download_button(
                label="Download Summary CSV",
                data=csv,
                file_name='model_comparison_summary.csv',
                mime='text/csv'
            )

    st.info("Select models and click 'Evaluate Selected Models' to see metrics and comparison.")
