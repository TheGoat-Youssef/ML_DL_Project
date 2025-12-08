import streamlit as st
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

from tensorflow.keras.models import load_model, Model
from src.dl.data_loader import load_processed_npz

# ============================================================
# SETTINGS
# ============================================================
BATCH_SIZE = 256
FEATURE_DIR = "data/processed/features"
CHECKPOINT_DIR = "experiments/dl_checkpoints"
VISUALIZE_SAMPLES = 2000  # number of points to show in t-SNE/UMAP

os.makedirs(FEATURE_DIR, exist_ok=True)

# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def list_trained_models():
    if not os.path.exists(CHECKPOINT_DIR):
        return []
    return [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".keras")]

def list_existing_features():
    if not os.path.exists(FEATURE_DIR):
        return []
    return [f for f in os.listdir(FEATURE_DIR) if f.endswith(".npy")]

@st.cache_resource
def load_data(npz_path):
    return load_processed_npz(npz_path)

@st.cache_resource
def load_feature_extractor(model_path):
    model = load_model(model_path)
    # Remove final classification layer
    feature_model = Model(inputs=model.input, outputs=model.layers[-2].output)
    return feature_model

def extract_features(X, extractor):
    feats = []
    for i in tqdm(range(0, len(X), BATCH_SIZE), desc="Extracting features"):
        batch = X[i:i+BATCH_SIZE]
        if batch.ndim == 3:
            batch = batch[..., None]
        if batch.shape[-1] == 1:
            batch = tf.image.grayscale_to_rgb(batch).numpy()
        f = extractor.predict(batch, verbose=0)
        feats.append(f)
    return np.vstack(feats)

def reduce_pca(X, n=50):
    return PCA(n_components=n, random_state=42).fit_transform(X)

def reduce_tsne(X):
    return TSNE(n_components=2, perplexity=40, random_state=42, max_iter=500).fit_transform(X)

def reduce_umap(X):
    return umap.UMAP(n_neighbors=20, min_dist=0.1, random_state=42).fit_transform(X)

def run_kmeans(X, k):
    return KMeans(n_clusters=k, random_state=42).fit_predict(X)

def run_dbscan(X, eps, min_samples):
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)

def cluster_analysis(labels, y_true, classes):
    result = []
    for cid in np.unique(labels):
        idx = np.where(labels == cid)[0]
        cname = "Outliers (DBSCAN)" if cid == -1 else f"Cluster {cid}"
        counts = {}
        for i in idx:
            cls = classes[y_true[i]]
            counts[cls] = counts.get(cls, 0) + 1
        result.append((cname, counts))
    return result

# ============================================================
# STREAMLIT APP
# ============================================================
def main():
    st.title("🔍 Unsupervised Clustering — CNN Features")
    st.markdown("""
    **Section 3.4 — Analyse Non Supervisée et Clustering**
    This tool allows you to extract features from pre-trained models and perform unsupervised clustering (K-Means and DBSCAN).
    You can also visualize the results in 2D using dimensionality reduction techniques (PCA, t-SNE, UMAP).
    """)

    npz_path = st.text_input("Dataset (NPZ)", "data/processed/processed_fer2013.npz")

    # ---------------------------
    # Feature source selection
    # ---------------------------
    existing_features = list_existing_features()
    trained_models = list_trained_models()

    if not trained_models:
        st.error("No trained .keras models found in experiments/dl_checkpoints/")
        return

    if existing_features:
        feature_mode = st.radio(
            "Choose feature source",
            ["Use existing features", "Extract new features"]
        )
    else:
        feature_mode = "Extract new features"

    # ---------------------------
    # Use existing features
    # ---------------------------
    if feature_mode == "Use existing features":
        selected_feature = st.selectbox("Select saved feature file", existing_features)
        feature_path = os.path.join(FEATURE_DIR, selected_feature)

    # ---------------------------
    # Extract features from model
    # ---------------------------
    else:
        model_name = st.selectbox("Choose trained model for feature extraction", trained_models)
        model_path = os.path.join(CHECKPOINT_DIR, model_name)
        feature_path = os.path.join(FEATURE_DIR, f"features_{model_name.replace('.keras','')}.npy")

        if st.button("Extract & Save Features"):
            X_train, X_test, y_train, y_test, classes = load_data(npz_path)
            X = np.concatenate([X_train, X_test])
            st.info(f"Images loaded: {X.shape[0]}")
            extractor = load_feature_extractor(model_path)
            features = extract_features(X, extractor)
            np.save(feature_path, features)
            st.success(f"Features saved to `{feature_path}`")
            st.session_state.features = features

    # ---------------------------
    # Load features
    # ---------------------------
    if not os.path.exists(feature_path):
        st.warning("Please extract or select features first.")
        return

    if "features" not in st.session_state:
        st.session_state.features = np.load(feature_path)
    features = st.session_state.features
    st.success(f"Features loaded: {features.shape}")

    _, _, y_train, y_test, classes = load_data(npz_path)
    y = np.concatenate([y_train, y_test])

    st.divider()

    # ---------------------------
    # K-Means Clustering
    # ---------------------------
    st.subheader("📌 K-Means Clustering")
    k = st.slider("Number of clusters", 2, 10, len(classes))

    if st.button("Run K-Means"):
        labels = run_kmeans(features, k)
        # PCA + t-SNE subsample for visualization
        if features.shape[0] > VISUALIZE_SAMPLES:
            sample_idx = np.random.choice(features.shape[0], VISUALIZE_SAMPLES, replace=False)
            X_vis = features[sample_idx]
            labels_vis = labels[sample_idx]
        else:
            X_vis = features
            labels_vis = labels

        X_vis = reduce_tsne(reduce_pca(X_vis))
        fig, ax = plt.subplots(figsize=(8,6))
        sc = ax.scatter(X_vis[:,0], X_vis[:,1], c=labels_vis, s=4, cmap="tab10")
        plt.colorbar(sc)
        ax.set_title("K-Means — Learned Feature Space")
        st.pyplot(fig)

        st.subheader("Cluster Analysis")
        cluster_results = cluster_analysis(labels, y, classes)
        st.table(cluster_results)  # Display as table for better readability

    st.divider()

    # ---------------------------
    # DBSCAN Clustering
    # ---------------------------
    st.subheader("📌 DBSCAN Clustering")
    eps = st.slider("eps", 0.1, 10.0, 2.5)
    min_samples = st.slider("min_samples", 1, 50, 5)

    if st.button("Run DBSCAN"):
        labels = run_dbscan(features, eps, min_samples)
        if features.shape[0] > VISUALIZE_SAMPLES:
            sample_idx = np.random.choice(features.shape[0], VISUALIZE_SAMPLES, replace=False)
            X_vis = features[sample_idx]
            labels_vis = labels[sample_idx]
        else:
            X_vis = features
            labels_vis = labels

        X_vis = reduce_umap(X_vis)
        fig, ax = plt.subplots(figsize=(8,6))
        sc = ax.scatter(X_vis[:,0], X_vis[:,1], c=labels_vis, s=4, cmap="tab10")
        plt.colorbar(sc)
        ax.set_title("DBSCAN — Learned Feature Space")
        st.pyplot(fig)

        st.subheader("Cluster Analysis")
        cluster_results = cluster_analysis(labels, y, classes)
        st.table(cluster_results)  # Display as table for better readability

# Required by navigation system
def render():
    main()  # Call the main function to start the app
