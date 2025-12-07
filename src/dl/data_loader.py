import numpy as np
from sklearn.model_selection import train_test_split

def load_processed_npz(path="data/processed/processed_fer2013.npz", test_size=0.2, random_state=42):
    """
    Load dataset saved by your preprocessing pipeline and return train/test splits.
    Expects keys: X, y, classes
    Converts string labels to integer indices and normalizes images to [0,1].
    Returns: X_train, X_test, y_train, y_test, classes (list)
    """
    data = np.load(path, allow_pickle=True)
    X = data["X"]            # (N,H,W,3)
    y = data["y"]            # array of strings
    classes = list(data["classes"])

    # map labels to int
    label_to_id = {c: i for i, c in enumerate(classes)}
    y_int = np.array([label_to_id[str(lbl)] for lbl in y], dtype=np.int32)

    # normalize images
    X = X.astype("float32") / 255.0

    # if grayscale images provided as (N,H,W), expand to 3 channels
    if X.ndim == 3:
        X = np.stack([X, X, X], axis=-1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_int, test_size=test_size, random_state=random_state, stratify=y_int
    )

    return X_train, X_test, y_train, y_test, classes
