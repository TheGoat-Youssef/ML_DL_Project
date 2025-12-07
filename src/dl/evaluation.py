import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def compute_metrics(y_true, y_pred, labels=None):
    """
    y_true: integer labels or one-hot
    y_pred: either logits/probs or integer preds
    returns classification_report dict
    """
    if y_true.ndim == 2:
        y_true_idx = np.argmax(y_true, axis=1)
    else:
        y_true_idx = y_true
    if y_pred.ndim == 2:
        y_pred_idx = np.argmax(y_pred, axis=1)
    else:
        y_pred_idx = y_pred
    report = classification_report(y_true_idx, y_pred_idx, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true_idx, y_pred_idx)
    return report, cm

def plot_confusion(cm, classes, out_path=None, figsize=(7,6)):
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted"); plt.ylabel("True")
    if out_path:
        plt.savefig(out_path, bbox_inches="tight")
    plt.close()
