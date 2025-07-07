#reporting.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from .helpers import compute_metrics, prettyPrintMetrics
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from .constants import TEST_DATA
from .feature_engineering import df_from_csv
from torch.utils.data import DataLoader



def evaluate_model(test_loader: DataLoader, model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    modelPerformance, y_true, y_preds  = compute_metrics(model, test_loader)
    prettyPrintMetrics(modelPerformance, "Performance: ", print_confusion_matrix=True, print_classification_report=True)
    return y_true, y_preds

def _infer_loader(model, loader, device):
    model.eval(); y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X).view(-1)
            prob   = torch.sigmoid(logits)
            y_prob.extend(prob.cpu().tolist())
            y_true.extend(y.cpu().tolist())
            y_pred.extend((prob >= 0.5).int().cpu().tolist())
    return np.array(y_true), np.array(y_pred), np.array(y_prob)


def plot_roc_curve(model, loader, device):
    y_true, _, y_prob = _infer_loader(model, loader, device)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score    = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False-Positive Rate"), plt.ylabel("True-Positive Rate")
    plt.title("ROC Curve"), plt.grid(True), plt.legend()
    plt.tight_layout(); plt.show()

    return y_true, y_prob


def plot_precision_recall_curve(y_true, y_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    plt.figure(figsize=(6,4))
    plt.plot(recall, precision)
    plt.xlabel("Recall"), plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.grid(True), plt.tight_layout(); plt.show()

    # Prediction-confidence histogram
    plt.figure(figsize=(6,4))
    plt.hist(y_prob, bins=50, color="steelblue", alpha=.8, edgecolor='white')
    plt.title("Prediction Confidence Histogram")
    plt.xlabel("Predicted Probability"), plt.ylabel("Frequency")
    plt.grid(True), plt.tight_layout(); plt.show()

    return precision, recall

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Phish (0)','Safe (1)'],
                yticklabels=['Phish (0)','Safe (1)'])
    plt.xlabel("Predicted"), plt.ylabel("Actual")
    plt.title("Confusion Matrix"); plt.tight_layout(); plt.show()
    return cm


def plot_batch_loss(model, loader, criterion, device):
    model.eval(); losses = []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            loss = criterion(model(X).view(-1), y)
            losses.append(loss.item())

    plt.figure(figsize=(8,5))
    plt.plot(losses, marker='o')
    plt.title("Batch-wise Loss (best checkpoint)")
    plt.xlabel("Batch #"), plt.ylabel("Loss")
    plt.grid(True), plt.tight_layout(); plt.show()
    return losses


def fp_fn_feature_distribution(
        df_or_path,              # pd.DataFrame  *or*  path/str to CSV
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        *,
        seed: int = 42,          # used only if we load from CSV
        sample_rows: int | None = None
    ) -> None:
    """
    Plot KDE distributions of selected features for False–Positives vs False–Negatives.
    
    Parameters
    ----------
    df_or_path : DataFrame | str | Path
        •  If a DataFrame is provided, it is used directly.  
        •  If a string / Path is provided, the CSV is loaded via `df_from_csv`.
    y_true, y_pred, y_prob : arrays
        Ground-truth labels, binary predictions, and prediction probabilities.
    seed : int
        RNG seed for reproducible sampling when `df_from_csv` is invoked.
    sample_rows : int | None
        Number of rows to sample from the CSV (pass `None` to load all).
    """
    # ------------------------------------------------------------------ #
    # Resolve the feature DataFrame
    # ------------------------------------------------------------------ #
    if isinstance(df_or_path, (str, Path)):        # a path → load CSV
        nrows = sample_rows if sample_rows is not None else 0
        features_df: pd.DataFrame = df_from_csv(df_or_path, nrows=nrows, seed=seed)
    elif isinstance(df_or_path, pd.DataFrame):     # already a DataFrame
        features_df = df_or_path.copy()
    else:
        raise TypeError("df_or_path must be a DataFrame or a CSV path")

    # ------------------------------------------------------------------ #
    # Merge predictions with features
    # ------------------------------------------------------------------ #
    res = features_df.reset_index(drop=True).copy()
    res["true_label"] = y_true
    res["pred_label"] = y_pred
    res["pred_prob"]  = y_prob

    fp = res[(res.true_label == 1) & (res.pred_label == 0)]   # Safe → Phish
    fn = res[(res.true_label == 0) & (res.pred_label == 1)]   # Phish → Safe

    # Pick a subset of informative numeric / binary columns
    features = ["body_count_of_words", "body_length", "url_count",
                "attachment_count", "has_url", "has_attachment", "pred_prob"]

    ncols, nrows = 3, int(np.ceil(len(features) / 3))
    plt.figure(figsize=(6 * ncols, 4 * nrows))

    idx = 1
    for feat in features:
        # skip if one group has zero variance (makes the KDE blow up)
        if fp[feat].nunique() <= 1 or fn[feat].nunique() <= 1:
            print(f"Skipping {feat!r} — zero variance in one group.")
            continue
        plt.subplot(nrows, ncols, idx); idx += 1
        sns.kdeplot(fp[feat], label="False Pos", color="crimson", fill=True)
        sns.kdeplot(fn[feat], label="False Neg", color="royalblue", fill=True)
        plt.title(feat.replace("_", " ").title())
        plt.xlabel(feat)
        plt.legend()

    plt.suptitle("Feature Distributions — False Positives vs False Negatives",
                 fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

