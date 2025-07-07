#helpers.py
import torch
import torch.nn as nn

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, \
    recall_score, f1_score


def save_model(model_state, path, loss, silent=False):
    torch.save(model_state, path)
    if not silent:
        print(f"\n Saved model at path: {path} with val_loss = {loss:.4f})")


def compute_metrics(model, test_loader, epochDetails='', device='cpu'):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    y_true = []
    y_pred = []
    total_loss = 0.0
    num_batches = 0
    summary = dict()

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch).view(-1)
            loss = criterion(outputs, y_batch.float())
            total_loss += loss.item()
            num_batches += 1

            # Computations
            preds = (torch.sigmoid(outputs) >= 0.5).int()
            y_true.extend(y_batch.cpu().int().tolist())
            y_pred.extend(preds.cpu().tolist())

    # Other metrices
    summary['average_loss'] = total_loss / num_batches if num_batches > 0 else 0.0
    summary['accuracy'] = accuracy_score(y_true, y_pred)
    summary['precision'] = precision_score(y_true, y_pred, zero_division=0)
    summary['recall'] = recall_score(y_true, y_pred, zero_division=0)
    summary['f1'] = f1_score(y_true, y_pred, zero_division=0)
    summary['conf_matrix'] = confusion_matrix(y_true, y_pred)
    summary['report'] = classification_report(y_true, y_pred, zero_division=0)
    return summary, y_true, y_pred

def prettyPrintMetrics(summary: dict, 
                       message: str = "", 
                       print_confusion_matrix: bool = False, 
                       print_classification_report: bool = False) -> None:
    print(f''' {message} \
    Loss: {summary['average_loss']:.4f}\
    | Accuracy: {summary['accuracy']*100:.2f}%\
    | Precision: {summary['precision']:.4f}\
    | Recall: {summary['recall']:.4f}\
    | F1-score: {summary['f1']:.4f}
    ''')
    if print_confusion_matrix:
        print("\n Confusion Matrix:\n", summary['conf_matrix'])
    if print_classification_report:
        print("\n Classification Report:\n", summary['report'])

def get_feature_count(train_loader):
    header, _ = train_loader.dataset[0]
    return header.numel()