from sklearn.metrics import precision_recall_curve, auc, roc_curve
import numpy as np

def calculate_roc_auc(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    return auc(fpr, tpr)

def metrics_evaluation(eval_pred):
    predictions, labels = eval_pred
    np.set_printoptions(precision=4)
    y_pred = 1 / (1 + np.exp(-predictions.squeeze()))
    y_true = labels
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    roc_auc = calculate_roc_auc(y_true, y_pred)
    return {"pr_auc": pr_auc, "roc_auc": roc_auc}