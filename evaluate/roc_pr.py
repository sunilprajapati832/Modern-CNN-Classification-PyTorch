import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import numpy as np


def save_roc_pr_curves(labels, preds, class_names, result_dir):
    labels = np.array(labels)
    preds = np.array(preds)

    plt.figure()
    fpr, tpr, _ = roc_curve(labels, preds, pos_label=1)
    plt.plot(fpr, tpr)
    plt.title("ROC Curve")
    plt.savefig(f"{result_dir}/roc_curve.png")
    plt.close()

    plt.figure()
    pr, rc, _ = precision_recall_curve(labels, preds, pos_label=1)
    plt.plot(rc, pr)
    plt.title("Precision-Recall Curve")
    plt.savefig(f"{result_dir}/pr_curve.png")
    plt.close()
