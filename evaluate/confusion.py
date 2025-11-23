import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def save_confusion_matrix(labels, preds, class_names, result_dir):
    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(f"{result_dir}/confusion_matrix.png")
    plt.close()
