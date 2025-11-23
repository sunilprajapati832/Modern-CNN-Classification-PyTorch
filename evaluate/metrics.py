import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import json
import torch


def evaluate_basic_metrics(model, dataloader, class_names, device, result_dir):
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            _, preds = torch.max(out, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)

    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0      # <-- FIXED WARNING
    )

    with open(f"{result_dir}/classification_report.json", "w") as f:
        json.dump(report, f, indent=4)

    print(f"\nAccuracy: {acc * 100:.2f}%")

    return np.array(all_preds), np.array(all_labels)
