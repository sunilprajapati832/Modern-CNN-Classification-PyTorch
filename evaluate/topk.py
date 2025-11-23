import torch
import json
import os

def compute_topk(model, dataloader, device, class_names, result_dir):

    total = 0
    top1 = 0
    top5 = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        out = model(x)

        _, pred1 = out.topk(1, dim=1)
        _, pred5 = out.topk(5, dim=1)

        top1 += (pred1.squeeze() == y).sum().item()
        top5 += sum(y[i].item() in pred5[i].tolist() for i in range(len(y)))

        total += y.size(0)

    results = {
        "top1_accuracy": top1 / total,
        "top5_accuracy": top5 / total
    }

    # save results
    with open(f"{result_dir}/topk.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\nTop-1 Accuracy:", results["top1_accuracy"])
    print("Top-5 Accuracy:", results["top5_accuracy"])

    return results["top1_accuracy"], results["top5_accuracy"]
