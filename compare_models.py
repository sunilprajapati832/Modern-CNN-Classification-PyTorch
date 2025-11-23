import argparse
import time
import torch
from torchvision import datasets, transforms
import os
from models.model_zoo import get_model_zoo
from evaluate.metrics import evaluate_basic_metrics
from evaluate.topk import compute_topk
from evaluate.save_utils import create_result_dir
import pandas as pd


def load_dataloader(data_dir, batch=32):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=False)
    return loader, dataset.classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--custom_model", required=True)
    parser.add_argument("--save_dir", default="model_comparison")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n➡ Running on: {device}")

    # Load data
    dataloader, class_names = load_dataloader(args.data_dir)
    num_classes = len(class_names)

    # Load all models (ResNet50, EfficientNet, MobileNet, ViT + custom)
    model_zoo = get_model_zoo(num_classes, args.custom_model, device=device)

    # Root result directory
    result_dir = create_result_dir(args.save_dir, "comparison")

    result_list = []

    for name, model in model_zoo.items():
        print(f"\n🔍 Evaluating {name} ...")
        model = model.to(device)
        model.eval()

        # Create a sub-folder for each model results
        model_dir = os.path.join(result_dir, name)
        os.makedirs(model_dir, exist_ok=True)

        # Measure evaluation time
        start = time.time()

        # FIXED: Added missing result_dir
        preds, labels = evaluate_basic_metrics(
            model, dataloader, class_names, device, model_dir
        )

        end = time.time()
        duration = end - start

        # Top-k accuracy
        top1, top5 = compute_topk(model, dataloader, device, class_names, model_dir)

        # Save results
        result_list.append({
            "Model": name,
            "Top-1 Accuracy": round(top1, 4),
            "Top-5 Accuracy": round(top5, 4),
            "Inference Time (s)": round(duration, 3),
            "Parameters (M)": round(sum(p.numel() for p in model.parameters()) / 1e6, 2)
        })

    # Convert to CSV
    df = pd.DataFrame(result_list)
    csv_path = f"{result_dir}/model_comparison.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n📄 Saved: {csv_path}")
    print("\n🎉 Model comparison complete!\n")


if __name__ == "__main__":
    main()
