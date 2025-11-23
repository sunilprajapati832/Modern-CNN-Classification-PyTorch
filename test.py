import argparse
import torch
from torchvision import datasets, transforms
from models.model_builder import build_model
from evaluate.metrics import evaluate_basic_metrics
from evaluate.confusion import save_confusion_matrix
from evaluate.roc_pr import save_roc_pr_curves
from evaluate.misclassified import save_misclassified_images
from evaluate.topk import compute_topk
from evaluate.gradcam import generate_gradcam
from evaluate.save_utils import create_result_dir


def load_dataloader(data_dir, batch=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=False)
    return loader, dataset.classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model_name", default="resnet50")
    parser.add_argument("--save_dir", default="results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n➡ Using device: {device}")

    dataloader, class_names = load_dataloader(args.data_dir)

    model = build_model(args.model_name, num_classes=len(class_names))
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()

    result_dir = create_result_dir(args.save_dir, args.model_name)

    print("\n📌 Running basic evaluation...")
    preds, labels = evaluate_basic_metrics(model, dataloader, class_names, device, result_dir)

    print("\n📌 Saving confusion matrix...")
    save_confusion_matrix(labels, preds, class_names, result_dir)

    print("\n📌 Saving ROC & PR curves...")
    save_roc_pr_curves(labels, preds, class_names, result_dir)

    print("\n📌 Saving misclassified images...")
    save_misclassified_images(model, dataloader, class_names, device, result_dir)

    print("\n📌 Computing Top-1 and Top-5 accuracy...")
    compute_topk(model, dataloader, device, class_names, result_dir)

    print("\n📌 Generating Grad-CAM visualizations...")
    generate_gradcam(model, dataloader, class_names, device, result_dir)

    print(f"\n🎉 All results saved in: {result_dir}\n")


if __name__ == "__main__":
    main()
