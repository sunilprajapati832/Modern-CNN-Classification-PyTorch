import torch
from torchvision import datasets, transforms
from models.model_builder import build_model
from evaluate.gradcam import generate_gradcam
import argparse


def load_dataloader(data_dir):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    return loader, dataset.classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model_name", default="resnet50")
    parser.add_argument("--save_dir", default="gradcam_results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n➡ Using device: {device}")

    dataloader, class_names = load_dataloader(args.data_dir)

    model = build_model(args.model_name, num_classes=len(class_names))
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()

    print("\n📌 Generating Grad-CAM... (Only this step, nothing else)")
    generate_gradcam(model, dataloader, class_names, device, args.save_dir)

    print(f"\n🎉 Grad-CAM saved in: {args.save_dir}/gradcam\n")


if __name__ == "__main__":
    main()
