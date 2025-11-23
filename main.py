import argparse
import torch
from utils.dataset_loader import get_dataloaders
from models.model_builder import build_model
from train import train_model


def main():
    parser = argparse.ArgumentParser(description="Modern CNN Classification - Caltech101")

    parser.add_argument("--model", type=str, default="resnet50",
                        choices=["resnet50", "vgg16", "mobilenetv2", "efficientnet_b0"],
                        help="Choose which model to train")

    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--data", type=str, default="data/Caltech101/101_ObjectCategories",
                        help="Dataset directory path")

    args = parser.parse_args()

    # ----------------------------
    # STEP 1: DEVICE SELECTION
    # ----------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}\n")

    # ----------------------------
    # STEP 2: LOAD DATASET
    # ----------------------------
    train_loader, val_loader, class_names = get_dataloaders(
        args.data, batch_size=args.batch
    )

    # ----------------------------
    # STEP 3: BUILD MODEL
    # ----------------------------
    model = build_model(args.model, num_classes=len(class_names))
    print(f"Model selected → {args.model}")

    # ----------------------------
    # STEP 4: TRAIN MODEL
    # ----------------------------
    trained_model = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        device=device
    )

    # ----------------------------
    # STEP 5: SAVE MODEL
    # ----------------------------
    save_path = f"saved_models/{args.model}_caltech101.pth"
    torch.save(trained_model.state_dict(), save_path)

    print(f"\nModel saved at: {save_path}")


if __name__ == "__main__":
    main()
