import os
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt


def save_misclassified_images(model, dataloader, class_names, device, result_dir):
    save_path = os.path.join(result_dir, "misclassified")
    os.makedirs(save_path, exist_ok=True)

    model.eval()

    idx = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            out = model(x)
            _, preds = torch.max(out, 1)

            for img, label, pred in zip(x, y, preds):
                if label != pred:
                    plt.imshow(TF.to_pil_image(img.cpu()))
                    plt.title(f"GT: {class_names[label]} | Pred: {class_names[pred]}")
                    plt.axis("off")
                    plt.savefig(f"{save_path}/{idx}.png")
                    plt.close()
                    idx += 1
