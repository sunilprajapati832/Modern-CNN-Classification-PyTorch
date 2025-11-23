import os
import cv2
import torch
import numpy as np
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt


def generate_gradcam(model, dataloader, class_names, device, result_dir):

    # Create output directory
    save_path = os.path.join(result_dir, "gradcam")
    os.makedirs(save_path, exist_ok=True)

    # Target layer (ResNet50 last block)
    target_layer = model.layer4[-1]

    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # Register hooks
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)

    model.eval()

    # Take only ONE batch for speed
    x, y = next(iter(dataloader))
    x = x.to(device)
    y = y.to(device)

    # Forward pass
    outputs = model(x)

    # Choose the max score from batch
    score = outputs.max(1)[0]
    score.sum().backward()

    # Extract saved activations & gradients
    acts = activations[-1]      # [B, C, H, W]
    grads = gradients[-1]       # [B, C, H, W]

    # Global average pooling (per-channel weight)
    weights = grads.mean(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]

    # Weighted sum — only first sample
    cam = (acts * weights).sum(dim=1)[0]  # [H, W]

    cam = cam.detach().cpu().numpy()
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam / (cam.max() + 1e-8)

    # Original image
    img = TF.to_pil_image(x[0].cpu())

    # Save heatmap
    plt.imshow(img)
    plt.imshow(cam, cmap="jet", alpha=0.4)
    plt.axis("off")
    plt.savefig(f"{save_path}/gradcam_sample.png")
    plt.close()

    print("✔ Grad-CAM saved:", f"{save_path}/gradcam_sample.png")
