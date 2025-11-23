import torch
import torchvision.models as models


def load_resnet50(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def load_efficientnet_b0(num_classes):
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
    )
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def load_mobilenet_v3(num_classes):
    model = models.mobilenet_v3_large(
        weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
    )
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)
    return model


def load_vit_small(num_classes):
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
    return model


def get_model_zoo(num_classes, custom_model_path=None, device="cpu"):
    """
    Returns a dictionary of models:
    - pretrained ResNet50
    - pretrained EfficientNet-B0
    - pretrained MobileNetV3-Large
    - pretrained ViT-B16
    - your OWN ResNet50 (with weights)
    """

    zoo = {
        "resnet50_pretrained": load_resnet50(num_classes),
        "efficientnet_b0": load_efficientnet_b0(num_classes),
        "mobilenet_v3": load_mobilenet_v3(num_classes),
        "vit_b16": load_vit_small(num_classes),
    }

    if custom_model_path:
        custom = load_resnet50(num_classes)

        # Load checkpoint safely
        state_dict = torch.load(custom_model_path, map_location=device, weights_only=True)


        # --- FIX CLASSIFIER SIZE MISMATCH ---
        # Remove old fc layer (trained on 102 classes)
        if "fc.weight" in state_dict:
            del state_dict["fc.weight"]
        if "fc.bias" in state_dict:
            del state_dict["fc.bias"]

        # Load remaining parameters
        custom.load_state_dict(state_dict, strict=False)

        zoo["resnet50_custom"] = custom

    return zoo
