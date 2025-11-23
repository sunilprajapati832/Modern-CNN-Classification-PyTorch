import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloaders(data_dir, batch_size=16, img_size=224):

    # Image preprocessing (same for all CNN architectures)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # -----------------------
    # LOAD TRAIN + VALIDATION
    # -----------------------
    full_dataset = datasets.ImageFolder(data_dir, transform=transform)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    # -----------------------
    # DATALOADERS
    # -----------------------
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # Class names
    class_names = full_dataset.classes

    return train_loader, val_loader, class_names
