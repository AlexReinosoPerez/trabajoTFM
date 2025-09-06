import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter
import numpy as np

def get_data_loaders(data_dir, batch_size, use_sampler=False):
    """
    Crea los DataLoaders de entrenamiento y validación, con opción de aplicar muestreo ponderado.
    """
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

    class_names = train_dataset.classes

    targets = [label for _, label in train_dataset.samples]
    class_counts = dict(Counter(targets))
    total_samples = sum(class_counts.values())

    if use_sampler:
        print("🧪 Usando WeightedRandomSampler por desbalance de clases.")
        class_sample_counts = np.array([class_counts[i] for i in range(len(class_names))])
        weights = 1.0 / class_sample_counts
        samples_weight = np.array([weights[t] for t in targets])
        sampler = WeightedRandomSampler(weights=samples_weight, num_samples=len(samples_weight), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader, class_names, class_counts

