# dataset.py
# DATASET MANAGEMENT AND EXPLORATORY DATA ANALYSIS
# Gestion des fichiers et affichage
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_transforms():
    """Retourne les transformations des images"""
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

def get_train_transforms():
    """Transformations pour l'entraînement (avec data augmentation)"""
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

def get_test_transforms():
    """Transformations pour le test (sans augmentation)"""
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

def load_datasets(dataset_dir="dataset"):
    """Charge les datasets train et test
    """
    # transform = get_transforms()
    train_transform = get_train_transforms()
    test_transform = get_test_transforms()

    dataset_path = os.path.join(PROJECT_ROOT, dataset_dir)

    train_data = datasets.ImageFolder(
        os.path.join(dataset_path, "train"),
        transform=train_transform
    )

    test_data = datasets.ImageFolder(
        os.path.join(dataset_path, "test"),
        transform=test_transform
    )

    return train_data, test_data

def get_dataloaders(dataset_dir="dataset", batch_size=16):
    """
    Retourne les DataLoaders train et test
    """
    train_data, test_data = load_datasets(dataset_dir)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False
    )
    return train_loader, test_loader

def check_dataset(dataset_dir="dataset", classes=("jackson", "divers")):
    """
    Affiche des infos sur la structure et la taille du dataset
    """
    dataset_path = os.path.join(PROJECT_ROOT, dataset_dir)

    for split in ["train", "test"]:
        for cls in classes:
            path = os.path.join(dataset_path, split, cls)

            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing folder: {path}")

            count = len(os.listdir(path))
            print(f"{split}/{cls}: {count} images")
