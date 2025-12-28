# ANALYSE EXPLORATOIRE DES DONNÉES

import matplotlib.pyplot as plt
import os
from PIL import Image
from datasets import get_dataloaders, check_dataset

def run_dataset_check(dataset_dir="dataset"):
    """
    Affiche la structure et la répartition du dataset
    """
    check_dataset(dataset_dir)

def inspect_dataloader(batch_size=16, dataset_dir="dataset"):
    """
    Vérifie les dimensions des images et labels
    """
    train_loader, _ = get_dataloaders(
        dataset_dir=dataset_dir,
        batch_size=batch_size
    )

    for images, labels in train_loader:
        print(f"Images shape : {images.shape}")
        print(f"Labels shape : {labels.shape}")
        break

def run_eda(dataset_dir="dataset", batch_size=16):
    """
    Lance l'EDA de base
    """
    print("=== Vérification du dataset ===")
    run_dataset_check(dataset_dir)

    print("\n=== Inspection du DataLoader ===")
    inspect_dataloader(batch_size, dataset_dir)

