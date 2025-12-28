# MAIN

import torch
import torch.nn as nn
import torch.optim as optim

from datasets import get_dataloaders, check_dataset
from eda import run_eda
from model import get_model
from train import train_model

from figures import (
    save_training_curves,
    save_confusion_matrix,
    save_misclassified_images,
    save_prediction_distribution
)


def main():
    print("=== PROJECT START ===")

    # =====================
    # 1. Dataset check
    # =====================
    print("\n[1] Checking dataset...")
    check_dataset()

    # =====================
    # 2. EDA
    # =====================
    print("\n[2] Running EDA...")
    run_eda(batch_size=16)

    # =====================
    # 3. Load data
    # =====================
    print("\n[3] Loading dataloaders...")
    train_loader, test_loader = get_dataloaders(batch_size=16)

    for images, labels in train_loader:
        print("Batch shape:", images.shape, labels.shape)
        break

    # =====================
    # 4. Model & training
    # =====================
    print("\n[4] Initializing model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(model_name="optimized", device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print("\n[5] Training model...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        n_epochs=20
    )

    print("\n=== TRAINING DONE ===")
    torch.save(model.state_dict(), "model.pth")

    print("\n[6] Saving curves...")
    class_names = ["jackson", "divers"]
    
    save_training_curves(history)
    save_confusion_matrix(model, test_loader, device, class_names)
    save_misclassified_images(model, test_loader, device, class_names)
    save_prediction_distribution(model, test_loader, device, class_names)

if __name__ == "__main__":
    main()
