# FIGURES MANAGEMENT
import os
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

os.makedirs(FIGURES_DIR, exist_ok=True)

def save_training_curves(history):
    """
    Sauvegarde les courbes de loss et accuracy
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train loss")
    plt.plot(epochs, history["test_loss"], label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss over epochs")
    plt.savefig(os.path.join(FIGURES_DIR, "loss_curve.png"))
    plt.close()

    # Accuracy
    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train acc")
    plt.plot(epochs, history["test_acc"], label="Test acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy over epochs")
    plt.savefig(os.path.join(FIGURES_DIR, "accuracy_curve.png"))
    plt.close()

def save_confusion_matrix(model, dataloader, device, class_names):
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(FIGURES_DIR, "confusion_matrix.png"))
    plt.close()


def save_misclassified_images(
    model, dataloader, device, class_names, max_images=9
):
    model.eval()
    images_saved = 0

    plt.figure(figsize=(10, 10))

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    img = images[i].cpu().permute(1, 2, 0)
                    img = (img * 0.5 + 0.5).clamp(0, 1)

                    plt.subplot(3, 3, images_saved + 1)
                    plt.imshow(img)
                    plt.title(
                        f"GT: {class_names[labels[i]]}\nPred: {class_names[preds[i]]}"
                    )
                    plt.axis("off")

                    images_saved += 1
                    if images_saved >= max_images:
                        plt.tight_layout()
                        plt.savefig(os.path.join(FIGURES_DIR, "misclassified_images.png"))
                        plt.close()
                        return

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "misclassified_images.png"))
    plt.close()

def save_prediction_distribution(model, dataloader, device, class_names):
    model.eval()
    preds_all = []

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            preds_all.extend(preds.cpu().numpy())

    plt.figure()
    plt.hist(preds_all, bins=len(class_names), rwidth=0.8)
    plt.xticks(range(len(class_names)), class_names)
    plt.xlabel("Predicted class")
    plt.ylabel("Count")
    plt.title("Prediction distribution")
    plt.savefig(os.path.join(FIGURES_DIR, "prediction_distribution.png"))
    plt.close()


