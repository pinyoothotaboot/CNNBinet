import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import logging
import os
import sys
import argparse
from cnn.layer import BinaryCNN
from cnn.evaluate import evaluate
from cnn.vitualize import show_predictions, plot_train_data
from cnn.model import load_model, save_model

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("train.log")],
)
log = logging.info


def train(epochs=200, test_only=False, patience=5, lr=1e-3, batch_size=2048):
    """
    Train or evaluate a BinaryCNN model on the MNIST dataset.

    Args:
        epochs (int): Number of training epochs (default: 200).
        test_only (bool): If True, skip training and run model evaluation only (default: False).
        patience (int): Number of epochs to wait for improvement before early stopping (default: 5).
        lr (float): Learning rate for the optimizer (default: 1e-3).
        batch_size (int): Batch size for training and testing (default: 2048).

    Returns:
        None

    Description:
        - Applies data augmentation to training data and normalization to both training and test sets.
        - Loads the MNIST dataset and prepares DataLoaders.
        - Loads a previously saved model if available.
        - If `test_only` is True:
            - Evaluates the model and visualizes predictions only.
        - Otherwise:
            - Trains the model using AdamW optimizer and ReduceLROnPlateau scheduler.
            - Uses CrossEntropyLoss.
            - Tracks training loss and test accuracy per epoch.
            - Applies early stopping based on best test accuracy.
            - Saves the model with the highest achieved accuracy.
            - Plots training loss and accuracy trends over epochs.

    Notes:
        - The model is trained on MNIST, but can be adapted to other datasets by modifying DataLoaders and label counts.
        - Assumes `BinaryCNN`, `load_model`, `save_model`, `evaluate`, `show_predictions`, and `plot_train_data` are defined and imported.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[⚙] Using device: {device}")

    transform_train = T.Compose(
        [
            T.RandomRotation(10),
            T.RandomAffine(0, translate=(0.2, 0.2), scale=(0.9, 1.1)),
            T.RandomPerspective(distortion_scale=0.2, p=0.5),
            T.ToTensor(),
            T.Normalize((0.1307,), (0.3081,)),
        ]
    )
    transform_test = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])

    trainset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform_test
    )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    model = BinaryCNN().to(device)

    if os.path.exists("binary_mnist.pth"):
        log("[ℹ] Found existing model.")
        model = load_model(model, device=device)

    if test_only:
        evaluate(model, testloader, device)
        show_predictions(model, testloader, device)
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    criterion = nn.CrossEntropyLoss()
    best_acc = evaluate(model, testloader, device, silent=True)
    epochs_no_improve = 0

    train_loss_list = []
    test_acc_list = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(trainloader)
        acc = evaluate(model, testloader, device, silent=True)

        train_loss_list.append(avg_loss)
        test_acc_list.append(acc)

        log(f"[Epoch {epoch + 1}/{epochs}] Loss = {avg_loss:.4f} | Accuracy = {acc:.2f}%")

        scheduler.step(acc)

        if acc > best_acc:
            best_acc = acc
            save_model(model)
            log(f"[🎉] New best model saved! Accuracy = {best_acc:.2f}%")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            log(f"[📉] No improvement for {epochs_no_improve} epoch(s)")

        if epochs_no_improve >= patience:
            log(f"[⛔] Early stopping triggered. Best Accuracy = {best_acc:.2f}%")
            break

    log(f"[🏁] Training complete. Best Accuracy = {best_acc:.2f}%")
    show_predictions(model, testloader, device)
    plot_train_data(train_loss_list, test_acc_list)


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Train Binary CNN on MNIST")

        parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
        parser.add_argument("--test-only", action="store_true", help="Run in test mode only")
        parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
        parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
        parser.add_argument("--batch-size", type=int, default=2048, help="Training batch size")

        args = parser.parse_args()

        train(
            epochs=args.epochs,
            test_only=args.test_only,
            patience=args.patience,
            lr=args.lr,
            batch_size=args.batch_size,
        )
    except KeyboardInterrupt:
        log("Exit CLI Training")
        sys.exit()
    except Exception as ex:
        log(f"Error : {ex}")
