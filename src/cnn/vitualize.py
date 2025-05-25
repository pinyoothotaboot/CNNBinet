import torch
import matplotlib.pyplot as plt


def show_predictions(model, testloader, device, img_file="prediction_examples.png"):
    """
    Display and save a visual comparison of true vs predicted labels for a batch of test images.

    Args:
        model (torch.nn.Module): Trained model for prediction.
        testloader (DataLoader): DataLoader containing the test dataset.
        device (torch.device): The device (CPU or CUDA) to perform computation on.
        img_file (str): Path to save the output image (default: 'prediction_examples.png').

    Returns:
        None

    Description:
        - Fetches a single batch of images from the test set.
        - Runs the model in evaluation mode to get predictions.
        - Plots 10 images with both ground truth and predicted labels.
        - Saves the resulting figure to the specified file.
    """
    model.eval()
    images, labels = next(iter(testloader))
    images, labels = images.to(device), labels.to(device)

    outputs = model(images)
    _, preds = torch.max(outputs, 1)

    for i in range(10):
        plt.subplot(2, 5, i + 1)
        img = images[i].cpu().squeeze()
        plt.imshow(img, cmap="gray")
        title = f"T:{labels[i].item()}/P:{preds[i].item()}"
        color = "green" if labels[i] == preds[i] else "red"
        plt.title(title, color=color, fontsize=10)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(img_file)
    print(f"[✔] Saved prediction image to {img_file}")


def plot_train_data(train_loss_list, test_acc_list, img_file="train_plot.png"):
    """
    Plot and save training loss and test accuracy per epoch.

    Args:
        train_loss_list (List[float]): List of average training loss values per epoch.
        test_acc_list (List[float]): List of test accuracy values per epoch.
        img_file (str): Path to save the output plot (default: 'train_plot.png').

    Returns:
        None

    Description:
        - Creates two side-by-side plots:
            1. Training Loss over epochs
            2. Test Accuracy over epochs
        - Saves the resulting figure to the specified file.
        - Useful for visualizing training performance and detecting overfitting.
    """
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_list, label="Train Loss")
    plt.title("Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(test_acc_list, label="Test Accuracy")
    plt.title("Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(img_file)
    print(f"[📊] Saved training graph to {img_file}")
    plt.close()
