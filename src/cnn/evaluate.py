import torch


def evaluate(model, testloader, device, silent=False):
    """
    Evaluate model accuracy on a test dataset.

    Args:
        model (torch.nn.Module): The trained PyTorch model to evaluate.
        testloader (DataLoader): DataLoader containing test dataset.
        device (torch.device): Device to run the evaluation on (e.g., 'cuda' or 'cpu').
        silent (bool): If True, suppresses printed accuracy output.

    Returns:
        float: Accuracy of the model on the test dataset (in percentage, 0–100).

    Description:
        - Sets the model to eval mode.
        - Disables gradient tracking for faster inference.
        - Computes the number of correct predictions.
        - Calculates and returns test accuracy.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    if not silent:
        print(f"[✔] Test Accuracy: {acc:.2f}%")
    return acc
