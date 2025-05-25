import torch


def save_model(model, path="binary_mnist.pth"):
    """
    Save the state dictionary (weights) of a PyTorch model to a file.

    Args:
        model (torch.nn.Module): The PyTorch model to save.
        path (str): File path to save the model (default: 'binary_mnist.pth').

    Returns:
        None

    Description:
        This function saves only the model's weights (not the full model).
        To reload, the model architecture must be instantiated beforehand.
    """
    torch.save(model.state_dict(), path)
    print(f"[✔] Model saved to: {path}")


def load_model(model, path="binary_mnist.pth", device="cpu"):
    """
    Load weights (state_dict) into a PyTorch model from a saved file.

    Args:
        model (torch.nn.Module): A PyTorch model instance with the same architecture as the saved one.
        path (str): Path to the saved state_dict file.
        device (str or torch.device): Device to map the model to (e.g., 'cpu' or 'cuda').

    Returns:
        torch.nn.Module: The model with loaded weights.

    Description:
        This function assumes the model architecture is already defined and matches the saved weights.
        It loads weights using map_location to ensure compatibility with the current device.
    """
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"[✔] Model loaded from: {path}")
    return model
