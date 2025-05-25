import torch
import torchvision.transforms as T
from PIL import Image
import argparse
import os
import sys
import logging
from cnn.layer import BinaryCNN
from cnn.model import load_model

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("train.log")],
)
log = logging.info

transform = T.Compose(
    [T.Grayscale(), T.Resize((28, 28)), T.ToTensor(), T.Normalize((0.1307,), (0.3081,))]
)


def predict_image(image_path, model_path="binary_mnist.pth"):
    """
    Predict the class of a single image using a trained BinaryCNN model.

    Args:
        image_path (str): Path to the input image file (e.g., PNG or JPG).
        model_path (str): Path to the saved PyTorch model weights (.pth file).

    Returns:
        int: The predicted class index (e.g., 0-9 for MNIST).

    Raises:
        FileNotFoundError: If the input image file does not exist.

    Description:
        - Loads the specified image and applies preprocessing (grayscale, resize, normalize).
        - Loads the trained BinaryCNN model with preloaded weights.
        - Runs inference on the image without gradient computation.
        - Returns the predicted class index as an integer.

    Notes:
        - Assumes `BinaryCNN` class and `load_model()` function are properly imported.
        - Requires `transform` to be defined globally or accessible within this scope.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Image not found: {image_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = BinaryCNN().to(device)
    model = load_model(model, path=model_path, device=device)
    model.eval()

    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        _, pred = torch.max(output, 1)

    return pred.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict MNIST digit with trained BinaryCNN")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--model", type=str, default="binary_mnist.pth", help="Path to trained model"
    )
    args = parser.parse_args()

    try:
        prediction = predict_image(args.image, args.model)
        log(f"[✅] Predicted digit: {prediction}")
    except Exception as e:
        log(f"[❌] Error: {e}")
    except KeyboardInterrupt:
        log("Exit CLI Predict")
        sys.exit()
