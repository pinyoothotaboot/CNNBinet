# CNNBitnet: Binary CNN for MNIST Classification

CNNBitnet is a PyTorch implementation of a binary convolutional neural network (BinaryCNN) designed for efficient image classification on the MNIST dataset. The model utilizes binarized weights in its convolutional and linear layers, making it memory- and energy-efficient while maintaining good classification performance.

## Features

- **Binary Convolutional Layers**: Implements custom binary convolution operations for efficient computation
- **Batch Normalization**: Each convolutional layer is followed by batch normalization for stable training
- **Hybrid Binary Weights**: Uses sign function for binarization with scaling to maintain model capacity
- **Modular Architecture**: Clean separation of model definition, training, and evaluation
- **Command Line Interface**: Easy-to-use CLI for training and inference
- **Visualization**: Includes utilities for visualizing model predictions and training progress

## Model Architecture

The BinaryCNN consists of:
- 6 binary convolutional layers with ReLU activation and batch normalization
- Max pooling for spatial dimension reduction
- Adaptive average pooling before the final classification layer
- Dropout for regularization
- Binary fully connected layers with batch normalization

## Requirements

- Python 3.8+
- PyTorch 2.5.1 (with CUDA 12.1 support recommended)
- torchvision
- matplotlib

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/pinyoothotaboot/CNNBitnet.git
   cd CNNBitnet
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

To train the model:
```bash
python -m src.cli_train --epochs 200 --batch_size 2048 --lr 1e-3
```

### Evaluation

To evaluate a pre-trained model:
```bash
python -m src.cli_train --test_only --model_path binary_mnist.pth
```

### Prediction

To make predictions on new images:
```bash
python -m src.cli_predict --image path/to/image.png --model_path binary_mnist.pth
```

## Project Structure

```
CNNBitnet/
├── data/                  # Directory for storing datasets
├── src/                    # Source code
│   ├── cnn/                # Core CNN implementation
│   │   ├── bitnet.py       # Binary layer implementations
│   │   ├── evaluate.py     # Model evaluation utilities
│   │   ├── layer.py        # Model architecture definition
│   │   ├── model.py        # Model loading/saving utilities
│   │   └── visualize.py    # Visualization utilities
│   ├── cli_predict.py      # Command line interface for inference
│   └── cli_train.py        # Command line interface for training
├── binary_mnist.pth        # Pre-trained model weights
├── requirements.txt        # Production dependencies
├── requirements-dev.txt    # Development dependencies
└── README.md              # This file
```

## Results

The model achieves competitive accuracy on the MNIST dataset while using binarized weights, making it more efficient than full-precision models.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- PyTorch Team for the amazing deep learning framework
- MNIST dataset creators
- Researchers who pioneered binary neural networks

## Author

__Pinyoo Yothotaboot__
