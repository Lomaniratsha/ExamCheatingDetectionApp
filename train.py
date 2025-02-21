import os
import sys
from pathlib import Path
import torch

# Validate paths
def validate_path(path, description):
    """Ensures the given path exists."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{description} path not found: {path}")

# Training script
def train_yolo(custom_epochs=30, weights="yolov5l.pt"):
    """
    Trains a YOLOv5 model on a specified dataset.

    Parameters:
        custom_epochs (int): Number of epochs to train.
        weights (str): Path to the weights file (default is YOLOv5-L pre-trained weights).
    """
    # Define paths
    repo_path = './yolov5'
    data_yaml = '/content/drive/MyDrive/Dataset~1/4500+dataset/Yaml/data.yaml'
    project_path = '/content/drive/MyDrive/Dataset~1/4500+dataset/runs'

    # Validate paths
    validate_path(repo_path, "YOLOv5 repository")
    validate_path(data_yaml, "Dataset configuration YAML")

    # Add YOLOv5 to system path
    sys.path.insert(0, repo_path)

    try:
        from train import run
    except ImportError:
        raise ImportError("Failed to import YOLOv5 'train' module. Ensure the repository is correctly set up.")

    # Determine device
    if torch.cuda.is_available():
        device = '0'  # Use the first GPU
    else:
        device = 'cpu'

    # Training configuration
    config = {
        'data': data_yaml,
        'weights': weights,
        'epochs': custom_epochs,
        'batch_size': 16,
        'img': 640,
        'device': device,
        'project': project_path,
        'name': 'yolov5l_training',
        'exist_ok': True,  # Allows overwriting existing project folders
    }

    print(f"Starting training for {custom_epochs} epochs using YOLOv5-L on {device}.")

    try:
        results = run(**config)
        print("\nTraining completed successfully!")
        return results
    except Exception as e:
        print(f"Error during training: {e}")
        raise

# Call the training function
if __name__ == "__main__":
    train_yolo(custom_epochs=30)
