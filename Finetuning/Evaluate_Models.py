import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score
import os
import numpy as np
from PIL import Image
import h5py
import pandas as pd
import random
from tqdm import tqdm  # For progress bars

# ============================ Reproducibility Setup ============================
random.seed(432)
np.random.seed(432)
torch.manual_seed(432)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(432)

# ============================ Configuration Parameters ============================

# Paths to data directories
TEST_DIR = ""  # Update with your test data path
TEST_HDF5 = os.path.join(TEST_DIR, 'test_stroke_dataset.h5')  # Update with your test HDF5 file path

# Directories for fine-tuned models and predictions
NEW_MODELS_DIR = "fine_tuned_models"        # Directory where fine-tuned models are saved
PREDICTIONS_DIR = "predictions/fine_tuned_models"  # Directory to save prediction CSVs

# Hyperparameters
TEST_BATCH_SIZE = 32
NUM_WORKERS = 8

# Ensure the predictions directory exists
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

# ============================ Data Transformations ============================

# Data transformations for testing (only resizing and normalization)
test_transforms = transforms.Compose([
    transforms.Resize((299, 299)),          # InceptionV3 input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ============================ Custom Dataset for HDF5 Files ============================

class HDF5Dataset(Dataset):
    def __init__(self, h5_file, transform=None):
        """
        Initializes the dataset by storing the path to the HDF5 file and the transformations.

        Args:
            h5_file (str): Path to the HDF5 file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.h5_file = h5_file
        self.transform = transform
        with h5py.File(self.h5_file, 'r') as hf:
            self.length = len(hf['images'])

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return self.length

    def __getitem__(self, idx):
        """
        Retrieves the image, label, and image path at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image, label, image_path)
        """
        with h5py.File(self.h5_file, 'r') as hf:
            image = hf['images'][idx]  # Access the image by index
            label = hf['labels'][idx]  # Access the corresponding label
            image_path_bytes = hf['image_paths'][idx]
            # Decode image_path from bytes to string if necessary
            image_path = image_path_bytes.decode('utf-8') if isinstance(image_path_bytes, bytes) else image_path_bytes

        # Map labels to binary (Assuming binary classification)
        # if label in [0, 1]:
        #     label = 0
        # else:
        #     label = 1

        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            image = Image.fromarray(image)
            image = self.transform(image)

        return image, label, image_path

def get_hdf5_dataloader(h5_file, batch_size=32, num_workers=4, transforms=None, shuffle=False):
    """
    Creates a DataLoader for the HDF5 dataset.

    Args:
        h5_file (str): Path to the HDF5 file.
        batch_size (int, optional): Number of samples per batch.
        num_workers (int, optional): Number of subprocesses for data loading.
        transforms (callable, optional): Optional transform to be applied on a sample.
        shuffle (bool, optional): Whether to shuffle the data.

    Returns:
        DataLoader: PyTorch DataLoader for the dataset.
    """
    dataset = HDF5Dataset(h5_file=h5_file, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return dataloader

# ============================ Evaluation Function ============================

def evaluate_model(model, dataloader, device):
    """
    Evaluates the model on the given dataloader.

    Args:
        model (torch.nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the evaluation data.
        device (torch.device): Device to perform computation on.

    Returns:
        tuple: (image_paths, probs, preds, labels)
    """
    model.eval()
    all_image_paths = []
    all_labels = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels, image_paths in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs = inputs.to(device)
            labels = labels.float().to(device)
            outputs = model(inputs)

            # If model has auxiliary logits (InceptionV3), use the primary output
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
            preds = (probs > 0.5).astype(int)

            # Handle image_paths decoding if necessary
            decoded_image_paths = [path.decode('utf-8') if isinstance(path, bytes) else path for path in image_paths]

            all_image_paths.extend(decoded_image_paths)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    return all_image_paths, np.array(all_probs), np.array(all_preds), np.array(all_labels)

# ============================ Model Initialization Function ============================

def initialize_model(pretrained_weights_path, device):
    """
    Initializes the InceptionV3 model, loads pre-trained weights, and sets up for evaluation.

    Args:
        pretrained_weights_path (str): Path to the pre-trained model weights.
        device (torch.device): Device to load the model onto.

    Returns:
        nn.Module: The initialized model ready for evaluation.
    """
    # Initialize the InceptionV3 model
    model = models.inception_v3(weights=None, aux_logits=True)

    # Replace the fully connected layers for binary classification
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, 1)

    # Load pre-trained weights
    try:
        model.load_state_dict(torch.load(pretrained_weights_path, map_location=device))
        print(f"Pre-trained weights loaded successfully from {pretrained_weights_path}.")
    except Exception as e:
        print(f"Error loading pre-trained weights from {pretrained_weights_path}: {e}")
        print("Proceeding without loading pre-trained weights.")

    # Move the model to the appropriate device
    model = model.to(device)

    return model

# ============================ Main Evaluation Loop ============================

def main():
    """
    Main function to evaluate multiple fine-tuned models and save their predictions to CSV files.
    """
    # Get the test dataloader
    test_loader = get_hdf5_dataloader(
        h5_file=TEST_HDF5,
        batch_size=TEST_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        transforms=test_transforms,
        shuffle=False  # Important for consistent ordering
    )

    # List all fine-tuned model files in the specified directory
    fine_tuned_model_files = [f for f in os.listdir(NEW_MODELS_DIR) if f.endswith('.pth')]

    if len(fine_tuned_model_files) == 0:
        print(f"No fine-tuned model files found in {NEW_MODELS_DIR}.")
        return

    print(f"Found {len(fine_tuned_model_files)} fine-tuned models in {NEW_MODELS_DIR}.")

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Iterate over each fine-tuned model
    for model_file in tqdm(fine_tuned_model_files, desc="Evaluating models"):
        model_name = os.path.splitext(model_file)[0]  # e.g., 'model1_fine_tuned' from 'model1_fine_tuned.pth'
        model_path = os.path.join(NEW_MODELS_DIR, model_file)
        print(f"\nEvaluating model: {model_name} from {model_path}")

        # Initialize and load the model
        model = initialize_model(model_path, device)

        # Ensure the model is in evaluation mode
        model.eval()

        # Evaluate the model
        image_paths, probs, preds, labels = evaluate_model(model, test_loader, device)

        # Calculate metrics
        try:
            auc = roc_auc_score(labels, probs)
            accuracy = accuracy_score(labels, preds)
            balanced_acc = balanced_accuracy_score(labels, preds)
            print(f"Model: {model_name} - Test AUC: {auc:.4f}, Test Accuracy: {accuracy:.4f}, Test Balanced Acc: {balanced_acc:.4f}")
        except ValueError as e:
            print(f"Error calculating metrics for {model_name}: {e}")
            auc = None
            accuracy = None
            balanced_acc = None

        # Create a DataFrame
        df = pd.DataFrame({
            'image_path': image_paths,
            'prob': probs,
            'prediction': preds,
            'label': labels
        })

        # Define the CSV filename and path
        csv_filename = f"{model_name}_predictions.csv"
        csv_path = os.path.join(PREDICTIONS_DIR, csv_filename)

        # Save the DataFrame to CSV
        try:
            df.to_csv(csv_path, index=False)
            print(f"Saved predictions to {csv_path}")
        except Exception as e:
            print(f"Error saving predictions for {model_name}: {e}")

    print("\nAll models have been evaluated and predictions saved.")

if __name__ == "__main__":
    main()
