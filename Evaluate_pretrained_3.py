import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.metrics import roc_auc_score, accuracy_score
import os
import numpy as np
from PIL import Image
import h5py
import pandas as pd
import random

# Set seeds for reproducibility
random.seed(432)
torch.manual_seed(432)

# Custom Dataset for HDF5 files
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

        # Map labels to binary
        if label in [0, 1]:
            label = 0
        else:
            label = 1

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

# Evaluation function
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
        for inputs, labels, image_paths in dataloader:
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

def main():
    """
    Main function to evaluate multiple models and save their predictions to CSV files.
    """
    # Constants
    test_dir = "../data/kaggle_data/test/"
    test_hdf5_path = os.path.join(test_dir, 'test_dataset_corrected.h5')
    test_batch_size = 32
    num_workers = 8
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # InceptionV3 input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Get the test dataloader
    test_loader = get_hdf5_dataloader(
        h5_file=test_hdf5_path,
        batch_size=test_batch_size,
        num_workers=num_workers,
        transforms=transform,
        shuffle=False  # Important for consistent ordering
    )
    
    # List of models to evaluate
    models_to_evaluate = [
        {
            'name': 'inceptionv3_scratch_Aux1',
            'path': 'model_weights/inceptionv3_scratch_Aux/model1.pth'
        },
        {
            'name': 'inceptionv3_scratch_Aux2',
            'path': 'model_weights/inceptionv3_scratch_Aux/model2.pth'
        },
        {
            'name': 'inceptionv3_scratch_Aux3',
            'path': 'model_weights/inceptionv3_scratch_Aux/model3.pth'
        },
        {
            'name': 'inceptionv3_scratch_Aux4',
            'path': 'model_weights/inceptionv3_scratch_Aux/model4.pth'
        },
        {
            'name': 'inceptionv3_scratch_Aux5',
            'path': 'model_weights/inceptionv3_scratch_Aux/model5.pth'
        },
        {
            'name': 'inceptionv3_scratch_Aux6',
            'path': 'model_weights/inceptionv3_scratch_Aux/model6.pth'
        },
        {
            'name': 'inceptionv3_scratch_Aux7',
            'path': 'model_weights/inceptionv3_scratch_Aux/model7.pth'
        },
        {
            'name': 'inceptionv3_scratch_Aux8',
            'path': 'model_weights/inceptionv3_scratch_Aux/model8.pth'
        },
        {
            'name': 'inceptionv3_scratch_Aux9',
            'path': 'model_weights/inceptionv3_scratch_Aux/model9.pth'
        },
        {
            'name': 'inceptionv3_scratch_Aux10',
            'path': 'model_weights/inceptionv3_scratch_Aux/model10.pth'
        },
        
        # Add more models as needed
    ]
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    for model_info in models_to_evaluate:
        model_name = model_info['name']
        model_path = model_info['path']
        print(f"Evaluating model: {model_name} from {model_path}")
        
        # Load the model (InceptionV3) without modifying the final layers
        model = models.inception_v3(weights=None, aux_logits=True)
        model = model.to(device)
        
        # Modify the final layers for binary classification
        model.fc = nn.Linear(model.fc.in_features, 1)
        
         # Modify the Auxilary layers for binary classification
        if hasattr(model, 'AuxLogits') and model.AuxLogits is not None:
            model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, 1)
        
        # Load the saved weights
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)  # Allow mismatched keys
            print(f"Successfully loaded weights for {model_name}.")
        except Exception as e:
            print(f"Error loading weights for {model_name}: {e}")
            continue  # Skip to the next model if loading fails
        
       
        
        model = model.to(device)
        
        model.eval()
        
        # Evaluate the model
        image_paths, probs, preds, labels = evaluate_model(model, test_loader, device)
        
        # Calculate metrics
        try:
            auc = roc_auc_score(labels, probs)
            accuracy = accuracy_score(labels, preds)
            print(f"Model: {model_name} - Test AUC: {auc:.4f}, Test Accuracy: {accuracy:.4f}")
        except ValueError as e:
            print(f"Error calculating metrics for {model_name}: {e}")
            auc = None
            accuracy = None
        
        # Create a DataFrame
        df = pd.DataFrame({
            'image_path': image_paths,
            'prob': probs,
            'prediction': preds,
            'label': labels
        })
        
        # Ensure the model name is safe for filenames
        safe_model_name = model_name.replace('/', '_').replace('\\', '_')
        csv_filename = f"predictions/inceptionv3_scratch_Aux/{safe_model_name}_predictions.csv"
        os.makedirs(os.path.dirname(csv_filename), exist_ok=True)  # Create directory if it doesn't exist
        df.to_csv(csv_filename, index=False)
        print(f"Saved predictions to {csv_filename}\n")

if __name__ == "__main__":
    main()


