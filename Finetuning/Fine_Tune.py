import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.metrics import roc_auc_score, balanced_accuracy_score  # Updated import
from jama_train_dataloader import get_hdf5_dataloader
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



# ============================ Configuration Parameters ============================

# Paths to data directories
TRAIN_DIR = ""  # Update with your training data path
VAL_DIR = ""      # Update with your validation data path
TRAIN_HDF5 = os.path.join(TRAIN_DIR, 'train_stroke_dataset.h5')
VAL_HDF5 = os.path.join(VAL_DIR, 'test_stroke_dataset.h5')

# Model saving and logging directories
SAVE_MODEL_PATH = "logs/only_oversampling_ensemble1/model_weights.pth"
SAVE_SUMMARIES_DIR = "logs/only_oversampling_ensemble1"
LOG_DIRECTORY = "logs/only_oversampling_ensemble1"

# Hyperparameters
LEARNING_RATE = 1e-4            # Smaller learning rate for fine-tuning
WEIGHT_DECAY = 4e-5
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 32
NUM_EPOCHS = 100                  # Fine-tuning usually requires fewer epochs
EARLY_STOPPING_PATIENCE = 10     # Early stopping patience
MIN_DELTA_LOSS = 0.001  # Minimum decrease in loss to qualify as improvement
MIN_DELTA_AUC = 0.01

# Ensure necessary directories exist
os.makedirs(LOG_DIRECTORY, exist_ok=True)
os.makedirs(SAVE_SUMMARIES_DIR, exist_ok=True)
os.makedirs(os.path.dirname(SAVE_MODEL_PATH), exist_ok=True)

# ============================ Data Transformations ============================

# Data transformations for training (including normalization)
train_transforms = transforms.Compose([
    transforms.Resize((299, 299)),          # InceptionV3 input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Data transformations for validation (only resizing and normalization)
val_transforms = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ============================ Data Loading ============================

# Load the training dataset
train_dataset = get_hdf5_dataloader(
    TRAIN_HDF5,
    batch_size=TRAIN_BATCH_SIZE,
    num_workers=8,
    transforms=train_transforms,
    return_dataset=True  # Ensure this flag makes 'get_hdf5_dataloader' return the dataset
)

# Extract labels from the training dataset using list comprehension
train_labels = [label.item() for _, label in train_dataset]

# Convert to NumPy array for easier manipulation
train_labels = np.array(train_labels)

# ============================ Class Imbalance Handling ============================

def compute_class_weights(dataset):
    """
    Computes class weights inversely proportional to class frequencies.

    Args:
        dataset (Dataset): Dataset for the training data.

    Returns:
        torch.Tensor: Tensor containing class weights.
    """
    all_labels = []
    for _, label in dataset:
        all_labels.append(label.item())
    class_counts = np.bincount(all_labels)

    # Handle case where a class might be missing
    if len(class_counts) < 2:
        class_counts = np.append(class_counts, [0]*(2 - len(class_counts)))

    # Avoid division by zero by adding a small epsilon
    class_weights = 1. / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * len(class_counts)  # Normalize weights
    return torch.tensor(class_weights, dtype=torch.float)

# Compute class counts and weights
class_counts = np.bincount(train_labels)
print(f"Class counts: {class_counts}")

class_weights = compute_class_weights(train_dataset)
print(f"Class Weights: {class_weights}")

# ============================ Create WeightedRandomSampler ============================

# Assign a weight to each sample based on its class
sample_weights = class_weights[train_labels]

# Convert to torch tensor
sample_weights = sample_weights.float()

# Create the WeightedRandomSampler
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# Initialize the training DataLoader with the sampler
train_loader = DataLoader(
    train_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    sampler=sampler,  # Use the sampler instead of shuffle
    num_workers=8
)

# Initialize the validation DataLoader without a sampler
val_loader = get_hdf5_dataloader(
    VAL_HDF5,
    batch_size=VAL_BATCH_SIZE,
    num_workers=8,
    transforms=val_transforms
)


# ============================ Model Definition ============================

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize the pre-trained InceptionV3 model
# Set aux_logits=True to enable auxiliary outputs during training
model = models.inception_v3(weights=None, aux_logits=True)

# Replace the fully connected layers for binary classification
model.fc = nn.Linear(model.fc.in_features, 1)
model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, 1)

# Optionally, load pre-trained weights if available
# Ensure that "model6.pth" is compatible with the current model architecture
try:
    model.load_state_dict(torch.load("model6.pth", map_location=device))
    print("Pre-trained weights loaded successfully.")
except Exception as e:
    print(f"Error loading pre-trained weights: {e}")
    print("Proceeding without loading pre-trained weights.")

# Freeze all layers initially
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last 3 Inception blocks (Mixed_7a, Mixed_7b, Mixed_7c) and the fully connected layers for fine-tuning
for param in model.Mixed_7a.parameters():
    param.requires_grad = True
for param in model.Mixed_7b.parameters():
    param.requires_grad = True
for param in model.Mixed_7c.parameters():
    param.requires_grad = True

# Also unfreeze the fully connected layers
for param in model.fc.parameters():
    param.requires_grad = True
for param in model.AuxLogits.fc.parameters():
    param.requires_grad = True

# Move the model to the appropriate device
model = model.to(device)

# ============================ Loss Function and Optimizer ============================

# Define the loss function with class weights for handling class imbalance
# Since it's a binary classification, use BCEWithLogitsLoss with pos_weight
# pos_weight is set to the ratio of negative to positive samples
# if len(class_weights) >= 2:
#     negative_count = class_weights[0].item()
#     positive_count = class_weights[1].item()
#     pos_weight = torch.tensor([positive_count / (negative_count + 1e-6)], device=device)
# else:
#     pos_weight = torch.tensor([1.0], device=device)

criterion = nn.BCEWithLogitsLoss()

# Define the optimizer to update only the parameters that require gradients
optimizer = optim.RMSprop(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

# ============================ Training and Evaluation Functions ============================

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for training data.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer.
        device (torch.device): Device to run the training on.

    Returns:
        tuple: Average loss and AUC for the epoch.
    """
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()

        outputs, aux_outputs = model(inputs)

        # Calculate primary loss and auxiliary loss
        loss1 = criterion(outputs.squeeze(), labels)
        loss2 = criterion(aux_outputs.squeeze(), labels)
        loss = loss1 + 0.4 * loss2  # Weight auxiliary loss

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Store predictions and labels for AUC calculation
        preds = torch.sigmoid(outputs).cpu().detach().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

    avg_loss = running_loss / len(dataloader)

    # Concatenate all predictions and labels
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Calculate AUC
    auc = roc_auc_score(all_labels, all_preds)

    return avg_loss, auc

def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run the evaluation on.

    Returns:
        tuple: Average loss and AUC for the validation set.
    """
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.float().to(device)

            outputs = model(inputs)  # In eval mode, only main output is returned

            # If outputs is a tuple, take the first element
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            loss = criterion(outputs.squeeze(), labels)
            running_loss += loss.item()

            # Store predictions and labels for AUC calculation
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    avg_loss = running_loss / len(dataloader)

    # Concatenate all predictions and labels
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Calculate AUC
    auc = roc_auc_score(all_labels, all_preds)

    return avg_loss, auc

def calculate_balanced_accuracy(dataloader, model, device):
    """
    Calculates the balanced accuracy of the model on a given dataset.

    Args:
        dataloader (DataLoader): DataLoader for the dataset.
        model (nn.Module): The neural network model.
        device (torch.device): Device to run the evaluation on.

    Returns:
        float: Balanced accuracy of the model.
    """
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.cpu().numpy()
            outputs = model(inputs)

            # If outputs is a tuple, take the first element
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            preds = torch.sigmoid(outputs).cpu().numpy()
            preds_binary = (preds >= 0.5).astype(int)

            all_labels.extend(labels)
            all_preds.extend(preds_binary)

    balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)
    return balanced_accuracy

# ============================ Training Loop with Early Stopping ============================

# Initialize lists to store metrics
train_losses = []
val_losses = []
train_aucs = []
val_aucs = []
train_balanced_accuracies = []
val_balanced_accuracies = []
best_val_loss = float('inf')     # Initialize to infinity for loss-based early stopping
waited_epochs = 0
best_auc = 0
best_val_accuracy = 0

# Open a log file to record training progress
log_file_path = os.path.join(LOG_DIRECTORY, 'model_log.txt')
with open(log_file_path, 'w') as log_file:
    # Redirect stdout to the log file
    original_stdout = sys.stdout
    sys.stdout = log_file

    print("Starting Fine-Tuning Process")
    print(f"Model will be saved to: {SAVE_MODEL_PATH}")
    print(f"Training Data: {TRAIN_HDF5}")
    print(f"Validation Data: {VAL_HDF5}")
    print("="*50)

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")

        # Train for one epoch
        train_loss, train_auc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_balanced_accuracy = calculate_balanced_accuracy(train_loader, model, device)

        # Evaluate on validation set
        val_loss, val_auc = evaluate_model(model, val_loader, criterion, device)
        val_balanced_accuracy = calculate_balanced_accuracy(val_loader, model, device)

        # Log metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_aucs.append(train_auc)
        val_aucs.append(val_auc)
        train_balanced_accuracies.append(train_balanced_accuracy)
        val_balanced_accuracies.append(val_balanced_accuracy)

        print(f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, Train Balanced Acc: {train_balanced_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, Val Balanced Acc: {val_balanced_accuracy:.4f}")
        print("-"*50)

        # Early Stopping Check based on validation loss
        if val_balanced_accuracy > best_val_accuracy:
            best_val_loss = val_loss
            waited_epochs = 0
            # Save the best model
            torch.save(model.state_dict(), SAVE_MODEL_PATH)
            print(f"New best validation acuracy: {val_balanced_accuracy:.4f}. Model saved to {SAVE_MODEL_PATH}.")
            best_val_accuracy = val_balanced_accuracy
            
        else:
            waited_epochs += 1
            print(f"No improvement in validation loss for {waited_epochs} epoch(s).")
            if waited_epochs >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
        # if val_balanced_accuracy > best_val_accuracy:
        #     # Save the best model
        #     torch.save(model.state_dict(), SAVE_MODEL_PATH)
        #     print(f"New best validation acuracy: {val_balanced_accuracy:.4f}. Model saved to {SAVE_MODEL_PATH}.")
        #     best_val_accuracy = val_balanced_accuracy
            
            
        # # Early Stopping Check
        # if val_auc < best_auc + MIN_DELTA_AUC:
        #     waited_epochs += 1
        #     print(f"No improvement in AUC for {waited_epochs} epoch(s).")
        #     if waited_epochs >= EARLY_STOPPING_PATIENCE:
        #         print(f"Early stopping triggered after {epoch+1} epochs.")
        #         break
        # else:
        #     best_auc = val_auc
        #     waited_epochs = 0
        #     # Save the best model
        #     torch.save(model.state_dict(), SAVE_MODEL_PATH)
        #     print(f"New best AUC: {best_auc:.4f}. Model saved to {SAVE_MODEL_PATH}.")

    print("Training Complete.")
    print("="*50)

    # Reset stdout to original
    sys.stdout = original_stdout

# ============================ Plotting Loss, AUC, and Balanced Accuracy ============================

# Plot Losses
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss per Epoch')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(SAVE_SUMMARIES_DIR, 'loss_plot.png'))
plt.close()

# Plot AUCs
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_aucs)+1), train_aucs, label='Train AUC')
plt.plot(range(1, len(val_aucs)+1), val_aucs, label='Validation AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.title('Training and Validation AUC per Epoch')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(SAVE_SUMMARIES_DIR, 'auc_plot.png'))
plt.close()

# Plot Balanced Accuracies
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_balanced_accuracies)+1), train_balanced_accuracies, label='Train Balanced Accuracy')
plt.plot(range(1, len(val_balanced_accuracies)+1), val_balanced_accuracies, label='Validation Balanced Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Balanced Accuracy')
plt.title('Training and Validation Balanced Accuracy per Epoch')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(SAVE_SUMMARIES_DIR, 'balanced_accuracy_plot.png'))
plt.close()

print("Training and evaluation plots have been saved.")
