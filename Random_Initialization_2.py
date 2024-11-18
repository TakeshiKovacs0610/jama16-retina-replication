
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from PreTraining.jama_train_dataloader_0 import get_hdf5_dataloader
import os
import argparse
import random
import csv
import numpy as np
from torchvision.models import Inception_V3_Weights
import matplotlib.pyplot as plt  # For plotting
import sys


# Command-line arguments for paths
parser = argparse.ArgumentParser(description="Train neural network for detection of diabetic retinopathy.")
parser.add_argument("-t", "--train_dir", default="../data/kaggle_data/train/", help="Path to training data")
parser.add_argument("-v", "--val_dir", default="../data/kaggle_data/train/", help="Path to validation data")
parser.add_argument("-sm", "--save_model_path", default="model_weights/inceptionv3_scratch_Aux/model", help="Where to save the model")
parser.add_argument("-ss", "--save_summaries_dir", default="logs/inception_scratch_Aux/", help="Where to save summaries")
# parser.add_argument("-so", "--save_operating_thresholds_path", default="/logs/inception_jama_scratch/", help="Where to save operating points")
parser.add_argument("-log_dir", "--log_directory", default="logs/inception_scratch_Aux", help="Where to save log files")
args = parser.parse_args()

os.makedirs(args.log_directory,exist_ok=True)

# Hyperparameters
learning_rate = 1e-3
decay = 4e-5
train_batch_size = 32
val_batch_size = 32
num_epochs = 200
wait_epochs = 10
min_delta_auc = 0.01
num_thresholds = 200
kepsilon = 1e-7

# Data transformations (for augmentation and normalization)
train_transforms = transforms.Compose([
    transforms.Resize((299, 299)),  # InceptionV3 input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

val_transforms = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_hdf5_path = os.path.join(args.train_dir, 'train_dataset_corrected.h5')
val_hdf5_path = os.path.join(args.val_dir, 'validation_dataset_corrected.h5')

# Get dataloaders
train_loader = get_hdf5_dataloader(train_hdf5_path, batch_size=train_batch_size, num_workers=8, transforms=train_transforms)
val_loader = get_hdf5_dataloader(val_hdf5_path, batch_size=val_batch_size, num_workers=8, transforms=val_transforms)

# Training and evaluation functions
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    # batch=0

    for inputs, labels in dataloader:
        # print(batch)
        # batch+=1
        inputs, labels = inputs.to(device), labels.float().to(device)
        optimizer.zero_grad()
        outputs,aux_outputs = model(inputs)
        
        # Primary Output
        # outputs = outputs.logits
        
        # Calculate loss
        loss = criterion(outputs.squeeze(), labels) +0.4 * criterion(aux_outputs.squeeze(),labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        # Store predictions and labels for AUC calculation
        all_preds.append(torch.sigmoid(outputs).cpu().detach().numpy())
        all_labels.append(labels.cpu().numpy())
    
    avg_loss = running_loss / len(dataloader)
    
    # Calculate AUC
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    auc = roc_auc_score(all_labels, all_preds)
    
    return avg_loss, auc

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.float().to(device)
            outputs = model(inputs)
            # Check if the outputs contain logits attribute
            # if hasattr(outputs, 'logits'):
            #     outputs = outputs.logits  # Access the primary output if using aux_logits
            loss = criterion(outputs.squeeze(), labels) 
            running_loss += loss.item()
            
            # Store predictions and labels for AUC calculation
            all_preds.append(torch.sigmoid(outputs).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    avg_loss = running_loss / len(dataloader)
    
    # Calculate AUC
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    auc = roc_auc_score(all_labels, all_preds)
    
    return avg_loss, auc

# Main training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


for i in range(10):
    print(f"\nStarting training for Model {i+1}")
    seed_value= 10**i
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    
    print(f"Model {i+1} is using seed: {seed_value}")
    


    log_file_path = os.path.join(args.log_directory,f'model_{i+1}_log.txt')
    with open(log_file_path, 'w') as f:
        sys.stdout = f  # Redirect stdout to the log file
        
        # weights = Inception_V3_Weights.DEFAULT
        model = models.inception_v3(weights=None,aux_logits = True)
        model.fc = nn.Linear(model.fc.in_features, 1)
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features,1)
        model = model.to(device)
    
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=decay)
        criterion = nn.BCEWithLogitsLoss()  # For binary classification
    
        train_losses = []
        val_losses = []
        train_aucs = []
        val_aucs = []
    
        best_auc = 0
        waited_epochs = 0
    
        for epoch in range(num_epochs):
            # Train for one epoch
            print("epoch:", epoch)
            train_loss, train_auc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_auc = evaluate_model(model, val_loader, criterion, device)
            
            # Store losses and AUCs
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_aucs.append(train_auc)
            val_aucs.append(val_auc)
            
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
            
            # Early stopping based on AUC improvement
            if val_auc < best_auc + min_delta_auc:
                waited_epochs += 1
                if waited_epochs >= wait_epochs:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                best_auc = val_auc
                waited_epochs = 0
                save_dir = os.path.dirname(f"{args.save_model_path}{i+1}.pth")
                os.makedirs(save_dir, exist_ok=True)
                torch.save(model.state_dict(), f"{args.save_model_path}{i+1}.pth")
                print(f"New best AUC: {best_auc:.8f}, model saved!")
        
        # Plotting AUCs
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f'Model {i+1}: Train and Validation loss vs epoch')
        plt.grid(True)
        plt.savefig(f"{args.save_summaries_dir}{i+1}_loss_plot.png")
    
        print(f"Model {i+1} training complete.")
    
    # Reset stdout to default after each model's training
    sys.stdout = sys.__stdout__

print("Training for all models complete.")