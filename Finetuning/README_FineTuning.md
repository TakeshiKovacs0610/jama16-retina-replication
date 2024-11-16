# jama16-retina-replication

## Introduction

This repository contains code for training, evaluating, and ensembling deep learning models for image classification of stroke images. The primary goal is to classify images into two categories: Stroke and NoStroke. The workflow includes preparing the dataset, training models, evaluating them, and performing ensemble methods to improve classification performance.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Dataset](#dataset)
    - [Dataset Structure](#dataset-structure)
    - [Preparing the HDF5 Files](#preparing-the-hdf5-files)
- [Usage](#usage)
    - [Step 1: Create HDF5 Datasets](#step-1-create-hdf5-datasets)
    - [Step 2: Train the Model](#step-2-train-the-model)
    - [Step 3: Evaluate Models](#step-3-evaluate-models)
    - [Step 4: Perform Ensemble Evaluation](#step-4-perform-ensemble-evaluation)
- [Code Overview](#code-overview)
- [Dependencies](#dependencies)
- [Notes](#notes)

## Prerequisites

- Python 3.x
- PyTorch
- torchvision
- h5py
- NumPy
- pandas
- scikit-learn
- Pillow (PIL)
- tqdm
- argparse
- glob

Ensure that you have a machine with sufficient computational resources. A GPU is highly recommended for training deep learning models.

## Dataset

### Dataset Structure

The dataset should be organized into two main directories representing the classes:

- `Stroke`: Contains images labeled as having a stroke (label 1).
- `NoStroke`: Contains images labeled as not having a stroke (label 0).

### Preparing the HDF5 Files

The images need to be converted into HDF5 files for efficient data loading. The script `create_hdf5.py` handles this conversion.

**Image Requirements:**

- Formats: `.jpeg`, `.jpg`, `.png`
- Color Mode: RGB
- Size: Images will be resized to `(299, 299)` pixels.

## Usage

### Step 1: Create HDF5 Datasets

The script `create_hdf5.py` creates HDF5 datasets from the image folders.

**Instructions:**

1. **Set the Paths:**

     Edit `create_hdf5.py` and set the paths to your image directories:

     ```python
     stroke_dir = 'path_to_stroke_images'
     nostroke_dir = 'path_to_nostroke_images'
     ```

2. **Run the Script:**

     ```bash
     python create_hdf5.py
     ```

**This script will:**

- Load images from the specified directories.
- Resize images to `(299, 299)`.
- Assign labels (1 for Stroke, 0 for NoStroke).
- Combine and optionally split the dataset.
- Save the combined dataset to HDF5 files.

**Output:**

- HDF5 files containing images, labels, and image paths.

### Step 2: Train the Model

The script `fine_tune.py` trains the InceptionV3 model using the HDF5 datasets.

**Instructions:**

1. **Set the Paths:**

     Edit `fine_tune.py` to set the paths for training and validation datasets:

     ```python
     TRAIN_HDF5 = 'path_to_train_hdf5_file'
     VAL_HDF5 = 'path_to_val_hdf5_file'
     ```

2. **Configure Training Parameters:**

     Adjust hyperparameters as needed:

     ```python
     LEARNING_RATE = 1e-4
     NUM_EPOCHS = 100
     ```

3. **Run the Script:**

     ```bash
     python fine_tune.py
     ```

**The script will:**

- Load the training data using `jama_train_dataloader.py`.
- Compute class weights to handle class imbalance.
- Initialize the InceptionV3 model.
- Fine-tune the model.
- Save the trained model weights.

**Output:**

- Trained model weights saved to the specified path.

### Step 3: Evaluate Models

The script `evaluate_models.py` evaluates the trained models on the test dataset and saves the predictions.

**Instructions:**

1. **Set the Paths:**

     Edit `evaluate_models.py` to set the paths:

     ```python
     TEST_HDF5 = 'path_to_test_hdf5_file'
     NEW_MODELS_DIR = 'path_to_trained_models_directory'
     ```

2. **Ensure Models are Available:**

     Place your trained `.pth` model files in `NEW_MODELS_DIR`.

3. **Run the Script:**

     ```bash
     python evaluate_models.py
     ```

**The script will:**

- Load each model from `NEW_MODELS_DIR`.
- Evaluate the model on the test dataset.
- Save the predictions to CSV files in `PREDICTIONS_DIR`.

**Output:**

- Prediction CSV files for each model.

### Step 4: Perform Ensemble Evaluation

The script `ensemble.py` performs ensemble methods on the predictions to improve performance.

**Instructions:**

1. **Set the Paths (if necessary):**

     By default, `ensemble.py` uses the predictions in `predictions/fine_tuned_models`.

2. **Run the Script:**

     ```bash
     python ensemble.py
     ```

     **Optional arguments:**

     - `-c`, `--csv_dir`: Directory containing prediction CSV files.
     - `-p`, `--pattern`: File pattern for CSV files.
     - `--threshold_start`: Starting threshold for averaging ensemble.
     - `--threshold_end`: Ending threshold for averaging ensemble.
     - `--threshold_step`: Step size for thresholds.
     - `--prediction_threshold`: Threshold to binarize individual model probabilities.

**Example with custom arguments:**

```bash
python ensemble.py -c predictions/fine_tuned_models -p '*_predictions.csv' --threshold_start 0.5 --threshold_end 0.9 --threshold_step 0.05
```

**Output:**

- Ensemble evaluation results printed to the console.
- Results saved to `ensemble_evaluation_results.csv` in `csv_dir`.

## Code Overview

**Scripts and Their Functions:**

- `create_hdf5.py`: Prepares the dataset by loading images, resizing them, assigning labels, and saving them into HDF5 files.
- `jama_train_dataloader.py`: Contains `HDF5Dataset` class for efficiently loading images and labels from HDF5 files and a function to get a `DataLoader`.
- `fine_tune.py`: Trains the InceptionV3 model using the HDF5 datasets. It handles data loading, model initialization, training, and saving the model.
- `evaluate_models.py`: Evaluates trained models on the test dataset. It loads models, performs predictions, and saves the predictions to CSV files.
- `ensemble.py`: Performs ensemble methods on model predictions. It loads prediction CSVs, merges them, performs averaging of probabilities, applies thresholds, and evaluates performance.

## Dependencies

Install the required packages using `pip`:

```bash
pip install torch torchvision h5py numpy pandas scikit-learn Pillow tqdm argparse glob
```

## Notes

- **Data Directory Structure:** Ensure that all file paths in the scripts are correctly set to your directory structure.
- **GPU Utilization:** The scripts are configured to use a GPU if available. For faster training and evaluation, a CUDA-enabled GPU is recommended.
- **Class Imbalance Handling:** The training script calculates class weights and uses a `WeightedRandomSampler` to handle class imbalance.
- **Reproducibility:** Random seeds are set for reproducibility. However, exact reproducibility might not be guaranteed due to certain operations in PyTorch.
- **Model Checkpoints:** The `fine_tune.py` script saves model checkpoints. Ensure that the directory to save the model exists.
- **Error Handling:** The scripts include basic error handling. If you encounter errors, check that all paths and filenames are correctly specified and that all dependencies are installed.
- **Customization:** Feel free to modify the scripts to suit your specific needs, such as changing the model architecture, altering hyperparameters, or adding data augmentation techniques.

By following the above steps, you should be able to replicate the process of training, evaluating, and ensembling models for image classification using the provided scripts.