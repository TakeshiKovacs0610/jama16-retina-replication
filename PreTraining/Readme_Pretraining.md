# PreTraining Guide

This guide provides step-by-step instructions to use the provided code files to generate results and model weights. It is designed to help you set up the dataset, run the code, and reproduce the results with minimal effort.

## Table of Contents

- [Dataset Preparation](#dataset-preparation)
- [Generating HDF5 Files](#generating-hdf5-files)
- [Training the Models](#training-the-models)
    - [Training with Weight Initialization](#training-with-weight-initialization)
    - [Training without Weight Initialization](#training-without-weight-initialization)
- [Evaluating the Models](#evaluating-the-models)
- [Ensembling the Results](#ensembling-the-results)
- [Recommended Folder Structure](#recommended-folder-structure)

## Dataset Preparation

1. **Initial Dataset Structure:**
     - **Train Dataset:** 35,000 images
     - **Test Dataset:** 55,000 images

2. **Moving Images from Test to Train:**
     - Move 10,000 images from the test dataset to the train dataset using the `create_hdf5_1.py` script.

3. **Resulting Dataset Structure:**
     - **Updated Train Dataset:** 45,000 images
     - **Updated Test Dataset:** 45,000 images

4. **Splitting Train Dataset:**
     - Split the updated train dataset into training and validation sets.
     - **Training Set:** 80% of train images (36,000 images)
     - **Validation Set:** 20% of train images (9,000 images)

5. **Directory Structure:**

     ```
     data/
         kaggle_data/
             train/
                 images/
                 trainLabels.csv
             test/
                 images/
                 testLabels.csv
     ```

## Generating HDF5 Files

Use the `create_hdf5_1.py` script to create HDF5 files for training, validation, and test datasets.

**Command:**

```bash
python create_hdf5_1.py
```

**Generated HDF5 Files:**

- `train_dataset_corrected.h5` (for training)
- `validation_dataset_corrected.h5` (for validation)
- `test_dataset_corrected.h5` (for testing)

These files are used for efficient data loading during training and evaluation.

## Training the Models

Train 10 models using two approaches:

### Training with Weight Initialization

Use the `pretrained_weights_2.py` script to train models with pretrained weights.

**Command:**

```bash
python pretrained_weights_2.py \
    -t ../data/kaggle_data/train/train_dataset_corrected.h5 \
    -v ../data/kaggle_data/train/validation_dataset_corrected.h5 \
    -sm model_weights/pretrained/model1 \
    -ss logs/pretrained/model1 \
    -log_dir logs/pretrained/model1
```

**Notes:**

- Repeat the command for `model1` to `model10`, changing the `-sm`, `-ss`, and `-log_dir` accordingly.
- The script trains the model and saves the weights and logs.

### Training without Weight Initialization

Use the `random_initialization_2.py` script to train models from scratch.

**Command:**

```bash
python random_initialization_2.py \
    -t ../data/kaggle_data/train/train_dataset_corrected.h5 \
    -v ../data/kaggle_data/train/validation_dataset_corrected.h5 \
    -sm model_weights/scratch/model1 \
    -ss logs/scratch/model1 \
    -log_dir logs/scratch/model1
```

**Notes:**

- Similarly, repeat for `model1` to `model10`.
- Adjust folder names to prevent overwriting.

## Evaluating the Models

Use the `evaluate_pretrained_3.py` script to generate predictions on the test set. The script automatically evaluates all 10 models and saves their predictions.

**Commands:**

Ensure to make update the `models_to_evaluate` list in the script to include the correct model weights for evaluation.
```bash
python evaluate_pretrained_3.py 
```

**Notes:**

- Replace `model1` with `model2`, ..., `model10` for evaluating other models
- Modify the script to load the correct model architecture matching your training setup
- Update the model loading section in `evaluate_pretrained_3.py` to match the architecture used in training
- The script outputs a CSV file with predictions for each model
- Ensure the model architecture in evaluation matches the one used during training


## Ensembling the Results

Use the `ensemble_4.py` script to combine predictions from all models.

**Command:**

```bash
python ensemble_4.py \
    --csv_dir predictions/ \
    --output_file results/final_predictions.csv
```

**Notes:**

- The `--input_dir` argument specifies the directory containing all prediction CSV files.
- The script computes the ensemble predictions and saves the final results to the file specified by `--output_file`.

## Recommended Folder Structure

```
jama16-retina-replication/
    PreTraining/
        create_hdf5_1.py
        pretrained_weights_2.py
        random_initialization_2.py
        evaluate_pretrained_3.py
        ensemble_4.py
        Readme_Pretraining.md
    data/
        kaggle_data/
            train/
                images/
                trainLabels.csv
            test/
                images/
                testLabels.csv
    model_weights/
        pretrained/
            model1/
            model2/
            ...
        scratch/
            model1/
            model2/
            ...
    logs/
        pretrained/
            model1/
            model2/
            ...
        scratch/
            model1/
            model2/
            ...
    predictions/
        model1_predictions.csv
        model2_predictions.csv
        ...
    results/
        final_predictions.csv
```

Ensure that all paths in the command line arguments match this folder structure for seamless execution.
