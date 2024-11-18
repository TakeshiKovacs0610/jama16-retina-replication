import os
import shutil
import h5py
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from PreTraining.preprocess_0 import resize_and_center_fundus

# Define root directories
train_dir = '../data/kaggle_data/train/'
test_dir = '../data/kaggle_data/test/'

# Define CSV file paths
train_csv = '../data/kaggle_data/trainLabels.csv'
test_csv = '../data/kaggle_data/test_Labels.csv'

# Desired image size
image_size = (299, 299)

# Number of images to move from test to train
num_move = 10000

# Output HDF5 file paths
train_hdf5 = os.path.join(train_dir, 'train_dataset_corrected.h5')
validation_hdf5 = os.path.join(train_dir, 'validation_dataset_corrected.h5')
test_hdf5 = os.path.join(test_dir, 'test_dataset_corrected.h5')

# Load CSV files
train_labels = pd.read_csv(train_csv)
test_labels = pd.read_csv(test_csv)

print(f"Initial Train Set: {len(train_labels)} images")
print(f"Initial Test Set: {len(test_labels)} images")


# Setting the seed 
seed = 42
# Move 10,000 images from test to train
images_to_move = test_labels.sample(n=num_move, random_state=seed).reset_index(drop=True)

for idx, row in images_to_move.iterrows():
    img_name = f"{row[0]}.jpeg"
    src_path = os.path.join(test_dir, img_name)
    dest_path = os.path.join(train_dir, img_name)
    
    if os.path.isfile(src_path):
        shutil.move(src_path, dest_path)
    else:
        print(f"File {src_path} does not exist and cannot be moved.")

# Update CSV files
test_labels_updated = test_labels[~test_labels.iloc[:,0].isin(images_to_move.iloc[:,0])].reset_index(drop=True)
train_labels_updated = pd.concat([train_labels, images_to_move], ignore_index=True)

print(f"Updated Train Set: {len(train_labels_updated)} images")
print(f"Updated Test Set: {len(test_labels_updated)} images")

# Save updated CSVs (optional)
train_labels_updated.to_csv('../../data/kaggle_data/train_Labels_updated.csv', index=False)
test_labels_updated.to_csv('../../data/kaggle_data/test_Labels_updated.csv', index=False)

seed = 42
test_labels_updated = pd.read_csv('../data/kaggle_data/test_Labels_updated.csv')
train_labels_updated = pd.read_csv('../data/kaggle_data/train_Labels_updated.csv')

# Perform stratified split
X = train_labels_updated.iloc[:, 0]
y = train_labels_updated.iloc[:, 1]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y, 
    random_state=seed
)

print(f"Final Train Set: {len(X_train)} images")
print(f"Validation Set: {len(X_val)} images")

# Function to create HDF5
def create_hdf5(csv_df, image_dir, hdf5_path, image_size=(224, 224)):
    data = []
    label_list = []
    image_paths = []
    count = 0

    for idx, row in csv_df.iterrows():
        img_name = os.path.join(image_dir, f"{row['image']}.jpeg")
        if os.path.isfile(img_name):
            count += 1
            if count % 1000 == 0:
                print(f"Processed {count} images for {hdf5_path}")
            try:
                processed_image = resize_and_center_fundus(img_name,diameter= 299)
                if processed_image is not None:
                    processed_image = processed_image.resize(image_size)
                    processed_image = np.array(processed_image)
                    data.append(processed_image)
                    label_list.append(row['level'])
                    image_paths.append(row['image'])

                else:
                    print(f"Fundus centering failed for {img_name}. Skipping.")
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
        else:
            print(f"File {img_name} does not exist.")

    if data:
        data = np.stack(data)
        label_list = np.array(label_list)
        image_paths = np.array(image_paths, dtype=h5py.string_dtype())

        with h5py.File(hdf5_path, 'w') as hf:
            hf.create_dataset('images', data=data)
            hf.create_dataset('labels', data=label_list)
            hf.create_dataset('image_paths', data=image_paths)

        print(f"Saved HDF5 file at {hdf5_path} with {len(data)} images.")
    else:
        print(f"No data to save for {hdf5_path}.")

# Create HDF5 files
create_hdf5(
    csv_df=pd.DataFrame({'image': X_train, 'level': y_train}),
    image_dir=train_dir,
    hdf5_path=train_hdf5,
    image_size=image_size
)

create_hdf5(
    csv_df=pd.DataFrame({'image': X_val, 'level': y_val}),
    image_dir=train_dir,
    hdf5_path=validation_hdf5,
    image_size=image_size
)

create_hdf5(
    csv_df=test_labels_updated,
    image_dir=test_dir,
    hdf5_path=test_hdf5,
    image_size=image_size
)
