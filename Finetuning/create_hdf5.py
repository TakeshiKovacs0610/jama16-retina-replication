import os
import h5py
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# Directories for images
stroke_dir = 'Stroke_test'  
nostroke_dir = 'NoStroke_test'

# Output HDF5 file paths
train_hdf5 = 'train_stroke_dataset.h5'
test_hdf5 = 'test_stroke_dataset.h5'

# Desired image size
image_size = (299, 299)  

# Function to load images and assign labels
def load_images_from_folder(folder, label, image_size):
    images = []
    labels = []
    image_paths = []
    i = 0
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path) and filename.lower().endswith(('.jpeg', '.jpg', '.png')):
            try:
                image = Image.open(img_path).convert('RGB')
                image = image.resize(image_size)
                image = np.array(image)
                images.append(image)
                labels.append(label)
                image_paths.append(filename)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
            print(f"Processed image {i+1}")
            i+= 1
    return np.array(images), np.array(labels), np.array(image_paths)

# Load Stroke images (label = 1)
stroke_images, stroke_labels, stroke_image_paths = load_images_from_folder(stroke_dir, 1, image_size)

# Load NoStroke images (label = 0)
nostroke_images, nostroke_labels, nostroke_image_paths = load_images_from_folder(nostroke_dir, 0, image_size)

# Combine the datasets
X = np.concatenate((stroke_images, nostroke_images), axis=0)
y = np.concatenate((stroke_labels, nostroke_labels), axis=0)
image_paths = np.concatenate((stroke_image_paths, nostroke_image_paths), axis=0)

# Perform 80:20 train-test split
X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(
    X, y, image_paths, test_size=0.2, stratify=y, random_state=42
)

# Function to create HDF5 datasets
def create_hdf5(images, labels, image_paths, hdf5_path):
    with h5py.File(hdf5_path, 'w') as hf:
        hf.create_dataset('images', data=images)
        hf.create_dataset('labels', data=labels)
        hf.create_dataset('image_paths', data=image_paths.astype(h5py.string_dtype()))
    print(f"Saved HDF5 file at {hdf5_path} with {len(images)} images.")

# Create the HDF5 train dataset
create_hdf5(X, y, image_paths, test_hdf5)

# # Create the HDF5 test dataset
# create_hdf5(X_test, y_test, paths_test, test_hdf5)

print(f"Final Test Set: {len(X)} images")
# print(f"Final Test Set: {len(X_test)} images")
