"""
This script preprocesses the EyePACS dataset for diabetic retinopathy detection by performing the following steps:

1. Argument Parsing: Parses the command-line argument to specify the directory where the EyePACS dataset resides.
2. Directory Setup: Creates directories for each diabetic retinopathy grade (0 to 4) if they do not already exist. It also creates a temporary directory for storing intermediate preprocessing results.
3. Data Preprocessing: Reads the CSV files containing the image labels (trainLabels.csv and testLabels.csv). For each image:
   - Finds the image path using a glob pattern.
   - Uses the `resize_and_center_fundus` function to resize the fundus to 299 pixels in diameter and center it.
   - Displays a status message indicating the progress.
   - If preprocessing is successful, moves the processed image to the appropriate grade directory. If it fails, logs the image as failed.
4. Cleanup: Deletes the temporary directory and prints a summary of images that could not be processed.

This script ensures that only preprocessed and properly organized images are used for training and evaluating the diabetic retinopathy detection model.
"""

import argparse
import csv
import sys
from shutil import rmtree
from PIL import Image
from glob import glob
from os import makedirs, rename
from os.path import join, splitext, basename, exists
from lib.preprocess import resize_and_center_fundus

parser = argparse.ArgumentParser(description='Preprocess EyePACS data set.')
parser.add_argument("--data_dir", help="Directory where EyePACS resides.",
                    default="data/eyepacs")

args = parser.parse_args()
data_dir = str(args.data_dir)

train_labels = join(data_dir, 'trainLabels.csv')
test_labels = join(data_dir, 'testLabels.csv')

# Create directories for grades.
[makedirs(join(data_dir, str(i))) for i in [0, 1, 2, 3, 4]
        if not exists(join(data_dir, str(i)))]

# Create a tmp directory for saving temporary preprocessing files.
tmp_path = join(data_dir, 'tmp')
if exists(tmp_path):
    rmtree(tmp_path)
makedirs(tmp_path)

failed_images = []

for labels in [train_labels, test_labels]:
    with open(labels, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)

        for i, row in enumerate(reader):
            basename, grade = row[:2]

            im_path = glob(join(data_dir, "{}*".format(basename)))[0]

            # Find contour of eye fundus in image, and scale
            #  diameter of fundus to 299 pixels and crop the edges.
            res = resize_and_center_fundus(save_path=tmp_path,
                                           image_path=im_path,
                                           diameter=299, verbosity=0)

            # Status message.
            msg = "\r- Preprocessing image: {0:>7}".format(i+1)
            sys.stdout.write(msg)
            sys.stdout.flush()

            if res != 1:
                failed_images.append(basename)
                continue

            new_filename = "{0}.jpg".format(basename)

            # Move the file from the tmp folder to the right grade folder.
            rename(join(tmp_path, new_filename),
                   join(data_dir, str(int(grade)), new_filename))

# Clean tmp folder.
rmtree(tmp_path)

print("Could not preprocess {} images.".format(len(failed_images)))
print(", ".join(failed_images))
