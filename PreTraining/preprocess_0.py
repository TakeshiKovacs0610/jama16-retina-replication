import cv2
from PIL import Image
import numpy as np

def _increase_contrast_pytorch(image):
    """
    Helper function for increasing contrast of image using PyTorch.
    """
    # Create a local copy of the image.
    copy = image.copy()

    maxIntensity = 255.0
    x = np.arange(maxIntensity)

    # Parameters for manipulating image data.
    phi = 1.3
    theta = 1.5
    y = (maxIntensity / phi) * (x / (maxIntensity / theta)) ** 0.5

    # Decrease intensity such that dark pixels become much darker,
    # and bright pixels become slightly dark.
    copy = (maxIntensity / phi) * (copy / (maxIntensity / theta)) ** 2
    copy = np.array(copy, dtype=np.uint8)

    return copy

def _find_contours_pytorch(image):
    """
    Helper function for finding contours of image using PyTorch.
    Returns coordinates of contours.
    """
    # Increase contrast in image to increase chances of finding contours.
    processed = _increase_contrast_pytorch(image)

    # Get the gray-scale of the image.
    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

    # Detect contour(s) in the image.
    cnts = cv2.findContours(
        gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    # Ensure that some contours were found.
    if len(cnts) > 0:
        # Find the largest contour in the mask.
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        # Assume the radius is of a certain size.
        if radius > 100:
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            return (center, radius)
    return None

def resize_and_center_fundus(image_path, diameter):
    """
    Helper function for scale normalizing image using PyTorch.
    Takes in image path and returns a PIL image.
    """
    # Read the image using OpenCV.
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Image at path {image_path} not found.")

    # Find largest contour in the image.
    contours = _find_contours_pytorch(image)

    if contours is None:
        return None

    center, radius = contours

    # Calculate the min and max-boundaries for cropping the image.
    x_min = max(0, int(center[0] - radius))
    y_min = max(0, int(center[1] - radius))
    z = int(radius * 2)
    x_max = x_min + z
    y_max = y_min + z

    # Crop the image.
    cropped_image = image[y_min:y_max, x_min:x_max]

    # Scale the image.
    fx = fy = (diameter / 2) / radius
    resized_image = cv2.resize(cropped_image, (0, 0), fx=fx, fy=fy)

    # Add padding to image.
    shape = resized_image.shape

    # Get the border shape size.
    top = bottom = int((diameter - shape[0]) / 2)
    left = right = int((diameter - shape[1]) / 2)

    # Add 1 pixel if necessary.
    if shape[0] + top + bottom == diameter - 1:
        top += 1

    if shape[1] + left + right == diameter - 1:
        left += 1

    # Add border.
    padded_image = cv2.copyMakeBorder(
        resized_image, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

    # Convert to PIL Image and return.
    pil_image = Image.fromarray(cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB))

    return pil_image


