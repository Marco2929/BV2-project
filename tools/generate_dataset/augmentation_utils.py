import random

import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import array_to_img


def apply_shear(image, level=15):
    array_inputs = tf.keras.preprocessing.image.img_to_array(image)

    # Apply vertical shear
    sheared_vertical = tf.keras.preprocessing.image.random_shear(array_inputs, level,
                                                                 row_axis=0, col_axis=1,
                                                                 channel_axis=2)

    # Apply horizontal shear
    sheared_horizontal = tf.keras.preprocessing.image.random_shear(sheared_vertical, level,
                                                                   row_axis=1, col_axis=0,
                                                                   channel_axis=2)

    # Convert the Keras image array to a PIL image
    pil_image = array_to_img(sheared_horizontal)

    # Convert the PIL image to a NumPy array
    numpy_image = np.array(pil_image)

    # Ensure the pixel values are in the correct range for OpenCV
    opencv_image = numpy_image.astype(np.uint8)

    return opencv_image

def augment_image(image):

    aug_image = tf.image.random_brightness(image, 0.6)
    aug_image = tf.image.random_contrast(aug_image, 0.7, 1.5)
    aug_image = tf.image.convert_image_dtype(aug_image, tf.float32)

    # Split the image into RGB and Alpha channels
    rgb_image = aug_image[:, :, :3]
    alpha_channel = aug_image[:, :, 3:]

    # Apply random hue adjustment to the RGB image
    rgb_image = tf.image.random_hue(rgb_image, 0.05)

    # Recombine the adjusted RGB image with the Alpha channel
    aug_image = tf.concat([rgb_image, alpha_channel], axis=-1)

    aug_image = apply_shear(aug_image)

    return aug_image


def plot_bounding_box(image_path, label_path):
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB (matplotlib expects RGB format)

    # Read label file to get bounding box coordinates
    with open(label_path, 'r') as f:
        line = f.readline().strip()
        data = line.split()
        class_id = int(data[0])  # YOLO format class ID
        x_center = float(data[1])  # Normalized x-coordinate of bounding box center
        y_center = float(data[2])  # Normalized y-coordinate of bounding box center
        width = float(data[3])  # Normalized width of the bounding box
        height = float(data[4])  # Normalized height of the bounding box

    # Convert normalized coordinates to image coordinates
    h, w, _ = image.shape
    x1 = int((x_center - width / 2) * w)
    y1 = int((y_center - height / 2) * h)
    x2 = int((x_center + width / 2) * w)
    y2 = int((y_center + height / 2) * h)

    # Plot bounding box on the image
    plt.imshow(image)
    plt.axis('off')  # Turn off axes
    plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none'))
    plt.show()
