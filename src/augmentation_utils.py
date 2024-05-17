import cv2
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf


def augment_image(image):
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('Original image')
    plt.imshow(image)

    plt.subplot(1, 2, 2)
    plt.title('Augmented image')
    flipped = tf.image.flip_left_right(image)
    plt.imshow(flipped)


if __name__ == '__main__':
    image_path = ''
    image = cv2.imread(image_path)
    augment_image(image)


def shear_image(image, shear_range):
    # Define the shear angle randomly within the specified range
    shear_angle = np.random.uniform(-shear_range, shear_range)

    # Get the image dimensions
    rows, cols = image.shape[:2]

    # Define the transformation matrix for shear
    shear_matrix = np.array([[1, np.tan(np.radians(shear_angle)), 0],
                             [0, 1, 0]])

    # Apply the transformation using warpAffine
    sheared_image = cv2.warpAffine(image, shear_matrix, (cols, rows), borderMode=cv2.BORDER_REFLECT_101)

    return sheared_image


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
