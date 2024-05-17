import os
import time
import uuid
import cv2
import math
import matplotlib.pyplot as plt
import random

import numpy as np

from src.augmentation_utils import shear_image


def is_overlap(existing_centers, new_center, min_distance=200):
    for center in existing_centers:
        distance = math.sqrt((center[0] - new_center[0]) ** 2 + (center[1] - new_center[1]) ** 2)
        if distance < min_distance:
            return True
    return False


def readin_yolo_coordinates_from_file(sign_label_path):
    # Read label file to get bounding box coordinates
    with open(sign_label_path, 'r') as f:
        line = f.readline().strip()
        data = line.split()
        class_id = int(data[0])  # YOLO format class ID
        x_center = float(data[1])  # Normalized x-coordinate of bounding box center
        y_center = float(data[2])  # Normalized y-coordinate of bounding box center
        width = float(data[3])  # Normalized width of the bounding box
        height = float(data[4])  # Normalized height of the bounding box

    return class_id, x_center, y_center, width, height


def resize_bounding_boxes(x, y, x_center, y_center, sign_w, sign_h, width, height, scale_factor):
    # Calculate new bounding box coordinates considering the sign overlay
    new_bb_x_center = int(x + (x_center * sign_w * scale_factor))
    new_bb_y_center = int(y + (y_center * sign_h * scale_factor))
    new_bb_width = int(width * sign_w * scale_factor)
    new_bb_height = int(height * sign_h * scale_factor)

    return new_bb_x_center, new_bb_y_center, new_bb_width, new_bb_height


def plot_bounding_box_on_background(background_image_path, sign_image_paths, output_path):
    # Load background image
    background_image = cv2.imread(background_image_path)
    background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)
    background_image = cv2.resize(background_image, (1920, 1080))
    h_bg, w_bg, _ = background_image.shape

    shear_range = 20

    loop_length = random.randint(2, 5)

    coordinate_list = []

    label_coordinates = None

    for i in range(loop_length):
        # Define name of new image and txt file
        sign_image_path = random.choice(sign_image_paths)
        sign_label_path = sign_image_path.replace('.png', '.txt')

        # Load sign image and its corresponding label
        sign_image = cv2.imread(sign_image_path, cv2.IMREAD_UNCHANGED)

        sign_image = shear_image(sign_image, shear_range)

        sign_image = cv2.cvtColor(sign_image, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # Randomly scale the sign image
        sign_h, sign_w, _ = sign_image.shape
        scale_factor = random.uniform(2, 5)  # Random scale factor between 1 and 5
        sign_image_resized = cv2.resize(sign_image, (int(sign_w * scale_factor), int(sign_h * scale_factor)))

        # Place image
        while True:
            # Ensure sign image fits within the background
            if sign_image_resized.shape[0] > h_bg or sign_image_resized.shape[1] > w_bg:
                # If resized image is too large, resize it again
                scale_factor = random.uniform(2, 5)
                sign_image_resized = cv2.resize(sign_image, (int(sign_w * scale_factor), int(sign_h * scale_factor)))

                continue
            else:
                # Randomly position the sign image on the background
                max_x = w_bg - sign_image_resized.shape[1]
                max_y = h_bg - sign_image_resized.shape[0]
                if max_x > 0 and max_y > 0:
                    x = random.randint(0, max_x)
                    y = random.randint(0, max_y)

                    class_id, x_center, y_center, width, height = readin_yolo_coordinates_from_file(sign_label_path)

                    new_bb_x_center, new_bb_y_center, new_bb_width, new_bb_height = resize_bounding_boxes(x, y,
                                                    x_center, y_center, sign_w, sign_h, width, height, scale_factor)

                    # Check if prev picture is in same location
                    if i > 0:
                        if is_overlap(coordinate_list, [new_bb_x_center, new_bb_y_center]):
                            continue

                    # Overlay sign image onto the background
                    overlay = background_image
                    overlay[y:y + sign_image_resized.shape[0], x:x + sign_image_resized.shape[1]] = sign_image_resized

                    if label_coordinates:
                        label_coordinates = f"{label_coordinates}{class_id} {new_bb_x_center / w_bg} {new_bb_y_center / h_bg} {new_bb_width / w_bg} {new_bb_height / h_bg}\n"
                    else:
                        label_coordinates = f"{class_id} {new_bb_x_center / w_bg} {new_bb_y_center / h_bg} {new_bb_width / w_bg} {new_bb_height / h_bg}\n"
                    coordinate_list.append([new_bb_x_center, new_bb_y_center])

                    break
            background_image = overlay

    # Save the overlay image
    my_uuid = uuid.uuid4()
    cv2.imwrite(os.path.join(output_path, f"{my_uuid}.jpg"), cv2.cvtColor(background_image, cv2.COLOR_RGB2BGR))

    # Write the new label file with adjusted bounding box coordinates
    with open(os.path.join(output_path, f"{my_uuid}.txt"), 'w') as f:
        f.write(label_coordinates)


if __name__ == '__main__':
    background_folder_path = r"C:\Users\Marco\dev\git\BV2-project\data\background\val2017"
    output_folder_path = r"C:\Users\Marco\dev\git\BV2-project\data\augmented_dataset"
    images_folder_path = r'C:\Users\Marco\dev\git\BV2-project\data\shorted_dataset\test'

    number_of_data_samples = 20000

    number_of_training_images = number_of_data_samples * 0.7

    # Get a list of all files in the folder
    background_images_files_list = os.listdir(background_folder_path)

    # Filter only the image files (you can adjust this if needed)
    background_image_files = [f for f in background_images_files_list if f.endswith(".jpg")]

    # List all files in the directory
    images_files = [os.path.join(images_folder_path, filename) for filename in os.listdir(images_folder_path)]

    # Filter only image files (extensions: .jpg, .jpeg, .png, .gif, etc.)
    image_files = [file for file in images_files if file.lower().endswith('.png')]

    for i in range(number_of_data_samples):
        background_image_path = os.path.join(background_folder_path, random.choice(background_image_files))
        if i < number_of_training_images:
            plot_bounding_box_on_background(background_image_path=background_image_path, sign_image_paths=image_files,
                                            output_path=os.path.join(output_folder_path, "train"))
        else:
            plot_bounding_box_on_background(background_image_path=background_image_path, sign_image_paths=image_files,
                                            output_path=os.path.join(output_folder_path, "test"))

        print(f"Image {i} saved")
