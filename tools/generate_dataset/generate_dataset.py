import os
import uuid
import re
import cv2
import math
import random
import numpy as np

from tools.generate_dataset.augmentation_utils import augment_image


def is_overlap(existing_centers, new_center, min_distance=200):
    for center in existing_centers:
        distance = math.sqrt((center[0] - new_center[0]) ** 2 + (center[1] - new_center[1]) ** 2)
        if distance < min_distance:
            return True
    return False


def generate_bounding_boxes(sign_image, sign_image_path):
    #pattern = re.compile(r'\\(\d+)(_?\d*)\.png$')
    pattern = re.compile(r'\\(\d+)_?.*\.png$')
    sign_class = pattern.search(sign_image_path)
    sign_class = int(sign_class.group(1))

    sign_h, sign_w, _ = sign_image.shape

    # Calculate the center coordinates
    x_center = sign_w / 2
    y_center = sign_h / 2

    # Normalize the center coordinates
    x_center_normalized = x_center / sign_w
    y_center_normalized = y_center / sign_h

    # Normalize the width and height
    width_normalized = sign_w / sign_w
    height_normalized = sign_h / sign_h

    return sign_class, x_center_normalized, y_center_normalized, width_normalized, height_normalized


def resize_bounding_boxes(x, y, x_center, y_center, sign_w, sign_h, width, height, scale_factor):
    # Calculate new bounding box coordinates considering the sign overlay
    new_bb_x_center = int(x + (x_center * sign_w * scale_factor))
    new_bb_y_center = int(y + (y_center * sign_h * scale_factor))
    new_bb_width = int(width * sign_w * scale_factor)
    new_bb_height = int(height * sign_h * scale_factor)

    return new_bb_x_center, new_bb_y_center, new_bb_width, new_bb_height


def adjust_brightness(image, value):
    # Überprüfen, ob das Bild einen Alpha-Kanal hat
    if image.shape[2] == 4:
        # Splitte das Bild in die BGR- und Alpha-Kanäle
        bgr = image[:, :, :3]
        alpha = image[:, :, 3]

        # Konvertiere das BGR-Bild in den HSV-Farbraum
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # Splitte die Kanäle
        h, s, v = cv2.split(hsv)

        # Helligkeit (Value) anpassen
        v = cv2.add(v, value)

        # Limitiere die Werte auf den Bereich [0, 255]
        v = np.clip(v, 0, 255)

        # Mische die Kanäle wieder zusammen
        final_hsv = cv2.merge((h, s, v))

        # Konvertiere das Bild von HSV zurück nach BGR
        bgr_adjusted = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

        # Füge den Alpha-Kanal wieder hinzu
        adjusted_image = cv2.merge((bgr_adjusted, alpha))
    else:
        # Wenn das Bild keinen Alpha-Kanal hat, wie gehabt fortfahren
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, value)
        v = np.clip(v, 0, 255)
        final_hsv = cv2.merge((h, s, v))
        adjusted_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    return adjusted_image

def contains_500(sign_image_path):
    pattern = re.compile(r'500')
    match = pattern.search(sign_image_path)
    return match is not None


def plot_bounding_box_on_background(background_image_path, sign_image_paths, output_path):
    # Load background image
    background_image = cv2.imread(background_image_path)
    background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)
    background_image = cv2.resize(background_image, (1920, 1080))
    h_bg, w_bg, _ = background_image.shape

    loop_length = random.randint(5, 10)

    coordinate_list = []

    label_coordinates = None

    custom_bias = np.ones(np.size(sign_image_paths))

    speed_prob = 1
    prio_prob = 2

    custom_bias[42] = speed_prob      # 3
    custom_bias[63:86] = speed_prob   # 3
    custom_bias[12:16] = prio_prob    # 16

    for i in range(loop_length):
        # Define name of new image and txt file
        sign_image_path = random.choices(sign_image_paths, custom_bias,k=1)[0]

        # Load sign image and its corresponding label
        sign_image = cv2.imread(sign_image_path, cv2.IMREAD_UNCHANGED)

        sign_image = cv2.resize(sign_image, (100, 100))

        sign_image = augment_image(sign_image)

        sign_image = adjust_brightness(sign_image, 60)

        sign_image = cv2.cvtColor(sign_image, cv2.COLOR_BGR2RGBA)  # Convert to RGB

        # Randomly scale the sign image
        sign_h, sign_w, _ = sign_image.shape
        # scale_factor = random.uniform(0.4, 1.5)  # Random scale factor between 1 and 5
        scale_factor = np.random.beta(a=2, b=5) * (1.7 - 0.4) + 0.4
        sign_image_resized = cv2.resize(sign_image, (int(sign_w * scale_factor), int(sign_h * scale_factor)))

        # Place image
        while True:
            # Ensure sign image fits within the background
            if sign_image_resized.shape[0] > h_bg or sign_image_resized.shape[1] > w_bg:
                # If resized image is too large, resize it again
                scale_factor = random.uniform(0.4, 1.5)
                sign_image_resized = cv2.resize(sign_image, (int(sign_w * scale_factor), int(sign_h * scale_factor)))

                continue
            else:
                # Randomly position the sign image on the background
                max_x = w_bg - sign_image_resized.shape[1]
                max_y = h_bg - sign_image_resized.shape[0]
                if max_x > 0 and max_y > 0:
                    x = random.randint(0, max_x)
                    y = random.randint(0, max_y)

                    class_id, x_center, y_center, width, height = generate_bounding_boxes(sign_image_resized,
                                                                                          sign_image_path)

                    new_bb_x_center, new_bb_y_center, new_bb_width, new_bb_height = resize_bounding_boxes(x, y,
                                                                                                          x_center,
                                                                                                          y_center,
                                                                                                          sign_w,
                                                                                                          sign_h, width,
                                                                                                          height,
                                                                                                          scale_factor)

                    # Check if prev picture is in same location
                    if i > 0:
                        if is_overlap(coordinate_list, [new_bb_x_center, new_bb_y_center]):
                            continue

                    # Ensure the background image has 4 channels (RGBA)
                    if background_image.shape[2] == 3:
                        background_image = cv2.cvtColor(background_image, cv2.COLOR_RGB2RGBA)

                    # Ensure the sign image has 4 channels (RGBA)
                    if sign_image_resized.shape[2] == 3:
                        sign_image_resized = cv2.cvtColor(sign_image_resized, cv2.COLOR_RGB2RGBA)

                    # Extract the alpha channel from the sign image
                    alpha_sign = sign_image_resized[:, :, 3] / 255.0
                    alpha_background = 1.0 - alpha_sign

                    # Define the region of interest (ROI) in the background image
                    y1, y2 = y, y + sign_image_resized.shape[0]
                    x1, x2 = x, x + sign_image_resized.shape[1]

                    # Blend the images
                    for c in range(0, 3):
                        background_image[y1:y2, x1:x2, c] = (alpha_sign * sign_image_resized[:, :, c] +
                                                             alpha_background * background_image[y1:y2, x1:x2, c])

                    # If you want to keep the alpha channel in the result
                    background_image[y1:y2, x1:x2, 3] = (alpha_sign * 255 +
                                                         alpha_background * background_image[y1:y2, x1:x2, 3])

                    # Overlay sign image onto the background
                    overlay = background_image
                    if contains_500(sign_image_path) is False:
                        if label_coordinates:
                            label_coordinates = (f"{label_coordinates}{class_id} {new_bb_x_center / w_bg} "
                                                 f"{new_bb_y_center / h_bg} {new_bb_width / w_bg} {new_bb_height / h_bg}\n")
                        else:
                            label_coordinates = (f"{class_id} {new_bb_x_center / w_bg} {new_bb_y_center / h_bg} "
                                                 f"{new_bb_width / w_bg} {new_bb_height / h_bg}\n")
                        coordinate_list.append([new_bb_x_center, new_bb_y_center])

                    break
            background_image = overlay

    # Save the overlay image
    my_uuid = uuid.uuid4()
    cv2.imwrite(os.path.join(output_path, f"{my_uuid}.jpg"), cv2.cvtColor(background_image, cv2.COLOR_RGBA2BGR))

    # Write the new label file with adjusted bounding box coordinates
    with open(os.path.join(output_path, f"{my_uuid}.txt"), 'w') as f:
        f.write(label_coordinates)


if __name__ == '__main__':

    # Define the base directory relative to your script location
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # Define relative paths
    background_folder_path = os.path.join(base_dir, "data", "background", "val2017")
    output_folder_path = os.path.join(base_dir, "data", "augmented_dataset")
    images_folder_path = os.path.join(base_dir, "data", "basic_images")

    number_of_dataset_images = 25000

    number_of_training_images = number_of_dataset_images * 0.7

    # Get a list of all files in the folder
    background_images_files_list = os.listdir(background_folder_path)

    # Filter only the image files (you can adjust this if needed)
    background_image_files = [f for f in background_images_files_list if f.endswith(".jpg")]

    # List all files in the directory
    images_files = [os.path.join(images_folder_path, filename) for filename in os.listdir(images_folder_path)]

    # Filter only image files (extensions: .jpg, .jpeg, .png, .gif, etc.)
    image_files = [file for file in images_files if file.lower().endswith('.png')]

    for i in range(number_of_dataset_images):
        background_image_path = os.path.join(background_folder_path, random.choice(background_image_files))
        if i < number_of_training_images:
            plot_bounding_box_on_background(background_image_path=background_image_path, sign_image_paths=image_files,
                                            output_path=os.path.join(output_folder_path, "train"))
        else:
            plot_bounding_box_on_background(background_image_path=background_image_path, sign_image_paths=image_files,
                                            output_path=os.path.join(output_folder_path, "test"))

        print(f"Image {i} saved")
