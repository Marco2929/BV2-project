import os
from PIL import Image


def resize_image_and_bboxes(image_path, txt_path, output_image_path, output_txt_path, target_size=(1920, 1080)):
    # Open the image
    image = Image.open(image_path)
    original_width, original_height = image.size

    # Resize the image
    image_resized = image.resize(target_size)
    image_resized.save(output_image_path)

    # Read the bounding boxes from the txt file
    with open(txt_path, 'r') as file:
        lines = file.readlines()

    # Scale the bounding boxes
    scaled_bboxes = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:
            class_id, x_center, y_center, width, height = parts
            x_center = float(x_center) * original_width
            y_center = float(y_center) * original_height
            width = float(width) * original_width
            height = float(height) * original_height

            # Scale the absolute coordinates
            x_center = x_center * target_size[0] / original_width
            y_center = y_center * target_size[1] / original_height
            width = width * target_size[0] / original_width
            height = height * target_size[1] / original_height

            # Convert back to relative coordinates
            x_center /= target_size[0]
            y_center /= target_size[1]
            width /= target_size[0]
            height /= target_size[1]

            scaled_bboxes.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    # Write the scaled bounding boxes to the new txt file
    with open(output_txt_path, 'w') as file:
        file.writelines(scaled_bboxes)


def process_folder(folder_path):
    output_folder_path = os.path.join(folder_path, "resized")
    os.makedirs(output_folder_path, exist_ok=True)

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.png'):
            image_path = os.path.join(folder_path, file_name)
            txt_path = os.path.join(folder_path, file_name.replace('.png', '.txt'))
            output_image_path = os.path.join(output_folder_path, file_name)
            output_txt_path = os.path.join(output_folder_path, file_name.replace('.png', '.txt'))

            if os.path.exists(txt_path):
                resize_image_and_bboxes(image_path, txt_path, output_image_path, output_txt_path)

# Specify the folder path containing the images and .txt files
folder_path = r'C:\Users\Benedikt Seeger\PycharmProjects\BV2_2\BV2-project\data\gtsdb'
process_folder(folder_path)
