import os
import shutil
import random
from math import floor

def get_all_images_from_subfolders(directory):
    images = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                images.append(os.path.join(root, file))
    return images

def get_corresponding_destination_path(source_path, source_base, destination_base):
    relative_path = os.path.relpath(source_path, source_base)
    return os.path.join(destination_base, relative_path)

def move_images(source_directory, destination_directory, percentage=30):
    all_images = get_all_images_from_subfolders(source_directory)
    number_to_move = floor(len(all_images) * (percentage / 100))
    images_to_move = random.sample(all_images, number_to_move)

    for image in images_to_move:
        destination_path = get_corresponding_destination_path(image, source_directory, destination_directory)
        destination_folder = os.path.dirname(destination_path)
        os.makedirs(destination_folder, exist_ok=True)
        shutil.move(image, destination_path)
        print(f"Moved {image} to {destination_path}")

source_directory = r'C:\Users\Benedikt Seeger\Desktop\train'
destination_directory = r'C:\Users\Benedikt Seeger\Desktop\test'

move_images(source_directory, destination_directory)
