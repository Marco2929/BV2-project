import os
import shutil
import random
from math import floor


def get_all_images_from_subfolders(directory):
    """
    Recursively retrieves all image files from the given directory and its subdirectories.

    Args:
    directory (str): The base directory to search for image files.

    Returns:
    list: A list of file paths to all found image files.
    """
    images = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                images.append(os.path.join(root, file))
    return images


def get_corresponding_destination_path(source_path, source_base, destination_base):
    """
    Creates the destination path for a given source path by maintaining the relative path structure.

    Args:
    source_path (str): The full path of the source file.
    source_base (str): The base directory of the source files.
    destination_base (str): The base directory for the destination files.

    Returns:
    str: The corresponding destination path.
    """
    relative_path = os.path.relpath(source_path, source_base)
    return os.path.join(destination_base, relative_path)


def move_images(source_directory, destination_directory, percentage=30):
    """
    Moves a percentage of images from the source directory to the destination directory,
    maintaining the subdirectory structure.

    Args:
    source_directory (str): The base directory containing the source images.
    destination_directory (str): The base directory where the images will be moved.
    percentage (int): The percentage of images to move (default is 30%).
    """
    all_images = get_all_images_from_subfolders(source_directory)
    number_to_move = floor(len(all_images) * (percentage / 100))
    images_to_move = random.sample(all_images, number_to_move)

    for image in images_to_move:
        destination_path = get_corresponding_destination_path(image, source_directory, destination_directory)
        destination_folder = os.path.dirname(destination_path)
        os.makedirs(destination_folder, exist_ok=True)
        shutil.move(image, destination_path)
        print(f"Moved {image} to {destination_path}")


if __name__ == "__main__":
    source_directory = r'C:\Users\Benedikt Seeger\Desktop\train'
    destination_directory = r'C:\Users\Benedikt Seeger\Desktop\test'

    move_images(source_directory, destination_directory)
