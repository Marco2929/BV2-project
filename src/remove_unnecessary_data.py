import os
from collections import defaultdict
import shutil

# Path to your dataset folder
dataset_folder = r'C:\Users\Marco\dev\git\BV2-project\data\dataset\train'
# Path to the new folder to save selected images
output_folder = r'C:\Users\Marco\dev\git\BV2-project\data\shorted_dataset\train'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Dictionary to store image files grouped by base image
image_groups = defaultdict(list)

# Iterate over all files in the dataset folder
for filename in os.listdir(dataset_folder):
    if filename.endswith('.png') or filename.endswith('.txt'):  # Adjust file extensions as needed
        # Extract the base image identifier (e.g., '00000_00000')
        base_image = '_'.join(filename.split('_')[:2])
        # Add the filename to the corresponding group
        image_groups[base_image].append(filename)

# Copy only 10 images per base image to the output folder
for base_image, filenames in image_groups.items():
    if len(filenames) > 10:
        # Sort the filenames (assuming they are named like '00000_00000_00000.jpg')
        sorted_filenames = sorted(filenames)
        # Keep only the first 10 filenames
        filenames_to_copy = sorted_filenames[:10]
        # Copy the selected files to the output folder
        for filename in filenames_to_copy:
            source_path = os.path.join(dataset_folder, filename)
            destination_path = os.path.join(output_folder, filename)
            shutil.copy(source_path, destination_path)
            print(f"Copied: {source_path} -> {destination_path}")

print("Copying complete.")
