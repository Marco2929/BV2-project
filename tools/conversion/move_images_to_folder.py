import os
import shutil

source_dir = r'C:\Users\Marco\dev\git\BV2-project\data\dataset\raw\Test'
destination_dir = r'C:\Users\Marco\dev\git\BV2-project\data\dataset\test'

# Ensure the destination directory exists
os.makedirs(destination_dir, exist_ok=True)

# Traverse the source directory
for root, dirs, files in os.walk(source_dir):
    for file in files:
        file_path = os.path.join(root, file)
        # Check if the file is an image (e.g., PNG, JPG, JPEG, etc.)
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            # Move the image file to the destination directory
            shutil.move(file_path, destination_dir)
