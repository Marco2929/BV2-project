import os
from PIL import Image

# Paths
gt_file_path = r'C:\Users\Benedikt Seeger\Downloads\archive\gt.txt'
images_directory = r'C:\Users\Benedikt Seeger\Downloads\gtsdb_new'
output_directory = r'C:\Users\Benedikt Seeger\Downloads\gtsdb_new'

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

def convert_to_yolo_format(left_col, top_row, right_col, bottom_row, img_width, img_height):
    # Calculate center coordinates
    x_center = (left_col + right_col) / 2.0
    y_center = (top_row + bottom_row) / 2.0

    # Calculate width and height
    width = right_col - left_col
    height = bottom_row - top_row

    # Normalize
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height

    return x_center, y_center, width, height

# Read the ground truth file
with open(gt_file_path, 'r') as file:
    lines = file.readlines()

# Parse each line and generate corresponding output files
for line in lines:
    parts = line.strip().split(';')
    img_name = parts[0]
    left_col = int(parts[1])
    top_row = int(parts[2])
    right_col = int(parts[3])
    bottom_row = int(parts[4])
    class_id = parts[5]

    # Construct the image path with the .png extension
    img_path = os.path.join(images_directory, f'{os.path.splitext(img_name)[0]}.png')
    if not os.path.exists(img_path):
        print(f"Image {img_name}.png not found, skipping.")
        continue

    # Get image dimensions
    with Image.open(img_path) as img:
        img_width, img_height = img.size

    # Convert to YOLO format
    x_center, y_center, width, height = convert_to_yolo_format(left_col, top_row, right_col, bottom_row, img_width, img_height)

    # Create the output file path
    output_file_path = os.path.join(output_directory, f'{os.path.splitext(img_name)[0]}.txt')

    # Write bounding box and label data to the output file
    with open(output_file_path, 'a') as output_file:
        output_file.write(f'{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n')

    print(f"Processed {os.path.splitext(img_name)[0]}.png")

print("Finished processing all images.")
