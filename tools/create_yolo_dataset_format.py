import cv2
import pandas as pd
from matplotlib import pyplot as plt
import os


def convert_coordinates_to_YOLO(img_width, img_height, x1, y1, x2, y2):
    # Calculate box center (x, y) and box width/height
    box_width = x2 - x1
    box_height = y2 - y1
    cx = x1 + (box_width / 2)
    cy = y1 + (box_height / 2)
    w = box_width
    h = box_height

    # Normalize coordinates
    cx /= img_width
    cy /= img_height
    w /= img_width
    h /= img_height

    return cx, cy, w, h


def convert_to_YOLO_format(img_width, img_height, x1, y1, x2, y2, class_label):
    # Convert coordinates to YOLO format
    cx, cy, w, h = convert_coordinates_to_YOLO(img_width, img_height, x1, y1, x2, y2)

    # Assuming class_label is the class index (integer) in YOLO format
    yolo_format = f"{class_label} {cx} {cy} {w} {h}"

    return yolo_format


if __name__ == '__main__':
    file_path = r"C:\Users\Marco\dev\git\BV2-project\data\dataset\raw\Test.csv"
    df = pd.read_csv(file_path)

    images_dir = r'C:\Users\Marco\dev\git\BV2-project\data\dataset\raw\Test'
    output_dir = r'C:\Users\Marco\dev\git\BV2-project\data\dataset\test'

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        image_filename = row['Path'].replace('/', '\\')  # Adjust path separator if needed
        image_path = os.path.join(images_dir, image_filename)
        # image = cv2.imread(image_path)

        width = row['Width']
        height = row['Height']
        roi_x1 = row['Roi.X1']
        roi_y1 = row['Roi.Y1']
        roi_x2 = row['Roi.X2']
        roi_y2 = row['Roi.Y2']
        class_id = row['ClassId']

        # Draw the bounding box on the image
        # cv2.rectangle(image, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 1)

        # Convert to YOLO format
        yolo_output = convert_to_YOLO_format(width, height, roi_x1, roi_y1, roi_x2, roi_y2, class_id)
        print("YOLO Format:", yolo_output)

        # Save YOLO format annotation to a text file with the same name as the image
        image_name = os.path.basename(image_path)
        yolo_filename = os.path.splitext(image_name)[0] + '.txt'
        yolo_filepath = os.path.join(output_dir, yolo_filename)

        with open(yolo_filepath, 'w') as f:
            f.write(yolo_output)
        # # Display the image with the bounding box
        # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR image to RGB for display
        # plt.title(f'Class ID: {class_id}, Width: {width}, Height: {height}')
        # plt.axis('off')  # Turn off axis
        # plt.show()
