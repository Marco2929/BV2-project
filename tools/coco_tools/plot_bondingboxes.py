import cv2
import matplotlib.pyplot as plt
import os


def plot_bounding_boxes(image_folder, txt_folder):
    """
    Plots bounding boxes in YOLO format on images from a specified folder.

    Parameters:
    - image_folder: Path to the folder containing the images.
    - txt_folder: Path to the folder containing the YOLO format text files.
    """
    # Get list of all image files in the specified folder
    image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

    for image_filename in image_files:
        image_path = os.path.join(image_folder, image_filename)

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image {image_path}")
            continue

        height, width, _ = image.shape

        # Get the corresponding text file path
        txt_filename = os.path.splitext(image_filename)[0] + '.txt'
        txt_path = os.path.join(txt_folder, txt_filename)

        # Check if the txt file exists
        if not os.path.exists(txt_path):
            print(f"Bounding box file {txt_path} not found.")
            continue

        # Read the bounding boxes from the text file
        with open(txt_path, 'r') as file:
            bboxes = file.readlines()

        # Process each bounding box
        for bbox in bboxes:
            class_id, x_center, y_center, bbox_width, bbox_height = map(float, bbox.split())

            # Convert from YOLO format to pixel values
            x_center_pixel = int(x_center * width)
            y_center_pixel = int(y_center * height)
            bbox_width_pixel = int(bbox_width * width)
            bbox_height_pixel = int(bbox_height * height)

            # Calculate the top-left and bottom-right coordinates of the bounding box
            x_min = int(x_center_pixel - (bbox_width_pixel / 2))
            y_min = int(y_center_pixel - (bbox_height_pixel / 2))
            x_max = int(x_center_pixel + (bbox_width_pixel / 2))
            y_max = int(y_center_pixel + (bbox_height_pixel / 2))

            # Draw the bounding box on the image
            color = (255, 0, 0)  # Red color in BGR
            thickness = 2
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)

        # Display the image with bounding boxes
        # Convert BGR to RGB for matplotlib display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.show()

        # Wait for user input to close the window and proceed to the next image
        input("Press Enter to continue to the next image...")

if __name__ == "__main__":
    # Example usage
    image_folder = './coco_selected_images_traffic_light'  # Relative path to the folder containing the images
    txt_folder = './coco_selected_annotations_traffic_light'  # Relative path to the folder containing the text files
    plot_bounding_boxes(image_folder, txt_folder)
