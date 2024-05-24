import cv2
import yaml
import os


def load_class_names(yaml_file_path):
    """
    Loads class names from a YAML file.

    Args:
    yaml_file_path (str): Path to the YAML file containing class names.

    Returns:
    dict: Dictionary mapping class numbers to class names.
    """
    try:
        with open(yaml_file_path, 'r') as file:
            data = yaml.safe_load(file)
        if 'names' in data:
            return data['names']
        else:
            print(f"Error: 'names' key not found in YAML file: {yaml_file_path}")
            return None
    except FileNotFoundError:
        print(f"Error: File not found: {yaml_file_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


def load_bounding_boxes(file_path):
    """
    Loads multiple bounding boxes from a file.

    Args:
    file_path (str): Path to the file containing bounding box coordinates in YOLO format.

    Returns:
    list: List of tuples with class number and bounding box (class_number, x_center, y_center, width, height).
    """
    bounding_boxes = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                class_number = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])
                bounding_boxes.append((class_number, x_center, y_center, width, height))
    except FileNotFoundError:
        print(f"Error: Bounding box file not found: {file_path}")
    except Exception as e:
        print(f"Unexpected error reading bounding box file: {e}")
    return bounding_boxes


def plot_bounding_boxes(image_path, bounding_boxes, class_names):
    """
    Plots multiple bounding boxes on the given image.

    Args:
    image_path (str): Path to the image file.
    bounding_boxes (list): List of bounding boxes in YOLO format (class_number, x_center, y_center, width, height).
                           The values should be normalized (between 0 and 1).
    class_names (dict): Dictionary mapping class numbers to class names.
    """
    if class_names is None:
        print("Error: Class names not loaded properly.")
        return

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image: {image_path}")
        return

    image_height, image_width, _ = image.shape

    for bounding_box in bounding_boxes:
        # Extract the class number and bounding box coordinates
        class_number, x_center, y_center, width, height = bounding_box

        # Convert YOLO format to (x_min, y_min, x_max, y_max)
        x_center *= image_width
        y_center *= image_height
        width *= image_width
        height *= image_height

        x_min = int(x_center - width / 2)
        y_min = int(y_center - height / 2)
        x_max = int(x_center + width / 2)
        y_max = int(y_center + height / 2)

        # Get the class name from the class number
        class_name = class_names.get(class_number, 'Unknown')

        # Draw the bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # Blue color box

        # Put the class name text above the bounding box
        cv2.putText(image, class_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the image with the bounding boxes in a separate window
    cv2.imshow('Image with Bounding Boxes', image)
    cv2.waitKey(0)  # Wait for a key press to go to the next image
    cv2.destroyAllWindows()


def process_folder(image_folder, bbox_folder, yaml_file_path):
    """
    Processes all images in the specified folder and displays them with bounding boxes.

    Args:
    image_folder (str): Path to the folder containing image files.
    bbox_folder (str): Path to the folder containing bounding box files.
    yaml_file_path (str): Path to the YAML file containing class names.
    """
    class_names = load_class_names(yaml_file_path)

    if class_names is None:
        print("Error: Class names could not be loaded. Exiting.")
        return

    # Iterate through all images in the folder
    for image_filename in os.listdir(image_folder):
        if image_filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, image_filename)
            bbox_path = os.path.join(bbox_folder, f"{os.path.splitext(image_filename)[0]}.txt")

            if os.path.exists(bbox_path):
                bounding_boxes = load_bounding_boxes(bbox_path)
                plot_bounding_boxes(image_path, bounding_boxes, class_names)
            else:
                print(f"Error: Bounding box file not found for image: {image_filename}")


# Example usage
image_folder = r'C:\Users\Benedikt Seeger\Downloads\gtsdb_new'
bbox_folder = image_folder
yaml_file_path = r'C:\Users\Benedikt Seeger\PycharmProjects\BV2_2\BV2-project\data\train.yaml'

process_folder(image_folder, bbox_folder, yaml_file_path)

