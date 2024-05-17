from ultralytics import YOLO
import os
import cv2


def predict_images_in_folder(folder_path, model):
    # List all files in the folder
    file_list = os.listdir(folder_path)

    # Filter files with image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.webp', 'avif']
    image_files = [f for f in file_list if os.path.splitext(f)[1].lower() in image_extensions]

    # Iterate over image files and perform predictions
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)

        # Perform prediction on the image
        results = model.predict(source=image_path, show=True)

        # Wait for a key press to continue to the next image
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # Path to the folder containing images
    folder_path = r"C:\Users\Marco\dev\git\BV2-project\data\test_images"

    # Initialize the YOLO model
    model_path = r"C:\Users\Marco\dev\git\BV2-project\src\runs\detect\train20\weights\best.pt"
    model = YOLO(model_path)

    # Call the function to predict images in the folder
    predict_images_in_folder(folder_path, model)
