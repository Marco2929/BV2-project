from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import cv2


def process_image(boxes):

    cropped_objects = []
    if boxes is not None:
        crop_obj = im0[int(boxes[0][1]):int(boxes[0][3]), int(boxes[0][0]):int(boxes[0][2])]
        cropped_objects.append(crop_obj)

    return cropped_objects


# Example usage:
image_path = r"C:\Users\Marco\dev\git\BV2-project\data\test_images\img.png"
# Initialize model
model = YOLO(r"C:\Users\Marco\dev\git\BV2-project\results\detection\train22\weights\best.pt")
names = model.names
# Read the image
im0 = cv2.imread(image_path)

# Perform object detection
results = model.predict(im0, show=False)
boxes = results[0].boxes.xyxy.cpu().tolist()
clss = results[0].boxes.cls.cpu().tolist()

cropped_objects = process_image(boxes)

# To verify the cropped objects
for i, crop in enumerate(cropped_objects):
    cv2.imshow(f"Cropped Object {i + 1}", crop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
