from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import cv2


def process_image(image_path):
    # Initialize model
    model = YOLO("yolov8n.pt")
    names = model.names

    # Read the image
    im0 = cv2.imread(image_path)
    assert im0 is not None, "Error reading image file"

    # Perform object detection
    results = model.predict(im0, show=False)
    boxes = results[0].boxes.xyxy.cpu().tolist()
    clss = results[0].boxes.cls.cpu().tolist()
    annotator = Annotator(im0, line_width=2, example=names)

    cropped_objects = []
    annotated_image = im0.copy()

    if boxes is not None:
        for box, cls in zip(boxes, clss):
            annotator.box_label(box, color=colors(int(cls), True), label=names[int(cls)])
            crop_obj = im0[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            cropped_objects.append(crop_obj)

    annotated_image = annotator.result()

    return annotated_image, cropped_objects


# Example usage:
image_path = r"C:\Users\Marco\dev\git\BV2-project\data\augmented_dataset\train\0a0e30c7-f14e-4700-b991-6f20a687d062.jpg"
annotated_image, cropped_objects = process_image(image_path)

# Display the image with annotations
cv2.imshow("ultralytics", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# To verify the cropped objects
for i, crop in enumerate(cropped_objects):
    cv2.imshow(f"Cropped Object {i + 1}", crop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
