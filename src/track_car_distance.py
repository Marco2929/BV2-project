import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2

model = YOLO(r"C:\Users\Marco\dev\git\BV2-project\yolov8n.pt")

if __name__ == '__main__':
    results = model.predict(source=r"C:\Users\Marco\dev\git\BV2-project\data\dataset\image3.jpeg", show=True, classes=[2])
    cv2.waitKey(0)
    cv2.destroyAllWindows()