import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2

model = YOLO(r"C:\Users\Marco\dev\git\BV2-project\src\runs\detect\train12\weights\best.pt")

if __name__ == '__main__':

    model.train(data=r"C:\Users\Marco\dev\git\BV2-project\data\dataset\train.yaml",
                epochs=200,
                patience=50,
                device=0,
                fliplr=0.0,
                copy_paste=0.5,
                perspective=0.0001
                )
    results = model.predict(source=r"C:\Users\Marco\dev\git\BV2-project\data\dataset\test_images\00000.png", show=True)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
