import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2

model = YOLO()

if __name__ == '__main__':

    model.train(data=r"C:\Users\Benedikt Seeger\PycharmProjects\BV2-project\data\train.yaml",
                epochs=400,
                patience=50,
                device=0,
                fliplr=0.0,
                flipud=0.0,
                copy_paste=0.5,
                perspective=0.0001,
                batch=-1
                )
    results = model.predict(source=r"C:\Users\Benedikt Seeger\PycharmProjects\BV2-project\data\test_images\Fotolia_170009629_S.jpg", show=True)
    cv2.waitKey(0)
    cv2.destroyAllWindows()