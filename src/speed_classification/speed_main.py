from ultralytics import YOLO
import cv2

model = YOLO("yolov8n-cls.pt")  # load a pretrained model (recommended for training)

if __name__ == '__main__':

    model.train(data=r"C:\Users\Benedikt Seeger\PycharmProjects\BV2-project\src\speed_classification\speed_data",
                epochs=100,
                patience=50,
                device=0,
                fliplr=0.0,
                degrees=0.5,
                flipud=0.0,
                perspective=0.0001,
                project= r"C:\Users\Benedikt Seeger\PycharmProjects\BV2-project\src\speed_classification",
                batch=-1
                )