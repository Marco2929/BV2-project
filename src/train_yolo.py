import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2

model = YOLO(r"C:\Users\Benedikt Seeger\PycharmProjects\BV2_2\BV2-project\runs\detect\train13\weights\best.pt")

if __name__ == '__main__':

    # #model.train(data=r"C:\Users\Benedikt Seeger\PycharmProjects\BV2-project\data\train.yaml",
    #             epochs=400,
    #             patience=50,
    #             device=0,
    #             fliplr=0.0,
    #             flipud=0.0,
    #             copy_paste=0.5,
    #             perspective=0.0001,
    #             batch=-1
    #             )
    results = model.predict(source=r"C:\Users\Benedikt Seeger\Documents\#Ben\05-Studium\8. Semester\04-Bildverarbeitung 2\Projekt\dataset\vidoes\Heilbronn Nacht.mp4", show=True)
    cv2.waitKey(0)
    cv2.destroyAllWindows()