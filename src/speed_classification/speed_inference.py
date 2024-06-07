from ultralytics import YOLO

model = YOLO(
    r"C:\Users\Marco\dev\git\BV2-project\src\speed_classification\train2\weights\best.pt")  # load a pretrained model (recommended for training)

if __name__ == '__main__':
    results = model.predict(source=r"C:\Users\Marco\dev\git\BV2-project\src\speed_classification\speed_data\train\70\00004_00000_00000.png")
    speed_sign_number = results[0].probs.top1

    speed_sign = results[0].names[speed_sign_number]

    print(f"Speed sign: {speed_sign}")