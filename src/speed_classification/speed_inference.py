from ultralytics import YOLO


def run_speed_sign_classification(image_path: str,
                                  model_path: str = r"C:\Users\Marco\dev\git\BV2-project\models\classification_model.pt"):
    model = YOLO(model_path)

    results = model.predict(
        source=image_path)
    speed_sign_number = results[0].probs.top1

    speed_sign = results[0].names[speed_sign_number]
    confidence = results[0].probs.top1conf.item()
    print(f"Speed sign: {speed_sign}, Confidence: {confidence}")

    return speed_sign, confidence


if __name__ == "__main__":
    run_speed_sign_classification(r"C:\Users\Marco\dev\git\BV2-project\data\test_images\classify_test.png")
