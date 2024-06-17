import numpy as np
from ultralytics import YOLO


def run_speed_sign_classification(image: np.ndarray,
                                  model_path: str = r"C:\Users\Marco\dev\git\BV2-project\models\classification_model.pt") -> [
    str, float]:
    model = YOLO(model_path)

    results = model.predict(
        source=image)
    speed_sign_number = results[0].probs.top1

    speed_sign = results[0].names[speed_sign_number]
    confidence = results[0].probs.top1conf.item()
    print(f"Speed sign: {speed_sign}, Confidence: {confidence}")

    return speed_sign, confidence
