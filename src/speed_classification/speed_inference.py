import os

import numpy as np
from ultralytics import YOLO


def run_speed_sign_classification(image: np.ndarray,
                                  model_path: str) -> [
    str, float]:

    """
    Run speed sign classification on the given image using a pre-trained YOLO model to further improve precision

    Args:
        image (np.ndarray): The image in which to detect and classify speed signs.
        model_path (str): path of the pre-trained YOLO model.

    Returns:
        str: The name of the detected speed sign.
        float: The confidence score of the detected speed sign.
    """
    model_path = os.path.join(model_path, "models/classification_model.pt")

    model = YOLO(model_path)

    results = model.predict(
        source=image)
    speed_sign_number = results[0].probs.top1

    speed_sign = results[0].names[speed_sign_number]
    confidence = results[0].probs.top1conf.item()
    print(f"Speed sign: {speed_sign}, Confidence: {confidence}")

    return speed_sign, confidence

