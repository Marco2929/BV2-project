from typing import Dict

import cv2
import numpy as np
from ultralytics.engine.results import Boxes


def draw_speed_sign_bboxes(detected_signs: Dict, box: Boxes, speed_sign_name: str, speed_confidence: float,
                           image: np.ndarray):
    """
    Draw bounding boxes and labels for detected speed signs on the image.

    Args:
        detected_signs (Dict): Dictionary to store detected sign names and their confidence scores.
        box (Boxes): Bounding box coordinates and confidence for the detected sign.
        speed_sign_name (str): Name of the detected speed sign.
        speed_confidence (float): Confidence score of the detected speed sign.
        image (np.ndarray): Image on which to draw the bounding box and label.

    Returns:
        None
    """
    detected_signs[speed_sign_name] = speed_confidence
    cv2.rectangle(image, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                  (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (0, 255, 0), 2)
    cv2.putText(image, f"{speed_sign_name} {speed_confidence:.2f}",
                (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)


def draw_other_bboxes(detected_signs: Dict, box: Boxes, sign_name: str, image: np.ndarray):
    """
    Draw bounding boxes and labels for other detected signs on the image.

    Args:
        detected_signs (Dict): Dictionary to store detected sign names and their confidence scores.
        box (Boxes): Bounding box coordinates and confidence for the detected sign.
        sign_name (str): Name of the detected sign.
        image (np.ndarray): Image on which to draw the bounding box and label.

    Returns:
        None
    """
    detected_signs[sign_name] = box.conf[0]
    cv2.rectangle(image, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                  (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (0, 0, 255), 2)
    cv2.putText(image, f"{sign_name} {box.conf[0]:.2f}",
                (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
