import cv2
import time

import numpy as np
from ultralytics import YOLO
import concurrent.futures
from typing import List, Tuple, Dict

from speed_classification.speed_inference import run_speed_sign_classification
from src.frontend.display_overlay import display_other_signs_and_frame, display_speed_sign
from src.frontend.draw_bboxes import draw_other_bboxes, draw_speed_sign_bboxes
from src.utils.count_labels import LabelCounter
from src.utils.utils import speed_numbers


def crop_image(image: np.ndarray, box: List) -> np.ndarray:
    cropped_image = image[int(box[0][1]):int(box[0][3]), int(box[0][0]):int(box[0][2])]
    return cropped_image


class VideoProcessor:
    def __init__(self, video_file: str, model_path: str, resize_width: int = 1080, resize_height: int = 920,
                 conf_threshold: float = 0.5):
        self.video_file = video_file
        self.model_path = model_path
        self.resize_width = resize_width
        self.resize_height = resize_height
        self.conf_threshold = conf_threshold
        self.cap = None
        self.model = None
        self.current_speed_sign = None
        self.display_sign_cache = []
        self.max_cache_size = 6
        self.label_counter = None

    def initialize_video_capture(self):
        self.cap = cv2.VideoCapture(self.video_file)
        frame_width = int(self.cap.get(3))
        frame_height = int(self.cap.get(4))
        if frame_width > 0:
            self.resize_height = int((self.resize_width / frame_width) * frame_height)

    def load_yolo_model(self):
        self.model = YOLO(self.model_path)

    def predict(self, image: cv2.Mat, classes: List[int], conf: float = None):
        if conf is None:
            conf = self.conf_threshold
        image = cv2.resize(image, (self.resize_width, self.resize_height))
        if classes:
            results = self.model.predict(image, classes=classes, conf=conf, save_txt=False)
        else:
            results = self.model.predict(image, conf=conf, save_txt=False)
        return results

    def predict_and_detect(self, image: cv2.Mat, classes=None) -> Tuple[cv2.Mat, List, Dict[str, float]]:
        if classes is None:
            classes = []
        image = cv2.resize(image, (self.resize_width, self.resize_height))
        results = self.predict(image=image, classes=classes, conf=self.conf_threshold)
        detected_signs = {}

        for result in results:
            for box in result.boxes:
                sign_name = result.names[int(box.cls[0])]
                self.label_counter.update_label_count(label=sign_name)
                if self.label_counter.check_label_count(label=sign_name):
                    if sign_name == "Geschwindigkeit":
                        cropped_image = crop_image(image=image, box=box.xyxy.cpu().tolist())
                        speed_sign_name, speed_confidence = run_speed_sign_classification(image=cropped_image)
                        draw_speed_sign_bboxes(detected_signs=detected_signs, box=box, speed_sign_name=speed_sign_name,
                                               speed_confidence=speed_confidence, image=image)
                    else:
                        draw_other_bboxes(detected_signs=detected_signs, box=box, sign_name=sign_name, image=image)
        return image, results, detected_signs

    def update_display_sign_cache(self, new_sign: str):
        # Avoid adding duplicate signs by removing it if it exists
        if new_sign in self.display_sign_cache:
            self.display_sign_cache.remove(new_sign)
        # Insert new sign at the beginning of the cache
        self.display_sign_cache.insert(0, new_sign)
        # Ensure cache does not exceed max size
        if len(self.display_sign_cache) > self.max_cache_size:
            self.display_sign_cache.pop()  # Remove the oldest sign (last in the list)

    def display_visuals(self, frame: cv2.Mat, detected_signs: Dict[str, float]):
        for sign, conf in detected_signs.items():
            if sign in speed_numbers:
                self.current_speed_sign = sign
            else:
                self.update_display_sign_cache(new_sign=sign)

        if self.current_speed_sign:
            frame = display_speed_sign(frame=frame, current_speed_sign=self.current_speed_sign)

        display_other_signs_and_frame(frame=frame, display_sign_cache=self.display_sign_cache)

    def process_frame(self, frame: cv2.Mat) -> cv2.Mat:
        result_frame, _, detected_signs = self.predict_and_detect(frame)
        self.display_visuals(frame=result_frame, detected_signs=detected_signs)
        return result_frame

    def run(self):
        self.initialize_video_capture()
        self.load_yolo_model()
        frame_count = 0
        self.label_counter = LabelCounter()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                frame_count += 1
                if frame_count % 4 == 0:
                    self.label_counter = LabelCounter()

                start_time = time.time()

                # Submit the frame for processing
                future = executor.submit(self.process_frame, frame)
                result_frame = future.result()

                end_time = time.time()
                loop_duration = end_time - start_time
                # Display duration of one frame
                print(f"Processing time for frame {frame_count}: {loop_duration:.3f} seconds")

                # Display the processed frame
                cv2.imshow("Processed Frame", result_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = r"C:\Users\Marco\dev\git\BV2-project\data\video\Verrücktes Überholmanöver Neben Polizei.mp4"
    model_path = r"C:\Users\Marco\dev\git\BV2-project\results\detection\train3\weights\best.pt"

    processor = VideoProcessor(video_file=video_path, model_path=model_path)
    processor.run()
