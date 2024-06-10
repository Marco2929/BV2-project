import os
import cv2
import concurrent.futures
from ultralytics import YOLO
from typing import List, Tuple, Dict

from speed_classification.speed_inference import run_speed_sign_classification


def crop_image(image, box):
    cropped_image = image[int(box[0][1]):int(box[0][3]), int(box[0][0]):int(box[0][2])]
    return cropped_image


def display_overlay(frame: cv2.Mat, detected_signs: Dict[str, float]):
    overlay_text = "Erkannte Schilder:\n"
    for sign, conf in detected_signs.items():
        overlay_text += f"{sign}: {conf:.2f}\n"

    y0, dy = 30, 30
    for i, line in enumerate(overlay_text.split('\n')):
        y = y0 + i * dy
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


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

    def initialize_video_capture(self):
        self.cap = cv2.VideoCapture(self.video_file)
        frame_width = int(self.cap.get(3))
        frame_height = int(self.cap.get(4))
        if frame_width > 0:
            self.resize_height = int((self.resize_width / frame_width) * frame_height)

    def load_yolo_model(self):
        self.model = YOLO(self.model_path)

    def predict(self, img: cv2.Mat, classes: List[int] = [], conf: float = None):
        if conf is None:
            conf = self.conf_threshold
        img = cv2.resize(img, (self.resize_width, self.resize_height))
        if classes:
            results = self.model.predict(img, classes=classes, conf=conf, save_txt=False)
        else:
            results = self.model.predict(img, conf=conf, save_txt=False)
        return results

    def predict_and_detect(self, image: cv2.Mat, classes: List[int] = []) -> Tuple[cv2.Mat, List, Dict[str, float]]:
        image = cv2.resize(image, (self.resize_width, self.resize_height))
        results = self.predict(image, classes, self.conf_threshold)
        detected_signs = {}

        for result in results:
            for box in result.boxes:
                sign_name = result.names[int(box.cls[0])]
                if sign_name == "120 kmh":
                    cropped_image = crop_image(image=image, box=box.xyxy.cpu().tolist())
                    speed_sign_name, speed_confidence = run_speed_sign_classification(cropped_image)
                    detected_signs[speed_sign_name] = speed_confidence
                    cv2.rectangle(image, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                                  (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (0, 255, 0), 2)
                    cv2.putText(image, f"{speed_sign_name} {speed_confidence:.2f}",
                                (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                                cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                else:
                    detected_signs[sign_name] = box.conf[0]
                    cv2.rectangle(image, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                                  (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (0, 0, 255), 2)
                    cv2.putText(image, f"{sign_name} {box.conf[0]:.2f}",
                                (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                                cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return image, results, detected_signs

    def process_frame(self, frame: cv2.Mat) -> cv2.Mat:
        result_frame, _, detected_signs = self.predict_and_detect(frame)
        display_overlay(result_frame, detected_signs)
        return result_frame

    def run(self, skip_frames: int = 1):
        self.initialize_video_capture()
        self.load_yolo_model()
        frame_count = 0

        with concurrent.futures.ThreadPoolExecutor() as executor:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                frame_count += 1
                if frame_count % skip_frames != 0:
                    continue  # Skip this frame

                # Submit the frame for processing
                future = executor.submit(self.process_frame, frame)
                result_frame = future.result()

                # Display the processed frame
                cv2.imshow("Processed Frame", result_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    video_path = os.path.join(base_dir, 'data', 'video', 'Schild Umgefahren.mp4')
    model_path = os.path.join(base_dir, 'results', 'detection', 'train3', 'weights', 'best.pt')

    processor = VideoProcessor(video_file=video_path, model_path=model_path)
    processor.run()
