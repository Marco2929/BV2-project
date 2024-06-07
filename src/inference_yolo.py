from ultralytics import YOLO
import cv2
import concurrent.futures
from typing import List, Tuple
from speed_classification.speed_inference import run_speed_sign_classification


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

    def predict_and_detect(self, img: cv2.Mat, classes: List[int] = [], conf: float = None) -> Tuple[cv2.Mat, List]:
        if conf is None:
            conf = self.conf_threshold
        img = cv2.resize(img, (self.resize_width, self.resize_height))
        results = self.predict(img, classes, conf)

        for result in results:
            for box in result.boxes:
                if result.names[int(box.cls[0])] == "speed":
                    cropped_image = 0
                    speed_sign_name, speed_confidence = run_speed_sign_classification(cropped_image)
                    cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                                  (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (0, 255, 0), 2)
                    cv2.putText(img, f"{speed_sign_name} {speed_confidence:.2f}",
                                (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                else:
                    cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                                  (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (0, 0, 255), 2)
                    cv2.putText(img, f"{result.names[int(box.cls[0])]} {box.conf[0]:.2f}",
                                (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                                cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return img, results

    def process_frame(self, frame: cv2.Mat) -> cv2.Mat:
        result_frame, _ = self.predict_and_detect(frame)
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
    # Define the source: 0 for webcam or the path to your video file
    video_path = r"C:\Users\Marco\dev\git\BV2-project\data\video\Schild Umgefahren.mp4"
    model_path = r"C:\Users\Marco\dev\git\BV2-project\src\runs\detect\train21\weights\best.pt"

    processor = VideoProcessor(video_file=video_path, model_path=model_path)
    processor.run()
