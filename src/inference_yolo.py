import cv2
import time
from ultralytics import YOLO
import concurrent.futures
from typing import List, Tuple, Dict
from speed_classification.speed_inference import run_speed_sign_classification


def crop_image(image, box):
    cropped_image = image[int(box[0][1]):int(box[0][3]), int(box[0][0]):int(box[0][2])]
    return cropped_image


def draw_rectangle(img, pt1, pt2, color, thickness):
    points = [
        pt1,
        (pt2[0], pt1[1]),
        pt2,
        (pt1[0], pt2[1]),
    ]
    for i in range(4):
        pt1, pt2 = points[i], points[(i + 1) % 4]
        cv2.line(img, pt1, pt2, color, thickness)


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
        self.sign_cache = []
        self.max_cache_size = 6

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
                if sign_name == "Geschwindigkeit":
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

    def update_sign_cache(self, new_sign: str):
        # Avoid adding duplicate signs by removing it if it exists
        if new_sign in self.sign_cache:
            self.sign_cache.remove(new_sign)
        # Insert new sign at the beginning of the cache
        self.sign_cache.insert(0, new_sign)
        # Ensure cache does not exceed max size
        if len(self.sign_cache) > self.max_cache_size:
            self.sign_cache.pop()  # Remove the oldest sign (last in the list)

    def display_overlay(self, frame: cv2.Mat, detected_signs: Dict[str, float]):
        for sign, conf in detected_signs.items():
            if sign in ["20", "30", "50", "60", "70", "90", "80", "100", "120"]:
                self.current_speed_sign = sign
            else:
                self.update_sign_cache(sign)

        if self.current_speed_sign:
            sign_image_path = fr"C:\Users\Marco\dev\git\BV2-project\src\frontend\image_utils\{self.current_speed_sign}.png"
            sign_image = cv2.imread(sign_image_path, cv2.IMREAD_UNCHANGED)
            # Resize sign image if necessary
            sign_height, sign_width = sign_image.shape[:2]
            scale_factor = 0.1  # Adjust this factor as needed
            sign_image = cv2.resize(sign_image, (int(sign_width * scale_factor), int(sign_height * scale_factor)))

            # Define position for the overlay (top-left corner)
            x_offset, y_offset = 650, 443  # Adjust position as needed

            alpha_s = sign_image[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            for c in range(0, 3):
                frame[y_offset:y_offset + sign_image.shape[0], x_offset:x_offset + sign_image.shape[1], c] = (
                        alpha_s * sign_image[:, :, c] +
                        alpha_l * frame[y_offset:y_offset + sign_image.shape[0],
                                  x_offset:x_offset + sign_image.shape[1], c]
                )

        # Draw rounded grey block at the bottom middle
        overlay_width = 400
        overlay_height = 70
        overlay_x = (frame.shape[1] - overlay_width) // 2
        overlay_y = frame.shape[0] - overlay_height - 10
        cv2.rectangle(frame, (overlay_x, overlay_y), (overlay_x + overlay_width, overlay_y + overlay_height),
                      (255, 255, 255), 3)

        # Display cached signs in the grey block
        x_offset = overlay_x + 10  # Starting x offset for the sign images
        y_offset = overlay_y + 10  # Starting y offset for the sign images

        for sign in self.sign_cache:
            sign_image_path = fr"C:\Users\Marco\dev\git\BV2-project\src\frontend\image_utils\{sign}.png"
            sign_image = cv2.imread(sign_image_path, cv2.IMREAD_UNCHANGED)
            if sign_image is None:
                continue  # Skip if image not found

            # Resize sign image to have a height of 50 pixels
            sign_height, sign_width = sign_image.shape[:2]
            new_height = 50
            new_width = int((new_height / sign_height) * sign_width)
            sign_image = cv2.resize(sign_image, (new_width, new_height))

            alpha_s = sign_image[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            # Ensure the overlay fits within the allocated space
            end_x = min(x_offset + sign_image.shape[1], overlay_x + overlay_width - 10)
            end_y = min(y_offset + sign_image.shape[0], overlay_y + overlay_height - 10)
            sign_image = sign_image[:end_y - y_offset, :end_x - x_offset]

            for c in range(0, 3):
                frame[y_offset:end_y, x_offset:end_x, c] = (
                        alpha_s * sign_image[:, :, c] +
                        alpha_l * frame[y_offset:end_y, x_offset:end_x, c]
                )

            x_offset += sign_image.shape[1] + 10

    def process_frame(self, frame: cv2.Mat) -> cv2.Mat:
        result_frame, _, detected_signs = self.predict_and_detect(frame)
        self.display_overlay(result_frame, detected_signs)
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

                start_time = time.time()  # Start time of the loop

                # Submit the frame for processing
                future = executor.submit(self.process_frame, frame)
                result_frame = future.result()

                # End time of the loop
                end_time = time.time()
                loop_duration = end_time - start_time
                print(f"Processing time for frame {frame_count}: {loop_duration:.3f} seconds")

                # Display the processed frame
                cv2.imshow("Processed Frame", result_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = r"C:\Users\Marco\dev\git\BV2-project\data\video\Schild Umgefahren.mp4"
    model_path = r"C:\Users\Marco\dev\git\BV2-project\results\detection\train3\weights\best.pt"

    processor = VideoProcessor(video_file=video_path, model_path=model_path)
    processor.run()
