import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255  # color to fill the mask

    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), 10)

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img

def process(image):
    height, width = image.shape[:2]

    # Define the region of interest (ROI)
    roi_vertices = [
        (0, height),
        (width / 2, height / 2),
        (width, height)
    ]

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    canny_image = cv2.Canny(blur_image, 50, 150)

    cropped_image = region_of_interest(canny_image, np.array([roi_vertices], np.int32))

    lines = cv2.HoughLinesP(
        cropped_image,
        rho=2,
        theta=np.pi / 180,
        threshold=100,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=25
    )

    image_with_lines = draw_lines(image, lines)
    return image_with_lines

def main(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = process(frame)
            cv2.imshow('Lane Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = r'C:\Users\Benedikt Seeger\Downloads\Vorfahrtsschild.mp4'
    main(video_path)
