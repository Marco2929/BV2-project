import cv2
import numpy as np

# Load the video file
video_path = r'C:\Users\Marco\dev\git\BV2-project\data\video\Verrücktes Überholmanöver Neben Polizei.mp4'
cap = cv2.VideoCapture(video_path)


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines):
    if lines is None:
        return img
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img


def draw_curve(img, poly_coeffs, color=(255, 0, 0), thickness=10):
    height = img.shape[0]
    y_vals = np.linspace(0, height - 1, height)
    x_vals = np.polyval(poly_coeffs, y_vals)

    pts = np.array([np.flipud(np.transpose(np.vstack([x_vals, y_vals])))])
    cv2.polylines(img, np.int32([pts]), isClosed=False, color=color, thickness=thickness)


def process_frame(frame):
    height, width = frame.shape[:2]
    # Define a trapezoidal region of interest
    top_width = width // 18
    bottom_width = width - 550
    top_height = height // 2
    bottom_height = height - 150
    offset = -125

    top_left = ((width // 2 - top_width // 2) + offset, top_height)
    top_right = ((width // 2 + top_width // 2) + offset, top_height)
    bottom_left = ((width // 2 - bottom_width // 2) + offset, bottom_height)
    bottom_right = ((width // 2 + bottom_width // 2) + offset, bottom_height)

    region_of_interest_vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], np.int32)

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canny_image = cv2.Canny(gray_image, 100, 200)
    cropped_image = region_of_interest(canny_image, region_of_interest_vertices)

    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(cropped_image, rho=3, theta=np.pi / 180, threshold=50, lines=np.array([]),
                            minLineLength=100, maxLineGap=120)

    # Draw straight lines
    frame_with_lines = draw_lines(frame, lines)

    # Extract points for polynomial fitting
    points = np.argwhere(cropped_image > 0)
    if points.shape[0] > 0:
        y_vals = points[:, 0]
        x_vals = points[:, 1]
        if len(x_vals) > 0 and len(y_vals) > 0:
            poly_coeffs = np.polyfit(y_vals, x_vals, 2)
            draw_curve(frame_with_lines, poly_coeffs)

    # Draw the ROI on the frame after processing
    draw_roi(frame_with_lines, region_of_interest_vertices)

    return frame_with_lines


def draw_roi(img, vertices):
    cv2.polylines(img, [vertices], isClosed=True, color=(0, 255, 0), thickness=2)


# Output video
output_path = r'C:\Users\Marco\dev\git\BV2-project\data\video\Verrücktes Überholmanöver Neben Polizei.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    processed_frame = process_frame(frame)
    out.write(processed_frame)
    cv2.imshow('Lane Detection', processed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
