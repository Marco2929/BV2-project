import cv2
import numpy as np

# Load the video file
video_path = r'C:\Users\Benedikt Seeger\Downloads\Vorfahrtsschild.mp4'
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
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float('inf')
            if abs(slope) < 0.5:  # Skip nearly horizontal lines
                continue
            cv2.line(blank_image, (x1, y1), (x2, y2), (255, 0, 0), 4)

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img

def draw_roi(img, vertices):
    cv2.polylines(img, [vertices], isClosed=True, color=(0, 255, 0), thickness=2)

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

    lines = cv2.HoughLinesP(cropped_image, rho=2, theta=np.pi / 180, threshold=90, lines=np.array([]), minLineLength=90, maxLineGap=60)
    frame_with_lines = draw_lines(frame, lines)

    # Draw the ROI on the frame after processing
    draw_roi(frame_with_lines, region_of_interest_vertices)

    return frame_with_lines

# Output video
output_path = r'C:\Users\Benedikt Seeger\Downloads\processed_Vorfahrtsschild.mp4'
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
