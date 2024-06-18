from typing import List

import cv2


def display_speed_sign(frame: cv2.Mat, current_speed_sign: str):
    sign_image_path = fr"C:\Users\Marco\dev\git\BV2-project\src\frontend\image_utils\{current_speed_sign}.png"
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

    return frame


def display_other_signs_and_frame(frame: cv2.Mat, display_sign_cache: List):
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

    for sign in display_sign_cache:
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
