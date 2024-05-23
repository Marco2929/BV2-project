import cv2
import numpy as np

# Load your background and sign images
background_image = cv2.imread(r'C:\Users\Benedikt Seeger\PycharmProjects\BV2-project\data\background\val2017\000000000139.jpg')
sign_image = cv2.imread(r'C:\Users\Benedikt Seeger\PycharmProjects\BV2-project\data\basic_images\0.png', cv2.IMREAD_UNCHANGED) # Load with alpha if available

# Ensure the background image has 4 channels (RGBA)
if background_image.shape[2] == 3:
    background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2BGRA)

# Resize the sign image as needed (example resizing, modify as required)
sign_image_resized = cv2.resize(sign_image, (150, 150))

# Ensure the sign image has 4 channels (RGBA)
if sign_image_resized.shape[2] == 3:
    sign_image_resized = cv2.cvtColor(sign_image_resized, cv2.COLOR_BGR2BGRA)

# Define the position where the sign image will be placed
x, y = 50, 50 # Example position, modify as needed

# Extract the alpha channel from the sign image
alpha_sign = sign_image_resized[:, :, 3] / 255.0
alpha_background = 1.0 - alpha_sign

# Define the region of interest (ROI) in the background image
y1, y2 = y, y + sign_image_resized.shape[0]
x1, x2 = x, x + sign_image_resized.shape[1]

# Blend the images
for c in range(0, 3):
    background_image[y1:y2, x1:x2, c] = (alpha_sign * sign_image_resized[:, :, c] +
                                         alpha_background * background_image[y1:y2, x1:x2, c])

# If you want to keep the alpha channel in the result
background_image[y1:y2, x1:x2, 3] = (alpha_sign * 255 +
                                     alpha_background * background_image[y1:y2, x1:x2, 3])

# Save or display the resulting image
cv2.imwrite('output_image.png', background_image)
# or
cv2.imshow('Overlay Image', background_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
