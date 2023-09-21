import cv2
import numpy as np
from PIL import Image



# Function to convert an image to grayscale
def grayscale(image):
    # Load the image
    attendance_sheet = Image.open(image)

    # Convert to grayscale
    gray_sheet = attendance_sheet.convert("L")

    # Convert to NumPy array for further processing
    sheet_array = np.array(gray_sheet)

    cv2.imshow("Grayscale image", gray_sheet)
    return sheet_array

# Apply Gaussian blur and Canny edge detection
def preprocess_image(image):
    img = grayscale(image)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    cv2.imshow("Blured image", blurred)
    edged = cv2.Canny(blurred, 50, 150)
    cv2.imshow("Canny image", edged)
    return edged