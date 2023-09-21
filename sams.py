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

# Function to crop the table from an image
def crop_table(img, img_cor, graph_range):
    pts1 = np.float32(img_cor)
    pts2 = np.float32(graph_range)
    M = cv2.getPerspectiveTransform(pts1, pts2)
    cropped_img = cv2.warpPerspective(img, M, (3000, 3000))
    return cropped_img

# Find biggest contour
def biggest_Contour(contours):
    biggest = np.array([])
    maxArea = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 60:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    return biggest, maxArea