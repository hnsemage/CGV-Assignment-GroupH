import cv2
import numpy as np
from PIL import Image

def grayscale():
    # Load the image
    image_path = "C:\\Users\\ASUS\\Desktop\\4th year 2nd semester\\CS402.3 - Computer Graphics and Visulation\\Assignment\\Final Assignment\\GroupH\\Images\\1.jpeg"
    attendance_sheet = Image.open(image_path)

    # Convert to grayscale
    gray_sheet = attendance_sheet.convert("L")

    # Convert to NumPy array for further processing
    sheet_array = np.array(gray_sheet)
    return sheet_array

# Apply Gaussian blur and Canny edge detection
blurred = cv2.GaussianBlur(grayscale(), (5, 5), 0)
edged = cv2.Canny(blurred, 50, 150)


kernel = np.ones((5, 5), np.uint8)

# Dilation

dilated_binary_crop = cv2.dilate(crop, kernel, iterations=1)

cv2.imwrite(
    'C:\\Users\\ASUS\Desktop\\4th year 2nd semester\\CS402.3 - Computer Graphics and Visulation\\Assignment\\Final Assignment\\GroupH\\ProcessedImages\\dilated_image.jpg',
    dilated_binary_crop)

# Erosion
eroded_binary_crop = cv2.erode(dilated_binary_crop, kernel, iterations=1)

cv2.imwrite(
    'C:\\Users\\ASUS\Desktop\\4th year 2nd semester\\CS402.3 - Computer Graphics and Visulation\\Assignment\\Final Assignment\\GroupH\\ProcessedImages\\eroded_image.jpg',
    eroded_binary_crop)

#Histogram before Otsu's algorithm
plt.hist(eroded_binary_crop.ravel(), 256, [0, 256])
plt.title( "Histogram")
plt.savefig("C:\\Users\\ASUS\\Desktop\\4th year 2nd semester\\CS402.3 - Computer Graphics and Visulation\\Assignment\\Final Assignment\\GroupH\\ProcessedImages\\crop_histogram_image.png")
#plt.show()
