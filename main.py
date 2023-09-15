import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

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

# Find lines in the image using Hough Line Transform
lines = cv2.HoughLinesP(edged, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

img_cor = [[450, 1300], [430, 2140], [2560, 1375], [2560, 2220]]
graph_range = [[10,10],[10,3000],[3000,10],[3000,3000]]

pts1 = np.float32(img_cor)
pts2 = np.float32(graph_range)
M = cv2.getPerspectiveTransform(pts1, pts2)

crop=cv2.warpPerspective(grayscale(), M, (3000,3000))
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

# Apply Otsu's thresholding
_, binary_crop = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Save or display the binary image
cv2.imwrite(
    'C:\\Users\\ASUS\\Desktop\\4th year 2nd semester\\CS402.3 - Computer Graphics and Visulation\\Assignment\\Final Assignment\\GroupH\\CroppedImages\\binary_image.jpg',
    binary_crop)

#Histogram after Otsu's algorithm
plt.hist(binary_crop.ravel(), 256, [0, 256])
plt.title( "Histogram")
plt.savefig("C:\\Users\\ASUS\\Desktop\\4th year 2nd semester\\CS402.3 - Computer Graphics and Visulation\\Assignment\\Final Assignment\\GroupH\\ProcessedImages\\histogram_image.png")
#plt.show()