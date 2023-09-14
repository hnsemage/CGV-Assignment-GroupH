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

