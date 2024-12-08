import cv2
import numpy as np
import pytesseract
import sys
import os.path

if len(sys.argv) != 3:
    print("%s input_file output_file" % (sys.argv[0]))
    sys.exit()
else:
    input_file = sys.argv[1]
    output_file = sys.argv[2]

# Check if the input file exists
if not os.path.isfile(input_file):
    print("No such file '%s'" % input_file)
    sys.exit()

DEBUG = 0

# Initialize Tesseract path if it's not in your PATH environment variable (update the path as per your installation)
# Uncomment and update the following line as needed:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Function to perform OCR and extract text
def extract_text_from_image(image):
    return pytesseract.image_to_string(image)

# Function to determine pixel intensity (same as before)
def ii(xx, yy):
    global img, img_y, img_x
    if yy >= img_y or xx >= img_x:
        return 0
    pixel = img[yy][xx]
    return 0.30 * pixel[2] + 0.59 * pixel[1] + 0.11 * pixel[0]

# A quick test to check whether the contour is a connected shape
def connected(contour):
    first = contour[0][0]
    last = contour[len(contour) - 1][0]
    return abs(first[0] - last[0]) <= 1 and abs(first[1] - last[1]) <= 1

# Helper function to return a given contour
def c(index):
    global contours
    return contours[index]

# Check if the contour should be kept based on size and shape
def keep(contour):
    return keep_box(contour) and connected(contour)

# Whether we should keep the containing box of this contour based on its shape
def keep_box(contour):
    xx, yy, w_, h_ = cv2.bounadingRect(contour)
    w_ *= 1.0
    h_ *= 1.0
    if w_ / h_ < 0.1 or w_ / h_ > 10:
        return False
    if ((w_ * h_) > ((img_x * img_y) / 5)) or ((w_ * h_) < 15):
        return False
    return True

# Load the image
orig_img = cv2.imread(input_file)

# Check if the image was loaded successfully
if orig_img is None:
    print(f"Error loading image '{input_file}'. Please check the file format or path.")
    sys.exit()

# Add a border to the image for processing sake
img = cv2.copyMakeBorder(orig_img, 50, 50, 50, 50, cv2.BORDER_CONSTANT)

# Calculate the width and height of the image
img_y = len(img)
img_x = len(img[0])

# Split out each channel
blue, green, red = cv2.split(img)

# Run canny edge detection on each channel
blue_edges = cv2.Canny(blue, 200, 250)
green_edges = cv2.Canny(green, 200, 250)
red_edges = cv2.Canny(red, 200, 250)

# Join edges back into image
edges = blue_edges | green_edges | red_edges

# Find the contours
contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

hierarchy = hierarchy[0]

# These are the boxes that we are determining
keepers = []

# For each contour, find the bounding rectangle and decide if it's one we care about
for index_, contour_ in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour_)

    # Check the contour and its bounding box
    if keep(contour_):
        keepers.append([contour_, [x, y, w, h]])

# Make a white copy of our image
new_image = edges.copy()
new_image.fill(255)

# For each of our keepers, we want to copy the region from the original image onto the new image
for keeper in keepers:
    contour, box = keeper
    x, y, w, h = box
    cv2.drawContours(new_image, [contour], 0, (0, 0, 0), -1)

# Apply OCR to the extracted image regions (new_image)
extracted_text = extract_text_from_image(new_image)

# Write the extracted text to the output file
with open(output_file, 'w') as file:
    file.write(extracted_text)

print(f"Text extracted from '{input_file}' and saved to '{output_file}'")
