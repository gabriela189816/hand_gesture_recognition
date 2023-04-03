# This is an example of how to use the bounding box function provided by the opencv library.

import cv2 as cv

# Read the image in gray scale
image = cv.imread("/Users/lapg/Documents/2do Cuatrimestre/Vision/Proyecto/example_star.png", cv.IMREAD_GRAYSCALE)
assert image is not None, "file could not be read, check with os.path.exists()"
# Perform a copy of the original image
original = image.copy()

# --- BINARIZE IMAGE ---
# Returns threshold value and the binarized image
threshold_value, threshold = cv.threshold(image, 60, 255, cv.THRESH_BINARY)
cv.imshow('Binarized Image', threshold)
cv.waitKey(0)
cv.destroyAllWindows()

# --- CONTOURS OF THE IMAGE ---
contours, hierarchy = cv.findContours(threshold, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
cnt = contours[0] #cnt contain only the contours of the binarized image.
epsilon = 0.1*cv.arcLength(cnt,True) # Epsilon is the maximum distance from contour to approximated contour (ACCURACY PARAMETER).
approx = cv.approxPolyDP(cnt,epsilon,True)

# --- DRAWING THE CONTOURS OF THE IMAGE ---
image_color = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
copy = image_color.copy() # This copy of the color image is used for the bounding rectangle.
cv.drawContours(image_color, contours, -1, (255,0,0), 2)
cv.imshow('CONTOURS OF THE IMAGE', image_color)
cv.waitKey(0)
cv.destroyAllWindows()

for contour in range(len(contours)):
# --- BOUNDING RECTANGLE ---
    x,y,w,h = cv.boundingRect(contours[contour])
#print(x,y,w,h)

# --- DRAW THE BOUNDING RECTANGLE ---
    cv.rectangle(copy, (x, y), (x+w, y+h), (0, 0, 255), 2)

cv.imshow('BOUNDING BOX', copy)
cv.waitKey(0)
cv.destroyAllWindows()