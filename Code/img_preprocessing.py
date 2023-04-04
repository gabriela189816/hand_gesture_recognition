# -*- coding: utf-8 -*-
"""
Created on Mon Apr 03 14:00:00 2023

@author: Gabriela Hilario Acuapan & Luis Alberto Pineda Gómez
File: img_preprocessing.py
Comments: Preprocessing of the images to be used
            Step 1. IMAGE CONTOURS: Find the contours of an image
            Step 2. BOUNDING BOX: Resizing images to 200x200 pixels
"""

# ----------------------------------------------------- BOUNDIN BOXING ---------------------------------------

# We use the bounding box function provided by the opencv library
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


# -- READ IMAGE --
original_image = cv.imread('images_code/frame_00_07_0004.png', cv.IMREAD_GRAYSCALE)      # Read the image in gray scale
assert original_image is not None, "file could not be read, check with os.path.exists()"

# hist = cv.calcHist([image], [0], None, [256], [0, 256]) # Histogram of an image
# plt.plot(hist, color='gray' )
# plt.show()

# --- BINARIZE IMAGE ---
threshold_value, threshold = cv.threshold(original_image, 60, 255, cv.THRESH_BINARY) # Returns threshold value and the binarized image
cv.imshow('Original Image', original_image)
cv.imshow('Binarized Image', threshold)

# --- CONTOURS OF THE BINARIZED IMAGE ---
contours, hierarchy = cv.findContours(threshold, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
#cnt = contours[0]                       # cnt contain only the contours of the binarized image.
#epsilon = 0.1*cv.arcLength(cnt,True)    # Epsilon is the maximum distance from contour to approximated contour (ACCURACY PARAMETER).
#approx = cv.approxPolyDP(cnt,epsilon,True)

# --- DRAWING THE CONTOURS IN THE ORIGINAL IMAGE ---
image_contour = cv.cvtColor(original_image, cv.COLOR_GRAY2RGB)
copy = image_contour.copy()               # This copy of the color image is used for the bounding rectangle.
cv.drawContours(image_contour, contours, -1, (255,0,0), 2)
cv.imshow('Countours of the Orginal Image', image_contour)

for contour in range(len(contours)):
    x,y,w,h = cv.boundingRect(contours[contour])            # Bounding rectangle
    #print(x,y,w,h)
    cv.rectangle(copy, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Draw the bounding rectangle

cv.imshow('Bounding box of the Original Image', copy)
cv.waitKey(0)
cv.destroyAllWindows()