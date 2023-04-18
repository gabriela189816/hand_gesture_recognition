"""
Created on Friday Apr 07 00:15:00 2023

@author: Gabriela Hilario Acuapan & Luis Alberto Pineda Gómez
File: Preprocessing.py
Comments: Preprocessing of images.
            ------------------------- Steps --------------------------
            (1) Binarize the loaded image
            (2) Obtain and draw the contours of the image
            (3) Crop of the image
            (4) Resize of the image to the standard 200 x 200 pixels
            (5) Save all the images in the standard size
"""

# --- IMPORT LIBRARIES ---
import numpy
import numpy as np
import cv2 as cv

def img_prepro():
    # LOAD THE INPUT IMAGE
    image = cv.imread('Input_image.png', cv.IMREAD_GRAYSCALE)

    # WE EVALUATE IF THE IMAGE LOADED IS AN IMAGE TYPE
    if type(image) is numpy.ndarray:
        original = image.copy()

        # (1) ----- BINARIZE THE IMAGE -----
        threshold_value, threshold = cv.threshold(image, 128, 255, cv.THRESH_BINARY)
        # (2.1) ----- FIND THE CONTOURS OF THE CURRENT IMAGE
        contours, hierarchy = cv.findContours(threshold, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

        # (2.2) ----- DRAWING THE CONTOURS OF THE IMAGE ----
        image_color = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        copy = image_color.copy()  # This copy of the color image is used for the bounding rectangle.
        cv.drawContours(image_color, contours, -1, (255, 0, 0), 2)

        value = 0
        num = 0
        for contour in range(len(contours)):
            if len(contours[contour]) > value:
                num = contour
                value = len(contours[contour])
            else:
                pass
        # (2.3) ----- BOUNDING RECTANGLE ----
        x, y, w, h = cv.boundingRect(contours[num])

        # (2.4) ----- DRAW THE BOUNDING RECTANGLE ---
        cv.rectangle(copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
        print(x, y, w, h)

        # (3) ----- CROPPING OFF THE IMAGE -----
        boundrie = 10
        if x <= 10:
            crop = original[y:y + h, x:x + w]
        else:
            crop = original[y - boundrie:y + h + boundrie, x - boundrie:x + w + boundrie]
        print("The dimensions of the cropped image are: ", np.shape(crop))

        # (4) ----- RESIZE OF THE IMAGE -----
        height = 200
        width = 200
        dim = (height, width)
        frame = cv.resize(crop, dim, interpolation=cv.INTER_AREA) # Resize of the cropped image
        print("The dimensions of the new resized image are:,", np.shape(frame))
        
        cv.imshow('RESIZED IMAGE', frame)
        cv.waitKey(0)
        cv.destroyAllWindows()

        save = input()
        if save == 's':
            cv.imwrite('Resized_image.png', frame)
            print("Saved frame")
        else: 
            print('Try again!')
        
