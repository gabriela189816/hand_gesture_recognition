# This is the first version of the Hand gesture recognition project using Principal Component Analysis (PCA).
# Given a training data set, we will attempt to classify different hand gestures when projecting them onto the hand space.
# Load multiple images using openCv and s module
# Binarize the loaded image
# Obtain and draw the contours of the image
# Crop of the image
# Resize of the image to the standard 200 x 200 pixels
# Save all the images in the standard size

# Created by Luis Alberto Pineda
# Date: 03/04/2023
# Version 1.2

# --- LIBRARIES ---
import os

import numpy
import numpy as np
import cv2 as cv

# PATH
path ="/Users/lapg/Documents/2do Cuatrimestre/Vision/Proyecto-Hand-Recognition/Images/palm_gesture"
list_dir = os.listdir(path)
print("Total amount of files in the current directory:", len(list_dir))
i = 0 # Variable for saving the frames

# --- LOAD EACH IMAGE AND BEGIN THE PRE-PROCESSING ---
for file in list_dir:
    # LOAD A SINGLE IMAGE
    image = cv.imread(
"/Users/lapg/Documents/2do Cuatrimestre/Vision/Proyecto-Hand-Recognition/Images/palm_gesture/" + str(file),
        cv.IMREAD_GRAYSCALE)
    if type(image) is numpy.ndarray: # WE EVALUATE IF THE IMAGE LOADED IS AN IMAGE TYPE
        original = image.copy()
        # BINARIZE THE IMAGE
        threshold_value, threshold = cv.threshold(image, 130, 255, cv.THRESH_BINARY)
        # cv.imshow('Binarized Image', threshold)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        # FIND THE CONTOURS OF THE CURRENT IMAGE
        contours, hierarchy = cv.findContours(threshold, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

        # --- DRAWING THE CONTOURS OF THE IMAGE ---
        image_color = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        copy = image_color.copy()  # This copy of the color image is used for the bounding rectangle.
        cv.drawContours(image_color, contours, -1, (255, 0, 0), 2)
        #cv.imshow('CONTOURS OF THE IMAGE', image_color)
        #cv.waitKey(0)
        #cv.destroyAllWindows()

        for contour in range(len(contours)):
            # --- BOUNDING RECTANGLE ---
            x, y, w, h = cv.boundingRect(contours[contour])
            # --- DRAW THE BOUNDING RECTANGLE ---
            cv.rectangle(copy, (x, y), (x + w, y + h), (0, 0, 255), 2)

        #print(x, y, w, h)
        #cv.imshow('BOUNDING BOX', copy)
        #cv.waitKey(0)
        #cv.destroyAllWindows()

        # --- CROPPING OFF THE IMAGE ---
        crop = original[y:y + h, x:x + w]
        print("The dimensions of the cropped image are: ", np.shape(crop))
        #cv.imshow('CROPPED IMAGE', crop)
        #cv.waitKey(0)
        #cv.destroyAllWindows()

        # --- RESIZE OF THE IMAGE ---
        # Resize of the cropped image
        height = 200
        width = 200
        dim = (height, width)
        frame = cv.resize(crop, dim, interpolation=cv.INTER_AREA)
        #print("The dimensions of the new resized image are:,", np.shape(resized))
        cv.imshow('RESIZED IMAGE', frame)
        cv.waitKey(0)
        cv.destroyAllWindows()

        # SAVE THE RESIZED IMAGE TO THE FOLDER
        save_frames_path = "/Users/lapg/Documents/2do Cuatrimestre/Vision/Proyecto-Hand-Recognition/Images/palm_gesture/frames/frame_0"+str(i)+".png"
        cv.imwrite(save_frames_path, frame)
        print("Saved frame")
        i += 1
    else:
        # PRINT THAT THE "IMAGE" IS NOT ACTUALLY AN IMAGE
        print(f"file {file} is not an image")

print("Done")