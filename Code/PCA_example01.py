# -*- coding: utf-8 -*-
"""
Created on Mon Apr 03 21:15:00 2023

@author: Gabriela Hilario Acuapan & Luis Alberto Pineda Gómez
File: PCA_example01.py
Comments: 
"""
import cv2 as cv
import numpy as np
from sys import getsizeof

# STEP 1. READ IMAGE (WITH PREPROCESING)
original_image = cv.imread('Frames_Palm_Gesture/frame_00.png', cv.IMREAD_GRAYSCALE)      # Read the image in gray scale
assert original_image is not None, "file could not be read, check with os.path.exists()"

# STEP 2. MATRIZ M
# -- FLATTENING IMAGE --
#Getting the multi-dimensional array from the image
array1 = np.array(original_image)
#Memory occupied by the multi-dimensional array
size1 = getsizeof(array1)
print(array1)
print(size1)
#Using Flatten function on array 1 to convert the multi-dimensional 
# array to 1-D array
array2 = array1.flatten()
#Memory occupied by array 2
size2 = getsizeof(array2)
#displaying the 1-D array
print(array2)
print(size2)


# STEP 3. AVERAGE FACE
average_face = np.mean(array1, 0)   #array2**
print(average_face)     # average of each colum of M**

# STEP 4. DEVIATION OF EACH IMAGE

# STEP 5. COVARIANCE MATRIX