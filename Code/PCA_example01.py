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
Img1 = cv.imread('Frames_Palm_Gesture/frame_00.png', cv.IMREAD_GRAYSCALE)      # Read the image in gray scale
#assert Img1 is not None, "file could not be read, check with os.path.exists()"
Img2 = cv.imread('Frames_Palm_Gesture/frame_01.png', cv.IMREAD_GRAYSCALE)
Img3 = cv.imread('Frames_Palm_Gesture/frame_02.png', cv.IMREAD_GRAYSCALE)
Img4 = cv.imread('Frames_Palm_Gesture/frame_03.png', cv.IMREAD_GRAYSCALE)
Img5 = cv.imread('Frames_Palm_Gesture/frame_04.png', cv.IMREAD_GRAYSCALE)
Img6 = cv.imread('Frames_Palm_Gesture/frame_05.png', cv.IMREAD_GRAYSCALE)
Img7 = cv.imread('Frames_Palm_Gesture/frame_06.png', cv.IMREAD_GRAYSCALE)
Img8 = cv.imread('Frames_Palm_Gesture/frame_07.png', cv.IMREAD_GRAYSCALE)
Img9 = cv.imread('Frames_Palm_Gesture/frame_08.png', cv.IMREAD_GRAYSCALE)
Img10 = cv.imread('Frames_Palm_Gesture/frame_09.png', cv.IMREAD_GRAYSCALE)

# cv.imshow('Original Image', Img1)
# cv.waitKey(0)
# cv.destroyAllWindows()

# STEP 2. MATRIZ M
# -- FLATTENING IMAGE --
#Getting the multi-dimensional array from the image
Img1_array = np.array(Img1)
Img2_array = np.array(Img2)
Img3_array = np.array(Img3)
Img4_array = np.array(Img4)
Img5_array = np.array(Img5)
Img6_array = np.array(Img6)
Img7_array = np.array(Img7)
Img8_array = np.array(Img8)
Img9_array = np.array(Img9)
Img10_array = np.array(Img10)
#Memory occupied by the multi-dimensional array
#size1 = getsizeof(array1)
#Using Flatten function on array 1 to convert the multi-dimensional 
# array to 1-D array
Img1_flat = Img1_array.flatten()
Img2_flat = Img2_array.flatten()
Img3_flat = Img3_array.flatten()
Img4_flat = Img4_array.flatten()
Img5_flat = Img5_array.flatten()
Img6_flat = Img6_array.flatten()
Img7_flat = Img7_array.flatten()
Img8_flat = Img8_array.flatten()
Img9_flat = Img9_array.flatten()
Img10_flat = Img10_array.flatten()
#Memory occupied by array 2
#size2 = getsizeof(array2)

M = []
M.append(Img1_flat)
M.append(Img2_flat)
M.append(Img3_flat)
M.append(Img4_flat)
M.append(Img5_flat)
M.append(Img6_flat)
M.append(Img7_flat)
M.append(Img8_flat)
M.append(Img9_flat)
M.append(Img10_flat)

print(np.shape(M))
# STEP 3. AVERAGE FACE
average_face = np.mean(M, 0)   #array2**
print(np.shape(average_face))     # average of each colum of M**

# --------- STEP 4. DEVIATION OF EACH IMAGE -------------
# Substract the mean from each column of M to 
desv_vect1 = Img1_flat - average_face
desv_vect2 = Img2_flat - average_face
desv_vect3 = Img3_flat - average_face
desv_vect4 = Img4_flat - average_face
desv_vect5 = Img5_flat - average_face
desv_vect6 = Img6_flat - average_face
desv_vect7 = Img7_flat - average_face
desv_vect8 = Img8_flat - average_face
desv_vect9 = Img9_flat - average_face
desv_vect10 = Img10_flat - average_face

# Matrix M centered at the mean
MC = []
MC.append(desv_vect1)
MC.append(desv_vect2)
MC.append(desv_vect3)
MC.append(desv_vect4)
MC.append(desv_vect5)
MC.append(desv_vect6)
MC.append(desv_vect7)
MC.append(desv_vect8)
MC.append(desv_vect9)
MC.append(desv_vect10)

print(np.shape(MC))


# STEP 5. COVARIANCE MATRIX

Cov = np.cov(MC)
#print(Cov)
CovT = np.transpose(Cov)
#print(CovT)

print(np.array_equal(Cov, CovT))

c = np.dot(np.transpose(Cov[1,:]), Cov[1,:])
print(c)

# STEP 6. EIGENVALUE DECOMPOSITION ON THE COVARIANCE MATRIX
