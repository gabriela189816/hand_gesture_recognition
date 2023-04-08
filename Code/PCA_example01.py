# -*- coding: utf-8 -*-
"""
Created on Mon Apr 03 21:15:00 2023

@author: Gabriela Hilario Acuapan & Luis Alberto Pineda Gómez
File: PCA_example01.py
Comments: Using row vectors
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
#from sys import getsizeof

# ------------- STEP 1. READ IMAGE (WITH PREPROCESING) -----------

m = 50
IMG = np.zeros((200,200,m), dtype=np.uint8) # Array to store images
for i in range(m):
    Img = cv.imread('Frames_Palm_Gesture/frame_0' + str(i) + '.png', cv.IMREAD_GRAYSCALE)      # Read the image in gray scale
    IMG[:,:,i] = Img
    assert IMG[:,:,i] is not None, "file could not be read, check with os.path.exists()"
    
# cv.imshow('Original Image', Img)
# cv.waitKey(0)
# cv.destroyAllWindows()

# ------------- STEP 2. MATRIZ M -------------------
# FLATTENING IMAGE
M = np.zeros((m,40000), dtype=np.uint8)
for i in range(m): 
    M[i,:] = IMG[:,:,i].flatten() #Using Flatten function on array 1 to convert the multi-dimensional


# ------------- STEP 3. AVERAGE FACE ----------------
average = np.mean(M, 0)      # average of each colum of M
average_face = np.array(average, dtype=np.uint8).reshape((200,200))
# cv.imwrite('average_face.png',average_face)
# cv.imshow('Average face', average_face)
# cv.waitKey(0)
# cv.destroyAllWindows()

# --------- STEP 4. DEVIATION OF EACH IMAGE -------------
# Substract the mean from each column of M
A = np.zeros((m,40000), dtype=np.uint8)
for i in range(m):
    A[i,:] = M[i,:] - average # Matrix A centered at the mean (Desviation Matrix)
#print(np.shape(A))

# --------------------- STEP 5. COVARIANCE MATRIX -------------------
L = np.dot(A,np.transpose(A))
#print(L)
#LT = np.transpose(L)
#print(np.array_equal(LT, L))

# STEP 6. EIGENVALUE DECOMPOSITION ON MATRIX L = AA'
e, v = np.linalg.eig(L)
print(np.amax(e))
print(np.amin(e))

# eigval = np.sort(e)[::-1]
# eigvec = np.sort(v)[::-1]
# print(eigval)

U = np.zeros((m,40000), dtype=np.uint8)
for l in range(m):
    for k in range(m):
        U[l,:] = U[l,:] + v[l,k]*A[k,:]         # eigenvectors u_l (eigenfaces)

eigen_face = np.array(U[4,:]).reshape((200,200))
#print(eigen_face)
# cv.imshow('Eigenface 5', eigen_face)
# cv.imwrite('Eigenface 5.png',eigen_face)

#cv.waitKey(0)
#cv.destroyAllWindows()

m2 = 30             # M'
# ------------------ STEP 7. WEIGHTS ------------------------------

## classes
Palm = cv.imread('Frames_Palm_Gesture/frame_00.png', cv.IMREAD_GRAYSCALE)      # Read the image in gray scale
New_imag1 = Palm.flatten()
W1 = np.zeros((1,m2), dtype=np.uint8)
for k in range(m2):
    W1[0,k] = np.dot(U[k,:],(New_imag1 - average))
print(W1)

LetterC = cv.imread('NEW_processed_images/P0_processed/p0_c0.png', cv.IMREAD_GRAYSCALE)      # Read the image in gray scale
New_imag2 = LetterC.flatten()
W2 = np.zeros((1,m2), dtype=np.uint8)
for k in range(m2):
    W2[0,k] = np.dot(U[k,:],(New_imag2 - average))
#print(W2)

Fist = cv.imread('NEW_processed_images/P0_processed/p0_fist0.png', cv.IMREAD_GRAYSCALE)      # Read the image in gray scale
New_imag3 = Fist.flatten()
W3 = np.zeros((1,m2), dtype=np.uint8)
for k in range(m2):
    W3[0,k] = np.dot(U[k,:],(New_imag3 - average))
#print(W3)

Index = cv.imread('NEW_processed_images/P0_processed/p0_index0.png', cv.IMREAD_GRAYSCALE)      # Read the image in gray scale
New_imag4 = Fist.flatten()
W4 = np.zeros((1,m2), dtype=np.uint8)
for k in range(m2):
    W4[0,k] = np.dot(U[k,:],(New_imag4 - average))
#print(W4)

# Euclidian distance
Input_image = cv.imread('NEW_processed_images/P0_processed/p0_index0.png', cv.IMREAD_GRAYSCALE)
input = Input_image.flatten()
O = np.zeros((1,m2), dtype=np.uint8)
for k in range(m2):
    O[0,k] = np.dot(U[k,:],(input - average))
print(O)

dist = (np.linalg.norm(O-W1))**2
print(dist)