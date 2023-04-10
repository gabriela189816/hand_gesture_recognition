"""
Created on Sat Apr 08 20:48:00 2023

@author: Gabriela Hilario Acuapan & Luis Alberto Pineda Gómez
File: PCA_recognition.py
Comments: 
"""
import cv2 as cv
import numpy as np

# **********************************************************
# ------------- STEP 1. READ IMAGES PREPROCESSED -----------
# **********************************************************

# global variables
gestures = 7
images = 10
j = 0
IMG = np.zeros((200,200, gestures*images), dtype=np.uint8) # Array to store images
for g in range(gestures):
    for i in range(images):
        # Read the image in gray scale
        Img = cv.imread('Trainingset_processed/Gesture' + str(g) + '_frame0' + str(i) +'.png', cv.IMREAD_GRAYSCALE)   
        IMG[:,:,j] = Img
        assert Img is not None, "file could not be read, check with os.path.exists()"
        j +=1

# for i in range(60,70):
#     cv.imshow('Original Image', IMG[:,:,i])
#     cv.waitKey(300)
# cv.waitKey(0)
# cv.destroyAllWindows()


# ------------- STEP 2. MATRIZ M -------------------
# FLATTENING IMAGE
M = np.zeros((gestures*images,40000), dtype=np.uint8)
for i in range(gestures*images): 
    M[i,:] = IMG[:,:,i].flatten() #Using Flatten function on array 1 to convert the multi-dimensional

# ------------- STEP 3. AVERAGE FACE ----------------
average = np.mean(M, 0)      # average of each colum of M
average_hand = np.array(average, dtype=np.uint8).reshape((200,200))
# cv.imwrite('average_hand.png', average_hand)
# cv.imshow('Average face', average_hand)
# cv.waitKey(0)
# cv.destroyAllWindows()

# --------- STEP 4. DEVIATION OF EACH IMAGE -------------
# Substract the mean from each column of M
A = np.zeros((gestures*images,40000), dtype=np.uint8)
for i in range(gestures*images):
    A[i,:] = M[i,:] - average # Matrix A centered at the mean (Desviation Matrix)
#print(np.shape(A))

# --------------------- STEP 5. COVARIANCE MATRIX -------------------
L = np.dot(A,np.transpose(A))
#print(np.shape(L))

# STEP 6. EIGENVALUE DECOMPOSITION ON MATRIX L = AA'
e, v = np.linalg.eig(L)
#print(e)
# print(np.amin(e))

U = np.zeros((gestures*images,40000), dtype=np.uint8)
for l in range(gestures*images):
    for k in range(gestures*images):
        U[l,:] = U[l,:] + v[l,k]*A[k,:]         # eigenvectors u_l (eigenfaces)

M2 = 10     # M'
for h in range(M2):
    eigen_hand = np.array(U[h,:]).reshape((200,200))
    #cv.imshow('Eigenface 1', eigen_hand)
    #cv.waitKey(200)
    cv.imwrite('Eigenhand ' + str(h) + '.png',eigen_hand)

#cv.waitKey(0)
#cv.destroyAllWindows()
