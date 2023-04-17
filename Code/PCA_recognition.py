"""
Created on Sat Apr 08 20:48:00 2023

@author: Gabriela Hilario Acuapan & Luis Alberto Pineda Gómez
File: PCA_recognition.py
Comments: 
"""
import cv2 as cv
import numpy as np
import pandas as pd

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

np.savetxt('Image1',IMG[:,:,0])

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
np.savetxt('Matriz_L',L)
# df = pd.DataFrame(L)
# df.to_excel('Matriz_L.xlsx', sheet_name='example')


# STEP 6. EIGENVALUE DECOMPOSITION ON MATRIX L = AA'
e, v = np.linalg.eig(L)
np.savetxt('eigenvalues',e)
E = sorted(e, reverse=True)
print(E)
np.savetxt('eigenvalues_sorted',E)
# df2 = pd.DataFrame(e)
# df3 = pd.DataFrame(v)
# df2.to_excel('eigenvalues_e.xlsx', sheet_name='e')
# df3.to_excel('eigenvectors_v.xlsx', sheet_name='v')
#print(e)
# print(np.amin(e))

M2 = 40     # M'
U = np.zeros((M2,40000), dtype=np.uint8)
for l in range(M2):
    for k in range(M2):
        U[l,:] = U[l,:] + v[l,k]*A[k,:]         # eigenvectors u_l (eigenfaces)

# plot eigenhands
#for h in range(M2):
    #eigen_hand = np.array(U[h,:]).reshape((200,200))
    #cv.imshow('Eigenface 1', eigen_hand)
    #cv.waitKey(200)
    #cv.imwrite('Eigenhand ' + str(h) + '.png',eigen_hand)

#cv.waitKey(0)
#cv.destroyAllWindows()

# ------------------ STEP 7. WEIGHTS ------------------------------

Omega_clases = np.zeros((7,M2))

# Class one - Palm
W_G1 = np.zeros((10,M2), dtype=np.uint8)
for i in range(0,10):
    for k in range(M2):
        W_G1[0,k] = np.dot(U[k,:],(M[i,:] - average))
Omega_clases[0,:] = np.mean(W_G1,0)

# Class two - C
W_G2 = np.zeros((10,M2), dtype=np.uint8)
for i in range(10,20):
    for k in range(M2):
        W_G2[0,k] = np.dot(U[k,:],(M[i,:] - average))
Omega_clases[1,:] = np.mean(W_G2,0)

# Class three - Fist
W_G3 = np.zeros((10,M2), dtype=np.uint8)
for i in range(20,30):
    for k in range(M2):
        W_G3[0,k] = np.dot(U[k,:],(M[i,:] - average))
Omega_clases[2,:] = np.mean(W_G3,0)

# Class Four - Ok
W_G4 = np.zeros((10,M2), dtype=np.uint8)
for i in range(30,40):
    for k in range(M2):
        W_G4[0,k] = np.dot(U[k,:],(M[i,:] - average))
Omega_clases[3,:] = np.mean(W_G4,0)

# Class five - Peace
W_G5 = np.zeros((10,M2), dtype=np.uint8)
for i in range(40,50):
    for k in range(M2):
        W_G5[0,k] = np.dot(U[k,:],(M[i,:] - average))
Omega_clases[4,:] = np.mean(W_G5,0)

# Class six - I love you
W_G6 = np.zeros((10,M2), dtype=np.uint8)
for i in range(50,60):
    for k in range(M2):
        W_G6[0,k] = np.dot(U[k,:],(M[i,:] - average))
Omega_clases[5,:] = np.mean(W_G6,0)

# Class seven - L
W_G7 = np.zeros((10,M2), dtype=np.uint8)
for i in range(60,70):
    for k in range(M2):
        W_G7[0,k] = np.dot(U[k,:],(M[i,:] - average))
Omega_clases[6,:] = np.mean(W_G7,0)


# print(Omega_G1)
# print(Omega_G2)
# print(Omega_G3)
# print(Omega_G4)
# print(Omega_G5)
# print(Omega_G6)
# print(Omega_G7)

# --------------------------- STEP 8: Clasification -----------------------------------------

Palm = cv.imread('Trainingset_processed/Gesture1_frame00.png', cv.IMREAD_GRAYSCALE)      # Read the image in gray scale
New_imag1 = Palm.flatten()
O_new = np.zeros((1,M2), dtype=np.uint8)
for k in range(M2):
    O_new[0,k] = np.dot(U[k,:],(New_imag1 - average))
print(O_new)

d = np.zeros((1,7))
for i in range(7):
    d[0,i] = (np.linalg.norm(O_new-Omega_clases[i,:]))**2
print(d)
