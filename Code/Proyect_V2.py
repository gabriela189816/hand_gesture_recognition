# This is the first version of the Hand gesture recognition project using Principal Component Analysis (PCA).
# Given a training data set, we will attempt to classify different hand gestures when projecting them onto the hand space.
# Load each and every image of the given path in order to compute the normalized version and add it to a general matrix

# Created by Luis Alberto Pineda
# Date: 03/04/2023
# Version 1.0

# --- LIBRARIES ---
import os
import numpy
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# --- PATH ---
path = "/Users/lapg/Documents/2do Cuatrimestre/Vision/Proyecto-Hand-Recognition/Images/palm_gesture/frames"
list_dir = os.listdir(path)
#print("Total amount of files in the current directory:", len(list_dir))
#print(list_dir)

# ARRAY where the normalized values of all images will be stored
width = 200
height= 200
dimensions = (height, width)
norm_array = np.zeros(dimensions, dtype= np.int8)
#print(np.shape(norm_array))
# --- NORMALIZE EACH IMAGE AND STORE IT IN A NEW MATRIX
for file in list_dir:
    image = cv.imread(
        "/Users/lapg/Documents/2do Cuatrimestre/Vision/Proyecto-Hand-Recognition/Images/palm_gesture/frames/" + str(file),
        cv.IMREAD_GRAYSCALE)
    if type(image) is numpy.ndarray:  # WE EVALUATE IF THE IMAGE LOADED IS AN IMAGE TYPE
        norm_array = norm_array + image/np.linalg.norm(image)
        #cv.imshow("Normalized input image", image)
        #cv.waitKey(0)
        #cv.destroyAllWindows()

    else:
        pass
print("The max/min values of the normalized array are:", np.amax(norm_array), np.amin(norm_array))

cv.imshow("Normalized Image", norm_array)
cv.waitKey(0)
cv.destroyAllWindows()

# --- STANDARIZATION OF THE DATA ---
standardized_data = (norm_array - norm_array.mean(axis = 0)) / norm_array.std(axis = 0)
cv.imshow("Standarized Image", standardized_data)
cv.waitKey(0)
cv.destroyAllWindows()

# --- COVARIANCE MATRIX ---
covariance_matrix = np.cov(standardized_data, ddof = 0, rowvar = False)
cv.imshow("Covariance Matrix", covariance_matrix)
cv.waitKey(0)
cv.destroyAllWindows()

# --- COMPUTE THE EIGENVALUES & EIGENVECTORS OF THE COVARIANCE MATRIX ---
#eigenvalues = np.linalg.eigvalsh(covariance_matrix)
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
#print(len(eigenvalues))
#print(eigenvalues)

# --- SORTING THE PRINCIPAL COMPONENTES ---
sorted_eigenvalues = np.sort(eigenvalues)[::-1]
#print(sorted_eigenvalues)
#print("\n")
sorted_eigenvectors = np.sort(eigenvectors)[::-1]

# --- EXPLAINED VARIANCE ---
explained_variance = sorted_eigenvalues / np.sum(sorted_eigenvalues)
#print(explained_variance)
plt.plot(np.cumsum(explained_variance))
plt.title("PCA vs Total Explained Variance")
plt.ylabel("Total Explained Variance")
plt.xlabel("Number of principal components")

# --- REDUCED DATA ---
reduction = 10
reduced_data = np.matmul(standardized_data, sorted_eigenvectors[:,:reduction])
#print(np.shape(reduced_data))

# --- SINGULAR VALUE DECOMPOSITION ---
u_matrix, singular, v_matrix = np.linalg.svd(norm_array)
#print(np.shape(u_matrix), np.shape(singular), np.shape(v_matrix))
plt.plot(np.cumsum(singular/sum(singular)))
plt.title("SVD vs Total Explained Variance")
plt.ylabel("Total Explained Variance")
plt.xlabel("Number of principal components")
plt.grid()
plt.show()

# --- REDUCING DIMENSIONALITY SVD ---
width = 200
height= 200
dimensions_reduced = (height, reduction)
# U Matrix
u_matrix_reduced = np.zeros(dimensions_reduced, dtype= np.int8)
u_matrix_reduced = u_matrix[:, 0:reduction]
#print(u_matrix_reduced.shape)
# Singular Array
reshape_singular = singular.reshape((200, 1))
singular_reduced = np.zeros([reduction, 1])
singular_reduced = reshape_singular[0:reduction] * np.identity(reduction)
#print(singular_reduced.shape)
# V Matrix
v_matrix_reduced = np.zeros((height, reduction), dtype=np.int8)
v_matrix_reduced = v_matrix[0:reduction, :]
#print(v_matrix_reduced.shape)

image_reconstruction = u_matrix_reduced @ singular_reduced @ v_matrix_reduced
#print(image_reconstruction.shape)
cv.imshow("Reconstructed Image with dimensionality reduction", image_reconstruction)
cv.waitKey(0)
cv.destroyAllWindows()