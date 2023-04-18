"""
Created on Friday Apr 07 00:15:00 2023

@author: Gabriela Hilario Acuapan & Luis Alberto Pineda Gómez
File: Clasification.py
Comments: The purpose of this function is to classify an input image, given the weights of each class obtainged by the PCA

            ---------- INPUTS ----------
            classes := A vector containing all the weights of each class [np.array]
            test_image := An image to be classified. [np.array]
            reduction := Value for reducing the dimension of the data set [int]
            reduced_data_set
            average_hands_flatten

            ---------- OUTPUT ----------
            classes := Array containing all the weights of each class of the gestures. [np.array (1x7)]
"""

# ---------- LIBRARIES ----------
import os
import numpy
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time
import math
t = time.time()

def classify(classes, test_image, reduction, reduced_data_set, average_hands_flatten):

    # ----------------- READ THE INPUT IMAGE
    test_image_path = test_image
    test_image = cv.imread(test_image_path, cv.IMREAD_GRAYSCALE)
    test_image = test_image / 255
    test_image_weights = np.zeros((1, reduction), dtype=np.float64)
    flatten_test_image = test_image.flatten()
    N_squared = 40_000
    flatten_test_image = flatten_test_image.reshape((N_squared, 1))

    # ---------------- COMPUTE THE WEIGHTS OF THE TEST IMAGE
    for k in range(np.shape(reduced_data_set)[0]):
        reduced_data_set_reshape = reduced_data_set[k, :]
        reduced_data_set_reshape = reduced_data_set_reshape.reshape((1, N_squared))
        weight = np.matmul(reduced_data_set_reshape, (flatten_test_image - average_hands_flatten))
        test_image_weights[0, k] = test_image_weights[0, k] + weight.item()
    #print(f"Test Image Max Weight:{np.amax(test_image_weights)}, Min weight:{np.amin(test_image_weights)}")

    euclidean_distance = np.zeros((1, 7))

    # Variable for comparison
    comparison = math.inf
    position = 0
    classes_name = ["PALM", "C", "FIST", "OK", "PEACE", "ROCK", "INDEX"]

    # ----------------- LOAD AN IMAGE OF EACH CLASS FOR REPRESENTATION
    palm_path = "C:/Users/gabri/OneDrive/Documentos/GitHub/Hand_gesture_recognition/Own_dataset_hands/Gesture0_frame00.png"
    palm = cv.imread(palm_path, cv.IMREAD_GRAYSCALE)
    c_path = "C:/Users/gabri/OneDrive/Documentos/GitHub/Hand_gesture_recognition/Own_dataset_hands/Gesture1_frame00.png"
    c = cv.imread(c_path, cv.IMREAD_GRAYSCALE)
    fist_path = "C:/Users/gabri/OneDrive/Documentos/GitHub/Hand_gesture_recognition/Own_dataset_hands/Gesture2_frame05.png"
    fist = cv.imread(fist_path, cv.IMREAD_GRAYSCALE)
    ok_path = "C:/Users/gabri/OneDrive/Documentos/GitHub/Hand_gesture_recognition/Own_dataset_hands/Gesture3_frame01.png"
    ok = cv.imread(ok_path, cv.IMREAD_GRAYSCALE)
    peace_path = "C:/Users/gabri/OneDrive/Documentos/GitHub/Hand_gesture_recognition/Own_dataset_hands/Gesture4_frame00.png"
    peace = cv.imread(peace_path, cv.IMREAD_GRAYSCALE)
    rock_path = "C:/Users/gabri/OneDrive/Documentos/GitHub/Hand_gesture_recognition/Own_dataset_hands/Gesture5_frame01.png"
    rock = cv.imread(rock_path, cv.IMREAD_GRAYSCALE)
    index_path = "C:/Users/gabri/OneDrive/Documentos/GitHub/Hand_gesture_recognition/Own_dataset_hands/Gesture6_frame00.png"
    index = cv.imread(index_path, cv.IMREAD_GRAYSCALE)
    images = [palm, c, fist, ok, peace, rock, index]

    # --------------------------- COMPUTE OF EUCLIDEAN DISTANCE 
    for cl in range(np.shape(classes)[0]):
        euclidean_distance[0, cl] = np.linalg.norm((test_image_weights[0, :] - classes[cl, :]), ord=2)
        if euclidean_distance[0, cl] < comparison:
            comparison = euclidean_distance[0, cl]
            position = cl
        else:
            pass
    # --------------------------- CLASSIFY THE INPUT IMAGE
    print("\n Comparación de distancia euclideana con respecto a cada clase: \n", euclidean_distance)
    print(f"\n \t La distancia Euclideana más corta es: {comparison} y pertenece a la clase {classes_name[position]}")

    # --------------------------- PLOT INPUT IMAGE AND ITS CLASS  
    fig, axis = plt.subplots(ncols=2, nrows=1)
    axis[0].set_title("Input Image")
    axis[0].imshow(test_image, cmap="gray")
    axis[1].set_title("Class to which it belongs")
    axis[1].imshow(images[position], cmap="gray")
    plt.show()