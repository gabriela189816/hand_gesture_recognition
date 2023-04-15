# This is the 5th version of the Hand gesture recognition project using Principal Component Analysis (PCA).
# Given a training data set, we will attempt to classify different hand gestures when projecting them onto the "hand space".
# Instead of using the features of each image i.e. (N^2) where N is the number of pixels of each image. We will use each image.
# In this script, we will use the dataset created by our own.

# Created by Luis Alberto Pineda
# Date: 08/04/2023
# Version 1.5

# ---------- LIBRARIES ----------
import os
import numpy
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time
t = time.time()

# ---------- PROCESSING OF THE FRAMES ----------
# We assume that we have already an existing database. We will fix the size of the frames if needed.
path_processed_frames = "/Users/lapg/Documents/2do Cuatrimestre/Vision/Proyecto-Hand-Recognition/Own_dataset_hands/"
directory = "Processed_frames"
# os.mkdir((str(path_processed_frames)+str(directory)))

# --- PATH FOR READING THE FRAMES TO BE PROCESSED ---
path = "/Users/lapg/Documents/2do Cuatrimestre/Vision/Proyecto-Hand-Recognition/Own_dataset_hands/"
list_dir = os.listdir(path)
print("Total amount of files in the current directory:", len(list_dir))

# --- LOAD EACH IMAGE IN THE PATH ---
# for img in list_dir:
#     image = cv.imread(str(path) + str(img), cv.IMREAD_GRAYSCALE)
#     if type(image) is numpy.ndarray:
#         original = image.copy()
#         thre  shold_value, threshold = cv.threshold(image, 20, 255, cv.THRESH_BINARY)
#         cv.imshow('Binarized Image', threshold)
#         cv.waitKey(0)
#         cv.destroyAllWindows()
#         # --- FIND THE CONTOURS OF THE CURRENT IMAGE ---
#         contours, hierarchy = cv.findContours(threshold, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
#
#         # --- DRAWING THE CONTOURS OF THE IMAGE ---
#         image_color = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
#         copy = image_color.copy()  # This copy of the color image is used for the bounding rectangle.
#         cv.drawContours(image_color, contours, -1, (255, 0, 0), 2)
#         # cv.imshow('CONTOURS OF THE IMAGE', image_color)
#         # cv.waitKey(0)
#         # cv.destroyAllWindows()
#         value = 0
#         num = 0
#         for contour in range(len(contours)):
#             if len(contours[contour]) > value:
#                 num = contour
#                 value = len(contours[contour])
#             else:
#                 pass
#         # --- BOUNDING RECTANGLE ---
#         x, y, w, h = cv.boundingRect(contours[num])
#         # --- DRAW THE BOUNDING RECTANGLE ---
#         cv.rectangle(copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
#         print(x, y, w, h)
#         cv.imshow('BOUNDING BOX', copy)
#         cv.waitKey(0)
#         cv.destroyAllWindows()
#
#         # --- CROPPING OFF THE IMAGE ---
#         boundrie = 10
#         if x <= 10:
#             crop = original[y:y + h, x:x + w]
#         else:
#             crop = original[y - boundrie:y + h + boundrie, x - boundrie:x + w + boundrie]
#         print("The dimensions of the cropped image are: ", np.shape(crop))
#         # cv.imshow('CROPPED IMAGE', crop)
#         # cv.waitKey(0)
#         # cv.destroyAllWindows()
#
#         # --- RESIZE OF THE IMAGE ---
#         # Resize of the cropped image
#         height = 200
#         width = 200
#         dim = (height, width)
#         frame = cv.resize(crop, dim, interpolation=cv.INTER_AREA)
#         # print("The dimensions of the new resized image are:,", np.shape(resized))
#         # cv.imshow('RESIZED IMAGE', frame)
#         # cv.waitKey(0)
#         # cv.destroyAllWindows()
#
#         # --- PATH TO SAVE THE PROCESSED FRAMES ---
#         processed_frames = "/Users/lapg/Documents/2do Cuatrimestre/Vision/Proyecto-Hand-Recognition/Own_dataset_hands/"
#         # --- SAVE THE RESIZED FRAMES TO THE DESTINY FOLDER ---
#         cv.imwrite(str(processed_frames) + str(directory) +str ("/")+ str(img), frame)
# print("Done")

# ---------- AVERAGE FRAME CREATION & CLASSIFICATION OF EACH GESTURE----------
# --- Path for reading the frames ---
frames_path = "/Users/lapg/Documents/GitHub/Hand_gesture_recognition/Own_dataset_hands/Processed_frames/"
list_dir_frames = os.listdir(frames_path)
print("Total amount of frames in the frame directory:", len(list_dir_frames))

width = 200
height = 200
dimensions = (height, width)
average_hands = np.zeros(dimensions, dtype= np.int32)

# --- PALM GESTURE ---
i0 = 0 # Variable for counting the number of PALM gestures
palm_gestures = np.zeros((10, width*height), dtype=np.uint8)

# --- C GESTURE ---
i1 = 0 # Variable for counting the number of C gestures
c_gestures = np.zeros((10, width*height), dtype=np.uint8)

# --- FIST GESTURE ---
i2 = 0 # Variable for counting the number of FIST gestures
fist_gestures = np.zeros((10, width*height), dtype=np.uint8)

# --- OK GESTURE ---
i3 = 0 # Variable for counting the number of OK gestures
ok_gestures = np.zeros((10, width*height), dtype=np.uint8)

# --- PEACE GESTURE ---
i4 = 0 # Variable for counting the number of PACE gestures
peace_gestures = np.zeros((10, width*height), dtype=np.uint8)

# --- ROCK GESTURE ---
i5 = 0 # Variable for counting the number of ROCK gestures
rock_gestures = np.zeros((10, width*height), dtype=np.uint8)

# --- INDEX GESTURE ---
i6 = 0 # Variable for counting the number of INDEX gestures
index_gestures = np.zeros((10, width*height), dtype=np.uint8)

# Variable for counting the number of images that are being loaded into the system.
count = 0

for file in list_dir_frames:
    image = cv.imread(str(frames_path) + str(file), cv.IMREAD_GRAYSCALE)
    # EVALUATE IF THE IMAGE LOADED IS AN IMAGE TYPE
    if type(image) is numpy.ndarray:
        # ADDITION OF THE IMAGES
        average_hands = average_hands + image
        count += 1
        # --- CLASSIFICATION OF EACH GESTURE ---
        # PALM GESTURE
        if str(file[7]) == "0":
            flat_palm_gesture = image.flatten()
            flat_palm_gesture = flat_palm_gesture.reshape(1, width*height)
            palm_gestures[i0, :] = flat_palm_gesture[0, :]
            i0 += 1
        # C GESTURE
        elif str(file[7]) == "1":
            flat_c_gesture = image.flatten()
            flat_c_gesture = flat_c_gesture.reshape(1, width * height)
            c_gestures[i1, :] = flat_c_gesture[0, :]
            i1 += 1
        # FIST GESTURE
        elif str(file[7]) == "2":
            flat_fist_gesture = image.flatten()
            flat_fist_gesture = flat_fist_gesture.reshape(1, width * height)
            fist_gestures[i2, :] = flat_fist_gesture[0, :]
            i2 += 1
        # OK GESTURE
        elif str(file[7]) == "3":
            flat_ok_gesture = image.flatten()
            flat_ok_gesture = flat_ok_gesture.reshape(1, width * height)
            ok_gestures[i3, :] = flat_ok_gesture[0, :]
            i3 += 1
        # PACE GESTURE
        elif str(file[7]) == "4":
            flat_peace_gesture = image.flatten()
            flat_peace_gesture = flat_peace_gesture.reshape(1, width * height)
            peace_gestures[i4, :] = flat_peace_gesture[0, :]
            i4 += 1
        # ROCK GESTURE
        elif str(file[7]) == "5":
            flat_rock_gesture = image.flatten()
            flat_rock_gesture = flat_rock_gesture.reshape(1, width * height)
            rock_gestures[i5, :] = flat_rock_gesture[0, :]
            i5 += 1
        # INDEX GESTURE
        elif str(file[7]) == "6":
            flat_index_gesture = image.flatten()
            flat_index_gesture = flat_index_gesture.reshape(1, width * height)
            index_gestures[i6, :] = flat_index_gesture[0, :]
            i6 += 1
    else:
        print(f"File {file} is not an image")

# ---------- COMPUTATION OF THE AVERAGE ----------
average_hands = average_hands / count
average_hands = average_hands.astype(np.uint8)
# Show the average of the hands
# cv.imshow("Average Hand", average_hands)
# cv.waitKey(0)
# cv.destroyAllWindows()

# ---------- COMPUTE THE DIFFERENCE FROM EACH IMAGE WITH RESPECT TO THE AVERAGE ----------
N_squared = width * height # Dimension of the images when flattened
A = np.zeros((N_squared, len(list_dir_frames)), dtype=np.uint8)
flat = np.zeros((N_squared, 1), dtype=np.uint8)
difference = np.zeros(dimensions, dtype=np.uint8)
index = 0

for file in list_dir_frames:
    image = cv.imread(str(frames_path) + str(file), cv.IMREAD_GRAYSCALE)
    if type(image) is numpy.ndarray:
        difference = image - average_hands
        flat = difference.flatten()
        flat = flat.reshape((N_squared, 1))
        A[:, index] = flat[:, 0]
        index += 1
    else:
        print(f"File {file} is not an image")

# ---------- COMPUTE THE TRANSPOSE OF THE A MATRIX ----------
covariance_matrix = np.zeros((index, index), dtype=np.uint8)
covariance_matrix = np.matmul(A.transpose(), A)

# cv.imshow("Covariance Matrix", cv.resize(covariance_matrix, (400,400), interpolation=cv.INTER_AREA))
# cv.waitKey(0)
# cv.destroyAllWindows()

# ---------- COMPUTE THE EIGENVALUES & EIGENVECTORS OF THE COVARIANCE MATRIX ----------
eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
# print(np.amax(eigenvalues), np.amin(eigenvalues))

# ---------- SORTING THE PRINCIPAL COMPONENTES ----------
idx = np.argsort(eigenvalues)
sorted_eigenvalues = eigenvalues[idx][::-1]
sorted_eigenvectors = eigenvectors[:, idx]
# print(sorted_eigenvalues)

# ---------- DRAWING THE EIGEN-HANDS ----------
u = np.zeros((index, N_squared), dtype=np.uint8)
A_transpose = np.zeros((index, N_squared), dtype=np.uint8)
A_transpose = A.transpose()

for l in range(index):
    result = np.zeros((1, N_squared), dtype=np.uint8)
    for k in range(np.shape(A)[1]):
        result = result + sorted_eigenvectors[l, k] * A_transpose[k, :]
        u[l, :] = result[:, :]
    print(f"Obtaining Eigen-hand {l}")
print("All Eigen-hands Calculated")

# ---------- PRINT THE 6 MOST REPRESENTATIVE EIGENHANDS ----------
# fig, axis = plt.subplots(ncols=3, nrows=2)
# eigenhand1 = u[0, :]
# eigenhand1 = np.reshape(eigenhand1, (dimensions))
# eigenhand1 = eigenhand1.astype(np.uint8)
#
# eigenhand2 = u[1, :]
# eigenhand2 = np.reshape(eigenhand2, (dimensions))
# eigenhand2 = eigenhand2.astype(np.uint8)
#
# eigenhand3 = u[2, :]
# eigenhand3 = np.reshape(eigenhand3, (dimensions))
# eigenhand3 = eigenhand3.astype(np.uint8)
#
# eigenhand4 = u[3, :]
# eigenhand4 = np.reshape(eigenhand4, (dimensions))
# eigenhand4 = eigenhand4.astype(np.uint8)
#
# eigenhand5 = u[4, :]
# eigenhand5 = np.reshape(eigenhand5, (dimensions))
# eigenhand5 = eigenhand5.astype(np.uint8)
#
# eigenhand6 = u[5, :]
# eigenhand6 = np.reshape(eigenhand6, (dimensions))
# eigenhand6 = eigenhand6.astype(np.uint8)
#
# axis[0, 0].imshow(eigenhand1, cmap="gray")
# axis[0, 1].imshow(eigenhand2, cmap="gray")
# axis[0, 2].imshow(eigenhand3, cmap="gray")
# axis[1, 0].imshow(eigenhand4, cmap="gray")
# axis[1, 1].imshow(eigenhand5, cmap="gray")
# axis[1, 2].imshow(eigenhand6, cmap="gray")
# plt.show()
#
# # ---------- PCA REPRESENTATION ----------
# u_matrix, singular, v_matrix = np.linalg.svd(covariance_matrix)
# normalize_eigenvalues = singular/sum(singular)
# fig, axs = plt.subplots()
# plt.title("PCA")
# plt.ylabel("Acumulated Sum")
# plt.xlabel("Number of principal components")
# axs.plot(np.cumsum(normalize_eigenvalues))
# plt.grid()
# plt.show()

# BY DOING THE PCA ANALYSIS, WE COME UP WITH THE CONCLUSION THAT 45 EIGEN-GESTURES IS ENOUGH TO REPRESENT ABOUT
# 95% OF THE TOTAL IMAGE
reduction = 45
reduced_data_set = np.zeros((reduction, N_squared), dtype=np.uint8)
reduced_data_set = u[0:reduction, :]
# print(np.shape(reduced_data_set)[0])

# ---------- AVERAGE HANDS FLATTEN ----------
average_hands_flatten = average_hands.flatten()
average_hands_flatten = average_hands_flatten.reshape((N_squared, 1))

# ---------- PALM GESTURE WEIGHTS----------
palm_gesture_weights = np.zeros((1, reduction), dtype=np.int32)
for l in range(np.shape(palm_gestures)[0]):
    train_image = palm_gestures[l, :]
    train_image = train_image.reshape(N_squared, 1)
    for k in range(np.shape(reduced_data_set)[0]):
        reduced_data_set_reshape = reduced_data_set[k, :]
        reduced_data_set_reshape = reduced_data_set_reshape.reshape((1, N_squared))
        weight = np.matmul(reduced_data_set_reshape, (train_image - average_hands_flatten))
        palm_gesture_weights[0, k] = palm_gesture_weights[0, k] + weight.item()
palm_gesture_weights = palm_gesture_weights/int(np.shape(palm_gestures)[0])
print(f"Palm Gesture Max Weight:{np.amax(palm_gesture_weights)}, Min weight:{np.amin(palm_gesture_weights)}")

# ---------- C GESTURE WEIGHTS----------
c_gesture_weights = np.zeros((1, reduction), dtype=np.int32)
for l in range(np.shape(c_gestures)[0]):
    train_image = c_gestures[l, :]
    train_image = train_image.reshape(N_squared, 1)
    for k in range(np.shape(reduced_data_set)[0]):
        reduced_data_set_reshape = reduced_data_set[k, :]
        reduced_data_set_reshape = reduced_data_set_reshape.reshape((1, N_squared))
        weight = np.matmul(reduced_data_set_reshape, (train_image - average_hands_flatten))
        c_gesture_weights[0, k] = c_gesture_weights[0, k] + weight.item()
c_gesture_weights = c_gesture_weights/int(np.shape(c_gestures)[0])
print(f"C Gesture Max Weight:{np.amax(c_gesture_weights)}, Min weight:{np.amin(c_gesture_weights)}")

# ---------- FIST GESTURE WEIGHTS----------
fist_gesture_weights = np.zeros((1, reduction), dtype=np.int32)
for l in range(np.shape(fist_gestures)[0]):
    train_image = fist_gestures[l, :]
    train_image = train_image.reshape(N_squared, 1)
    for k in range(np.shape(reduced_data_set)[0]):
        reduced_data_set_reshape = reduced_data_set[k, :]
        reduced_data_set_reshape = reduced_data_set_reshape.reshape((1, N_squared))
        weight = np.matmul(reduced_data_set_reshape, (train_image - average_hands_flatten))
        fist_gesture_weights[0, k] = fist_gesture_weights[0, k] + weight.item()
fist_gesture_weights = fist_gesture_weights/int(np.shape(fist_gestures)[0])
print(f"Fist Gesture Max Weight:{np.amax(fist_gesture_weights)}, Min weight:{np.amin(fist_gesture_weights)}")

# ---------- OK GESTURE WEIGHTS----------
ok_gesture_weights = np.zeros((1, reduction), dtype=np.int32)
for l in range(np.shape(ok_gestures)[0]):
    train_image = ok_gestures[l, :]
    train_image = train_image.reshape(N_squared, 1)
    for k in range(np.shape(reduced_data_set)[0]):
        reduced_data_set_reshape = reduced_data_set[k, :]
        reduced_data_set_reshape = reduced_data_set_reshape.reshape((1, N_squared))
        weight = np.matmul(reduced_data_set_reshape, (train_image - average_hands_flatten))
        ok_gesture_weights[0, k] = ok_gesture_weights[0, k] + weight.item()
ok_gesture_weights = ok_gesture_weights/int(np.shape(ok_gestures)[0])
print(f"Ok Gesture Max Weight:{np.amax(ok_gesture_weights)}, Min weight:{np.amin(ok_gesture_weights)}")

# ---------- PEACE GESTURE WEIGHTS----------
peace_gesture_weights = np.zeros((1, reduction), dtype=np.int32)
for l in range(np.shape(peace_gestures)[0]):
    train_image = peace_gestures[l, :]
    train_image = train_image.reshape(N_squared, 1)
    for k in range(np.shape(reduced_data_set)[0]):
        reduced_data_set_reshape = reduced_data_set[k, :]
        reduced_data_set_reshape = reduced_data_set_reshape.reshape((1, N_squared))
        weight = np.matmul(reduced_data_set_reshape, (train_image - average_hands_flatten))
        peace_gesture_weights[0, k] = peace_gesture_weights[0, k] + weight.item()
peace_gesture_weights = peace_gesture_weights/int(np.shape(peace_gestures)[0])
print(f"Peace Gesture Max Weight:{np.amax(peace_gesture_weights)}, Min weight:{np.amin(peace_gesture_weights)}")

# ---------- ROCK GESTURE WEIGHTS----------
rock_gesture_weights = np.zeros((1, reduction), dtype=np.int32)
for l in range(np.shape(rock_gestures)[0]):
    train_image = rock_gestures[l, :]
    train_image = train_image.reshape(N_squared, 1)
    for k in range(np.shape(reduced_data_set)[0]):
        reduced_data_set_reshape = reduced_data_set[k, :]
        reduced_data_set_reshape = reduced_data_set_reshape.reshape((1, N_squared))
        weight = np.matmul(reduced_data_set_reshape, (train_image - average_hands_flatten))
        rock_gesture_weights[0, k] = rock_gesture_weights[0, k] + weight.item()
rock_gesture_weights = rock_gesture_weights/int(np.shape(rock_gestures)[0])
print(f"Rock Gesture Max Weight:{np.amax(rock_gesture_weights)}, Min weight:{np.amin(rock_gesture_weights)}")

# ---------- INDEX GESTURE WEIGHTS----------
index_gesture_weights = np.zeros((1, reduction), dtype=np.int32)
for l in range(np.shape(index_gestures)[0]):
    train_image = index_gestures[l, :]
    train_image = train_image.reshape(N_squared, 1)
    for k in range(np.shape(reduced_data_set)[0]):
        reduced_data_set_reshape = reduced_data_set[k, :]
        reduced_data_set_reshape = reduced_data_set_reshape.reshape((1, N_squared))
        weight = np.matmul(reduced_data_set_reshape, (train_image - average_hands_flatten))
        index_gesture_weights[0, k] = index_gesture_weights[0, k] + weight.item()
index_gesture_weights = index_gesture_weights/int(np.shape(index_gestures)[0])
print(f"Index Gesture Max Weight:{np.amax(index_gesture_weights)}, Min weight:{np.amin(index_gesture_weights)}")

# ---------- CLASSES FOR EACH GESTURE ----------
classes = np.zeros((7, reduction))

# PALM GESTURE CLASS
classes[0, :] = palm_gesture_weights

# C GESTURE CLASS
classes[1, :] = c_gesture_weights

# FIST GESTURE CLASS
classes[2, :] = fist_gesture_weights

# OK GESTURE CLASS
classes[3, :] = ok_gesture_weights

# PEACE GESTURE CLASS
classes[4, :] = peace_gesture_weights

# ROCK GESTURE CLASS
classes[5, :] = rock_gesture_weights

# INDEX GESTURE CLASS
classes[6, :] = index_gesture_weights

# ---------- CLASSIFYING AN INPUT IMAGE ----------
# LOAD THE TEST IMAGE
# PALM GESTURE (SUCCESS)
# test_image_path = "/Users/lapg/Documents/2do Cuatrimestre/Vision/Proyecto-Hand-Recognition/Own_dataset_hands/Test_images/Gesture0_frame01.png"
# C GESTURE (FAIL)
#test_image_path = "/Users/lapg/Documents/2do Cuatrimestre/Vision/Proyecto-Hand-Recognition/Own_dataset_hands/Test_images/Gesture1_frame05.png"
# FIST GESTURE (SUCCESS)
#test_image_path = "/Users/lapg/Documents/2do Cuatrimestre/Vision/Proyecto-Hand-Recognition/Own_dataset_hands/Test_images/Gesture2_frame03.png"
# OK GESTURE (FAIL)
#test_image_path = "/Users/lapg/Documents/2do Cuatrimestre/Vision/Proyecto-Hand-Recognition/Own_dataset_hands/Test_images/Gesture3_frame04.png"
# PEACE GESTURE (SUCCESS)
#test_image_path = "/Users/lapg/Documents/2do Cuatrimestre/Vision/Proyecto-Hand-Recognition/Own_dataset_hands/Test_images/Gesture4_frame04.png"
# ROCK GESTURE (SUCCESS)
#test_image_path = "/Users/lapg/Documents/2do Cuatrimestre/Vision/Proyecto-Hand-Recognition/Own_dataset_hands/Test_images/Gesture5_frame02.png"
#INDEX GESTURE (FAIL)
test_image_path = "/Users/lapg/Documents/2do Cuatrimestre/Vision/Proyecto-Hand-Recognition/Own_dataset_hands/Test_images/Gesture6_frame04.png"

test_image = cv.imread(test_image_path, cv.IMREAD_GRAYSCALE)
test_image_weights = np.zeros((1, reduction), dtype=np.uint8)
flatten_test_image = test_image.flatten()
flatten_test_image = flatten_test_image.reshape((N_squared, 1))

# COMPUTE THE WEIGHTS OF THE TEST IMAGE
for k in range(np.shape(reduced_data_set)[0]):
    reduced_data_set_reshape = reduced_data_set[k, :]
    reduced_data_set_reshape = reduced_data_set_reshape.reshape((1, N_squared))
    weight = np.matmul(reduced_data_set_reshape, (flatten_test_image - average_hands_flatten))
    test_image_weights[0, k] = test_image_weights[0, k] + weight.item()
print(f"Test Image Max Weight:{np.amax(test_image_weights)}, Min weight:{np.amin(test_image_weights)}")

euclidean_distance = np.zeros((1, 7))

for cl in range(np.shape(classes)[0]):
    euclidean_distance[0, cl] = np.linalg.norm((test_image_weights[0, :] - classes[cl, :]), ord=2)
print(euclidean_distance)

elapsed = time.time() - t
print("Computed time:", elapsed)