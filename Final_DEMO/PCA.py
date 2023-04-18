"""
Created on Friday Apr 07 00:15:00 2023

@author: Gabriela Hilario Acuapan & Luis Alberto Pineda Gómez
File: PCA.py
Comments: The purpose of this function is to compute the PCA of a given data set.

            ---------- INPUTS ----------
            path: := Path from where to read all the images already pre-processed. [String]
            disp_avg := Display the average hand of the data set [Boolean]
            disp_cov := Display the covariance matrix of the data set [Boolean]
            disp_eig := Display the 6 most representative eigen-hands of the data set [Boolean]
            disp_sum := Display the accumulated sum of the eigenvalues of the data set [Boolean]
            reduction := Value for reducing the dimension of the data set [int]

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
t = time.time()

def compute_pca(path, disp_avg, disp_cov, disp_eig, disp_sum, reduction):

    frames_path = str(path)
    list_dir_frames = os.listdir(frames_path)
    print("Total amount of files in the current directory:", len(list_dir_frames))

    width = 200
    height = 200
    dimensions = (height, width)
    average_hands = np.zeros(dimensions, dtype=np.float64)

    # ---  GESTURES ---
    i0 = 0  # Variable for counting the number of gestures
    palm_gestures = np.zeros((10, width * height), dtype=np.float64)

    # --- C GESTURE ---
    i1 = 0  # Variable for counting the number of C gestures
    c_gestures = np.zeros((10, width * height), dtype=np.float64)

    # --- FIST GESTURE ---
    i2 = 0  # Variable for counting the number of FIST gestures
    fist_gestures = np.zeros((10, width * height), dtype=np.float64)

    # --- OK GESTURE ---
    i3 = 0  # Variable for counting the number of OK gestures
    ok_gestures = np.zeros((10, width * height), dtype=np.float64)

    # --- PEACE GESTURE ---
    i4 = 0  # Variable for counting the number of PEACE gestures
    peace_gestures = np.zeros((10, width * height), dtype=np.float64)

    # --- ILOVEU GESTURE ---
    i5 = 0  # Variable for counting the number of ILOVEU gestures
    rock_gestures = np.zeros((10, width * height), dtype=np.float64)

    # --- L GESTURE ---
    i6 = 0  # Variable for counting the number of L gestures
    index_gestures = np.zeros((10, width * height), dtype=np.float64)

    # Variable for counting the number of images that are being loaded into the system.
    count = 0
    for file in list_dir_frames:
        image = cv.imread(str(frames_path) + str(file), cv.IMREAD_GRAYSCALE)
        # EVALUATE IF THE IMAGE LOADED IS AN IMAGE TYPE
        if type(image) is numpy.ndarray:
            image = image / 255
            # ADDITION OF THE IMAGES
            average_hands = average_hands + image
            count += 1
            # --- CLASSIFICATION OF EACH GESTURE ---
            # PALM GESTURE
            if str(file[7]) == "0":
                flat_palm_gesture = image.flatten()
                flat_palm_gesture = flat_palm_gesture.reshape(1, width * height)
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
            # PEACE GESTURE
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
    average_hands = average_hands.astype(np.float64)

    if disp_avg == True:
        # Show the average of the hands
        cv.imshow("Average Hand", cv.resize(average_hands, (600,600), interpolation=cv.INTER_AREA))
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        pass

    # ---------- COMPUTE THE DIFFERENCE FROM EACH IMAGE WITH RESPECT TO THE AVERAGE ----------
    N_squared = width * height  # Dimension of the images when flattened
    A = np.zeros((N_squared, len(list_dir_frames)), dtype=np.float64)
    flat = np.zeros((N_squared, 1), dtype=np.float64)
    difference = np.zeros(dimensions, dtype=np.float64)
    index = 0

    for file in list_dir_frames:
        image = cv.imread(str(frames_path) + str(file), cv.IMREAD_GRAYSCALE)
        if type(image) is numpy.ndarray:
            image = image / 255
            difference = image - average_hands
            flat = difference.flatten()
            flat = flat.reshape((N_squared, 1))
            A[:, index] = flat[:, 0]
            index += 1
        else:
            print(f"File {file} is not an image")

    # ---------- COMPUTE THE TRANSPOSE OF THE A MATRIX ----------
    covariance_matrix = np.zeros((index, index), dtype=np.float64)
    covariance_matrix = np.dot(A.transpose(), A)

    if disp_cov == True:
        cv.imshow("Covariance Matrix", cv.resize(covariance_matrix, (600,600), interpolation=cv.INTER_AREA))
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        pass

    # ---------- COMPUTE THE EIGENVALUES & EIGENVECTORS OF THE COVARIANCE MATRIX ----------
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # ---------- SORTING THE PRINCIPAL COMPONENTES ----------
    idx = np.argsort(eigenvalues)
    sorted_eigenvalues = eigenvalues[idx][::-1]
    sorted_eigenvectors = eigenvectors[:, idx]

    # ---------- DRAWING THE EIGEN-HANDS ----------
    u = np.zeros((index, N_squared), dtype=np.float64)
    A_transpose = np.zeros((index, N_squared), dtype=np.float64)
    A_transpose = A.transpose()

    for l in range(index):
        result = np.zeros((1, N_squared), dtype=np.float64)
        for k in range(np.shape(A)[1]):
            result = result + sorted_eigenvectors[l, k] * A_transpose[k, :]
            u[l, :] = result[:, :]

    if disp_eig == True:
        # ---------- PRINT THE 6 MOST REPRESENTATIVE EIGENHANDS ----------
        fig, axis = plt.subplots(ncols=3, nrows=2)
        eigenhand1 = u[0, :]
        eigenhand1 = np.reshape(eigenhand1, (dimensions))

        eigenhand2 = u[1, :]
        eigenhand2 = np.reshape(eigenhand2, (dimensions))

        eigenhand3 = u[2, :]
        eigenhand3 = np.reshape(eigenhand3, (dimensions))

        eigenhand4 = u[3, :]
        eigenhand4 = np.reshape(eigenhand4, (dimensions))

        eigenhand5 = u[4, :]
        eigenhand5 = np.reshape(eigenhand5, (dimensions))

        eigenhand6 = u[5, :]
        eigenhand6 = np.reshape(eigenhand6, (dimensions))

        axis[0, 0].imshow(eigenhand1, cmap="gray")
        axis[0, 1].imshow(eigenhand2, cmap="gray")
        axis[0, 2].imshow(eigenhand3, cmap="gray")
        axis[1, 0].imshow(eigenhand4, cmap="gray")
        axis[1, 1].imshow(eigenhand5, cmap="gray")
        axis[1, 2].imshow(eigenhand6, cmap="gray")
        plt.show()
    else:
        pass

    if disp_sum == True:
        # ---------- PCA REPRESENTATION ----------
        u_matrix, singular, v_matrix = np.linalg.svd(covariance_matrix)
        normalize_eigenvalues = singular/sum(singular)
        fig, axs = plt.subplots()
        plt.title("PCA")
        plt.ylabel("Acumulated Sum")
        plt.xlabel("Number of principal components")
        axs.plot(np.cumsum(normalize_eigenvalues))
        plt.grid()
        plt.show()
    else:
        pass

    # BY DOING THE PCA ANALYSIS, WE COME UP WITH THE CONCLUSION THAT 30 EIGEN-GESTURES IS ENOUGH TO REPRESENT ABOUT
    reduced_data_set = np.zeros((reduction, N_squared), dtype=np.float64)
    reduced_data_set = u[0:reduction, :]

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
    palm_gesture_weights = palm_gesture_weights / int(np.shape(palm_gestures)[0])
    #print(f"Palm Gesture Max Weight:{np.amax(palm_gesture_weights)}, Min weight:{np.amin(palm_gesture_weights)}")

    # ---------- C GESTURE WEIGHTS----------
    c_gesture_weights = np.zeros((1, reduction), dtype=np.float64)
    for l in range(np.shape(c_gestures)[0]):
        train_image = c_gestures[l, :]
        train_image = train_image.reshape(N_squared, 1)
        for k in range(np.shape(reduced_data_set)[0]):
            reduced_data_set_reshape = reduced_data_set[k, :]
            reduced_data_set_reshape = reduced_data_set_reshape.reshape((1, N_squared))
            weight = np.matmul(reduced_data_set_reshape, (train_image - average_hands_flatten))
            c_gesture_weights[0, k] = c_gesture_weights[0, k] + weight.item()
    c_gesture_weights = c_gesture_weights / int(np.shape(c_gestures)[0])
    #print(f"C Gesture Max Weight:{np.amax(c_gesture_weights)}, Min weight:{np.amin(c_gesture_weights)}")

    # ---------- FIST GESTURE WEIGHTS----------
    fist_gesture_weights = np.zeros((1, reduction), dtype=np.float64)
    for l in range(np.shape(fist_gestures)[0]):
        train_image = fist_gestures[l, :]
        train_image = train_image.reshape(N_squared, 1)
        for k in range(np.shape(reduced_data_set)[0]):
            reduced_data_set_reshape = reduced_data_set[k, :]
            reduced_data_set_reshape = reduced_data_set_reshape.reshape((1, N_squared))
            weight = np.matmul(reduced_data_set_reshape, (train_image - average_hands_flatten))
            fist_gesture_weights[0, k] = fist_gesture_weights[0, k] + weight.item()
    fist_gesture_weights = fist_gesture_weights / int(np.shape(fist_gestures)[0])
    #print(f"Fist Gesture Max Weight:{np.amax(fist_gesture_weights)}, Min weight:{np.amin(fist_gesture_weights)}")

    # ---------- OK GESTURE WEIGHTS----------
    ok_gesture_weights = np.zeros((1, reduction), dtype=np.float64)
    for l in range(np.shape(ok_gestures)[0]):
        train_image = ok_gestures[l, :]
        train_image = train_image.reshape(N_squared, 1)
        for k in range(np.shape(reduced_data_set)[0]):
            reduced_data_set_reshape = reduced_data_set[k, :]
            reduced_data_set_reshape = reduced_data_set_reshape.reshape((1, N_squared))
            weight = np.matmul(reduced_data_set_reshape, (train_image - average_hands_flatten))
            ok_gesture_weights[0, k] = ok_gesture_weights[0, k] + weight.item()
    ok_gesture_weights = ok_gesture_weights / int(np.shape(ok_gestures)[0])
    #print(f"Ok Gesture Max Weight:{np.amax(ok_gesture_weights)}, Min weight:{np.amin(ok_gesture_weights)}")

    # ---------- PEACE GESTURE WEIGHTS----------
    peace_gesture_weights = np.zeros((1, reduction), dtype=np.float64)
    for l in range(np.shape(peace_gestures)[0]):
        train_image = peace_gestures[l, :]
        train_image = train_image.reshape(N_squared, 1)
        for k in range(np.shape(reduced_data_set)[0]):
            reduced_data_set_reshape = reduced_data_set[k, :]
            reduced_data_set_reshape = reduced_data_set_reshape.reshape((1, N_squared))
            weight = np.matmul(reduced_data_set_reshape, (train_image - average_hands_flatten))
            peace_gesture_weights[0, k] = peace_gesture_weights[0, k] + weight.item()
    peace_gesture_weights = peace_gesture_weights / int(np.shape(peace_gestures)[0])
    #print(f"Peace Gesture Max Weight:{np.amax(peace_gesture_weights)}, Min weight:{np.amin(peace_gesture_weights)}")

    # ---------- ILOVEU GESTURE WEIGHTS----------
    rock_gesture_weights = np.zeros((1, reduction), dtype=np.float64)
    for l in range(np.shape(rock_gestures)[0]):
        train_image = rock_gestures[l, :]
        train_image = train_image.reshape(N_squared, 1)
        for k in range(np.shape(reduced_data_set)[0]):
            reduced_data_set_reshape = reduced_data_set[k, :]
            reduced_data_set_reshape = reduced_data_set_reshape.reshape((1, N_squared))
            weight = np.matmul(reduced_data_set_reshape, (train_image - average_hands_flatten))
            rock_gesture_weights[0, k] = rock_gesture_weights[0, k] + weight.item()
    rock_gesture_weights = rock_gesture_weights / int(np.shape(rock_gestures)[0])
    #print(f"LOVEUGesture Max Weight:{np.amax(rock_gesture_weights)}, Min weight:{np.amin(rock_gesture_weights)}")

    # ---------- L GESTURE WEIGHTS----------
    index_gesture_weights = np.zeros((1, reduction), dtype=np.float64)
    for l in range(np.shape(index_gestures)[0]):
        train_image = index_gestures[l, :]
        train_image = train_image.reshape(N_squared, 1)
        for k in range(np.shape(reduced_data_set)[0]):
            reduced_data_set_reshape = reduced_data_set[k, :]
            reduced_data_set_reshape = reduced_data_set_reshape.reshape((1, N_squared))
            weight = np.matmul(reduced_data_set_reshape, (train_image - average_hands_flatten))
            index_gesture_weights[0, k] = index_gesture_weights[0, k] + weight.item()
    index_gesture_weights = index_gesture_weights / int(np.shape(index_gestures)[0])
    #print(f"L Gesture Max Weight:{np.amax(index_gesture_weights)}, Min weight:{np.amin(index_gesture_weights)}")

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

    # ILOVEU GESTURE CLASS
    classes[5, :] = rock_gesture_weights

    # L GESTURE CLASS
    classes[6, :] = index_gesture_weights

    elapsed = time.time() - t
    print("\n Computed time:", elapsed)

    return classes, reduced_data_set, average_hands_flatten