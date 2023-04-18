"""
Created on Friday Apr 14 00:15:00 2023

@author: Gabriela Hilario Acuapan & Luis Alberto Pineda Gómez
File: demo.py
Comments: This is the main program, you need to run the sensorIR.py code first.

        ---------- INPUTS ----------
        path: := Path from where to read all the frames already pre-processed. [String]
        disp_avg := Display the average hand of the data set [Boolean]
        disp_cov := Display the covariance matrix of the data set [Boolean]
        disp_eig := Display the 6 most representative eigen-hands of the data set [Boolean]
        disp_sum := Display the accumulated sum of the eigenvalues of the data set [Boolean]
        reduction := Value for reducing the dimension of the data set [int]

        ---------- OUTPUT ----------
        classes := Array containing all the weights of each class of the gestures. [np.array (1x7)]
"""

# --- IMPORT LIBRARIES ---
from PCA import compute_pca
from Classification import classify

path = "C:/Users/gabri/OneDrive/Documentos/GitHub/Hand_gesture_recognition/Trainingset_processed/"
disp_avg = False
disp_cov = False
disp_eig = False
disp_sum = False
reduction = 30

# ---------- COMPUTE THE PCA (Phase I) ----------
output, reduced_dataset, average_hands_flatten = compute_pca(path, disp_avg, disp_cov, disp_eig, disp_sum, reduction)

# ---------- CLASSIFY A GIVEN IMAGE (Phase II) ----------
test_image_path = "C:/Users/gabri/OneDrive/Documentos/GitHub/Hand_gesture_recognition/Final_DEMO/Resized_image.png"
classify(output, test_image_path, reduction, reduced_dataset, average_hands_flatten)

