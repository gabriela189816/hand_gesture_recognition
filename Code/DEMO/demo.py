from PCA import compute_pca
from Classification import classify

# ---------- INPUTS ----------
# path: := Path from where to read all the frames already pre-processed. [String]
# disp_avg := Display the average hand of the data set [Boolean]
# disp_cov := Display the covariance matrix of the data set [Boolean]
# disp_eig := Display the 6 most representative eigen-hands of the data set [Boolean]
# disp_sum := Display the accumulated sum of the eigenvalues of the data set [Boolean]
# reduction := Value for reducing the dimension of the data set [int]

# ---------- OUTPUT ----------
# classes := Array containing all the weights of each class of the gestures. [np.array (1x7)]

path = "/Users/lapg/Documents/2do Cuatrimestre/Vision/Proyecto-Hand-Recognition/Own_dataset_hands/Processed_frames/"
disp_avg = False
disp_cov = False
disp_eig = False
disp_sum = False
reduction = 30

# ---------- COMPUTE THE PCA ----------
output, reduced_dataset, average_hands_flatten = compute_pca(path, disp_avg, disp_cov, disp_eig, disp_sum, reduction)

# ---------- CLASSIFY A GIVEN IMAGE ----------
#test_image_path = "/Users/lapg/Documents/2do Cuatrimestre/Vision/Proyecto-Hand-Recognition/Own_dataset_hands/Test_images/Gesture0_frame01.png"
#test_image_path = "/Users/lapg/Documents/2do Cuatrimestre/Vision/Proyecto-Hand-Recognition/Own_dataset_hands/Test_images/Gesture1_frame05.png"
#test_image_path = "/Users/lapg/Documents/2do Cuatrimestre/Vision/Proyecto-Hand-Recognition/Own_dataset_hands/Test_images/Gesture2_frame03.png" # WRONG CLASSIFICATION
#test_image_path = "/Users/lapg/Documents/2do Cuatrimestre/Vision/Proyecto-Hand-Recognition/Own_dataset_hands/Test_images/Gesture3_frame04.png"
#test_image_path = "/Users/lapg/Documents/2do Cuatrimestre/Vision/Proyecto-Hand-Recognition/Own_dataset_hands/Test_images/Gesture4_frame04.png"
#test_image_path = "/Users/lapg/Documents/2do Cuatrimestre/Vision/Proyecto-Hand-Recognition/Own_dataset_hands/Test_images/Gesture5_frame02.png"
#test_image_path = "/Users/lapg/Documents/2do Cuatrimestre/Vision/Proyecto-Hand-Recognition/Own_dataset_hands/Test_images/Gesture6_frame04.png"
classify(output, test_image_path, reduction, reduced_dataset, average_hands_flatten)
