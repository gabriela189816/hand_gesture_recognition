# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 23:04:00 2023

@author: Gabriela Hilario Acuapan
Ejemplo tomado de:
    https://www.analyticsvidhya.com/blog/2021/06/feature-detection-description-and-matching-of-images-using-opencv/
"""

#import numpy as np
import cv2 as cv

# Ejemplo del Algoritmo SIFT aplicado a la imagen de un mango
ori = cv.imread('Hapus_Mango.jpg')
img = cv.imread('Hapus_Mango.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
sift = cv.SIFT_create()
kp, des = sift.detectAndCompute(gray,None)
img=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv.imshow('Original',ori) 
cv.imwrite("Mango_gray.jpg",ori)
cv.imshow('SIFT',img)
cv.imwrite("Mango_sift.jpg",img)


if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()