import cv2
import numpy as np

img = cv2.imread("H:/Sem5/Day3/Images/finger_print.jpg",0)
#bin_img1 = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY_INV)[1]
bin_img = cv2.adaptiveThreshold(img, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11,2)

#kernel = np.ones((5,5),np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

cv2.imshow('bin', bin_img)

erode = cv2.erode(bin_img, kernel)
closing = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)

cv2.imshow('erode', erode)
cv2.imshow('closing', closing)

cv2.waitKey(0)
cv2.destroyAllWindows()

