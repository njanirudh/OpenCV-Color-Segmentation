import cv2 as cv
import numpy as np

img1 = cv.imread("data/5.jpg")
img1 = cv.resize(img1,(480,640))
print(img1.shape)

bg = cv.imread("/home/anirudh/HBRS/HomeLab/Dataset/FG/2.png")
bg = cv.resize(bg,(480,640))
print(bg.shape)

val = img1 - bg
cv2_subt = cv.bitwise_and(img1,bg)
#cv.imwrite("result.jpg",cv2_subt)
cv.imshow("A",cv2_subt)
cv.waitKey(0)