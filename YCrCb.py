import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img1 = cv.imread("data/3.jpg")
ycrcb = cv.cvtColor(img1,cv.COLOR_BGR2HSV_FULL)

h = ycrcb[:, :, 0]
#print(np.max(h))
s = ycrcb[:, :, 1]
#print(np.max(s))
v = ycrcb[:, :, 2]
#print(np.max(v))
#cv.imwrite("result/5.jpg",v)


f, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(10,5))
ax1.set_title("H")
ax1.imshow(h)
ax2.set_title("S")
ax2.imshow(s)
ax3.set_title("V")
ax3.imshow(v)
plt.show()