import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Open an image using Opencv
img1 = cv.imread("data/3.jpg")

# Convert the image into different colorspaces
cvt_image = cv.cvtColor(img1,cv.COLOR_BGR2HSV_FULL)
#cvt_image = cv.cvtColor(img1,cv.COLOR_BGR2LAB)
#cvt_image = cv.cvtColor(img1,cv.COLOR_BGR2LUV)

# Save each channel of the image seperately
c1 = cvt_image[:, :, 0]
c2 = cvt_image[:, :, 1]
c3 = cvt_image[:, :, 2]

# Create a new window to show all the three channels seperately
f, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(10,5))
ax1.set_title("Channel 1")
ax1.imshow(c1)
ax2.set_title("Channel 2")
ax2.imshow(c2)
ax3.set_title("Channel 3")
ax3.imshow(c3)
plt.show()