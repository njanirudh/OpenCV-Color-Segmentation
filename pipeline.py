import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow

        buf = cv.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

def process(input):
    YUV = cv.cvtColor(input,cv.COLOR_BGR2YUV)
    V = YUV[:, :, 2]

    #eql = cv.equalizeHist(V)

    clahe = cv.createCLAHE(clipLimit=1.0, tileGridSize=(5, 5))
    eql = clahe.apply(V)

    eql = cv.blur(eql,(5,5))

    out = apply_brightness_contrast(eql,0,60)
    #out = cv.inRange(out, 0, 50)
    #ret , out = cv.threshold(out,0,80,cv.THRESH_BINARY_INV)
    out = cv.bitwise_and(input, input, mask=out)
    #
    #plt.hist(out.ravel(), bins=256, range=(0.0, 255))  # calculating histogram
    # hist = cv.calcHist(eql,[0],mask=None,histSize=[256],ranges=[0,256])
    # plt.plot(hist)
    # plt.show()
    #
    # out[:,:,1]=0
    #cv.imwrite("result2.jpg",out)
    # cv.imwrite("result1.jpg",img)
    # cv.imwrite("result.jpg",eqn)
    cv.imshow("AAA",out)
    cv.waitKey(0)



if __name__ == "__main__":

    img = cv.imread("data/5.jpg")
    img = cv.resize(img,(640,480))
    process(img)

