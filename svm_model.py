import cv2 as cv
import numpy as np
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set up training data
def threshold_slow(image):
    # grab the image dimensions
    image = cv.imread(image)
    image = cv.resize(image, (640, 480))
    image = cv.cvtColor(image,cv.COLOR_BGR2YUV)

    h = image.shape[0]
    w = image.shape[1]

    color_list_b ,color_list_g,color_list_r = [],[],[]
    color_list_all = []
    # loop over the image, pixel by pixel
    for y in range(0, h):
        for x in range(0, w):
            # threshold the pixel
            #image[y, x] = 255 if image[y, x] >= T else 0
            # color_list_b.append(image[y,x][0])
            # color_list_g.append(image[y,x][1])
            # color_list_r.append(image[y,x][2])
            color_list_all.append(image[y,x])


    # return the thresholded image
    #return color_list_b , color_list_g ,color_list_r
    return color_list_all

def svm_model(data,label):
    svm = cv.ml.SVM_create()
    svm.setType(cv.ml.SVM_C_SVC)
    svm.setKernel(cv.ml.SVM_LINEAR)
    svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 500, 1e-6))
    svm.train(data, cv.ml.ROW_SAMPLE, label)
    #svm.save("model.xml")
    print(svm.isTrained())

    #print(svm.predict(np.matrix([0,0,0],dtype=np.float32))[1])
    #print(svm.predict(np.matrix([165,168,176],dtype=np.float32))[1])

    return svm

def plot3d(x,y,z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

if __name__ == "__main__":
    labels = []
    data_set = []
    fg =[]
    # BG
    file_list = glob.glob("/home/anirudh/HBRS/HomeLab/Dataset/dataset_green_segmentation/bg/*.jpg")
    for file in file_list:
        for i in threshold_slow(file):
            data_set.append(np.array(i))
            labels.append(1)


    #FG
    fg_list = glob.glob("/home/anirudh/HBRS/HomeLab/Dataset/FG/*.png")
    for file in fg_list:
        for i in threshold_slow(file):
            data_set.append(np.array(i))
            labels.append(-1)



    print((np.array(data_set).shape))
    #labels = [1 for x in range(0,len(data_set))]
    print((np.array(labels).shape))
    # plot3d(np.array(data_set).T[0],
    #        np.array(data_set).T[1],
    #        np.array(data_set).T[2])
    #
    # plot3d(np.array(fg).T[0],
    #        np.array(fg).T[1],
    #        np.array(fg).T[2])

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # ax.scatter(np.array(data_set).T[0],
    #             np.array(data_set).T[1],
    #             np.array(data_set).T[2],marker = "*")
    # ax.scatter(np.array(fg).T[0],
    #             np.array(fg).T[1],
    #             np.array(fg).T[2])
    #
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    #
    # plt.show()


    #svm_model(np.matrix([[1,1],[2,3]],dtype=np.float32),np.array([1,1]))
    svm = svm_model(np.matrix(data_set,dtype=np.float32),np.array(labels))

    # Prediction
    image = cv.imread("data/4.jpg")
    image = cv.resize(image, (640, 480))

    h = image.shape[0]
    w = image.shape[1]

    # loop over the image, pixel by pixel
    for y in range(0, h):
        for x in range(0, w):
            if svm.predict(np.matrix(image[y,x],dtype=np.float32))[1][0] == 1.:
                image[y, x] = 0

    cv.imwrite("svm.jpg",image)

