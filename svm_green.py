import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def threshold_slow(image):
    # grab the image dimensions
    image = cv2.resize(image, (640, 480))

    h = image.shape[0]
    w = image.shape[1]

    color_list_b ,color_list_g,color_list_r = [],[],[]
    # loop over the image, pixel by pixel
    for y in range(0, h):
        for x in range(0, w):
            # threshold the pixel
            #image[y, x] = 255 if image[y, x] >= T else 0
            color_list_b.append(image[y,x][0])
            color_list_g.append(image[y,x][1])
            color_list_r.append(image[y,x][2])
            print(image[y,x])


    # return the thresholded image
    return color_list_b , color_list_g ,color_list_r

def plot3d(x,y,z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, 0)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


if __name__ == "__main__":
    img = cv2.imread("data/1.jpg")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    cols_b, cols_g, cols_r = threshold_slow(hsv)
    plot3d(cols_b, cols_g, cols_r)