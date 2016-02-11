from scipy.interpolate import interp1d
import numpy as np
import math
from pylab import *
from scipy import ndimage
import cv2

def main():
	img = cv2.imread('images/flyeronground.png', 0)
	perspective_image = perspectiveTransform(img, 560, 202, 840, 190, 605, 605, 355, 570, 220, 300)

	show_images([img, perspective_image])

def perspectiveTransform(image, x1, y1, x2, y2, x3, y3, x4, y4, M, N):
	#the 4 point correspondences matrix
	matrix = np.array([
				[x1, y1, 1, 0, 0, 0, 0*x1, 0*y1, 0],
				[0, 0, 0, x1, y1, 1, 0*x1, 0*y1, 0],
				[x2, y2, 1, 0, 0, 0, -M*x2, -M*y2, -M],
				[0, 0, 0, x2, y2, 1, 0*x2, 0*y2, 0],
				[x3, y3, 1, 0 ,0, 0, -M*x3, -M*y3, -M],
				[0, 0, 0, x3, y3, 1, -N*x3, -N*y3, -N],
				[x4, y4, 1, 0, 0, 0, 0*x4, 0*y4, 0],
				[0, 0, 0, x4, y4, 1, -N*x4, -N*y4, -N]
				])

	#create the SVD
	U, D, V = svd(matrix)
	
	#use V to create p (last column)
	P = np.array([[V[8][0], V[8][1], V[8][2], V[8][3],
		 V[8][4],V[8][5], V[8][6], V[8][7], V[8][8]]])

	A = P.reshape((3,3))


	warp_image = cv2.warpPerspective(image, A, (M, N))

	return warp_image

# This function displays the given images together with their titles
def show_images(images, cm=plt.cm.gray, axis='off', titles=[]):
    number_images = len(images)
    fig = plt.figure()

    for i, img in enumerate(images):
        fig.add_subplot(1, number_images, i + 1)
        plt.axis(axis)

        if len(titles) != 0:
            plt.suptitle(titles[0])
            plt.title(titles[i+1])

        plt.imshow(img, cmap=cm)
    plt.show()

if __name__ == "__main__":
	main()