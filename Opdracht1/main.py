from __future__ import division
import numpy as np
import cv2
import matplotlib.pyplot as plt

def main():
	m = 255
	img = cv2.imread('cameraman.png', 0)
	histo = histogramEqualization(img, m)
	
	cv2.imshow(histo, img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def histogramEqualization(f, m, bins= 100):
    his, be = np.histogram(f, range=(0,m), bins=bins)
    his = his.astype(float)/sum(his)
    return np.interp(f, be[1:], np.cumsum(his))

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
