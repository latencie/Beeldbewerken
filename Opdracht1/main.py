from __future__ import division
import numpy as np
import cv2
import matplotlib.pyplot as plt

def main():
	m = 255
	img = cv2.imread('cameraman.png', 0)
	histo = histogramEqualization(img, m)
	show_image('Original', img, 'Equalized', histo)


def show_image(name, image, name2, image2):
    cv2.imshow(name, image)
    cv2.imshow(name2, image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def histogramEqualization(f, m, bins= 100):
    his, be = np.histogram(f, range=(0,m), bins=bins)
    his = his.astype(float)/sum(his)
    return np.interp(f, be[1:], np.cumsum(his))

if __name__ == "__main__":
	main()
