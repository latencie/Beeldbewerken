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

if __name__ == "__main__":
	main()
