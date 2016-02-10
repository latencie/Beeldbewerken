from __future__ import division
import numpy as np
import cv2
import matplotlib.pyplot as plt

def main():
	m = 255
	img = cv2.imread('cameraman.png', 0)
	histo = histogramEqualization(img, m)
	draw_histogram(histo, m)

def draw_histogram(data, m):
	n, bins, patches = plt.hist(data, 10)
	plt.show()

def histogramEqualization(f, m, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ,1.0]):
    his, be = np.histogram(f, range=(0,m), bins=bins)
    his = his.astype(float)/sum(his)
    return np.interp(f, be[1:], np.cumsum(his))

if __name__ == "__main__":
	main()
