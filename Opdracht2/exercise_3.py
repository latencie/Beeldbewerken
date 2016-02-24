from pylab import *
import numpy as np
from scipy.ndimage import convolve1d;
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

def main():
    s = 2.0
    m = gauss_1(s)

    img = imread('images/cameraman.png')
    img2 = convolve1d(img, m, axis=0, mode='nearest')
    img2 = convolve1d(img2, m, axis=1, mode='nearest')
    imshow(img2, cmap=cm.gray)
    show()

def gauss_1(s):
    # Determine sample space where the sum of the kernell is more than 0.95
    sample = s * 6 + 1

    if(sample % 2 == 0): 
        sample +=1

    half_sample = sample/2

    x = arange(-half_sample, half_sample + 1)

    return g(x, s)

def g(x, s=2):
    return 1 / (sqrt(2 * pi) * s) * e**(-(x**2 / (2* s**2)))


if __name__ == "__main__":
    main()