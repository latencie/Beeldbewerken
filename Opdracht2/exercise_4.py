from pylab import *
import numpy as np
from scipy.ndimage import convolve;
import cv2
from scipy.ndimage import convolve1d;

def main():

    s = 2.0
    img = imread('images/cameraman.png')
    img2 = gD(img, s, 1, 1)
    imshow(img2, cmap=cm.gray)
    show()


def gD(F, s, iorder, jorder):
    if(iorder == 0):
        A = gauss(g, s)
    elif(iorder == 1):
        A = gauss(g1, s)
    elif(iorder == 2):
        A = gauss(g2, s)

    if(jorder == 0):
        B = gauss(g, s)
    elif(jorder == 1):
        B = gauss(g1, s)
    elif(jorder == 2):
        B = gauss(g2, s)

    Fc = convolve1d(F, A, axis=0, mode='nearest')
    Fc = convolve1d(Fc, B, axis=1, mode='nearest')

    return Fc



def g(x, s=2):
    return 1 / (sqrt(2 * pi) * s) * e**(-(x**2 / (2* s**2)))

def g1(x, s=2):
    return g(x, s) * (-x/s**2)

def g2(x, s=2):
    return  g1(x, s) * (-x/s**2) + (g1(x, s) / x)



def gauss(funct, s=2):
    # Determine sample space where the sum of the kernell is more than 0.95
    sample = s * 6 + 1

    if(sample % 2 == 0): 
        sample +=1

    half_sample = sample/2

    x = arange(-half_sample, half_sample + 1)

    return funct(x, s)

if __name__ == "__main__":
    main()
