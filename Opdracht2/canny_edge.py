from pylab import *
import numpy as np
from scipy.ndimage import convolve;
import cv2
from scipy.ndimage import convolve1d;

def main():

    s = 2.0
    img = imread('images/cameraman.png')
    img2 = canny(img, s)
    imshow(img2, cmap=cm.gray)
    show()
    
def canny(F, s = 2):
    Fx = gD(F, s, 1, 0)
    Fy = gD(F, s, 0, 1)
    Fxx = gD(F, s, 2, 0)
    Fyy = gD(F, s, 0, 2)
    Fxy = gD(F, s, 1, 1)

    width, height = F.shape
    new_img = empty([width, height])

    t = 0.007
    for i in range(0, width - 1):
        for j in range(0, height - 1):
            first_check = sqrt(Fx[i, j] * Fx[i, j] + Fy[i, j] * Fy[i, j])

            if(first_check > t):
                if(zero_crossings(F, s, Fx, Fy, Fxx, Fyy, Fxy, i, j)):
                    new_img[i, j] = first_check

    return new_img

def zero_crossings(F, s, Fx, Fy, Fxx, Fyy, Fxy, x, y):
    for i in range(0, 3):
        if i == 0:
            x_check = 1
            y_check = 0
        elif i == 1:
            x_check = 1
            y_check = 1
        elif i == 2:
            x_check = 0
            y_check = 1
        else:
            x_check = -1
            y_check = 1


        point_1 = (Fx[x + x_check, y + y_check] * 
                   Fx[x + x_check, y + y_check] * 
                   Fxx[x + x_check, y + y_check] + 2.0 * 
                   Fx[x + x_check, y + y_check] * 
                   Fy[x + x_check, y + y_check] * 
                   Fxy[x + x_check, y + y_check] +  
                   Fy[x + x_check, y + y_check] * 
                   Fy[x + x_check, y + y_check] * 
                   Fyy[x + x_check, y + y_check])

        point_2 = (Fx[x - x_check, y - y_check] * 
                   Fx[x - x_check, y - y_check] * 
                   Fxx[x - x_check, y - y_check] + 2.0 * 
                   Fx[x - x_check, y - y_check] * 
                   Fy[x - x_check, y - y_check] * 
                   Fxy[x - x_check, y - y_check] +  
                   Fy[x - x_check, y - y_check] * 
                   Fy[x - x_check, y - y_check] * 
                   Fyy[x - x_check, y - y_check])

        if((point_1 < 0 and point_2 > 0) or (point_1 > 0 and point_2 < 0)):
            return True

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
    return 1 / (sqrt(2.0 * pi) * s) * e**(-(x**2.0 / (2.0* s**2)))

def g1(x, s=2):
    return -x / (sqrt(2.0 * pi) * s**3) * e**(-(x**2.0 / (2.0* s**2)))

def g2(x, s=2):
    return  (x**2 / (sqrt(2.0 * pi) * s**5) * e**(-(x**2.0 / (2.0* s**3)))) - (1 / (sqrt(2.0 * pi) * s**3) * e**(-(x**2 / (2.0* s**2))))

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
