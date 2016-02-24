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
<<<<<<< HEAD
    
=======
    imshow(img, cmap=cm.gray)
    show()

>>>>>>> c7372d5fdcf4bebab7ab86bd3c49483e41c773b6
def canny(F, s):
    Fx = gD(F, s, 1, 0)
    Fy = gD(F, s, 0, 1)
    Fxx = gD(F, s, 2, 0)
    Fyy = gD(F, s, 0, 2)
    Fxy = gD(F, s, 1, 1)


    width, height = F.shape
    new_img = zeros((width, height))
    # Fw = np.sqrt(Fx ** 2 + Fy ** 2) 

<<<<<<< HEAD
    t = 0.007
    for i in range(0, height - 1):
        for j in range(0, width - 1):
=======
    t = 0.02
    for i in range(0, width - 1):
        for j in range(0, height - 1):
>>>>>>> c7372d5fdcf4bebab7ab86bd3c49483e41c773b6
            first_check = sqrt(Fx[i, j] * Fx[i, j] + Fy[i, j] * Fy[i, j])

            if(first_check > t):
                if(zero_crossings(F, s, Fx, Fy, Fxx, Fyy, Fxy, i, j)):
                    new_img[i, j] = first_check

    return new_img

def zero_crossings(F, s, Fx, Fy, Fxx, Fyy, Fxy, x, y):
    for i in range(0, 4):
        if i == 0:
            x_check = 1
            y_check = 1
        elif i == 1:
            x_check = 1
            y_check = 0
        elif i == 2:
            x_check = 0
            y_check = 1
        elif i == 3:
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

        if(point_1 < 0 and point_2 > 0 or point_1 > 0 and point_2 < 0):
            return True

def gD(F, s, iorder, jorder):
    # Create empty kernel of size 5*s by 5*s and fill it
    A = empty([5*s])
    B = empty([5*s])
    for i in range(5 * int(s)):
            x = i - (5.0*s)/2.0

            # Looks what the iorder and picks the right function.
            if(iorder == 0):
                A[i] = g(x, s)

            elif(iorder == 1):
                A[i] = g1(x, s)

            elif(iorder == 2):
                A[i] = g2(x, s)

            # Looks what the jorder and picks the right function.
            if(jorder == 0):
                B[i] = g(x, s)

            elif(jorder == 1):
                B[i] = g1(x, s)

            elif(jorder == 2):
                B[i] = g2(x, s)

    H = convolve1d(F, A, 0, mode='nearest')
    H = convolve1d(H, B, 1, mode='nearest')
    return H



def g(x, s):
<<<<<<< HEAD
    return 1.0 / (sqrt(2.0 * pi) * s) * exp(-((x**2) / (2.0* s**2)))
=======
    return 1 / (sqrt(2.0 * pi) * s) * e**(-(x**2.0 / (2.0* s**2)))
>>>>>>> c7372d5fdcf4bebab7ab86bd3c49483e41c773b6

def g1(x, s):
    return ((-(x) / (s**3 * sqrt(2.0*pi)))
                              * exp(-(x**2 / (2.0 * s**2))))

def g2(x, s):
    return  ((-(s**2 - x**2) / s**5 * sqrt(2.0 * pi))
                              * exp(-(x**2 / (2.0 * s**2))))

def gauss(funct, s):
    # Determine sample space where the sum of the kernell is more than 0.95
    sample = s * 6 + 1

    if(sample % 2 == 0): 
        sample +=1

    half_sample = sample/2.0

    x = arange(-half_sample, half_sample + 1)

    return funct(x, s)

if __name__ == "__main__":
    main()
