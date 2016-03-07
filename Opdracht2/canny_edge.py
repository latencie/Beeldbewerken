# Students: Ajit Jena & Lars Lokhoff
# Student id's: 5730066 & 10606165
# 24-02-2016
# This file outputs the edges of the cameraman
# image using the canny edge detection method

from pylab import imread, arange, sqrt, pi, exp, zeros, imshow, cm, show
from scipy.ndimage import convolve1d


# The main function does all the function calling
def main():
    s = 2.0
    img = imread('cameraman.png')
    img2 = canny(img, s)
    imshow(img2, cmap=cm.gray)
    show()


# This function returns an image with only the edges of the input image shown
def canny(F, s):
    # Calculate the convolutions of the image
    Fx = gD(F, s, 1, 0)
    Fy = gD(F, s, 0, 1)
    Fxx = gD(F, s, 2, 0)
    Fyy = gD(F, s, 0, 2)
    Fxy = gD(F, s, 1, 1)

    # Create a new image to fill with only edges
    width, height = F.shape
    new_img = zeros((width, height))

    # Threshold and the loop over the image and do the checks for an edge
    t = 0.02
    for i in range(0, width - 1):
        for j in range(0, height - 1):

            # The first check fw, has to be bigger than our threshold
            first_check = sqrt(Fx[i, j] * Fx[i, j] + Fy[i, j] * Fy[i, j])
            if(first_check > t):

                # The second check uses the zero
                # crossings method to check for an edge
                if(zero_crossings(F, s, Fx, Fy, Fxx, Fyy, Fxy, i, j)):
                    new_img[i, j] = first_check

    return new_img


# This function does the zero crossing
# check of a point and its 3x3 neighbourhood
def zero_crossings(F, s, Fx, Fy, Fxx, Fyy, Fxy, x, y):

    # First we check which 2 pixels we have to compare
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

    # Calculate the fww of the first pixels
    point_1 = (Fx[x + x_check, y + y_check] *
               Fx[x + x_check, y + y_check] *
               Fxx[x + x_check, y + y_check] + 2.0 *
               Fx[x + x_check, y + y_check] *
               Fy[x + x_check, y + y_check] *
               Fxy[x + x_check, y + y_check] +
               Fy[x + x_check, y + y_check] *
               Fy[x + x_check, y + y_check] *
               Fyy[x + x_check, y + y_check])

    # Calculate the fww of the second pixel
    point_2 = (Fx[x - x_check, y - y_check] *
               Fx[x - x_check, y - y_check] *
               Fxx[x - x_check, y - y_check] + 2.0 *
               Fx[x - x_check, y - y_check] *
               Fy[x - x_check, y - y_check] *
               Fxy[x - x_check, y - y_check] +
               Fy[x - x_check, y - y_check] *
               Fy[x - x_check, y - y_check] *
               Fyy[x - x_check, y - y_check])

    # Check if one pixel is bigger than 0 and the othe smaller than 0
    if(point_1 < 0 and point_2 > 0 or point_1 > 0 and point_2 < 0):
            return True


# This function returns the gaussian kernel of the appropriate order
def gD(F, s, iorder, jorder):

    # Create the first half of the right order
    if(iorder == 0):
        A = gauss(g, s)
    elif(iorder == 1):
        A = gauss(g1, s)
    elif(iorder == 2):
        A = gauss(g2, s)

    # Create the second half of the right order
    if(jorder == 0):
        B = gauss(g, s)
    elif(jorder == 1):
        B = gauss(g1, s)
    elif(jorder == 2):
        B = gauss(g2, s)

    # Now we convolve in the right order to create the image
    Fc = convolve1d(F, A, axis=0, mode='nearest')
    Fc = convolve1d(Fc, B, axis=1, mode='nearest')

    return Fc


# Returns the Gaussian formula
def g(x, s):
    return 1.0 / (sqrt(2.0 * pi) * s) * exp(-((x**2) / (2.0 * s**2)))


# Returns the first derivative of the gaussian formula
def g1(x, s):
    return ((-(x) / (s**3 * sqrt(2.0*pi))) * exp(-(x**2 / (2.0 * s**2))))


# Returns the second derivative of the Gaussian formula
def g2(x, s):
    return ((-(s**2 - x**2) / s**5 * sqrt(2.0 * pi)) *
            exp(-(x**2 / (2.0 * s**2))))


# This function returns the kernel
def gauss(funct, s):
    #  Determine sample space where the sum of the kernell is more than 0.95
    sample = s * 6 + 1

    if(sample % 2 == 0):
        sample += 1

    half_sample = sample / 2.0

    x = arange(-half_sample, half_sample + 1)

    return funct(x, s)


if __name__ == "__main__":
    main()
