# Students: Ajit Jena & Lars Lokhoff
# Student id's: 5730066 & 10606165
# 24-02-2016
# This file outputs the convolved image of the cameraman
from pylab import imread, imshow, cm, show, arange, sqrt, pi, e
from scipy.ndimage import convolve1d


def main():
    s = 2.0
    m = gauss_1(s)

    img = imread('cameraman.png')
    img2 = convolve1d(img, m, axis=0, mode='nearest')
    img2 = convolve1d(img2, m, axis=1, mode='nearest')
    imshow(img2, cmap=cm.gray)
    show()


# This function returns the guassian kernal
def gauss_1(s):
    #  Determine sample space where the sum of the kernell is more than 0.95
    sample = s * 6 + 1

    if(sample % 2 == 0):
        sample += 1

    half_sample = sample / 2

    x = arange(-half_sample, half_sample + 1)

    return g(x, s)


# Returns the guassian function
def g(x, s=2):
    return 1 / (sqrt(2 * pi) * s) * e**(-(x**2 / (2 * s**2)))


if __name__ == "__main__":
    main()
