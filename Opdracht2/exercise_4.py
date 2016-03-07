# Students: Ajit Jena & Lars Lokhoff
# Student id's: 5730066 & 10606165
# 24-02-2016
# This file outputs all the convolved images of each different order
from pylab import imread, plt, cm, show, sqrt, e, arange, pi
from scipy.ndimage import convolve1d


def main():
    s = 2.0
    img = imread('cameraman.png')

    # Create all the images with each a differen order of convolution
    img1 = gD(img, s, 0, 0)
    img2 = gD(img, s, 1, 0)
    img3 = gD(img, s, 0, 1)
    img4 = gD(img, s, 2, 0)
    img5 = gD(img, s, 0, 2)
    img6 = gD(img, s, 1, 1)

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.set_title("Fzero")
    ax1.imshow(img1, cmap=cm.gray)
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.set_title("Fx")
    ax2.imshow(img2, cmap=cm.gray)
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.set_title("Fy")
    ax3.imshow(img3, cmap=cm.gray)
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.set_title("Fxx")
    ax4.imshow(img4, cmap=cm.gray)
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set_title("Fyy")
    ax5.imshow(img5, cmap=cm.gray)
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.set_title("Fxy")
    ax6.imshow(img6, cmap=cm.gray)
    show()


# This function convolves an image according to the given order
def gD(F, s, iorder, jorder):
    # Create the first half of the convolution
    if(iorder == 0):
        A = gauss(g, s)
    elif(iorder == 1):
        A = gauss(g1, s)
    elif(iorder == 2):
        A = gauss(g2, s)

    # Create the second half of the convolution
    if(jorder == 0):
        B = gauss(g, s)
    elif(jorder == 1):
        B = gauss(g1, s)
    elif(jorder == 2):
        B = gauss(g2, s)

    # Convolve in te right order
    Fc = convolve1d(F, A, axis=0, mode='nearest')
    Fc = convolve1d(Fc, B, axis=1, mode='nearest')

    return Fc


# Return the Gaussian formula
def g(x, s=2):
    return 1 / (sqrt(2 * pi) * s) * e**(-(x**2 / (2 * s**2)))


# Return the first order derivative of the Gaussian formula
def g1(x, s=2):
    return g(x, s) * (-x/s**2)


# Return the second order of the derivative of the Gaussian formula
def g2(x, s=2):
    return g1(x, s) * (-x/s**2) + (g1(x, s) / x)


# This function creates the guass kernel
def gauss(funct, s=2):
    #  Determine sample space where the sum of the kernell is more than 0.95
    sample = s * 6 + 1

    if(sample % 2 == 0):
        sample += 1

    half_sample = sample / 2

    x = arange(-half_sample, half_sample + 1)

    return funct(x, s)

if __name__ == "__main__":
    main()
