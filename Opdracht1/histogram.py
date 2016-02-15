#Ajit Jena & Lars Lokhoff
#5730066 & 10606165

from __future__ import division
import numpy as np
import cv2
import matplotlib.pyplot as plt

#Main runs all the questions (except question one and 4) in sequence
def main():
    #This part shows the cameraman original and the cameraman after equalization
    m = 255
    img = cv2.imread('images/cameraman.png', 0)
    histo = histogramEqualization(img, m)
    show_images([img, histo])

    #Show the histograms
    show_histograms(img, histo)

    #This part shows the same object from 4 angles equalized
    img_angle_1 = cv2.imread('images/view1.jpg', 0)
    histo_1 = histogramEqualization(img_angle_1, m)

    img_angle_2 = cv2.imread('images/view2.jpg', 0)
    histo_2 = histogramEqualization(img_angle_2, m)

    img_angle_3 = cv2.imread('images/view3.jpg', 0)
    histo_3 = histogramEqualization(img_angle_3, m)

    img_angle_4 = cv2.imread('images/view4.jpg', 0)
    histo_4 = histogramEqualization(img_angle_4, m)

    show_images([histo_1, histo_2, histo_3, histo_4])


#This function displays the given images together with their titles
def show_images(images, cm=plt.cm.gray, axis='off', titles=[]):
    number_images = len(images)
    fig = plt.figure()

    for i, img in enumerate(images):
        fig.add_subplot(1, number_images, i + 1)
        plt.axis(axis)

        if len(titles) != 0:
            plt.suptitle(titles[0])
            plt.title(titles[i+1])

        plt.imshow(img, cmap=cm)
    plt.show()

#This is the function from the assignment but changed to work of a range from 0 to m.
def histogramEqualization(f, m, bins= 100):
    his, be = np.histogram(f, range=(0,m), bins=bins)
    his = his.astype(float)/sum(his)
    return np.interp(f, be[1:], np.cumsum(his))

#This function shows the 2 histograms of the given pictures to compare in one figure
def show_histograms(img, histo):
    plt.figure(1)

    #For original image
    plt.subplot(211)
    plt.hist(img.flatten(), bins=20, range=[0,255], color='b');

    #The equalized image
    plt.subplot(212)
    plt.hist(histo.flatten(), bins=20, range=[0,1], color='b');

    plt.show()

if __name__ == "__main__":
    main()
