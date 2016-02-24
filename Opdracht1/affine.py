# Subject: Beeld Bewerken

# Student name .... Ajit Jena, Daan van Ingen
# Student email ... <ajit.jena@gmail.com>, <lars.lokhoff@hotmail.com>.
# Collegekaart .... 5730066, 10606165
# Date ............ 10-02-2016
# Comments ........ This file contains the function to perform an affine
#                    transformation
# affine.py

import numpy as np
import cv2
import matplotlib.pyplot as plt


# Main function that calls the functions to warp images and print them
def main():

    img = cv2.imread('cameraman.png', 0)
    warp_img = affineTransform(img, 0, 100, 150, 0, 250, 100, 255, 255)
    show_images([img, warp_img])


# Function responsible for the affine transformation of an image. Output
# Image is returned
def affineTransform(image, x1, y1, x2, y2, x3, y3, X, Y):

    M = np.array([
        [x1, y1, 1, 0, 0, 0],
        [0, 0, 0, x1, y1, 1],
        [x2, y2, 1, 0, 0, 0],
        [0, 0, 0, x2, y2, 1],
        [x3, y3, 1, 0, 0, 0],
        [0, 0, 0, x3, y3, 1]
        ])

    # vector q to which the points are mapped to
    q = np.array([0, Y-1, 0, 0, X - 1, 0])
    # Obtain vector p with least squares
    p = np.linalg.lstsq(M, q)[0]
    # Reshape vector p to the affine transformation matrix
    A = p.reshape((2, 3))
    # flags = 4: Lanczos interpolation over 8x8 neighborhood
    warp_image = cv2.warpAffine(image, A, (X-1, Y-1), flags=4)
    return warp_image


# This function displays the given images together with their titles
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


if __name__ == "__main__":
    main()
