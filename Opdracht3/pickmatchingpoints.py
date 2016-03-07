# -*- coding: utf-8 -*-
"""
Pick matching points in two images
"""
from pylab import *
import cv2
from ginput import ginput
import drawMatches as dm
import matplotlib.pyplot as plt
import random

def marker(im, x, y, r,s):
    cv2.circle(im, (x,y), r, (0,255,255))
    cv2.line(im, (x,y-r), (x,y+r), (0,255,255))
    cv2.line(im, (x-r,y), (x+r,y), (0,255,255))
    cv2.putText(im,s,(x,y+r),cv2.FONT_HERSHEY_PLAIN,1.0,(0,255,255))
    

def pickMatchingPoints(im1, im2, n):
    dispim1 = im1.copy()
    dispim2 = im2.copy()
    
    cv2.imshow('Image 1',dispim1)
    cv2.imshow('Image 2',dispim2)

    xy = zeros((4,2))
    for i in range(n):
        print 'Click at point %s in image 1' % (i+1)
        x, y = ginput('Image 1', 1)
        marker(dispim1, x, y, 3, str(i+1))
        cv2.imshow('Image 1', dispim1)
        xy[i,0]=x
        xy[i,1]=y
       
    xaya = zeros((4,2))
    for i in range(n):
        print 'Click at point %s in image 2' % (i+1)
        x, y = ginput('Image 2', 1)
        marker(dispim2, x, y, 3, str(i+1))
        cv2.imshow('Image 2', dispim2)
        xaya[i,0]=x
        xaya[i,1]=y
    
    return xy, xaya

def perspectiveTransform(xy, xaya):
    matrix = zeros(shape=(2*len(xy), 9))
    
    for i in range(len(xy)):
        matrix[i*2 -1] = [xy[i][0], xy[i][1], 1, 0, 0, 0, -xaya[i][0]*xy[i][0], -xaya[i][0]*xy[i][1], -xaya[i][0]]
        matrix[i*2] = [0, 0, 0, xy[i][0], xy[i][1], 1, -xaya[i][1]*xy[i][0], -xaya[i][1]*xy[i][1], -xaya[i][1]]

    U, D, V = svd(matrix)
    P = V[-1].reshape(3,3)

    return P

def sift(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # BF matching
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)

    # Gives back the 40 closest matching datapoint from image1 and image2
    data = array([]).reshape(0, 4)
    for i in range(40):
        data = concatenate((data, [[kp1[matches[i].queryIdx].pt[0], kp1[matches[i].queryIdx].pt[1],
                                   kp2[matches[i].trainIdx].pt[0], kp2[matches[i].trainIdx].pt[1]]]))

    return data

def RANSAC(data, n, k , t, d):
    bestfit = array([])
    besterr = 1000000
    thiserr = 0.0
    random.seed()

    # For k iterations.
    for i in range(k):
        maybeinliers = zeros(shape=(n, 4))
        alsoinLiers = array([]).reshape(0,2)

        # Picks n random points. 
        for j in range(n):
            index = random.randrange(len(data))
            maybeinliers[j] = data[index]

        # Creates model for the random points.
        maybemodel = perspectiveTransform(maybeinliers[:, 0:2], maybeinliers[:, 2:4])

        # Checks for all points what their error is.
        for l in range(len(data)):
            point = dot(maybemodel, [data[l][0], data[l][1], 1])
            point = point / point[2]
            curerr = sqrt((point[0] - data[l][2])**2 + (point[1] - data[l][3])**2)

            # If points are close it is in the line.
            if curerr < t:
                alsoinLiers= concatenate((alsoinLiers, [data[l][0:2]]))
                thiserr += curerr

        #  If there are more than d points in the line.
        if len(alsoinLiers) > d:
            # Checks if this is the model with the smallest error.
            if besterr > thiserr:
                bestfit = maybemodel
                besterr = thiserr
    return bestfit

if __name__=="__main__":
    im1 = cv2.imread('../images/nachtwacht1.jpg')
    im2 = cv2.imread('../images/nachtwacht2.jpg')

    # xy, xaya = pickMatchingPoints(im1, im2, 4)

    data = sift(im1, im2)

    P = RANSAC(data, 10, 1000, 1, 1)
    
    # I warp from image2 to image1 because image 1 is nicely upright
    # The sizez of images are set by trial and error...
    # The final code for the exercise should not contain these magic numbers
    # the size of the final image should be calculated to exactly contain
    # both (warped) images.
    # In the warped version of image2 i simply overwrite with data
    # from image 1.
    tim = cv2.warpPerspective(im2, linalg.inv(P), (450,300))
    M,N = im1.shape[:2]
    tim[0:M,0:N,:]=im1
    cv2.waitKey(1)
    cv2.imshow('result',tim)

    cv2.waitKey()