# -*- coding: utf-8 -*-
"""
Pick matching points in two images
"""
from pylab import *
import cv2
import random
from ginput import ginput


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

    # xy = zeros((4,2))
    # for i in range(n):
    #    print 'Click at point %s in image 1' % (i+1)
    #    x, y = ginput('Image 1', 1)
    #    marker(dispim1, x, y, 3, str(i+1))
    #    cv2.imshow('Image 1', dispim1)
    #    xy[i,0]=x
    #    xy[i,1]=y
       
    # xaya = zeros((4,2))
    # for i in range(n):
    #    print 'Click at point %s in image 2' % (i+1)
    #    x, y = ginput('Image 2', 1)
    #    marker(dispim2, x, y, 3, str(i+1))
    #    cv2.imshow('Image 2', dispim2)
    #    xaya[i,0]=x
       # xaya[i,1]=y

    # the are points i have clicked that lead to reasonable result    
    xy = array([[ 157, 32],
                [ 211, 37],
                [ 222,107],
                [ 147,124]])
    xaya = array([[  6, 38],
                  [ 56, 31],
                  [ 82, 87],
                  [ 22,118]])
    
    return xy, xaya
    

def perspectiveTransform(xy, xaya):
    #create the matrix
    matrix = zeros(shape = (2*len(xy), 9))

    #change the appropriate lines in the matrix according to xy length
    for i in range(len(xy)):
        matrix[i*2 -1] = [xy[i][0], xy[i][1], 1, 0, 0, 0, -xaya[i][0]*xy[i][0],
        -xaya[i][0]*xy[i][1], -xaya[i][0]]

        matrix[i*2] = [0, 0, 0, xy[i][0], xy[i][1], 1, -xaya[i][1]*xy[i][0],
         -xaya[i][1]*xy[i][1], -xaya[i][1]]
            

    #create the SVD
    U, D, V = svd(matrix)
    
    #create P by using V and reshaping
    P = V[-1].reshape(3,3)

    return P

def distance(match):
    return match.distance


def sift(im1, im2):

    # Convert images to gray scale
    g_im1 = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    g_im2 = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)

    # Initialize Sift
    sift = cv2.xfeatures2d.SIFT_create()

    # Get keypoints and descriptors from images
    kp1, des1 = sift.detectAndCompute(g_im1, None)
    kp2, des2 = sift.detectAndCompute(g_im2, None)

    # Brute-Force matching, creates a sorted array of closest matches to
    # furthest matches of all descriptors in both images. 
    best_fit = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    sort_match = sorted(best_fit.match(des1, des2), key=distance)

    data = array([])
    # Pick the 40 best matching data points from both images
    for i in range(0,40):
        # Concatenate all corresponding pair points one after the other
        data = concatenate((data, [
            kp1[sort_match[i].queryIdx].pt[0],
            kp1[sort_match[i].queryIdx].pt[1],
            kp2[sort_match[i].trainIdx].pt[0],
            kp2[sort_match[i].trainIdx].pt[1]
            ]))

    # Reshape array so that there are 40 pairs of corresponding closest 
    # matching points.
    data = data.reshape(40,4)
    return data


def ransac(data, n, k, t, d):

    best_fit = None
    # Start with a large error value
    best_error = 1000
    this_error = .0
    random.seed()

    # No more itterations than k
    for i in range(0, k):

        maybe_inliers = array([]).reshape(0, 4)
        also_inliers = array([]).reshape(0, 2)
        # Temporary data points ()
        temp_data = copy(data)

        # Place n random points in maybe_inliers
        for _ in range(0, n):

            rand_num = random.randint(0, len(temp_data) - 1)
            maybe_inliers = concatenate((maybe_inliers, [temp_data[rand_num]]))      
            # Deletion occurs with flattened array, therefore multiply 
            # by 4 and delete four consecutive values 
            # Points are deleted for tests that occur in next forloop.
            temp_data = delete(temp_data, [
                rand_num*4, rand_num*4 + 1, rand_num*4 + 2, rand_num*4 + 3
                ])
            # after flatenning reshape
            temp_data = temp_data.reshape(len(temp_data)/4, 4)
        # Create potential best fit model
        maybe_model = perspectiveTransform(
            maybe_inliers[:, 0:2], maybe_inliers[:, 2:4]
            )

        # Test model with every point in data and not in maybe_inliers
        # Calculate errors
        for j in range(0, len(temp_data)):

            point = dot(maybe_model, [temp_data[j][0], temp_data[j][1], 1])
            point = point / point[2]

            curr_error = (point[0] - temp_data[j][2])**2 
            curr_error += (point[1] - temp_data[j][3])**2
            curr_error = sqrt(curr_error)

            # If error of point is smaller than threshold add to also_inliers
            if(curr_error < t):
                also_inliers = concatenate((also_inliers, [temp_data[j][0:2]]))
                this_error += curr_error

        # Make sure there are more than d points in also_inliers
        if(len(also_inliers) > d):
            # If current model fits beter than previous best model make 
            # current model best model. 
            if(this_error < best_error):
                best_error = this_error
                best_fit = maybe_model

    return best_fit


if __name__=="__main__":
    
    im1 = cv2.imread('images/nachtwacht1.jpg')
    im2 = cv2.imread('images/nachtwacht2.jpg')
    xy, xaya = pickMatchingPoints(im1, im2, 4)

    data = sift(im1,im2)

    P = ransac(data, 10, 1000, 1, 4)
    
    # P = perspectiveTransform(xy, xaya)
    
    # I warp from image2 to image1 because image 1 is nicely upright
    # The sizez of images are set by trial and error...
    # The final code for the exercise should not contain these magic numbers
    # the size of the final image should be calculated to exactly contain
    # both (warped) images.
    # In the warped version of image2 i simply overwrite with data
    # from image 1.
    tim = cv2.warpPerspective(im2, linalg.inv(P), (450,400))
    M,N = im1.shape[:2]
    tim[0:M,0:N,:]=im1
    cv2.waitKey(1)
    cv2.imshow('result',tim)

    
    cv2.waitKey()