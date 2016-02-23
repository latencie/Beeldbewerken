from pylab import *
import numpy as np
from scipy.ndimage import convolve;
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D


# This is the main function that prints a guassian blurred image and a 3d plot 
# 
def main():

	s = 10.0
	m = gauss(s)
	img = imread('images/cameraman.png')

	# figure()
	# add_subplot()
	img2 = convolve(img, m, mode='nearest')
	# imshow(img2, cmap=cm.gray)
	# show()
	
	xs = ys = linspace(-3*s+0.5, 3*s+0.5)
	Xs, Ys = meshgrid(xs,ys)
	Zs = g(Xs,Ys,s)

	fig = plt.figure()
	ax1 = fig.add_subplot(2,4,(1,2))
	ax1.imshow(img, cmap=cm.gray)
	ax2 = fig.add_subplot(2,4,(3,4))
	ax2.imshow(img2, cmap=cm.gray)
	ax3 = fig.add_subplot(2,4,(6,7), projection='3d')
	ax3.plot_wireframe(Xs, Ys, Zs, rstride=3, cstride=3)
	ax3.set_xlabel('$x$')
	ax3.set_ylabel('$y$')
	ax3.set_zlabel('$z$')
	plt.show()
	

# This is the 2dimensional guassian function
def g(x, y, s=2):
	A = 1/(2*pi*s*s)
	B = e**(-(x*x + y*y)/float(2*s*s))
	return A * B


# Given an s value the Gaussian kernel is returned
def gauss(s=2):
	
	# Determine sample space where the sum of the kernell is more than 0.95
	sample = s * 6 + 1
	# If sample space is even
	if(sample % 2 == 0): sample +=1
	half_sample = sample/2
	# Create meshgrid of the sample size in X and Y
	x = y = arange(-half_sample, half_sample + 1)
	X, Y = meshgrid(x,y)
	# Returns Gaussian kernel 
	print sum(g(X,Y, s))
	return g(X,Y, s)



if __name__ == "__main__":
    main()