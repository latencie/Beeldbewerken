from pylab import *
import cv2

def main():
	x = arange(-100, 101)
	y = arange(-100, 101)

	Y, X = meshgrid(x, y)
	A = 1
	B = 2
	V = 6*pi / 201
	W = 4*pi / 201
	F = A*sin(V*X) + B*cos(W*Y)
	Fx = V*A*cos(V*X)
	Fy = W*B*-sin(W*Y)
	
	show_image(F)
	show_image(Fx)
	show_image(Fy)
	
def show_image(F):
	clf()
	imshow(F, cmap=cm.gray)
	show()

if __name__ == "__main__":
	main()