from pylab import *

def main():
	x = arange(-100, 101)
	y = arange(-100, 101)

	X, Y = meshgrid(x, y)
	A = 1
	B = 2
	V = 6*pi / 201
	W = 4*pi / 201
	F = A*sin(V*X) + B*cos(W*Y)
	clf()
	imshow(F, cmap=cm.gray)

if __name__ == "__main__":
	main()