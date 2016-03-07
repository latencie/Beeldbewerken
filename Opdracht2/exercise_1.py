# Students: Ajit Jena & Lars Lokhoff
# Student id's: 5730066 & 10606165
# 24-02-2016
# This file outputs the visualised derivatives
# of the function from the assignment
from pylab import arange, meshgrid, pi, sin, cos, clf, imshow, cm, quiver, show


def main():
    # Create the grid
    x = arange(-100, 101)
    y = arange(-100, 101)

    # Create the meshgrid
    Y, X = meshgrid(x, y)
    A = 1
    B = 2
    V = 6*pi / 201
    W = 4*pi / 201
    F = A*sin(V*X) + B*cos(W*Y)
    Fx = V*A*cos(V*X)
    Fy = W*B*-sin(W*Y)

    # Show the images
    show_image(F)
    show_image(Fx)
    show_image(Fy)

    # Create the grid for the quivers
    xs = arange(-100, 101, 10)
    ys = arange(-100, 101, 10)

    # Here we determine the direction of the quivers
    Ys, Xs = meshgrid(ys, xs)
    FFx = V*A*cos(V*Xs)
    FFy = W*B*-sin(W*Ys)

    # Draw the quivers and the image
    clf()
    imshow(F, cmap=cm.gray, extent=(-100, 100, -100, 100))
    quiver(ys, xs, -FFy, FFx, color='red')
    show()


# This function shows an image
def show_image(F):
    clf()
    imshow(F, cmap=cm.gray)
    show()


if __name__ == "__main__":
    main()
