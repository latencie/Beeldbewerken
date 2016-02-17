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
    

    # show_image(F)
    # show_image(Fx)
    # show_image(Fy)

    ys = arange(-100,101, 10)
    xs = arange(-100, 101, 10)

    Ys, Xs = meshgrid(ys, xs)
    clf()
    imshow(F, cmap=cm.gray, extent=(-100,100, -100,100))
    # component_x = list()
    # component_y = list()
    # # Add each x gradient component to list for each point in mesh
    # # Add each y gradient component to list for each point in mesh
    # # Order of component_x list corresponds with order of component_y list
    # # xs : Each row is equal to each other. Within the row the elements differ.
    # # therefore we only need the elements from the first row (xs[0][i])
    # # ys: Each row is different, all the elements within a row are the same
    # # therefore we only need the first element of each row (ys[j[0]])
    # for i in xs[0]:
    #     for j in ys:
    #         component_x.append(Fx[i][j[0]])
    #         component_y.append(-Fy[i][j[0]])

    quiver(ys, xs, Fy, -Fx, color='red')
    show()

    
def show_image(F):
    clf()
    imshow(F, cmap=cm.gray)
    show()

if __name__ == "__main__":
    main()