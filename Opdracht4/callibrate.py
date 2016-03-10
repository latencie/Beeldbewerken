from pylab import *


def main():
    XYZ = array([
                [0, -5, 5], [0, -3, 5], [0, -1, 5], [-1, 0 ,5],
                [-3, 0, 5], [-5, 0 , 5], [0, -5, 3], [0, -3, 3],
                [0, -1, 3], [-1, 0, 3], [-3, 0, 3], [-5, 0 ,3],
                [0, -5, 1], [0, -3, 1], [0, -1, 1], [-1, 0 ,1],
                [-3, 0, 1], [-5, 0 ,1]
                ])

    xy = array([
                [213.1027, 170.0499], [258.1908, 181.3219],
                [306.41, 193.8464], [351.498, 183.8268],
                [382.8092, 155.6468], [411.6155, 130.5978],
                [223.7485, 218.2691], [267.5841, 230.7935],
                [314.5509, 244.5705], [357.7603, 235.1771],
                [387.819, 205.1184], [415.3728, 178.1908],
                [234.3943, 263.9834], [276.9775, 277.1341],
                [323.318, 291.5372], [363.3963, 282.1438],
                [392.8288, 251.4589], [419.1301, 223.9051]
                ])

    P = callibrate(xy, XYZ)
    reprojection_error(P, XYZ, xy)

    im = imread("calibrationpoints.jpg")
    imshow(im)

    animate_cube(P)

    show()

def callibrate(xy, XYZ):
    matrix = zeros(shape = (2 * len(xy), 12))

    for i in range(len(xy)):
        matrix[2 * i - 1] = [XYZ[i][0], XYZ[i][1], XYZ[i][2], 1, 0, 0, 0, 0,
                            -xy[i][0] * XYZ[i][0], -xy[i][0] * XYZ[i][1],
                            -xy[i][0] * XYZ[i][2], -xy[i][0]] 
        matrix[2 * i] = [0, 0, 0, 0, XYZ[i][0], XYZ[i][1], XYZ[i][2], 1,
                        -xy[i][1] * XYZ[i][0], -xy[i][1] * XYZ[i][1],
                        -xy[i][1] * XYZ[i][2], -xy[i][1]]

    U, D, V = svd(matrix)

    P = V[-1].reshape(3, 4)

    return P

def reprojection_error(P, XYZ, xy):
    homogenous = zeros(shape = (len(XYZ), 4))
    euclidean = 0

    for i in range(len(XYZ)):
        #Make homogenous
        homogenous[i] = np.append(XYZ[i], [1])

        #Calculate dot product and calculate real points by dividing
        dotprod = dot(P, homogenous[i])
        dotprod[0] = dotprod[0] / dotprod[2]
        dotprod[1] = dotprod[1] / dotprod[2]

        #Calculate euclidean error and add to total
        euclidean += sqrt((xy[i][0] - dotprod[0])**2 +  (xy[i][1] - dotprod[1])**2)

    euclidean = euclidean / len(XYZ)

    print euclidean

def draw_cube(P, x, y, z):
    left_coords = zeros(shape = (4, 2))
    right_coords = zeros(shape = (4, 2))

    #Calculate points of the left and right square (divided the 8 points in 2 halves)
    for i in range(4):
        if i == 0:
            left_point = dot(P, [x, y, z, 1])
            right_point = dot(P, [x, y + 1, z, 1])
        elif i == 1:
            left_point = dot(P, [x + 1, y, z, 1])
            right_point = dot(P, [x + 1, y + 1, z, 1])
        elif i == 2:
            left_point = dot(P, [x + 1, y, z + 1, 1])
            right_point = dot(P, [x + 1, y + 1, z + 1, 1])
        elif i == 3:
            left_point = dot(P, [x, y, z + 1, 1])
            right_point = dot(P, [x, y + 1, z + 1, 1])

        #Set the coordinates in the array
        left_coords[i][0] = left_point[0] / left_point[2]
        left_coords[i][1] = left_point[1] / left_point[2]
        right_coords[i][0] = right_point[0] / right_point[2]
        right_coords[i][1] = right_point[1] / right_point[2]

    #Connect all coordinates and draw them
    for i in range(len(left_coords)):
        plot([left_coords[i][0], left_coords[(i + 1) % 4][0]], [left_coords[i][1], left_coords[(i + 1) % 4][1]], color ="cyan", lw = 1.5)
        plot([right_coords[i][0], right_coords[(i + 1) % 4][0]], [right_coords[i][1], right_coords[(i + 1) % 4][1]], color ="cyan", lw = 1.5)
        plot([left_coords[i][0], right_coords[i][0]], [left_coords[i][1], right_coords[i][1]], color ="cyan", lw = 1.5)


def animate_cube(P):
    x = 0
    y = -7
    
    draw_cube(P, x, y, 0)

if __name__ == "__main__":
    main()