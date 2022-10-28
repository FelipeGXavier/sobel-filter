import numpy as np
import math 
import cv2

def kernel_x(image, i, j):
    kernel = [
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ]
    gx = kernel[0][0] * image[i - 1][j - 1] + \
         kernel[1][0] * image[i][j - 1] + \
         kernel[2][0] * image[i + 1][j - 1] + \
         kernel[0][1] * image[i - 1][j] + \
         kernel[1][1] * image[i][j] + \
         kernel[2][1] * image[i + 1][j] + \
         kernel[0][2] * image[i - 1][j + 1] + \
         kernel[1][2] * image[i][j + 1] + \
         kernel[2][2] * image[i + 1][j + 1]
    return gx


def kernel_y(image, i, j):
    kernel = [
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ]
    gy = kernel[0][0] * image[i - 1][j - 1] + \
         kernel[0][1] * image[i - 1][j] + \
         kernel[0][2] * image[i - 1][j + 1] + \
         kernel[1][0] * image[i][j - 1] + \
         kernel[1][1] * image[i][j] + \
         kernel[1][2] * image[i][j + 1] + \
         kernel[2][0] * image[i + 1][j - 1] + \
         kernel[2][1] * image[i + 1][j] + \
         kernel[2][2] * image[i + 1][j + 1]
    return gy 

def gradient(gx, gy):
    return math.sqrt(gx ** 2 + gy ** 2) 

def truncate_pixel(pixel):
    if pixel > 255:
        return 255
    if pixel < 0:
        return 0
    return pixel

def flatten(l):
    return [item for sublist in l for item in sublist]

def sobel_filter(in_dir, out_dir):
    base_image = cv2.cvtColor(cv2.imread(in_dir), cv2.COLOR_BGR2GRAY)
    copy_image_gx = np.copy(base_image)
    copy_image_gy = np.copy(base_image)
    copy_image_g = np.copy(base_image)
    size = base_image.shape
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            gx = kernel_x(base_image, i, j)
            gy = kernel_y(base_image, i, j)
            copy_image_gx[i][j] = truncate_pixel(gx)
            copy_image_gy[i][j] = truncate_pixel(gy)
            copy_image_g[i][j] = truncate_pixel(gradient(gx, gy))
    return {'g': copy_image_g, 'gx': copy_image_gx, 'gy': copy_image_gy}

img = sobel_filter('valve1.png', 'sobel_valve.png')
img_g = cv2.cvtColor(img['g'], cv2.COLOR_GRAY2RGB)
img_gx = cv2.cvtColor(img['gx'], cv2.COLOR_GRAY2RGB)
img_gy = cv2.cvtColor(img['gy'], cv2.COLOR_GRAY2RGB)
cv2.imwrite('valve_g.ppm', img_g)
cv2.imwrite('valve_gx.ppm', img_gx)
cv2.imwrite('valve_gy.ppm', img_gy)