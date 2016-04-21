import numpy as np
# from scipy.misc import imread, imsave
# from imageIO import imread
import cv2

import os.path
import time

import halide as h
from halide import Var, Func, RDom, UInt, Int

from bot_vision.imshow_utils import imshow_cv
from pybot_vision import scaled_color_disp

def main(): 
    # First we'll load the input image we wish to brighten.
    left = cv2.imread(os.path.join("../data/left.png"), cv2.IMREAD_COLOR).transpose(1,0,2)
    right = cv2.imread(os.path.join("../data/right.png"), cv2.IMREAD_COLOR).transpose(1,0,2)
    assert left.dtype == np.uint8
    assert right.dtype == np.uint8

    print left.shape

    # We create an Image object to wrap the numpy array
    left_image = h.Image(left)
    right_image = h.Image(right)
    W, H, C = left_image.width(), left_image.height(), left_image.channels()
    print 'W {} x H {} x C {}'.format(W,H,C)
    
    # width, height = left.shape[:2]
    SADWindowSize = 15
    numDisparities = 128

    disp_image = stereoBM(left_image, right_image, W, H, SADWindowSize, 0, numDisparities)

    # # Realize xsobel
    # result = h.Image(Int(8), disp_image)
    # disp = h.image_to_ndarray(result).transpose(1,0)
    # print disp[:10, :10]
    # imshow_cv('disp', (disp + 31) / 62.0, block=True)

    # Realize disparity
    result = h.Image(UInt(16), disp_image)
    disp = h.image_to_ndarray(result).transpose(1,0)
    
    print 'Input', left.shape[:2]
    print 'Disparity', disp.shape[:2]

    disp_color = scaled_color_disp(disp)
    imshow_cv('disp', disp_color, block=True)


def profile(func, W, H): 
    func.compile_jit()
    niters = 10
    st = time.time()
    for j in xrange(niters): 
        func.realize(W,H)
    print('Total time taken to realize {} ms'.format((time.time() - st) * 1000.0 / niters))

def prefilterXSobel(image, W, H): 
    x, y = Var("x"), Var("y")
    clamped, gray = Func("clamped"), Func("gray")
    gray[x, y] = 0.2989*image[x, y, 0] + 0.5870*image[x, y, 1] + 0.1140*image[x, y, 2]
    clamped[x, y] = gray[h.clamp(x, 0, W-1), h.clamp(y, 0, H-1)]

    temp, xSobel = Func("temp"), Func("xSobel")
    temp[x, y] = clamped[x+1, y] - clamped[x-1, y]
    xSobel[x, y] = h.cast(Int(8), h.clamp(temp[x, y-1] + 2 * temp[x, y] + temp[x, y+1], -31, 31))

    xi, xo, yi, yo = Var("xi"), Var("xo"), Var("yi"), Var("yo")
    xSobel.compute_root().tile(x, y, xo, yo, xi, yi, 64, 32).parallel(yo).parallel(xo)
    temp.compute_at(xSobel, yi).vectorize(x, 8)
    return xSobel

def findStereoCorrespondence(left, right, SADWindowSize, minDisparity, numDisparities,
                             xmin, xmax, ymin, ymax,
                             x_tile_size=32, y_tile_size=32, test=False, uniquenessRatio=0.15, disp12MaxDiff=1): 
    """ Returns Func (left: Func, right: Func) """

    x, y, c, d = Var("x"), Var("y"), Var("c"), Var("d")

    diff = Func("diff")
    diff[d, x, y] = h.cast(UInt(16), h.abs(left[x, y] - right[x-d, y]))

    win2 = SADWindowSize/2

    diff_T = Func("diff_T")
    xi, xo, yi, yo = Var("xi"), Var("xo"), Var("yi"), Var("yo")
    diff_T[d, xi, yi, xo, yo] = diff[d, xi + xo * x_tile_size + xmin, yi + yo * y_tile_size + ymin]

    cSAD, vsum = Func("cSAD"), Func("vsum")
    rk = RDom(-win2, SADWindowSize, "rk")
    rxi, ryi = RDom(1, x_tile_size - 1, "rxi"), RDom(1, y_tile_size - 1, "ryi")

    if test: 
        vsum[d, xi, yi, xo, yo] = h.sum(diff_T[d, xi, yi+rk, xo, yo])
        cSAD[d, xi, yi, xo, yo] = h.sum(vsum[d, xi+rk, yi, xo, yo])
    else: 
        vsum[d, xi, yi, xo, yo] = h.select(yi != 0, h.cast(UInt(16), 0), h.sum(diff_T[d, xi, rk, xo, yo]))
        vsum[d, xi, ryi, xo, yo] = vsum[d, xi, ryi-1, xo, yo] + diff_T[d, xi, ryi+win2, xo, yo] - diff_T[d, xi, ryi-win2-1, xo, yo]

        cSAD[d, xi, yi, xo, yo] = h.select(xi != 0, h.cast(UInt(16), 0), h.sum(vsum[d, rk, yi, xo, yo]))
        cSAD[d, rxi, yi, xo, yo] = cSAD[d, rxi-1, yi, xo, yo] + vsum[d, rxi+win2, yi, xo, yo] - vsum[d, rxi-win2-1, yi, xo, yo]

    rd = RDom(minDisparity, numDisparities)
    disp_left = Func("disp_left")
    disp_left[xi, yi, xo, yo] = h.Tuple(h.cast(UInt(16), minDisparity), h.cast(UInt(16), (2<<16)-1))
    disp_left[xi, yi, xo, yo] = h.tuple_select(
            cSAD[rd, xi, yi, xo, yo] < disp_left[xi, yi, xo, yo][1],
            h.Tuple(h.cast(UInt(16), rd), cSAD[rd, xi, yi, xo, yo]), 
            h.Tuple(disp_left[xi, yi, xo, yo]))

    FILTERED = -16
    disp = Func("disp")

    disp[x, y] = h.select(
        # x > xmax-xmin or y > ymax-ymin,
        x < xmax, 
        h.cast(UInt(16), disp_left[x % x_tile_size, y % y_tile_size, x / x_tile_size, y / y_tile_size][0]), 
        h.cast(UInt(16), FILTERED))
        

    # Schedule
    vector_width = 8
    disp.compute_root() \
        .tile(x, y, xo, yo, xi, yi, x_tile_size, y_tile_size).reorder(xi, yi, xo, yo) \
        .vectorize(xi, vector_width).parallel(xo).parallel(yo)

    # reorder storage
    disp_left.reorder_storage(xi, yi, xo, yo)
    diff_T   .reorder_storage(xi, yi, xo, yo, d)
    vsum     .reorder_storage(xi, yi, xo, yo, d)
    cSAD     .reorder_storage(xi, yi, xo, yo, d)

    disp_left.compute_at(disp, xo).reorder(xi, yi, xo, yo) \
                                  .vectorize(xi, vector_width) \
                                  .update() \
                                  .reorder(xi, yi, rd, xo, yo).vectorize(xi, vector_width)

    if test: 
        cSAD.compute_at(disp_left, rd).reorder(xi,  yi, xo, yo, d).vectorize(xi, vector_width)
        vsum.compute_at(disp_left, rd).reorder(xi,  yi, xo, yo, d).vectorize(xi, vector_width)
    else: 
        cSAD.compute_at(disp_left, rd).reorder(xi,  yi, xo, yo, d).vectorize(xi, vector_width) \
                                                                  .update() \
                                                                  .reorder(yi, rxi, xo, yo, d).vectorize(yi, vector_width)
        vsum.compute_at(disp_left, rd).reorder(xi,  yi, xo, yo, d).vectorize(xi, vector_width) \
                                                                  .update() \
                                                                  .reorder(xi, ryi, xo, yo, d).vectorize(xi, vector_width)
    
    return disp


def stereoBM(left_image, right_image, width, height, 
             SADWindowSize, minDisparity, numDisparities): 
    
    x, y, c = Var("x"), Var("y"), Var("c")
    left, right = Func("left"), Func("right")
    left[x, y, c] = left_image[x, y, c]
    right[x, y, c] = right_image[x, y, c]

    W, H, C = left_image.width(), left_image.height(), left_image.channels()
    
    filteredLeft = prefilterXSobel(left, W, H)
    filteredRight = prefilterXSobel(right, W, H)

    # profile(filteredLeft, W, H)
    # return filteredLeft.realize(W, H)

    win2 = SADWindowSize / 2
    maxDisparity = numDisparities - 1
    xmin = maxDisparity + win2
    xmax = width - win2 - 1
    ymin = win2
    ymax = height - win2 - 1

    x_tile_size, y_tile_size = 32, 32
    disp = findStereoCorrespondence(filteredLeft, filteredRight, SADWindowSize, minDisparity, numDisparities, 
                                    xmin, xmax, ymin, ymax, x_tile_size, y_tile_size)

    args = h.ArgumentsVector()
    disp.compile_to_lowered_stmt("disp.html", args, h.HTML)

    # W = (xmax-xmin) / x_tile_size * x_tile_size + x_tile_size
    # H = (ymax-ymin) / x_tile_size * x_tile_size + x_tile_size

    # Compile
    profile(disp, W, H)

    # Start with a target suitable for the machine you're running
    # this on.
    target = h.get_host_target()
    disp_image = disp.realize(W, H)

    return disp_image

if __name__ == "__main__":
    main()
    
