# Author: Sudeep Pillai (spillai@csail.mit.edu)
# License: MIT
# Python port of https://github.com/mzhang94/stereo/blob/master/alg-StereoBM-Halide/src/stereoBM.cpp

import cv2
import numpy as np

import os.path
import time

import halide as h
from halide import Image, Var, Func, RDom, UInt, Int

def colormap(im, min_threshold=0.01):
    mask = im<min_threshold
    hsv = np.zeros((im.shape[0], im.shape[1], 3), np.uint8)
    hsv[...,0] = (im * 180).astype(np.uint8)
    hsv[...,1] = 255
    hsv[...,2] = 255
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    bgr[mask] = 0
    return bgr

def main(): 
    left = cv2.imread(os.path.join("../data/left.png"), cv2.IMREAD_COLOR).transpose(1,0,2)
    right = cv2.imread(os.path.join("../data/right.png"), cv2.IMREAD_COLOR).transpose(1,0,2)
    assert left.dtype == np.uint8
    assert right.dtype == np.uint8

    # We create an Image object to wrap the numpy array
    left_image = Image(left)
    right_image = Image(right)
    W, H, C = left_image.width(), left_image.height(), left_image.channels()
    print 'W {} x H {} x C {}'.format(W,H,C)

    # StereoBM
    SADWindowSize = 15
    numDisparities = 128
    disp_image = stereoBM(left_image, right_image, W, H, SADWindowSize, 0, numDisparities)

    # Realize disparity
    result = Image(UInt(16), disp_image)
    disp = h.image_to_ndarray(result).transpose(1,0)
    
    print 'Input', left.shape[:2]
    print 'Disparity', disp.shape[:2]

    # Colorize stereo disparity
    disp_color = colormap(disp.astype(np.float32) / numDisparities)
    cv2.imwrite('disp.png', disp_color)
    cv2.imshow('disp', disp_color)
    cv2.waitKey(0)

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

    # Sobel filtering
    filteredLeft = prefilterXSobel(left, W, H)
    filteredRight = prefilterXSobel(right, W, H)

    # Stereo block-matching 
    win2 = SADWindowSize / 2
    maxDisparity = numDisparities - 1
    xmin = maxDisparity + win2
    xmax = width - win2 - 1
    ymin = win2
    ymax = height - win2 - 1
    x_tile_size, y_tile_size = 32, 32
    disp = findStereoCorrespondence(filteredLeft, filteredRight, SADWindowSize, minDisparity, numDisparities, 
                                    xmin, xmax, ymin, ymax, x_tile_size, y_tile_size)

    # Write to pretty html
    args = h.ArgumentsVector()
    disp.compile_to_lowered_stmt("disp.html", args, h.HTML)

    # Compile / Profile
    profile(disp, W, H)

    # Start with a target suitable for the machine you're running
    # this on.
    target = h.get_host_target()
    disp_image = disp.realize(W, H)

    return disp_image

if __name__ == "__main__":
    main()
    
