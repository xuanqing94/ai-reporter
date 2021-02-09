#!/usr/bin/env python
import pickle
import glob
#from tifffile import imread, imwrite
from imageio import imread, imwrite
import numpy as np
import cv2 as cv
import os
from scipy.spatial.distance import pdist
from scipy.spatial import ConvexHull
import pandas as pd
from polygon_inclusion import PolygonRegion
import ray


def find_centroid(points):
    return np.mean(points, axis=0)


def area(points):
    x = points[:, 0]
    y = points[:, 1]
    A = 0.5 * np.sum(y[:-1] * np.diff(x) - x[:-1] * np.diff(y))
    A = np.abs(A)
    return A


def perimeter(points):
    points_shift = np.concatenate((points[1:], points[0:1]), axis=0)
    d = points_shift - points
    return np.sum(np.sqrt(np.sum(d * d, axis=1)))


def radius_EAC(points):
    A = area(points)
    r = np.sqrt(A / np.pi)
    return r


def proximity(points):
    cen = find_centroid(points)
    d = points - cen
    d = np.sqrt(np.sum(d * d, axis=1))
    return np.mean(d, axis=0)


def normalized_proximity(points):
    prox_EAC = radius_EAC(points) * 2 / 3
    prox_shape = proximity(points)
    return prox_EAC / prox_shape


def points_inside(points):
    min_x = max(0, np.min(points[:, 0]) - 1)
    max_x = np.max(points[:, 0])
    min_y = max(0, np.min(points[:, 1]) - 1)
    max_y = np.max(points[:, 1])
    inside_points = []
    mesh_x, mesh_y = np.meshgrid(
        np.arange(min_x, max_x + 1),
        np.arange(min_y, max_y + 1),
        sparse=False,
        indexing="ij",
    )
    mesh_points = np.stack((mesh_x, mesh_y), axis=0).reshape(2, -1)
    points = points.astype(float)
    polygon = PolygonRegion(points.T)
    is_inside = polygon.contains(mesh_points)
    inside_points = mesh_points[:, is_inside].T
    return inside_points


def spin_index(points):
    inside_points = points_inside(points)
    c = find_centroid(points)
    d = inside_points - c
    return np.sum(d * d) / points.shape[0]


def normalized_spin_index(points):
    spin_shape = spin_index(points)
    r = radius_EAC(points)
    spin_EAC = 0.5 * r * r
    return spin_EAC / spin_shape


def dispersion(points):
    c = find_centroid(points)
    d = points - c
    d = np.sqrt(np.sum(d * d, axis=1))
    return np.mean(d), np.std(d)


def normalized_dispersion(points):
    mean, std = dispersion(points)
    return (mean - std) / mean


def cohesion(points):
    inside_points = points_inside(points)
    D = pdist(inside_points)
    return np.mean(D)


def normalized_cohesion(points):
    cohesion_shape = cohesion(points)
    r = radius_EAC(points)
    cohesion_EAC = 0.9054 * r
    return cohesion_EAC / cohesion_shape


def find_nearest(point, points):
    d = points - point
    d = np.sqrt(np.sum(d * d, axis=1))
    return np.min(d)


def depth_index(points):
    inside_points = points_inside(points)
    avg_d = 0.0
    for i in range(inside_points.shape[0]):
        min_d = find_nearest(inside_points[i], points)
        avg_d += min_d
    return avg_d / len(inside_points)


def normalized_depth_index(points):
    depth_index_shape = depth_index(points)
    r = radius_EAC(points)
    depth_index_EAC = r / 3.0
    return depth_index_shape / depth_index_EAC


def largest_inscribed_circle(points):
    inside_points = points_inside(points)
    max_circ = -1
    for i in range(inside_points.shape[0]):
        min_d = find_nearest(inside_points[i], points)
        if min_d > max_circ:
            max_circ = min_d
    return max_circ


def normalized_inscribed_circle(points):
    r_EAC = radius_EAC(points)
    r_shape = largest_inscribed_circle(points)
    return r_shape / r_EAC


def convex_hull(points):
    hull = ConvexHull(points)
    vert_idx = hull.vertices
    vert = points[vert_idx]
    perimeter = 0.0
    for i in range(len(vert_idx)):
        i_next = (i + 1) % (len(vert_idx))
        # dist of i - i_next
        diff = vert[i] - vert[i_next]
        dist = np.sqrt(np.sum(diff * diff))
        perimeter += dist
    return perimeter


def normalized_convex_hull(points):
    perimeter_shape = convex_hull(points)
    r_EAC = radius_EAC(points)
    perimeter_EAC = 2 * np.pi * r_EAC
    return perimeter_EAC / perimeter_shape


def fit_ellipse(points):
    points = points - np.mean(points, axis=0, keepdims=True)
    U, S, V = np.linalg.svd(points)
    return np.max(S) / np.min(S)


@ray.remote
def process_img(f):
    print(f)
    img = imread(f)[:, :, 0]
    corner_pix = int(img[0, 0])
    img = cv.copyMakeBorder(img,
                            32,
                            32,
                            32,
                            32,
                            cv.BORDER_CONSTANT,
                            value=corner_pix)
    kernel = np.ones((5, 5), dtype=np.float) / 25
    smoothed_img = cv.filter2D(img, -1,
                               kernel).astype(np.uint8)[:, :, np.newaxis]
    if corner_pix < 2:
        thresh_c = 1
    elif corner_pix > 254:
        thresh_c = 254
    else:
        print("Unhandled case")
        exit(0)
    # pad image with pixels
    _, thresh = cv.threshold(smoothed_img, thresh_c, 255, 0)
    if thresh_c == 254:
        # invert
        thresh = 255 - thresh
    #im2, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE,
    #                                           cv.CHAIN_APPROX_NONE)
    contours = cv.findContours(thresh, cv.RETR_TREE,
                                               cv.CHAIN_APPROX_NONE)[0]
    # find the longest contour
    contour = sorted(contours, key=lambda c: c.shape[0], reverse=True)[0]
    points = contour[:, 0, :]
    features = [
        area(points),
        0,
        perimeter(points),
        normalized_proximity(points),
        normalized_spin_index(points),
        normalized_cohesion(points),
        normalized_depth_index(points),
        normalized_inscribed_circle(points),
        normalized_convex_hull(points),
        fit_ellipse(points),
    ]

    # TODO do whatever with contour variable, it is a numpy array of shape (circumference, 1)

    # convert to RGB image and draw red contour
    color_img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    cv.drawContours(color_img, [contour], 0, (255, 0, 0), 3)

    # save
    out_f = os.path.join("./single_cell_boundary_v3/", os.path.basename(f))
    imwrite(out_f, color_img)
    return features


if __name__ == "__main__":
    ray.init()
    files = sorted(glob.glob("./C-NMC_Leukemia/validation_data/C-NMC_test_prelim_phase_data/*.bmp"))
    promise = []
    for f in files:
        promise.append(process_img.remote(f))
    shape_data = ray.get(promise)
    df = pd.DataFrame(data=np.asarray(shape_data),
                      columns=[
                          "area", "area_ratio", "perimeter", "proximity",
                          "spin_index", "cohesion", "depth_index",
                          "inscribed_circle", "convex_hull", "aspect_ratio"
                      ])
    df['files'] = files
    df.to_csv("./validation_data.csv", index=False)
    #pickle.dump(shape_data, open("./shape_data_v3.pkl", "wb+"))
