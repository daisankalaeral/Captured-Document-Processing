import cv2
import numpy as np
import sympy


def appx_best_fit_ngon(mask_cv2_gray, n: int = 4) -> list[(int, int)]:
    # convex hull of the input mask
    contours, _ = cv2.findContours(
        mask_cv2_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    hull = cv2.convexHull(contours[0])
    hull = np.array(hull).reshape((len(hull), 2))

    # to sympy land
    hull = [sympy.Point(*pt) for pt in hull]

    # run until we cut down to n vertices
    while len(hull) > n:
        best_candidate = None

        # for all edges in hull ( <edge_idx_1>, <edge_idx_2> ) ->
        for edge_idx_1 in range(len(hull)):
            edge_idx_2 = (edge_idx_1 + 1) % len(hull)

            adj_idx_1 = (edge_idx_1 - 1) % len(hull)
            adj_idx_2 = (edge_idx_1 + 2) % len(hull)

            edge_pt_1 = sympy.Point(*hull[edge_idx_1])
            edge_pt_2 = sympy.Point(*hull[edge_idx_2])
            adj_pt_1 = sympy.Point(*hull[adj_idx_1])
            adj_pt_2 = sympy.Point(*hull[adj_idx_2])

            subpoly = sympy.Polygon(adj_pt_1, edge_pt_1, edge_pt_2, adj_pt_2)
            angle1 = subpoly.angles[edge_pt_1]
            angle2 = subpoly.angles[edge_pt_2]

            # we need to first make sure that the sum of the interior angles the edge
            # makes with the two adjacent edges is more than 180°
            if sympy.N(angle1 + angle2) <= sympy.pi:
                continue

            # find the new vertex if we delete this edge
            adj_edge_1 = sympy.Line(adj_pt_1, edge_pt_1)
            adj_edge_2 = sympy.Line(edge_pt_2, adj_pt_2)
            intersect = adj_edge_1.intersection(adj_edge_2)[0]

            # the area of the triangle we'll be adding
            area = sympy.N(sympy.Triangle(edge_pt_1, intersect, edge_pt_2).area)
            # should be the lowest
            if best_candidate and best_candidate[1] < area:
                continue

            # delete the edge and add the intersection of adjacent edges to the hull
            better_hull = list(hull)
            better_hull[edge_idx_1] = intersect
            del better_hull[edge_idx_2]
            best_candidate = (better_hull, area)

        if not best_candidate:
            raise ValueError("Could not find the best fit n-gon!")

        hull = best_candidate[0]

    # back to python land
    hull = [(int(x), int(y)) for x, y in hull]
    hull, closest_points = find_closest_points(hull, contours[0])

    return hull, closest_points

def find_closest_points(hull, contour):

    closest_points = []

    for hull_point in hull:
        min_distance = np.inf
        closest_point = None

        for contour_point in contour:
            # Calculate Euclidean distance
            print(hull_point[0])
            print(contour_point[0][0])
            print(hull_point[0]-contour_point[0][0])
            distance = np.sqrt((hull_point[0] - contour_point[0][0])**2 + (hull_point[1] - contour_point[0][1])**2)
            if distance < min_distance:
                min_distance = distance
                closest_point = contour_point[0]

        closest_points.append(closest_point)

    return hull, closest_points

from itertools import combinations
from shapely.geometry import Polygon

def max_area_quad(points):
    # Generate all combinations of 4 points
    all_combinations = list(combinations(points, 4))

    max_area = 0
    max_quad = None

    for quad in all_combinations:
        # Calculate the area of the quadrilateral
        polygon = Polygon([(q[0][0], q[0][1]) for q in quad])
        area = polygon.area

        # Update max_area and max_quad if this quadrilateral has a larger area
        if area > max_area:
            max_area = area
            max_quad = quad

    return max_quad