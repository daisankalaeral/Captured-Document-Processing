import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from random import random
from utils import appx_best_fit_ngon
from scipy.interpolate import interp1d
import utils

def interpolate_points(points, n, cornerIndex):
    cornerIndex = sorted(cornerIndex)
    n_corners = len(cornerIndex)
    
    result = []

    for i in range(n_corners):
        next_idx = (i + 1) % n_corners
        if not next_idx:
            line = points[cornerIndex[i]:]
            temp = points[:cornerIndex[0]]
            print(line.shape, temp.shape)
            line = np.append(line, temp, axis=0)
        else:
            line = points[cornerIndex[i]:cornerIndex[next_idx]]
        print(line)

        line = [x[0] for x in line]

        line = np.stack(line)

        t = np.linspace(0, 1, len(line))

        fx = interp1d(t, line[:, 0], kind='nearest')
        fy = interp1d(t, line[:, 1], kind='nearest')

        t_new = np.linspace(0, 1, n)

        new_points = np.column_stack((fx(t_new), fy(t_new))).astype(int)
        result.append(new_points[:-1])

    return result

rand_int = int(random()*1000)
dir_path = "img/1"

files = os.listdir(dir_path)

for file in files:
    filepath = f"{dir_path}/{file}"
    print(filepath)
    # filepath = "alb/1/3-90_-12.5-tc_Page_314-7S00001.png"
    # Load the image
    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask_image_path = filepath.replace("img", "alb").split("-L")[0]
    mask_image_path = mask_image_path + ".png" if ".png" not in mask_image_path else mask_image_path
    print(mask_image_path)
    mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
    
    contours, _ = cv2.findContours(mask_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    try:
        cnt = contours[0]
    except Exception as e:
        print(e)
        continue

    # Apply the Douglas-Peucker algorithm
    epsilon = 0.02*cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    if len(approx) == 4:
        max_quad = approx
    else:
        max_quad = utils.max_area_quad(approx)

    # Draw
    for corner in max_quad:
        cv2.circle(image, tuple(corner[0]), 2, (255,255,255), 2)
    
    contourList = cnt.tolist()
    cornerIndex = []
    for corner in max_quad:
        cornerIndex.append(contourList.index(corner.tolist()))
    
    result = interpolate_points(cnt, 21, cornerIndex)
    
    color = {
        0: (255, 0, 0),
        1: (0, 255, 0),
        2: (0, 0, 255),
        3: (255, 255, 0)
    }
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2RGB)
    for i, line in enumerate(result):
        # line = np.expand_dims(line, 1)
        # print(line.shape)
        l = len(line)
        for j, point in enumerate(line):
            next_j = (j + 1) % l
            if next_j == 0:
                cv2.circle(image, tuple(point), 1, color[i], 1)
                cv2.circle(mask_image, tuple(point), 1, color[i], 1)
            else:
                cv2.line(image, tuple(point), tuple(line[next_j]), color[i], 1)
                cv2.line(mask_image, tuple(point), tuple(line[next_j]), color[i], 1)
            # cv2.circle(image, tuple(point), 2, color[i], 2)
        # cv2.drawContours(image, line, -1, color[i], 2)

    black_image = np.zeros(image.shape, dtype=np.uint8)
    cv2.drawContours(black_image, cnt, -1, (255, 255, 255), 1)

    plt.subplot(131)
    plt.imshow(image)
    plt.subplot(132)
    plt.imshow(mask_image, cmap="gray")
    plt.subplot(133)
    plt.imshow(black_image, cmap="gray")

    # print(filepath)
    # print(mask_image_path)
    # # Apply binary thresholding
    # _, mask = cv2.threshold(mask_image, 0.0000000001, 255, cv2.THRESH_BINARY)

    # hull, closest_points = appx_best_fit_ngon(mask, 4)
    # for idx in range(len(closest_points)):
    #     next_idx = (idx + 1) % len(hull)
    #     cv2.line(mask, closest_points[idx], closest_points[next_idx], (0, 255, 0), 1)

    # for pt in closest_points:
    #     cv2.circle(mask, pt, 2, (255, 0, 0), 2)
    
    # plt.imshow(mask)
    plt.show()
    # break