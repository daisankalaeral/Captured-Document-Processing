import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from random import random
from utils import appx_best_fit_ngon
from scipy.interpolate import interp1d
import utils

def interpolate_points(points, n, cornerIndex):
    n_corners = len(cornerIndex)
    
    result = []
    cornerIndex = sorted(cornerIndex)
    
    for i in range(n_corners):
        next_idx = (i + 1) % n_corners
        if not next_idx:
            line = points[cornerIndex[i]:]
            temp = points[:cornerIndex[0]]
            line = np.append(line, temp, axis=0)
        else:
            line = points[cornerIndex[i]:cornerIndex[next_idx]]

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
from tqdm import tqdm

background_image = cv2.imread("background.jpeg")

for file in tqdm(files):
    filepath = f"{dir_path}/{file}"

    # Load image and mask
    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    copied_image = np.copy(image)
    mask_image_path = filepath.replace("img", "alb").split("-")[0:4]
    mask_image_path = "-".join(mask_image_path) + ".png"

    # find the mask image
    if not os.path.isfile(mask_image_path):
        mask_image_path = filepath.replace("img", "alb").split("-")[0:5]
        mask_image_path = "-".join(mask_image_path) + ".png"
        if not os.path.isfile(mask_image_path):
            print(f"File not found: {mask_image_path}")
            continue
    mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)

    # combine image with new background

    # rotate image and mask image
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    copied_image = cv2.rotate(copied_image, cv2.ROTATE_90_CLOCKWISE)
    mask_image = cv2.rotate(mask_image, cv2.ROTATE_90_CLOCKWISE)

    # resize image and mask image to half
    # image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    # copied_image = cv2.resize(copied_image, (0, 0), fx=0.5, fy=0.5)
    # mask_image = cv2.resize(mask_image, (0, 0), fx=0.5, fy=0.5)

    kernel = np.ones((3,3),np.uint8)
    try:
        processed_mask = cv2.threshold(mask_image, 0, 255, cv2.THRESH_BINARY)[1]
        processed_mask = cv2.GaussianBlur(processed_mask, (3,3), 0)
        processed_mask = cv2.threshold(processed_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # processed_mask = cv2.erode(processed_mask, kernel, iterations=1)
        # cv2.imwrite("mask.png", processed_mask)
        # processed_mask = processed_mask[-1]
        # cv2.imwrite("t_mask.png", processed_mask)
        # processed_mask = cv2.erode(processed_mask, kernel, iterations=1)
        # cv2.imwrite("mask.png", processed_mask)
        # processed_mask = cv2.GaussianBlur(processed_mask, (5, 5), 0)

    except Exception as e:
        print(e)
        cv2.imwrite("error.png", mask_image)
    combined_image = utils.combine(image, background_image, processed_mask)
    contours, _ = cv2.findContours(processed_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(contours, key=cv2.contourArea)
    except Exception as e:
        print(e)
        continue

    # Apply the Douglas-Peucker algorithm
    epsilon = 0.02*cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    if len(approx) == 4:
        max_quad = approx
        # continue
    else:
        max_quad = utils.max_area_quad(approx)

    if max_quad is None:
        continue

    # Draw
    for corner in max_quad:
        cv2.circle(image, tuple(corner[0]), 2, (255,255,255), 2)
    
    contourList = cnt.tolist()
    cornerIndex = []

    print(len(contours))
    for corner in max_quad:
        cornerIndex.append(contourList.index(corner.tolist()))
    try:
        result = interpolate_points(cnt, 21, cornerIndex)
    except Exception as e:
        print(e)
        print(max_quad)
        plt.figure(figsize=(13, 13))
        plt.subplot(221)
        plt.imshow(image)
        plt.subplot(222)
        plt.imshow(mask_image, cmap="gray")
        plt.subplot(223)
        cv2.drawContours(copied_image, cnt, -1, (0, 255, 0), 1)
        plt.imshow(copied_image)
        plt.show()
        break

    color = {
        0: (255, 0, 0),
        1: (0, 255, 0),
        2: (0, 0, 255),
        3: (255, 255, 0)
    }
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2RGB)
    for i, line in enumerate(result):
        next_i = (i + 1) % 4
        # line = np.expand_dims(line, 1)
        # print(line.shape)
        l = len(line)
        for j, point in enumerate(line):
            # next_j = (j + 1) % l
            # if next_j == 0:
            #     cv2.line(image, tuple(point), tuple(result[next_i][0]), color[i], 1)
            #     cv2.line(mask_image, tuple(point), tuple(result[next_i][0]), color[i], 1)
            #     # cv2.circle(image, tuple(point), 1, color[i], 1)
            #     # cv2.circle(mask_image, tuple(point), 1, color[i], 1)
            # else:
            #     cv2.line(image, tuple(point), tuple(line[next_j]), color[i], 1)
            #     cv2.line(mask_image, tuple(point), tuple(line[next_j]), color[i], 1)
            cv2.circle(image, tuple(point), 1, color[i], 1)
        # cv2.drawContours(image, line, -1, color[i], 2)

    black_image = np.zeros(image.shape, dtype=np.uint8)
    # cv2.drawContours(copied_image, cnt, -1, (0, 255, 0), 1)

    fig = plt.figure(figsize=(13, 13))
    plt.subplot(221)
    plt.imshow(image)
    plt.subplot(222)
    plt.imshow(mask_image, cmap="gray")
    plt.subplot(223)
    plt.imshow(copied_image)
    plt.subplot(224)
    cv2.drawContours(combined_image, cnt, -1, (0, 255, 0), 1)
    plt.imshow(combined_image)
    plt.show()
    # break