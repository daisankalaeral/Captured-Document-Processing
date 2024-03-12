import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from random import random
from utils import appx_best_fit_ngon
import utils
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
        continue

    max_quad = utils.max_area_quad(approx)
    print(max_quad)

    # Draw
    black_image = np.zeros(image.shape, dtype=np.uint8)

    for corner in max_quad:
        cv2.circle(image, tuple(corner[0]), 10, (255,0,0), -1)

    cv2.drawContours(image, cnt, -1, (0,255,0), 2)

    plt.subplot(121)
    plt.imshow(image)
    plt.subplot(122)
    plt.imshow(mask_image, cmap="gray")

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


    # # Find contours
    # black_image = np.zeros(image.shape, dtype=np.uint8)
    # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # cnt = max(contours, key=cv2.contourArea)

    # # Apply the Douglas-Peucker algorithm
    # epsilon = 0.02*cv2.arcLength(cnt, True)
    # approx = cv2.approxPolyDP(cnt, epsilon, True)

    # # Draw
    # for corner in approx:
    #     cv2.circle(black_image, tuple(corner[0]), 10, (255,255,255), -1)
    # output = cv2.drawContours(black_image, contours, -1, (255,255,255), 2)
    
    # plt.figure()
    # plt.subplot(211)
    # plt.imshow(black_image, cmap="gray")
    
    # plt.subplot(212)
    # plt.imshow(image)

    # plt.show()