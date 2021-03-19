#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fundamentals of Artificial Intelligence
Project 1
Jakub Ciemięga, Kszysztof Piątek 2021
Warsaw University of Technology
"""

import numpy as np
import cv2
from individual import generate_individual, create_image

img_size = 400
pop_size = 5

if __name__ == "__main__":
    # for tests use random seed: 300418
    np.random.seed(300418)

    # load input image
    input_img = cv2.imread("image/rubens.jpg")[:img_size, :img_size]
    cv2.imshow("Input image", input_img)
    cv2.waitKey()

    pop = []
    for i in range(pop_size):
        pop.append(generate_individual(img_size))
        print(pop[i])
        img = create_image(img_size, pop[i])
        cv2.imshow('Individual '+ str(i), img)

    cv2.waitKey()

    # # current_path = Path(__file__).parent
    # img = cv2.imread("image/rubens.jpg")[:256, :256]
    # rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    #
    # rgba[:100, :100, 3] = 0
    # rgba = rgba.astype('float') / 255
    # print(rgba[:, :, 3])
    #
    # rgba[:, :, 0] = rgba[:, :, 0] * rgba[:, :, 3]
    # rgba[:, :, 1] = rgba[:, :, 1] * rgba[:, :, 3]
    # rgba[:, :, 2] = rgba[:, :, 2] * rgba[:, :, 3]
    #
    # # cv2.imshow('test', rgba)
    # # cv2.waitKey(0)
    #
    # ones = np.ones((500, 500)).astype('float')
    # zeros = np.zeros((500, 500)).astype('float')
    #
    # red = cv2.merge((zeros, zeros, zeros))
    # red[:200, :200, 2] = 1
    # red /= 2
    #
    # blue = cv2.merge((zeros, zeros, zeros))
    # blue[200:, 200:, 2] = 1
    # blue /= 2
    #
    # combined = red+blue
    #
    # cv2.imshow('red', red)
    # cv2.imshow('blue', blue)
    # cv2.imshow('combined', combined)
    # cv2.waitKey(0)
