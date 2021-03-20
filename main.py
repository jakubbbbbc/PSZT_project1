#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fundamentals of Artificial Intelligence
Project 1
Jakub Ciemięga, Krzysztof Piątek 2021
Warsaw University of Technology
"""

import numpy as np
import cv2
from individual import generate_individual, create_image
from evolution_algorithm import objective_function

img_size = 400
pop_size = 20

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
        # print(pop[i])
        img = create_image(img_size, pop[i])
        print(objective_function(input_img, img))
        # cv2.imshow('Individual '+ str(i), img)

    cv2.waitKey()
