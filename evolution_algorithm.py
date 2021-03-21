#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" evolution algorithm
Authors: Jakub Ciemięga, Krzysztof Piątek
"""

import numpy as np
from individual import generate_individual, create_image


def initialize_population(pop_size: int, img_size: int) -> list:
    """ Generate iniital population

    :param pop_size: size of the population
    :type pop_size: int

    :param img_size: size of the square input image
    :type img_size: int

    :return pop: initial population pop_size individuals
    :rtype pop: list
    """

    pop = []
    for i in range(pop_size):
        pop.append(generate_individual(img_size))

    return pop


def objective_function(input_image: np.ndarray, individuals: list) -> np.ndarray:
    """
    calculate value of objective function by comparing the given image with input_image and calculating average absolute
    error

    :param input_image: image the algorithm is trying to achieve
    :type input_image: np.ndarray with shape (height, width, 3) where height=width

    :param individuals: list of individuals to calculate the objective function for
    :type individuals: list of np.ndarray

    :return scores: values of the objective function calculated for the given individuals
    :rtype scores: np.ndarray of floats
    """

    # convert input_image to to int to perform subtraction correctly (uint8 only supports 0-255 so 2-4 = 254)
    input_image = input_image.copy().astype(int)

    scores = np.zeros(len(individuals))

    for i in range(len(individuals)):
        img = create_image(input_image.shape[0], individuals[i]).astype(int)

        # calculate errors for each color channel
        abs_error_b = np.sum(np.abs(input_image[:, :, 0] - img[:, :, 0]))
        abs_error_g = np.sum(np.abs(input_image[:, :, 1] - img[:, :, 1]))
        abs_error_r = np.sum(np.abs(input_image[:, :, 2] - img[:, :, 2]))

        # average absolute error for all colors
        scores[i] = (abs_error_b + abs_error_g + abs_error_r) / img.shape[0] ** 2

    return scores
