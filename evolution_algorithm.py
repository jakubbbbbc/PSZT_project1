#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" evolution algorithm
Authors: Jakub Ciemięga, Kszysztof Piątek
"""

import numpy as np


def objective_function(input_image: np.ndarray, img: np.ndarray) -> float:
    """
    calculate value of objective function by comparing the given image with input_image and calculating average absolute
    error

    :param input_image: image the algorithm is trying to achieve
    :type input_image: np.ndarray with shape (height, width, 3) where height=width

    :param img: image to calculate the objective function for
    :type img: np.ndarray: np.ndarray with shape (height, width, 3) where height=width

    :return val: value of the objective function calculated for two given images
    :rtype: float
    """

    abs_error_b = np.sum(np.abs(input_image[:, :, 0] - img[:, :, 0]))
    abs_error_g = np.sum(np.abs(input_image[:, :, 1] - img[:, :, 1]))
    abs_error_r = np.sum(np.abs(input_image[:, :, 2] - img[:, :, 2]))

    # average absolute error for all colors
    val = (abs_error_b + abs_error_g + abs_error_r) / img.shape[0] ** 2

    return val
