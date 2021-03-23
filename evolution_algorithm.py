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


def selection_tournament(individuals: list, scores: np.ndarray) -> list:
    """
        creates a new population using selection tournament

        :param individuals: list of individuals in current population
        :type individuals: list of np.ndarray

        :param scores: values of the objective function for individuals in current population
        :type scores: np.ndarray of floats

        :return new_individuals: population created in tournament
        :rtype new_individuals: list of np.ndarrays
        """
    pop_size = len(individuals)
    new_individuals = []

    for i in range(pop_size):
        first = np.random.randint(0, pop_size)
        second = np.random.randint(0, pop_size)
        # pick individual with lower value
        if scores[first] < scores[second]:
            new_individuals.append(individuals[first])
        else:
            new_individuals.append(individuals[second])

    return new_individuals


def succession_elite(old_pop: list, old_scores: np.ndarray, new_pop: list, new_scores: np.ndarray, k: int = 1) -> \
        (list, np.ndarray):
    """ Perform elite succession, replaces k worst individuals from old_pop with k best individuals from new_pop

    :param old_pop: current population
    :type old_pop: list of pop_size individuals

    :param old_scores: values of the objective function for individuals in current population
    :type old_scores: np.ndarray of floats

    :param new_pop: new population
    :type new_pop: list of pop_size individuals

    :param new_scores: values of the objective function for individuals in new population
    :type new_scores: np.ndarray of floats

    :param k: decides how many worst individuals from old_pop are to be replaced by best individuals from new_pop
    :type k: int

    :return: (combined_pop, combined_scores): combined_pop: population consisting of k best individuals from new_pop and
                                                            (pop_size-k) best individuals from old_pop
                                              combined_scores: values of the objective function for individuals in
                                                               combined population
    :rtype: (list, np.ndarray)
    """
    combined_pop = old_pop.copy()
    combined_scores = old_scores.copy()
    for i in range(k):
        # delete worst element from the population
        if i != 0:
            worst_pos = np.argmax(combined_scores[:-i])  # [:-i] not to delete the elements already added from new_pop
        else:
            worst_pos = np.argmax(combined_scores)  # if else statement because indexing array[:-0] not possible
        combined_pop.pop(worst_pos)
        combined_scores = np.delete(combined_scores, worst_pos)

        # add best element from new population
        best_pos = np.argmin(new_scores)
        combined_pop.append(new_pop[best_pos])
        combined_scores = np.append(combined_scores, new_scores[best_pos])
        # delete best element from new population so it's not added more than once
        new_pop.pop(best_pos)
        new_scores = np.delete(new_scores, best_pos)

    return combined_pop, combined_scores

