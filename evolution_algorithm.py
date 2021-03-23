#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" evolution algorithm
Authors: Jakub Ciemięga, Krzysztof Piątek
"""

import numpy as np
from individual import generate_individual, create_image


def initialize_population(pop_size: int, img_size: int) -> list:
    """ Generate initial population

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


def mutation(pop: list, img_size: int) -> list:
    """ Perform mutation on a given population

    For each individual of k rectangles at first draw rectangle index and which part to change (0 - top-left corner, 1 -
    bottom-right corner, 2-5 - BGRA channels) and change the appropriate trait (for each individual one trait of one
    rectangle is changed). Then draw a number 0-9 to decide if to add or remove a rectangle (0 - remove, 1-4 - remove).

    :param pop:  population to perform mutation on
    :type pop: list of pop_size individuals

    :param img_size: size of the square input image
    :type img_size: int

    :return pop_mutation: mutated population
    :rtype pop_mutation: list of pop_size individuals
    """
    pop_mutation = pop.copy()
    pop_size = len(pop_mutation)

    # perform mutation for each individual
    for i in range(pop_size):
        ind = pop_mutation[i]

        # mutate one trait of one rectangle
        rect_idx = np.random.randint(0, ind.shape[0])
        trait_idx = np.random.randint(0, 6)
        if 0 == trait_idx:
            # x1, y1
            ind[rect_idx, :2] = np.random.randint(0, img_size - 1, 2)
        elif 1 == trait_idx:
            # x2, y2
            ind[rect_idx, 2] = np.random.randint(ind[rect_idx, 0] + 1, img_size)
            ind[rect_idx, 3] = np.random.randint(ind[rect_idx, 1] + 1, img_size)
        elif 2 == trait_idx:
            # B channel
            ind[rect_idx, 4] = np.random.randint(0, 256)
        elif 3 == trait_idx:
            # G channel
            ind[rect_idx, 5] = np.random.randint(0, 256)
        elif 4 == trait_idx:
            # R channel
            ind[rect_idx, 6] = np.random.randint(0, 256)
        elif 5 == trait_idx:
            # A channel
            ind[rect_idx, 7] = np.random.randint(0, 256)

        # decide if to add or remove a rectangle remove-10%, add-40%
        add_remove = np.random.randint(0, 10)
        if 0 == add_remove:
            # remove rectangle
            ind = np.delete(ind, np.random.randint(0, ind.shape[0]), axis=0)
        elif 4 >= add_remove:
            # add rectangle
            new_rect = np.zeros((1, 8), int)
            # fill rectangle
            # x1, y1
            new_rect[0, :2] = np.random.randint(0, img_size - 1, 2)
            # x2, y2
            new_rect[0, 2] = np.random.randint(new_rect[0, 0] + 1, img_size)
            new_rect[0, 3] = np.random.randint(new_rect[0, 1] + 1, img_size)
            # BGRA channels
            new_rect[0, 4:] = np.random.randint(0, 256, 4)
            # add ready rectangle to individual
            ind = np.r_[ind, new_rect]

        pop_mutation[i] = ind

    return pop_mutation
