#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" evolution algorithm
Authors: Jakub Ciemięga, Krzysztof Piątek
"""

import numpy as np
import copy
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
    individuals = copy.deepcopy(individuals)
    new_individuals = []

    for i in range(pop_size):
        first = np.random.randint(0, pop_size)
        second = first
        while second == first:
            second = np.random.randint(0, pop_size)
        # pick individual with lower value
        if scores[first] < scores[second]:
            new_individuals.append(individuals[first])
        else:
            new_individuals.append(individuals[second])

    return new_individuals


def succession_elite(old_pop: list, old_scores: np.ndarray, new_pop: list, new_scores: np.ndarray,
                     num_kept: int = 1) -> (list, np.ndarray):
    """ Perform elite succession, keeps num_kept best individuals from old_pop and replaces the rest with best
        individuals from new_pop

    :param old_pop: current population
    :type old_pop: list of pop_size individuals

    :param old_scores: values of the objective function for individuals in current population
    :type old_scores: np.ndarray of floats

    :param new_pop: new population
    :type new_pop: list of pop_size individuals

    :param new_scores: values of the objective function for individuals in new population
    :type new_scores: np.ndarray of floats

    :param num_kept: how many best individuals from old_pop are not to be replaced by best individuals from new_pop
    :type num_kept: int

    :return: (combined_pop, combined_scores): combined_pop: population comprising num_kept best individuals from old_pop
                                                            and (pop_size-num_kept) best individuals from new_pop
                                              combined_scores: values of the objective function for individuals in
                                                               combined population
    :rtype: (list, np.ndarray)
    """

    combined_pop = copy.deepcopy(new_pop)
    combined_scores = new_scores.copy()

    worst_positions = np.argpartition(combined_scores, -num_kept)[-num_kept:]
    best_positions = np.argpartition(old_scores, num_kept)[:num_kept]

    combined_scores[worst_positions] = old_scores[best_positions]
    for i in range(num_kept):
        combined_pop[worst_positions[i]] = old_pop[best_positions[i]]

    return combined_pop, combined_scores


def succession_steady_state(old_pop: list, old_scores: np.ndarray, new_pop: list, new_scores: np.ndarray) -> \
        (list, np.ndarray):
    """ Perform steady state succession
    Places 1 best individual from new_pop in old_pop. Individual from old_pop to be replaced is found in a reverse
    tournament.

    :param old_pop: current population
    :type old_pop: list of pop_size individuals

    :param old_scores: values of the objective function for individuals in current population
    :type old_scores: np.ndarray of floats

    :param new_pop: new population
    :type new_pop: list of pop_size individuals

    :param new_scores: values of the objective function for individuals in new population
    :type new_scores: np.ndarray of floats

    :return: (combined_pop, combined_scores): combined_pop: generated successive population
                                              combined_scores: values of the objective function for individuals in
                                                               combined population
    :rtype: (list, np.ndarray)
    """
    combined_pop = copy.deepcopy(old_pop)
    combined_scores = old_scores.copy()
    pop_size = len(combined_pop)

    # find index in combined population to be replaced in reverse tournament
    first = np.random.randint(0, pop_size)
    second = first
    while second == first:
        second = np.random.randint(0, pop_size)
    # pick individual with higher (worse) value
    if combined_scores[first] > combined_scores[second]:
        replaced_pos = first
    else:
        replaced_pos = second

    # find best
    best_pos = np.argmin(new_scores)

    # replace
    combined_pop[replaced_pos] = new_pop[best_pos]
    combined_scores[replaced_pos] = new_scores[best_pos]

    return combined_pop, combined_scores


def mutation(pop: list, img_size: int, max_rectangles: int, prob_edit: float, prob_add: float, mut_std: float) -> list:
    """ Perform mutation on a given population

    For each individual of k rectangles at first decide if to change existing rectangles or change their number.
    If changing existing ones: draw rectangle index and which part to change (0 - top-left corner, 1 -
    bottom-right corner, 2-5 - BGRA channels) and change the appropriate trait (for each individual two traits of many
    rectangles are changed). If changing number of rectangles: decide if to add or remove based on prob_add.

    :param pop:  population to perform mutation on
    :type pop: list of pop_size individuals

    :param img_size: size of the square input image
    :type img_size: int

    :param max_rectangles: maximum number of rectangles an individual can comprise
    :type max_rectangles: int

    :param prob_edit: probability that individual's rectangles are edited and nothing is added/removed. Probability that
                      a rectangle is removed or added = 1-prob_edit
    :type prob_edit: float

    :param prob_add: probability a rectangle is added and not removed. final_prob_add = (1-prob_edit)*prob_add
    :type prob_add: float

    :param mut_std: defines mutation std in relation to appropriate maximum value so: coordinates_std=mut_std*img_size,
                    colors_std = std*255
    :type mut_std: float

    :return pop_mutation: mutated population
    :rtype pop_mutation: list of pop_size individuals
    """
    pop_mutation = copy.deepcopy(pop)
    pop_size = len(pop_mutation)

    # perform mutation for each individual
    for i in range(pop_size):
        ind = pop_mutation[i].copy()

        # flag to mark if a rectangle was added or removed
        already_changed = False

        mutation_type = np.random.random()  # change rectangles or add/remove
        if mutation_type >= prob_edit:
            # decide if to add or remove a rectangle
            add_remove = np.random.random()
            if add_remove >= prob_add and ind.shape[0] > 2:
                # remove rectangle
                ind = np.delete(ind, np.random.randint(0, ind.shape[0]), axis=0)
                already_changed = True
            elif ind.shape[0] < max_rectangles:
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
                already_changed = True

        if mutation_type < prob_edit or not already_changed:  # if could not add or remove rectangle, this mutation is performed
            # decide how many rectangles to change
            change_num = np.random.randint(1, ind.shape[0] + 1)
            for temp in range(change_num):
                rect_idx = np.random.randint(0, ind.shape[0])
                # change two traits of each chosen rectangle
                for temp2 in range(2):
                    trait_idx = np.random.randint(0, 6)
                    if 0 == trait_idx:
                        # x1, y1
                        ind[rect_idx, 0] = np.maximum(0, np.minimum(ind[rect_idx, 2] - 1, ind[rect_idx, 0] +
                                                                    np.random.normal(0, mut_std*img_size)))
                        # ind[rect_idx, 0] = np.random.randint(0, ind[rect_idx, 2])
                        ind[rect_idx, 1] = np.maximum(0, np.minimum(ind[rect_idx, 3] - 1, ind[rect_idx, 1] +
                                                                    np.random.normal(0, mut_std*img_size)))
                        # ind[rect_idx, 1] = np.random.randint(0, ind[rect_idx, 3])
                    elif 1 == trait_idx:
                        # x2, y2
                        ind[rect_idx, 2] = np.maximum(ind[rect_idx, 0] + 1,
                                                      np.minimum(img_size, ind[rect_idx, 2] +
                                                                 np.random.normal(0, mut_std*img_size)))
                        # ind[rect_idx, 2] = np.random.randint(ind[rect_idx, 0]+1, img_size)

                        ind[rect_idx, 3] = np.maximum(ind[rect_idx, 1] + 1,
                                                      np.minimum(img_size, ind[rect_idx, 3] +
                                                                 np.random.normal(0, mut_std*img_size)))
                        # ind[rect_idx, 3] = np.random.randint(ind[rect_idx, 1]+1, img_size)
                    elif 2 == trait_idx:
                        # B channel
                        ind[rect_idx, 4] = np.maximum(0, np.minimum(255, ind[rect_idx, 4] + np.random.normal(0, mut_std*255)))
                    elif 3 == trait_idx:
                        # G channel
                        ind[rect_idx, 5] = np.maximum(0, np.minimum(255, ind[rect_idx, 5] + np.random.normal(0, mut_std*255)))
                    elif 4 == trait_idx:
                        # R channel
                        ind[rect_idx, 6] = np.maximum(0, np.minimum(255, ind[rect_idx, 6] + np.random.normal(0, mut_std*255)))
                    elif 5 == trait_idx:
                        # A channel
                        ind[rect_idx, 7] = np.maximum(0, np.minimum(255, ind[rect_idx, 7] + np.random.normal(0, mut_std*255)))

        pop_mutation[i] = ind.copy()

    return pop_mutation

# TODO add save and read population

