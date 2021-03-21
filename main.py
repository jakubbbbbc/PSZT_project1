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
from individual import create_image
from evolution_algorithm import initialize_population, objective_function, selection_tournament

img_size = 400
pop_size = 5
num_generations = 300

if __name__ == "__main__":
    # for tests use random seed: 300418
    np.random.seed(300418)

    # load input image
    input_img = cv2.imread("image/rubens.jpg")[:img_size, :img_size]
    cv2.imshow("Input image", input_img)
    cv2.waitKey()

    # create initial population and its scores
    pop = initialize_population(pop_size, img_size)
    scores = objective_function(input_img, pop)

    # find best individual and their score
    # noinspection PyTypeChecker
    best_ind = pop[np.argmin(scores)]
    best_score = np.min(scores)

    cur_gen = 0

    print('STARTING:')
    for i in range(len(scores)):
        print(scores[i])

    while cur_gen < num_generations:

        pop_selection = selection_tournament(pop, scores)
        # TODO implement mutation() in evolution_algorithm.py
        # pop_mutation = mutation(pop_selection)

        # new_scores = objective_function(input_img, pop_mutation)

        # TODO implement succession_elite() in evolution_algorithm.py
        # pop, scores = succession_elite(pop, scores, pop_mutation, new_scores)

        # if np.min(scores) < best_score:
        #     best_score = np.min(scores)
        #     best_ind = pop[np.argmin(scores)]

        if cur_gen % 100 == 0:
            cv2.imshow('Generation ' + str(cur_gen) + ', score: ' + str(best_score), create_image(img_size, best_ind))
            cv2.waitKey()
            scores_selection = objective_function(input_img, pop_selection)
            print('SELECTION:')
            for i in range(len(scores_selection)):
                print(scores_selection[i])


        cur_gen += 1
