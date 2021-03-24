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
from evolution_algorithm import initialize_population, objective_function, selection_tournament, succession_elite, \
    mutation, succession_steady_state

img_size = 100
pop_size = 20
num_generations = 300000

# for elite succession
k = 1

if __name__ == "__main__":
    # for tests use random seed: 300418
    # np.random.seed(300418)

    # load input image
    input_img = cv2.imread("image/rubens.jpg")
    input_img = input_img[::4, ::4]
    input_img = input_img[:img_size, :img_size]
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

    while cur_gen < num_generations:
        # print('cur_gen:', cur_gen)

        # selection
        pop_selection = selection_tournament(pop, scores)

        # mutation
        pop_mutation = mutation(pop_selection, img_size)
        # pop_mutation = mutation(pop, img_size)

        # calculate new scores
        new_scores = objective_function(input_img, pop_mutation)
        # new_scores = objective_function(input_img, pop_selection)

        # succession
        # pop, scores = succession_elite(pop, scores, pop_mutation, new_scores, k)
        pop, scores = succession_steady_state(pop, scores, pop_mutation, new_scores)
        # pop, scores = succession_elite(pop, scores, pop_selection, new_scores, k)
        # pop, scores = pop_mutation, new_scores

        if np.min(scores) < best_score:
            best_score = np.min(scores)
            best_ind = pop[np.argmin(scores)].copy()
            print('gen:', cur_gen, 'best score:', best_score)
            cv2.imwrite('results/' + 'generation_' + str(cur_gen) + ".png", create_image(img_size, best_ind))
            # cv2.imshow('Generation ' + str(cur_gen) + ', score: ' + str(best_score), create_image(img_size, best_ind))
            # cv2.waitKey()

        if cur_gen % 500 == 0:
            print('cur_gen:', cur_gen)

            print(scores)
            print(np.std(scores))
            print('cur best score:', best_score)
        #     cv2.imshow('Generation ' + str(cur_gen) + ', score: ' + str(best_score), create_image(img_size, best_ind))
        #     cv2.waitKey()

        cur_gen += 1

    print(scores)
    print(np.std(scores))
    print('final score:', best_score)
    cv2.imshow('Final image, score: ' + str(best_score), create_image(img_size, best_ind))
    cv2.waitKey()
    cv2.destroyAllWindows()
    # for i in range(pop_size):
    #     cv2.imshow('ind ' + str(i) + ', score: ' + str(scores[i]), create_image(img_size, pop[i]))
    # cv2.waitKey()


