#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fundamentals of Artificial Intelligence
Project 1
Jakub Ciemięga, Krzysztof Piątek 2021
Warsaw University of Technology
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from individual import create_image
from evolution_algorithm import initialize_population, objective_function, selection_tournament, succession_elite, \
    mutation, succession_steady_state

image_path = "image/Jakub.jpeg"

img_size = 100
pop_size = 20
num_generations = 30000

# for mutation
max_rectangles = 10000
# probability that individual's rectangles are edited; probability that a rectangle is removed or added = 1-prob_edit
prob_edit = 0.3
# probability a rectangle is added and not removed; final_prob_add = (1-prob_edit)*prob_add
prob_add = 0.8
# defines mutation std in relation to max value: coordinates_std=mut_std*img_size, colors_std = std*255
mut_std = 0.4

# for elite succession
num_kept = 5

# decide if to show additional information
debug = False

if __name__ == "__main__":

    # load input image
    input_img = cv2.imread(image_path)
    input_img = input_img[::8, ::8]
    # y, x = 40, 20
    y, x = 0, 0
    input_img = input_img[y:y+img_size, x:x+img_size]
    cv2.imshow("Input image", input_img)
    cv2.waitKey()

    # create initial population and its scores
    pop = initialize_population(pop_size, img_size)
    scores = objective_function(input_img, pop)

    # find best individual and their score
    # store best scores and generation they were obtained in to plot graphs
    best_ind = pop[np.argmin(scores)]
    best_scores = [np.min(scores)]
    best_scores_gen = [0]

    cur_gen = 1

    while cur_gen < num_generations:
        # selection
        pop_selection = selection_tournament(pop, scores)

        # mutation
        pop_mutation = mutation(pop_selection, img_size, max_rectangles, prob_edit, prob_add, mut_std)

        # calculate new scores
        new_scores = objective_function(input_img, pop_mutation)

        # succession
        pop, scores = succession_elite(pop, scores, pop_mutation, new_scores, num_kept)
        # pop, scores = succession_steady_state(pop, scores, pop_mutation, new_scores)

        # check if better solution found
        if np.min(scores) < best_scores[-1]:
            best_scores.append(round(np.min(scores), 4))
            best_scores_gen.append(cur_gen)
            best_ind = pop[np.argmin(scores)].copy()
            print('gen:', cur_gen, 'best score:', best_scores[-1])

        # for development debugging purposes
        if debug and cur_gen % 500 == 0:
            print('cur_gen:', cur_gen)

            print(np.sort(scores))
            print(np.std(scores))
            print('cur best score:', best_scores[-1])
            # number of rectangles best individual has
            print('best_ind_rect_num:', best_ind.shape[0])
            plt.plot(best_scores_gen[100:], best_scores[100:])
            plt.show()
            cv2.imwrite(
                'results/' + 'gen_' + str(best_scores_gen[-1]) + "_score_" + str(best_scores[-1]) + ".png",
                create_image(img_size, best_ind))

        cur_gen += 1

    print()
    print('final score:', best_scores[-1])
    cv2.imwrite('results/' + 'final_gen_' + str(best_scores_gen[-1]) + "_score_" + str(best_scores[-1]) + ".png",
                create_image(img_size, best_ind))
    cv2.imshow('Final image, score: ' + str(best_scores[-1]), create_image(img_size, best_ind))
    cv2.waitKey()
    cv2.destroyAllWindows()
