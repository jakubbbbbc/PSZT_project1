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

img_size = 100
pop_size = 20
num_generations = 15000

# for elite succession
k = 1

if __name__ == "__main__":
    # for tests use random seed: 300418
    # np.random.seed(300418)

    # load input image
    input_img = cv2.imread("image/sunset.jpg")
    input_img = input_img[::7, ::7]
    # y, x = 80, 40
    y, x = 0, 0
    input_img = input_img[y:y+img_size, x:x+img_size]
    cv2.imshow("Input image", input_img)
    cv2.waitKey()

    # create initial population and its scores
    pop = initialize_population(pop_size, img_size)
    scores = objective_function(input_img, pop)

    # find best individual and their score
    best_ind = pop[np.argmin(scores)]
    best_scores = [np.min(scores)]
    best_scores_gen = [0]

    cur_gen = 1

    while cur_gen < num_generations:
        # if cur_gen % 100 == 0:
        #     print('cur_gen:', cur_gen)

        # print('old pop:')
        # print(pop)
        # print(scores)
        # selection
        pop_selection = selection_tournament(pop, scores)

        # print('selection:')
        # print(pop_selection)

        # mutation
        # print('before:')
        # print(pop_selection)
        pop_mutation = mutation(pop_selection, img_size)
        # pop_mutation = mutation(pop, img_size)
        # print('after:')
        # print(pop_selection)
        # print(pop_mutation)

        # calculate new scores
        new_scores = objective_function(input_img, pop_mutation)
        # new_scores = objective_function(input_img, pop_selection)

        # print('mutation:')
        # print(pop_mutation)
        # print(new_scores)

        # succession
        pop, scores = succession_elite(pop, scores, pop_mutation, new_scores, k)
        # pop, scores = succession_steady_state(pop, scores, pop_mutation, new_scores)
        # pop, scores = succession_elite(pop, scores, pop_selection, new_scores, k)

        # print('new pop:')
        # print(pop)
        # print(scores)
        # print()
        # pop, scores = pop_mutation, new_scores

        if np.min(scores) < best_scores[-1]:
            best_scores.append(np.min(scores))
            best_scores_gen.append(cur_gen)
            best_ind = pop[np.argmin(scores)].copy()
            print('gen:', cur_gen, 'best score:', best_scores[-1])
            # cv2.imwrite('results/' + 'gen_' + str(cur_gen) + "_score_" + str(int(best_scores[-1])) + ".png",
            #             create_image(img_size, best_ind))
            # cv2.imshow('Generation ' + str(cur_gen) + ', score: ' + str(best_scores[-1]),
            #            create_image(img_size, best_ind))
            # cv2.waitKey()

        if cur_gen % 500 == 0:
            print('cur_gen:', cur_gen)

            print(np.sort(scores))
            print(np.std(scores))
            print('cur best score:', best_scores[-1])
            print('best_ind_rect_num:', best_ind.shape[0])
            plt.plot(best_scores_gen[100:], best_scores[100:])
            plt.show()
            cv2.imwrite(
                'results/' + '_gen_' + str(best_scores_gen[-1]) + "_score_" + str(best_scores[-1]) + ".png",
                create_image(img_size, best_ind))
            # cv2.imshow('Generation ' + str(cur_gen) + ', score: ' + str(best_scores[-1]),
            #            create_image(img_size, best_ind))
        #     cv2.waitKey()

        cur_gen += 1

    print(scores)
    print(np.std(scores))
    print('final score:', best_scores[-1])
    cv2.imwrite('results/' + 'final_gen_' + str(best_scores_gen[-1]) + "_score_" + str(best_scores[-1]) + ".png",
                create_image(img_size, best_ind))
    cv2.imshow('Final image, score: ' + str(best_scores[-1]), create_image(img_size, best_ind))
    cv2.waitKey()
    cv2.destroyAllWindows()
    # for i in range(pop_size):
    #     cv2.imshow('ind ' + str(i) + ', score: ' + str(scores[i]), create_image(img_size, pop[i]))
    # cv2.waitKey()
