import utility.utility as utility
import os
import random


def fitness(n_class, d, ind, target_name, method):
    data, cols = utility.preparation(data=d, ind=ind, target=target_name)
    accuracy, precision, recall, f_score, matrix = utility.learning(n_class=n_class, data=data, target=target_name,
                                                                    method=method)

    return accuracy, precision, recall, f_score, matrix, cols


if __name__ == '__main__':
    dataset = "als"
    target = "survived"

    data = utility.read(filename=(os.path.dirname(os.getcwd()) + '/in/' + dataset))

    vars = [False, True, False, False, False, False, False, False, True, False, True, False, True, False, True,
            False, False, False, False, True, True, False, True, False, True, False, False, False, False, True,
            False, False, False, False, False, True, False, False, False, True, False, False, True, False, True,
            True, True, False, True, False, True, True, True, True, False, False, False, False, False, False, False,
            False, True, False, True, False, False, False, False, False, False, False, False, False, False, True, True,
            False, True, True, False, False, False, False, False, False, True, True, True, False, False, False, False,
            False, True, True, False, True, False, False, False, False, True, True, True, True, True, True, False,
            True, False, False, True, False, True, False, True, False, False, True, False, False, True, False, True,
            True, False, True, False, False, True, False, False, False, False, False, True, False, False, False, False,
            True, False, False, True, True, False, True, False, True, False, True, False, False, True, False, True,
            True, True, True, False, True, True, True, False, True, False, True, False, False, False, False, False,
            False, False, False, False, False, False]

    accuracy, precision, recall, f_score, matrix, cols = fitness(n_class=2, d=data, ind=vars, target_name=target,
                                                                 method="dtc")

    print("score_initial:", recall)

    origin_score = 0
    max_score = recall
    max_ind = vars

    cpt = 1

    while max_score != origin_score:
        origin_score = max_score
        for i in range(179):
            ind = vars.copy()
            ind[i] = not ind[i]

            accuracy, precision, recall, f_score, matrix, cols = \
                fitness(n_class=2, d=data, ind=ind, target_name=target, method="dtc")

            print(i, ": ", recall)

            if recall > max_score:
                max_score = recall
                max_ind = ind

            '''
            for j in range(179):
                ind2 = ind.copy()
                if i != j:
                    ind2[j] = not ind2[j]
                    accuracy, precision, recall, f_score, matrix, cols =\
                        fitness(n_class=2, d=data, ind=ind2, target_name=target, method="LR")
        
                    print(i, "/", j, ": ", recall)
        
                    if recall > max_score:
                        max_score = recall
                        max_ind = ind2
            '''

        print()
        print("iteration ", cpt, ": ", max_score)
        print()
        cpt = cpt + 1

        vars = max_ind

    print(max_score)
    print(max_ind)
