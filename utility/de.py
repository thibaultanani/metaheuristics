import utility.utility as utility
import utility.strategy as strategy

import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')

np.set_printoptions(threshold=sys.maxsize)


def crossover(n_ind, ind, mutant, cross_proba):
    cross_points = np.random.rand(n_ind) <= cross_proba

    trial = np.where(cross_points, mutant, ind)

    idxs = [idx for idx in range(len(ind))]
    selected = np.random.choice(idxs, 1, replace=False)

    trial[selected] = not trial[selected]

    return trial


def mutate(F, pop, bestInd, ind_pos, strat):
    try:
        mut_strategy = eval("strategy." + strat)
    except:
        mut_strategy = eval("strategy.de_rand_1")

    mutant = mut_strategy(F, pop, bestInd, ind_pos)

    mutant = mutant.astype(bool)

    return mutant


def mutate_jade(n_ind, F, pop, pInd, ind_pos, pop_archive):

    idxs = [idx for idx in range(len(pop)) if idx != ind_pos]
    selected = np.random.choice(idxs, 1, replace=False)
    xr1 = pop[selected[0]]

    idxs = [idx for idx in range(len(pop_archive)) if idx != ind_pos and idx != selected[0]]
    selected2 = np.random.choice(idxs, 1, replace=False)
    xr2 = pop_archive[selected2[0]]

    mutant = pop[ind_pos].astype(np.float32) + F * (pInd.astype(np.float32) - pop[ind_pos].astype(np.float32)) + \
             F * (xr1.astype(np.float32) - xr2.astype(np.float32))

    for i in range(n_ind):
        if mutant[i] < 0.5:
            mutant[i] = 0
        else:
            mutant[i] = 1

    mutant = mutant.astype(bool)

    return mutant


def merge_feature(pop, mutants):
    pop_list = []
    for ind, score, model, col, scoreA, scoreP, scoreR, scoreF in pop:
        pop_list.append(list([list(ind), score, model, col, scoreA, scoreP, scoreR, scoreF]))
    mut_list = []
    for ind, score, model, col, scoreA, scoreP, scoreR, scoreF in mutants:
        mut_list.append(list([list(ind), score, model, col, scoreA, scoreP, scoreR, scoreF]))
    newpop = []
    scores = []
    models = []
    cols = []
    scoresA = []
    scoresP = []
    scoresR = []
    scoresF = []
    for i in range(len(pop_list)):
        if pop_list[i][1] > mut_list[i][1]:
            newpop.append((pop_list[i][0]))
            scores.append((pop_list[i][1]))
            models.append((pop_list[i][2]))
            cols.append((pop_list[i][3]))
            scoresA.append((pop_list[i][4]))
            scoresP.append((pop_list[i][5]))
            scoresR.append((pop_list[i][6]))
            scoresF.append((pop_list[i][7]))
        else:
            newpop.append((mut_list[i][0]))
            scores.append((mut_list[i][1]))
            models.append((mut_list[i][2]))
            cols.append((mut_list[i][3]))
            scoresA.append((mut_list[i][4]))
            scoresP.append((mut_list[i][5]))
            scoresR.append((mut_list[i][6]))
            scoresF.append((mut_list[i][7]))
    return np.array(newpop), scores, models, cols, scoresA, scoresP, scoresR, scoresF


def merge_knapsack(pop, mutants):
    pop_list = []
    for ind, score, weight in pop:
        pop_list.append(list([list(ind), score, weight]))
    mut_list = []
    for ind, score, weight in mutants:
        mut_list.append(list([list(ind), score, weight]))
    newpop = []
    scores = []
    weights = []
    for i in range(len(pop_list)):
        if pop_list[i][1] > mut_list[i][1]:
            newpop.append((pop_list[i][0]))
            scores.append((pop_list[i][1]))
            weights.append((pop_list[i][2]))
        else:
            newpop.append((mut_list[i][0]))
            scores.append((mut_list[i][1]))
            weights.append((mut_list[i][2]))
    return np.array(newpop), scores, weights
