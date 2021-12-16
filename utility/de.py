import random

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

    idxs = [idx for idx in range(n_ind)]
    selected = np.random.choice(idxs, 1, replace=False)

    trial[selected] = not trial[selected]

    return trial


def crossover2(n_ind, ind, mutant, cross_proba, pop):
    cross_points = np.random.rand(n_ind) <= cross_proba

    trial = np.where(cross_points, mutant, ind)

    idxs = [idx for idx in range(n_ind)]
    selected = np.random.choice(idxs, 1, replace=False)

    trial[selected] = not trial[selected]

    while trial.tolist() in pop.tolist():
        rand = random.randint(1, int(n_ind/10))
        mutate_index = random.sample(range(0, n_ind), rand)
        for x in mutate_index:
            trial[x] = not trial[x]

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


def mutate_new_jade(n_ind, pop, pInd, ind_pos, ind_pbest):

    idxs = [idx for idx in range(len(pop)) if idx != ind_pos and idx != ind_pbest]
    selected = np.random.choice(idxs, 2, replace=False)
    xr1, xr2 = pop[selected]

    rand = np.random.rand(n_ind) <= [0.5]*n_ind

    mutant = []

    for i in range(n_ind):
        mutant.append(((pInd[i] ^ xr2[i]) and rand[i]) or (not (pInd[i] ^ xr2[i]) and pInd[i]))

    return mutant


def mutate_new_jade2(n_ind, F, pop, pInd, pbilInd, ind_pos, pop_archive, ind_pbest, CR):

    idxs = [idx for idx in range(len(pop)) if idx != ind_pos and idx != ind_pbest]
    selected = np.random.choice(idxs, 2, replace=False)
    xr1, xr2 = pop[selected]

    rand = np.random.rand(n_ind) <= [0.5]*n_ind

    mutant = []
    cross_choices = []

    for i in range(n_ind):
        # mutant.append(((xr1[i] ^ xr2[i]) and rand[i]) or (not (xr1[i] ^ xr2[i]) and xr1[i]))
        mutant.append(((pInd[i] ^ xr2[i]) and rand[i]) or (not (pInd[i] ^ xr2[i]) and pInd[i]))
        if pInd[i] ^ xr2[i]:
            cross_choices.append(CR[0])
        else:
            cross_choices.append(CR[1])


    # mutant = pInd.astype(np.float32) + F * (xr1.astype(np.float32) - xr2.astype(np.float32))

    '''
    for i in range(n_ind):
        if mutant[i] < 0.5:
            mutant[i] = 0
        else:
            mutant[i] = 1

    mutant = mutant.astype(bool)
    '''

    return mutant, cross_choices


def mutate_new(n_ind, F, pop, bestInd, pInd, ind_pos):

    idxs = [idx for idx in range(len(pop)) if idx != ind_pos]
    selected = np.random.choice(idxs, 1, replace=False)
    xr1 = pop[selected[0]]

    mutant = bestInd.astype(np.float32) + F * (pInd.astype(np.float32) - xr1.astype(np.float32))

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
