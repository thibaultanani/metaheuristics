import numpy as np


def clip(mutant, size):
    for i in range(size):
        if mutant[i] < 0.5:
            mutant[i] = 0
        else:
            mutant[i] = 1
    return mutant


# Les stratégies de mutations
def de_rand_1(F, pop, bestInd, ind_pos):
    # Selection des 3 individus aléatoires de la population actuelle
    idxs = [idx for idx in range(len(pop)) if idx != ind_pos]
    selected = np.random.choice(idxs, 3, replace=False)
    xr1, xr2, xr3 = pop[selected]

    mutant = xr1.astype(np.float32) + F * (xr2.astype(np.float32) - xr3.astype(np.float32))

    return clip(mutant, len(bestInd))


def de_best_1(F, pop, bestInd, ind_pos):
    # Selection des 2 individus aléatoires de la population actuelle
    idxs = [idx for idx in range(len(pop)) if idx != ind_pos]
    selected = np.random.choice(idxs, 2, replace=False)
    xr1, xr2 = pop[selected]

    mutant = bestInd.astype(np.float32) + F * (xr1.astype(np.float32) - xr2.astype(np.float32))

    return clip(mutant, len(bestInd))


def de_current_to_rand_1(F, pop, bestInd, ind_pos):
    # Selection des 3 individus aléatoires de la population actuelle
    idxs = [idx for idx in range(len(pop)) if idx != ind_pos]
    selected = np.random.choice(idxs, 3, replace=False)
    xr1, xr2, xr3 = pop[selected]

    mutant = pop[ind_pos].astype(np.float32) + F * (xr1.astype(np.float32) - pop[ind_pos].astype(np.float32)) + \
             F * (xr2.astype(np.float32) - xr3.astype(np.float32))

    return clip(mutant, len(bestInd))


def de_current_to_best_1(F, pop, bestInd, ind_pos):
    # Selection des 2 individus aléatoires de la population actuelle
    idxs = [idx for idx in range(len(pop)) if idx != ind_pos]
    selected = np.random.choice(idxs, 2, replace=False)
    xr1, xr2 = pop[selected]

    mutant = pop[ind_pos].astype(np.float32) + F * (bestInd.astype(np.float32) - pop[ind_pos].astype(np.float32)) + \
             F * (xr1.astype(np.float32) - xr2.astype(np.float32))

    return clip(mutant, len(bestInd))


def de_rand_to_best_1(F, pop, bestInd, ind_pos):
    # Selection des 2 individus aléatoires de la population actuelle
    idxs = [idx for idx in range(len(pop)) if idx != ind_pos]
    selected = np.random.choice(idxs, 3, replace=False)
    xr1, xr2, xr3 = pop[selected]

    mutant = pop[ind_pos].astype(np.float32) + F * (bestInd.astype(np.float32) - xr1.astype(np.float32)) + \
             F * (xr2.astype(np.float32) - xr3.astype(np.float32))

    return clip(mutant, len(bestInd))


def de_rand_2(F, pop, bestInd, ind_pos):
    # Selection des 5 individus aléatoires de la population actuelle
    idxs = [idx for idx in range(len(pop)) if idx != ind_pos]
    selected = np.random.choice(idxs, 5, replace=False)
    xr1, xr2, xr3, xr4, xr5 = pop[selected]

    mutant = xr1.astype(np.float32) + F * (xr2.astype(np.float32) - xr3.astype(np.float32)) + \
             F * (xr4.astype(np.float32) - xr5.astype(np.float32))

    return clip(mutant, len(bestInd))


def de_best_2(F, pop, bestInd, ind_pos):
    # Selection des 4 individus aléatoires de la population actuelle
    idxs = [idx for idx in range(len(pop)) if idx != ind_pos]
    selected = np.random.choice(idxs, 4, replace=False)
    xr1, xr2, xr3, xr4 = pop[selected]

    mutant = bestInd.astype(np.float32) + F * (xr1.astype(np.float32) - xr2.astype(np.float32)) + \
             F * (xr3.astype(np.float32) - xr4.astype(np.float32))

    return clip(mutant, len(bestInd))

