import utility.de as de
import utility.utility as utility

import multiprocessing
import numpy as np
import sys
import os
import heapq
import psutil
import time
from datetime import timedelta
import random
from scipy import stats

import warnings

warnings.filterwarnings('ignore')

np.set_printoptions(threshold=sys.maxsize)


class Differential:

    def __init__(self, list_exp, seed, pop, gen, cross_proba, F, strat):
        self.path1 = os.path.dirname(os.getcwd()) + '/in/'
        self.path2 = os.path.dirname(os.getcwd()) + '/out'
        N = 19
        self.vertices = N
        self.n_ind = int(N*(N-1)/2)
        self.list_exp = list_exp
        self.seed = seed
        self.n_pop = pop
        self.n_gen = gen
        self.cross_proba = cross_proba
        self.F = F
        self.strat = strat
        self.c = 0.1
        utility.cleanOut()

    def write_res(self, folderName, y1, y2, bestScorePro, bestScore, bestInd, debut, out, yTps):
        a = os.path.join(os.path.join(self.path2, folderName), 'resultat.txt')
        f = open(a, "w")
        string = "heuristique: Evolution différentielle" + os.linesep + \
                 "population: " + str(self.n_pop) + os.linesep + \
                 "générations: " + str(self.n_gen) + os.linesep + \
                 "probabilité de croisement: " + str(self.cross_proba) + os.linesep +\
                 "F: " + str(self.F) + os.linesep +\
                 "stratégie de mutation: " + str(self.strat) + os.linesep +\
                 "moyenne: " + str(y1) + os.linesep + "meilleur: " + str(y2) + os.linesep + \
                 "temps: " + str(yTps) + os.linesep + \
                 "scores:" + str(bestScorePro) + \
                 "meilleur score: " + str(bestScore) + os.linesep + \
                 "meilleur individu: " + str(bestInd) + os.linesep + \
                 "temps total: " + str(timedelta(seconds=(time.time() - debut))) + os.linesep +\
                 "mémoire: " + str(psutil.virtual_memory())
        f.write(string)
        f.close()
        a = os.path.join(os.path.join(self.path2, folderName), 'print.txt')
        f = open(a, "w")
        f.write(out)

    @staticmethod
    def update_param(muCR, c, SCR):
        try:
            muCR = (1 - c) * muCR + c * (sum(SCR) / len(SCR))
        except ZeroDivisionError:
            pass
        return muCR

    def differential_evolution(self, part, besties, names, iters, times, names2):

        debut = time.time()
        print_out = ""

        for mode in part:

            np.random.seed(None)

            folderName = mode.upper()

            utility.createDirectory(path=self.path2, folderName=folderName)

            # Les axes pour le graphique
            x1 = []
            y1 = []
            y2 = []
            yTps = []

            scoreMax = -9999999
            indMax = -999999

            # Progression des meilleurs éléments
            bestScorePro = []
            bestIndsPro = []

            # Mesurer le temps d'execution
            instant = time.time()

            # Initalise les paramètres
            muCR = 0.5

            # Initialise la population
            pop = utility.create_population_feature(inds=self.n_pop, size=self.n_ind)

            scores, inds = utility.calcScore(pop=pop, n_vertices=self.vertices)

            bestScore, bestInd, bestScorePro, bestIndsPro = \
                utility.conjecture_add_list(scores=scores, inds=pop, bestScorePro=bestScorePro, bestIndsPro=bestIndsPro)

            generation = 0

            mean_scores = np.mean(heapq.nlargest(int(self.n_pop / 2), scores))
            time_instant = timedelta(seconds=(time.time() - instant))
            time_debut = timedelta(seconds=(time.time() - debut))

            x1, y1, y2, yTps = utility.add_axis(bestScore=bestScore, meanScore=mean_scores, iter=generation,
                                                time_debut=time_debut, x1=x1, y1=y1, y2=y2, yTps=yTps)

            print_out = utility.my_print_conjecture(print_out=print_out, mode=mode, mean=mean_scores,
                                                    bestScore=bestScore, time_exe=time_instant,
                                                    time_total=time_debut, iter=generation + 1)

            print_out = print_out + "\n"

            # pbest = int(self.n_pop * self.p)
            # pbest = int(self.n_pop * 0.5)
            pbest = self.n_pop

            div = int(self.n_gen/5)
            div_pbest = int(self.n_pop/5)

            for generation in range(self.n_gen):

                if (generation % div) == 0 and generation != 0:
                    pbest = pbest - div_pbest
                    print("nouvelle valeur de pbest: ", pbest)
                    if pbest < 1:
                        pbest = 1

                instant = time.time()

                # Liste des bons croisements
                cross_probas = []

                # Liste des mutants
                # mutants = []

                # Création des mutants
                for i in range(self.n_pop):

                    # Les indices des individus avec les scores les plus élevés
                    indices = (-np.array(scores)).argsort()[:pbest]

                    cross_proba = -1
                    while cross_proba > 1 or cross_proba < 0:
                        cross_proba = stats.norm.rvs(loc=muCR, scale=0.1)

                    pindex = indices[random.randint(0, len(indices) - 1)]
                    # print(pbest, indices, pindex)
                    pInd = pop[pindex]

                    # mutation
                    # mutant = de.mutate(F=self.F, pop=pop, bestInd=bestInd, ind_pos=i, strat=self.strat)
                    mutant = de.mutate_new_jade(n_ind=self.n_ind, pop=pop, pInd=pInd,
                                                pbilInd="hfhfhf", ind_pos=i, ind_pbest=pindex,
                                                pop_archive="hhfhfhf", F=2)

                    # croisement
                    # trial = de.crossover(n_ind=self.n_ind, ind=pop[i], mutant=mutant, cross_proba=self.cross_proba)
                    trial = de.crossover(n_ind=self.n_ind, ind=pop[i], mutant=mutant, cross_proba=cross_proba)

                    score_m = utility.calcScore_ind(ind=trial, n_vertices=self.vertices)
                    if scores[i] < score_m:
                        pop[i] = trial
                        scores[i] = score_m

                        cross_probas.append(cross_proba)

                        bestScore, bestInd, bestScorePro, bestIndsPro = \
                            utility.conjecture_add_list(scores=scores, inds=pop, bestScorePro=bestScorePro,
                                                        bestIndsPro=bestIndsPro)
                    # mutants.append(trial)

                '''
                # Calcul du score pour l'ensemble des mutants
                scores_m, weights_m, inds_m = utility.fitness_knapsack(self=self, pop=mutants)

                pop_score = zip(pop, scores, weights)
                mut_score = zip(mutants, scores_m, weights_m)

                pop, scores, weights = de.merge_knapsack(pop=pop_score, mutants=mut_score)

                bestScore, bestWeight, bestInd, bestScorePro, bestWeightPro, bestIndsPro = \
                    utility.knapsack_add_list(scores=scores, weights=weights, inds=pop, bestScorePro=bestScorePro,
                                              bestWeightPro=bestWeightPro, bestIndsPro=bestIndsPro)
                '''

                # Enlever les doublons de nos listes
                cross_probas = utility.f7(cross_probas)

                muCR = self.update_param(muCR=muCR, c=self.c, SCR=cross_probas)

                generation = generation + 1

                mean_scores = np.mean(heapq.nlargest(int(self.n_pop / 2), scores))
                time_instant = timedelta(seconds=(time.time() - instant))
                time_debut = timedelta(seconds=(time.time() - debut))

                print_out = utility.my_print_conjecture(print_out=print_out, mode=mode, mean=mean_scores,
                                                        bestScore=bestScore, time_exe=time_instant,
                                                        time_total=time_debut, iter=generation + 1)

                print_out = print_out + "\n"

                x1, y1, y2, yTps = utility.add_axis(bestScore=bestScore, meanScore=mean_scores, iter=generation,
                                                    time_debut=time_debut, x1=x1, y1=y1, y2=y2, yTps=yTps)

                utility.plot_knapsack(x1=x1, y1=y1, y2=y2, yTps=yTps, n_pop=self.n_pop, n_gen=self.n_gen,
                                      heuristic="Evolution différentielle", folderName=folderName, path=self.path2,
                                      bestScore=bestScore, mean_scores=mean_scores,
                                      time_total=time_debut.total_seconds())

                if bestScore > scoreMax:
                    scoreMax = bestScore
                    indMax = bestInd

                self.write_res(folderName=folderName, y1=y1, y2=y2, bestScorePro=bestScorePro,
                               bestScore=bestScore, bestInd=indMax,
                               debut=debut, out=print_out, yTps=yTps)

            besties, names, iters, times, names2 = \
                utility.queues_put_knapsack(y2=y2, folderName=folderName, scoreMax=scoreMax, iter=generation, yTps=yTps,
                                            time=time_debut.total_seconds(), besties=besties, names=names,
                                            names2=names2, iters=iters, times=times)

    def init(self):

        print("#######################################")
        print("#ALGORITHME A EVOLUTION DIFFERENTIELLE#")
        print("#######################################")
        print()

        besties, names, iters, times, names2 = utility.queues_init()

        mods = self.list_exp

        n = len(mods)
        mods = [mods[i::n] for i in range(n)]

        processes = []

        for part in mods:
            process = multiprocessing.Process(target=self.differential_evolution,
                                              args=(part, besties, names, iters, times, names2))
            processes.append(process)
            process.start()

        bestiesLst, namesLst, itersLst, timesLst, names2Lst =\
            utility.queues_get(n_process=len(processes), besties=besties, names=names, names2=names2, iters=iters,
                               times=times)

        for process in processes:
            process.join()

        return utility.res(heuristic="Evolution différentielle", besties=bestiesLst, names=namesLst, iters=itersLst,
                           times=timesLst, names2=names2Lst, path=self.path2, dataset="conjecture")


if __name__ == '__main__':

    diff = Differential(list_exp=["EXP1", "EXP2"],
                        seed=42, pop=200, gen=1000, cross_proba=0.5, F=1, strat='de_best_1')

    diff.init()
